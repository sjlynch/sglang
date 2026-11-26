"""CPU gates A and B for the QISO-x per-step width selector (Infera task sal25).

Gate A: the device rule (torch reference path, and the fused Triton kernel under
TRITON_INTERPRET when Triton imports) chooses the same widths as the host
reference on random batches, bit for bit, and the decision reads no q beyond the
drafted prefix (q[:, W-1:] cannot change a width W). The Infera repo's
``sim/qiso_gate_a.py`` runs the same comparison against the replay's own
``exact_stop_width`` on served round logs.

Gate B: chain speculative sampling whose verify length is chosen by the rule from
the drafted tokens' q commits IDs distributed exactly as the target (chi-square,
>= 1e5 chains); a rule that peeks at the q of the token it is about to include
(QISO, section 1.5) is detected.
"""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sel = _load("inkling_width_selector", "python/sglang/srt/speculative/inkling_width_selector.py")

try:
    import triton  # noqa: F401

    qiso_kernel = _load("qiso_width", "python/sglang/kernels/ops/speculative/qiso_width.py")
except Exception:  # pragma: no cover - environment dependent
    qiso_kernel = None


def stock_like_table(seed=0, steep=False):
    rng = random.Random(seed)
    iso = []
    for k in range(sel.DEPTHS):
        lo, hi = (0.05, 0.98) if steep else (0.35 - 0.02 * k, 0.95 - 0.01 * k)
        row = sorted(min(1.0, max(0.0, lo + (hi - lo) * (b / 19) + rng.uniform(-0.02, 0.02))) for b in range(20))
        iso.append(row)
    payload = {
        "version": 1, "kind": sel.TABLE_KIND, "bins": 20, "head_sha256": "7c" * 32,
        "iso": iso,
        "prior": [0.764, 0.697, 0.684, 0.700, 0.703, 0.710, 0.699],
        "answer": [0.965, 0.959, 0.930, 0.958, 0.958, 0.958, 0.958],
    }
    return payload, sel.WidthTable.from_payload(payload)


ROUND_MS = {4: 10.65, 5: 11.1, 6: 11.6, 8: 12.5}  # c1-like belief times


def ms_tensor(times):
    return torch.tensor([times[w] for w in sel.WIDTHS], dtype=torch.float64)


def random_batch(rng, rows):
    q = [[float(torch.tensor(rng.random() ** rng.choice((0.3, 1, 3)), dtype=torch.float32)) for _ in range(7)]
         for _ in range(rows)]
    thinking = [rng.random() < 0.8 for _ in range(rows)]
    return q, thinking


def random_times(rng):
    base = rng.uniform(8, 20)
    inc = sorted(rng.uniform(0.1, 3.0) for _ in range(3))
    return {4: base, 5: base + inc[0], 6: base + inc[1], 8: base + inc[2]}


class TableTests(unittest.TestCase):
    def test_reads_a_pinned_table_and_rejects_bad_ones(self):
        payload, _ = stock_like_table()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            data = json.dumps(payload).encode()
            path.write_bytes(data)
            table = sel.WidthTable.read(path, hashlib.sha256(data).hexdigest())
            self.assertEqual(len(table.iso), 7)
            with self.assertRaises(ValueError):
                sel.WidthTable.read(path, "0" * 64)
        broken = json.loads(json.dumps(payload))
        broken["iso"][2][5] = 0.99  # not monotone
        with self.assertRaises(ValueError):
            sel.WidthTable.from_payload(broken)
        broken = json.loads(json.dumps(payload))
        broken["prior"] = broken["prior"][:6]
        with self.assertRaises(ValueError):
            sel.WidthTable.from_payload(broken)

    def test_fit_reproduces_monotone_tables(self):
        rng = random.Random(4)
        rounds = []
        for _ in range(3000):
            q = [rng.random() for _ in range(7)]
            a = [min(1.0, max(0.0, 0.3 + 0.6 * qq + rng.uniform(-0.2, 0.2))) for qq in q]
            rounds.append([0, 0, 0, *a, *q])
        payload = sel.fit_table(rounds, "ab" * 32, [0.958] * 7)
        table = sel.WidthTable.from_payload(payload)
        self.assertGreater(table.iso[0][19], table.iso[0][0] + 0.3)


class GateATests(unittest.TestCase):
    def test_device_rule_matches_reference_bit_for_bit(self):
        rng = random.Random(1795640482)
        widths = {w: 0 for w in sel.WIDTHS}
        for trial in range(1500):
            payload, table = stock_like_table(seed=trial % 7, steep=trial % 3 == 0)
            rule = sel.ExactStopWidth(table)
            rows = rng.choice((1, 2, 3, 4))
            q, thinking = random_batch(rng, rows)
            times = random_times(rng) if trial % 2 else ROUND_MS
            want = sel.exact_stop_width_reference(table, q, thinking, times)
            got = rule(torch.tensor(q, dtype=torch.float32), torch.tensor(thinking), ms_tensor(times))
            self.assertEqual(int(got), want, (trial, q, thinking, times))
            widths[want] += 1
        # The rule must actually choose among the widths on this mix.
        self.assertTrue(all(count > 20 for count in widths.values()), widths)

    def test_padded_rows_do_not_vote(self):
        rng = random.Random(3)
        _, table = stock_like_table(steep=True)
        rule = sel.ExactStopWidth(table)
        for _ in range(200):
            q, thinking = random_batch(rng, 2)
            want = sel.exact_stop_width_reference(table, q, thinking, ROUND_MS)
            padded_q = torch.tensor(q + [[0.999] * 7, [0.0] * 7], dtype=torch.float32)
            padded_t = torch.tensor(thinking + [True, True])
            active = torch.tensor([True, True, False, False])
            self.assertEqual(int(rule(padded_q, padded_t, ms_tensor(ROUND_MS), active)), want)

    def test_decision_reads_no_q_beyond_the_drafted_prefix(self):
        rng = random.Random(11)
        _, table = stock_like_table(steep=True)
        rule = sel.ExactStopWidth(table)
        for _ in range(600):
            rows = rng.choice((1, 2, 4))
            q, thinking = random_batch(rng, rows)
            q_t = torch.tensor(q, dtype=torch.float32)
            width = int(rule(q_t, torch.tensor(thinking), ms_tensor(ROUND_MS)))
            # Poison every depth the chosen width did not need to decide.
            poisoned = q_t.clone()
            poisoned[:, width - 1 :] = float("nan") if rng.random() < 0.5 else torch.rand(rows, 8 - width)
            self.assertEqual(int(rule(poisoned, torch.tensor(thinking), ms_tensor(ROUND_MS))), width)
            ref = sel.exact_stop_width_reference(
                table, [row[: width - 1] + [float("nan")] * (8 - width) for row in q], thinking, ROUND_MS
            )
            self.assertEqual(ref, width)

    @unittest.skipIf(qiso_kernel is None, "triton not importable")
    def test_fused_kernel_matches_reference(self):
        if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
            self.skipTest("needs a GPU or TRITON_INTERPRET=1")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = random.Random(77)
        for trial in range(300):
            _, table = stock_like_table(seed=trial % 5, steep=trial % 2 == 0)
            rule = sel.ExactStopWidth(table, device=device)
            rows = rng.choice((1, 2, 3, 4))
            q, thinking = random_batch(rng, rows)
            times = random_times(rng)
            want = sel.exact_stop_width_reference(table, q, thinking, times)
            out = torch.zeros(1, dtype=torch.int64, device=device)
            qiso_kernel.qiso_width(
                torch.tensor(q, dtype=torch.float32, device=device),
                torch.tensor(thinking, device=device),
                torch.ones(rows, dtype=torch.bool, device=device),
                rule.iso, rule.prior, rule.answer,
                ms_tensor(times).to(device), out,
            )
            self.assertEqual(int(out.item()), want, (trial, q, thinking, times))

    def test_reference_cost_is_small(self):
        _, table = stock_like_table()
        rng = random.Random(5)
        q, thinking = random_batch(rng, 4)
        start = time.perf_counter()
        for _ in range(2000):
            sel.exact_stop_width_reference(table, q, thinking, ROUND_MS)
        per = (time.perf_counter() - start) / 2000 * 1e6
        print(f"QISO-x host reference: {per:.1f} us per 4-row decision (CPU)")


def chi2_sf(x, dof):
    a, z = dof / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:
        term = total = 1.0 / a
        n = a
        while abs(term) > abs(total) * 1e-15:
            n += 1
            term *= z / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    tiny = 1e-300
    b, c = z + 1 - a, 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def chi2(counts, probs):
    n = counts.sum().item()
    expected = probs.double().flatten() * n
    observed = counts.double().flatten()
    big = expected >= 5
    stat = (((observed[big] - expected[big]) ** 2) / expected[big]).sum().item()
    dof = int(big.sum()) - 1
    if (~big).any():
        e, o = expected[~big].sum().item(), observed[~big].sum().item()
        stat += (o - e) ** 2 / max(e, 1e-12)
        dof += 1
    return chi2_sf(stat, max(dof, 1))


def run_chains(n, vocab, seed, peek=False, draft_noise=1.5, threshold=None):
    """Bigram toy target p and draft q; each chain drafts 7 tokens per round, the
    rule picks the verify width from the drafted q (``peek``: the broken QISO
    variant that reads the q of the token it is deciding to include), chain
    speculative sampling verifies, and chains run until 4 IDs are committed.
    Returns counts of committed (x1) and of (x3, x4)."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(vocab, vocab, generator=gen) * 1.2
    draft = target + torch.randn(vocab, vocab, generator=gen) * draft_noise
    p_rows = torch.softmax(target.double(), -1)
    q_rows = torch.softmax(draft.double(), -1)
    _, table = stock_like_table(steep=True)
    times = {4: 10.0, 5: 10.25, 6: 10.5, 8: 11.0}
    rng = random.Random(seed)

    def width_of(qs):
        if threshold is not None:  # extreme peek: include a depth iff its own q is high
            width = 4
            for _, nxt in sel.STEPS:
                if min(qs[width - 1 : nxt - 1]) >= threshold:
                    width = nxt
                else:
                    break
            return width
        if not peek:
            return sel.exact_stop_width_reference(table, [qs], [True], times)
        width = 4
        for _, nxt in sel.STEPS:
            prefix = [qs[: nxt - 1]]  # reads the q of depths it is about to include
            cur = sel.expected_ids(table, prefix, [True], width, nxt)
            ext = sel.expected_ids(table, prefix, [True], nxt, nxt)
            if (ext - cur) / (times[nxt] - times[width]) >= cur / times[width]:
                width = nxt
            else:
                break
        return width

    committed = torch.full((n, 4), -1, dtype=torch.int64)
    count = torch.zeros(n, dtype=torch.int64)
    prev = torch.zeros(n, dtype=torch.int64)
    while bool((count < 4).any()):
        live = (count < 4).nonzero().flatten()
        ctx = prev[live]
        drafts, qx = [], []
        for _ in range(7):
            token = torch.multinomial(q_rows[ctx].float(), 1, generator=gen).squeeze(1)
            drafts.append(token)
            qx.append(q_rows[ctx].gather(1, token[:, None]).squeeze(1))
            ctx = token
        drafts = torch.stack(drafts, 1)
        qx = torch.stack(qx, 1).float().tolist()
        widths = [width_of(row) for row in qx]
        for j, chain in enumerate(live.tolist()):
            cur = int(prev[chain])
            for depth in range(widths[j]):
                p = p_rows[cur]
                if depth < widths[j] - 1:
                    x = int(drafts[j, depth])
                    q = q_rows[cur]
                    if rng.random() * q[x].item() < p[x].item():
                        out, accepted = x, True
                    else:
                        res = (p - q).clamp(min=0)
                        out = int(torch.multinomial((res / res.sum()).float(), 1, generator=gen))
                        accepted = False
                else:
                    out = int(torch.multinomial(p.float(), 1, generator=gen))
                    accepted = False
                committed[chain, count[chain]] = out
                count[chain] += 1
                cur = out
                if not accepted or count[chain] >= 4:
                    break
            prev[chain] = cur
    x1 = torch.bincount(committed[:, 0], minlength=vocab)
    pair = torch.bincount(committed[:, 2] * vocab + committed[:, 3], minlength=vocab * vocab)
    two = p_rows @ p_rows  # p'(x3 | anchor) = row 0 of p^3
    p3 = (p_rows @ two)[0]
    exact_pair = p3[:, None] * p_rows
    return x1, p_rows[0], pair, exact_pair, widths


class GateBTests(unittest.TestCase):
    N = int(os.environ.get("QISO_GATE_B_CHAINS", "100000"))

    def test_stopping_rule_keeps_speculative_sampling_exact(self):
        x1, p1, pair, exact_pair, _ = run_chains(self.N, 4, seed=21, draft_noise=2.5)
        self.assertGreater(chi2(x1, p1), 1e-3)
        self.assertGreater(chi2(pair, exact_pair), 1e-3)

    def test_peeking_rule_is_detected(self):
        _, _, pair, exact_pair, _ = run_chains(self.N, 4, seed=21, peek=True, draft_noise=2.5)
        self.assertLess(chi2(pair, exact_pair), 1e-3)


policy = _load("inkling_draft_policy_qiso", "python/sglang/srt/speculative/inkling_draft_policy.py")


class ShadowManifestTests(unittest.TestCase):
    def _manifest(self, **extra):
        payload = {
            "version": 1, "stock_sha256": "7c" * 32, "batch_mode": "width",
            "widths": [{"max_load": None, "thinking": 8, "answer": 8}],
            "pin": {"head": 0, "width": 8},
            "width_selector": {"kind": "qiso-x-table", "path": str(Path(__file__).resolve().parent / "qiso.json"),
                               "sha256": "ab" * 32, "mode": "shadow"},
        }
        payload.update(extra)
        return payload

    def _read(self, payload):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(payload))
            return policy.DraftPolicy.read(path)

    def test_shadow_manifest_round_trips(self):
        rules = self._read(self._manifest())
        self.assertEqual(rules.width_selector_mode, "shadow")
        self.assertEqual(rules.width_selector.sha256, "ab" * 32)
        self.assertEqual(rules.pin, (0, 8))

    def test_shadow_needs_pinned_width_eight_and_a_known_mode(self):
        with self.assertRaises(ValueError):
            self._read(self._manifest(pin={"head": 0, "width": 4},
                                      widths=[{"max_load": None, "thinking": 8, "answer": 4}]))
        bad = self._manifest()
        del bad["pin"]
        with self.assertRaises(ValueError):
            self._read(bad)
        bad = self._manifest()
        bad["width_selector"]["mode"] = "device"
        with self.assertRaises(ValueError):
            self._read(bad)
        bad = self._manifest()
        bad["width_selector"]["path"] = "relative.json"
        with self.assertRaises(ValueError):
            self._read(bad)

    def test_counterfactual_telemetry_pairs_every_policy_on_the_same_rounds(self):
        from types import SimpleNamespace

        req = SimpleNamespace()
        for accepted, width in ((7, 8), (2, 4), (5, 6), (0, 5)):
            policy.record_round(req, True, False, 7, accepted, qiso_width=width)
        shadow = policy.telemetry(req)["thinking"]["stock"]["qiso_x_shadow"]
        self.assertEqual(shadow["rounds"], 4)
        self.assertEqual(shadow["widths"], {"4": 1, "5": 1, "6": 1, "8": 1})
        self.assertEqual(shadow["accepted"], 7 + 2 + 5 + 0)
        self.assertEqual(shadow["fixed_accepted"], {"4": 3 + 2 + 3 + 0, "5": 4 + 2 + 4 + 0,
                                                    "6": 5 + 2 + 5 + 0, "8": 7 + 2 + 5 + 0})
        # Rounds verified below width 8 carry no counterfactual.
        policy.record_round(req, True, False, 3, 3, qiso_width=4)
        self.assertEqual(policy.telemetry(req)["thinking"]["stock"]["qiso_x_shadow"]["rounds"], 4)


class DraftFilterTelemetryTests(unittest.TestCase):
    def test_opt_out_is_validated_and_counted(self):
        from types import SimpleNamespace

        policy.validate_override({"dspark_draft_filter": False}, None)
        with self.assertRaises(ValueError):
            policy.validate_override({"dspark_draft_filter": "no"}, None)
        req = SimpleNamespace()
        policy.record_round(req, True, False, 3, 2, draft_filter=True)
        policy.record_round(req, True, False, 3, 1, draft_filter=False)
        policy.record_round(req, True, False, 3, 1, draft_filter=True)
        summary = policy.telemetry(req)["thinking"]["stock"]
        self.assertEqual(summary["draft_filter"], {"off": 1, "on": 2})


if __name__ == "__main__":
    unittest.main()
