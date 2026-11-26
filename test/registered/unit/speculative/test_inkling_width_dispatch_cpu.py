"""CPU checks for QISO-x active dispatch and the phase tables (Infera task jf1ay).

1. Phase tables: a version 1 table decides exactly as the sal25 rule did (a frozen
   copy of that reference is compared on random batches); version 2 tables with a
   tool table match the device rule bit for bit; the fitter and validation.
2. Decision parity: shadow and active modes share ``QisoDecision``; the active
   read returns the device width, and ``force_width`` changes only the verify.
3. Exactness: chain speculative sampling over batches of four rows (mixed
   thinking / answer / tool phases, joint decision, dead rows dropped) whose
   verify reads ``slice_proposal`` of a width-8 draft commits IDs distributed as
   the target (chi-square, >= 1e5 chains); a slice that misaligns q by one depth
   is detected.
4. Pinned-width identity (the CPU model of gate C's engine half): greedy output
   through the active path at each forced width equals the target's greedy
   sequence and native width-W decoding, 16/16 prompts per width.
5. The worker's dispatch protocol (``Inkling2MTPWorker._qiso_dispatch_round``)
   under stubbed SGLang modules: the chosen width's worker verifies the sliced
   width-8 draft under nested spec overrides; width 8 continues in the drafter.
"""

import dataclasses
import importlib.util
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from types import ModuleType, SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = "python/sglang/srt/speculative/"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sel = _load("sglang.srt.speculative.inkling_width_selector", SPEC + "inkling_width_selector.py")
disp = _load("sglang.srt.speculative.inkling_width_dispatch", SPEC + "inkling_width_dispatch.py")
policy = _load("inkling_draft_policy_dispatch", SPEC + "inkling_draft_policy.py")


# --- helpers -----------------------------------------------------------------


def iso_rows(rng, lo, hi, steep=False):
    rows = []
    for k in range(sel.DEPTHS):
        a, b = (0.05, 0.98) if steep else (lo - 0.02 * k, hi - 0.01 * k)
        rows.append(sorted(min(1.0, max(0.0, a + (b - a) * (i / 19) + rng.uniform(-0.02, 0.02))) for i in range(20)))
    return rows


def v1_payload(seed=0, steep=False):
    rng = random.Random(seed)
    return {
        "version": 1, "kind": sel.TABLE_KIND, "bins": 20, "head_sha256": "7c" * 32,
        "iso": iso_rows(rng, 0.35, 0.95, steep),
        "prior": [0.764, 0.697, 0.684, 0.700, 0.703, 0.710, 0.699],
        "answer": [0.965, 0.959, 0.930, 0.958, 0.958, 0.958, 0.958],
    }


def v2_payload(seed=0, tool=True, round_ms=None, context=None):
    rng = random.Random(seed)
    phases = {
        "thinking": {"iso": iso_rows(rng, 0.35, 0.95, seed % 2 == 0), "prior": [0.76, 0.70, 0.68, 0.70, 0.70, 0.71, 0.70]},
        "answer": {"iso": iso_rows(rng, 0.70, 0.99), "prior": [0.96, 0.95, 0.93, 0.95, 0.95, 0.95, 0.95]},
    }
    if tool:
        phases["tool"] = {"iso": iso_rows(rng, 0.10, 0.999, True), "prior": [0.90, 0.88, 0.86, 0.85, 0.85, 0.84, 0.84]}
    payload = {"version": 2, "kind": sel.TABLE_KIND, "bins": 20, "head_sha256": "7c" * 32, "phases": phases}
    if round_ms is not None:
        payload["round_ms"] = round_ms
    if context is not None:
        payload["context_ms_per_token"] = context
    return payload


def legacy_reference(payload, rows_q, thinking, round_ms):
    """The sal25 (version 1) host reference, frozen: answer rows on the answer prior."""
    iso, prior, answer = payload["iso"], payload["prior"], payload["answer"]

    def expected(prefix, width, drafted):
        total = 0.0
        for q, think in zip(prefix, thinking):
            survive, e = 1.0, 1.0
            for k in range(width - 1):
                if not think:
                    c = answer[k]
                elif k < drafted - 1:
                    c = iso[k][min(int(q[k] * 20), 19)]
                else:
                    c = prior[k]
                survive *= c
                e += survive
            total += e
        return total

    width = 4
    for _, nxt in sel.STEPS:
        prefix = [list(q[: width - 1]) for q in rows_q]
        cur_e, nxt_e = expected(prefix, width, width), expected(prefix, nxt, width)
        if (nxt_e - cur_e) / max(round_ms[nxt] - round_ms[width], 1e-9) >= cur_e / round_ms[width]:
            width = nxt
        else:
            break
    return width


def random_q(rng, rows):
    return [[float(torch.tensor(rng.random() ** rng.choice((0.3, 1, 3)), dtype=torch.float32)) for _ in range(7)]
            for _ in range(rows)]


def random_times(rng):
    base = rng.uniform(8, 20)
    inc = sorted(rng.uniform(0.1, 3.0) for _ in range(3))
    return {4: base, 5: base + inc[0], 6: base + inc[1], 8: base + inc[2]}


def ms_tensor(times):
    return torch.tensor([times[w] for w in sel.WIDTHS], dtype=torch.float64)


FALLBACK = ((10.65, 11.1, 11.6, 12.5), (13.0, 13.6, 14.2, 15.8), (14.6, 15.6, 16.5, 18.0), (16.2, 17.5, 18.8, 20.2))


@dataclasses.dataclass(frozen=True)
class Block:
    draft_tokens: torch.Tensor
    corrected_logits: torch.Tensor
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor
    greedy_draft: bool = False
    vocab_shard: object = None
    noise: object = None


@dataclasses.dataclass(frozen=True)
class Proposal:
    draft_block_ids: torch.Tensor
    draft_block: Block
    confidence: torch.Tensor = None
    confidence_raw: torch.Tensor = None
    acceptance_features: torch.Tensor = None
    folded: bool = False
    verify_ids: torch.Tensor = None


def block_from_q(q_rows, tokens):
    """A width-8 draft block whose kept logits give q_k = q_rows[k][token_k]."""
    bs = len(q_rows)
    vocab = 6
    logits = torch.full((bs, 7, vocab), -30.0)
    for r, row in enumerate(q_rows):
        for k, q in enumerate(row):
            q = min(max(q, 1e-6), 1 - 1e-6)
            logits[r, k, tokens[r][k]] = math.log(q)
            rest = (1 - q) / (vocab - 1)
            for v in range(vocab):
                if v != tokens[r][k]:
                    logits[r, k, v] = math.log(rest)
    return Block(torch.tensor(tokens), logits, torch.zeros(bs, dtype=torch.bool), torch.ones(bs))


# --- 1. phase tables -------------------------------------------------------------


class PhaseTableTests(unittest.TestCase):
    def test_version_one_decides_exactly_as_before(self):
        rng = random.Random(1795669400)
        widths = {w: 0 for w in sel.WIDTHS}
        for trial in range(1500):
            payload = v1_payload(seed=trial % 7, steep=trial % 3 == 0)
            table = sel.WidthTable.from_payload(payload)
            rule = sel.ExactStopWidth(table)
            rows = rng.choice((1, 2, 3, 4))
            q = random_q(rng, rows)
            thinking = [rng.random() < 0.7 for _ in range(rows)]
            times = random_times(rng)
            want = legacy_reference(payload, q, thinking, times)
            self.assertEqual(sel.exact_stop_width_reference(table, q, thinking, times), want)
            got = rule(torch.tensor(q, dtype=torch.float32), torch.tensor(thinking), ms_tensor(times))
            self.assertEqual(int(got), want, trial)
            widths[want] += 1
        self.assertTrue(all(n > 20 for n in widths.values()), widths)

    def test_tool_table_device_rule_matches_reference(self):
        rng = random.Random(7)
        differs = 0
        for trial in range(1200):
            table = sel.WidthTable.from_payload(v2_payload(seed=trial % 5))
            no_tool = sel.WidthTable.from_payload(v2_payload(seed=trial % 5, tool=False))
            rule = sel.ExactStopWidth(table)
            rows = rng.choice((1, 2, 3, 4))
            q = random_q(rng, rows)
            phases = [rng.choice((0, 1, 2)) for _ in range(rows)]
            times = random_times(rng)
            want = sel.exact_stop_width_reference(table, q, phases, times)
            got = rule(torch.tensor(q, dtype=torch.float32), torch.tensor(phases), ms_tensor(times))
            self.assertEqual(int(got), want, trial)
            # Tool rows priced as answers (no tool table) decide differently somewhere.
            differs += sel.exact_stop_width_reference(no_tool, q, phases, times) != want
            # Answer-only rows ignore the tool table.
            answer = [1] * rows
            self.assertEqual(
                sel.exact_stop_width_reference(table, q, answer, times),
                sel.exact_stop_width_reference(no_tool, q, answer, times),
            )
        self.assertGreater(differs, 20)
        self.assertEqual(table.tool_source, "table")
        self.assertEqual(no_tool.tool_source, "answer")

    def test_version_two_validation_and_round_costs(self):
        good = v2_payload(round_ms=[[10.0, 10.5, 11.0, 12.0], [13.0, 13.6, 14.2, 15.8]], context=0.00003)
        table = sel.WidthTable.from_payload(good)
        self.assertEqual(table.version, 2)
        self.assertEqual(table.round_costs(1, FALLBACK), (10.0, 10.5, 11.0, 12.0))
        self.assertEqual(table.round_costs(9, FALLBACK), (13.0, 13.6, 14.2, 15.8))  # last row covers more rows
        got = sel.round_ms_for(table, 2, 10_000, FALLBACK, 0.5)
        self.assertEqual(got, [v + 0.3 for v in (13.0, 13.6, 14.2, 15.8)])
        plain = sel.WidthTable.from_payload(v1_payload())
        self.assertEqual(sel.round_ms_for(plain, 3, 1000, FALLBACK, 0.001), [v + 1.0 for v in FALLBACK[2]])
        for mutate in (
            lambda p: p["phases"].pop("thinking"),
            lambda p: p["phases"].update(code=p["phases"]["answer"]),
            lambda p: p.update(round_ms=[[10.0, 9.0, 11.0, 12.0]]),
            lambda p: p.update(round_ms=[[10.0, 11.0, 12.0]]),
            lambda p: p.update(context_ms_per_token=-1),
            lambda p: p.update(extra=1),
            lambda p: p["phases"]["tool"]["iso"][0].reverse(),
        ):
            bad = json.loads(json.dumps(v2_payload()))
            mutate(bad)
            with self.assertRaises(ValueError):
                sel.WidthTable.from_payload(bad)

    def test_fit_version_two_per_phase_with_tool_fallback(self):
        rng = random.Random(3)

        def rounds(n, slope):
            out = []
            for _ in range(n):
                q = [rng.random() for _ in range(7)]
                a = [min(1.0, max(0.0, 0.2 + slope * x + rng.uniform(-0.1, 0.1))) for x in q]
                out.append([0, 0, 0, *a, *q])
            return out

        payload = sel.fit_table_v2(
            {"thinking": rounds(2000, 0.6), "answer": rounds(50, 0.7), "tool": rounds(800, 0.75)},
            "ab" * 32, answer_prior=[0.958] * 7, round_ms=FALLBACK, sampling="t1",
        )
        table = sel.WidthTable.from_payload(payload)
        self.assertEqual(payload["fit"]["phases"]["answer"]["source"], "prior")
        self.assertEqual(table.tool_source, "table")
        self.assertGreater(table.phases[sel.TOOL].iso[0][19], table.phases[sel.TOOL].iso[0][0] + 0.4)
        self.assertEqual(table.phases[sel.ANSWER].iso[3], (0.958,) * 20)
        few = sel.fit_table_v2({"thinking": rounds(300, 0.6), "tool": rounds(10, 0.7)}, "ab" * 32, answer_prior=[0.95] * 7)
        self.assertEqual(few["fit"]["phases"]["tool"]["source"], "answer-fallback")
        self.assertEqual(sel.WidthTable.from_payload(few).tool_source, "answer")
        with self.assertRaises(ValueError):
            sel.fit_table_v2({"thinking": rounds(10, 0.6)}, "ab" * 32, answer_prior=[0.95] * 7)


# --- 2. decision parity -------------------------------------------------------


class DecisionParityTests(unittest.TestCase):
    def test_shadow_and_active_choose_the_reference_width(self):
        rng = random.Random(11)
        for trial in range(400):
            payload = v2_payload(seed=trial % 4, round_ms=[list(r) for r in FALLBACK] if trial % 2 else None)
            table = sel.WidthTable.from_payload(payload)
            decision = disp.QisoDecision(table, "cpu", FALLBACK, 0.0000332)
            dispatcher = disp.QisoDispatcher(decision)
            rows = rng.choice((1, 2, 3, 4))
            q = random_q(rng, rows)
            tokens = [[rng.randrange(6) for _ in range(7)] for _ in range(rows)]
            block = block_from_q(q, tokens)
            q_dev = sel.drafted_confidence(block.corrected_logits, block.draft_tokens, block.temperatures)
            phases = [rng.choice((0, 1, 2)) for _ in range(rows)]
            context = rng.randrange(0, 80_000)
            times = dict(zip(sel.WIDTHS, decision.round_ms(rows, context)))
            want = sel.exact_stop_width_reference(table, q_dev.tolist(), phases, times)
            shadow = int(decision.decide(block, rows, phases, context)[0])
            active, wait_ns, fallback = dispatcher.read_width(block, rows, phases, context)
            self.assertEqual((shadow, active, fallback), (want, want, False), trial)
            self.assertGreaterEqual(wait_ns, 0)

    def test_force_width_pins_the_verify_and_greedy_draft_falls_back(self):
        table = sel.WidthTable.from_payload(v1_payload())
        decision = disp.QisoDecision(table, "cpu", FALLBACK, 0.0)
        block = block_from_q([[0.99] * 7], [[1] * 7])
        for width in sel.WIDTHS:
            self.assertEqual(disp.QisoDispatcher(decision, force_width=width).read_width(block, 1, [0], 0)[0], width)
        with self.assertRaises(ValueError):
            disp.QisoDispatcher(decision, force_width=7)
        greedy = dataclasses.replace(block, corrected_logits=None)
        self.assertEqual(disp.QisoDispatcher(decision).read_width(greedy, 1, [0], 0), (8, 0, True))

    def test_dispatch_stats_percentiles(self):
        stats = disp.DispatchStats(every=0)
        for i, width in enumerate((4, 5, 6, 8, 8)):
            stats.note(width, 1000 * i, 2000 * (i + 1))
        summary = stats.summary()
        self.assertEqual(summary["widths"], {"4": 1, "5": 1, "6": 1, "8": 2})
        self.assertEqual(summary["bubble_us_p50"], 6.0)


# --- 3. slicing and exactness -------------------------------------------------


class SliceTests(unittest.TestCase):
    def test_slice_keeps_the_drafted_prefix(self):
        gen = torch.Generator().manual_seed(0)
        bs, vocab = 3, 11
        block = Block(
            torch.randint(0, vocab, (bs, 7), generator=gen), torch.randn(bs, 7, vocab, generator=gen),
            torch.tensor([True, False, False]), torch.tensor([1.0, 0.7, 0.9]),
        )
        ids = torch.randint(0, vocab, (bs, 8), generator=gen)
        verify = torch.cat([ids[:, :1], block.draft_tokens], 1)
        proposal = Proposal(ids, block, confidence=torch.rand(bs, 7), confidence_raw=torch.rand(bs, 7),
                            acceptance_features=torch.rand(bs, 7, 2), folded=True, verify_ids=verify)
        for width in sel.WIDTHS:
            cut = disp.slice_proposal(proposal, width)
            self.assertTrue(torch.equal(cut.draft_block.draft_tokens, block.draft_tokens[:, : width - 1]))
            self.assertTrue(torch.equal(cut.draft_block.corrected_logits, block.corrected_logits[:, : width - 1]))
            self.assertTrue(cut.draft_block.corrected_logits.is_contiguous())
            self.assertTrue(torch.equal(cut.verify_ids, verify[:, :width]))
            self.assertTrue(torch.equal(cut.draft_block_ids, ids[:, :width]))
            self.assertTrue(torch.equal(cut.confidence, proposal.confidence[:, : width - 1]))
            self.assertIs(cut.draft_block.greedy_mask, block.greedy_mask)
            self.assertIs(cut.draft_block.temperatures, block.temperatures)
            self.assertFalse(cut.folded)
            self.assertIsNone(cut.acceptance_features)
        with self.assertRaises(ValueError):
            disp.slice_proposal(dataclasses.replace(proposal, draft_block=dataclasses.replace(block, noise=object())), 4)
        with self.assertRaises(ValueError):
            disp.slice_proposal(proposal, 7)


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


def run_active_chains(n, vocab=4, seed=5, misalign=False, group=4):
    """Chains in fixed groups of ``group`` rows (one batch each, phases mixed);
    every round drafts 7 tokens (the width-8 draft), the batch's width comes from
    the rule on the live rows, and verify reads ``slice_proposal`` of the draft.
    Runs until 4 IDs are committed per chain; returns x1 and (x3, x4) counts with
    their exact target laws."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(vocab, vocab, generator=gen) * 1.2
    draft = target + torch.randn(vocab, vocab, generator=gen) * 2.5
    p_rows = torch.softmax(target.double(), -1)
    q_rows = torch.softmax(draft.double(), -1)
    table = sel.WidthTable.from_payload(v2_payload(seed=0, round_ms=[[10.0, 10.25, 10.5, 11.0]] * 4))
    phase_of = [i % 3 for i in range(n)]
    committed = torch.full((n, 4), -1, dtype=torch.int64)
    count = torch.zeros(n, dtype=torch.int64)
    prev = torch.zeros(n, dtype=torch.int64)
    while bool((count < 4).any()):
        live = (count < 4).nonzero().flatten()
        ctx = prev[live]
        tokens, logits = [], []
        for _ in range(7):
            logits.append(draft[ctx].float())
            token = torch.multinomial(q_rows[ctx].float(), 1, generator=gen).squeeze(1)
            tokens.append(token)
            ctx = token
        tokens = torch.stack(tokens, 1)
        logits = torch.stack(logits, 1)  # [m, 7, V], the kept step logits
        q = sel.drafted_confidence(logits, tokens, torch.ones(len(live))).tolist()
        # One decision per live group (dead rows leave the batch).
        widths = torch.empty(len(live), dtype=torch.int64)
        live_list = live.tolist()
        start = 0
        while start < len(live_list):
            g = live_list[start] // group
            end = start
            while end < len(live_list) and live_list[end] // group == g:
                end += 1
            rows = list(range(start, end))
            times = dict(zip(sel.WIDTHS, sel.round_ms_for(table, len(rows), 0, FALLBACK, 0.0)))
            width = sel.exact_stop_width_reference(
                table, [q[r] for r in rows], [phase_of[live_list[r]] for r in rows], times
            )
            widths[start:end] = width
            start = end
        ids = torch.cat([prev[live].view(-1, 1), tokens], 1)
        block = Block(tokens, logits, torch.zeros(len(live), dtype=torch.bool), torch.ones(len(live)))
        for width in sel.WIDTHS:
            pick = (widths == width).nonzero().flatten()
            if not len(pick):
                continue
            sub_block = Block(tokens[pick], logits[pick], block.greedy_mask[pick], block.temperatures[pick])
            cut = disp.slice_proposal(Proposal(ids[pick], sub_block, verify_ids=ids[pick]), width)
            kept = cut.draft_block.corrected_logits
            if misalign:  # q read one depth late: the verify no longer sees the q x was drawn from
                kept = torch.cat([sub_block.corrected_logits[:, 1:width], sub_block.corrected_logits[:, -1:]], 1)[:, : width - 1]
            qprob = torch.softmax(kept.double(), -1)
            chains = live[pick]
            cur = prev[chains].clone()
            running = torch.ones(len(pick), dtype=torch.bool)
            for depth in range(width - 1):
                x = cut.draft_block.draft_tokens[:, depth]
                p = p_rows[cur]
                qd = qprob[:, depth]
                u = torch.rand(len(pick), generator=gen, dtype=torch.float64)
                px, qx = p.gather(1, x.view(-1, 1)).squeeze(1), qd.gather(1, x.view(-1, 1)).squeeze(1)
                accept = u * qx < px
                residual = (p - qd).clamp(min=0)
                residual = residual / residual.sum(1, keepdim=True).clamp(min=1e-300)
                resample = torch.multinomial(residual.float(), 1, generator=gen).squeeze(1)
                out = torch.where(accept, x, resample)
                write = running & (count[chains] < 4)
                idx = chains[write]
                committed[idx, count[idx]] = out[write]
                count[idx] += 1
                cur = torch.where(write, out, cur)
                running = running & accept & (count[chains] < 4)
            write = running & (count[chains] < 4)
            if write.any():
                bonus = torch.multinomial(p_rows[cur].float(), 1, generator=gen).squeeze(1)
                idx = chains[write]
                committed[idx, count[idx]] = bonus[write]
                count[idx] += 1
                cur = torch.where(write, bonus, cur)
            prev[chains] = cur
    x1 = torch.bincount(committed[:, 0], minlength=vocab)
    pair = torch.bincount(committed[:, 2] * vocab + committed[:, 3], minlength=vocab * vocab)
    p3 = (p_rows @ (p_rows @ p_rows))[0]
    return x1, p_rows[0], pair, p3[:, None] * p_rows


class ExactnessTests(unittest.TestCase):
    N = int(os.environ.get("QISO_DISPATCH_CHAINS", "100000"))

    def test_active_dispatch_keeps_speculative_sampling_exact(self):
        x1, p1, pair, exact_pair = run_active_chains(self.N)
        self.assertGreater(chi2(x1, p1), 1e-3)
        self.assertGreater(chi2(pair, exact_pair), 1e-3)

    def test_misaligned_slice_is_detected(self):
        x1, p1, pair, exact_pair = run_active_chains(self.N, misalign=True)
        self.assertLess(min(chi2(x1, p1), chi2(pair, exact_pair)), 1e-3)


# --- 4. pinned-width identity -----------------------------------------------


def greedy_decode(target, start, length, width, draft_tokens_fn):
    """Greedy chain verify: accept a draft iff it is the target argmax; the
    mismatch and the bonus commit the target argmax. Returns (ids, rounds)."""
    out, cur, rounds = [], start, 0
    while len(out) < length:
        rounds += 1
        drafts = draft_tokens_fn(cur)[: width - 1]
        for x in drafts:
            best = int(target[cur].argmax())
            out.append(best)
            cur = best
            if x != best or len(out) >= length:
                break
        else:
            best = int(target[cur].argmax())
            out.append(best)
            cur = best
    return out[:length], rounds


class PinnedWidthIdentityTests(unittest.TestCase):
    def test_forced_widths_reproduce_greedy_output_16_of_16(self):
        vocab = 12
        gen = torch.Generator().manual_seed(9)
        target = torch.randn(vocab, vocab, generator=gen)
        draft = target + torch.randn(vocab, vocab, generator=gen) * 0.8
        table = sel.WidthTable.from_payload(v1_payload())
        decision = disp.QisoDecision(table, "cpu", FALLBACK, 0.0)

        def width8_draft(seed):
            g = torch.Generator().manual_seed(seed)

            def fn(cur):
                tokens, ctx = [], cur
                for _ in range(7):
                    ctx = int(torch.multinomial(torch.softmax(draft[ctx], -1), 1, generator=g))
                    tokens.append(ctx)
                return tokens
            return fn

        identical = {w: 0 for w in sel.WIDTHS}
        for prompt in range(16):
            reference, _ = greedy_decode(target, prompt % vocab, 64, 1, lambda cur: [])
            for width in sel.WIDTHS:
                dispatcher = disp.QisoDispatcher(decision, force_width=width)

                def active_fn(cur, fn=width8_draft(prompt), dispatcher=dispatcher):
                    tokens = fn(cur)
                    logits = draft[[cur] + tokens[:-1]].unsqueeze(0)
                    block = Block(torch.tensor([tokens]), logits, torch.zeros(1, dtype=torch.bool), torch.ones(1))
                    chosen, _, _ = dispatcher.read_width(block, 1, [0], 0)
                    cut = disp.slice_proposal(Proposal(torch.tensor([[cur] + tokens]), block), chosen)
                    return cut.draft_block.draft_tokens[0].tolist()

                active, active_rounds = greedy_decode(target, prompt % vocab, 64, width, active_fn)
                native, _ = greedy_decode(target, prompt % vocab, 64, width, width8_draft(10_000 + prompt))
                pinned, pinned_rounds = greedy_decode(target, prompt % vocab, 64, width, width8_draft(prompt))
                identical[width] += active == native == pinned == reference
                self.assertEqual(active_rounds, pinned_rounds)  # same drafts: same rounds
        self.assertEqual(identical, {w: 16 for w in sel.WIDTHS})


# --- manifests and telemetry ---------------------------------------------------


class ActiveManifestTests(unittest.TestCase):
    def _manifest(self, mode="active", **selector):
        row = {"kind": "qiso-x-table", "path": str(Path(__file__).resolve().parent / "qiso.json"),
               "sha256": "ab" * 32, "mode": mode}
        row.update(selector)
        return {
            "version": 1, "stock_sha256": "7c" * 32, "batch_mode": "width",
            "widths": [{"max_load": None, "thinking": 8, "answer": 8}],
            "pin": {"head": 0, "width": 8}, "width_selector": row,
        }

    def _read(self, payload):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(payload))
            return policy.DraftPolicy.read(path)

    def test_active_manifest_captures_every_width(self):
        rules = self._read(self._manifest())
        self.assertEqual(rules.width_selector_mode, "active")
        self.assertEqual(rules.capture_widths, policy.WIDTHS)
        self.assertIsNone(rules.width_selector_force)
        self.assertEqual(self._read(self._manifest(force_width=5)).width_selector_force, 5)
        shadow = self._read(self._manifest("shadow"))
        self.assertEqual(shadow.capture_widths, (4, 8))

    def test_active_manifest_rejections(self):
        for payload in (
            self._manifest(force_width=7),
            self._manifest("shadow", force_width=4),
            self._manifest(extra=True),
            dict(self._manifest(), batch_mode="control"),
        ):
            with self.assertRaises(ValueError):
                self._read(payload)

    def test_phase_ids_cover_tool_and_grammar_rows(self):
        reqs = [
            SimpleNamespace(_inkling_draft_phase=SimpleNamespace(channel="thinking"), grammar=None),
            SimpleNamespace(_inkling_draft_phase=SimpleNamespace(channel="answer"), grammar=None),
            SimpleNamespace(_inkling_draft_phase=SimpleNamespace(channel="tool"), grammar=None),
            SimpleNamespace(_inkling_draft_phase=SimpleNamespace(channel="answer"), grammar=object()),
            SimpleNamespace(grammar=None),
        ]
        self.assertEqual(policy.qiso_phase_ids(reqs), [0, 1, 2, 2, 1])
        self.assertEqual(sel.PHASES[policy.QISO_TOOL], "tool")

    def test_active_telemetry(self):
        req = SimpleNamespace()
        policy.record_round(req, True, False, 3, 2, qiso_active=(4, 900, 30_000, False))
        policy.record_round(req, True, False, 7, 7, qiso_active=(8, 1100, None, True))
        active = policy.telemetry(req)["thinking"]["stock"]["qiso_x_active"]
        self.assertEqual(active["rounds"], 2)
        self.assertEqual(active["widths"], {"4": 1, "5": 0, "6": 0, "8": 1})
        self.assertEqual(active["accepted"], 9)
        self.assertEqual((active["wait_ns"], active["bubble_ns"], active["bubble_rounds"], active["fallback"]),
                         (2000, 30_000, 1, 1))
        self.assertNotIn("qiso_x_shadow", policy.telemetry(req)["thinking"]["stock"])


class RoundLogTests(unittest.TestCase):
    def test_round_log_rows(self):
        gen = torch.Generator().manual_seed(1)
        bs, vocab = 2, 5
        tokens = torch.randint(0, vocab, (bs, 7), generator=gen)
        block = Block(tokens, torch.randn(bs, 7, vocab, generator=gen), torch.zeros(bs, dtype=torch.bool), torch.ones(bs))
        target = torch.randn(bs * 8, vocab, generator=gen)
        reqs = [SimpleNamespace(rid="a", output_ids=[1, 2]), SimpleNamespace(rid="b", output_ids=[])]
        with TemporaryDirectory() as tmp:
            log = disp.QisoRoundLog(Path(tmp) / "rounds.jsonl", "cpu")
            log.record(block, bs, target, torch.tensor([3, 8]), reqs, [0, 2])
            log._file.close()
            rows = [json.loads(line) for line in (Path(tmp) / "rounds.jsonl").read_text().splitlines()]
        self.assertEqual([(r["rid"], r["pos"], r["phase"], r["cl"]) for r in rows], [("a", 2, 0, 2), ("b", 0, 2, 7)])
        q = sel.drafted_confidence(block.corrected_logits, tokens, block.temperatures)
        p = torch.softmax(target.view(bs, 8, vocab)[:, :7], -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        for r, row in enumerate(rows):
            for k in range(7):
                self.assertAlmostEqual(row["q"][k], float(q[r, k]), places=4)
                self.assertAlmostEqual(row["a"][k], min(1.0, float(p[r, k] / q[r, k])), places=4)


# --- 5. worker dispatch protocol (stubbed SGLang) -------------------------------


def _load_worker():
    """inkling2_mtp_worker.py with SGLang's runtime replaced by stubs."""
    overrides = []

    @contextmanager
    def override(**values):
        overrides.append(values["speculative_num_draft_tokens"])
        yield

    stubs = {
        "sglang.srt.mem_cache.swa_memory_pool": {"SWAKVPool": type("SWAKVPool", (), {})},
        "sglang.srt.model_executor.runner": {"DecodeCudaGraphRunner": type("DecodeCudaGraphRunner", (), {})},
        "sglang.srt.runtime_context": {
            name: (lambda: SimpleNamespace(override=override))
            for name in ("get_disagg", "get_exec", "get_lora", "get_parallel", "get_schedule", "get_spec")
        },
        "sglang.srt.speculative.draft_worker_common": {"DraftWorkerBundle": object},
        "sglang.srt.speculative.dspark_components.dspark_verify": {
            "TargetVerifyExecutor": type("TargetVerifyExecutor", (), {"__init__": lambda self, **kw: None}),
        },
        "sglang.srt.speculative.dspark_components.dspark_worker_v2": {"DSparkWorkerV2": type("DSparkWorkerV2", (), {})},
        "sglang.srt.utils.device": {"async_h2d": lambda args, *, dtype, device: torch.tensor(args, dtype=dtype)},
    }
    saved = {name: sys.modules.get(name) for name in [*stubs, "sglang.srt.speculative.inkling_draft_policy"]}
    try:
        for name, attrs in stubs.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
        sys.modules["sglang.srt.speculative.inkling_draft_policy"] = policy
        worker_mod = _load("inkling2_mtp_worker_stubbed", SPEC + "inkling2_mtp_worker.py")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return worker_mod, overrides


class FakeWidthWorker:
    def __init__(self, width):
        self.width = width
        self._verify_executor = SimpleNamespace(launched_ns=-1)
        self.calls = []

    def _forward_decode(self, batch, on_publish, grammar_barrier, *, after_propose=None, injected_proposal=None):
        self.calls.append(injected_proposal)
        if injected_proposal is None:
            gamma = self.width - 1  # this worker's own native draft
            tokens = [[1, 2, 3, 4, 5, 0, 1][:gamma]] * len(batch.reqs)
            q = [[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]] * len(batch.reqs)
            block = block_from_q(q, [row + [0] * (7 - gamma) for row in tokens])
            block = dataclasses.replace(
                block, draft_tokens=block.draft_tokens[:, :gamma], corrected_logits=block.corrected_logits[:, :gamma]
            )
            proposal = Proposal(torch.zeros(len(batch.reqs), self.width, dtype=torch.long), block, folded=True)
            if after_propose is not None:
                assert self.width == 8
                redirected = after_propose(proposal)
                if redirected is not None:
                    return redirected
        else:
            proposal = injected_proposal
        self._verify_executor.launched_ns = perf_counter_ns()
        return SimpleNamespace(verified=proposal.draft_block.draft_tokens.shape[1] + 1, folded=proposal.folded)


class WorkerProtocolTests(unittest.TestCase):
    def test_dispatch_round_routes_the_sliced_draft_to_the_chosen_width(self):
        worker_mod, overrides = _load_worker()
        table = sel.WidthTable.from_payload(v1_payload())
        decision = disp.QisoDecision(table, "cpu", FALLBACK, 0.0)
        batch = SimpleNamespace(reqs=[
            SimpleNamespace(origin_input_ids=[1] * 10, output_ids=[2] * 3, grammar=None,
                            _inkling_draft_phase=SimpleNamespace(channel="thinking")),
        ])
        for forced in (None, *sel.WIDTHS):
            workers = {w: FakeWidthWorker(w) for w in sel.WIDTHS}
            worker = worker_mod.Inkling2MTPWorker.__new__(worker_mod.Inkling2MTPWorker)
            worker._batch_width_index = {w: i for i, w in enumerate(sel.WIDTHS)}
            worker._batch_workers = tuple(workers[w] for w in sel.WIDTHS)
            worker._qiso_dispatcher = disp.QisoDispatcher(decision, force_width=forced)
            overrides.clear()
            result = worker._qiso_dispatch_round(batch, None, None)
            width, wait_ns, bubble, fallback = result.inkling_qiso_active
            expected = forced if forced is not None else 8  # confident drafts extend to 8
            self.assertEqual((width, result.verified, fallback), (expected, expected, False))
            self.assertIsNotNone(bubble)
            self.assertGreaterEqual(bubble, 0)
            if expected == 8:
                self.assertEqual(overrides, [8])
                self.assertEqual(workers[8].calls, [None])
                self.assertTrue(result.folded)  # the drafter's own graphs: fold allowed
            else:
                self.assertEqual(overrides, [8, expected])
                injected = workers[expected].calls[0]
                self.assertEqual(injected.draft_block.draft_tokens.shape[1], expected - 1)
                self.assertFalse(result.folded)
                self.assertTrue(all(not workers[w].calls for w in sel.WIDTHS if w not in (8, expected)))
        self.assertEqual(worker._qiso_dispatcher.stats.rounds, 1)

    def test_width_arms_share_one_process(self):
        worker_mod, overrides = _load_worker()
        table = sel.WidthTable.from_payload(v1_payload())
        manifest = ActiveManifestTests()._manifest()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(manifest))
            rules = policy.DraftPolicy.read(path)

        def req(arm):
            custom = {} if arm is None else {"inkling_width_arm": arm}
            return SimpleNamespace(
                origin_input_ids=[policy.THINK_START], output_ids=[5] * 4, grammar=None,
                sampling_params=SimpleNamespace(custom_params=custom),
            )

        for arms, want_width, want_arm in (
            ((None, None), 8, ("qiso-x", False)),
            (("w4", "w4"), 4, ("w4", False)),
            (("w6", "w6"), 6, ("w6", False)),
            (("p3", "p3"), None, ("p3", False)),  # P3's own choice from its 0.70 prior
            (("w6", "p3"), 4, ("w4", True)),  # rows disagree: the w4 control
            (("qiso-x@5", "qiso-x@5"), 5, ("qiso-x@5", False)),  # dispatch path, verify forced to 5
            (("qiso-x@8", "qiso-x@8"), 8, ("qiso-x@8", False)),
        ):
            workers = {w: FakeWidthWorker(w) for w in sel.WIDTHS}
            worker = worker_mod.Inkling2MTPWorker.__new__(worker_mod.Inkling2MTPWorker)
            worker.inkling_draft_policy = rules
            worker._filtered_draft = False
            worker._batch_width_index = {w: i for i, w in enumerate(sel.WIDTHS)}
            worker._batch_workers = tuple(workers[w] for w in sel.WIDTHS)
            worker._qiso_dispatcher = disp.QisoDispatcher(disp.QisoDecision(table, "cpu", FALLBACK, 0.0))
            batch = SimpleNamespace(
                reqs=[req(arm) for arm in arms], is_extend_in_batch=False,
                forward_mode=SimpleNamespace(is_decode=lambda: True),
            )
            result = worker.forward_batch_generation(batch)
            if want_width is None:
                want_width = policy.select_tracking_width(batch.reqs, (True, True))
                self.assertNotEqual(want_width, 8)  # differs from the qiso-x round above
            self.assertEqual(result.verified, want_width, arms)
            self.assertEqual(result.inkling_width_arm, want_arm)
            self.assertEqual(result.inkling_track_acceptance, want_arm[0] == "p3")
            self.assertEqual(getattr(result, "inkling_qiso_active", None) is not None,
                             want_arm[0].startswith("qiso-x"))
            if want_arm[0].startswith("qiso-x@"):
                self.assertEqual(result.inkling_qiso_active[0], want_width)
            self.assertEqual(result.inkling_draft_route[0], (True, True))

    def test_width_arm_validation_and_telemetry(self):
        manifest = ActiveManifestTests()._manifest()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(manifest))
            active = policy.DraftPolicy.read(path)
            path.write_text(json.dumps(ActiveManifestTests()._manifest("shadow")))
            shadow = policy.DraftPolicy.read(path)
        for arm in policy.WIDTH_ARMS:
            policy.validate_override({"inkling_width_arm": arm}, active)
        for custom, rules in (({"inkling_width_arm": "w7"}, active), ({"inkling_width_arm": "p3"}, shadow),
                              ({"inkling_width_arm": "p3"}, None)):
            with self.assertRaises(ValueError):
                policy.validate_override(custom, rules)
        r = SimpleNamespace()
        policy.record_round(r, True, False, 3, 1, width_arm=("w4", False))
        policy.record_round(r, True, False, 3, 2, width_arm=("w4", True))
        policy.record_round(r, True, False, 7, 2, width_arm=("qiso-x", False), qiso_active=(8, 0, None, False))
        arm = policy.telemetry(r)["thinking"]["stock"]["width_arm"]
        self.assertEqual(arm, {"rounds": {"w4": 2, "qiso-x": 1}, "mixed": 1})


if __name__ == "__main__":
    unittest.main()
