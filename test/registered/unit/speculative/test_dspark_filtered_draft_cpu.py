"""CPU exactness checks for SGLANG_DSPARK_FILTERED_DRAFT (Infera task sal25).

The filtered draft applies the request's top-k/top-p to the draft rows before the
Gumbel-max draw and keeps the filtered logits, which verify turns into q'. These
tests check (1) the keep set against a full-sort reference, (2) that chain
speculative sampling with the filtered q' outputs exactly the filtered target p'
(closed form and a chi-square test over >= 1e5 sampled rounds, plus a broken
control that must be detected), (3) that the kernel's torch path leaves rows
without a filter, and argmax rows, bit-identical, and (4) a micro-benchmark of
the filter on a 200K vocabulary.

``draft_filter.py`` loads standalone (torch only); the kernel test imports
``markov_sample`` and skips when SGLang's kernel stack is not importable.
"""

import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import unittest

import torch

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/speculative/dspark_components/draft_filter.py"
)
spec = importlib.util.spec_from_file_location("dspark_draft_filter", SOURCE)
draft_filter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = draft_filter
spec.loader.exec_module(draft_filter)

DraftFilter = draft_filter.DraftFilter
TOP_K_ALL = 1 << 30


def chi2_sf(x: float, dof: int) -> float:
    """Upper tail of the chi-square distribution (regularized gamma Q)."""
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
    b = z + 1 - a
    c = 1 / tiny
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
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def chi2_test(counts: torch.Tensor, probs: torch.Tensor, min_expected: float = 5.0):
    """Pearson chi-square with cells of expected count < min_expected pooled."""
    n = counts.sum().item()
    expected = probs.double() * n
    big = expected >= min_expected
    observed = counts.double()
    stat = (((observed[big] - expected[big]) ** 2) / expected[big]).sum().item()
    dof = int(big.sum().item()) - 1
    rest_e = expected[~big].sum().item()
    if rest_e > 0:
        rest_o = observed[~big].sum().item()
        stat += (rest_o - rest_e) ** 2 / rest_e
        dof += 1
    # Mass the target gives zero probability must never be observed.
    assert observed[probs == 0].sum().item() == 0, "a zero-probability ID was sampled"
    return stat, max(dof, 1), chi2_sf(stat, max(dof, 1))


def filtered_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    keep = draft_filter.reference_keep_mask(logits, temperature, top_k, top_p)
    scaled = torch.where(keep, logits.double() / temperature, torch.tensor(-math.inf))
    return torch.softmax(scaled, dim=-1)


class KeepSetTests(unittest.TestCase):
    def test_matches_full_sort_reference(self):
        gen = torch.Generator().manual_seed(7)
        vocab = 300
        cases = [
            (0.7, 20, 0.8),
            (0.7, TOP_K_ALL, 0.95),
            (1.0, 5, 1.0),
            (0.3, TOP_K_ALL, 0.5),
            (1.3, 40, 0.99),
            (1.0, 1, 1.0),
            (0.7, 64, 1.0),
        ]
        for temperature, top_k, top_p in cases:
            for trial in range(20):
                logits = torch.randn(vocab, generator=gen) * (1 + trial % 4)
                if trial % 5 == 0:  # ties at the boundary
                    logits = torch.round(logits * 2) / 2
                if trial % 7 == 0:
                    logits[:10] = -math.inf  # vocab-limit padding
                row = logits.clone().view(1, -1)
                flt = DraftFilter(
                    torch.tensor([top_k]), torch.tensor([top_p]), max_top_k=64
                )
                draft_filter.apply_draft_filter_(row, torch.tensor([temperature]), flt)
                kept = torch.isfinite(row[0]) | (logits == -math.inf)
                want, ranks = draft_filter.reference_keep_mask(
                    logits, temperature, top_k, top_p, with_rank_count=True
                )
                expressible = ranks <= 64 or (top_k >= vocab and top_p >= 1)
                if expressible:
                    self.assertTrue(
                        torch.equal(kept, want | (logits == -math.inf)),
                        (temperature, top_k, top_p, trial),
                    )
                else:  # a keep set wider than the cap stays unfiltered (exact)
                    self.assertTrue(torch.equal(row[0], logits), (temperature, top_k, top_p))

    def test_identity_rows_are_bit_identical(self):
        gen = torch.Generator().manual_seed(11)
        for dtype in (torch.float32, torch.bfloat16):
            logits = (torch.randn(4, 1000, generator=gen) * 3).to(dtype)
            before = logits.clone()
            flt = DraftFilter(
                torch.tensor([TOP_K_ALL, 20, 1, 20]),
                torch.tensor([1.0, 0.8, 1.0, 0.8]),
                max_top_k=64,
            )
            # Row 0: no filter. Row 2: top-k 1 (an argmax request). Row 3: argmax row (T = 0).
            temps = torch.tensor([0.7, 0.7, 0.0, 0.0])
            draft_filter.apply_draft_filter_(logits, temps, flt)
            for row in (0, 2, 3):
                self.assertTrue(torch.equal(logits[row], before[row]), (dtype, row))
            self.assertLess(int(torch.isfinite(logits[1]).sum()), 21)

    def test_overflowing_nucleus_is_left_unfiltered(self):
        logits = torch.zeros(1, 500)  # flat: a 0.9 nucleus needs 450 tokens
        before = logits.clone()
        flt = DraftFilter(torch.tensor([TOP_K_ALL]), torch.tensor([0.9]), max_top_k=64)
        draft_filter.apply_draft_filter_(logits, torch.tensor([1.0]), flt)
        self.assertTrue(torch.equal(logits, before))


def _toy(vocab, seed, mismatch, scale=0.8):
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(vocab, vocab, generator=gen) * scale
    draft = target + mismatch * torch.randn(vocab, vocab, generator=gen) * scale
    return target, draft


def _rows_probs(logits_rows, temperature, top_k, top_p):
    return torch.stack(
        [filtered_probs(row, temperature, top_k, top_p) for row in logits_rows]
    )


def simulate_two_ids(target, draft, temperature, top_k, top_p, n, seed, broken=False):
    """Run width-4 chain speculative sampling rounds on n parallel bigram chains
    until each has committed two IDs; return the [vocab, vocab] counts of the first
    two committed IDs. The draft is drawn with apply_draft_filter_ + Gumbel-max
    (the kernel's rule); verify uses the same filtered q' unless ``broken``."""
    vocab = target.shape[0]
    gen = torch.Generator().manual_seed(seed)
    p_rows = _rows_probs(target, temperature, top_k, top_p)  # p'(. | prev)
    q_plain = torch.softmax(draft.double() / temperature, dim=-1)
    flt = DraftFilter(
        torch.full((n,), top_k, dtype=torch.int64),
        torch.full((n,), top_p, dtype=torch.float32),
        max_top_k=64,
    )
    temps = torch.full((n,), temperature)
    prev = torch.zeros(n, dtype=torch.int64)  # anchor 0
    committed = torch.full((n, 2), -1, dtype=torch.int64)
    count = torch.zeros(n, dtype=torch.int64)
    while bool((count < 2).any()):
        active = count < 2
        cur = prev.clone()
        drafts, q_used = [], []
        ctx = cur
        for _ in range(3):
            logits = draft[ctx].clone()
            draft_filter.apply_draft_filter_(logits, temps, flt)
            gumbel = -torch.log(-torch.log(torch.rand(n, vocab, generator=gen, dtype=torch.float64)))
            token = torch.argmax(logits.double() / temperature + gumbel, dim=-1)
            q_filt = torch.softmax(logits.double() / temperature, dim=-1)
            drafts.append(token)
            q_used.append(q_plain[ctx] if broken else q_filt)
            ctx = token
        alive = active.clone()
        ctx = cur
        for depth in range(4):
            p = p_rows[ctx]
            if depth < 3:
                token = drafts[depth]
                q = q_used[depth]
                p_x = p.gather(1, token[:, None]).squeeze(1)
                q_x = q.gather(1, token[:, None]).squeeze(1)
                u = torch.rand(n, generator=gen, dtype=torch.float64)
                accept = u * q_x < p_x
                residual = (p - q).clamp(min=0)
                mass = residual.sum(-1, keepdim=True)
                residual = torch.where(mass > 0, residual / mass.clamp(min=1e-300), p)
                resampled = torch.multinomial(residual.float(), 1, generator=gen).squeeze(1)
                out = torch.where(accept, token, resampled)
            else:
                accept = torch.zeros(n, dtype=torch.bool)
                out = torch.multinomial(p.float(), 1, generator=gen).squeeze(1)
            write = alive & (count < 2)
            idx = count.clamp(max=1)
            committed[write, idx[write]] = out[write]
            count = count + write.long()
            prev = torch.where(alive, out, prev)
            alive = alive & accept & (count < 2)
            ctx = out
            if not bool(alive.any()):
                break
    pairs = committed[:, 0] * vocab + committed[:, 1]
    return torch.bincount(pairs, minlength=vocab * vocab).view(vocab, vocab), p_rows


class ExactnessTests(unittest.TestCase):
    SETTINGS = [(0.7, 20, 0.8), (0.7, TOP_K_ALL, 0.95), (1.0, 3, 1.0), (0.5, 5, 0.7)]

    def test_closed_form_one_row_outputs_filtered_target(self):
        gen = torch.Generator().manual_seed(3)
        worst = 0.0
        for temperature, top_k, top_p in self.SETTINGS:
            for vocab in (8, 30):
                for _ in range(10):
                    t_logits = torch.randn(vocab, generator=gen) * 2
                    d_logits = t_logits + torch.randn(vocab, generator=gen)
                    p = filtered_probs(t_logits, temperature, top_k, top_p)
                    row = d_logits.clone().view(1, -1)
                    flt = DraftFilter(torch.tensor([top_k]), torch.tensor([top_p]), 64)
                    draft_filter.apply_draft_filter_(row, torch.tensor([temperature]), flt)
                    q = torch.softmax(row[0].double() / temperature, dim=-1)
                    accepted = torch.minimum(p, q)
                    residual = (p - q).clamp(min=0)
                    out = accepted + (1 - accepted.sum()) * residual / residual.sum().clamp(min=1e-300)
                    worst = max(worst, (out - p).abs().max().item())
        self.assertLess(worst, 1e-12)

    def test_sampled_rounds_match_the_filtered_target(self):
        n = int(os.environ.get("FILTERED_DRAFT_TEST_ROUNDS", "100000"))
        vocab = 8
        pvals = []
        for i, (temperature, top_k, top_p) in enumerate(self.SETTINGS):
            target, draft = _toy(vocab, seed=100 + i, mismatch=0.6)
            counts, p_rows = simulate_two_ids(
                target, draft, temperature, top_k, top_p, n, seed=200 + i
            )
            exact = p_rows[0][:, None] * p_rows  # p'(x1 | 0) p'(x2 | x1)
            _, _, pval = chi2_test(counts.flatten(), exact.flatten())
            pvals.append(pval)
        # Fisher's method across settings, and no single setting far in the tail.
        fisher = -2 * sum(math.log(max(p, 1e-300)) for p in pvals)
        self.assertGreater(chi2_sf(fisher, 2 * len(pvals)), 1e-3, pvals)
        self.assertGreater(min(pvals), 1e-4, pvals)

    def test_broken_control_is_detected(self):
        # Draft from the filtered q' but verify with the unfiltered q: not exact.
        n = int(os.environ.get("FILTERED_DRAFT_TEST_ROUNDS", "100000"))
        target, draft = _toy(8, seed=101, mismatch=0.6)
        counts, p_rows = simulate_two_ids(
            target, draft, 0.7, 3, 0.8, n, seed=301, broken=True
        )
        exact = p_rows[0][:, None] * p_rows
        _, _, pval = chi2_test(counts.flatten(), exact.flatten())
        self.assertLess(pval, 1e-3)


def _load_markov_sample():
    try:
        from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, exc
    return MarkovSampleStep, None


class KernelPathTests(unittest.TestCase):
    def setUp(self):
        self.step, error = _load_markov_sample()
        if self.step is None:
            self.skipTest(f"markov_sample not importable here: {error}")

    def _run(self, base, bias, temps, flt, dtype, offset=0):
        logits_out = torch.empty(base.shape, dtype=dtype)
        tokens = self.step.torch(
            base_logits=base,
            bias=bias,
            temperatures=temps,
            key=1234,
            offset=torch.tensor([offset], dtype=torch.int64),
            step=1,
            step_dtype=dtype,
            logits_out=logits_out,
            softcap=30.0,
            vocab_limit=base.shape[1] - 8,
            draft_filter=flt,
        )
        return tokens, logits_out

    def test_unfiltered_and_argmax_rows_match_the_plain_kernel(self):
        gen = torch.Generator().manual_seed(5)
        for dtype in (torch.float32, torch.bfloat16):
            base = (torch.randn(4, 1024, generator=gen) * 4).to(dtype)
            bias = (torch.randn(4, 1024, generator=gen)).to(dtype)
            temps = torch.tensor([0.7, 0.0, 1.0, 0.7], dtype=torch.float32)
            flt = DraftFilter(
                torch.tensor([20, 20, TOP_K_ALL, TOP_K_ALL]),
                torch.tensor([0.8, 0.8, 1.0, 1.0]),
                64,
            )
            for offset in range(8):
                plain_t, plain_l = self._run(base, bias, temps, None, dtype, offset)
                filt_t, filt_l = self._run(base, bias, temps, flt, dtype, offset)
                for row in (1, 2, 3):  # argmax row, two rows without a filter
                    self.assertEqual(int(plain_t[row]), int(filt_t[row]))
                    self.assertTrue(torch.equal(plain_l[row], filt_l[row]))
                keep = torch.isfinite(filt_l[0])
                self.assertLessEqual(int(keep.sum()), 20)
                self.assertTrue(bool(keep[filt_t[0]]))
                # Kept entries are the plain kernel's values, unchanged.
                self.assertTrue(torch.equal(filt_l[0][keep], plain_l[0][keep]))

    def test_filtered_draw_follows_the_kept_q(self):
        gen = torch.Generator().manual_seed(9)
        base = torch.randn(1, 256, generator=gen) * 2
        bias = torch.zeros(1, 256)
        temps = torch.tensor([0.7])
        flt = DraftFilter(torch.tensor([12]), torch.tensor([0.9]), 64)
        counts = torch.zeros(256, dtype=torch.int64)
        q = None
        for offset in range(20000):
            token, logits_out = self._run(base, bias, temps, flt, torch.float32, offset)
            counts[token] += 1
            if q is None:
                q = torch.softmax(logits_out[0].double() / 0.7, dim=-1)
        _, _, pval = chi2_test(counts, q)
        self.assertGreater(pval, 1e-3)


class OverheadTests(unittest.TestCase):
    def test_filter_cost_on_a_full_vocab_row_cpu(self):
        """Prints CPU timings (the GPU number gates serving; see the Infera plan)."""
        vocab = 200_064
        for rows in (1, 4):
            logits = torch.randn(rows, vocab).to(torch.bfloat16)
            flt = DraftFilter(
                torch.full((rows,), 20), torch.full((rows,), 0.8), max_top_k=64
            )
            temps = torch.full((rows,), 0.7)
            for _ in range(3):
                draft_filter.apply_draft_filter_(logits.clone(), temps, flt)
            reps = 20
            start = time.perf_counter()
            for _ in range(reps):
                draft_filter.apply_draft_filter_(logits.clone(), temps, flt)
            per_call = (time.perf_counter() - start) / reps * 1e6
            print(f"filtered draft CPU: rows={rows} vocab={vocab} {per_call:.0f} us/step")
            self.assertLess(per_call, 5_000_000)  # informational on CPU


if __name__ == "__main__":
    unittest.main()
