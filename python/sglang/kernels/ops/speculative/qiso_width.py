"""QISO-x verify width in one kernel launch (Infera FLARE-LESSONS section 1.6).

One program evaluates the exact stopping rule of
``sglang.srt.speculative.inkling_width_selector.ExactStopWidth`` in float64 with
the same summation order, so the chosen width matches the reference bit for bit.
Each decision W -> W' reads q of depths 1..W-1 only (``DRAFTED`` is a constexpr
of the decision, not of the input), which keeps the verify length a stopping
time of the proposal sequence. No host reads: it can sit inside a CUDA graph.
"""

import torch
import triton
import triton.language as tl

_WIDTHS = (4, 5, 6, 8)


@triton.jit
def _expected_ids(
    Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, DRAFTED: tl.constexpr, BINS: tl.constexpr,
):
    total = tl.zeros((), tl.float64)
    for r in tl.static_range(ROWS):
        think = tl.load(THINK + r) != 0
        active = tl.load(ACTIVE + r) != 0
        survive = tl.full((), 1.0, tl.float64)
        expected = tl.full((), 1.0, tl.float64)
        for k in tl.static_range(WIDTH - 1):
            if k < DRAFTED:
                q = tl.load(Q + r * QS + k).to(tl.float64)
                q = tl.where(q == q, q, 0.0)  # masked decisions may read NaN later depths
                b = tl.minimum(tl.maximum(tl.floor(q * BINS), 0.0), BINS - 1.0).to(tl.int32)
                c_think = tl.load(ISO + k * BINS + b)
            else:
                c_think = tl.load(PRIOR + k)
            c = tl.where(think, c_think, tl.load(ANSWER + k))
            survive = survive * c
            expected = expected + survive
        total = total + tl.where(active, expected, 0.0)
    return total


@triton.jit
def _step(
    Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, t_cur, t_nxt,
    ROWS: tl.constexpr, CUR: tl.constexpr, NXT: tl.constexpr, BINS: tl.constexpr,
):
    cur_e = _expected_ids(Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, ROWS, CUR, CUR - 1, BINS)
    nxt_e = _expected_ids(Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, ROWS, NXT, CUR - 1, BINS)
    gain = (nxt_e - cur_e) / tl.maximum(t_nxt - t_cur, 1e-9)
    return gain >= cur_e / t_cur


@triton.jit
def _qiso_width_kernel(
    Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, MS, OUT,
    ROWS: tl.constexpr, BINS: tl.constexpr,
):
    t4 = tl.load(MS + 0).to(tl.float64)
    t5 = tl.load(MS + 1).to(tl.float64)
    t6 = tl.load(MS + 2).to(tl.float64)
    t8 = tl.load(MS + 3).to(tl.float64)
    ext5 = _step(Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, t4, t5, ROWS, 4, 5, BINS)
    ext6 = ext5 & _step(Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, t5, t6, ROWS, 5, 6, BINS)
    ext8 = ext6 & _step(Q, QS, THINK, ACTIVE, ISO, PRIOR, ANSWER, t6, t8, ROWS, 6, 8, BINS)
    width = 4 + ext5.to(tl.int64) + ext6.to(tl.int64) + 2 * ext8.to(tl.int64)
    tl.store(OUT, width)


def qiso_width(
    q: torch.Tensor,
    thinking: torch.Tensor,
    active: torch.Tensor,
    iso: torch.Tensor,
    prior: torch.Tensor,
    answer: torch.Tensor,
    round_ms: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """q [rows, >= 5] float32 (stride 1 on depth), thinking/active [rows] bool,
    iso [7, 20] / prior [7] / answer [7] float64, round_ms [4] (WIDTHS order),
    out [1] int64 receives the width. ``rows`` is static per captured graph."""
    rows = q.shape[0]
    assert q.shape[1] >= 5 and q.stride(1) == 1
    assert iso.dtype == prior.dtype == answer.dtype == torch.float64
    assert iso.shape == (7, 20) and iso.is_contiguous()
    _qiso_width_kernel[(1,)](
        q, q.stride(0), thinking, active, iso, prior, answer, round_ms, out,
        ROWS=rows, BINS=iso.shape[1], num_warps=1,
    )
    return out
