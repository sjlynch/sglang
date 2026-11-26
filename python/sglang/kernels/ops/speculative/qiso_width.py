"""QISO-x verify width in one kernel launch (Infera FLARE-LESSONS section 1.6).

One program evaluates the exact stopping rule of
``sglang.srt.speculative.inkling_width_selector.ExactStopWidth`` in float64 with
the same summation order, so the chosen width matches the reference bit for bit.
Each decision W -> W' reads q of depths 1..W-1 only (``DRAFTED`` is a constexpr
of the decision, not of the input), which keeps the verify length a stopping
time of the proposal sequence. No host reads: it can sit inside a CUDA graph.

Rows are priced with the table of their phase (thinking / answer / tool, task
jf1ay): ``iso`` is [phases, 7, 20] and ``prior`` [phases, 7]; a version 1 table's
answer rows are constant, which reproduces the earlier answer-prior kernel.
"""

import torch
import triton
import triton.language as tl

_WIDTHS = (4, 5, 6, 8)
_DEPTHS = 7


@triton.jit
def _expected_ids(
    Q, QS, PHASE, ACTIVE, ISO, PRIOR,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, DRAFTED: tl.constexpr, BINS: tl.constexpr,
    DEPTHS: tl.constexpr,
):
    total = tl.zeros((), tl.float64)
    for r in tl.static_range(ROWS):
        phase = tl.load(PHASE + r).to(tl.int32)
        active = tl.load(ACTIVE + r) != 0
        survive = tl.full((), 1.0, tl.float64)
        expected = tl.full((), 1.0, tl.float64)
        for k in tl.static_range(WIDTH - 1):
            if k < DRAFTED:
                q = tl.load(Q + r * QS + k).to(tl.float64)
                q = tl.where(q == q, q, 0.0)  # masked decisions may read NaN later depths
                b = tl.minimum(tl.maximum(tl.floor(q * BINS), 0.0), BINS - 1.0).to(tl.int32)
                c = tl.load(ISO + (phase * DEPTHS + k) * BINS + b)
            else:
                c = tl.load(PRIOR + phase * DEPTHS + k)
            survive = survive * c
            expected = expected + survive
        total = total + tl.where(active, expected, 0.0)
    return total


@triton.jit
def _step(
    Q, QS, PHASE, ACTIVE, ISO, PRIOR, t_cur, t_nxt,
    ROWS: tl.constexpr, CUR: tl.constexpr, NXT: tl.constexpr, BINS: tl.constexpr, DEPTHS: tl.constexpr,
):
    cur_e = _expected_ids(Q, QS, PHASE, ACTIVE, ISO, PRIOR, ROWS, CUR, CUR - 1, BINS, DEPTHS)
    nxt_e = _expected_ids(Q, QS, PHASE, ACTIVE, ISO, PRIOR, ROWS, NXT, CUR - 1, BINS, DEPTHS)
    gain = (nxt_e - cur_e) / tl.maximum(t_nxt - t_cur, 1e-9)
    return gain >= cur_e / t_cur


@triton.jit
def _qiso_width_kernel(
    Q, QS, PHASE, ACTIVE, ISO, PRIOR, MS, OUT,
    ROWS: tl.constexpr, BINS: tl.constexpr, DEPTHS: tl.constexpr,
):
    t4 = tl.load(MS + 0).to(tl.float64)
    t5 = tl.load(MS + 1).to(tl.float64)
    t6 = tl.load(MS + 2).to(tl.float64)
    t8 = tl.load(MS + 3).to(tl.float64)
    ext5 = _step(Q, QS, PHASE, ACTIVE, ISO, PRIOR, t4, t5, ROWS, 4, 5, BINS, DEPTHS)
    ext6 = ext5 & _step(Q, QS, PHASE, ACTIVE, ISO, PRIOR, t5, t6, ROWS, 5, 6, BINS, DEPTHS)
    ext8 = ext6 & _step(Q, QS, PHASE, ACTIVE, ISO, PRIOR, t6, t8, ROWS, 6, 8, BINS, DEPTHS)
    width = 4 + ext5.to(tl.int64) + ext6.to(tl.int64) + 2 * ext8.to(tl.int64)
    tl.store(OUT, width)


def qiso_width(
    q: torch.Tensor,
    phase: torch.Tensor,
    active: torch.Tensor,
    iso: torch.Tensor,
    prior: torch.Tensor,
    round_ms: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """q [rows, >= 5] float32 (stride 1 on depth); phase [rows] int64 PHASES ids
    (``ExactStopWidth.phase_tensor``); active [rows] bool; iso [phases, 7, 20] /
    prior [phases, 7] float64 (``ExactStopWidth.iso`` / ``.prior``); round_ms [4]
    (WIDTHS order); out [1] int64 receives the width. ``rows`` is static per
    captured graph."""
    rows = q.shape[0]
    assert q.shape[1] >= 5 and q.stride(1) == 1
    assert iso.dtype == prior.dtype == torch.float64
    assert iso.dim() == 3 and iso.shape[1:] == (_DEPTHS, 20) and iso.is_contiguous()
    assert prior.shape == iso.shape[:2] and prior.is_contiguous()
    assert phase.dtype in (torch.int32, torch.int64) and phase.shape == (rows,)
    _qiso_width_kernel[(1,)](
        q, q.stride(0), phase, active, iso, prior, round_ms, out,
        ROWS=rows, BINS=iso.shape[2], DEPTHS=_DEPTHS, num_warps=1,
    )
    return out
