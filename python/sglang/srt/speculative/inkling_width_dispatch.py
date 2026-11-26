"""QISO-x active dispatch: the per-step width drives the verify (Infera task jf1ay).

Shadow mode (task sal25) decides a width on the device every round and only
records it; verify stays at width 8. Active mode makes the decision act:

1. Draft once with the width-8 graph (head 0): seven Markov depths and their kept
   step logits.
2. Decide W in {4, 5, 6, 8} on the device from those drafts' q
   (``ExactStopWidth`` / the fused ``qiso_width`` kernel), with the row phases
   (thinking / answer / tool) and the host-written round costs of this batch.
3. Read W on the host: one bounded sync on an event recorded right after the
   decision (``read_width``). It waits for the draft graph the host has already
   launched, so its wait is draft time, not added time; the added time is the
   host work between the wake-up and the verify graph's launch (``bubble``).
4. Verify with the pre-captured width-W verify graph and its rows, fed the first
   W - 1 drafts of the width-8 proposal (``slice_proposal``). W == 8 continues in
   the width-8 worker unchanged.

Why one sync and not a CUDA conditional node: every per-width verify graph needs
host-staged inputs (the verify window plan, attention metadata, KV slot table) and
returns host-visible shapes (accept lengths at its own stride) to the scheduler.
A SWITCH node must hold the whole verify + accept + commit of every width with
inputs staged for all four widths before the decision; that is the follow-up if
the measured bubble fails gate C (bench_qiso_gate_c.py measures both).

Exactness: W reads q of depths 1..W-1 only (the stopping-time slice in
``ExactStopWidth``), and the width-W verify sees exactly the width-8 draft's first
W - 1 tokens with the logits they were drawn from. Chain speculative sampling on
a prefix chosen by a stopping time of the proposal is exact.

Standalone (stdlib + torch); SGLang structs are replaced by field.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import deque
from time import perf_counter_ns
from typing import Optional, Sequence

from sglang.srt.speculative.inkling_width_selector import (
    WIDTHS,
    ExactStopWidth,
    WidthTable,
    drafted_confidence,
    round_ms_for,
)

logger = logging.getLogger(__name__)

ROUND_LOG_ENV = "SGLANG_INKLING_QISO_ROUND_LOG"
DRAFT_WIDTH = WIDTHS[-1]


def _replace(obj, **changes):
    try:
        import msgspec
    except ImportError:  # pragma: no cover - SGLang always has msgspec
        msgspec = None
    if msgspec is not None and isinstance(obj, msgspec.Struct):
        return msgspec.structs.replace(obj, **changes)
    return dataclasses.replace(obj, **changes)


def slice_proposal(proposal, width: int):
    """The width-``width`` proposal a width-8 draft implies: the anchor and the
    first ``width - 1`` drafted tokens with their kept step logits, confidence
    and folded verify ids. ``folded`` is cleared: the width-W graphs' own draft
    buffers were not written this round, so the accept reads these tensors."""
    gamma = width - 1
    block = proposal.draft_block
    if width not in WIDTHS:
        raise ValueError(f"QISO-x dispatch width must be one of {WIDTHS}")
    if block.draft_tokens.shape[1] < gamma:
        raise ValueError("QISO-x dispatch needs the width-8 draft (7 depths)")
    if block.vocab_shard is not None or block.noise is not None:
        raise ValueError("QISO-x dispatch does not slice a vocab-sharded draft")

    def cut(tensor):
        return None if tensor is None else tensor[:, :gamma].contiguous()

    sliced = _replace(
        block,
        draft_tokens=cut(block.draft_tokens),
        corrected_logits=cut(block.corrected_logits),
    )
    return _replace(
        proposal,
        draft_block_ids=proposal.draft_block_ids[:, :width].contiguous(),
        draft_block=sliced,
        confidence=cut(proposal.confidence),
        confidence_raw=cut(proposal.confidence_raw),
        acceptance_features=None,
        folded=False,
        verify_ids=None if proposal.verify_ids is None else proposal.verify_ids[:, :width].contiguous(),
    )


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


class DispatchStats:
    """Process-level dispatch counters (logged every ``every`` rounds)."""

    def __init__(self, window: int = 4096, every: int = 2000):
        self.rounds = 0
        self.widths = {w: 0 for w in WIDTHS}
        self.fallback = 0
        self.wait_ns = deque(maxlen=window)
        self.bubble_ns = deque(maxlen=window)
        self.every = every

    def note(self, width, wait_ns, bubble_ns, fallback=False):
        self.rounds += 1
        self.widths[width] += 1
        self.fallback += bool(fallback)
        if wait_ns is not None:
            self.wait_ns.append(wait_ns)
        if bubble_ns is not None:
            self.bubble_ns.append(bubble_ns)
        if self.every and self.rounds % self.every == 0:
            logger.info("QISO-x dispatch: %s", json.dumps(self.summary()))

    def summary(self):
        def us(values, fraction):
            value = _percentile(list(values), fraction)
            return None if value is None else round(value / 1000, 2)

        return {
            "rounds": self.rounds,
            "widths": {str(w): n for w, n in self.widths.items()},
            "fallback": self.fallback,
            "wait_us_p50": us(self.wait_ns, 0.5),
            "bubble_us_p50": us(self.bubble_ns, 0.5),
            "bubble_us_p90": us(self.bubble_ns, 0.9),
        }


class QisoDecision:
    """The device decision shared by shadow and active modes, so both choose the
    same width on the same inputs (decision parity)."""

    def __init__(self, table: WidthTable, device, fallback_costs, fallback_context_ms: float, h2d=None):
        import torch

        self.table = table
        self.device = torch.device(device)
        self.rule = ExactStopWidth(table, device=self.device)
        self.fallback_costs = fallback_costs
        self.fallback_context_ms = fallback_context_ms
        self._h2d = h2d or self._default_h2d
        self._cuda = self.device.type == "cuda"

    def _default_h2d(self, values, *, dtype, device):
        import torch

        host = torch.tensor(values, dtype=dtype, pin_memory=self._cuda)
        return host.to(device, non_blocking=True)

    def round_ms(self, rows: int, context_tokens: int):
        return round_ms_for(self.table, rows, context_tokens, self.fallback_costs, self.fallback_context_ms)

    def decide(self, block, bs: int, phases: Sequence[int], context_tokens: int, out=None):
        """Write the width into ``out`` ([1] int64 on the device; allocated when
        None) from this round's drafted q; no host read. None if the block kept
        no step logits (greedy draft): the caller then verifies at width 8."""
        import torch

        if block.corrected_logits is None:
            return None
        q = drafted_confidence(
            block.corrected_logits[:bs], block.draft_tokens[:bs], block.temperatures[:bs]
        )
        phase = self._h2d(list(phases), dtype=torch.int64, device=self.device)
        round_ms = self._h2d(self.round_ms(bs, context_tokens), dtype=torch.float64, device=self.device)
        if out is None:
            out = torch.empty(1, dtype=torch.int64, device=self.device)
        if self._cuda:
            from sglang.kernels.ops.speculative.qiso_width import qiso_width

            qiso_width(
                q, phase, torch.ones(bs, dtype=torch.bool, device=self.device),
                self.rule.iso, self.rule.prior, round_ms, out,
            )
        else:
            out.copy_(self.rule(q, phase, round_ms).view(1))
        return out


class QisoDispatcher:
    """Active mode: decide on the device, then one bounded host read."""

    def __init__(self, decision: QisoDecision, force_width: Optional[int] = None, stats: Optional[DispatchStats] = None):
        import torch

        if force_width is not None and force_width not in WIDTHS:
            raise ValueError(f"QISO-x force_width must be one of {WIDTHS}")
        self.decision = decision
        self.force_width = force_width
        self.stats = stats or DispatchStats()
        cuda = decision.device.type == "cuda"
        self._width_dev = torch.empty(1, dtype=torch.int64, device=decision.device)
        self._width_host = torch.empty(1, dtype=torch.int64, pin_memory=cuda)
        self._ready = torch.cuda.Event() if cuda else None
        self.sync_return_ns = None

    def read_width(self, block, bs, phases, context_tokens, force_width=None):
        """Returns (width, wait_ns, fallback). The only host sync of the round.
        ``force_width`` (or the manifest's) pins the verify after the decision ran."""
        force_width = force_width if force_width is not None else self.force_width
        out = self.decision.decide(block, bs, phases, context_tokens, out=self._width_dev)
        if out is None:
            self.sync_return_ns = perf_counter_ns()
            return DRAFT_WIDTH, 0, True
        self._width_host.copy_(out, non_blocking=self._ready is not None)
        start = perf_counter_ns()
        if self._ready is not None:
            self._ready.record()
            self._ready.synchronize()
        self.sync_return_ns = perf_counter_ns()
        width = int(self._width_host[0])
        if width not in WIDTHS:
            raise RuntimeError(f"QISO-x device rule returned width {width}")
        if force_width is not None:
            # Pinned-width identity: the decision still runs (same cost path), the
            # verify uses the pinned width.
            width = force_width
        return width, self.sync_return_ns - start, False

    def bubble_ns(self, launched_ns):
        """Host time from the sync's return to the verify graph's launch."""
        if launched_ns is None or self.sync_return_ns is None or launched_ns < self.sync_return_ns:
            return None
        return launched_ns - self.sync_return_ns


class QisoRoundLog:
    """Width-8 round log for fitting phase tables (SGLANG_INKLING_QISO_ROUND_LOG,
    shadow mode only): per row and round ``{rid, pos, phase, cl, a, q}`` with
    a_k = min(1, p_k / q_k), p the target's temperature softmax at the drafted
    token (no top-k/top-p filter: fit at the sampling regime of the log) and cl
    the accepted drafts. The device copy is read only once its event completed,
    so logging adds no sync."""

    def __init__(self, path, device):
        self.path = path
        self.pending = deque()
        self.device = device
        self._file = open(path, "a", encoding="utf-8")
        self.rows = 0

    @classmethod
    def from_environment(cls, device):
        path = os.environ.get(ROUND_LOG_ENV)
        return cls(path, device) if path else None

    def record(self, block, bs, target_logits, accept_lens, reqs, phases):
        import torch

        if block.corrected_logits is None or target_logits is None or accept_lens is None:
            return
        gamma = block.draft_tokens.shape[1]
        if gamma != DRAFT_WIDTH - 1:
            return
        tokens = block.draft_tokens[:bs].long()
        temps = block.temperatures[:bs].float().clamp(min=1e-5).view(-1, 1, 1)
        q = drafted_confidence(block.corrected_logits[:bs], tokens, block.temperatures[:bs])
        logits = target_logits.view(-1, DRAFT_WIDTH, target_logits.shape[-1])[:bs, :gamma].float() / temps
        p = torch.exp(logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1))
        a = torch.minimum(torch.ones_like(p), p / q.clamp(min=1e-30))
        packed = torch.cat([a, q, (accept_lens[:bs].float() - 1).view(-1, 1)], dim=1)
        cuda = packed.is_cuda
        host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=cuda)
        host.copy_(packed, non_blocking=cuda)
        event = None
        if cuda:
            event = torch.cuda.Event()
            event.record()
        meta = [(req.rid, len(req.output_ids), phase) for req, phase in zip(reqs, phases)]
        self.pending.append((event, host, meta))
        self.flush()

    def flush(self, wait=False):
        while self.pending:
            event, host, meta = self.pending[0]
            if event is not None and not event.query():
                if not wait:
                    return
                event.synchronize()
            self.pending.popleft()
            gamma = DRAFT_WIDTH - 1
            for row, (rid, pos, phase) in zip(host.tolist(), meta):
                self._file.write(json.dumps({
                    "rid": rid, "pos": pos, "phase": phase, "cl": int(row[2 * gamma]),
                    "a": [round(v, 5) for v in row[:gamma]], "q": [round(v, 5) for v in row[gamma : 2 * gamma]],
                }) + "\n")
                self.rows += 1
        self._file.flush()
