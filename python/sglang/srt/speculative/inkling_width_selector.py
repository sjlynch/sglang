"""QISO-x: per-step verify width from fresh draft confidence (opt-in).

Infera FLARE-LESSONS.md section 1.6 (task pa9it). After drafting depths 1..W-1
the device decides whether to draft further and verify at the next captured width
(4 -> 5 -> 6 -> 8). The decision maps the drafted tokens' draft probabilities
q_k through a static per-head isotonic q -> conditional-acceptance table and
uses the table's per-depth prior for undrafted depths; answer rows use the answer
prior. Extend from W to W' iff

    (E(W') - E(W)) / (T(W') - T(W)) >= E(W) / T(W)

with E summed over the batch's rows and T the host-written predicted round time
of each width (rows and context are known on the host before the round).

Exactness: the decision to extend from W reads q of depths 1..W-1 only, all of
which are verified whatever is decided. No drafted token's inclusion depends on
its own probability, so the verify length is a stopping time of the proposal
sequence and chain speculative sampling stays exact. ``ExactStopWidth`` slices
q to the drafted prefix *before* each decision, so a later depth cannot leak in.

Standalone (stdlib + torch) so the CPU tests load it without SGLang.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

WIDTHS = (4, 5, 6, 8)  # Captured verify widths, including the anchor.
STEPS = ((4, 5), (5, 6), (6, 8))  # The stopping rule tries these in order.
DEPTHS = 7  # Depths 1..7 drafted at width 8.
BINS = 20
TABLE_KIND = "qiso-x-table"


def _sha256(value) -> str:
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("QISO-x table requires lowercase SHA-256 digests")
    return value


def _probabilities(values, length, name):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"QISO-x table {name} must have {length} values")
    out = []
    for value in values:
        if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"QISO-x table {name} values must lie in [0, 1]")
        out.append(float(value))
    return tuple(out)


@dataclass(frozen=True)
class WidthTable:
    """Per head: iso[k][bin] is the conditional acceptance of depth k + 1 given
    the drafted token's q in bin floor(20 q); prior[k] the conditional acceptance
    of an undrafted depth; answer[k] the answer-phase conditional."""

    iso: tuple[tuple[float, ...], ...]
    prior: tuple[float, ...]
    answer: tuple[float, ...]
    head_sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping) -> "WidthTable":
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("kind") != TABLE_KIND
            or payload.get("bins") != BINS
        ):
            raise ValueError("Invalid QISO-x table (version 1, kind qiso-x-table, 20 bins)")
        iso = payload.get("iso")
        if not isinstance(iso, list) or len(iso) != DEPTHS:
            raise ValueError(f"QISO-x table iso must have {DEPTHS} depths")
        rows = tuple(_probabilities(row, BINS, "iso") for row in iso)
        for row in rows:
            if any(b < a for a, b in zip(row, row[1:])):
                raise ValueError("QISO-x table iso rows must be non-decreasing in q")
        return cls(
            iso=rows,
            prior=_probabilities(payload.get("prior"), DEPTHS, "prior"),
            answer=_probabilities(payload.get("answer"), DEPTHS, "answer"),
            head_sha256=_sha256(payload.get("head_sha256")),
        )

    @classmethod
    def read(cls, path, sha256: str) -> "WidthTable":
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != _sha256(sha256):
            raise ValueError("QISO-x table digest mismatch")
        return cls.from_payload(json.loads(data))


def q_bin(q: float) -> int:
    return min(int(q * BINS), BINS - 1)


def expected_ids(
    table: WidthTable,
    rows_q: Sequence[Sequence[float]],
    thinking: Sequence[bool],
    width: int,
    drafted_width: int,
) -> float:
    """Expected committed IDs of the batch at ``width``, knowing the q of depths
    1..drafted_width-1 (the same summation order as the device rule)."""
    total = 0.0
    for q, think in zip(rows_q, thinking):
        survive, expected = 1.0, 1.0
        for k in range(width - 1):
            if not think:
                c = table.answer[k]
            elif k < drafted_width - 1:
                c = table.iso[k][q_bin(q[k])]
            else:
                c = table.prior[k]
            survive *= c
            expected += survive
        total += expected
    return total


def exact_stop_width_reference(
    table: WidthTable,
    rows_q: Sequence[Sequence[float]],
    thinking: Sequence[bool],
    round_ms: Mapping[int, float],
) -> int:
    """Host reference of the device rule; the replay's ``exact_stop_width``
    (sim/flare_replay.py) with its belief round times passed as ``round_ms``."""
    width = 4
    for _, nxt in STEPS:
        # Only depths 1..width-1 are drafted when this decision is made.
        prefix = [list(q[: width - 1]) for q in rows_q]
        cur_e = expected_ids(table, prefix, thinking, width, width)
        nxt_e = expected_ids(table, prefix, thinking, nxt, width)
        t_cur, t_nxt = round_ms[width], round_ms[nxt]
        if (nxt_e - cur_e) / max(t_nxt - t_cur, 1e-9) >= cur_e / t_cur:
            width = nxt
        else:
            break
    return width


class ExactStopWidth:
    """The device rule: no host reads, fixed launch sequence, so it can run inside
    the draft graph. ``torch`` is the reference path (float64, the host's
    summation order); a fused kernel must match it bit for bit."""

    def __init__(self, table: WidthTable, device=None):
        import torch

        self.table = table
        self.iso = torch.tensor(table.iso, dtype=torch.float64, device=device)
        self.prior = torch.tensor(table.prior, dtype=torch.float64, device=device)
        self.answer = torch.tensor(table.answer, dtype=torch.float64, device=device)

    def _expected(self, q_prefix, thinking, active, width: int):
        import torch

        rows = q_prefix.shape[0]
        drafted = q_prefix.shape[1]  # == drafted width - 1 by construction
        if drafted:
            # A decision the rule never reaches is still evaluated (and masked);
            # keep its binning in range whatever the later depths hold.
            clean = torch.nan_to_num(q_prefix.double(), nan=0.0, posinf=1.0, neginf=0.0)
            bins = torch.clamp((clean * BINS).floor(), 0, BINS - 1).long()
            depth = torch.arange(drafted, device=q_prefix.device).view(1, drafted)
            iso = self.iso[depth.expand(rows, drafted), bins]  # [rows, drafted]
        survive = torch.ones(rows, dtype=torch.float64, device=q_prefix.device)
        expected = torch.ones_like(survive)
        for k in range(width - 1):
            think_c = iso[:, k] if k < drafted else self.prior[k].expand(rows)
            c = torch.where(thinking, think_c, self.answer[k].expand(rows))
            survive = survive * c
            expected = expected + survive
        expected = torch.where(active, expected, torch.zeros_like(expected))
        total = expected[0] * 0.0
        for row in range(rows):  # the host's left-to-right order
            total = total + expected[row]
        return total

    def __call__(self, q, thinking, round_ms, active=None):
        """q [rows, >= 5] drafted-token probabilities (columns beyond the drafted
        prefix of each decision are never read); thinking, active [rows] bool;
        round_ms [4] float64 predicted round time per width in WIDTHS order.
        Returns a 0-dim int64 tensor: the verify width."""
        import torch

        rows = q.shape[0]
        if active is None:
            active = torch.ones(rows, dtype=torch.bool, device=q.device)
        times = {w: round_ms[i].double() for i, w in enumerate(WIDTHS)}
        extend = torch.ones((), dtype=torch.bool, device=q.device)
        width = torch.full((), 4, dtype=torch.int64, device=q.device)
        for cur, nxt in STEPS:
            prefix = q[:, : cur - 1]  # the stopping-time slice
            cur_e = self._expected(prefix, thinking, active, cur)
            nxt_e = self._expected(prefix, thinking, active, nxt)
            gain = (nxt_e - cur_e) / torch.clamp(times[nxt] - times[cur], min=1e-9)
            extend = extend & (gain >= cur_e / times[cur])
            width = width + extend.long() * (nxt - cur)
        return width


def drafted_confidence(step_logits, draft_tokens, temperatures):
    """q_k = softmax(step_logits / T)[x_k]: the probability the draft gave each
    drafted token ([bs, gamma] float32). The step logits are the kept (and, with
    SGLANG_DSPARK_FILTERED_DRAFT, filtered) draft rows verify reads."""
    import torch

    scaled = step_logits.float() / temperatures.float().clamp(min=1e-5).view(-1, 1, 1)
    picked = scaled.gather(-1, draft_tokens.long().unsqueeze(-1)).squeeze(-1)
    return torch.exp(picked - torch.logsumexp(scaled, dim=-1))


def counterfactual_accepted(accepted_at_width8: int, width: int) -> int:
    """Chain verification is prefix-consistent: with the same draws, a verify at
    ``width`` accepts min(L8, width - 1) drafts where width 8 accepted L8."""
    return min(int(accepted_at_width8), width - 1)


def fit_table(rounds: Sequence[Sequence[float]], head_sha256: str, answer: Sequence[float]):
    """Fit a table from width-8 round-log rows laid out as the Flare replay's
    ``[pos, phase, ?, a_1..a_7, q_1..q_7]`` (a_k = min(1, p/q), thinking rows):
    per depth a survival-weighted mean of a_k per q bin, gaps filled forward,
    then pool-adjacent-violators (sim/flare_jensen.py ``fit_iso``)."""
    iso = []
    for k in range(DEPTHS):
        sums, weights = [0.0] * BINS, [0.0] * BINS
        for r in rounds:
            b = q_bin(r[10 + k])
            w = math.prod(r[3 : 3 + k])
            sums[b] += w * r[3 + k]
            weights[b] += w
        values, last = [], 0.5
        for i in range(BINS):
            value = sums[i] / weights[i] if weights[i] > 0 else last
            values.append(value)
            last = value
        blocks = [[values[i], max(weights[i], 1e-9), 1] for i in range(BINS)]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                total = blocks[i][1] + blocks[i + 1][1]
                merged = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / total
                blocks[i] = [merged, total, blocks[i][2] + blocks[i + 1][2]]
                del blocks[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        row = []
        for value, _, count in blocks:
            row.extend([value] * count)
        iso.append(row)
    prior = []
    for k in range(DEPTHS):
        num = sum(math.prod(r[3 : 3 + k + 1]) for r in rounds)
        den = sum(math.prod(r[3 : 3 + k]) for r in rounds)
        prior.append(num / den if den else 0.0)
    return {
        "version": 1,
        "kind": TABLE_KIND,
        "bins": BINS,
        "head_sha256": head_sha256,
        "iso": iso,
        "prior": prior,
        "answer": list(answer),
    }


def load_table_from_artifact(artifact) -> Optional[WidthTable]:
    if artifact is None:
        return None
    return WidthTable.read(artifact.path, artifact.sha256)
