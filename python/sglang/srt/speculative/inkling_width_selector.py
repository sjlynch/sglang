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

Phase tables (Infera ADAPTIVE-STACK, task jf1ay): each row is priced with the
table of its phase: thinking, answer, or tool (agentic tool-call output and
grammar rows). Version 1 tables carry a thinking table and an answer prior; they
map to phase tables with constant answer rows and the tool phase reuses the answer
table, so a version 1 table decides exactly as before. Version 2 tables carry one
(iso, prior) pair per phase and may carry their own round-cost tensor
``round_ms[rows - 1][width]`` and context cost (the rule's load and context
signals), measured on the serving PC.

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
PHASES = ("thinking", "answer", "tool")
THINKING, ANSWER, TOOL = range(len(PHASES))


def phase_ids(phases) -> list[int]:
    """Row phases as PHASES indices; a bool means thinking (True) or answer."""
    out = []
    for phase in phases:
        if isinstance(phase, bool):
            out.append(THINKING if phase else ANSWER)
        elif isinstance(phase, str):
            out.append(PHASES.index(phase))
        elif type(phase) is int and 0 <= phase < len(PHASES):
            out.append(phase)
        else:
            raise ValueError(f"Unknown QISO-x row phase {phase!r}")
    return out


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


def _iso_rows(rows, name):
    if not isinstance(rows, list) or len(rows) != DEPTHS:
        raise ValueError(f"QISO-x table {name} must have {DEPTHS} depths")
    out = tuple(_probabilities(row, BINS, name) for row in rows)
    for row in out:
        if any(b < a for a, b in zip(row, row[1:])):
            raise ValueError(f"QISO-x table {name} rows must be non-decreasing in q")
    return out


@dataclass(frozen=True)
class PhaseTable:
    """iso[k][bin]: conditional acceptance of depth k + 1 given the drafted
    token's q in bin floor(20 q); prior[k]: that of an undrafted depth."""

    iso: tuple[tuple[float, ...], ...]
    prior: tuple[float, ...]

    @classmethod
    def constant(cls, values):
        values = tuple(float(v) for v in values)
        return cls(tuple((value,) * BINS for value in values), values)

    @classmethod
    def from_payload(cls, row, name):
        if (
            not isinstance(row, dict)
            or row.keys() - {"iso", "prior", "source"}
            or not {"iso", "prior"} <= row.keys()
        ):
            raise ValueError(f"QISO-x {name} table requires iso and prior")
        return cls(_iso_rows(row["iso"], f"{name}.iso"), _probabilities(row["prior"], DEPTHS, f"{name}.prior"))


@dataclass(frozen=True)
class WidthTable:
    """Per head: ``iso`` / ``prior`` are the thinking table and ``answer`` the
    answer prior (the version 1 fields, kept for their readers). ``phases`` holds
    one PhaseTable per PHASES entry. ``round_ms`` (rows x WIDTHS, ms) and
    ``context_ms_per_token`` replace the worker's built-in round costs when set."""

    iso: tuple[tuple[float, ...], ...]
    prior: tuple[float, ...]
    answer: tuple[float, ...]
    head_sha256: str
    phases: tuple[PhaseTable, ...] = ()
    round_ms: Optional[tuple[tuple[float, ...], ...]] = None
    context_ms_per_token: Optional[float] = None
    # "table": a fitted tool table; "answer": tool rows are priced as answers
    # (every version 1 table, and a version 2 table without one).
    tool_source: str = "answer"
    version: int = 1

    def __post_init__(self):
        if not self.phases:
            thinking = PhaseTable(self.iso, self.prior)
            answer = PhaseTable.constant(self.answer)
            object.__setattr__(self, "phases", (thinking, answer, answer))

    @classmethod
    def from_payload(cls, payload: Mapping) -> "WidthTable":
        if (
            not isinstance(payload, dict)
            or payload.get("version") not in (1, 2)
            or type(payload.get("version")) is not int
            or payload.get("kind") != TABLE_KIND
            or payload.get("bins") != BINS
        ):
            raise ValueError("Invalid QISO-x table (version 1 or 2, kind qiso-x-table, 20 bins)")
        head = _sha256(payload.get("head_sha256"))
        if payload["version"] == 1:
            return cls(
                iso=_iso_rows(payload.get("iso"), "iso"),
                prior=_probabilities(payload.get("prior"), DEPTHS, "prior"),
                answer=_probabilities(payload.get("answer"), DEPTHS, "answer"),
                head_sha256=head,
            )
        allowed = {
            "version", "kind", "bins", "head_sha256", "phases", "round_ms",
            "context_ms_per_token", "sampling", "fit",
        }
        if payload.keys() - allowed:
            raise ValueError(f"QISO-x table version 2 keys are {sorted(allowed)}")
        rows = payload.get("phases")
        if not isinstance(rows, dict) or rows.keys() - set(PHASES) or not {"thinking", "answer"} <= rows.keys():
            raise ValueError("QISO-x table version 2 needs thinking and answer phase tables (tool optional)")
        thinking = PhaseTable.from_payload(rows["thinking"], "thinking")
        answer = PhaseTable.from_payload(rows["answer"], "answer")
        tool = PhaseTable.from_payload(rows["tool"], "tool") if "tool" in rows else answer
        round_ms = payload.get("round_ms")
        if round_ms is not None:
            if not isinstance(round_ms, list) or not round_ms:
                raise ValueError("QISO-x round_ms must list rows 1..R")
            checked = []
            for row in round_ms:
                if (
                    not isinstance(row, list) or len(row) != len(WIDTHS)
                    or any(type(v) not in (int, float) or not 0 < v < 1e4 for v in row)
                    or any(b <= a for a, b in zip(row, row[1:]))
                ):
                    raise ValueError("QISO-x round_ms rows are positive ms, strictly increasing over widths")
                checked.append(tuple(float(v) for v in row))
            round_ms = tuple(checked)
        context = payload.get("context_ms_per_token")
        if context is not None and (type(context) not in (int, float) or not 0 <= context < 1):
            raise ValueError("QISO-x context_ms_per_token must lie in [0, 1)")
        return cls(
            iso=thinking.iso,
            prior=thinking.prior,
            answer=answer.prior,
            head_sha256=head,
            phases=(thinking, answer, tool),
            round_ms=round_ms,
            context_ms_per_token=None if context is None else float(context),
            tool_source="table" if "tool" in rows else "answer",
            version=2,
        )

    @classmethod
    def read(cls, path, sha256: str) -> "WidthTable":
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != _sha256(sha256):
            raise ValueError("QISO-x table digest mismatch")
        return cls.from_payload(json.loads(data))

    def round_costs(self, rows: int, fallback) -> tuple[float, ...]:
        """Predicted round ms per width at ``rows`` active rows, before context."""
        table = self.round_ms if self.round_ms is not None else fallback
        return tuple(table[min(max(rows, 1), len(table)) - 1])


def round_ms_for(table: WidthTable, rows: int, context_tokens: int, fallback, fallback_context_ms: float):
    """The round-cost vector (WIDTHS order) the host writes before a round: the
    load row of the table's cost tensor (or ``fallback``) plus the batch's
    context term. Shadow and active modes both build it here."""
    per_token = table.context_ms_per_token if table.context_ms_per_token is not None else fallback_context_ms
    context = per_token * context_tokens
    return [cost + context for cost in table.round_costs(rows, fallback)]


def q_bin(q: float) -> int:
    return min(int(q * BINS), BINS - 1)


def expected_ids(
    table: WidthTable,
    rows_q: Sequence[Sequence[float]],
    thinking: Sequence,
    width: int,
    drafted_width: int,
) -> float:
    """Expected committed IDs of the batch at ``width``, knowing the q of depths
    1..drafted_width-1 (the same summation order as the device rule).
    ``thinking`` holds the row phases (bool thinking / answer, or PHASES ids)."""
    total = 0.0
    for q, phase in zip(rows_q, phase_ids(thinking)):
        rows = table.phases[phase]
        survive, expected = 1.0, 1.0
        for k in range(width - 1):
            if k < drafted_width - 1:
                c = rows.iso[k][q_bin(q[k])]
            else:
                c = rows.prior[k]
            survive *= c
            expected += survive
        total += expected
    return total


def exact_stop_width_reference(
    table: WidthTable,
    rows_q: Sequence[Sequence[float]],
    thinking: Sequence,
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
        # [phase, depth, bin] and [phase, depth] in PHASES order.
        self.iso = torch.tensor([p.iso for p in table.phases], dtype=torch.float64, device=device).contiguous()
        self.prior = torch.tensor([p.prior for p in table.phases], dtype=torch.float64, device=device).contiguous()

    @staticmethod
    def phase_tensor(thinking):
        """bool rows (True thinking, False answer) or integer PHASES ids, as int64."""
        import torch

        if thinking.dtype == torch.bool:
            return torch.where(
                thinking,
                torch.full_like(thinking, THINKING, dtype=torch.int64),
                torch.full_like(thinking, ANSWER, dtype=torch.int64),
            )
        return thinking.long()

    def _expected(self, q_prefix, phase, active, width: int):
        import torch

        rows = q_prefix.shape[0]
        drafted = q_prefix.shape[1]  # == drafted width - 1 by construction
        if drafted:
            # A decision the rule never reaches is still evaluated (and masked);
            # keep its binning in range whatever the later depths hold.
            clean = torch.nan_to_num(q_prefix.double(), nan=0.0, posinf=1.0, neginf=0.0)
            bins = torch.clamp((clean * BINS).floor(), 0, BINS - 1).long()
            depth = torch.arange(drafted, device=q_prefix.device).view(1, drafted)
            iso = self.iso[phase.view(rows, 1).expand(rows, drafted), depth.expand(rows, drafted), bins]
        survive = torch.ones(rows, dtype=torch.float64, device=q_prefix.device)
        expected = torch.ones_like(survive)
        for k in range(width - 1):
            c = iso[:, k] if k < drafted else self.prior[phase, k]
            survive = survive * c
            expected = expected + survive
        expected = torch.where(active, expected, torch.zeros_like(expected))
        total = expected[0] * 0.0
        for row in range(rows):  # the host's left-to-right order
            total = total + expected[row]
        return total

    def __call__(self, q, thinking, round_ms, active=None):
        """q [rows, >= 5] drafted-token probabilities (columns beyond the drafted
        prefix of each decision are never read); thinking [rows] bool (thinking /
        answer) or int64 PHASES ids; active [rows] bool; round_ms [4] float64
        predicted round time per width in WIDTHS order.
        Returns a 0-dim int64 tensor: the verify width."""
        import torch

        rows = q.shape[0]
        phase = self.phase_tensor(thinking)
        if active is None:
            active = torch.ones(rows, dtype=torch.bool, device=q.device)
        times = {w: round_ms[i].double() for i, w in enumerate(WIDTHS)}
        extend = torch.ones((), dtype=torch.bool, device=q.device)
        width = torch.full((), 4, dtype=torch.int64, device=q.device)
        for cur, nxt in STEPS:
            prefix = q[:, : cur - 1]  # the stopping-time slice
            cur_e = self._expected(prefix, phase, active, cur)
            nxt_e = self._expected(prefix, phase, active, nxt)
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


def fit_phase(rounds: Sequence[Sequence[float]]) -> dict:
    """(iso, prior) of one phase from width-8 round-log rows laid out as the Flare
    replay's ``[pos, phase, cl, a_1..a_7, q_1..q_7]`` (a_k = min(1, p/q)): per
    depth a survival-weighted mean of a_k per q bin, gaps filled forward, then
    pool-adjacent-violators (sim/flare_jensen.py ``fit_iso``)."""
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
    return {"iso": iso, "prior": prior}


def fit_table(rounds: Sequence[Sequence[float]], head_sha256: str, answer: Sequence[float]):
    """Version 1 table: thinking rounds fitted, answers on a fixed prior."""
    fitted = fit_phase(rounds)
    return {
        "version": 1,
        "kind": TABLE_KIND,
        "bins": BINS,
        "head_sha256": head_sha256,
        "iso": fitted["iso"],
        "prior": fitted["prior"],
        "answer": list(answer),
    }


def fit_table_v2(
    rounds_by_phase: Mapping[str, Sequence[Sequence[float]]],
    head_sha256: str,
    *,
    answer_prior: Optional[Sequence[float]] = None,
    min_rounds: int = 200,
    round_ms=None,
    context_ms_per_token=None,
    sampling: Optional[str] = None,
):
    """Version 2 table: one fitted (iso, prior) per phase that has at least
    ``min_rounds`` width-8 rounds. An answer phase short of rounds falls back to
    ``answer_prior`` (constant rows). A tool phase short of rounds is left out,
    so tool rows are priced as answers (``tool_source`` "answer") and the fit
    record says so."""
    phases, sources = {}, {}
    for name in PHASES:
        rows = rounds_by_phase.get(name) or []
        if len(rows) >= min_rounds:
            phases[name] = fit_phase(rows)
            sources[name] = {"rounds": len(rows), "source": "fit"}
        elif name == "thinking":
            raise ValueError(f"QISO-x needs >= {min_rounds} thinking rounds, got {len(rows)}")
        elif name == "answer":
            if answer_prior is None:
                raise ValueError("QISO-x answer phase has too few rounds and no answer prior")
            const = PhaseTable.constant(answer_prior)
            phases[name] = {"iso": [list(r) for r in const.iso], "prior": list(const.prior)}
            sources[name] = {"rounds": len(rows), "source": "prior"}
        else:
            sources[name] = {"rounds": len(rows), "source": "answer-fallback"}
    payload = {
        "version": 2,
        "kind": TABLE_KIND,
        "bins": BINS,
        "head_sha256": head_sha256,
        "phases": phases,
        "fit": {"phases": sources, "min_rounds": min_rounds},
    }
    if round_ms is not None:
        payload["round_ms"] = [list(row) for row in round_ms]
    if context_ms_per_token is not None:
        payload["context_ms_per_token"] = context_ms_per_token
    if sampling is not None:
        payload["sampling"] = sampling
    return payload


def load_table_from_artifact(artifact) -> Optional[WidthTable]:
    if artifact is None:
        return None
    return WidthTable.read(artifact.path, artifact.sha256)
