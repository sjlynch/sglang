"""Host-side policy for the opt-in Inkling DSpark experiment."""

import hashlib
import json
import os
import re
from functools import lru_cache
from itertools import islice
from time import perf_counter_ns
from dataclasses import dataclass, field, replace
from pathlib import Path

DRAFT_MANIFEST_ENV = "SGLANG_INKLING_DRAFT_MANIFEST"
DRAFT_MANIFEST_SHA_ENV = DRAFT_MANIFEST_ENV + "_SHA256"
# Inkling chat framing (inkling_tokenizer.SPECIAL_TOKEN_IDS).
THINK_START = 200008
THINK_END = 200010
# Also recognize direct transitions to model text and tool invocation.
OTHER_CHANNELS = frozenset(
    (200000, 200001, 200002, 200003, 200004, 200006, 200022, 200024, 200049, 200057)
)
CHANNELS = dict.fromkeys(OTHER_CHANNELS | {THINK_END}, "answer")
CHANNELS.update({THINK_START: "thinking", 200049: "tool", 200057: "tool"})
WIDTHS = (4, 5, 6, 8)  # Includes the target anchor.
BATCH_MODES = ("control", "width", "head", "both")
# QISO-x (Infera FLARE-LESSONS 1.6): "shadow" decides on the device every round and
# records the counterfactual while verify stays pinned at width 8. A dispatching
# mode needs the conditional-node graph that gate C qualifies first.
WIDTH_SELECTOR_MODES = ("shadow",)
# Pilot costs from the held-out-calibrated policy simulator (ms, w4/5/6/8).
SWARM_ROUND_MS = ((10.8, 11.35, 11.9, 12.9), (13.3, 13.95, 14.6, 16.3),
                  (14.95, 15.95, 16.95, 18.5), (16.6, 17.95, 19.3, 20.7))
TRACKED_ALPHA = (0.764, 0.697, 0.684, 0.700, 0.703, 0.710, 0.699)
CONTEXT_MS_PER_TOKEN = 0.0000332
ALPHA_WINDOW = 16


def sha256_field(value):
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("Drafter manifest requires lowercase SHA-256 digests")
    return value


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class WidthTier:
    max_load: int | None
    thinking: int
    answer: int
    long_context_tokens: int | None = None
    thinking_long: int | None = None
    answer_structured: int | None = None
    suffix_short: bool = True

    def is_long(self, prompt_tokens):
        return self.long_context_tokens is not None and prompt_tokens >= self.long_context_tokens


@dataclass(frozen=True)
class DraftPolicy:
    widths: tuple[WidthTier, ...]
    stock_sha256: str
    thinking: Artifact | None = None
    anchor_suffix_width8: bool = False
    batch_mode: str | None = None
    # Diagnostic: (head, width) every decode round uses while the mode's full graph
    # table is still captured. Isolates one captured graph for identity tests.
    pin: tuple[int, int] | None = None
    acceptance_tracking: bool = False
    # QISO-x table (kind "qiso-x-table") and its mode; None keeps the selector off.
    width_selector: Artifact | None = None
    width_selector_mode: str | None = None

    @property
    def capture_widths(self):
        if self.batch_mode in ("control", "head"):
            return (4,)
        if self.acceptance_tracking:
            return WIDTHS
        widths = {
            width for tier in self.widths
            for width in (tier.thinking, tier.answer, tier.thinking_long, tier.answer_structured)
            if width is not None
        }
        if self.anchor_suffix_width8:
            widths.add(8)
        if self.batch_mode is not None:
            widths.add(4)
        return tuple(sorted(widths))

    @property
    def capture_heads(self):
        return 2 if self.batch_mode in ("head", "both") else 1

    def tier(self, active, queued):
        load = active + queued
        for tier in self.widths:
            if tier.max_load is None or load <= tier.max_load:
                return tier
        raise AssertionError("Width policy has no final tier")

    @classmethod
    def read(cls, path, sha256=None):
        data = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256_field(sha256):
            raise ValueError("Inkling drafter manifest digest mismatch")
        payload = json.loads(data)
        if (
            not isinstance(payload, dict)
            or payload.keys() - {"version", "stock_sha256", "thinking", "widths", "anchor_suffix_width8", "batch_mode", "pin", "acceptance_tracking", "width_selector"}
            or type(payload.get("version")) is not int
            or payload["version"] != 1
        ):
            raise ValueError("Invalid Inkling drafter manifest")
        rows = payload.get("widths")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Drafter width policy requires at least one tier")
        widths = []
        previous = 0
        for i, row in enumerate(rows):
            required = {"max_load", "thinking", "answer"}
            optional = {"long_context_tokens", "thinking_long", "answer_structured", "suffix_short"}
            if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - optional:
                raise ValueError("Each width tier requires max_load, thinking and answer")
            limit = row["max_load"]
            if i == len(rows) - 1:
                if limit is not None:
                    raise ValueError("The last width tier must have max_load null")
            elif type(limit) is not int or limit <= previous:
                raise ValueError("Width tier limits must be positive and strictly increasing")
            else:
                previous = limit
            for phase in ("thinking", "answer", "thinking_long", "answer_structured"):
                if phase in row and (type(row[phase]) is not int or row[phase] not in WIDTHS):
                    raise ValueError(f"Draft width must be one of {WIDTHS}")
            threshold = row.get("long_context_tokens")
            if ("long_context_tokens" in row) != ("thinking_long" in row) or (
                "long_context_tokens" in row and (type(threshold) is not int or threshold <= 0)
            ):
                raise ValueError("Long thinking width requires a positive long_context_tokens threshold")
            if type(row.get("suffix_short", True)) is not bool:
                raise ValueError("suffix_short must be a boolean")
            widths.append(WidthTier(**row))
        suffix_flag = payload.get("anchor_suffix_width8", False)
        if type(suffix_flag) is not bool:
            raise ValueError("The anchor suffix flag must be boolean")
        batch_mode = payload.get("batch_mode")
        if "batch_mode" in payload and batch_mode not in BATCH_MODES:
            raise ValueError(f"Batch routing mode must be one of {BATCH_MODES}")
        if batch_mode is not None and suffix_flag:
            raise ValueError("Batch routing uses scheduler demand without suffix indexing")
        artifact = None
        if "thinking" in payload:
            row = payload["thinking"]
            if (
                not isinstance(row, dict)
                or row.keys() != {"kind", "path", "sha256"}
                or row["kind"] not in ("markov-head", "markov-lora", "native-head")
                or not isinstance(row["path"], str)
                or not Path(row["path"]).is_absolute()
            ):
                raise ValueError("Thinking variant requires a pinned local head or adapter")
            artifact = Artifact(row["kind"], Path(row["path"]), sha256_field(row["sha256"]))
        if batch_mode in ("head", "both") and artifact is None:
            raise ValueError("Batch head switching requires a resident thinking artifact")
        if artifact and artifact.kind == "native-head" and batch_mode is None:
            raise ValueError("A native head requires batch routing")
        tracking = payload.get("acceptance_tracking", False)
        if type(tracking) is not bool or (tracking and batch_mode not in ("width", "both")):
            raise ValueError("Acceptance tracking requires an overlap width arm")
        policy = cls(tuple(widths), sha256_field(payload.get("stock_sha256")), artifact,
                     suffix_flag, batch_mode, acceptance_tracking=tracking)
        if "pin" in payload:
            row = payload["pin"]
            if (
                batch_mode is None or not isinstance(row, dict) or row.keys() != {"head", "width"}
                or type(row["head"]) is not int or type(row["width"]) is not int
                or not 0 <= row["head"] < policy.capture_heads
                or row["width"] not in policy.capture_widths
            ):
                raise ValueError("A pin requires batch routing and a captured head and width")
            policy = replace(policy, pin=(row["head"], row["width"]))
        if "width_selector" in payload:
            row = payload["width_selector"]
            if (
                not isinstance(row, dict)
                or row.keys() != {"kind", "path", "sha256", "mode"}
                or row["kind"] != "qiso-x-table"
                or row["mode"] not in WIDTH_SELECTOR_MODES
                or not isinstance(row["path"], str)
                or not Path(row["path"]).is_absolute()
            ):
                raise ValueError(
                    "width_selector requires kind qiso-x-table, an absolute pinned path, "
                    f"sha256 and mode in {WIDTH_SELECTOR_MODES}"
                )
            if policy.pin is None or policy.pin[1] != 8 or tracking:
                raise ValueError(
                    "QISO-x shadow needs verify pinned at width 8 (every depth verified) "
                    "and no acceptance tracking"
                )
            policy = replace(
                policy,
                width_selector=Artifact(row["kind"], Path(row["path"]), sha256_field(row["sha256"])),
                width_selector_mode=row["mode"],
            )
        return policy

    @classmethod
    def from_environment(cls):
        filename = os.environ.get(DRAFT_MANIFEST_ENV)
        return cls.read(filename, sha256_field(os.environ.get(DRAFT_MANIFEST_SHA_ENV))) if filename else None


def validate_override(custom_params, policy):
    if custom_params and "dspark_draft_filter" in custom_params and type(
        custom_params["dspark_draft_filter"]
    ) is not bool:
        raise ValueError("dspark_draft_filter must be true or false")
    if not custom_params or "inkling_draft_variant" not in custom_params:
        return
    override = custom_params["inkling_draft_variant"]
    if policy is not None and policy.batch_mode is not None and override != "auto":
        raise ValueError("Batch routing supports only the automatic variant policy")
    if policy is None or (
        override != "auto" and override != "stock"
        and not (override == "thinking" and policy.thinking)
    ):
        raise ValueError("inkling_draft_variant requires an enabled variant (auto, stock, thinking)")


def channel_after(tokens, channel="answer"):
    for token in reversed(tokens):
        transition = CHANNELS.get(token)
        if transition is not None:
            return transition
    return channel


@dataclass
class RequestPhase:
    channel: str = "answer"
    seen: int = 0
    override: str = "auto"
    acceptance_rounds: int = 0
    ewma_accepted: float = 0
    ewma_reached: float = 0

    @classmethod
    def from_request(cls, req):
        custom_params = req.sampling_params.custom_params or {}
        return cls(
            channel=channel_after(req.origin_input_ids),
            override=custom_params.get("inkling_draft_variant", "auto"),
        )

    def update(self, req):
        output = req.output_ids
        if len(output) < self.seen:
            self.channel = channel_after(req.origin_input_ids)
            self.seen = 0
        if self.seen < len(output):
            self.channel = channel_after(output[self.seen:], self.channel)
            self.seen = len(output)
        return self.channel == "thinking"

    def variant(self, thinking_variant_loaded):
        return thinking_variant_loaded and (
            self.override == "thinking" or (self.override == "auto" and self.channel == "thinking")
        )

    def observe_acceptance(self, drafted, accepted):
        accepted = min(drafted, max(0, accepted))
        reached = min(drafted, accepted + 1)
        if reached:
            decay = (ALPHA_WINDOW - 1) / (ALPHA_WINDOW + 1)
            self.ewma_accepted = self.ewma_accepted * decay + accepted
            self.ewma_reached = self.ewma_reached * decay + reached
            self.acceptance_rounds += 1

    def alpha(self):
        return self.ewma_accepted / self.ewma_reached if self.acceptance_rounds >= ALPHA_WINDOW else 0.70


@dataclass
class AnchorSuffixMatch:
    """Exact ≥4 predicate. A longer match necessarily includes its matching last four IDs."""

    prior: set[tuple[int, ...]] = field(default_factory=set)
    tail: tuple[int, ...] = ()
    seen: int = 0
    initialized: bool = False

    def add(self, tokens):
        tail = self.tail
        for token in tokens:
            if len(tail) == 4:
                # Index the previous suffix only after observing its continuation.
                self.prior.add(tail)
            tail = (*tail[-3:], token)
        self.tail = tail

    def update(self, req):
        if len(req.output_ids) < self.seen:
            self.seen = 0
            self.initialized = False
        if not self.initialized:
            prompt = req.origin_input_ids
            starts = len(prompt) - 4
            self.prior = set(zip(*(islice(prompt, i, i + starts) for i in range(4)))) if starts > 0 else set()
            self.tail = tuple(prompt[-4:])
            self.initialized = True
        if self.seen < len(req.output_ids):
            self.add(req.output_ids[self.seen:])
            self.seen = len(req.output_ids)
        return self.tail in self.prior


def completed_batch_channels(batch, result):
    """Peek at the existing host readback without waiting for the forward stream."""
    lens = getattr(result, "accept_lens", None)
    tokens = getattr(result, "next_token_ids", None)
    if (
        not batch.forward_mode.is_decode()
        or lens is None or not lens.is_cpu
        or tokens is None or not tokens.is_cpu
        or result.copy_done is None or not result.copy_done.query()
    ):
        return None
    stride = result.speculative_output_stride
    if stride is None:
        stride = result.speculative_num_draft_tokens
    if stride is None or stride < 1:
        return None
    tokens = tokens.tolist()
    channels = {}
    for row, (req, count) in enumerate(zip(batch.reqs, lens.tolist())):
        if getattr(req, "grammar", None) is None:
            start = row * stride
            channel = channel_after(tokens[start:start + count], None)
            if channel is not None:
                channels[id(req)] = channel
    return channels or None


def select_round(policy, requests, queued, suffix_stats=None, completed_channels=None):
    phases, variants = [], []
    widths = []
    tier = policy.tier(len(requests), queued)
    for req in requests:
        state = getattr(req, "_inkling_draft_phase", None)
        if state is None:
            state = req._inkling_draft_phase = RequestPhase.from_request(req)
        state.update(req)
        channel = state.channel
        if completed_channels:
            channel = completed_channels.get(id(req), channel)
        thinking = channel == "thinking"
        phases.append(thinking)
        variants.append(state.variant(policy.thinking is not None))
        observation = None
        if thinking:
            long_prompt = tier.is_long(len(req.origin_input_ids))
            width = tier.thinking_long if long_prompt else tier.thinking
            if policy.anchor_suffix_width8 and width < 8 and (long_prompt or tier.suffix_short):
                start = perf_counter_ns()
                matcher = getattr(req, "_inkling_anchor_suffix", None)
                if matcher is None:
                    matcher = req._inkling_anchor_suffix = AnchorSuffixMatch()
                hit = matcher.update(req)
                observation = (hit, perf_counter_ns() - start)
                if hit:
                    width = 8
        else:
            structured = channel == "tool" or getattr(req, "grammar", None) is not None
            width = tier.answer_structured if structured and tier.answer_structured is not None else tier.answer
        widths.append(width)
        if suffix_stats is not None:
            suffix_stats.append(observation)
    # One native rectangular graph must fit every request in the round.
    return min(widths), tuple(phases), tuple(variants)


def _expectations(alpha):
    # Retain the measured depth profile after shifting its mean to the observed alpha.
    if alpha <= 0 or alpha >= 1:
        return tuple(1.0 + alpha * (width - 1) for width in WIDTHS)
    odds = tuple(p / (1 - p) for p in TRACKED_ALPHA)
    lo, hi = 0.0, 1000.0
    for _ in range(23):
        scale = (lo + hi) / 2
        probs = tuple(o * scale / (1 + o * scale) for o in odds)
        if sum(probs) / len(probs) > alpha:
            hi = scale
        else:
            lo = scale
    total = survival = 1.0
    scores = [total]
    for p in probs:
        survival *= p
        total += survival
        scores.append(total)
    return tuple(scores[width - 1] for width in WIDTHS)


@lru_cache(maxsize=1)
def tracking_expectations():
    return tuple(_expectations(alpha / 1000) for alpha in range(1001))


def _round_costs():
    scale1 = (10.649 - 1100 * CONTEXT_MS_PER_TOKEN) / SWARM_ROUND_MS[0][0]
    scale4 = (16.999 - 4400 * CONTEXT_MS_PER_TOKEN) / SWARM_ROUND_MS[3][0]
    return tuple(
        tuple((scale1 + row * (scale4 - scale1) / 3)
              * (times[0] + 1.109 * (time - times[0])) for time in times)
        for row, times in enumerate(SWARM_ROUND_MS)
    )


ROUND_COSTS = _round_costs()


def select_tracking_width(requests, phases):
    expected = [0.0] * len(WIDTHS)
    context_tokens = 0
    lookup = tracking_expectations()
    for req, thinking in zip(requests, phases):
        alpha = req._inkling_draft_phase.alpha() if thinking else 0.958
        for index, value in enumerate(lookup[round(alpha * 1000)]):
            expected[index] += value
        context_tokens += len(req.origin_input_ids) + len(req.output_ids)
    context_cost = context_tokens * CONTEXT_MS_PER_TOKEN
    costs = ROUND_COSTS[min(len(requests), len(ROUND_COSTS)) - 1]
    best = max(range(len(WIDTHS)), key=lambda index: expected[index] / (costs[index] + context_cost))
    return WIDTHS[best]


def select_batch_round(policy, requests, queued, completed_channels=None):
    """Choose the next batch from committed host state and scheduler counts."""
    requested_width, phases, _ = select_round(
        policy, requests, queued, completed_channels=completed_channels
    )
    if policy.pin is not None:
        # Phases still advance and are reported; only the graph choice is fixed.
        return policy.pin[0], policy.pin[1], phases
    width = requested_width if policy.batch_mode in ("width", "both") else 4
    if policy.acceptance_tracking:
        width = select_tracking_width(requests, phases)
    head = int(policy.capture_heads == 2 and 2 * sum(phases) > len(phases))
    return head, width, phases


@dataclass
class DepthCounts:
    drafted: list[int] = field(default_factory=list)
    reached: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    suffix_checks: int = 0
    suffix_matches: int = 0
    suffix_match_width8: int = 0
    suffix_ns: int = 0
    # QISO-x shadow: per verified-at-8 round, the width the device chose and the
    # drafts each policy would have accepted on the same draws (chain verify is
    # prefix-consistent: a width-W verify accepts min(L8, W - 1)).
    # SGLANG_DSPARK_FILTERED_DRAFT: rounds drafted with the request's filter off / on.
    draft_filter_rounds: list[int] = field(default_factory=lambda: [0, 0])
    qiso_rounds: int = 0
    qiso_widths: list[int] = field(default_factory=lambda: [0] * len(WIDTHS))
    qiso_accepted: int = 0
    fixed_accepted: list[int] = field(default_factory=lambda: [0] * len(WIDTHS))

    def record(self, drafted, accepted):
        accepted = min(drafted, max(0, accepted))
        missing = drafted - len(self.drafted)
        if missing > 0:
            self.drafted.extend([0] * missing)
            self.reached.extend([0] * missing)
            self.accepted.extend([0] * missing)
        for i in range(drafted):
            self.drafted[i] += 1
            self.reached[i] += i <= accepted
            self.accepted[i] += i < accepted

    def summary(self):
        summary = {
            "drafted": self.drafted.copy(),
            "reached": self.reached.copy(),
            "accepted": self.accepted.copy(),
            "alpha": [a / r if r else None for a, r in zip(self.accepted, self.reached)],
        }
        if any(self.draft_filter_rounds):
            summary["draft_filter"] = {
                "off": self.draft_filter_rounds[0], "on": self.draft_filter_rounds[1]
            }
        if self.qiso_rounds:
            summary["qiso_x_shadow"] = {
                "rounds": self.qiso_rounds,
                "widths": dict(zip(map(str, WIDTHS), self.qiso_widths)),
                "accepted": self.qiso_accepted,
                "fixed_accepted": dict(zip(map(str, WIDTHS), self.fixed_accepted)),
            }
        if self.suffix_checks:
            summary["anchor_suffix"] = {
                "checks": self.suffix_checks,
                "matches": self.suffix_matches,
                "matches_at_width8": self.suffix_match_width8,
                "cpu_ns": self.suffix_ns,
            }
        return summary


def record_round(req, thinking, tuned, drafted, accepted, suffix_observation=None, *, track_acceptance=False,
                 qiso_width=None, draft_filter=None):
    if track_acceptance and thinking:
        req._inkling_draft_phase.observe_acceptance(drafted, accepted)
    counts = getattr(req, "_inkling_draft_depth", None)
    if counts is None:
        counts = req._inkling_draft_depth = {}
    key = ("thinking" if thinking else "answer", "thinking" if tuned else "stock")
    if key not in counts:
        counts[key] = DepthCounts()
    depth = counts[key]
    depth.record(drafted, accepted)
    if draft_filter is not None:
        depth.draft_filter_rounds[bool(draft_filter)] += 1
    if qiso_width is not None and drafted == WIDTHS[-1] - 1:
        accepted = min(drafted, max(0, accepted))
        depth.qiso_rounds += 1
        depth.qiso_widths[WIDTHS.index(qiso_width)] += 1
        depth.qiso_accepted += min(accepted, qiso_width - 1)
        for i, width in enumerate(WIDTHS):
            depth.fixed_accepted[i] += min(accepted, width - 1)
    if suffix_observation is not None:
        match, elapsed_ns = suffix_observation
        depth.suffix_checks += 1
        depth.suffix_matches += match
        depth.suffix_match_width8 += match and drafted == 7
        depth.suffix_ns += elapsed_ns


def telemetry(req):
    counts = getattr(req, "_inkling_draft_depth", None)
    if not counts:
        return
    phases = {}
    for (phase, variant), depth in counts.items():
        phases.setdefault(phase, {})[variant] = depth.summary()
    return phases
