"""Host-side policy for the opt-in Inkling DSpark experiment."""

import hashlib
import json
import os
import re
from itertools import islice
from time import perf_counter_ns
from dataclasses import dataclass, field
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

    @property
    def capture_widths(self):
        widths = {
            width for tier in self.widths
            for width in (tier.thinking, tier.answer, tier.thinking_long, tier.answer_structured)
            if width is not None
        }
        if self.anchor_suffix_width8:
            widths.add(8)
        return tuple(sorted(widths))

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
            or payload.keys() - {"version", "stock_sha256", "thinking", "widths", "anchor_suffix_width8"}
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
        artifact = None
        if "thinking" in payload:
            row = payload["thinking"]
            if (
                not isinstance(row, dict)
                or row.keys() != {"kind", "path", "sha256"}
                or row["kind"] not in ("markov-head", "markov-lora")
                or not isinstance(row["path"], str)
                or not Path(row["path"]).is_absolute()
            ):
                raise ValueError("Thinking variant requires a pinned local Markov head or LoRA")
            artifact = Artifact(row["kind"], Path(row["path"]), sha256_field(row["sha256"]))
        return cls(tuple(widths), sha256_field(payload.get("stock_sha256")), artifact, suffix_flag)

    @classmethod
    def from_environment(cls):
        filename = os.environ.get(DRAFT_MANIFEST_ENV)
        return cls.read(filename, sha256_field(os.environ.get(DRAFT_MANIFEST_SHA_ENV))) if filename else None


def validate_override(custom_params, policy):
    if not custom_params or "inkling_draft_variant" not in custom_params:
        return
    override = custom_params["inkling_draft_variant"]
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


def select_round(policy, requests, queued, suffix_stats=None):
    phases, variants = [], []
    widths = []
    tier = policy.tier(len(requests), queued)
    for req in requests:
        state = getattr(req, "_inkling_draft_phase", None)
        if state is None:
            state = req._inkling_draft_phase = RequestPhase.from_request(req)
        thinking = state.update(req)
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
            structured = state.channel == "tool" or getattr(req, "grammar", None) is not None
            width = tier.answer_structured if structured and tier.answer_structured is not None else tier.answer
        widths.append(width)
        if suffix_stats is not None:
            suffix_stats.append(observation)
    # One native rectangular graph must fit every request in the round.
    return min(widths), tuple(phases), tuple(variants)


@dataclass
class DepthCounts:
    drafted: list[int] = field(default_factory=list)
    reached: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    suffix_checks: int = 0
    suffix_matches: int = 0
    suffix_match_width8: int = 0
    suffix_ns: int = 0

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
        if self.suffix_checks:
            summary["anchor_suffix"] = {
                "checks": self.suffix_checks,
                "matches": self.suffix_matches,
                "matches_at_width8": self.suffix_match_width8,
                "cpu_ns": self.suffix_ns,
            }
        return summary


def record_round(req, thinking, tuned, drafted, accepted, suffix_observation=None):
    counts = getattr(req, "_inkling_draft_depth", None)
    if counts is None:
        counts = req._inkling_draft_depth = {}
    key = ("thinking" if thinking else "answer", "thinking" if tuned else "stock")
    if key not in counts:
        counts[key] = DepthCounts()
    depth = counts[key]
    depth.record(drafted, accepted)
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
