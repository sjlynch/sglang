"""Filtered draft: the request's top-k/top-p applied to the draft row (opt-in).

When a request carries top-k or top-p, the verifier filters the *target* row
(``dflash_utils.build_speculative_verify_target_probs``) but the drafter samples
from the unfiltered draft distribution q, so draft mass outside the nucleus is
always rejected. With ``SGLANG_DSPARK_FILTERED_DRAFT=1`` each Markov step writes
-inf to the draft logits outside the request's keep set *before* the Gumbel-max
draw, into the kept step logits that verify turns into q (``_draft_probs``).
Proposal and verify therefore read the same filtered q', and the chain rule
"accept x ~ q' with min(1, p'(x)/q'(x)), else draw from max(p' - q', 0)" still
outputs exactly p' (Infera THINKING-CEILINGS.md, task 7adin; +1.1% committed IDs
per round at width 4 on chat-default sampling, 0 for plain-temperature rows).

The keep set mirrors the verifier's: top-k on the logits, top-p mass measured on
the temperature-scaled *full* row. Exactness never depends on matching the
target's keep set (any q' that both sides read is exact); the match only decides
the gain. Rows the filter cannot express exactly inside ``max_top_k`` candidates
(a nucleus wider than that, or top-k above it) keep their whole row (identity),
as do argmax rows (temperature 0) and rows with no top-k/top-p. Identity rows
are bit-identical to the unfiltered path.

Standalone (torch only) so the CPU tests load it without SGLang.
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

import torch

FILTERED_DRAFT_ENV = "SGLANG_DSPARK_FILTERED_DRAFT"
FILTERED_DRAFT_MAX_TOP_K_ENV = "SGLANG_DSPARK_FILTERED_DRAFT_MAX_TOP_K"
# Covers Infera's chat default (top-k 20) with room for typical nuclei; a row whose
# keep set is wider stays unfiltered (exact, no gain on that row).
DEFAULT_MAX_TOP_K = 64


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def filtered_draft_enabled() -> bool:
    """Read through SGLang's ``envs`` when importable (the registered switch)."""
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_DSPARK_FILTERED_DRAFT.get())
    except Exception:  # standalone CPU tests
        return _env_bool(FILTERED_DRAFT_ENV)


def filtered_draft_max_top_k() -> int:
    try:
        from sglang.srt.environ import envs

        value = int(envs.SGLANG_DSPARK_FILTERED_DRAFT_MAX_TOP_K.get())
    except Exception:
        value = int(os.environ.get(FILTERED_DRAFT_MAX_TOP_K_ENV, DEFAULT_MAX_TOP_K))
    if value < 1:
        raise ValueError(f"{FILTERED_DRAFT_MAX_TOP_K_ENV} must be >= 1, got {value}")
    return value


class DraftFilter(NamedTuple):
    """Per-row request filter, staged next to the step temperatures.

    ``top_ks`` int [rows] (values >= vocab mean no top-k; SGLang's TOP_K_ALL is
    1 << 30), ``top_ps`` float32 [rows] (>= 1 means no top-p). ``max_top_k`` is
    static: a CUDA graph captures it."""

    top_ks: torch.Tensor
    top_ps: torch.Tensor
    max_top_k: int


def _default_topk() -> Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]:
    try:
        from sglang.kernels.ops.sampling.topk import radix_topk

        return radix_topk
    except Exception:
        return lambda scores, k: torch.topk(scores, k, dim=-1)


_TOPK: Optional[Callable] = None


def _topk(scores: torch.Tensor, k: int):
    global _TOPK
    if _TOPK is None:
        _TOPK = _default_topk()
    return _TOPK(scores, k)


def keep_thresholds(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_filter: DraftFilter,
) -> torch.Tensor:
    """[rows] float32: the smallest kept logit of each row, or -inf (keep all).

    A token is kept iff its logit >= the row's threshold, so ties at the k-th
    place are all kept (q' stays exact; the target's tie choice may differ).
    No host reads: safe inside a CUDA graph capture."""
    rows, vocab = logits.shape
    k_cap = max(1, min(int(draft_filter.max_top_k), vocab))
    values = _topk(logits, k_cap)[0].float()  # [rows, k_cap], descending
    temps = temperatures.float().view(rows, 1)
    sampled = temps > 0
    safe_t = torch.where(sampled, temps, torch.ones_like(temps))
    scaled = values / safe_t
    log_norm = torch.logsumexp(logits.float() / safe_t, dim=-1, keepdim=True)
    probs = torch.exp(scaled - log_norm)
    # Mass of the strictly higher-ranked candidates, measured on the full row.
    cum_before = torch.cumsum(probs, dim=-1) - probs
    top_ks = draft_filter.top_ks.view(rows, 1).to(torch.int64)
    top_ps = draft_filter.top_ps.view(rows, 1).float()
    has_p = top_ps < 1.0
    ranks = torch.arange(k_cap, device=logits.device).view(1, k_cap)
    keep = (ranks < top_ks) & (~has_p | (cum_before < top_ps))
    keep[:, :1] = True  # the top token is always kept
    n_keep = keep.sum(dim=-1, keepdim=True)  # keep is a prefix of the ranking
    # The next rank (beyond the cap) would also be kept: not expressible here.
    total = cum_before[:, -1:] + probs[:, -1:]
    open_k = top_ks > k_cap
    open_p = ~has_p | (total < top_ps)
    overflow = (n_keep == k_cap) & open_k & open_p
    no_filter = (top_ks >= vocab) & ~has_p
    identity = overflow | no_filter | ~sampled
    threshold = torch.gather(values, 1, n_keep - 1)
    return torch.where(identity, torch.full_like(threshold, float("-inf")), threshold).view(rows)


def apply_draft_filter_(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_filter: DraftFilter,
) -> torch.Tensor:
    """Writes -inf in place to every entry of ``logits`` [rows, vocab] outside
    its row's keep set; identity rows are left bit-identical."""
    threshold = keep_thresholds(logits, temperatures, draft_filter)
    drop = logits < threshold.view(-1, 1).to(logits.dtype)
    logits.masked_fill_(drop, float("-inf"))
    return logits


def reference_keep_mask(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    *,
    with_rank_count: bool = False,
):
    """Plain full-sort reference of one row's keep set (tests only); with
    ``with_rank_count`` also the number of kept ranks before ties are added."""
    vocab = logits.shape[-1]
    if temperature <= 0 or (top_k >= vocab and top_p >= 1.0):
        everything = torch.ones_like(logits, dtype=torch.bool)
        return (everything, vocab) if with_rank_count else everything
    order = torch.argsort(logits.double(), descending=True, stable=True)
    probs = torch.softmax(logits.double() / temperature, dim=-1)[order]
    cum_before = torch.cumsum(probs, 0) - probs
    keep_rank = torch.arange(vocab) < top_k
    if top_p < 1.0:
        keep_rank &= cum_before < top_p
    keep_rank[0] = True
    n = int(keep_rank.sum())
    threshold = logits[order[n - 1]]
    keep = logits >= threshold
    return (keep, n) if with_rank_count else keep
