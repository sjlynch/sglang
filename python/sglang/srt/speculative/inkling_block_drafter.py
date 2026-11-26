"""Research block-drafter heads for Inkling2 MTP: K1-K3 and C1/C2 (Infera route A).

Infera tasks xl5h9 (K1-K3 training + config), a1c96 (C1/C2 training) and 5k0kt (this
serving path). Flag: ``SGLANG_INKLING2_MTP_BLOCK_ARCH`` = the ``block_arch`` JSON the
Infera trainer writes into ``mtp-block.safetensors``. Empty (default): the shipped head
and every code path are unchanged.

How a block head proposes (the DSPARK path, unchanged by every arm):

1. **One drafter forward drafts the whole block.** Row 0 is the anchor (the last
   committed token), rows 1..w-1 are the mask embedding; every row attends to the
   draft KV of the committed context and, bidirectionally, to its own block. One
   forward over ``bs * w`` rows gives w-1 hidden rows, then one lm_head GEMM gives the
   base logits ``[bs, w-1, V]``. Nothing in this forward depends on a drafted token.
2. **A short Markov walk makes it partially parallel.** Depth j's draft distribution
   is ``q_j = softmax(base_j + fold(rnn(state, x_{j-1}, h_j, f_rnn)))``: a rank-256
   recurrence on the previous draft (``dspark_markov_walk.cuh`` / ``MarkovSampleStep``).
   The walk samples x_j ~ q_j (Gumbel-max on the kept step logits) and keeps those
   logits, so verify reads exactly the q each draft was drawn from.
3. **Verify is chain speculative sampling** (``dspark_verify``): accept x_j with
   ``min(1, p_j(x_j) / q_j(x_j))``, on the first rejection draw from
   ``max(p_j - q_j, 0)`` normalised, after w-1 acceptances draw the bonus from p_w.
   This outputs exactly p for ANY proposal whose recorded q is the distribution its
   token was drawn from given the earlier drafts: a block forward + Markov walk is such
   a proposal. ``chain_verify_reference`` below is the literal rule the CPU tests use.

Arms (all exactly their init head at initialisation; none touches steps 2-3):

* K1 ``block_conv 2``: ``DFlashGroupedConv`` around every sublayer (config only).
* K2 ``tap_fc``: a full-rank tap residual on every scalar mix (context KV and f_rnn).
* K3 ``n_layers`` > 3: extra local layers (config only).
* C1 ``ffn_extra N``: N more SwiGLU channels per layer. SwiGLU is channel-wise, so the
  loader concatenates them onto gate/up rows and down columns (``merge_ffn_extra``);
  the served MLP is one GEMM pair of intermediate 9216 + N.
* C2 ``depth_experts R``: a learned per-depth embedding on mask rows 1..w-1 at the
  block input and one low-rank SwiGLU expert per depth on the final residual before
  the final norm (``DepthExperts``).
* K4 (pairwise selector) is refused: it needs a top-k walk kernel.

After partial acceptance (k drafts accepted + one corrected/bonus token) nothing of the
block survives: the k+1 committed positions get their draft KV from the *target's*
verify hidden taps (``write_target_hidden_kv``, K2's tap term included) and the f_rnn
row from the last committed position (``store_verify_commit``). Mask rows are never
cached. So the state update is per committed position and width-independent.

QISO-x (active dispatch) drafts at the checkpoint width 8 and verifies a prefix W in
{4, 5, 6, 8} chosen by a stopping time of the drafted q; the width-W verify reads the
width-8 proposal's first W-1 drafts with their kept logits, which is exact. Width views
that draft at their own width (batch routing without QISO-x) get their own block size:
``Inkling2ForConditionalGenerationMTP.forward`` sets every block-size-indexed module
(conv, depth experts) from the view before the graph captures. The filtered draft
(``SGLANG_DSPARK_FILTERED_DRAFT``) acts on the Markov step logits, which no arm changes,
so it composes unchanged (proposal and verify read the same filtered q').

Standalone (stdlib + torch; no SGLang import) so the CPU tests and the Infera tools load
it directly. Also a CLI:

    python inkling_block_drafter.py inspect HEAD.safetensors
    python inkling_block_drafter.py prepare --head mtp-block.safetensors \
        --stock-dir STOCK_DRAFT_DIR --out PRIVATE_DRAFT_DIR [--lift-from CONTROL.safetensors]
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import torch
from torch import nn

BLOCK_ARCH_ENV = "SGLANG_INKLING2_MTP_BLOCK_ARCH"
BLOCK_ARCH_KEY = "block_arch"  # safetensors metadata of mtp-block.safetensors
QISO_TABLE_TRANSFER_ENV = "SGLANG_INKLING_QISO_TABLE_TRANSFER"

BLOCK_ARCH_KEYS = frozenset(
    {
        "n_layers",
        "local_layers",
        "window",
        "init_layers",
        "block_conv",
        "conv_group",
        "tap_fc",
        "selector_rank",
        "selector_topk",
        "ffn_extra",
        "depth_experts",
    }
)
FFN_EXTRA_CHANNELS = (2304, 4608, 9216)
DEPTH_EXPERT_RANKS = (128, 256, 512, 1024)
DEPTH_EXPERT_DEPTHS = 7  # the trainer's depths 1..7 (block 8)
STOCK_LAYERS = 3


# ---------------------------------------------------------------------------
# Architecture: parse, validate, describe
# ---------------------------------------------------------------------------


def validate_block_arch(arch: Any, draft: Mapping[str, Any]) -> dict:
    """The validated K/C-arm architecture, or ValueError. ``draft`` is the target
    config's ``mtp_config`` (stock values: num_hidden_layers, sliding_window_size,
    block_size)."""
    if not isinstance(arch, dict) or set(arch) - BLOCK_ARCH_KEYS:
        raise ValueError(f"Unknown Inkling2 MTP block architecture: {arch!r}")
    if arch.get("selector_rank") or arch.get("selector_topk"):
        raise ValueError(
            "Inkling2 MTP selector heads (K4) need a candidate walk; not servable here"
        )
    if arch.get("block_conv", 0) not in (0, 2) or (
        arch.get("block_conv") and arch.get("conv_group") != 16
    ):
        raise ValueError("Inkling2 MTP block convolution must be 2 taps x 16 channels")
    if arch.get("window", draft["sliding_window_size"]) != draft["sliding_window_size"]:
        raise ValueError("Inkling2 MTP block head window differs from the checkpoint's")
    if int(arch.get("n_layers", draft["num_hidden_layers"])) < draft["num_hidden_layers"]:
        raise ValueError("Inkling2 MTP block heads only add layers")
    ffn = int(arch.get("ffn_extra", 0) or 0)
    if ffn and ffn not in FFN_EXTRA_CHANNELS:
        raise ValueError(f"Inkling2 MTP ffn_extra must be one of {FFN_EXTRA_CHANNELS}")
    rank = int(arch.get("depth_experts", 0) or 0)
    if rank:
        if rank not in DEPTH_EXPERT_RANKS:
            raise ValueError(f"Inkling2 MTP depth_experts must be one of {DEPTH_EXPERT_RANKS}")
        if int(draft["block_size"]) - 1 > DEPTH_EXPERT_DEPTHS:
            raise ValueError("The block is wider than the trained depth experts")
        if arch.get("block_conv"):
            # The fused ROCm conv epilogue would skip the experts' final-norm hook.
            raise ValueError("Inkling2 MTP serves depth experts without the block conv")
    return dict(arch)


def parse_block_arch(raw: Optional[str], draft: Mapping[str, Any]) -> Optional[dict]:
    """``SGLANG_INKLING2_MTP_BLOCK_ARCH``'s value -> validated arch, or None when empty."""
    if not raw:
        return None
    try:
        arch = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{BLOCK_ARCH_ENV} is not JSON: {error}") from None
    return validate_block_arch(arch, draft)


def arm_name(arch: Optional[Mapping[str, Any]]) -> str:
    """A short label (stock, k1, k2, k3, c1, c2, or a '+'-joined mix)."""
    if not arch:
        return "stock"
    # K arms are cumulative as the trainer's campaign defines them: k1 = conv,
    # k2 = conv + tap fc, k3 = k2 + extra layers.
    conv, tap = bool(arch.get("block_conv")), bool(arch.get("tap_fc"))
    deep = int(arch.get("n_layers", STOCK_LAYERS)) > int(arch.get("init_layers", STOCK_LAYERS))
    parts = []
    if conv and tap and deep:
        parts.append("k3")
    elif conv and tap:
        parts.append("k2")
    else:
        parts += [name for name, on in (("k1", conv), ("tapfc", tap), ("layers", deep)) if on]
    if arch.get("ffn_extra"):
        parts.append("c1")
    if arch.get("depth_experts"):
        parts.append("c2")
    return "+".join(parts) or "stock"


def extra_tensor_shapes(
    arch: Mapping[str, Any],
    *,
    hidden: int,
    n_features: int,
    stored_layers: int = STOCK_LAYERS,
) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensors an arm adds to the stock head (names as the trainer exports
    them, which are the serving names). K3's extra layers are whole stock-layout layers
    and are checked by the strict loader, not listed here."""
    n_layers = int(arch.get("n_layers", stored_layers))
    shapes: dict[str, tuple[int, ...]] = {}
    if arch.get("block_conv"):
        taps, group = int(arch["block_conv"]), int(arch["conv_group"])
        for i in range(n_layers):
            for name in ("attention_conv", "mlp_conv"):
                shapes[f"layers.{i}.{name}.base_kernel"] = (2, taps, hidden)
                shapes[f"layers.{i}.{name}.kernel_projection.weight"] = (
                    2 * taps * (hidden // group),
                    hidden,
                )
    if arch.get("tap_fc"):
        shapes["tap_fc.weight"] = (hidden, n_features * hidden)
    ffn = int(arch.get("ffn_extra", 0) or 0)
    if ffn:
        for i in range(n_layers):
            shapes[f"layers.{i}.mlp_extra.gate_proj.weight"] = (ffn, hidden)
            shapes[f"layers.{i}.mlp_extra.up_proj.weight"] = (ffn, hidden)
            shapes[f"layers.{i}.mlp_extra.down_proj.weight"] = (hidden, ffn)
    rank = int(arch.get("depth_experts", 0) or 0)
    if rank:
        d = DEPTH_EXPERT_DEPTHS
        shapes.update(
            {
                "depth_experts.row_embed": (d, hidden),
                "depth_experts.norm": (hidden,),
                "depth_experts.gate": (d, rank, hidden),
                "depth_experts.up": (d, rank, hidden),
                "depth_experts.down": (d, hidden, rank),
            }
        )
    return shapes


def added_parameters(arch: Mapping[str, Any], *, hidden: int = 2304, n_features: int = 4,
                     intermediate: int = 9216, stored_layers: int = STOCK_LAYERS) -> int:
    """Parameters the arm adds (K3's layers counted at the stock layer size)."""
    total = sum(math.prod(s) for s in extra_tensor_shapes(
        arch, hidden=hidden, n_features=n_features, stored_layers=stored_layers).values())
    extra_layers = int(arch.get("n_layers", stored_layers)) - int(arch.get("init_layers", stored_layers))
    if extra_layers > 0:
        # q, k, v, o (24 heads x 128, 8 KV heads), MLP, two norms, context k/v + norms
        heads, kv_heads, head_dim = 24, 8, 128
        attention = hidden * head_dim * (2 * heads + 2 * kv_heads)
        context = 2 * hidden * kv_heads * head_dim
        total += extra_layers * (attention + context + 3 * hidden * intermediate + 4 * hidden)
    return int(total)


# ---------------------------------------------------------------------------
# Safetensors header (stdlib) and the head loader / draft-dir preparer
# ---------------------------------------------------------------------------


def read_safetensors_header(path) -> tuple[dict, dict]:
    """(tensor table {name: {dtype, shape, data_offsets}}, metadata) without torch."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length <= 0 or length > 64 << 20:
            raise ValueError(f"{path}: not a safetensors file")
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata


def head_block_arch(path) -> Optional[dict]:
    """The ``block_arch`` metadata of a head file, or None for a stock-format head."""
    _, metadata = read_safetensors_header(path)
    raw = metadata.get(BLOCK_ARCH_KEY)
    return None if raw is None else json.loads(raw)


def sha256_file(path, chunk=8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_head(path, *, stock_path=None, draft: Optional[Mapping[str, Any]] = None) -> dict:
    """Inspect a head: its arch, arm, the tensors the arch adds (present, shapes) and,
    with ``stock_path``, that every other tensor has the stock name and shape.
    Raises ValueError on any mismatch; returns a JSON-able receipt."""
    table, metadata = read_safetensors_header(path)
    raw = metadata.get(BLOCK_ARCH_KEY)
    arch = None if raw is None else json.loads(raw)
    hidden = None
    n_features = None
    for name, entry in table.items():
        if name == "fuse_weight":
            n_features = int(entry["shape"][1])
        if name == "mask_embed":
            hidden = int(entry["shape"][0])
    if hidden is None or n_features is None:
        raise ValueError(f"{path}: not an Inkling2 MTP head (no mask_embed / fuse_weight)")
    bad_dtype = sorted(n for n, e in table.items() if e["dtype"] != "BF16")
    if bad_dtype:
        raise ValueError(f"{path}: MTP heads are BF16; not: {bad_dtype[:4]}")
    extra = {}
    if arch is not None:
        draft = dict(draft or {})
        draft.setdefault("num_hidden_layers", STOCK_LAYERS)
        draft.setdefault("sliding_window_size", arch.get("window", 1024))
        draft.setdefault("block_size", 8)
        validate_block_arch(arch, draft)
        extra = extra_tensor_shapes(arch, hidden=hidden, n_features=n_features)
        for name, shape in extra.items():
            if name not in table:
                raise ValueError(f"{path}: arch {arm_name(arch)} needs {name}")
            if tuple(table[name]["shape"]) != shape:
                raise ValueError(f"{path}: {name} is {table[name]['shape']}, expected {list(shape)}")
    if stock_path is not None:
        stock, _ = read_safetensors_header(stock_path)
        own = {n: tuple(e["shape"]) for n, e in table.items() if n not in extra}
        stock_layers = 1 + max(int(n.split(".")[1]) for n in stock if n.startswith("layers."))
        n_layers = int((arch or {}).get("n_layers", stock_layers))
        expected = {}
        for name, entry in stock.items():
            match = re.match(r"layers\.(\d+)\.(.*)$", name)
            if match and int(match.group(1)) == stock_layers - 1:
                # K3's new layers copy the stock layout of the last stock layer
                for i in range(stock_layers - 1, n_layers):
                    expected[f"layers.{i}.{match.group(2)}"] = tuple(entry["shape"])
            else:
                expected[name] = tuple(entry["shape"])
        if own != expected:
            missing = sorted(set(expected) - set(own))[:4]
            unknown = sorted(set(own) - set(expected))[:4]
            changed = sorted(n for n in set(own) & set(expected) if own[n] != expected[n])[:4]
            raise ValueError(
                f"{path}: stock tensors differ (missing {missing}, unknown {unknown}, shape {changed})"
            )
    return {
        "path": str(path),
        "arm": arm_name(arch),
        "block_arch": arch,
        "env": {BLOCK_ARCH_ENV: "" if arch is None else json.dumps(arch, sort_keys=True)},
        "tensors": len(table),
        "added_tensors": len(extra),
        "added_parameters": None if arch is None else added_parameters(arch, hidden=hidden, n_features=n_features),
    }


def identity_tensors(arch: Mapping[str, Any], *, hidden: int, n_features: int,
                     dtype=torch.bfloat16, seed: int = 61) -> dict[str, torch.Tensor]:
    """The tensors that make an arm exactly its init head: conv base 1/0 with a zero
    projection, zero tap fc, C1 channels with zero down projection, C2 with zero row
    embedding and zero down. Used to LIFT a stock-format head (the T8 control) into an
    arm's architecture, so one engine booted with the arm's flag can swap between the
    arm and its control (head_ab ``transport: swap``). Function-preserving: gate/up
    values never reach the output while down is zero."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for name, shape in extra_tensor_shapes(arch, hidden=hidden, n_features=n_features).items():
        if name.endswith("base_kernel"):
            value = torch.zeros(shape)
            value[:, 0] = 1.0
        elif name.endswith((".gate_proj.weight", ".up_proj.weight")) or name in (
            "depth_experts.gate",
            "depth_experts.up",
        ):
            bound = 1 / math.sqrt(hidden)
            value = torch.empty(shape).uniform_(-bound, bound, generator=generator)
        elif name == "depth_experts.norm":
            value = torch.ones(shape)
        else:
            value = torch.zeros(shape)
        out[name] = value.to(dtype)
    return out


def lift_head(stock_tensors: Mapping[str, torch.Tensor], arch: Mapping[str, Any]) -> dict:
    """A stock-format head in an arm's architecture (K3's extra layers are refused:
    a lifted control would need identity layers the trainer copied, not stock ones)."""
    if int(arch.get("n_layers", STOCK_LAYERS)) != STOCK_LAYERS:
        raise ValueError("Lifting into K3 needs the trainer's identity-layer export (_roundtrip/)")
    hidden = int(stock_tensors["mask_embed"].shape[0])
    n_features = int(stock_tensors["fuse_weight"].shape[1])
    extra = identity_tensors(arch, hidden=hidden, n_features=n_features)
    clash = set(extra) & set(stock_tensors)
    if clash:
        raise ValueError(f"Head already has arm tensors: {sorted(clash)[:3]}")
    return {**dict(stock_tensors), **extra}


def prepare_draft_dir(head, stock_dir, out_dir, *, lift_from=None) -> dict:
    """A private ``--speculative-draft-model-path`` for a block head: every file of
    the stock draft dir except ``mtp.safetensors`` (copied), the head as
    ``mtp.safetensors``, ``block-arch.env`` (the flag) and ``block-arch.json`` (the
    receipt). With ``lift_from`` (a stock-format head, e.g. the T8 control), also
    writes ``control-lifted.safetensors`` in the head's architecture for swaps."""
    head, stock_dir, out_dir = Path(head), Path(stock_dir), Path(out_dir)
    stock = stock_dir / "mtp.safetensors"
    receipt = check_head(head, stock_path=stock)
    if receipt["block_arch"] is None:
        raise ValueError(f"{head}: no block_arch metadata; serve it as a stock head")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in stock_dir.iterdir():
        if entry.name != "mtp.safetensors" and entry.is_file():
            shutil.copy2(entry, out_dir / entry.name)
    shutil.copy2(head, out_dir / "mtp.safetensors")
    receipt["sha256"] = sha256_file(out_dir / "mtp.safetensors")
    receipt["stock_sha256"] = sha256_file(stock)
    value = receipt["env"][BLOCK_ARCH_ENV]
    (out_dir / "block-arch.env").write_text(f"{BLOCK_ARCH_ENV}='{value}'\n")
    if lift_from is not None:
        from safetensors.torch import load_file, save_file

        control = load_file(str(lift_from))
        if any(k.startswith(("tap_fc.", "depth_experts.")) or "_conv." in k or ".mlp_extra." in k for k in control):
            raise ValueError(f"{lift_from}: already carries arm tensors")
        lifted = lift_head(control, receipt["block_arch"])
        target = out_dir / "control-lifted.safetensors"
        save_file(lifted, str(target), metadata={BLOCK_ARCH_KEY: value})
        check_head(target, stock_path=stock)
        receipt["control_lifted"] = {"from": str(lift_from), "sha256": sha256_file(target)}
    (out_dir / "block-arch.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


# ---------------------------------------------------------------------------
# C1: extra SwiGLU channels, merged at load
# ---------------------------------------------------------------------------

_STOCK_MLP = re.compile(r"^(layers\.(\d+)\.(?:mlp\.)?)(gate|up|down)_proj\.weight$")
_EXTRA_MLP = re.compile(r"^layers\.(\d+)\.mlp_extra\.(gate|up|down)_proj\.weight$")


def merge_ffn_extra(
    weights: Iterable[tuple[str, torch.Tensor]], ffn_extra: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """C1 loader: stream every tensor through, except each layer's stock MLP and
    ``mlp_extra`` tensors, which come out as ONE gate, up and down tensor under the
    stock names: gate/up rows ``[stock; extra]``, down columns ``[stock | extra]``.
    SwiGLU is channel-wise, so the merged MLP equals stock + extra (fp32 accumulation
    in one GEMM instead of two summed in BF16). Any layer missing a piece fails."""
    if not ffn_extra:
        yield from weights
        return
    pending: dict[int, dict] = {}
    for name, tensor in weights:
        extra = _EXTRA_MLP.match(name)
        stock = None if extra else _STOCK_MLP.match(name)
        if extra:
            layer, kind = int(extra.group(1)), extra.group(2)
            pending.setdefault(layer, {})[("extra", kind)] = tensor
        elif stock:
            layer, kind = int(stock.group(2)), stock.group(3)
            slot = pending.setdefault(layer, {})
            slot[("stock", kind)] = tensor
            slot.setdefault("names", {})[kind] = name
        else:
            yield name, tensor
            continue
        slot = pending[layer]
        if all((side, kind) in slot for side in ("stock", "extra") for kind in ("gate", "up", "down")):
            for kind in ("gate", "up", "down"):
                base, add = slot[("stock", kind)], slot[("extra", kind)]
                dim = 1 if kind == "down" else 0
                if add.shape[dim] != ffn_extra or add.shape[1 - dim] != base.shape[1 - dim]:
                    raise ValueError(f"layers.{layer}.mlp_extra.{kind}_proj has shape {tuple(add.shape)}")
                yield slot["names"][kind], torch.cat((base, add.to(base.dtype)), dim=dim)
            del pending[layer]
    if pending:
        raise ValueError(
            f"C1 (ffn_extra {ffn_extra}) needs stock and extra gate/up/down for layers {sorted(pending)}"
        )


# ---------------------------------------------------------------------------
# C2: depth experts
# ---------------------------------------------------------------------------


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class DepthExperts(nn.Module):
    """C2 serving module, the trainer's ``drafter.DepthExperts`` on the flattened
    ``[bs * block, H]`` rows: row j of each block (depth j, 1..block-1) gets
    ``row_embed[j-1]`` at the input and ``down_j(silu(gate_j h) * up_j h)`` on its
    final residual, h = rms(residual, norm), rank r_j = ceil(R j / 7) (a fixed channel
    mask). Row 0 (the anchor) is untouched. ``block_size`` is set per width view
    before capture. BF16 einsums as the trainer; one rounding per op."""

    def __init__(self, rank: int, hidden: int, depths: int = DEPTH_EXPERT_DEPTHS,
                 dtype=torch.bfloat16):
        super().__init__()
        self.rank, self.depths, self.hidden = int(rank), int(depths), int(hidden)
        self.block_size = depths + 1
        self.row_embed = nn.Parameter(torch.zeros(depths, hidden, dtype=dtype), requires_grad=False)
        self.norm = nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)
        self.gate = nn.Parameter(torch.zeros(depths, rank, hidden, dtype=dtype), requires_grad=False)
        self.up = nn.Parameter(torch.zeros(depths, rank, hidden, dtype=dtype), requires_grad=False)
        self.down = nn.Parameter(torch.zeros(depths, hidden, rank, dtype=dtype), requires_grad=False)
        active = [math.ceil(rank * (j + 1) / depths) for j in range(depths)]
        mask = (torch.arange(rank)[None] < torch.tensor(active)[:, None]).to(dtype)
        self.register_buffer("mask", mask, persistent=False)

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        block = int(self.block_size)
        if block - 1 > self.depths:
            raise ValueError("The block is wider than the trained depth experts")
        if x.shape[0] % block:
            raise ValueError("Depth experts need whole blocks of rows")
        return x.view(-1, block, x.shape[-1])

    def embed(self, hidden: torch.Tensor) -> torch.Tensor:
        """[bs * block, H] block input -> rows 1..block-1 plus their depth embedding."""
        x = self._rows(hidden)
        rows = x.shape[1] - 1
        out = torch.cat((x[:, :1], x[:, 1:] + self.row_embed[:rows].to(x.dtype)), dim=1)
        return out.view_as(hidden)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """[bs * block, H] final residual -> rows 1..block-1 plus their expert's output."""
        x = self._rows(residual)
        rows = x[:, 1:]
        depth = rows.shape[1]
        h = _rms(rows, self.norm)
        a = torch.einsum("mdh,drh->mdr", h, self.gate[:depth].to(h.dtype))
        b = torch.einsum("mdh,drh->mdr", h, self.up[:depth].to(h.dtype))
        code = nn.functional.silu(a) * b * self.mask[:depth].to(h.dtype)
        out = torch.einsum("mdr,dhr->mdh", code, self.down[:depth].to(code.dtype))
        return torch.cat((x[:, :1], rows + out), dim=1).view_as(residual)

    def final_norm_pre_hook(self, module, args, kwargs):
        """Forward pre-hook for the draft model's final RMSNorm: ``norm(x, residual)``
        becomes ``norm(experts(x + residual), 0)``, i.e. rms of the expert-updated
        residual (an exact zero add), so the norm's weight keeps its checkpoint name."""
        hidden = args[0]
        residual = args[1] if len(args) > 1 else kwargs.pop("residual", None)
        merged = hidden if residual is None else hidden + residual
        updated = self(merged)
        if residual is None:
            return (updated,) + tuple(args[2:]), kwargs
        return (updated, torch.zeros_like(residual)) + tuple(args[2:]), kwargs


# ---------------------------------------------------------------------------
# Reference proposal and verification (tests, microbench)
# ---------------------------------------------------------------------------


def markov_block_reference(
    base_logits: torch.Tensor,
    step_bias,
    anchor: torch.Tensor,
    generator: torch.Generator,
    *,
    temperature: float = 1.0,
    keep=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DSPARK proposal on [bs, gamma, V] base logits (one block forward): walk
    gamma depths, x_j ~ q_j = softmax((base_j + step_bias(j, x_{j-1})) / T) with an
    optional filter ``keep(j, logits) -> bool mask`` (the filtered draft's q').
    Returns (drafts [bs, gamma], q [bs, gamma, V]: the distribution each draft was
    drawn from)."""
    bs, gamma, vocab = base_logits.shape
    prev = anchor.long()
    drafts, qs = [], []
    for j in range(gamma):
        logits = (base_logits[:, j].double() + step_bias(j, prev).double()) / temperature
        if keep is not None:
            logits = logits.masked_fill(~keep(j, logits), float("-inf"))
        q = torch.softmax(logits, dim=-1)
        prev = torch.multinomial(q, 1, generator=generator).view(bs)
        drafts.append(prev)
        qs.append(q)
    return torch.stack(drafts, 1), torch.stack(qs, 1)


def chain_verify_reference(
    p: torch.Tensor,
    q: torch.Tensor,
    drafts: torch.Tensor,
    generator: torch.Generator,
    widths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal chain speculative sampling, vectorised over rows. ``drafts`` [bs, k],
    q [bs, k, V] the distribution each draft was drawn from, p [bs, k+1, V] target rows
    (p[:, j] follows the j-th drafted prefix; p[:, k] is the bonus row). ``widths``
    [bs] (optional, 1..k+1 incl. the anchor) verifies only each row's first
    ``widths - 1`` drafts: the QISO-x prefix, whose bonus row is p[:, widths - 1].
    Row i accepts x_j with prob min(1, p_j(x_j) / q_j(x_j)); at the first rejection it
    draws from max(p_j - q_j, 0) normalised; after all its drafts it draws the bonus.
    Returns (committed [bs, k+1] padded with -1, accepted drafts [bs])."""
    bs, k = drafts.shape
    p, q = p.double(), q.double()
    limit = torch.full((bs,), k, dtype=torch.long) if widths is None else widths.long() - 1
    committed = torch.full((bs, k + 1), -1, dtype=torch.long)
    accepted = torch.zeros(bs, dtype=torch.long)
    live = torch.ones(bs, dtype=torch.bool)
    rows = torch.arange(bs)
    for j in range(k + 1):
        bonus = live & (limit == j)
        if bonus.any():
            committed[bonus, j] = torch.multinomial(p[bonus, j], 1, generator=generator).view(-1)
            live &= ~bonus
        if j == k or not live.any():
            break
        x = drafts[:, j].long()
        px, qx = p[rows, j, x], q[rows, j, x]
        u = torch.rand(bs, dtype=torch.float64, generator=generator)
        accept = live & (qx > 0) & (u * qx <= px)
        committed[accept, j] = x[accept]
        accepted += accept.long()
        reject = live & ~accept
        if reject.any():
            residual = torch.clamp(p[reject, j] - q[reject, j], min=0)
            total = residual.sum(-1, keepdim=True)
            row = torch.where(total > 0, residual / total.clamp(min=1e-300), p[reject, j])
            committed[reject, j] = torch.multinomial(row, 1, generator=generator).view(-1)
            live &= ~reject
    return committed, accepted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect = sub.add_parser("inspect", help="print a head's arch, arm and flag value")
    inspect.add_argument("head")
    inspect.add_argument("--stock", help="stock mtp.safetensors to check the other tensors against")
    prepare = sub.add_parser("prepare", help="make a private draft dir for a block head")
    prepare.add_argument("--head", required=True)
    prepare.add_argument("--stock-dir", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--lift-from", help="a stock-format control head to lift into the arm's architecture")
    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        print(json.dumps(check_head(args.head, stock_path=args.stock), indent=1, sort_keys=True))
    else:
        receipt = prepare_draft_dir(args.head, args.stock_dir, args.out, lift_from=args.lift_from)
        print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
