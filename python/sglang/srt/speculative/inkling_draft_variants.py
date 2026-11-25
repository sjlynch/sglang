"""Startup loading for resident Inkling draft variants."""

import hashlib
import logging
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from sglang.srt.utils.device import async_h2d

logger = logging.getLogger(__name__)
PROJECTIONS = ("rnn_proj", "hidden_out")
PREFIX = "markov_head."


def verify_file(path, expected):
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Inkling drafter artifact digest mismatch: {path}")


def checked_tensor(source, name, shape, *, bf16=False):
    tensor = source.get_tensor(name)
    dtypes = (torch.bfloat16,) if bf16 else (torch.bfloat16, torch.float32, torch.float16)
    if tensor.shape != shape or tensor.dtype not in dtypes or not torch.isfinite(tensor).all():
        raise ValueError(f"Invalid Inkling drafter tensor: {name}")
    return tensor


def read_adapter(source, name, weight):
    a = source.get_tensor(name + ".a")
    if a.ndim != 2 or not 1 <= a.shape[0] <= 256:
        raise ValueError(f"Invalid Inkling drafter adapter rank: {name}")
    rank = a.shape[0]
    return (
        checked_tensor(source, name + ".a", (rank, weight.shape[1])),
        checked_tensor(source, name + ".b", (weight.shape[0], rank)),
    )


def fold_delta(lm_head, matrix, multiplier):
    # Match the stock fold's chunking and accumulation precision at startup.
    result = lm_head.new_empty(lm_head.shape[0], matrix.shape[1])
    matrix = matrix.to(device=lm_head.device, dtype=torch.float64)
    for start in range(0, lm_head.shape[0], 4096):
        rows = slice(start, start + 4096)
        result[rows].copy_((lm_head[rows].double() @ matrix) / (multiplier or 1))
    return result


def writes_shared_context(name):
    # These parameters produce the committed draft KV and recurrent state, not the proposal alone.
    return name == "fuse_weight" or any(
        segment in name for segment in (".dflare_norm.", ".kv_proj_ctx.", ".k_proj_ctx.", ".v_proj_ctx.", ".k_norm_ctx.")
    )


def native_head_tensors(source, stock):
    if set(source.keys()) != set(stock.keys()):
        raise ValueError("Resident native head schema differs from the stock checkpoint")
    for name in source.keys():
        original = stock.get_tensor(name)
        tensor = checked_tensor(source, name, original.shape, bf16=True)
        if writes_shared_context(name) and not torch.equal(tensor, original):
            raise ValueError(f"Resident head changes shared prefix state: {name}")
        yield name, tensor


@torch.no_grad()
def build_batch_head(artifact, model, stock_path):
    """Materialize independent draft parameters before the variant graphs capture their addresses."""
    verify_file(artifact.path, artifact.sha256)
    # Target embeddings and unembed are immutable. The shared committed recurrent buffer
    # is valid only for variants with unchanged context projections and normalization.
    if model._f_rnn_buf is None:
        raise ValueError("Allocate the committed recurrent state before building resident heads")
    shared = (model.embed_tokens, model.lm_head, model._f_rnn_buf, model.config)
    memo = {id(value): value for value in shared if value is not None}
    if artifact.kind != "native-head":
        # A Markov variant changes only rnn_proj/hidden_out and its fold. Share every other
        # module and tensor, so the 14 context writers are the stock tensors by identity and
        # the draft layers are not duplicated. The fold buffers stay private: refresh_fold
        # writes them in place.
        for name, module in model.named_children():
            if name != "markov_head":
                memo[id(module)] = module
        tensors = (*model.parameters(recurse=False), *model.buffers(recurse=False),
                   *(value for value in vars(model).values() if isinstance(value, torch.Tensor)))
        for value in tensors:
            memo[id(value)] = value
        for name, module in model.markov_head.named_children():
            if name not in PROJECTIONS:
                memo[id(module)] = module
    variant = deepcopy(model, memo)
    if variant._f_rnn_buf is not model._f_rnn_buf:
        raise ValueError("Resident heads must share the committed recurrent state")
    head = variant.markov_head
    head._get_staged_f = lambda: variant._f_rnn_rows
    head._gather_logits = variant._gather_logits
    variant._f_rnn_rows = None
    with safe_open(artifact.path, framework="pt", device="cpu") as source:
        if artifact.kind == "native-head":
            # Parameter.__deepcopy__ drops the custom packed/sharded loaders.
            # Bind loaders to the cloned layers before reading checkpoint shards.
            for name, parameter in model.named_parameters():
                if not name.startswith(("embed_tokens.", "lm_head.")):
                    vars(memo[id(parameter)]).update(deepcopy(vars(parameter), memo))
            with safe_open(stock_path, framework="pt", device="cpu") as stock:
                variant.load_weights(native_head_tensors(source, stock))
        elif artifact.kind == "markov-head":
            for name in PROJECTIONS:
                layer = getattr(head, name)
                for suffix in ("weight", "bias"):
                    parameter = getattr(layer, suffix)
                    parameter.copy_(checked_tensor(source, f"{PREFIX}{name}.{suffix}", parameter.shape, bf16=True))
        else:
            keys = set(source.keys())
            supported = {f"{name}.{suffix}" for name in PROJECTIONS for suffix in ("a", "b")}
            if not keys or keys - supported:
                raise ValueError("Expected Markov-only LoRA tensors")
            for name in PROJECTIONS:
                pair = {f"{name}.{suffix}" for suffix in ("a", "b")}
                if not keys & pair:
                    continue
                if not pair <= keys:
                    raise ValueError(f"Missing LoRA matrix for {name}")
                weight = getattr(head, name).weight
                a, b = read_adapter(source, name, weight)
                a = a.to(device=weight.device, dtype=torch.float32)
                b = b.to(device=weight.device, dtype=torch.float32)
                weight.copy_(weight.float() + b @ a)
    head.refresh_fold(variant.lm_head.weight, variant.logits_mup_width_multiplier)
    return variant


class ThinkingVariant:
    def __init__(self, artifact, model, max_bs):
        head = model.markov_head
        lm_head = model.lm_head.weight
        self.init_mask(max_bs, lm_head.device)
        self.recurrent = self.output = None
        self.low_rank = artifact.kind == "markov-lora"
        verify_file(artifact.path, artifact.sha256)
        with safe_open(artifact.path, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            if self.low_rank:
                supported = {
                    f"{name}.{suffix}"
                    for name in PROJECTIONS for suffix in ("a", "b")
                }
                if not keys or keys - supported:
                    raise ValueError("Expected Markov-only LoRA tensors")
                for name in PROJECTIONS:
                    pair = {f"{name}.{suffix}" for suffix in ("a", "b")}
                    if not keys & pair:
                        continue
                    if not pair <= keys:
                        raise ValueError(f"Missing LoRA matrix for {name}")
                    a, b = read_adapter(source, name, getattr(head, name).weight)
                    if name == "rnn_proj":
                        self.recurrent = a.to(lm_head), b.to(lm_head)
                    else:
                        self.output = a.to(lm_head), fold_delta(
                            lm_head, b, model.logits_mup_width_multiplier
                        )
            else:
                for name in PROJECTIONS:
                    module = getattr(head, name)
                    prefix = PREFIX + name
                    weight = checked_tensor(
                        source, prefix + ".weight", module.weight.shape, bf16=True
                    )
                    bias = checked_tensor(
                        source, prefix + ".bias", module.bias.shape, bf16=True
                    )
                    if name == "rnn_proj":
                        self.recurrent = weight.to(lm_head), bias.to(lm_head)
                    else:
                        multiplier = model.logits_mup_width_multiplier
                        self.output = (
                            fold_delta(lm_head, weight, multiplier),
                            fold_delta(lm_head, bias[:, None], multiplier)[:, 0],
                        )
        logger.info("Loaded resident Inkling thinking variant (%s): sha256=%s",
                    artifact.kind, artifact.sha256)

    def init_mask(self, max_bs, device):
        self.mask = torch.zeros(max_bs, 1, dtype=torch.bool, device=device)
        self.last_mask = None
        self.mask_patterns = None
        if self.mask.is_cuda and max_bs <= 8:
            # The indexed view and copy stay on device. The destination address is fixed for graph replay.
            self.mask_patterns = torch.tensor(
                [[bool(bits & (1 << row)) for row in range(max_bs)] for bits in range(1 << max_bs)],
                dtype=torch.bool, device=device,
            )

    def stage(self, variants):
        variants = tuple(variants)
        if variants != self.last_mask:
            if len(variants) > len(self.mask):
                raise ValueError("Inkling drafter mask is larger than its captured batch")
            if self.mask_patterns is not None:
                bits = sum((1 << row) for row, selected in enumerate(variants) if selected)
                self.mask[:, 0].copy_(self.mask_patterns[bits], non_blocking=True)
            else:
                padded = list(variants) + [False] * (len(self.mask) - len(variants))
                self.mask[:, 0].copy_(
                    async_h2d(padded, dtype=torch.bool, device=self.mask.device)
                )
            self.last_mask = variants

    def recurrent_projection(self, inputs, stock):
        if self.recurrent is None:
            return stock
        a, b = self.recurrent
        if self.low_rank:
            return self._apply_lora(inputs, a, b, stock)
        return torch.where(self.mask[:len(stock)], F.linear(inputs, a, b), stock)

    def output_projection(self, latent, stock):
        if self.output is None:
            return stock
        a, b = self.output
        if self.low_rank:
            return self._apply_lora(latent, a, b, stock)
        return torch.where(self.mask[:len(stock)], F.linear(latent, a, b), stock)

    def _apply_lora(self, inputs, a, b, stock):
        if (
            stock.is_cuda and stock.dtype == inputs.dtype == a.dtype == b.dtype == torch.bfloat16
            and stock.ndim == inputs.ndim == 2 and 0 < len(stock) == len(inputs) <= len(self.mask)
            and a.shape[0] <= 32 and a.shape[1] <= 8192
            and all(tensor.is_contiguous() for tensor in (inputs, a, b, stock))
        ):
            from sglang.srt.speculative.inkling_draft_variant_kernels import lora_projection

            return lora_projection(inputs, a, b, stock, self.mask)
        tuned = stock + F.linear(F.linear(inputs, a), b)
        return torch.where(self.mask[:len(stock)], tuned, stock)
