"""Resident Markov projections shared by all native widths."""

import hashlib
import logging
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


class ThinkingVariant:
    def __init__(self, artifact, model, max_bs):
        head = model.markov_head
        lm_head = model.lm_head.weight
        self.mask = torch.zeros(max_bs, 1, dtype=torch.bool, device=lm_head.device)
        self.last_mask = None
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

    def stage(self, variants):
        if variants != self.last_mask:
            self.last_mask = variants
            padded = list(variants) + [False] * (len(self.mask) - len(variants))
            self.mask[:, 0].copy_(
                async_h2d(padded, dtype=torch.bool, device=self.mask.device)
            )

    def recurrent_projection(self, inputs, stock):
        if self.recurrent is None:
            return stock
        a, b = self.recurrent
        if self.low_rank:
            tuned = stock + F.linear(F.linear(inputs, a), b)
        else:
            tuned = F.linear(inputs, a, b)
        return torch.where(self.mask[:len(stock)], tuned, stock)

    def output_projection(self, latent, stock):
        if self.output is None:
            return stock
        a, b = self.output
        if self.low_rank:
            tuned = stock + F.linear(F.linear(latent, a), b)
        else:
            tuned = F.linear(latent, a, b)
        return torch.where(self.mask[:len(stock)], tuned, stock)
