"""CUDA projection epilogues for the resident Inkling Markov adapter."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lora_down(inputs, weight, variants, out, features: tl.constexpr, rank: tl.constexpr,
               block_features: tl.constexpr):
    adapter_row, request = tl.program_id(0), tl.program_id(1)
    result = 0.0
    if tl.load(variants + request):
        lane = tl.arange(0, block_features)
        valid = lane < features
        activation = tl.load(inputs + request * features + lane, mask=valid, other=0)
        factor = tl.load(weight + adapter_row * features + lane, mask=valid, other=0)
        result = tl.sum(activation.to(tl.float32) * factor.to(tl.float32), axis=0)
    tl.store(out + request * rank + adapter_row, result)


@triton.jit
def _lora_residual(
    low, weight, stock, variants, out,
    features: tl.constexpr, rank: tl.constexpr,
    block_features: tl.constexpr, block_rank: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * block_features + tl.arange(0, block_features)
    valid = cols < features
    offset = row * features + cols
    base = tl.load(stock + offset, mask=valid, other=0)
    if tl.load(variants + row):
        lanes = tl.arange(0, block_rank)
        activation = tl.load(low + row * rank + lanes, mask=lanes < rank, other=0)
        factors = tl.load(
            weight + cols[:, None] * rank + lanes[None, :],
            mask=valid[:, None] & (lanes[None, :] < rank), other=0,
        )
        residual = tl.sum(factors.to(tl.float32) * activation[None, :].to(tl.float32), axis=1)
        # The unfused path materializes the BF16 projection before adding the stock value.
        base = (base.to(tl.float32) + residual.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(out + offset, base, mask=valid)


def lora_projection(inputs, a, b, stock, variants):
    """Add the selected BF16 adapter rows while leaving stock rows exact."""
    rows, features = stock.shape
    rank, input_features = a.shape
    low = torch.empty((rows, rank), dtype=inputs.dtype, device=inputs.device)
    _lora_down[(rank, rows)](
        inputs, a, variants, low, input_features, rank,
        triton.next_power_of_2(input_features), num_warps=4,
    )
    block_features = 128 if rank <= 32 and features >= 4096 else 32
    result = torch.empty_like(stock)
    _lora_residual[(triton.cdiv(features, block_features), rows)](
        low, b, stock, variants, result, features, rank,
        block_features, triton.next_power_of_2(rank), num_warps=4,
    )
    return result
