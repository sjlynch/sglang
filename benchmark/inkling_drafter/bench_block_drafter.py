"""Microbench: per-round draft cost of the research block-drafter arms (Infera 5k0kt).

Times, at Inkling-2-Mini drafter dimensions, the work one verify round adds for each
T8 arm (K1 conv, K2 tap fc, C1 extra FFN channels, C2 depth experts) next to the stock
drafter's own pieces: the three-layer block forward's MLPs, the base-logit lm_head GEMM
and the seven-step Markov walk (the "MTP chain": rnn_proj + folded output projection +
sample per depth). Every op set runs eagerly and, on CUDA, also replayed from a captured
CUDA graph (the serving path's draft graph), so the launch cost of small added kernels
shows up as it does in serving.

    # GPU (drafter agent, under a lease; nothing else on the card):
    python benchmark/inkling_drafter/bench_block_drafter.py --device cuda \
        --batches 1 4 8 --width 8 --out block-drafter-microbench.json
    # CPU smoke (reduced dimensions):
    python benchmark/inkling_drafter/bench_block_drafter.py --device cpu --small

Reads: ``added_us`` per arm and batch is the ARM's added draft time per round; compare
with ``stock_us`` (block MLPs + lm_head + chain) and the measured round times
(10.8 / 16.6 / 21.4 ms at c1 / c4 / c8, w4, 5090) in Infera's BLOCK-DRAFTER-SERVING.md.
The K1 conv uses the fork's Triton kernel (``dflash_grouped_conv``) when SGLang imports
on CUDA, else a torch reference of the same math.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location(
    "inkling_block_drafter",
    HERE.parents[2] / "python/sglang/srt/speculative/inkling_block_drafter.py",
)
bd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bd
_spec.loader.exec_module(bd)

FULL = dict(hidden=2304, inter=9216, vocab=201024, layers=3, features=4, rank=256,
            ffn_extra=4608, experts=512, conv_group=16)
SMALL = dict(hidden=256, inter=1024, vocab=8192, layers=3, features=4, rank=64,
             ffn_extra=512, experts=128, conv_group=16)


def _conv_reference(x, delta, base, block, group, taps):
    """dflash _grouped_conv math on flattened [bs*block, H] rows (causal in-block)."""
    rows, hidden = x.shape
    xb = x.float().view(-1, block, hidden)
    coeff = base.float().view(1, 1, taps, hidden) + delta.float().view(-1, block, taps, hidden // group) \
        .repeat_interleave(group, dim=-1)
    out = coeff[:, :, 0] * xb
    for tap in range(1, taps):
        shifted = F.pad(xb[:, :-tap], (0, 0, tap, 0))
        out = out + coeff[:, :, tap] * shifted
    return out.to(x.dtype).view(rows, hidden)


def _conv_fn(device):
    if device.type == "cuda":
        try:
            from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv

            def run(x, delta, base, block, group, taps):
                return dflash_grouped_conv(hidden_states=x, delta=delta, base=base,
                                           block_size=block, group_size=group, taps=taps)
            return run, "triton"
        except Exception:
            pass
    return _conv_reference, "torch-reference"


class Ops:
    """Weights and op sets at one batch size and width."""

    def __init__(self, dims, bs, width, device, dtype):
        g = torch.Generator(device="cpu").manual_seed(0)
        H, I, V, L = dims["hidden"], dims["inter"], dims["vocab"], dims["layers"]

        def w(*shape, scale=None):
            scale = scale or 1 / math.sqrt(shape[-1])
            return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)

        self.dims, self.bs, self.width = dims, bs, width
        rows = bs * width
        self.x = w(rows, H, scale=1.0)
        self.gate_up = [w(2 * I, H) for _ in range(L)]
        self.down = [w(H, I) for _ in range(L)]
        self.extra_gate_up = [w(2 * (I + dims["ffn_extra"]), H) for _ in range(L)]
        self.extra_down = [w(H, I + dims["ffn_extra"]) for _ in range(L)]
        self.lm_head = w(V, H)
        R = dims["rank"]
        self.rnn_proj = w(3 * R, 2 * R + 2 * H)
        self.fold = w(V, R)
        self.markov_w1 = w(V, R)
        groups = H // dims["conv_group"]
        self.conv_proj = [w(2 * 2 * groups, H) for _ in range(2 * L)]
        self.conv_base = torch.zeros(2, 2, H, device=device, dtype=torch.float32)
        self.conv_base[:, 0] = 1
        self.tap_fc = w(H, dims["features"] * H)
        self.taps = w(bs * width, dims["features"], H, scale=1.0)  # <= w committed rows per round
        self.experts = bd.DepthExperts(dims["experts"], H, depths=width - 1, dtype=dtype).to(device)
        with torch.no_grad():
            for name in ("gate", "up", "down", "row_embed"):
                getattr(self.experts, name).copy_(w(*getattr(self.experts, name).shape))
        self.experts.block_size = width
        self.conv, self.conv_kind = _conv_fn(device)

    # --- stock pieces -------------------------------------------------------
    def block_mlps(self):
        h = self.x
        for gu, d in zip(self.gate_up, self.down):
            a, b = F.linear(h, gu).chunk(2, -1)
            h = F.linear(F.silu(a) * b, d)
        return h

    def lm_head_logits(self):
        rows = self.x.view(self.bs, self.width, -1)[:, 1:].reshape(-1, self.x.shape[-1])
        return F.linear(rows, self.lm_head)

    def markov_chain(self):
        R = self.dims["rank"]
        state = self.x.new_zeros(self.bs, R)
        prev = torch.zeros(self.bs, dtype=torch.long, device=self.x.device)
        rows = self.x.view(self.bs, self.width, -1)
        f_rnn = rows[:, 0]
        for j in range(self.width - 1):
            z = torch.cat([state, self.markov_w1[prev], rows[:, j + 1], f_rnn], -1)
            gc, cc, oc = F.linear(z, self.rnn_proj).float().chunk(3, -1)
            gate = torch.sigmoid(gc)
            state = (gate * state.float() + (1 - gate) * torch.tanh(cc)).to(self.x.dtype)
            bias = F.linear(torch.tanh(oc).to(self.x.dtype), self.fold)
            prev = torch.argmax(bias.float(), -1)  # Gumbel-max on kept logits ~ one pass
        return prev

    # --- arm additions ------------------------------------------------------
    def k1_conv(self):
        h = self.x
        group = self.dims["conv_group"]
        for proj in self.conv_proj:
            coeff = F.linear(h, proj).view(h.shape[0], 2, 2, -1)
            h = self.conv(h, coeff[:, 0].contiguous(), self.conv_base[0], self.width, group, 2)
            h = self.conv(h, coeff[:, 1].contiguous(), self.conv_base[1], self.width, group, 2)
        return h

    def k2_tap_fc(self):
        f = self.taps.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
        return F.linear(f.flatten(1).to(self.tap_fc.dtype), self.tap_fc)

    def c1_mlps(self):
        h = self.x
        for gu, d in zip(self.extra_gate_up, self.extra_down):
            a, b = F.linear(h, gu).chunk(2, -1)
            h = F.linear(F.silu(a) * b, d)
        return h

    def c2_experts(self):
        return self.experts(self.experts.embed(self.x))


def _time(fn, device, iters, graph):
    if device.type == "cuda" and graph:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            with torch.cuda.graph(g, stream=stream):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        run = g.replay
    else:
        run = fn
    for _ in range(3):
        run()
    samples = []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)
        else:
            t0 = time.perf_counter()
            run()
            samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--small", action="store_true", help="reduced dimensions (CPU smoke)")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    dims = SMALL if args.small else FULL
    dtype = torch.bfloat16
    rows = []
    with torch.inference_mode():
        for bs in args.batches:
            ops = Ops(dims, bs, args.width, device, dtype)
            t = {name: _time(getattr(ops, name), device, args.iters, not args.no_graph)
                 for name in ("block_mlps", "lm_head_logits", "markov_chain", "k1_conv", "k2_tap_fc",
                              "c1_mlps", "c2_experts")}
            stock = t["block_mlps"] + t["lm_head_logits"] + t["markov_chain"]
            added = {"k1": t["k1_conv"], "k2": t["k1_conv"] + t["k2_tap_fc"],
                     "c1": t["c1_mlps"] - t["block_mlps"], "c2": t["c2_experts"]}
            row = {"batch": bs, "width": args.width, "op_us": {k: round(v, 2) for k, v in t.items()},
                   "stock_us": round(stock, 2), "added_us": {k: round(v, 2) for k, v in added.items()},
                   "added_share_of_stock_drafter": {k: round(v / stock, 4) for k, v in added.items()},
                   "conv_kernel": ops.conv_kind}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del ops
    report = {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "dims": dims, "dtype": str(dtype), "graph": device.type == "cuda" and not args.no_graph,
              "torch": torch.__version__, "rows": rows,
              "note": "added_us is the arm's draft-time increase per round (K2 includes K1: T8's k2 arm is "
                      "--block-conv 2 --tap-fc); C1 is measured as the merged MLP minus the stock MLP"}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    main()
