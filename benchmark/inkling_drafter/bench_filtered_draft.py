"""GPU micro-benchmark: added time of SGLANG_DSPARK_FILTERED_DRAFT per draft round.

Times one Markov draft step (``MarkovSampleStep.triton``) with and without the
draft filter, captured in a CUDA graph as the folded proposal runs it, over the
Inkling vocabulary, rows 1-4, and reports the added microseconds per round
(three draft steps at width 4, seven at width 8). No model, no engine: needs a
GPU lease, about 2 minutes.

Pass (Infera sal25 plan): added p50 <= 40 us per width-4 round at one row, about a
third of the +1.1% IDs/round the filter buys on a ~12 ms round.

    python benchmark/inkling_drafter/bench_filtered_draft.py --out filtered-draft-overhead.json
"""

import argparse
import json
import statistics

import torch

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.srt.speculative.dspark_components.draft_filter import DraftFilter

VOCAB = 200_064
UNPADDED = 200_058


def bench(rows: int, filtered: bool, max_top_k: int, reps: int, steps: int) -> list[float]:
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(rows)
    base = (torch.randn(rows, VOCAB, device=dev, generator=gen) * 4).to(torch.bfloat16)
    bias = torch.randn(rows, VOCAB, device=dev, generator=gen).to(torch.bfloat16)
    temps = torch.full((rows,), 0.7, device=dev)
    offset = torch.zeros(1, dtype=torch.int64, device=dev)
    logits_out = torch.empty(rows, steps, VOCAB, dtype=torch.bfloat16, device=dev)
    flt = (
        DraftFilter(
            torch.full((rows,), 20, dtype=torch.int64, device=dev),
            torch.full((rows,), 0.8, device=dev),
            max_top_k,
        )
        if filtered
        else None
    )

    def round_():
        for step in range(steps):
            MarkovSampleStep.triton(
                base_logits=base,
                bias=bias,
                temperatures=temps,
                key=1234,
                offset=offset,
                step=step,
                step_dtype=torch.bfloat16,
                logits_out=logits_out[:, step],
                softcap=30.0,
                vocab_limit=UNPADDED,
                draft_filter=flt,
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            round_()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        round_()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--max-top-k", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows_out = []
    for steps in (3, 7):
        for rows in (1, 2, 3, 4):
            plain = bench(rows, False, args.max_top_k, args.reps, steps)
            filt = bench(rows, True, args.max_top_k, args.reps, steps)
            p50 = statistics.median(filt) - statistics.median(plain)
            row = {
                "width": steps + 1,
                "rows": rows,
                "plain_us_p50": round(statistics.median(plain), 2),
                "filtered_us_p50": round(statistics.median(filt), 2),
                "added_us_p50": round(p50, 2),
            }
            rows_out.append(row)
            print(json.dumps(row))
    gate = next(r for r in rows_out if r["width"] == 4 and r["rows"] == 1)
    verdict = {"gate": "added p50 <= 40 us per width-4 round at one row",
               "pass": gate["added_us_p50"] <= 40.0}
    print(json.dumps(verdict))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "max_top_k": args.max_top_k, "rows": rows_out, "verdict": verdict}, f, indent=1)


if __name__ == "__main__":
    main()
