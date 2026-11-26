"""QISO-x gate C, mechanism half (Infera FLARE-LESSONS 1.6, task sal25). GPU, ~10 min.

Two costs decide whether a device-chosen verify width can pay (break-even about
0.2 ms per round at c1; gate: <= 50 us p50 added per round):

1. The decision: ``drafted_confidence`` over the kept [rows, 7, vocab] step
   logits plus the fused ``qiso_width`` kernel, captured in a CUDA graph.
2. The dispatch: a CUDA SWITCH conditional node (CUDA >= 12.8) whose device-set
   value picks one of four body graphs (stand-ins for the per-width verify
   graphs: one spin kernel each, 4 / 5 / 6 / 8 x a base duration), against
   launching the chosen body graph directly. Built with the explicit node API in
   an inline CUDA extension, so no torch allocation happens during a capture.

The engine half of gate C (16/16 greedy identity per pinned width through the
conditional graph) needs the dispatch wired into the routing worker and runs after
this passes.

    python benchmark/inkling_drafter/bench_qiso_gate_c.py --out qiso-gate-c.json
"""

import argparse
import json
import statistics

import torch

VOCAB = 200_064

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x) == cudaSuccess, cudaGetErrorString(x))

__global__ void spin_kernel(long long ns) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < (unsigned long long)ns);
}

__global__ void set_switch_kernel(cudaGraphConditionalHandle handle, const long long* index) {
  cudaGraphSetConditional(handle, (unsigned int)index[0]);
}

static std::vector<cudaGraphExec_t> g_execs;

static cudaGraphNode_t add_spin(cudaGraph_t graph, const cudaGraphNode_t* deps, size_t ndeps, long long ns) {
  static std::vector<long long*> keep;
  long long* arg = new long long(ns);
  keep.push_back(arg);
  void* args[] = {arg};
  cudaKernelNodeParams p = {};
  p.func = (void*)spin_kernel;
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = args;
  cudaGraphNode_t node;
  CHECK_CUDA(cudaGraphAddKernelNode(&node, graph, deps, ndeps, &p));
  return node;
}

int64_t build_direct(int64_t ns) {
  cudaGraph_t graph;
  CHECK_CUDA(cudaGraphCreate(&graph, 0));
  add_spin(graph, nullptr, 0, ns);
  cudaGraphExec_t exec;
  CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  g_execs.push_back(exec);
  return (int64_t)g_execs.size() - 1;
}

int64_t build_switch(std::vector<int64_t> body_ns, torch::Tensor index) {
  TORCH_CHECK(index.is_cuda() && index.scalar_type() == torch::kLong && index.numel() == 1);
  cudaGraph_t graph;
  CHECK_CUDA(cudaGraphCreate(&graph, 0));
  cudaGraphConditionalHandle handle;
  CHECK_CUDA(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
  const long long* index_ptr = (const long long*)index.data_ptr<int64_t>();
  void* set_args[] = {&handle, &index_ptr};
  cudaKernelNodeParams sp = {};
  sp.func = (void*)set_switch_kernel;
  sp.gridDim = dim3(1);
  sp.blockDim = dim3(1);
  sp.kernelParams = set_args;
  cudaGraphNode_t setter;
  CHECK_CUDA(cudaGraphAddKernelNode(&setter, graph, nullptr, 0, &sp));
  cudaGraphNodeParams cp = {};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = handle;
  cp.conditional.type = cudaGraphCondTypeSwitch;
  cp.conditional.size = (unsigned int)body_ns.size();
  cudaGraphNode_t cond;
#if CUDART_VERSION >= 13000
  CHECK_CUDA(cudaGraphAddNode(&cond, graph, &setter, nullptr, 1, &cp));
#else
  CHECK_CUDA(cudaGraphAddNode(&cond, graph, &setter, 1, &cp));
#endif
  for (size_t i = 0; i < body_ns.size(); ++i) {
    add_spin(cp.conditional.phGraph_out[i], nullptr, 0, body_ns[i]);
  }
  cudaGraphExec_t exec;
  CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  g_execs.push_back(exec);
  return (int64_t)g_execs.size() - 1;
}

void launch(int64_t id) {
  CHECK_CUDA(cudaGraphLaunch(g_execs.at(id), at::cuda::getCurrentCUDAStream()));
}
"""

CPP_DECL = """
int64_t build_direct(int64_t ns);
int64_t build_switch(std::vector<int64_t> body_ns, torch::Tensor index);
void launch(int64_t id);
"""


def timed(fn, reps):
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times), sorted(times)[int(0.9 * len(times))]


def decision_cost(reps):
    from sglang.kernels.ops.speculative.qiso_width import qiso_width
    from sglang.srt.speculative.inkling_width_selector import (
        ExactStopWidth,
        WidthTable,
        drafted_confidence,
    )

    payload = {
        "version": 1, "kind": "qiso-x-table", "bins": 20, "head_sha256": "0" * 64,
        "iso": [[min(1.0, 0.3 + 0.035 * b) for b in range(20)] for _ in range(7)],
        "prior": [0.76, 0.70, 0.68, 0.70, 0.70, 0.71, 0.70],
        "answer": [0.965, 0.959, 0.93, 0.958, 0.958, 0.958, 0.958],
    }
    rule = ExactStopWidth(WidthTable.from_payload(payload), device="cuda")
    rows_out = []
    for rows in (1, 2, 3, 4):
        logits = torch.randn(rows, 7, VOCAB, device="cuda").to(torch.bfloat16)
        tokens = torch.randint(0, VOCAB, (rows, 7), device="cuda")
        temps = torch.full((rows,), 0.7, device="cuda")
        thinking = torch.ones(rows, dtype=torch.bool, device="cuda")
        active = torch.ones(rows, dtype=torch.bool, device="cuda")
        round_ms = torch.tensor([10.6, 11.1, 11.6, 12.5], dtype=torch.float64, device="cuda")
        out = torch.empty(1, dtype=torch.int64, device="cuda")

        def body():
            q = drafted_confidence(logits, tokens, temps)
            qiso_width(q, thinking, active, rule.iso, rule.prior, rule.answer, round_ms, out)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        empty = torch.cuda.CUDAGraph()
        with torch.cuda.graph(empty):
            torch.cuda._sleep(0)
        p50, p90 = timed(graph.replay, reps)
        base50, _ = timed(empty.replay, reps)
        row = {"part": "decision", "rows": rows, "graph_us_p50": round(p50, 2),
               "graph_us_p90": round(p90, 2), "empty_graph_us_p50": round(base50, 2),
               "added_us_p50": round(p50 - base50, 2)}
        print(json.dumps(row))
        rows_out.append(row)
    return rows_out


def switch_cost(reps, base_us):
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        name="qiso_gate_c_switch",
        cpp_sources=CPP_DECL,
        cuda_sources=CUDA_SRC,
        functions=["build_direct", "build_switch", "launch"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    widths = (4, 5, 6, 8)
    body_ns = [int(base_us * 1000 * w / 4) for w in widths]
    index = torch.zeros(1, dtype=torch.int64, device="cuda")
    switch = ext.build_switch(body_ns, index)
    rows_out = []
    for i, width in enumerate(widths):
        direct = ext.build_direct(body_ns[i])
        index.fill_(i)
        torch.cuda.synchronize()
        d50, _ = timed(lambda: ext.launch(direct), reps)
        s50, s90 = timed(lambda: ext.launch(switch), reps)
        row = {"part": "switch", "width": width, "body_us": body_ns[i] / 1000,
               "direct_us_p50": round(d50, 2), "switch_us_p50": round(s50, 2),
               "switch_us_p90": round(s90, 2), "added_us_p50": round(s50 - d50, 2)}
        print(json.dumps(row))
        rows_out.append(row)
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--body-us", type=float, default=2000.0,
                    help="stand-in verify body at width 4 (scaled by width)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    decision = decision_cost(args.reps)
    switch = switch_cost(args.reps, args.body_us)
    worst = max(r["added_us_p50"] for r in decision) + max(r["added_us_p50"] for r in switch)
    verdict = {"gate": "decision + switch added p50 <= 50 us per round (rows 1-4)",
               "worst_added_us_p50": round(worst, 2), "pass": worst <= 50.0}
    print(json.dumps(verdict))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "cuda": torch.version.cuda, "decision": decision, "switch": switch,
                       "verdict": verdict}, f, indent=1)


if __name__ == "__main__":
    main()
