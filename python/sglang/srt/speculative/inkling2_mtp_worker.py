import itertools
import logging
from copy import copy
from pathlib import Path

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.draft_worker_common import DraftWorkerBundle
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.inkling_draft_policy import DraftPolicy, select_round
from sglang.srt.utils.device import async_h2d

logger = logging.getLogger(__name__)


def _worker_view(worker):
    view = copy(worker)
    runner = copy(worker.model_runner)
    runner.capture_head_hooks = list(runner.capture_head_hooks)
    runner.capture_tail_hooks = list(runner.capture_tail_hooks)
    view._model_runner = runner
    view.model_runner_list = [runner]
    return view


class Inkling2MTPVerifyExecutor(TargetVerifyExecutor):
    def __init__(self, *, draft_model, **kwargs):
        super().__init__(**kwargs)
        self.draft_model = draft_model

    def commit_hidden(
        self,
        *,
        batch,
        layout,
        hidden_strided,
        verify_window,
        logits_output,
        commit_lens,
        bs,
        run_compact,
    ):
        hidden = hidden_strided if run_compact else logits_output.hidden_states
        super().commit_hidden(
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
        )
        assert hidden is not None
        if bs:
            self.draft_model.store_verify_commit(
                req_pool_indices=batch.req_pool_indices,
                verify_hidden=hidden,
                commit_lens=commit_lens,
                stride=self.verify_num_draft_tokens,
            )


class Inkling2MTPWorker(DSparkWorkerV2):
    def __init__(self, *args, **kwargs):
        if get_disagg().disaggregation_mode != "null":
            raise ValueError("Inkling2 MTP does not support disaggregated serving")
        shared = kwargs.get("shared_draft") is not None
        self.inkling_draft_policy = None if shared else DraftPolicy.from_environment()
        self._width_workers = {}
        if self.inkling_draft_policy is not None:
            parallel = get_parallel()
            if (
                parallel.tp_size != 1 or parallel.pp_size != 1 or parallel.attn_dp_enabled
                or not get_schedule().disable_overlap_schedule or get_lora().enable_lora
            ):
                raise ValueError("Inkling drafter routing requires TP1/PP1, no overlap or target LoRA")
            policy = self.inkling_draft_policy
            if get_spec().speculative_num_draft_tokens != policy.capture_widths[-1]:
                raise ValueError("DSpark startup width must equal the largest routing width")
            stock = Path(get_spec().speculative_draft_model_path) / "mtp.safetensors"
            from sglang.srt.speculative.inkling_draft_variants import verify_file

            verify_file(stock, policy.stock_sha256)
        super().__init__(*args, **kwargs)
        if shared or self.inkling_draft_policy is not None:
            if (
                self._verify_planner.mode_value != "static" or not self._decode_graph_allowed
                or self._simulate_acc_len != 0 or self._deferred_draft_kv is not None
            ):
                raise ValueError("Inkling routing requires static native graphs and real inline verification")
            if not 2 <= self.query_token_num <= self.draft_model.config.block_size:
                raise ValueError("Inkling routing width exceeds the checkpoint block size")
            self.draft_model.block_size = self.query_token_num
            self.draft_model.gamma = self.gamma
        else:
            self.draft_model.set_block_size(self.query_token_num)
        if self.inkling_draft_policy is not None and self.inkling_draft_policy.thinking:
            from sglang.srt.speculative.inkling_draft_variants import ThinkingVariant

            variant = ThinkingVariant(
                self.inkling_draft_policy.thinking,
                self.draft_model,
                max(get_exec().graph.cuda_graph_config.decode.bs),
            )
            self.draft_model.markov_head.thinking_variant = variant

    def _resolve_target_embed_tokens(self, target_model):
        return target_model.get_draft_input_embeddings()

    def _build_verify_executor(self, **kwargs):
        return Inkling2MTPVerifyExecutor(draft_model=self.draft_model, **kwargs)

    def init_attention_backends(self):
        super().init_attention_backends()
        draft_runner = self.draft_model_runner
        if (
            draft_runner.prefill_attention_backend_str != "fa4"
            or draft_runner.decode_attention_backend_str != "fa4"
        ):
            raise ValueError("Inkling2 MTP requires FA4 draft attention")
        draft_pool = draft_runner.token_to_kv_pool
        if isinstance(draft_pool, SWAKVPool):
            draft_window = draft_runner.sliding_window_size + 1
            target_window = self.target_worker.sliding_window_size
            if target_window is None or draft_window > target_window:
                raise ValueError("MTP sliding window must not exceed the target window")
        req_pool = self.model_runner.req_to_token_pool
        self.draft_model.ensure_propose_staging(req_pool.req_to_token.shape[0])

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        policy = self.inkling_draft_policy
        if policy is None:
            return
        self._assert_native_graphs()
        for width in policy.capture_widths:
            self._width_workers[width] = (
                self if width == self.verify_num_draft_tokens else self._capture_width(width)
            )
        logger.info("Inkling drafter captured native width set %s", policy.capture_widths)

    def _assert_native_graphs(self):
        for runner in (self.model_runner, self.draft_model_runner):
            if not isinstance(runner.decode_cuda_graph_runner, DecodeCudaGraphRunner):
                raise ValueError("Inkling drafter routing requires both native decode graphs")

    def _capture_width(self, width):
        target = _worker_view(self.target_worker)
        draft = _worker_view(self.draft_worker)
        target_runner = target.model_runner
        draft_runner = draft.model_runner
        target_runner.capture_tail_hooks = [
            hook for hook in target_runner.capture_tail_hooks
            if getattr(hook, "__self__", None) is not self._verify_epilogue
        ]
        draft_runner.capture_tail_hooks = [
            hook for hook in draft_runner.capture_tail_hooks
            if "make_draft_sampler_capture_hook" not in hook.__qualname__
        ]
        model = copy(self.draft_model)
        model._modules = model._modules.copy()
        model.markov_head = copy(model.markov_head)
        model.markov_head._get_staged_f = lambda: model._f_rnn_rows
        model.markov_head._gather_logits = model._gather_logits
        draft_runner.model = model
        bundle = DraftWorkerBundle(draft, draft_runner, model, draft_runner.decode_attention_backend_str)
        with get_spec().override(speculative_num_draft_tokens=width):
            target_runner.attn_backend = target_runner._get_attention_backend(init_new_workspace=True)
            target_runner.runner_ids.bind_and_verify_backends([target_runner.attn_backend])
            target_runner.eager_runner = copy(target_runner.eager_runner)
            target_runner.eager_runner.model_runner = target_runner
            worker = Inkling2MTPWorker(
                self.server_args, self.gpu_id, self.nccl_port, target,
                shared_draft=bundle, shared_noise_from=self,
            )
            worker.init_attention_backends()
            target_runner.decode_cuda_graph_runner = DecodeCudaGraphRunner(
                target_runner, speculative_num_draft_tokens=width
            )
            worker.init_cuda_graphs()
            worker._assert_native_graphs()
            return worker

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, **kwargs):
        policy = self.inkling_draft_policy
        if policy is None or not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch, on_publish, grammar_barrier, **kwargs)
        width, phases, variants = select_round(policy, batch.reqs, batch._inkling_draft_queue)
        resident = self.draft_model.markov_head.thinking_variant
        if resident is not None:
            resident.stage(variants)
        worker = self._width_workers[width]
        result = worker._forward_decode(batch, on_publish, grammar_barrier)
        result.inkling_draft_route = phases, variants
        return result

    def _on_prefill_target_hidden(self, batch, target_hidden):
        assert target_hidden is not None
        assert batch.extend_lens and all(n > 0 for n in batch.extend_lens)
        last_row_ids = async_h2d(
            list(itertools.accumulate(batch.extend_lens)),
            dtype=torch.int64,
            device=target_hidden.device,
        )
        last_rows = target_hidden.index_select(0, last_row_ids - 1)
        self.draft_model.store_f_rnn(batch.req_pool_indices, last_rows)
