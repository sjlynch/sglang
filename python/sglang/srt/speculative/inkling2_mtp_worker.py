import itertools
import logging
from copy import copy
from pathlib import Path
from time import perf_counter_ns

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
from sglang.srt.speculative.inkling_draft_policy import (
    CONTEXT_MS_PER_TOKEN,
    ROUND_COSTS,
    WIDTHS,
    DraftPolicy,
    arm_width,
    batch_width_arm,
    RequestPhase,
    p3_shadow_width,
    qiso_phase_ids,
    select_batch_round,
    select_round,
    tracking_expectations,
)
from sglang.srt.utils.device import async_h2d

logger = logging.getLogger(__name__)


def _context_tokens(requests):
    return sum(len(req.origin_input_ids) + len(req.output_ids) for req in requests)


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
        # perf_counter_ns once the target verify forward (the graph replay) was
        # launched: the end of a QISO-x dispatch bubble.
        self.launched_ns = None

    def _forward_prepared_verify(self, **kwargs):
        result = super()._forward_prepared_verify(**kwargs)
        self.launched_ns = perf_counter_ns()
        return result

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
        self._batch_workers = ()
        self._batch_width_index = {}
        self._qiso_table = self._qiso_rule = None
        self._qiso_decision = self._qiso_dispatcher = self._qiso_round_log = None
        self._qiso_shadow_proposal = None
        from sglang.srt.speculative.dspark_components.draft_filter import filtered_draft_enabled

        self._filtered_draft = filtered_draft_enabled()
        if self.inkling_draft_policy is not None:
            policy = self.inkling_draft_policy
            if policy.width_selector is not None:
                from sglang.srt.environ import envs
                from sglang.srt.speculative.inkling_width_selector import WidthTable

                self._qiso_table = WidthTable.read(
                    policy.width_selector.path, policy.width_selector.sha256
                )
                if policy.pin[0] != 0 or self._qiso_table.head_sha256 != policy.stock_sha256:
                    raise ValueError("The QISO-x table must be fitted for the pinned stock head")
                if envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get():
                    raise ValueError("QISO-x reads the sampled draft's kept step logits")
            parallel = get_parallel()
            if (
                parallel.tp_size != 1 or parallel.pp_size != 1 or parallel.attn_dp_enabled
                or get_lora().enable_lora
            ):
                raise ValueError("Inkling drafter routing requires TP1/PP1 and unadapted target weights")
            if get_schedule().disable_overlap_schedule != (policy.batch_mode is None):
                raise ValueError("Batch routing requires overlap and legacy mixed-row routing requires non-overlap")
            if get_spec().speculative_num_draft_tokens != policy.capture_widths[-1]:
                raise ValueError("DSpark startup width must equal the largest routing width")
            stock = Path(get_spec().speculative_draft_model_path) / "mtp.safetensors"
            from sglang.srt.speculative.inkling_draft_variants import verify_file

            verify_file(stock, policy.stock_sha256)
        super().__init__(*args, **kwargs)
        if shared or self.inkling_draft_policy is not None:
            if (
                self._verify_planner.mode_value != "static" or not self._decode_graph_allowed
                or self._simulate_acc_len > 0 or self._deferred_draft_kv is not None
            ):
                raise ValueError("Inkling routing requires static native graphs and real inline verification")
            if not 2 <= self.query_token_num <= self.draft_model.config.block_size:
                raise ValueError("Inkling routing width exceeds the checkpoint block size")
            if any(
                getattr(layer, name, None) is not None
                for layer in self.draft_model.layers for name in ("attention_conv", "mlp_conv")
            ):
                # Width views share draft layers; a block-size-indexed convolution would
                # take the last view's block size for every width.
                raise ValueError("Inkling routing cannot share block-size-dependent draft layers")
            self.draft_model.block_size = self.query_token_num
            self.draft_model.gamma = self.gamma
        else:
            self.draft_model.set_block_size(self.query_token_num)
        if (
            self.inkling_draft_policy is not None and self.inkling_draft_policy.thinking
            and self.inkling_draft_policy.batch_mode is None
        ):
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
        policy = self.inkling_draft_policy
        models = [self.draft_model]
        if policy is not None and policy.capture_heads == 2:
            from sglang.srt.speculative.inkling_draft_variants import build_batch_head

            stock = Path(get_spec().speculative_draft_model_path) / "mtp.safetensors"
            models.append(build_batch_head(policy.thinking, self.draft_model, stock))
        super().init_cuda_graphs()
        if policy is None:
            return
        self._assert_native_graphs()
        if policy.batch_mode is not None:
            if policy.acceptance_tracking:
                tracking_expectations()
            widths = policy.capture_widths
            self._batch_width_index = {width: i for i, width in enumerate(widths)}
            self._batch_workers = tuple(
                self if head == 0 and width == self.verify_num_draft_tokens
                else self._capture_width(width, model)
                for head, model in enumerate(models) for width in widths
            )
            logger.info("Inkling batch routing captured %d resident heads at widths %s", len(models), widths)
            if self._qiso_table is not None:
                self._init_qiso(policy)
            return
        for width in policy.capture_widths:
            self._width_workers[width] = (
                self if width == self.verify_num_draft_tokens else self._capture_width(width)
            )
        logger.info("Inkling drafter captured native width set %s", policy.capture_widths)

    def _init_qiso(self, policy):
        from sglang.srt.speculative.inkling_width_dispatch import (
            QisoDecision,
            QisoDispatcher,
            QisoRoundLog,
        )

        self._qiso_decision = QisoDecision(
            self._qiso_table, self.device, ROUND_COSTS, CONTEXT_MS_PER_TOKEN, h2d=async_h2d
        )
        self._qiso_rule = self._qiso_decision.rule
        head, width = policy.pin
        if self._qiso_table.tool_source != "table":
            logger.warning(
                "QISO-x table %s has no tool table: tool-call and grammar rows are priced as answers",
                policy.width_selector.sha256[:12],
            )
        if policy.width_selector_mode == "active":
            tracking_expectations()  # the in-process P3 arm
            missing = set(WIDTHS) - set(self._batch_width_index)
            if missing:
                raise ValueError(f"QISO-x active dispatch needs captured verify widths {WIDTHS}")
            if getattr(getattr(self._observers, "_info_dumper", None), "enabled", False):
                raise ValueError("QISO-x active dispatch does not support the DSpark info dumper")
            self._qiso_dispatcher = QisoDispatcher(
                self._qiso_decision, force_width=policy.width_selector_force
            )
            logger.info(
                "Inkling QISO-x ACTIVE dispatch on (table %s v%d%s): draft at width %d, verify at the device-chosen width",
                policy.width_selector.sha256[:12], self._qiso_table.version,
                "" if policy.width_selector_force is None else f", forced width {policy.width_selector_force}",
                width,
            )
            return
        tracking_expectations()  # the shadow replays P3
        self._qiso_round_log = QisoRoundLog.from_environment(self.device)
        worker = self._batch_workers[head * len(self._batch_width_index) + self._batch_width_index[width]]
        proposer = worker._proposer
        propose = proposer.propose

        def remember(*args, **kwargs):
            proposal = propose(*args, **kwargs)
            worker._qiso_last_proposal = proposal
            return proposal

        worker._qiso_last_proposal = None
        proposer.propose = remember
        logger.info("Inkling QISO-x shadow on (table %s): verify pinned at width %d", policy.width_selector.sha256[:12], width)

    def _qiso_shadow(self, worker, batch, phases):
        """Decide the QISO-x width on the device from this round's drafted q (no
        host read) and queue a pinned copy; the result processor reads it after
        copy_done, beside the realized width-8 acceptance."""
        proposal = getattr(worker, "_qiso_last_proposal", None)
        worker._qiso_last_proposal = None
        self._qiso_shadow_proposal = proposal
        if proposal is None:
            return None
        block = proposal.draft_block
        bs = len(batch.reqs)
        # The same decision the active dispatch makes (inkling_width_dispatch.py).
        width = self._qiso_decision.decide(
            block, bs, qiso_phase_ids(batch.reqs), _context_tokens(batch.reqs)
        )
        if width is None:
            return None
        host = torch.empty(1, dtype=torch.int64, pin_memory=width.is_cuda)
        host.copy_(width, non_blocking=True)
        return host

    def _qiso_log_round(self, proposal, batch, result):
        """SGLANG_INKLING_QISO_ROUND_LOG: one width-8 round per row for refitting
        the phase tables (tool output on agentic panels, a sampling regime)."""
        if proposal is None:
            return
        logits = getattr(result.logits_output, "next_token_logits", None)
        self._qiso_round_log.record(
            proposal.draft_block, len(batch.reqs), logits, result.accept_lens,
            batch.reqs, qiso_phase_ids(batch.reqs),
        )

    def _qiso_dispatch_round(self, batch, on_publish, grammar_barrier):
        """QISO-x active: draft with the width-8 graph, decide on the device, read
        the width once (the bounded sync), verify with that width's graph."""
        index = self._batch_width_index
        drafter = self._batch_workers[index[8]]
        dispatcher = self._qiso_dispatcher
        bs = len(batch.reqs)
        phase_ids = qiso_phase_ids(batch.reqs)
        context = _context_tokens(batch.reqs)
        round_state = {}

        def after_propose(proposal):
            width, wait_ns, fallback = dispatcher.read_width(proposal.draft_block, bs, phase_ids, context)
            round_state.update(width=width, wait_ns=wait_ns, fallback=fallback)
            worker = drafter if width == 8 else self._batch_workers[index[width]]
            worker._verify_executor.launched_ns = None
            round_state["executor"] = worker._verify_executor
            if width == 8:
                return None
            from sglang.srt.speculative.inkling_width_dispatch import slice_proposal

            with get_spec().override(speculative_num_draft_tokens=width):
                return worker._forward_decode(
                    batch, on_publish, grammar_barrier, injected_proposal=slice_proposal(proposal, width)
                )

        with get_spec().override(speculative_num_draft_tokens=8):
            result = drafter._forward_decode(batch, on_publish, grammar_barrier, after_propose=after_propose)
        width = round_state.get("width", 8)
        executor = round_state.get("executor")
        bubble = dispatcher.bubble_ns(None if executor is None else executor.launched_ns)
        wait_ns = round_state.get("wait_ns", 0)
        fallback = round_state.get("fallback", True)
        dispatcher.stats.note(width, wait_ns, bubble, fallback)
        result.inkling_qiso_active = (width, wait_ns, bubble, fallback)
        return result

    def _assert_native_graphs(self):
        for runner in (self.model_runner, self.draft_model_runner):
            if not isinstance(runner.decode_cuda_graph_runner, DecodeCudaGraphRunner):
                raise ValueError("Inkling drafter routing requires both native decode graphs")

    def _capture_width(self, width, resident_model=None):
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
        model = copy(self.draft_model if resident_model is None else resident_model)
        if model._f_rnn_buf is None or model._f_rnn_buf is not self.draft_model._f_rnn_buf:
            raise ValueError("Every Inkling routing graph must read the one committed recurrent state")
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
            if self._verify_epilogue is not None and worker._verify_epilogue is not None:
                # All graphs publish by request slot into the one FutureMap relay adopted
                # from the stock capture (None unless POST_FOLD). A width view must never
                # build and offer its own relay, which would replace the adopted one.
                worker._verify_epilogue.future_relay = self._verify_epilogue.future_relay
            worker.init_attention_backends()
            target_runner.decode_cuda_graph_runner = DecodeCudaGraphRunner(
                target_runner, speculative_num_draft_tokens=width
            )
            worker.init_cuda_graphs()
            worker._assert_native_graphs()
            return worker

    def _stage_draft_filter_opt_out(self, batch):
        """SGLANG_DSPARK_FILTERED_DRAFT: rows whose request sets custom_params
        dspark_draft_filter false draft unfiltered, so one warm process can serve
        both arms of an in-process A/B (Infera head_ab, task sal25)."""
        sampling_info = getattr(batch, "sampling_info", None)
        if sampling_info is None or not batch.reqs:
            return
        off = []
        for req in batch.reqs:
            custom = req.sampling_params.custom_params or {}
            enabled = custom.get("dspark_draft_filter", True) is not False
            req._dspark_draft_filter = enabled
            off.append(not enabled)
        sampling_info.draft_filter_off = (
            async_h2d(off, dtype=torch.bool, device=self.device) if any(off) else None
        )

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, **kwargs):
        policy = self.inkling_draft_policy
        if self._filtered_draft:
            self._stage_draft_filter_opt_out(batch)
        if policy is None or not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
            if policy is not None and policy.batch_mode is not None and batch.forward_mode.is_extend():
                for req in batch.reqs:
                    if not hasattr(req, "_inkling_draft_phase"):
                        req._inkling_draft_phase = RequestPhase.from_request(req)
            return super().forward_batch_generation(batch, on_publish, grammar_barrier, **kwargs)
        if policy.batch_mode is not None:
            head, width, phases = select_batch_round(
                policy, batch.reqs, getattr(batch, "_inkling_draft_queue", 0),
                getattr(batch, "_inkling_draft_completed_channels", None),
            )
            batch._inkling_draft_completed_channels = None
            if self._qiso_dispatcher is not None:
                arm, mixed = batch_width_arm(batch.reqs)
                if arm == "qiso-x":
                    result = self._qiso_dispatch_round(batch, on_publish, grammar_barrier)
                else:
                    # P3 or a fixed width on its own native graph (in-process M1 arms).
                    width = arm_width(arm, batch.reqs, phases)
                    worker = self._batch_workers[self._batch_width_index[width]]
                    with get_spec().override(speculative_num_draft_tokens=width):
                        result = worker._forward_decode(batch, on_publish, grammar_barrier)
                result.inkling_draft_route = phases, (False,) * len(phases)
                result.inkling_track_acceptance = arm == "p3"
                result.inkling_width_arm = arm, mixed
                return result
            index = head * len(self._batch_width_index) + self._batch_width_index[width]
            worker = self._batch_workers[index]
            with get_spec().override(speculative_num_draft_tokens=width):
                result = worker._forward_decode(batch, on_publish, grammar_barrier)
            result.inkling_draft_route = phases, (bool(head),) * len(phases)
            result.inkling_track_acceptance = policy.acceptance_tracking
            if self._qiso_rule is not None:
                result.inkling_qiso_shadow = self._qiso_shadow(worker, batch, phases)
                result.inkling_p3_shadow = p3_shadow_width(batch.reqs, phases)
                if self._qiso_round_log is not None:
                    self._qiso_log_round(self._qiso_shadow_proposal, batch, result)
                self._qiso_shadow_proposal = None
            return result
        suffix_stats = [] if policy.anchor_suffix_width8 else None
        width, phases, variants = select_round(
            policy, batch.reqs, batch._inkling_draft_queue, suffix_stats
        )
        resident = self.draft_model.markov_head.thinking_variant
        if resident is not None:
            resident.stage(variants)
        worker = self._width_workers[width]
        result = worker._forward_decode(batch, on_publish, grammar_barrier)
        result.inkling_draft_route = phases, variants
        result.inkling_suffix_stats = suffix_stats
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
