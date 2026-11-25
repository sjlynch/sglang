"""An online draft-head load must refresh the folded Markov output projection (CPU, no CUDA)."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/models/inkling2_mtp.py"


def _load_module():
    def rms_norm(hidden, eps=1e-6):
        return nn.LayerNorm(hidden, eps=eps)

    stubs = {
        "sglang.kernels.ops.speculative.inkling2_mtp": SimpleNamespace(
            mix_rmsnorm=None, rnn_gates=None, rnn_inputs=None),
        "sglang.srt.distributed.device_communicators.vocab_gather": SimpleNamespace(make_vocab_gather=None),
        "sglang.srt.layers.layernorm": SimpleNamespace(RMSNorm=rms_norm),
        "sglang.srt.layers.linear": SimpleNamespace(QKVParallelLinear=object),
        "sglang.srt.layers.logits_processor": SimpleNamespace(should_apply_lm_head_quant_method=None),
        "sglang.srt.mem_cache.memory_pool": SimpleNamespace(KVWriteLoc=object, layer_window_locs=None),
        "sglang.srt.mem_cache.swa_memory_pool": SimpleNamespace(SWAKVPool=object),
        "sglang.srt.model_loader.weight_utils": SimpleNamespace(
            default_weight_loader=lambda param, weight: param.data.copy_(weight)),
        "sglang.srt.models.dflash": SimpleNamespace(DFlashDraftModel=nn.Module),
        "sglang.srt.models.dspark": SimpleNamespace(
            BaseDSparkMarkovHead=nn.Module, DSparkDraftModel=nn.Module, VocabShard=object,
            gather_and_crop_vocab=None, project_through_lm_head=None),
        "sglang.srt.runtime_context": SimpleNamespace(get_parallel=None, get_platform=None),
    }
    spec = importlib.util.spec_from_file_location("inkling2_mtp_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


mtp = _load_module()
BF16 = torch.bfloat16


def small_model(vocab=7, rank=2, hidden=3):
    model = object.__new__(mtp.Inkling2ForConditionalGenerationMTP)
    nn.Module.__init__(model)
    model.markov_head = mtp.Inkling2MTPHead(vocab, rank, hidden, 1e-6).to(BF16)
    model.fuse_weight = nn.Parameter(torch.zeros(2, 3, dtype=BF16), requires_grad=False)
    model._fuse_alpha = None
    model.lm_head = None
    model.logits_mup_width_multiplier = 2.0
    return model


def checkpoint(model, seed):
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(param.shape, generator=generator).to(BF16)
            for name, param in model.named_parameters()}


def expected_fold(lm_head, hidden_out, multiplier):
    return ((lm_head.double() @ hidden_out.weight.double()) / multiplier).to(BF16)


class OnlineFoldTests(unittest.TestCase):
    def test_startup_load_before_attach_leaves_the_fold_to_attach(self):
        model = small_model()
        model.load_weights(checkpoint(model, 1).items())
        self.assertIsNone(model.markov_head._fold_weight)

    def test_online_load_refreshes_the_fold_in_place(self):
        model = small_model()
        model.load_weights(checkpoint(model, 1).items())
        lm_head = torch.randn(7, 3, generator=torch.Generator().manual_seed(9)).to(BF16)
        model.lm_head = SimpleNamespace(weight=lm_head)
        model.markov_head.attach_fold(lm_head, model.logits_mup_width_multiplier)
        fold = model.markov_head._fold_weight
        address = fold.data_ptr()
        before = fold.clone()
        model.load_weights(checkpoint(model, 2).items())
        head = model.markov_head
        self.assertEqual(head._fold_weight.data_ptr(), address)  # graph-captured address kept
        self.assertFalse(torch.equal(head._fold_weight, before))
        self.assertTrue(torch.equal(head._fold_weight, expected_fold(lm_head, head.hidden_out, 2.0)))
        bias = ((lm_head.double() @ head.hidden_out.bias.double()) / 2.0).to(BF16)
        self.assertTrue(torch.equal(head._fold_bias, bias))

    def test_swapping_back_restores_the_original_fold_exactly(self):
        model = small_model()
        first = checkpoint(model, 1)
        model.load_weights(first.items())
        lm_head = torch.randn(7, 3, generator=torch.Generator().manual_seed(9)).to(BF16)
        model.lm_head = SimpleNamespace(weight=lm_head)
        model.markov_head.attach_fold(lm_head, model.logits_mup_width_multiplier)
        original = model.markov_head._fold_weight.clone()
        model.load_weights(checkpoint(model, 2).items())
        model.load_weights(first.items())
        self.assertTrue(torch.equal(model.markov_head._fold_weight, original))


if __name__ == "__main__":
    unittest.main()
