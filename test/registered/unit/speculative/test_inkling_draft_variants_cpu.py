"""Exercise resident projection selection and artifact validation without CUDA."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/speculative/inkling_draft_variants.py"
spec = importlib.util.spec_from_file_location("inkling_draft_variants", SOURCE)
variants = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"sglang.srt.utils.device": SimpleNamespace(async_h2d=torch.tensor)}):
    spec.loader.exec_module(variants)


def bf16(value):
    return torch.tensor(value, dtype=torch.bfloat16)


class ThinkingVariantTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "weights.safetensors"
        self.model = SimpleNamespace(
            markov_head=SimpleNamespace(
                rnn_proj=SimpleNamespace(weight=bf16([[0, 0], [0, 0]]), bias=bf16([0, 0])),
                hidden_out=SimpleNamespace(weight=bf16([[0], [0]]), bias=bf16([0, 0])),
            ),
            lm_head=SimpleNamespace(weight=bf16([[1, 0], [0, 2], [1, 1]])),
            logits_mup_width_multiplier=2,
        )

    def artifact(self, kind, tensors):
        save_file(tensors, self.path)
        return SimpleNamespace(
            kind=kind, path=self.path, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        )

    def test_lora_mixed_rows_and_second_round(self):
        artifact = self.artifact("markov-lora", {
            "rnn_proj.a": bf16([[1, 1]]), "rnn_proj.b": bf16([[1], [2]]),
            "hidden_out.a": bf16([[1]]), "hidden_out.b": bf16([[2], [4]]),
        })
        runtime = variants.ThinkingVariant(artifact, self.model, 4)
        runtime.stage((True, False))
        rnn_stock = bf16([[0, 1], [2, 3]])
        logits_stock = bf16([[0, 1, 2], [3, 4, 5]])
        inputs, latent = bf16([[1, 2], [3, 4]]), bf16([[1], [2]])
        self.assertTrue(torch.equal(
            runtime.recurrent_projection(inputs, rnn_stock), bf16([[3, 7], [2, 3]])
        ))
        self.assertTrue(torch.equal(
            runtime.output_projection(latent, logits_stock), bf16([[1, 5, 5], [3, 4, 5]])
        ))
        runtime.stage((False, True))
        self.assertTrue(torch.equal(
            runtime.output_projection(latent, logits_stock), bf16([[0, 1, 2], [5, 12, 11]])
        ))

    def test_merged_bf16_markov_head_shares_unembed_and_keeps_stock_rows(self):
        artifact = self.artifact("markov-head", {
            "markov_head.rnn_proj.weight": bf16([[1, 0], [0, 1]]),
            "markov_head.rnn_proj.bias": bf16([2, 3]),
            "markov_head.hidden_out.weight": bf16([[2], [4]]),
            "markov_head.hidden_out.bias": bf16([2, 0]),
        })
        runtime = variants.ThinkingVariant(artifact, self.model, 4)
        runtime.stage((False, True))
        self.assertTrue(torch.equal(
            runtime.recurrent_projection(bf16([[1, 2], [3, 4]]), bf16([[1, 2], [3, 4]])),
            bf16([[1, 2], [5, 7]]),
        ))
        self.assertTrue(torch.equal(
            runtime.output_projection(bf16([[1], [2]]), bf16([[0, 1, 2], [3, 4, 5]])),
            bf16([[0, 1, 2], [3, 8, 7]]),
        ))

    def test_bad_digest_and_factor_pair_fail_at_startup(self):
        artifact = self.artifact("markov-lora", {"rnn_proj.a": bf16([[1, 1]])})
        with self.assertRaisesRegex(ValueError, "Missing LoRA matrix"):
            variants.ThinkingVariant(artifact, self.model, 4)
        artifact.sha256 = "a" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            variants.ThinkingVariant(artifact, self.model, 4)

    def test_resident_mask_copy_keeps_padding_clear(self):
        runtime = variants.ThinkingVariant.__new__(variants.ThinkingVariant)
        runtime.init_mask(4, 'cpu')
        runtime.mask_patterns = torch.tensor(
            [[bool(bits & (1 << row)) for row in range(4)] for bits in range(16)],
        )
        pointer = runtime.mask.data_ptr()
        runtime.stage((True, False, True, True))
        runtime.stage((False, True))
        self.assertEqual(pointer, runtime.mask.data_ptr())
        self.assertEqual(runtime.mask[:, 0].tolist(), [False, True, False, False])
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'larger than its captured batch'):
                runtime.stage((True,) * 5)
        self.assertEqual(runtime.mask[:, 0].tolist(), [False, True, False, False])


if __name__ == "__main__":
    unittest.main()
