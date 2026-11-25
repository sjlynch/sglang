"""Standalone CPU checks and c4 routing micro-benchmark (no SGLang imports)."""

import importlib.util
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/speculative/inkling_draft_policy.py"
spec = importlib.util.spec_from_file_location("inkling_draft_policy", SOURCE)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def request(prompt=(), output=(), variant="auto"):
    return SimpleNamespace(
        origin_input_ids=list(prompt),
        output_ids=list(output),
        sampling_params=SimpleNamespace(custom_params={"inkling_draft_variant": variant}),
    )


def config(artifact=True):
    return policy.DraftPolicy(
        (policy.WidthTier(2, 6, 4), policy.WidthTier(None, 8, 6)),
        "a" * 64,
        policy.Artifact("markov-lora", Path("/local/adapter"), "b" * 64) if artifact else None,
    )


class DraftPolicyTests(unittest.TestCase):
    def test_prompt_and_only_new_committed_tokens_drive_phase(self):
        req = request((policy.THINK_START, 123), (456,))
        self.assertEqual(policy.select_round(config(), [req], 0), (6, (True,), (True,)))
        req.output_ids += [policy.THINK_END, 789]
        self.assertEqual(policy.select_round(config(), [req], 0), (4, (False,), (False,)))
        req.output_ids = [policy.THINK_START, 999]
        req._inkling_draft_phase.seen = 3  # output was rewound by scheduler
        self.assertEqual(policy.select_round(config(), [req], 0), (6, (True,), (True,)))
        req.output_ids += [200049]
        self.assertEqual(policy.select_round(config(), [req], 0), (4, (False,), (False,)))

    def test_mixed_rows_keep_variants_and_use_common_native_width(self):
        requests = [
            request((policy.THINK_START,)),
            request(),
            request(variant="thinking"),
            request((policy.THINK_START,), variant="stock"),
        ]
        width, phases, variants = policy.select_round(config(), requests, 2)
        self.assertEqual((width, phases, variants),
                         (6, (True, False, False, True), (True, False, True, False)))
        self.assertEqual(policy.select_round(config(), requests[:1], 1)[0], 6)
        self.assertEqual(policy.select_round(config(), requests[:1], 2)[0], 8)
        self.assertFalse(any(policy.select_round(config(False), requests, 0)[2]))

    def test_override_validation(self):
        for override in ("auto", "stock", "thinking"):
            policy.validate_override({"inkling_draft_variant": override}, config())
        for override in ([], False, "absent", None):
            with self.assertRaises(ValueError):
                policy.validate_override({"inkling_draft_variant": override}, config())
        with self.assertRaises(ValueError):
            policy.validate_override({"inkling_draft_variant": "thinking"}, config(False))

    def test_conditional_acceptance_by_depth_and_variant(self):
        req = request()
        policy.record_round(req, True, True, 5, 3)
        policy.record_round(req, True, True, 3, 1)
        policy.record_round(req, True, False, 3, 0)
        policy.record_round(req, False, False, 3, 2)
        counts = policy.telemetry(req)
        self.assertEqual(counts["thinking"]["thinking"], {
            "drafted": [2, 2, 2, 1, 1],
            "reached": [2, 2, 1, 1, 0],
            "accepted": [2, 1, 1, 0, 0],
            "alpha": [1.0, 0.5, 1.0, 0.0, None],
        })
        self.assertEqual(counts["answer"]["stock"]["accepted"], [1, 1, 0])

    def test_manifest_rejects_unbounded_prefix_and_unknown_width(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = {
                "version": 1, "stock_sha256": "a" * 64,
                "widths": [
                    {"max_load": 2, "thinking": 6, "answer": 4},
                    {"max_load": None, "thinking": 8, "answer": 6},
                ],
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(policy.DraftPolicy.read(path, digest).capture_widths, (4, 6, 8))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                policy.DraftPolicy.read(path, "0" * 64)
            for change in ({"max_load": None}, {"thinking": 5}, {"answer": True}):
                path.write_text(json.dumps(dict(
                    manifest, widths=[dict(manifest["widths"][0], **change),
                                     manifest["widths"][1]]
                )), encoding="utf-8")
                with self.assertRaises(ValueError):
                    policy.DraftPolicy.read(path)


def bench(rounds=100_000):
    rules = config()
    batch = [request() for _ in range(4)]
    for req in batch:
        # A long prompt is scanned once at admission and is not copied per round.
        req.origin_input_ids = [42] * 32768
    policy.select_round(rules, batch, 0)
    def run(i):
        event = policy.THINK_START if i % 2 else policy.THINK_END
        for j, req in enumerate(batch):
            # Keep the output history bounded and feed a boundary on every round.
            req.output_ids[:] = [event if j % 2 else 42, 999]
            req._inkling_draft_phase.seen = 0
        start = time.perf_counter_ns()
        policy.select_round(rules, batch, i % 5)
        return time.perf_counter_ns() - start

    for i in range(4_000):
        run(i)
    samples = sorted(run(i) for i in range(rounds))
    p50, p95, p99 = median(samples), samples[int(rounds * .95)], samples[int(rounds * .99)]
    print(f"c4 CPU selection ({rounds} rounds): p50={p50 / 1000:.2f} us "
          f"p95={p95 / 1000:.2f} us p99={p99 / 1000:.2f} us")
    return p99 < 200_000


if __name__ == "__main__":
    if "--bench" in sys.argv:
        sys.exit(0 if bench() else 1)
    unittest.main()
