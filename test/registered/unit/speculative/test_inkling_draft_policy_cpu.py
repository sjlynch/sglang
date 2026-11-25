"""Standalone CPU checks and c4 routing micro-benchmark (no SGLang imports)."""

import importlib.util
from dataclasses import replace
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
    def test_overlap_factorial_modes_and_queue_pressure(self):
        base = config()
        reqs = [request((policy.THINK_START,)) for _ in range(2)]
        for mode, expected in {
            "control": (0, 4), "width": (0, 6), "head": (1, 4), "both": (1, 6),
        }.items():
            rules = replace(base, batch_mode=mode)
            self.assertEqual(policy.select_batch_round(rules, reqs, 0)[:2], expected)
        rules = replace(base, batch_mode="both")
        self.assertEqual(policy.select_batch_round(rules, reqs, 1)[:2], (1, 8))
        self.assertEqual(policy.select_batch_round(rules, [reqs[0], request()], 0)[:2], (0, 4))

    def test_overlap_uses_ready_previous_stride_without_advancing_phase(self):
        old = request((policy.THINK_START,))
        other = request()
        mode = SimpleNamespace(is_decode=lambda: True)
        batch = SimpleNamespace(reqs=[old, other], forward_mode=mode)
        ready = False

        def cpu(values):
            def tolist():
                self.assertTrue(ready)
                return values
            return SimpleNamespace(is_cpu=True, tolist=tolist)

        result = SimpleNamespace(
            accept_lens=cpu([1, 2]),
            next_token_ids=cpu([policy.THINK_END, policy.THINK_START, 0, 0, 0, 0,
                                1, policy.THINK_START, policy.THINK_END, 0, 0, 0]),
            copy_done=SimpleNamespace(query=lambda: ready),
            speculative_output_stride=6,
            speculative_num_draft_tokens=8,
        )
        rules = replace(config(), batch_mode="both")
        self.assertIsNone(policy.completed_batch_channels(batch, result))
        self.assertEqual(policy.select_batch_round(rules, [old], 0), (1, 6, (True,)))
        ready = True
        channels = policy.completed_batch_channels(batch, result)
        self.assertEqual(channels, {id(old): "answer", id(other): "thinking"})
        self.assertEqual(policy.select_batch_round(rules, [old], 0, channels), (0, 4, (False,)))
        self.assertEqual(old._inkling_draft_phase.seen, 0)
        self.assertEqual(old._inkling_draft_phase.channel, "thinking")
        old.output_ids.append(policy.THINK_END)
        self.assertEqual(policy.select_batch_round(rules, [old], 0), (0, 4, (False,)))
        result.next_token_ids.is_cpu = False
        self.assertIsNone(policy.completed_batch_channels(batch, result))

    def test_context_load_table_and_structured_answer(self):
        rules = policy.DraftPolicy(
            (
                policy.WidthTier(2, 6, 8, 8192, 8),
                policy.WidthTier(4, 5, 8, 8192, 8),
                policy.WidthTier(None, 4, 4, 131072, 8, 8, False),
            ),
            "a" * 64, anchor_suffix_width8=True,
        )
        self.assertEqual(rules.capture_widths, (4, 5, 6, 8))
        for load, tokens, expected in (
            (1, 8191, 6), (2, 8192, 8), (4, 8191, 5), (4, 8192, 8),
            (8, 131071, 4), (8, 131072, 8),
        ):
            req = request([0] * (tokens - 1) + [policy.THINK_START])
            self.assertEqual(policy.select_round(rules, [req], load - 1)[0], expected)
        repeats = request((1, 2, 3, 4, 99, policy.THINK_START), (1, 2, 3, 4))
        self.assertEqual(policy.select_round(rules, [repeats], 3)[0], 8)
        self.assertEqual(policy.select_round(rules, [repeats], 7)[0], 4)
        self.assertEqual(policy.select_round(rules, [request()], 3)[0], 8)
        self.assertEqual(policy.select_round(rules, [request()], 7)[0], 4)
        self.assertEqual(policy.select_round(rules, [request((200049, 7))], 7)[0], 8)
        schema_req = request()
        schema_req.grammar = object()
        self.assertEqual(policy.select_round(rules, [schema_req], 7)[0], 8)

    def test_suffix_uses_prompt_and_commits_only(self):
        rules = policy.DraftPolicy(
            (policy.WidthTier(None, 4, 4),), "a" * 64, anchor_suffix_width8=True
        )
        self.assertEqual(rules.capture_widths, (4, 8))
        req = request((1, 2, 3, 4, 99, policy.THINK_START))
        req.output_ids = [1, 2, 3]
        self.assertEqual(policy.select_round(rules, [req], 0)[0], 4)
        req.output_ids.append(4)
        observations = []
        self.assertEqual(policy.select_round(rules, [req], 0, observations)[0], 8)
        self.assertTrue(observations[0][0])
        self.assertEqual(policy.select_round(rules, [req, request()], 0)[0], 4)
        req.output_ids += [policy.THINK_END, 1, 2, 3, 4]
        observations.clear()
        self.assertEqual(policy.select_round(rules, [req], 0, observations)[0], 4)
        self.assertEqual(observations, [None])

    def test_suffix_excludes_current_occurrence_and_handles_rewind(self):
        req = request(output=(1, 2, 3, 4))
        matcher = policy.AnchorSuffixMatch()
        self.assertFalse(matcher.update(req))
        req.output_ids += [5, 1, 2, 3, 4]
        self.assertTrue(matcher.update(req))
        req.output_ids = [1, 2, 3, 4]
        self.assertFalse(matcher.update(req))
        req.output_ids = [4, 4, 4, 4, 4]
        matcher = policy.AnchorSuffixMatch()
        self.assertTrue(matcher.update(req))

    def test_suffix_telemetry_reports_matches_held_to_width_four(self):
        req = request()
        policy.record_round(req, True, False, 3, 0, (True, 7))
        policy.record_round(req, True, False, 7, 6, (True, 11))
        policy.record_round(req, True, False, 3, 2, (False, 5))
        summary = policy.telemetry(req)["thinking"]["stock"]["anchor_suffix"]
        self.assertEqual(summary, {"checks": 3, "matches": 2,
                                   "matches_at_width8": 1, "cpu_ns": 23})

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
            for change in ({"max_load": None}, {"thinking": 7}, {"answer": True},
                           {"thinking_long": 8}, {"long_context_tokens": 0, "thinking_long": 8}):
                path.write_text(json.dumps(dict(
                    manifest, widths=[dict(manifest["widths"][0], **change),
                                     manifest["widths"][1]]
                )), encoding="utf-8")
                with self.assertRaises(ValueError):
                    policy.DraftPolicy.read(path)
            for flag in ("true", 1):
                path.write_text(json.dumps(dict(manifest, anchor_suffix_width8=flag)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    policy.DraftPolicy.read(path)


    def test_pin_fixes_one_captured_graph_and_keeps_phases(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = {
                "version": 1, "stock_sha256": "a" * 64, "batch_mode": "width",
                "widths": [{"max_load": None, "thinking": 8, "answer": 4}],
                "pin": {"head": 0, "width": 4},
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            rules = policy.DraftPolicy.read(path)
            self.assertEqual((rules.capture_widths, rules.pin), ((4, 8), (0, 4)))
            thinking = request((policy.THINK_START,))
            self.assertEqual(policy.select_batch_round(rules, [thinking], 0), (0, 4, (True,)))
            for pin, mode in (({"head": 1, "width": 4}, "width"), ({"head": 0, "width": 5}, "width"),
                              ({"head": 0, "width": 4}, None), ({"head": 0}, "width")):
                path.write_text(json.dumps(dict(manifest, pin=pin, batch_mode=mode)), encoding="utf-8")
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


def bench_suffix(rounds=20_000):
    passed = True
    for offered in (1, 4, 8):
        active = min(offered, 4)
        rules = policy.DraftPolicy(
            (policy.WidthTier(None, 4, 4),), "a" * 64, anchor_suffix_width8=True
        )
        prompt = list(range(32_768)) + [policy.THINK_START]
        batch = [request(prompt) for _ in range(active)]
        stats = []
        policy.select_round(rules, batch, offered - active, stats)
        cold_us = sum(elapsed for _, elapsed in stats) / 1000
        samples = []
        for i in range(rounds):
            for req in batch:
                req.output_ids.extend([i % 256, 1, 2, 3])
            stats.clear()
            start = time.perf_counter_ns()
            policy.select_round(rules, batch, offered - active, stats)
            samples.append(time.perf_counter_ns() - start)
        samples.sort()
        p99 = samples[int(rounds * .99)] / 1000
        print(f"c{offered} ({active} active): first 32K indexing={cold_us:.1f} us, "
              f"steady p50={median(samples) / 1000:.2f} us p99={p99:.2f} us ({rounds} rounds)")
        passed &= p99 < 200
    return passed


def bench_overlap(rounds=100_000):
    rules = replace(config(), batch_mode="both")
    passed = True
    widths = {width: index for index, width in enumerate(rules.capture_widths)}
    graph_table = tuple(object() for _ in range(rules.capture_heads * len(widths)))
    for active in (1, 4):
        reqs = [request((policy.THINK_START,)) for _ in range(active)]
        policy.select_batch_round(rules, reqs, 0)
        batch = SimpleNamespace(reqs=reqs, forward_mode=SimpleNamespace(is_decode=lambda: True))
        result = SimpleNamespace(
            accept_lens=SimpleNamespace(is_cpu=True, tolist=lambda: [4] * active),
            next_token_ids=SimpleNamespace(
                is_cpu=True, tolist=lambda: [42, policy.THINK_END, 17, 99, 0, 0, 0, 0] * active,
            ),
            copy_done=SimpleNamespace(query=lambda: True),
            speculative_output_stride=8,
        )
        samples = []
        for i in range(rounds + 4_000):
            start = time.perf_counter_ns()
            channels = policy.completed_batch_channels(batch, result)
            head, width, _ = policy.select_batch_round(rules, reqs, i % 5, channels if i % 2 else None)
            graph_table[head * len(widths) + widths[width]]
            elapsed = time.perf_counter_ns() - start
            if i >= 4_000:
                samples.append(elapsed)
        samples.sort()
        p95, p99 = samples[int(rounds * .95)] / 1000, samples[int(rounds * .99)] / 1000
        print(f"c{active} overlap dispatch ({rounds} rounds, simulated ready copy): "
              f"p50={median(samples) / 1000:.2f} us p95={p95:.2f} us p99={p99:.2f} us")
        passed &= p99 <= 100
    return passed


if __name__ == "__main__":
    if "--bench" in sys.argv:
        sys.exit(0 if bench() else 1)
    if "--bench-suffix" in sys.argv:
        sys.exit(0 if bench_suffix() else 1)
    if "--bench-overlap" in sys.argv:
        sys.exit(0 if bench_overlap() else 1)
    unittest.main()
