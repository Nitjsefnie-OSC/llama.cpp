#!/usr/bin/env python3
"""Mock-service correctness and artifact tests; no GPU or live llama-server access."""

from argparse import Namespace
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import io


SPEC = importlib.util.spec_from_file_location("growth", Path(__file__).with_name("bonsai-server-growth-check.py"))
GROWTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROWTH)


def response(body):
    tokens = list(range(10, 10 + body["n_predict"]))
    probabilities = []
    for token in tokens:
        item = dict(id=token, token="x", bytes=[120], logprob=-0.2)
        item["top_logprobs"] = [dict(id=token+i, token="x", bytes=[120], logprob=-0.2-i) for i in range(5)]
        probabilities.append(item)
    return dict(id_slot=body["id_slot"], stop=True, truncated=False, content="completion", tokens=tokens,
                tokens_predicted=len(tokens), tokens_evaluated=len(body["prompt"]),
                timings=dict(prompt_n=len(body["prompt"]), predicted_n=len(tokens)),
                completion_probabilities=probabilities)


class MockService:
    def __init__(self, busy=False, fail_slot=None, probability_delta=0):
        self.busy, self.fail_slot, self.delta = busy, fail_slot, probability_delta
        self.barrier = threading.Barrier(4)
        self.completions = []
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def __call__(self, url, route, body, timeout):
        if route == "/slots":
            return [dict(id=i, is_processing=self.busy) for i in range(4)]
        if route == "/props":
            return dict(total_slots=4, default_generation_settings=dict(n_ctx=188416),
                        model_path="model.gguf", model_alias="bonsai")
        if route == "/tokenize":
            return dict(tokens=list(range(20000)))
        if route != "/completion":
            raise AssertionError(route)
        with self.lock:
            self.completions.append(copy.deepcopy(body))
        if len(body["prompt"]) == 128:
            with self.lock:
                self.active += 1
                self.max_active = max(self.active, self.max_active)
            self.barrier.wait(timeout=2)
            with self.lock:
                self.active -= 1
            if body["id_slot"] == self.fail_slot:
                raise RuntimeError("mock HTTP 500: retained server error")
        result = response(body)
        for item in result["completion_probabilities"]:
            item["logprob"] += self.delta
            for top in item["top_logprobs"]:
                top["logprob"] += self.delta
        return result


class GrowthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.args = Namespace(url="http://mock", output=self.path/"baseline.jsonl", reference=None,
                              tokens=2, timeout=2, logprob_atol=1e-4)

    def rows(self):
        return [json.loads(line) for line in self.args.output.read_text().splitlines()]

    def test_growth_order_concurrency_full_artifact_and_reference(self):
        service = MockService()
        GROWTH.run(self.args, service)
        self.assertEqual([len(body["prompt"]) for body in service.completions[:4]], [512, 4096, 16384, 512])
        self.assertEqual({body["id_slot"] for body in service.completions[4:]}, set(range(4)))
        self.assertEqual(service.max_active, 4)
        rows = self.rows()
        self.assertEqual(len([row for row in rows if row["kind"] == "request"]), 8)
        self.assertEqual(len([row for row in rows if row["kind"] == "result"]), 8)
        requests = [row for row in rows if row["kind"] in ("request", "http_request")]
        responses = [row for row in rows if row["kind"] in ("result", "http_response")]
        self.assertEqual(len(requests), 16)
        self.assertEqual(len({row["request_id"] for row in requests}), len(requests))
        self.assertEqual({row["request_id"] for row in requests}, {row["request_id"] for row in responses})
        self.assertEqual(rows[-1]["status"], "passed")
        self.args.reference = self.args.output
        self.args.output = self.path/"compared.jsonl"
        GROWTH.run(self.args, MockService(probability_delta=0.00005))
        self.assertTrue(self.rows()[-1]["compared"])

    def test_probability_difference_preserves_failed_response(self):
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.output = self.path/"mismatch.jsonl"
        with self.assertRaisesRegex(ValueError, "Logprob differs"):
            GROWTH.run(self.args, MockService(probability_delta=0.001))
        rows = self.rows()
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(len([row for row in rows if row["kind"] == "result"]), 1)

    def test_busy_and_existing_output_never_launch(self):
        service = MockService(busy=True)
        with self.assertRaisesRegex(ValueError, "busy"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)
        self.assertEqual(self.rows()[-1]["status"], "failed")
        original = self.args.output.read_bytes()
        with self.assertRaises(FileExistsError):
            GROWTH.run(self.args, MockService())
        self.assertEqual(self.args.output.read_bytes(), original)

    def test_concurrent_failure_collects_other_results(self):
        service = MockService(fail_slot=2)
        with self.assertRaisesRegex(ValueError, "Concurrent cases failed"):
            GROWTH.run(self.args, service)
        rows = self.rows()
        self.assertEqual(len(service.completions), 8)
        self.assertEqual(len([row for row in rows if row["kind"] == "result"]), 7)
        errors = [row for row in rows if row["kind"] == "error"]
        self.assertEqual(errors[0]["case"], "concurrent-slot-2")
        self.assertIn("retained server error", errors[0]["error"])
        http_error = next(row for row in rows if row["kind"] == "http_error")
        sent = next(row for row in rows if row.get("request_id") == http_error["request_id"] and row["kind"] == "request")
        self.assertEqual(sent["request"]["id_slot"], 2)
        self.assertEqual(http_error["route"], "/completion")
        self.assertIn("retained server error", http_error["error"])

    def test_late_busy_slots_response_retained(self):
        service = MockService()
        slot_calls = 0

        def late_busy(url, route, body, timeout):
            nonlocal slot_calls
            result = service(url, route, body, timeout)
            if route == "/slots":
                slot_calls += 1
                if slot_calls == 2:
                    result[2]["is_processing"] = True
                    result[2]["diagnostic"] = "external request arrived"
            return result

        with self.assertRaisesRegex(ValueError, "busy"):
            GROWTH.run(self.args, late_busy)
        rows = self.rows()
        slots = [row for row in rows if row["kind"] == "http_response" and row["route"] == "/slots"]
        self.assertEqual(len(slots), 2)
        self.assertFalse(slots[0]["response"][2]["is_processing"])
        self.assertTrue(slots[1]["response"][2]["is_processing"])
        self.assertEqual(slots[1]["response"][2]["diagnostic"], "external request arrived")
        sent = next(row for row in rows if row.get("request_id") == slots[1]["request_id"] and row["kind"] == "http_request")
        self.assertEqual(sent["route"], "/slots")
        self.assertIsNone(sent["request"])
        self.assertFalse(service.completions)
        self.assertEqual(rows[-1]["status"], "failed")

    def test_invalid_response_shapes(self):
        body = GROWTH.make_cases(list(range(20000)), 2)[0][1]
        good = response(body)
        for change in (dict(tokens=[]), dict(content=""), dict(tokens_evaluated=1), dict(truncated=True),
                       dict(id_slot=3), dict(completion_probabilities=[])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                GROWTH.validate_response(body, dict(good, **change))
        for field, value in (("logprob", math.nan), ("logprob", 1), ("id", 999),
                             ("bytes", [256]), ("top_logprobs", [])):
            bad = copy.deepcopy(good)
            bad["completion_probabilities"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                GROWTH.validate_response(body, bad)

    def test_reference_exact_requests_tokens_and_top_ids(self):
        body = GROWTH.make_cases(list(range(20000)), 2)[0][1]
        good = response(body)
        ref = dict(request=body, response=good)
        with self.assertRaisesRegex(ValueError, "Request differs"):
            GROWTH.compare_response(dict(body, seed=9), good, ref, 1e-4)
        bad = copy.deepcopy(good)
        bad["tokens"][0] += 1
        with self.assertRaisesRegex(ValueError, "Output tokens differ"):
            GROWTH.compare_response(body, bad, ref, 1e-4)
        bad = copy.deepcopy(good)
        bad["completion_probabilities"][0]["top_logprobs"][0]["id"] += 99
        with self.assertRaisesRegex(ValueError, "identity differs"):
            GROWTH.compare_response(body, bad, ref, 1e-4)

    def test_incomplete_reference_rejected(self):
        self.args.output.write_text('{"kind":"summary","status":"failed"}\n')
        with self.assertRaisesRegex(ValueError, "completed passing"):
            GROWTH.load_reference(self.args.output)

    def test_reference_without_http_audit_fields_still_loads(self):
        cases = GROWTH.make_cases(list(range(20000)), 2)
        rows = [{"kind": "metadata", "props": {"model_alias": "old artifact"}}]
        rows.extend(dict(kind="result", case=name, request=body, response=response(body)) for name, body in cases)
        rows.append(dict(kind="summary", status="passed", cases=8))
        self.args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        metadata, results = GROWTH.load_reference(self.args.output)
        self.assertEqual(metadata["props"]["model_alias"], "old artifact")
        self.assertEqual(set(results), {name for name, _ in cases})

    def test_repeat_detects_growth_state_contamination(self):
        service = MockService()

        def contaminated(url, route, body, timeout):
            result = service(url, route, body, timeout)
            if route == "/completion" and len(service.completions) == 4:
                result["content"] = "changed after growth"
            return result

        with self.assertRaisesRegex(ValueError, "content differs"):
            GROWTH.run(self.args, contaminated)
        self.assertEqual(len(service.completions), 4)
        self.assertEqual(self.rows()[-1]["status"], "failed")

    def test_http_timeout_and_error_body(self):
        error = urllib.error.HTTPError("http://mock", 500, "failed", {}, io.BytesIO(b'diagnostic detail'))
        with patch.object(GROWTH.urllib.request, "urlopen", side_effect=error) as open_url:
            with self.assertRaisesRegex(RuntimeError, "HTTP 500 /completion: diagnostic detail"):
                GROWTH.request_json("http://mock", "/completion", {"seed": 1234}, 17)
            self.assertEqual(open_url.call_args.kwargs["timeout"], 17)
        with patch.object(GROWTH.urllib.request, "urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(TimeoutError):
                GROWTH.request_json("http://mock", "/slots", None, 17)

    def test_bad_corpus_and_slot_shapes(self):
        for slots in ([], [dict(id=0, is_processing=False)]*4):
            with self.assertRaises(ValueError):
                GROWTH.idle_slots(slots)
        with self.assertRaisesRegex(ValueError, "too short"):
            GROWTH.make_cases([1, 2], 2)


if __name__ == "__main__":
    unittest.main()
