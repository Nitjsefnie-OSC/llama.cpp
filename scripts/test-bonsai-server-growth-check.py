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
                              tokens=2, timeout=2, logprob_atol=1e-4, concurrent_only=False, atomic_concurrent=False,
                              growth_only=False)

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

    def test_growth_only_order_and_complete_audit(self):
        self.args.growth_only = True
        service = MockService()
        with patch.object(GROWTH, "ThreadPoolExecutor", side_effect=AssertionError("concurrent work started")):
            GROWTH.run(self.args, service)
        self.assertEqual([len(body["prompt"]) for body in service.completions], [512, 4096, 16384, 512])
        self.assertEqual({body["id_slot"] for body in service.completions}, {0})
        rows = self.rows()
        self.assertEqual(rows[-1], dict(kind="summary", status="passed", mode="growth-only", cases=4, compared=False))
        self.assertEqual(next(row for row in rows if row["kind"] == "metadata")["mode"], "growth-only")
        requests = [row for row in rows if row["kind"] in ("request", "http_request")]
        responses = [row for row in rows if row["kind"] in ("result", "http_response")]
        self.assertEqual(len(requests), 12)
        self.assertEqual({row["request_id"] for row in requests}, {row["request_id"] for row in responses})

    def test_growth_only_accepts_full_and_growth_references(self):
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.growth_only = True
        for name in ("growth-from-full.jsonl", "growth-from-growth.jsonl"):
            self.args.output = self.path/name
            service = MockService(probability_delta=0.00005)
            GROWTH.run(self.args, service)
            self.assertEqual(len(service.completions), 4)
            self.assertTrue(self.rows()[-1]["compared"])
            self.args.reference = self.args.output
        self.args.growth_only = False
        self.args.output = self.path/"full-from-growth.jsonl"
        with self.assertRaisesRegex(ValueError, "result count"):
            GROWTH.run(self.args, MockService())
        self.assertFalse(self.args.output.exists())
        with self.assertRaisesRegex(ValueError, "mode/result count"):
            GROWTH.load_reference(self.args.reference, concurrent_only=True)

    def test_growth_only_rejects_incomplete_or_wrong_references(self):
        GROWTH.run(self.args, MockService())
        full = self.rows()
        partial = [row for row in full if row["kind"] != "result" or row["case"].startswith("growth-")]
        wrong_mode = copy.deepcopy(partial)
        wrong_mode[-1]["cases"] = 4
        three = [row for row in partial if row.get("case") != "growth-3-512"]
        three[-1] = dict(kind="summary", status="passed", cases=3, mode="growth-only")
        for index, rows in enumerate((partial, wrong_mode, three)):
            path = self.path/f"invalid-growth-{index}.jsonl"
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            with self.subTest(index=index), self.assertRaises(ValueError):
                GROWTH.load_reference(path, growth_only=True)
        self.args.growth_only = True
        self.args.output = self.path/"growth-valid.jsonl"
        GROWTH.run(self.args, MockService())
        valid_rows = self.rows()
        for index, mutation in enumerate((
                lambda rows: next(r for r in rows if r["kind"] == "result").update(case="unexpected-growth"),
                lambda rows: next(r for r in rows if r["kind"] == "metadata").pop("mode"))):
            rows = copy.deepcopy(valid_rows)
            mutation(rows)
            reference = self.path/f"bad-growth-mode-name-{index}.jsonl"
            reference.write_text("".join(json.dumps(row)+"\n" for row in rows))
            self.args.reference = reference
            self.args.output = self.path/f"rejected-growth-{index}.jsonl"
            service = MockService()
            with self.subTest(index=index), self.assertRaises(ValueError):
                GROWTH.run(self.args, service)
            self.assertFalse(service.completions)

    def test_growth_only_conflicts_before_requests(self):
        self.args.growth_only = True
        for concurrent, atomic in ((True, False), (False, True), (True, True)):
            self.args.concurrent_only, self.args.atomic_concurrent = concurrent, atomic
            service = unittest.mock.Mock(side_effect=AssertionError("HTTP attempted"))
            with self.subTest(concurrent=concurrent, atomic=atomic), self.assertRaisesRegex(ValueError, "conflicts"):
                GROWTH.run(self.args, service)
            service.assert_not_called()
        for flag in ("--concurrent-only", "--atomic-concurrent"):
            argv = ["checker", "--growth-only", flag, "--output", str(self.args.output)]
            with patch("sys.argv", argv), patch.object(GROWTH, "run") as run, patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    GROWTH.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
        with patch("sys.argv", ["checker", "--growth-only", "--output", str(self.args.output)]), patch.object(GROWTH, "run") as run:
            GROWTH.main()
            parsed = run.call_args.args[0]
            self.assertTrue(parsed.growth_only)
            self.assertFalse(parsed.concurrent_only or parsed.atomic_concurrent)
        self.assertFalse(self.args.output.exists())

    def test_growth_only_probability_failure_retains_response(self):
        self.args.growth_only = True
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.output = self.path/"growth-mismatch.jsonl"
        with self.assertRaisesRegex(ValueError, "Logprob differs"):
            GROWTH.run(self.args, MockService(probability_delta=0.001))
        rows = self.rows()
        result = next(row for row in rows if row["kind"] == "result")
        self.assertEqual(result["case"], "growth-0-512")
        self.assertAlmostEqual(result["response"]["completion_probabilities"][0]["logprob"], -0.199)
        self.assertTrue(any(row["kind"] == "error" and row["case"] == result["case"] for row in rows))
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["mode"], "growth-only")

    def test_growth_only_requires_exact_reference_requests(self):
        self.args.growth_only = True
        GROWTH.run(self.args, MockService())
        rows = self.rows()
        next(row for row in rows if row["kind"] == "result")["request"]["seed"] = 99
        self.args.reference = self.path/"growth-wrong-request.jsonl"
        self.args.reference.write_text("".join(json.dumps(row)+"\n" for row in rows))
        self.args.output = self.path/"growth-request-rejected.jsonl"
        service = MockService()
        with self.assertRaisesRegex(ValueError, "Reference request differs"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)

    def test_concurrent_only_accepts_full_and_four_case_references(self):
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.concurrent_only = True
        for filename in ("four-from-full.jsonl", "four-from-four.jsonl"):
            self.args.output = self.path/filename
            service = MockService()
            GROWTH.run(self.args, service)
            self.assertEqual(len(service.completions), 4)
            self.assertEqual({len(body["prompt"]) for body in service.completions}, {128})
            self.assertEqual(service.max_active, 4)
            rows = self.rows()
            self.assertEqual(rows[-1]["cases"], 4)
            self.assertEqual(rows[-1]["mode"], "concurrent-only")
            self.assertTrue(rows[-1]["compared"])
            self.assertEqual(next(row for row in rows if row["kind"] == "metadata")["mode"], "concurrent-only")
            self.assertEqual(len([row for row in rows if row["kind"] == "result"]), 4)
            self.args.reference = self.args.output
        self.args.concurrent_only = False
        self.args.output = self.path/"full-from-four.jsonl"
        service = MockService()
        with self.assertRaisesRegex(ValueError, "result count"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)
        self.assertFalse(self.args.output.exists())

    def test_concurrent_only_retains_all_mismatched_responses(self):
        self.args.concurrent_only = True
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.output = self.path/"four-mismatch.jsonl"
        with self.assertRaisesRegex(ValueError, "Concurrent cases failed"):
            GROWTH.run(self.args, MockService(probability_delta=0.001))
        rows = self.rows()
        self.assertEqual(len([row for row in rows if row["kind"] == "result"]), 4)
        self.assertEqual(len([row for row in rows if row["kind"] == "error"]), 4)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["mode"], "concurrent-only")

    def test_concurrent_reference_requires_exact_selected_cases(self):
        self.args.concurrent_only = True
        GROWTH.run(self.args, MockService())
        rows = self.rows()
        next(row for row in rows if row["kind"] == "result")["case"] = "unexpected-slot"
        reference = self.path/"bad-reference.jsonl"
        reference.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.args.reference = reference
        self.args.output = self.path/"bad-reference-run.jsonl"
        service = MockService()
        with self.assertRaisesRegex(ValueError, "case names differ"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)

    def test_partial_full_reference_is_not_a_four_case_reference(self):
        GROWTH.run(self.args, MockService())
        rows = [row for row in self.rows() if row["kind"] != "result" or row["case"].startswith("concurrent-")]
        reference = self.path/"partial-full.jsonl"
        reference.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "summary/result count"):
            GROWTH.load_reference(reference, concurrent_only=True)

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


class AtomicService(MockService):
    def __init__(self, slots=(2, 0, 3, 1), delta=0, mutate=None, **kwargs):
        super().__init__(**kwargs)
        self.slots, self.delta, self.mutate = slots, delta, mutate

    def __call__(self, url, route, body, timeout):
        if route != "/completion":
            return super().__call__(url, route, body, timeout)
        assert "id_slot" not in body and body["stream"] is False
        assert len(body["prompt"]) == 4 and all(p == body["prompt"][0] for p in body["prompt"])
        self.completions.append(copy.deepcopy(body))
        results = []
        for index, slot in enumerate(self.slots):
            item = response(dict(body, prompt=body["prompt"][index], id_slot=slot))
            item["index"] = index
            # Distinct per-slot results detect accidental comparison by prompt index.
            item["content"] = f"slot-{slot}"
            for probability in item["completion_probabilities"]:
                probability["logprob"] -= slot * 0.01 + self.delta
                for top in probability["top_logprobs"]:
                    top["logprob"] -= slot * 0.01 + self.delta
            results.append(item)
        if self.mutate:
            self.mutate(results)
        return list(reversed(results))


class AtomicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.args = Namespace(url="http://mock", output=self.path/"atomic.jsonl", reference=None,
                              tokens=2, timeout=2, logprob_atol=1e-4, concurrent_only=True, atomic_concurrent=True,
                              growth_only=False)

    def rows(self):
        return [json.loads(line) for line in self.args.output.read_text().splitlines()]

    def test_bulk_once_and_reference_compares_actual_slots(self):
        service = AtomicService()
        GROWTH.run(self.args, service)
        self.assertEqual(len(service.completions), 1)
        rows = self.rows()
        sent = [r for r in rows if r.get("route") == "/completion" and r["kind"] == "request"]
        results = [r for r in rows if r["kind"] == "result"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(sent[0]["request_id"], results[0]["request_id"])
        self.assertEqual(len(results[0]["response"]), 4)
        self.assertEqual({(r["index"], r["id_slot"]) for r in results[0]["response"]},
                         {(0, 2), (1, 0), (2, 3), (3, 1)})
        self.assertEqual(rows[-1]["cases"], 4)
        self.assertEqual(rows[-1]["mode"], "atomic-concurrent")
        self.assertEqual(next(r for r in rows if r["kind"] == "metadata")["mode"], "atomic-concurrent")
        self.args.reference = self.args.output
        self.args.output = self.path/"remapped.jsonl"
        GROWTH.run(self.args, AtomicService(slots=(0, 1, 2, 3), delta=0.00005))
        self.assertTrue(self.rows()[-1]["compared"])

    def test_malformed_bulk_retained_before_validation(self):
        mutations = [lambda r: r.pop(), lambda r: r[1].update(id_slot=r[0]["id_slot"]),
                     lambda r: r[1].update(index=r[0]["index"]), lambda r: r[1].pop("index"),
                     lambda r: r[1].pop("id_slot"), lambda r: r[1].update(index=True),
                     lambda r: r[1].update(id_slot=4)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.args.output = self.path/f"malformed-{index}.jsonl"
                with self.assertRaises(ValueError):
                    GROWTH.run(self.args, AtomicService(mutate=mutate))
                rows = self.rows()
                self.assertEqual(len([r for r in rows if r["kind"] == "result"]), 1)
                self.assertEqual(rows[-1]["status"], "failed")

    def test_atomic_and_http_references_cannot_mix(self):
        GROWTH.run(self.args, AtomicService())
        atomic_path = self.args.output
        self.args.atomic_concurrent = False
        self.args.reference = atomic_path
        self.args.output = self.path/"http-reject.jsonl"
        with self.assertRaises(ValueError):
            GROWTH.run(self.args, MockService())
        self.args.reference = None
        self.args.output = self.path/"http.jsonl"
        GROWTH.run(self.args, MockService())
        self.args.reference = self.args.output
        self.args.atomic_concurrent = True
        self.args.output = self.path/"atomic-reject.jsonl"
        with self.assertRaises(ValueError):
            GROWTH.run(self.args, AtomicService())

    def test_probability_mismatch_and_exact_bulk_request(self):
        GROWTH.run(self.args, AtomicService())
        self.args.reference = self.args.output
        self.args.output = self.path/"mismatch.jsonl"
        with self.assertRaisesRegex(ValueError, "Logprob differs"):
            GROWTH.run(self.args, AtomicService(delta=0.001))
        self.assertEqual(len([r for r in self.rows() if r["kind"] == "result"]), 1)
        self.assertEqual(self.rows()[-1]["status"], "failed")
        rows = [json.loads(l) for l in self.args.reference.read_text().splitlines()]
        next(r for r in rows if r["kind"] == "result")["request"]["seed"] = 999
        self.args.reference = self.path/"changed-request.jsonl"
        self.args.reference.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.args.output = self.path/"request-mismatch.jsonl"
        service = AtomicService()
        with self.assertRaisesRegex(ValueError, "request differs"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)

    def test_atomic_requires_explicit_mode_and_concurrent_only(self):
        self.args.concurrent_only = False
        with self.assertRaisesRegex(ValueError, "requires --concurrent-only"):
            GROWTH.run(self.args, AtomicService())
        self.args.concurrent_only = True
        GROWTH.run(self.args, AtomicService())
        rows = self.rows()
        next(r for r in rows if r["kind"] == "metadata").pop("mode")
        reference = self.path/"missing-mode.jsonl"
        reference.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaises(ValueError):
            GROWTH.load_reference(reference, concurrent_only=True, atomic_concurrent=True)

    def test_busy_and_configuration_checks_before_bulk(self):
        service = AtomicService(busy=True)
        with self.assertRaisesRegex(ValueError, "busy"):
            GROWTH.run(self.args, service)
        self.assertFalse(service.completions)
        self.args.output = self.path/"bad-context.jsonl"
        service = AtomicService()
        def wrong_context(url, route, body, timeout):
            result = service(url, route, body, timeout)
            if route == "/props": result["default_generation_settings"]["n_ctx"] = 4096
            return result
        with self.assertRaisesRegex(ValueError, "unchanged context"):
            GROWTH.run(self.args, wrong_context)
        self.assertFalse(service.completions)


if __name__ == "__main__":
    unittest.main()
