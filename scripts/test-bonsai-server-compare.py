#!/usr/bin/env python3
"""CPU-only comparator regression tests; no service or CUDA calls."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("compare", Path(__file__).with_name("bonsai-server-compare.py"))
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def artifact(tag, ingest=(100, 100, 100), decode=(100, 100, 100), wall=(2, 2, 2), sizes=(4, 8), tokens=3):
    rows = [{"kind": "metadata", "sizes": list(sizes), "tokens": tokens, "reps": len(ingest), "tag": tag,
             "props": {"model_path": "model.gguf", "model_alias": "bonsai", "total_slots": 4,
                       "default_generation_settings": {"n_ctx": 188416}}}]
    for rep in range(len(ingest) + 1):
        for size in sizes:
            body = {"prompt": list(range(size)), "n_predict": tokens, "temperature": 0, "seed": 1234,
                    "ignore_eos": True, "cache_prompt": False, "id_slot": 0, "return_tokens": True}
            content = f"answer-{size}-{rep}"
            pp, tg = (999999, 999999) if rep == 0 else (ingest[rep - 1], decode[rep - 1])
            if tokens == 1:
                tg = 0
            response = {"content": content, "tokens": list(range(100, 100 + tokens)), "id_slot": 0,
                        "stop": True, "truncated": False, "tokens_predicted": tokens, "tokens_evaluated": size,
                        "timings": {"cache_n": 0, "prompt_n": size, "predicted_n": tokens,
                                    "prompt_ms": 1000 * size / pp, "prompt_per_second": pp,
                                    "predicted_ms": 1000 * (tokens - 1) / tg if tg else 0, "predicted_per_second": tg}}
            rows.append({"kind": "trial", "repetition": rep, "warmup": rep == 0, "prompt_tokens": size,
                         "wall_seconds": 999999 if rep == 0 else wall[rep - 1], "request": body, "response": response,
                         "content_sha256": hashlib.sha256(content.encode()).hexdigest()})
    rows.extend({"kind": "summary", "prompt_tokens": size, "ingest_median": 123456,
                 "decode_median": 123456, "wall_median": 123456} for size in sizes)
    return rows


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="bonsai-compare-tests-"))
        print(f"Test artifacts: {cls.directory}")

    def setUp(self):
        self.root = self.directory / self._testMethodName
        self.root.mkdir()

    def write(self, name, rows):
        path = self.root / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def pair(self, **kwargs):
        return self.write("a", artifact("a")), self.write("b", artifact("b", **kwargs))

    def test_abba_pooling_excludes_warmup_and_ignores_summaries(self):
        a1 = self.write("a1", artifact("a1", ingest=(10, 20, 30), decode=(10, 20, 30), wall=(1, 2, 3)))
        a2 = self.write("a2", artifact("a2", ingest=(40, 50, 600), decode=(40, 50, 600), wall=(4, 5, 60)))
        b1 = self.write("b1", artifact("b1", ingest=(40, 50, 60), decode=(40, 50, 60), wall=(4, 5, 6)))
        b2 = self.write("b2", artifact("b2", ingest=(70, 80, 900), decode=(70, 80, 900), wall=(7, 8, 90)))
        result = COMPARE.compare([a1, a2], [b1, b2], 3, 2)
        self.assertEqual(result["status"], "passed")
        for p in result["prompts"]:
            self.assertEqual(p["baseline"], {"samples": 6, "ingest_tps": 35, "decode_tps": 35, "wall_seconds": 3.5})
            self.assertEqual(p["candidate"], {"samples": 6, "ingest_tps": 65, "decode_tps": 65, "wall_seconds": 6.5})
            self.assertAlmostEqual(p["change_pct"]["decode_tps"], 100 * 30 / 35)

    def test_gate_exact_boundary_and_just_outside(self):
        a, b = self.pair(ingest=(98, 98, 98), decode=(103, 103, 103))
        self.assertEqual(COMPARE.compare([a], [b], 3, 2)["status"], "passed")
        self.assertEqual(COMPARE.compare([a], [b], 3.000001, 2)["status"], "failed")
        self.assertEqual(COMPARE.compare([a], [b], 3, 1.999999)["status"], "failed")

    def test_gate_checks_each_prompt(self):
        a, b = self.pair(decode=(110, 110, 110))
        data = artifact("bad-second", decode=(110, 110, 110))
        for row in data:
            if row["kind"] == "trial" and row["prompt_tokens"] == 8:
                row["response"]["timings"].update(predicted_ms=2000/99, predicted_per_second=99)
        bad = self.write("bad-second", data)
        result = COMPARE.compare([a], [bad], 3, 2)
        self.assertEqual(result["correctness"], "passed")
        self.assertEqual([p["gate_passed"] for p in result["prompts"]], [True, False])

    def test_malformed_and_truncated_json(self):
        for index, text in enumerate(("", "{}\n", '{"kind":', '[]\n', '{"kind":"metadata","kind":"metadata"}\n')):
            path = self.root / f"bad-{index}.jsonl"
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.load_run(path)

    def test_coverage_rejections(self):
        original = artifact("a")
        cases = []
        cases.append(original[:-1])
        cases.append(original[:2] + original[3:])
        cases.append(original + [original[-1]])
        duplicated = copy.deepcopy(original); duplicated[2] = copy.deepcopy(duplicated[1]); cases.append(duplicated)
        wrong = copy.deepcopy(original); wrong[1]["warmup"] = False; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong[3]["repetition"] = True; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong[-1] = copy.deepcopy(wrong[-2]); cases.append(wrong)
        wrong = copy.deepcopy(original); wrong[0]["sizes"] = [4, 4]; cases.append(wrong)
        for i, rows in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.load_run(self.write(str(i), rows))

    def test_trial_validation(self):
        changes = [lambda r: r.update(wall_seconds=0), lambda r: r.update(wall_seconds=float("nan")),
                   lambda r: r["request"].update(n_predict=True), lambda r: r["request"].update(prompt=[True] * 4),
                   lambda r: r["response"].update(tokens=[1]), lambda r: r["response"].update(stop=False),
                   lambda r: r["response"].update(truncated=True), lambda r: r.update(content_sha256="wrong"),
                   lambda r: r["response"]["timings"].update(cache_n=1),
                   lambda r: r["response"]["timings"].update(prompt_n=0),
                   lambda r: r["response"]["timings"].update(predicted_ms=-1),
                   lambda r: r["response"]["timings"].update(predicted_per_second=123),
                   lambda r: r["response"]["timings"].update(prompt_ms=float("inf"))]
        for i, change in enumerate(changes):
            data = artifact(str(i)); change(data[1])
            with self.subTest(i=i), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.load_run(self.write(str(i), data))

    def test_request_and_configuration_mismatch(self):
        a, _ = self.pair()
        changes = [lambda d: d[1]["request"].update(seed=999),
                   lambda d: d[1]["request"]["prompt"].__setitem__(0, 999),
                   lambda d: d[0]["props"].update(model_alias="other")]
        for i, change in enumerate(changes):
            data = artifact(str(i)); change(data)
            with self.subTest(i=i), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.compare([a], [self.write(str(i), data)])
        for i, kwargs in enumerate(({"sizes": (4, 9)}, {"tokens": 4}, {"ingest": (100,), "decode": (100,), "wall": (2,)})):
            with self.subTest(kwargs=kwargs), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.compare([a], [self.write(f"config-{i}", artifact(str(i), **kwargs))])

    def test_exact_output_tokens_and_content_including_warmup(self):
        a, _ = self.pair()
        for field, row_index in (("tokens", 1), ("tokens", 3), ("content", 1), ("content", 3)):
            data = artifact("changed")
            row = data[row_index]
            row["response"][field] = [901, 902, 903] if field == "tokens" else "different answer"
            row["content_sha256"] = hashlib.sha256(row["response"]["content"].encode()).hexdigest()
            with self.subTest(field=field, row=row_index), self.assertRaises(COMPARE.OutputMismatch):
                COMPARE.compare([a], [self.write(f"{field}-{row_index}", data)])

    def test_duplicate_paths_and_copied_artifacts(self):
        a, b = self.pair()
        with self.assertRaises(COMPARE.InvalidRun): COMPARE.compare([a, a], [b])
        with self.assertRaises(COMPARE.InvalidRun): COMPARE.compare([a], [a])
        copied = self.write("copied", artifact("a"))
        with self.assertRaises(COMPARE.InvalidRun): COMPARE.compare([a], [copied])

    def test_threshold_validation_and_single_token(self):
        a, b = self.pair()
        for low, high in ((3, None), (None, 2), (-1, 2), (float("nan"), 2), (3, float("inf"))):
            with self.subTest(low=low, high=high), self.assertRaises(COMPARE.InvalidRun):
                COMPARE.compare([a], [b], low, high)
        a1 = self.write("one-a", artifact("one-a", tokens=1))
        b1 = self.write("one-b", artifact("one-b", tokens=1))
        self.assertEqual(COMPARE.compare([a1], [b1])["status"], "passed")
        self.assertIsNone(COMPARE.compare([a1], [b1])["prompts"][0]["change_pct"]["decode_tps"])
        self.assertEqual(COMPARE.compare([a1], [b1], 0, 0)["status"], "failed")

    def test_cli_exclusive_output_and_retained_errors(self):
        a, b = self.pair()
        output = self.root / "result.json"
        args = ["--baseline", str(a), "--candidate", str(b), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(COMPARE.main(args), 0)
            original = output.read_bytes()
            self.assertEqual(COMPARE.main(args), 2)
            self.assertEqual(output.read_bytes(), original)
            for name, extra in (("gate", ["--min-decode-gain-pct", "3", "--max-ingest-regression-pct", "2"]),
                                ("arguments", ["--min-decode-gain-pct", "3"])):
                args[-1] = str(self.root / (name + ".json"))
                self.assertEqual(COMPARE.main(args + extra), 1)
                self.assertIn("error", json.loads(Path(args[-1]).read_text()))
            missing = self.root / "missing.jsonl"
            args[1] = str(missing); args[-1] = str(self.root / "malformed.json")
            self.assertEqual(COMPARE.main(args), 1)
            self.assertIn(str(missing), json.loads(Path(args[-1]).read_text())["error"]["message"])
            changed = artifact("changed", decode=(200, 200, 200))
            changed[1]["response"]["tokens"][0] = 900
            args[1] = str(a); args[3] = str(self.write("changed", changed))
            args[-1] = str(self.root / "correctness.json")
            self.assertEqual(COMPARE.main(args + ["--min-decode-gain-pct", "3", "--max-ingest-regression-pct", "2"]), 1)
            result = json.loads(Path(args[-1]).read_text())
            self.assertEqual(result["correctness"], "failed")
            self.assertEqual(result["error"]["type"], "OutputMismatch")


if __name__ == "__main__":
    unittest.main()
