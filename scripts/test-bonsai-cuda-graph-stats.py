#!/usr/bin/env python3
"""Stdlib tests for the graph diagnostic parser and exclusive CLI output."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("bonsai-cuda-graph-stats.py")
SPEC = importlib.util.spec_from_file_location("graph_stats", SCRIPT)
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


def record(**changes):
    fields = dict(event="execute", time_us=100, device=0, key="000001AB", uid=3,
                  host_us=0, idle_us=0, tokens=1, mode="replay", reason="stable", node=-1)
    fields.update(changes)
    return "CUDA_GRAPH_STATS," + ",".join(f"{key}={value}" for key, value in fields.items())


class ParserTests(unittest.TestCase):
    def test_mixed_summary(self):
        lines = ["unrelated DEBUG_CUDA_GRAPH_STATS=1 log line",
                 "[server prefix] " + record(tokens=508, mode="direct", reason="warmup_reset"),
                 record(tokens=4, mode="capture", reason="warmup_complete"), record(),
                 record(tokens=0, mode="direct", reason="disabled"),
                 record(event="miss", mode="-", uid=0, tokens=0, reason="-"),
                 record(event="change", mode="-", reason="node_properties", node=17),
                 record(event="change", mode="-", reason="node_count"),
                 record(event="capture", mode="-", host_us=40),
                 record(event="capture", mode="-", host_us=10),
                 record(event="instantiate", mode="-", host_us=7),
                 record(event="update", mode="-", host_us=3, reason="success"),
                 record(event="evict", mode="-", host_us=2, idle_us=10000000)]
        result = STATS.summarize(STATS.read_records(lines))
        self.assertEqual(result["record_count"], 12)
        self.assertEqual(result["event_counts"]["execute"], 4)
        self.assertEqual(result["execution_counts"], {
            "prefill": dict(direct=1, capture=1, replay=0),
            "decode": dict(direct=0, capture=0, replay=1),
            "unknown": dict(direct=1, capture=0, replay=0)})
        self.assertEqual(result["host_lifecycle"]["capture"], dict(count=2, total_us=50, max_us=40))
        self.assertEqual(result["evictions"], [dict(device=0, key="000001AB", uid=3, idle_us=10000000)])
        self.assertEqual(result["property_change_counts"], dict(node_properties=1, node_count=1))
        self.assertEqual(result["warmup_reset_count"], 1)
        self.assertNotIn("prefill", result["host_lifecycle"])

    def test_all_fields_required_and_duplicates_rejected(self):
        parts = record().split(",")
        for index in range(1, len(parts)):
            with self.subTest(field=parts[index]):
                with self.assertRaisesRegex(ValueError, "line 1"):
                    STATS.read_records([",".join(parts[:index] + parts[index+1:])])
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    STATS.read_records([record() + "," + parts[index]])

    def test_invalid_values(self):
        cases = [dict(event="other"), dict(mode="other"), dict(mode="-"),
                 dict(event="capture", mode="replay"), dict(key="not-a-pointer"),
                 dict(reason=""), dict(reason="two words"), dict(node=-2)]
        for field in STATS.NUMBERS - {"node"}:
            cases.extend({field: value} for value in (-1, "1.5", "nan", "True"))
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "line 2"):
                STATS.read_records(["unrelated", record(**changes)])

    def test_empty_truncated_and_unexpected_fields(self):
        for line in ["CUDA_GRAPH_STATS", "CUDA_GRAPH_STATS truncated", record() + ",extra=1",
                     record() + ",", record() + ",broken"]:
            with self.subTest(line=line), self.assertRaises(ValueError):
                STATS.read_records([record(), line])

    def test_no_execute(self):
        for lines in [[], ["unrelated"], [record(event="miss", mode="-")]]:
            with self.subTest(lines=lines), self.assertRaisesRegex(ValueError, "no execute"):
                STATS.read_records(lines)

    def test_pointer_formats(self):
        for key in ("0xabc", "0000ABCD", "(nil)"):
            self.assertEqual(STATS.read_records([record(key=key)])[0]["key"], key)

    def test_cli_success_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.log"
            output = Path(directory) / "output.json"
            source.write_text(record() + "\n", encoding="utf-8")
            command = [sys.executable, str(SCRIPT), str(source), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["event_counts"], dict(execute=1))
            original = output.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_invalid_input_never_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.log"
            output = Path(directory) / "output.json"
            for content in (None, "", record() + ",uid=8", record(host_us=-1)):
                if content is not None:
                    source.write_text(content, encoding="utf-8")
                result = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
