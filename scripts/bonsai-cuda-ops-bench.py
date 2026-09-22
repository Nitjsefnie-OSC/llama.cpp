#!/usr/bin/env python3
"""Compare Bonsai CUDA operations in ABBA order using test-backend-ops.

These are isolated operation timings, not end-to-end model token rates.
Put the CUDA runtime DLL directory on PATH on Windows.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def gpu():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()


def binary_hashes(directory):
    paths = [p for p in directory.iterdir() if p.is_file() and
             (p.name.startswith("ggml") or p.stem == "test-backend-ops") and
             p.suffix in (".dll", ".exe", ".so", ".dylib", "")]
    hashes = {}
    for path in paths:
        with path.open("rb") as source:
            hashes[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="type_a=pq2_0")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if os.environ.get("DEBUG_CUDA_TIMING", "0") != "0":
        parser.error("Unset DEBUG_CUDA_TIMING before measuring throughput")
    binaries = {}
    for label in ("baseline", "candidate"):
        directory = getattr(args, label).resolve(strict=True)
        matches = [directory / name for name in ("test-backend-ops.exe", "test-backend-ops")]
        binaries[label] = next((p for p in matches if p.is_file()), None)
        if binaries[label] is None:
            parser.error(f"test-backend-ops missing in {directory}")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"started": datetime.now(timezone.utc).isoformat(), "pattern": args.pattern,
                "rounds": args.rounds, "gpu": gpu(),
                "environment": {k: v for k, v in os.environ.items() if k.startswith("GGML_CUDA_")},
                "binaries": {
                    label: {"path": str(exe), "sha256": binary_hashes(exe.parent)}
                    for label, exe in binaries.items()}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    samples = {label: {} for label in binaries}
    expected_cases = None
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    timing = re.compile(r"^\s*(MUL_MAT\(.*\)):\s+\d+ runs\s+-\s+([\d.]+) us/run", re.MULTILINE)
    with (args.output / "trials.jsonl").open("x", encoding="utf-8") as trials:
        for index, label in enumerate(["baseline", "candidate", "candidate", "baseline"] * args.rounds):
            command = [str(binaries[label]), "perf", "-o", "MUL_MAT", "-b", "CUDA0", "-p", args.pattern]
            before = gpu()
            start = time.perf_counter()
            result = subprocess.run(command, capture_output=True)
            wall = time.perf_counter() - start
            prefix = args.output / f"{index:02d}-{label}"
            prefix.with_suffix(".stdout.txt").write_bytes(result.stdout)
            prefix.with_suffix(".stderr.txt").write_bytes(result.stderr)
            values = dict((case, float(us)) for case, us in timing.findall(ansi.sub("", result.stdout.decode(errors="replace"))))
            trials.write(json.dumps({"index": index, "label": label, "command": command,
                                     "returncode": result.returncode, "wall_seconds": wall,
                                     "gpu_before": before, "gpu_after": gpu(), "microseconds": values}) + "\n")
            trials.flush()
            if result.returncode or not values:
                raise RuntimeError(f"{label} failed or ran no matching cases; see {prefix}")
            if expected_cases is None:
                expected_cases = set(values)
            if set(values) != expected_cases:
                raise RuntimeError("Baseline and candidate case sets differ")
            for case, us in values.items():
                samples[label].setdefault(case, []).append(us)
            print(f"{index}: {label}, {len(values)} cases, {wall:.1f}s", flush=True)
    summary = []
    for case in sorted(expected_cases):
        baseline = samples["baseline"][case]
        candidate = samples["candidate"][case]
        row = {"case": case, "baseline_us": baseline, "candidate_us": candidate,
               "baseline_median_us": statistics.median(baseline),
               "candidate_median_us": statistics.median(candidate),
               "speedup": statistics.median(baseline) / statistics.median(candidate)}
        summary.append(row)
        print(f"{row['speedup']:.3f}x {case}", flush=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
