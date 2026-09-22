#!/usr/bin/env python3
"""Measure uncached Bonsai ingestion and decode through a running llama server."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", default="512,4096")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    sizes = [int(n) for n in args.prompts.split(",")]
    if min(sizes) < 1 or args.tokens < 1 or args.reps < 1:
        parser.error("Prompt lengths, tokens and repetitions must be positive")

    def request(route, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url.rstrip("/") + route, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)

    def gpu():
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    props = request("/props")
    if max(sizes) + args.tokens > props["default_generation_settings"]["n_ctx"]:
        parser.error("Benchmark exceeds server context")
    text = ("Binary search finds an element in a sorted sequence by halving the search interval. "
            "Compare the middle value with the target, then search the remaining half. ")
    tokenized = request("/tokenize", {"content": text * (max(sizes) // 8 + 1)})["tokens"]
    if len(tokenized) < max(sizes):
        raise RuntimeError("Tokenized corpus is too short")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        def record(row):
            output.write(json.dumps(row) + "\n")
            output.flush()

        record({"kind": "metadata", "url": args.url, "props": props,
                "gpu_fields": "temperature,sm_clock,memory_clock,power,memory_used,utilization",
                "gpu": gpu(), "sizes": sizes, "tokens": args.tokens, "reps": args.reps})
        rows = []
        for repetition in range(args.reps + 1):
            for size in sizes:
                if any(slot["is_processing"] for slot in request("/slots")):
                    raise RuntimeError("Server is busy; refusing a contaminated measurement")
                body = {"prompt": tokenized[:size], "n_predict": args.tokens,
                        "temperature": 0, "seed": 1234, "ignore_eos": True,
                        "cache_prompt": False, "id_slot": 0, "return_tokens": True}
                before = gpu()
                start = time.perf_counter()
                response = request("/completion", body)
                elapsed = time.perf_counter() - start
                timings = response["timings"]
                if timings["prompt_n"] != size or timings["predicted_n"] != args.tokens:
                    raise RuntimeError(f"Unexpected token counts: {timings}")
                row = {"kind": "trial", "warmup": repetition == 0, "repetition": repetition,
                       "prompt_tokens": size, "wall_seconds": elapsed, "gpu_before": before,
                       "gpu_after": gpu(), "request": body, "response": response,
                       "content_sha256": hashlib.sha256(response["content"].encode()).hexdigest()}
                record(row)
                if repetition:
                    rows.append(row)
                print(f"{'warmup' if not repetition else repetition}: pp{size} "
                      f"{timings['prompt_per_second']:.2f} tok/s; tg{args.tokens} "
                      f"{timings['predicted_per_second']:.2f} tok/s; wall {elapsed:.2f}s", flush=True)
        for size in sizes:
            subset = [r for r in rows if r["prompt_tokens"] == size]
            summary = {"kind": "summary", "prompt_tokens": size,
                       "ingest_median": statistics.median(r["response"]["timings"]["prompt_per_second"] for r in subset),
                       "decode_median": statistics.median(r["response"]["timings"]["predicted_per_second"] for r in subset),
                       "wall_median": statistics.median(r["wall_seconds"] for r in subset)}
            record(summary)
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
