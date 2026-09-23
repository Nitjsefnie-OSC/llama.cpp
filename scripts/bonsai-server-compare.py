#!/usr/bin/env python3
"""Validate and compare completed Bonsai server benchmark JSONL artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


class InvalidRun(ValueError):
    pass


class OutputMismatch(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidRun(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def number(value, minimum=0, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > minimum if positive else value >= minimum))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise InvalidRun(f"Non-finite JSON constant: {value}")


def validate_trial(row, metadata):
    size, count = row["prompt_tokens"], metadata["tokens"]
    body, response = row["request"], row["response"]
    require(isinstance(body, dict) and isinstance(response, dict), "Invalid request/response")
    prompt = body["prompt"]
    require(isinstance(prompt, list) and len(prompt) == size and all(integer(t) for t in prompt),
            "Invalid prompt tokens")
    require(integer(body["n_predict"], 1) and body["n_predict"] == count, "Wrong requested output count")
    require(body.get("cache_prompt") is False and body.get("return_tokens") is True and
            body.get("ignore_eos") is True, "Expected uncached requests with returned tokens and ignored EOS")
    require(integer(body.get("id_slot")) and body["id_slot"] == 0 and
            integer(body.get("seed")) and number(body.get("temperature")) and body["temperature"] == 0,
            "Invalid deterministic request settings")
    require(response.get("stop") is True and response.get("truncated") is False, "Incomplete response")
    require(integer(response.get("id_slot")) and response["id_slot"] == body["id_slot"], "Wrong response slot")
    tokens, content = response["tokens"], response["content"]
    require(isinstance(tokens, list) and len(tokens) == count and all(integer(t) for t in tokens),
            "Invalid output tokens")
    require(isinstance(content, str), "Invalid output content")
    require(row["content_sha256"] == hashlib.sha256(content.encode()).hexdigest(), "Content hash differs")
    timings = response["timings"]
    require(isinstance(timings, dict), "Invalid timings")
    for obj, key, expected in ((response, "tokens_evaluated", size), (response, "tokens_predicted", count),
                               (timings, "prompt_n", size), (timings, "predicted_n", count),
                               (timings, "cache_n", 0)):
        require(integer(obj.get(key)) and obj[key] == expected, f"Wrong {key}")
    require(number(row["wall_seconds"], positive=True), "Invalid wall_seconds")
    metrics = {"wall_seconds": row["wall_seconds"]}
    # server_slot_stats excludes the first generated token from timed decode steps.
    for prefix, steps, label in (("prompt", size, "ingest_tps"), ("predicted", count - 1, "decode_tps")):
        milliseconds = timings[prefix + "_ms"]
        require(number(milliseconds, positive=steps > 0), f"Invalid {prefix}_ms")
        rate = 1000.0 / milliseconds * steps if milliseconds else 0.0
        require(number(rate), f"Invalid calculated {label}")
        reported = timings[prefix + "_per_second"]
        require(number(reported) and math.isclose(rate, reported, rel_tol=1e-9, abs_tol=1e-9),
                f"Inconsistent {prefix}_per_second")
        metrics[label] = rate
    return metrics


def load_run(path):
    try:
        raw = path.read_bytes()
        require(raw.endswith(b"\n"), "Missing final JSONL newline")
        rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=invalid_constant)
                for line in raw.decode("utf-8").splitlines()]
        require(rows and all(isinstance(row, dict) for row in rows), "Expected JSON objects")
        metadata = rows[0]
        require(metadata.get("kind") == "metadata", "First record must be metadata")
        sizes, count, reps = (metadata[key] for key in ("sizes", "tokens", "reps"))
        require(isinstance(sizes, list) and sizes and all(integer(n, 1) for n in sizes) and
                len(set(sizes)) == len(sizes), "Invalid/duplicate prompt sizes")
        require(integer(count, 1) and integer(reps, 1), "Invalid tokens/reps")
        props = metadata["props"]
        identity = {key: props[key] for key in ("model_path", "model_alias", "total_slots")}
        identity["n_ctx"] = props["default_generation_settings"]["n_ctx"]
        require(isinstance(identity["model_path"], str) and isinstance(identity["model_alias"], str) and
                integer(identity["total_slots"], 1) and integer(identity["n_ctx"], 1), "Invalid model/context metadata")
        require(max(sizes) + count <= identity["n_ctx"], "Benchmark exceeds context")
        keys = [(size, rep) for rep in range(reps + 1) for size in sizes]
        require(len(rows) == 1 + len(keys) + len(sizes), "Incomplete or extra benchmark records")
        trials = {}
        for row, (size, rep) in zip(rows[1:], keys):
            require(row.get("kind") == "trial" and integer(row.get("prompt_tokens"), 1) and
                    row["prompt_tokens"] == size and integer(row.get("repetition")) and row["repetition"] == rep,
                    f"Missing, duplicate or out-of-order trial: prompt={size}, repetition={rep}")
            require(type(row.get("warmup")) is bool and row["warmup"] == (rep == 0), "Wrong warmup flag")
            trials[(size, rep)] = {"row": row, "metrics": validate_trial(row, metadata)}
        for row, size in zip(rows[1 + len(keys):], sizes):
            require(row.get("kind") == "summary" and integer(row.get("prompt_tokens"), 1) and
                    row["prompt_tokens"] == size, "Missing, duplicate or out-of-order summary")
            require(all(number(row.get(key)) for key in ("ingest_median", "decode_median", "wall_median")),
                    "Invalid summary values")
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "config": {"sizes": sizes, "tokens": count, "reps": reps, "identity": identity}, "trials": trials}
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise InvalidRun(f"{path}: {error}") from error


def prompt_gain_overrides(values):
    if values is None:
        return None
    overrides = {}
    for value in values:
        try:
            prompt, pct = value.split("=")
            require(prompt.isascii() and prompt.isdecimal(), f"Invalid prompt threshold: {value}")
            prompt, pct = int(prompt), float(pct)
            require(integer(prompt, 1) and number(pct, minimum=-math.inf), f"Invalid prompt threshold: {value}")
            require(prompt not in overrides, f"Duplicate prompt threshold: {prompt}")
        except (ValueError, OverflowError) as error:
            raise InvalidRun(f"Invalid --min-ingest-gain-pct-for {value}: {error}") from error
        overrides[prompt] = pct
    return overrides


def compare(baseline, candidate, min_decode_gain=None, max_ingest_regression=None, *,
            min_ingest_gain=None, max_decode_regression=None, min_ingest_gain_for=None):
    require(bool(baseline) and bool(candidate), "Both conditions require at least one artifact")
    require(min_ingest_gain_for is None or (isinstance(min_ingest_gain_for, dict) and min_ingest_gain_for and
            all(integer(size, 1) and number(pct, minimum=-math.inf) for size, pct in min_ingest_gain_for.items())),
            "Prompt thresholds must map positive integer sizes to finite gains")
    overrides = min_ingest_gain_for or {}
    require((min_decode_gain is None) == (max_ingest_regression is None), "Both performance thresholds are required together")
    require((min_ingest_gain is None) == (max_decode_regression is None), "Both ingest performance thresholds are required together")
    require(min_decode_gain is None or min_ingest_gain is None, "Decode and ingest gate pairs are mutually exclusive")
    thresholds = (min_decode_gain, max_ingest_regression, max_decode_regression)
    require(all(value is None or number(value) for value in thresholds), "Thresholds must be finite and nonnegative")
    require(min_ingest_gain is None or number(min_ingest_gain, minimum=-math.inf if overrides else 0),
            "Ingest gain must be finite and nonnegative unless prompt overrides are supplied")
    axis = "decode" if min_decode_gain is not None else "ingest" if min_ingest_gain is not None else None
    require(not overrides or axis == "ingest", "Prompt overrides require the ingest performance gate pair")
    gain, regression = (min_decode_gain, max_ingest_regression) if axis == "decode" else (min_ingest_gain, max_decode_regression)
    paths = [Path(p).resolve() for p in [*baseline, *candidate]]
    require(len(set(paths)) == len(paths), "Duplicate artifact path")
    runs = [load_run(path) for path in paths]
    require(len({r["sha256"] for r in runs}) == len(runs), "Duplicate artifact contents")
    reference = runs[0]
    require(set(overrides) <= set(reference["config"]["sizes"]),
            f"Unknown prompt threshold sizes: {sorted(set(overrides) - set(reference['config']['sizes']))}")
    for run in runs[1:]:
        require(run["config"] == reference["config"], f"{run['path']}: benchmark configuration differs")
    gate = {"requested": axis is not None, "passed": None, "axis": axis,
            "min_decode_gain_pct": min_decode_gain, "max_ingest_regression_pct": max_ingest_regression,
            "min_ingest_gain_pct": min_ingest_gain, "max_decode_regression_pct": max_decode_regression}
    if overrides:
        gate.update(thresholds_resolved=True,
                    min_ingest_gain_pct_for={str(size): pct for size, pct in sorted(overrides.items())},
                    effective_prompt_thresholds={str(size): {"min_ingest_gain_pct": overrides.get(size, gain),
                                                            "max_decode_regression_pct": max_decode_regression}
                                                 for size in reference["config"]["sizes"]})
    try:
        for run in runs[1:]:
            for key, entry in run["trials"].items():
                actual, expected = entry["row"], reference["trials"][key]["row"]
                where = f"{run['path']}: prompt={key[0]}, repetition={key[1]}"
                require(actual["request"] == expected["request"], f"{where}: request differs")
                for field in ("tokens", "content"):
                    if actual["response"][field] != expected["response"][field]:
                        raise OutputMismatch(f"{where}: output {field} differs from {reference['path']}")
    except (InvalidRun, OutputMismatch) as error:
        if overrides:
            error.gate = gate
        raise
    result = {"status": "passed", "correctness": "passed", "reference": reference["path"],
              "config": reference["config"], "conditions": {}, "prompts": [],
              "gate": gate}
    groups = {"baseline": runs[:len(baseline)], "candidate": runs[len(baseline):]}
    for name, group in groups.items():
        result["conditions"][name] = [{k: run[k] for k in ("path", "sha256")} for run in group]
    for size in reference["config"]["sizes"]:
        prompt = {"prompt_tokens": size}
        for name, group in groups.items():
            samples = [entry["metrics"] for run in group for (n, rep), entry in run["trials"].items()
                       if n == size and rep > 0]
            prompt[name] = {"samples": len(samples), **{key: statistics.median(s[key] for s in samples)
                           for key in ("ingest_tps", "decode_tps", "wall_seconds")}}
        prompt["change_pct"] = {key: 100 * (prompt["candidate"][key] - prompt["baseline"][key]) / prompt["baseline"][key]
                                if prompt["baseline"][key] else None
                                for key in ("ingest_tps", "decode_tps", "wall_seconds")}
        require(all(value is None or math.isfinite(value) for value in prompt["change_pct"].values()),
                f"Non-finite percentage change for prompt={size}")
        if axis is not None:
            delta = prompt["change_pct"]
            other = "ingest" if axis == "decode" else "decode"
            prompt_gain = overrides.get(size, gain)
            if overrides:
                prompt["gate_thresholds"] = {"min_ingest_gain_pct": prompt_gain,
                                             "max_decode_regression_pct": max_decode_regression}
            prompt["gate_passed"] = (delta["decode_tps"] is not None and delta[axis + "_tps"] >= prompt_gain and
                                     delta[other + "_tps"] >= -regression)
        result["prompts"].append(prompt)
    if axis is not None:
        result["gate"]["passed"] = all(p["gate_passed"] for p in result["prompts"])
        if not result["gate"]["passed"]:
            result["status"] = "failed"
            result["error"] = {"type": "PerformanceGate", "message": "One or more prompts failed the performance gate"}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, help="exclusive new JSON result; also printed to stdout")
    parser.add_argument("--min-decode-gain-pct", type=float)
    parser.add_argument("--max-ingest-regression-pct", type=float)
    parser.add_argument("--min-ingest-gain-pct", type=float)
    parser.add_argument("--max-decode-regression-pct", type=float)
    parser.add_argument("--min-ingest-gain-pct-for", action="append", metavar="PROMPT=PCT",
                        help="override one prompt's signed ingest gain floor; repeatable, requires the ingest gate pair")
    args = parser.parse_args(argv)
    output = None
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("x", encoding="utf-8")
        try:
            result = compare(args.baseline, args.candidate, args.min_decode_gain_pct, args.max_ingest_regression_pct,
                             min_ingest_gain=args.min_ingest_gain_pct, max_decode_regression=args.max_decode_regression_pct,
                             min_ingest_gain_for=prompt_gain_overrides(args.min_ingest_gain_pct_for))
        except (InvalidRun, OutputMismatch) as error:
            result = {"status": "failed", "correctness": "failed" if isinstance(error, OutputMismatch) else "not_established",
                      "baseline": [str(p) for p in args.baseline], "candidate": [str(p) for p in args.candidate],
                      "error": {"type": type(error).__name__, "message": str(error)}}
            if args.min_ingest_gain_pct_for is not None:
                result["gate"] = getattr(error, "gate", {"thresholds_resolved": False})
        if args.min_ingest_gain_pct_for is not None:
            # Text keeps invalid non-finite arguments serializable in failure artifacts.
            result["threshold_arguments"] = {
                key: None if getattr(args, key) is None else str(getattr(args, key))
                for key in ("min_decode_gain_pct", "max_ingest_regression_pct",
                            "min_ingest_gain_pct", "max_decode_regression_pct")}
            result["threshold_arguments"]["min_ingest_gain_pct_for"] = args.min_ingest_gain_pct_for
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if output:
            output.write(encoded)
        sys.stdout.write(encoded)
        for p in result.get("prompts", []):
            b, c = p["baseline"], p["candidate"]
            print(f"pp{p['prompt_tokens']}: n={b['samples']}/{c['samples']} "
                  f"ingest {b['ingest_tps']:.3f}->{c['ingest_tps']:.3f} tok/s; "
                  f"decode {b['decode_tps']:.3f}->{c['decode_tps']:.3f} tok/s; "
                  f"wall {b['wall_seconds']:.3f}->{c['wall_seconds']:.3f}s", file=sys.stderr)
        print(result["status"].upper() + (": " + result["error"]["message"] if "error" in result else ""), file=sys.stderr)
        return 0 if result["status"] == "passed" else 1
    except OSError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
