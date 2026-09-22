#!/usr/bin/env python3
"""Check live Bonsai prompt growth and four slots; retain correctness artifacts, not speed scores."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import threading
import urllib.error
import urllib.request


SIZES = (512, 4096, 16384, 512)
CORPUS = ("Binary search finds an element in a sorted sequence by halving the search interval. "
          "Compare the middle value with the target, then search the remaining half. ")
LOGPROB_ATOL = 1e-4  # Absolute natural-log units; no relative tolerance, including near zero.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_json(url, route, body, timeout):
    request = urllib.request.Request(url.rstrip("/") + route,
                                    data=None if body is None else json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} {route}: {error.read().decode('utf-8', errors='replace')}") from error
    try:
        return json.loads(raw)
    except ValueError as error:
        raise ValueError(f"Invalid JSON from {route}: {raw}") from error


def idle_slots(slots):
    require(isinstance(slots, list) and len(slots) == 4, "Expected four server slots")
    require({slot["id"] for slot in slots} == set(range(4)), "Expected slot IDs 0..3")
    require(all(slot["is_processing"] is False for slot in slots), "Server is busy; refusing requests")


def make_cases(tokens, n_predict):
    require(isinstance(tokens, list) and len(tokens) >= max(SIZES), "Tokenized corpus is too short")
    require(all(type(token) is int and token >= 0 for token in tokens), "Invalid corpus token IDs")
    cases = []
    for name, size, slot in ([(f"growth-{index}-{size}", size, 0) for index, size in enumerate(SIZES)] +
                             [(f"concurrent-slot-{slot}", 128, slot) for slot in range(4)]):
        cases.append((name, {"prompt": tokens[:size], "n_predict": n_predict,
                            "temperature": 0, "seed": 1234, "ignore_eos": True,
                            "cache_prompt": False, "id_slot": slot, "return_tokens": True,
                            "n_probs": 5, "post_sampling_probs": False, "stream": False}))
    return cases


def validate_response(body, response):
    require(isinstance(response, dict), "Response must be an object")
    require(response.get("id_slot") == body["id_slot"], "Wrong response slot")
    require(response.get("stop") is True and response.get("truncated") is False, "Incomplete/truncated response")
    require(isinstance(response.get("content"), str) and bool(response["content"]), "Empty completion content")
    tokens = response.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == body["n_predict"], "Wrong output token count")
    require(all(type(token) is int and token >= 0 for token in tokens), "Invalid output token IDs")
    for name, expected in (("tokens_predicted", body["n_predict"]), ("tokens_evaluated", len(body["prompt"]))):
        require(response.get(name) == expected, f"Wrong {name}")
    timings = response.get("timings", {})
    require(timings.get("prompt_n") == len(body["prompt"]) and timings.get("predicted_n") == body["n_predict"],
            "Wrong timing token counts")
    probabilities = response.get("completion_probabilities")
    require(isinstance(probabilities, list) and len(probabilities) == len(tokens), "Wrong probability count")
    for index, item in enumerate(probabilities):
        require(isinstance(item, dict) and item.get("id") == tokens[index], "Probability token ID mismatch")
        top = item.get("top_logprobs")
        require(isinstance(top, list) and 1 <= len(top) <= body["n_probs"], "Invalid top_logprobs count")
        for entry in [item] + top:
            require(isinstance(entry, dict) and type(entry.get("id")) is int and entry["id"] >= 0,
                    "Invalid probability token ID")
            require(isinstance(entry.get("token"), str), "Missing probability token text")
            octets = entry.get("bytes")
            require(isinstance(octets, list) and all(type(n) is int and 0 <= n <= 255 for n in octets),
                    "Invalid probability token bytes")
            value = entry.get("logprob")
            require(type(value) in (int, float) and math.isfinite(value) and value <= 0, "Invalid logprob")
        ids = [entry["id"] for entry in top]
        require(len(ids) == len(set(ids)), "Duplicate top token IDs")
        require(all(a["logprob"] >= b["logprob"] for a, b in zip(top, top[1:])), "Unsorted top_logprobs")


def compare_response(body, response, reference, atol):
    require(body == reference["request"], "Request differs from reference")
    expected = reference["response"]
    validate_response(body, expected)
    require(response["tokens"] == expected["tokens"], "Output tokens differ from reference")
    require(response["content"] == expected["content"], "Output content differs from reference")
    for index, (actual, wanted) in enumerate(zip(response["completion_probabilities"], expected["completion_probabilities"])):
        require(len(actual["top_logprobs"]) == len(wanted["top_logprobs"]), f"Top count differs at token {index}")
        for got, ref in zip([actual] + actual["top_logprobs"], [wanted] + wanted["top_logprobs"]):
            require(all(got[key] == ref[key] for key in ("id", "token", "bytes")),
                    f"Probability token identity differs at token {index}")
            require(math.isclose(got["logprob"], ref["logprob"], abs_tol=atol, rel_tol=0),
                    f"Logprob differs at token {index}, id={got['id']}: {got['logprob']} vs {ref['logprob']}")


def load_reference(path, concurrent_only=False):
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    require(rows and rows[-1].get("kind") == "summary" and rows[-1].get("status") == "passed",
            "Reference must be a completed passing run")
    require(not any(row.get("kind") == "error" for row in rows), "Reference contains errors")
    metadata = [row for row in rows if row.get("kind") == "metadata"]
    results = [row for row in rows if row.get("kind") == "result"]
    allowed_counts = (4, 8) if concurrent_only else (8,)
    require(len(metadata) == 1 and len(results) in allowed_counts,
            f"Reference requires metadata and result count in {allowed_counts}")
    require(rows[-1].get("cases") == len(results), "Reference summary/result count differs")
    expected_mode = "concurrent-only" if len(results) == 4 else "full"
    require(all(row.get("mode", expected_mode) == expected_mode for row in (metadata[0], rows[-1])),
            "Reference mode/result count differs")
    by_name = {row["case"]: row for row in results}
    require(len(by_name) == len(results), "Duplicate reference cases")
    for row in results:
        validate_response(row["request"], row["response"])
    return metadata[0], by_name


def run(args, request=request_json):
    reference = load_reference(args.reference, args.concurrent_only) if args.reference else None
    mode = "concurrent-only" if args.concurrent_only else "full"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        lock = threading.Lock()
        request_number = 0

        def record(row):
            with lock:
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()

        def call(route, body=None, case=None):
            nonlocal request_number
            with lock:
                request_number += 1
                request_id = request_number
            audit = {"request_id": request_id, "route": route}
            if case is not None:
                audit["case"] = case
            record(dict(audit, kind="request" if case is not None else "http_request", request=body))
            try:
                response = request(args.url, route, body, args.timeout)
            except Exception as error:
                record(dict(audit, kind="http_error", error=str(error)))
                raise
            if case is not None:
                # Keep the reference artifact schema without duplicating completion responses.
                record(dict(audit, kind="result", request=body, response=response))
            else:
                record(dict(audit, kind="http_response", response=response))
            return response

        try:
            record({"kind": "start", "url": args.url, "reference": str(args.reference) if args.reference else None,
                    "mode": mode,
                    "logprob_atol": args.logprob_atol, "probability_tolerance": "absolute natural-log units; relative tolerance zero"})
            slots = call("/slots")
            record({"kind": "slots", "stage": "start", "response": slots})
            idle_slots(slots)
            props = call("/props")
            record({"kind": "metadata", "mode": mode, "props": props, "slots": slots})
            require(props["total_slots"] == 4 and props["default_generation_settings"]["n_ctx"] == 188416,
                    "Expected unchanged context 188416 and four slots")
            if reference:
                previous = reference[0]["props"]
                for key in ("model_path", "model_alias", "total_slots"):
                    require(props[key] == previous[key], f"Reference {key} differs")
                require(props["default_generation_settings"]["n_ctx"] == previous["default_generation_settings"]["n_ctx"],
                        "Reference context differs")
            tokenize_body = {"content": CORPUS * (max(SIZES) // 8 + 1)}
            tokenized = call("/tokenize", tokenize_body)
            cases = make_cases(tokenized["tokens"], args.tokens)
            selected_cases = cases[4:] if args.concurrent_only else cases
            if reference:
                reference_cases = cases if len(reference[1]) == 8 else cases[4:]
                require(set(reference[1]) == {name for name, _ in reference_cases}, "Reference case names differ")
                for name, body in selected_cases:
                    require(body == reference[1][name]["request"], f"Reference request differs: {name}")

            def exercise(name, body, barrier=None):
                try:
                    if barrier:
                        barrier.wait(timeout=args.timeout)
                    response = call("/completion", body, case=name)
                    validate_response(body, response)
                    if reference:
                        compare_response(body, response, reference[1][name], args.logprob_atol)
                    print(f"PASS {name}: {len(body['prompt'])} prompt tokens, {len(response['tokens'])} output tokens", flush=True)
                    return {"request": body, "response": response}
                except Exception as error:
                    record({"kind": "error", "case": name, "error": str(error)})
                    raise

            first_result = None
            for index, (name, body) in enumerate([] if args.concurrent_only else cases[:4]):
                idle_slots(call("/slots"))
                print(f"Checking {name}", flush=True)
                result = exercise(name, body)
                if index == 0:
                    first_result = result
                elif index == 3:
                    compare_response(body, result["response"], first_result, args.logprob_atol)
                    print("PASS repeated 512-token prompt matches before growth", flush=True)
            idle_slots(call("/slots"))
            print("Checking four concurrent explicit slots", flush=True)
            barrier = threading.Barrier(4)
            errors = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(exercise, name, body, barrier) for name, body in cases[4:]]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
            require(not errors, f"Concurrent cases failed: {errors}")
            record({"kind": "summary", "status": "passed", "mode": mode,
                    "cases": len(selected_cases), "compared": reference is not None})
            print(f"PASS {mode} correctness checks", flush=True)
        except Exception as error:
            record({"kind": "summary", "status": "failed", "mode": mode, "error": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", required=True, type=Path, help="exclusive new JSONL artifact")
    parser.add_argument("--reference", type=Path, help="completed passing artifact to compare")
    parser.add_argument("--concurrent-only", action="store_true",
                        help="run only slots 0..3 concurrently; accepts a passing four- or eight-case reference")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600, help="finite HTTP socket/barrier timeout in seconds")
    parser.add_argument("--logprob-atol", type=float, default=LOGPROB_ATOL,
                        help="absolute log-probability tolerance (default 1e-4 natural-log units; relative tolerance zero)")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 512 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("tokens must be 1..512 and timeout finite and positive")
    if not math.isfinite(args.logprob_atol) or not 0 <= args.logprob_atol <= 0.01:
        parser.error("logprob-atol must be finite and between 0 and 0.01")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
