#!/usr/bin/env python3
"""Summarize DEBUG_CUDA_GRAPH_STATS server logs; all durations are host microseconds."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


MARKER = re.compile(r"(?<![A-Za-z0-9_])CUDA_GRAPH_STATS\b")
NUMBERS = {"time_us", "device", "uid", "host_us", "idle_us", "tokens", "node"}
FIELDS = NUMBERS | {"event", "key", "mode", "reason"}
EVENTS = {"miss", "evict", "change", "execute", "capture", "instantiate", "update"}
MODES = ("direct", "capture", "replay")
LIFECYCLE = ("capture", "instantiate", "update", "evict")


def read_records(lines):
    records = []
    for line_number, line in enumerate(lines, 1):
        marker = MARKER.search(line)
        if not marker:
            continue
        try:
            parts = line[marker.start():].strip().split(",")
            if parts[0] != "CUDA_GRAPH_STATS":
                raise ValueError("expected comma after marker")
            record = {}
            for part in parts[1:]:
                key, separator, value = part.partition("=")
                if not separator or not value or key in record:
                    raise ValueError("empty, malformed, or duplicate field")
                record[key] = value
            if record.keys() != FIELDS:
                raise ValueError("missing or unexpected fields")
            for field in NUMBERS:
                value = record[field]
                if not re.fullmatch(r"[0-9]+", value) and not (field == "node" and value == "-1"):
                    raise ValueError(f"invalid integer {field}={value}")
                record[field] = int(value)
            if record["event"] not in EVENTS:
                raise ValueError("unsupported event")
            allowed_modes = MODES if record["event"] == "execute" else ("-",)
            if record["mode"] not in allowed_modes:
                raise ValueError("unsupported mode for event")
            if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]+|\(nil\)", record["key"]):
                raise ValueError("invalid pointer key")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", record["reason"]):
                raise ValueError("invalid reason")
            records.append(record)
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
    if not any(record["event"] == "execute" for record in records):
        raise ValueError("no execute records found")
    return records


def summarize(records):
    events = Counter(record["event"] for record in records)
    executions = {phase: dict.fromkeys(MODES, 0) for phase in ("prefill", "decode", "unknown")}
    lifecycle = {event: {"count": 0, "total_us": 0, "max_us": 0} for event in LIFECYCLE}
    changes = Counter()
    resets = 0
    evictions = []
    for record in records:
        event = record["event"]
        if event == "execute":
            tokens = record["tokens"]
            phase = "prefill" if tokens > 1 else "decode" if tokens == 1 else "unknown"
            executions[phase][record["mode"]] += 1
            resets += record["reason"] == "warmup_reset"
        if event == "change":
            changes[record["reason"]] += 1
        if event in lifecycle:
            item = lifecycle[event]
            item["count"] += 1
            item["total_us"] += record["host_us"]
            item["max_us"] = max(item["max_us"], record["host_us"])
        if event == "evict":
            evictions.append({key: record[key] for key in ("device", "key", "uid", "idle_us")})
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(events.items())),
        "execution_counts": executions,
        "host_lifecycle": lifecycle,
        "evictions": evictions,
        "property_change_counts": dict(sorted(changes.items())),
        "warmup_reset_count": resets,
        "timing_note": "Host microseconds only; lifecycle records are not assigned execution phases.",
        "phase_criterion": "Quantized MUL_MAT token width: >1 prefill, 1 decode, 0 unknown.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new JSON output path (never overwritten)")
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8") as source:
            result = summarize(read_records(source))
        result["source"] = str(args.log.resolve())
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print("Events: " + ", ".join(f"{event}={count}" for event, count in result["event_counts"].items()))
    for phase, counts in result["execution_counts"].items():
        print(f"{phase}: " + ", ".join(f"{mode}={count}" for mode, count in counts.items()))
    for event, timing in result["host_lifecycle"].items():
        print(f"host {event}: count={timing['count']} total_us={timing['total_us']} max_us={timing['max_us']}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
