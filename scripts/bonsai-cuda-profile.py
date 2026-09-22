#!/usr/bin/env python3
"""Summarize DEBUG_CUDA_TIMING records from a llama-server --log-file log."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


MARKER = re.compile(r"CUDA_TIMING(?:_TOTAL)?(?=,|\s|$)")
LAYER = re.compile(r"-\d+(?=$|[ (])")
ANONYMOUS_NODE = re.compile(r"^node_\d+$")


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise ValueError("expected a nonnegative integer")
    return result


def milliseconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("expected finite nonnegative milliseconds")
    return result


def read_graphs(lines):
    graphs = []
    operations = []
    errors = []
    first_line = None
    for line_number, line in enumerate(lines, 1):
        marker = MARKER.search(line)
        if not marker:
            continue
        if first_line is None:
            first_line = line_number
        fields = line[marker.start():].strip().split(",")
        is_total = fields[0] == "CUDA_TIMING_TOTAL"
        total = None
        try:
            if is_total:
                if len(fields) != 4:
                    raise ValueError("expected 4 total fields")
                total = (nonnegative_int(fields[1]), milliseconds(fields[2]), milliseconds(fields[3]))
            else:
                if len(fields) != 8 or not all(fields[1:4]):
                    raise ValueError("expected 8 operation fields with op, name, and type")
                operations.append({
                    "op": fields[1], "name": ANONYMOUS_NODE.sub("node_<id>", LAYER.sub("-<layer>", fields[2])),
                    "src0_type": fields[3],
                    "src0_ne0": nonnegative_int(fields[4]), "dst_ne0": nonnegative_int(fields[5]),
                    "dst_ne1": nonnegative_int(fields[6]), "ms": milliseconds(fields[7]),
                })
        except ValueError as error:
            errors.append(f"line {line_number}: {error}")
        if is_total:
            if total is not None and total[0] != len(operations):
                errors.append(f"operation count mismatch: total={total[0]}, parsed={len(operations)}")
            graphs.append(make_graph(len(graphs), first_line, line_number, operations, total, errors))
            operations, errors, first_line = [], [], None
    if first_line is not None:
        errors.append("incomplete graph: missing CUDA_TIMING_TOTAL at end of input")
        graphs.append(make_graph(len(graphs), first_line, line_number, operations, None, errors))
    return graphs


def make_graph(index, first_line, last_line, operations, total, errors):
    widths = sorted({op["dst_ne1"] for op in operations if op["op"] == "MUL_MAT"})
    quantized_widths = sorted({op["dst_ne1"] for op in operations
                              if op["op"] == "MUL_MAT" and "q" in op["src0_type"].lower()})
    phase_widths = quantized_widths or widths
    phase = "prefill" if any(width > 1 for width in phase_widths) else "decode" if phase_widths == [1] else "unknown"
    parsed_ms = math.fsum(op["ms"] for op in operations)
    graph = {
        "index": index, "first_line": first_line, "last_line": last_line,
        "phase": phase, "mul_mat_dst_ne1": widths, "complete": total is not None and not errors,
        "phase_mul_mat_dst_ne1": phase_widths,
        "phase_source": "quantized_mul_mat" if quantized_widths else "all_mul_mat",
        "errors": errors, "operations": operations, "parsed_operation_ms": parsed_ms,
        "reported_operation_count": total[0] if total else None,
        "graph_ms": total[1] if total else None, "reported_operation_ms": total[2] if total else None,
        "operation_sum_delta_ms": parsed_ms - total[2] if total else None,
        "graph_minus_operations_ms": total[1] - parsed_ms if total else None,
        "sum_reconciles": total is not None and math.isclose(
            parsed_ms, total[2], rel_tol=1e-4, abs_tol=1e-6 * (len(operations) + 1)),
    }
    return graph


def summarize(graphs):
    groups = {}
    for graph in graphs:
        for op in graph["operations"]:
            key = tuple(op[field] for field in ("op", "name", "src0_type", "src0_ne0", "dst_ne0", "dst_ne1"))
            if key not in groups:
                groups[key] = {field: value for field, value in op.items() if field != "ms"}
                groups[key].update(count=0, total_ms=0.0)
            groups[key]["count"] += 1
            groups[key]["total_ms"] += op["ms"]
    operation_ms = math.fsum(graph["parsed_operation_ms"] for graph in graphs)
    reported_ms = math.fsum(graph["reported_operation_ms"] for graph in graphs)
    graph_ms = math.fsum(graph["graph_ms"] for graph in graphs)
    for group in groups.values():
        group["mean_ms"] = group["total_ms"] / group["count"]
        group["operation_percent"] = 100 * group["total_ms"] / operation_ms if operation_ms else 0.0
    return {
        "graph_count": len(graphs), "operation_count": sum(len(graph["operations"]) for graph in graphs),
        "graph_ms": graph_ms, "parsed_operation_ms": operation_ms, "reported_operation_ms": reported_ms,
        "operation_sum_delta_ms": operation_ms - reported_ms, "graph_minus_operations_ms": graph_ms - operation_ms,
        "groups": sorted(groups.values(), key=lambda group: (-group["total_ms"], group["op"], group["name"])),
    }


def profile(lines, skip_graphs=0):
    graphs = read_graphs(lines)
    selected = [graph for graph in graphs if graph["index"] >= skip_graphs and graph["complete"]]
    issues = []
    if not graphs:
        issues.append("no CUDA timing graphs found")
    for graph in graphs:
        issues.extend(f"graph {graph['index']}: {error}" for error in graph["errors"])
        if graph["complete"] and not graph["sum_reconciles"]:
            issues.append(f"graph {graph['index']}: reported operation sum differs from parsed sum")
    if graphs and not selected:
        issues.append("no complete graphs remain after exclusions")
    return {
        "schema_version": 1, "skip_graphs": skip_graphs, "graphs_seen": len(graphs),
        "complete_graphs": sum(graph["complete"] for graph in graphs),
        "excluded_incomplete_graphs": sum(not graph["complete"] for graph in graphs),
        "skipped_graphs": min(skip_graphs, len(graphs)), "issues": issues,
        "notes": [
            "Graph phase prefers MUL_MAT with quantized src0_type (case-insensitive name contains q); falls back to all MUL_MAT if none exist.",
            "Using those dst_ne1 widths: any >1 is prefill; all 1 is decode; otherwise unknown. Auxiliary F32 transforms are ignored when quantized projections exist.",
            "Multi-token decode batches also classify as prefill; the log has no request phase identifier.",
            "Operation percentages use the operation sum; concurrent streams can make it exceed graph elapsed time.",
            "DEBUG_CUDA_TIMING disables graph replay and adds synchronization; these are diagnostic timings.",
        ],
        "totals": summarize(selected),
        "phases": {phase: summarize([graph for graph in selected if graph["phase"] == phase])
                   for phase in ("prefill", "decode", "unknown")},
        "graphs": [{key: value for key, value in graph.items() if key != "operations"} for graph in graphs],
    }


def print_summary(result):
    print(f"Graphs: {result['graphs_seen']} seen, {result['complete_graphs']} complete, "
          f"{result['skipped_graphs']} skipped, {result['excluded_incomplete_graphs']} incomplete")
    for phase, summary in [("all", result["totals"]), *result["phases"].items()]:
        if not summary["graph_count"]:
            continue
        print(f"{phase}: graphs={summary['graph_count']} graph={summary['graph_ms']:.6f} ms "
              f"ops={summary['parsed_operation_ms']:.6f} ms reported={summary['reported_operation_ms']:.6f} ms "
              f"sum_delta={summary['operation_sum_delta_ms']:.6f} ms "
              f"graph_minus_ops={summary['graph_minus_operations_ms']:.6f} ms")
        for group in summary["groups"][:10]:
            print(f"  {group['total_ms']:10.4f} ms {group['operation_percent']:6.2f}% n={group['count']} "
                  f"{group['op']} {group['name']} {group['src0_type']} "
                  f"src0={group['src0_ne0']} dst={group['dst_ne0']}x{group['dst_ne1']}")
    for issue in result["issues"]:
        print(f"WARNING: {issue}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; existing files are rejected")
    parser.add_argument("--skip-graphs", type=nonnegative_int, default=0,
                        help="exclude the first N graph records, including incomplete records (default: 0)")
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8-sig", errors="replace") as source:
            result = profile(source, args.skip_graphs)
        result["input"] = str(args.log.resolve())
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print_summary(result)
    return 2 if result["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
