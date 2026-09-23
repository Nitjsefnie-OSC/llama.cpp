#!/usr/bin/env python3
"""Validate retained044 inspection logs. No GPU work and no throughput inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys

GRAPH_TOOL = Path(__file__).with_name("bonsai-cuda-graph-stats.py")
GRAPH = runpy.run_path(str(GRAPH_TOOL))
SOURCE_SHA256 = "b054bcbcfa51d69e34f28f72c0c9c1147e4aa0190c2d0a812157617c03c998f4"
# Enum/flag values below are bound to this diagnostic source contract, not arbitrary GGML versions.
MARKER = re.compile(r"(?<![A-Za-z0-9_])(CUDA_SWIGLU_FWHT|CUDA_GRAPH_STATS)\b")
ROLES = ("glu", "mul", "reshape", "output", "gate", "up", "signs", "hadamard")
COMMON = set("event invocation uid index".split())
SCHEMAS = {
    "candidate": COMMON | set("observation key mode device cc compiled_cc tokens eligible reason links plain shape f32 contiguous allocated views_safe extra_sources fusion_disabled can_fuse ranges gate_overlap up_overlap signs_overlap hadamard_overlap".split()),
    "tensor": COMMON | set("role tensor data type op ne nb src0 src1 view_src flags uses metadata contiguous buffer range_safe range".split()),
    "null": COMMON | {"role", "tensor"},
    "view": COMMON | set("role depth tensor parent constant in_pattern".split()),
    "graph": set("event observation invocation uid key mode device candidates eligible rejected".split()),
}
FACTS = set("links plain shape f32 contiguous allocated views_safe extra_sources fusion_disabled can_fuse ranges gate_overlap up_overlap signs_overlap hadamard_overlap metadata buffer range_safe constant".split())
STRICT_FACTS = set("links plain shape views_safe extra_sources fusion_disabled constant".split())
LIMITS = [
    "All logged inspections, including startup and replay, are included; none are kernel submissions.",
    "Interval facts are independently checked. Unlogged external graph edges and canonical predicate internals cannot be reconstructed.",
    "Log consistency does not establish complete collection, binary provenance, actual fusion dispatch, correctness of requests or a performance gain.",
    "Diagnostic source/binary provenance must be bound independently to the expected source hash.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, signed=False):
    require(bool(re.fullmatch(r"(?:0|[1-9][0-9]*|-?[1-9][0-9]*)" if signed else r"(?:0|[1-9][0-9]*)", value)), "noncanonical integer: " + value)
    n = int(value)
    require(-(1 << 63) <= n < (1 << (63 if signed else 64)), "integer out of range")
    return n


def pointer(value):
    if value in ("null", "(nil)"):
        return 0
    require(bool(re.fullmatch(r"(?:0x)?[0-9a-fA-F]{1,16}", value)), "invalid pointer: " + value)
    return int(value, 16)


def fields(line, marker):
    payload = line[marker.end():].strip()
    require(payload.startswith(","), "missing marker delimiter")
    # Split only field boundaries, retaining delimiters inside a field for its own grammar.
    result = {}
    for part in re.split(r",(?=[a-z][a-z0-9_]*=)", payload[1:]):
        key, sep, value = part.partition("=")
        require(sep and value and key not in result, "missing/duplicate field")
        result[key] = value
    return result


def f32_contiguous(ne, nb):
    # ggml_is_contiguous_m_n(tensor, 0, 4): singleton strides are immaterial.
    expected = 4
    for extent, stride in zip(ne, nb):
        if extent != 1 and stride != expected:
            return False
        expected *= extent
    return True


def normalize(raw):
    r = dict(raw)
    event = r.get("event")
    schema = "null" if event == "tensor" and r.get("tensor") == "null" else event
    require(schema in SCHEMAS and set(r) == SCHEMAS[schema], "unknown event or missing/extra fields")
    for key in set(r) & set("invocation uid index device depth candidates rejected flags".split()):
        r[key] = integer(r[key])
    for key in set(r) & set("cc compiled_cc tokens type op in_pattern".split()):
        r[key] = integer(r[key], signed=True)
    for key in set(r) & set("index device depth candidates rejected flags cc compiled_cc type op in_pattern".split()):
        require(-(1 << 31) <= r[key] < (1 << 31), "32-bit field out of range: " + key)
    for key in set(r) & FACTS:
        require(r[key] in ("true", "false", "unchecked"), "invalid fact: " + key)
        require(key not in STRICT_FACTS or r[key] != "unchecked", "unchecked Boolean: " + key)
    if "eligible" in r:
        if event == "graph":
            r["eligible"] = integer(r["eligible"])
            require(r["eligible"] < (1 << 31), "eligible count out of range")
        else:
            require(r["eligible"] in ("true", "false"), "invalid eligibility")
    for key in set(r) & set("key tensor data src0 src1 view_src parent".split()):
        r[key] = pointer(r[key])
    if event in ("candidate", "graph"):
        require(r["observation"] == "inspection" and r["mode"] in GRAPH["MODES"], "not an inspection/valid mode")
    if "role" in r:
        require(r["role"] in ROLES, "unknown tensor role")
    if schema == "tensor":
        require(r["tensor"] != 0, "non-null tensor record has null identity")
        for key in ("ne", "nb"):
            r[key] = [integer(x, signed=key == "ne") for x in r[key].split(":")]
            require(len(r[key]) == 4, "expected four dimensions/strides")
        if r["uses"] != "unchecked":
            r["uses"] = integer(r["uses"])
        if r["type"] == 0:
            valid = min(r["ne"]) > 0 and math.prod(r["ne"])*4 <= (1 << 63)-1 and 4+sum((n-1)*s for n,s in zip(r["ne"],r["nb"])) <= (1 << 63)-1
            require(r["metadata"] == str(valid).lower(), "metadata validity contradicts dimensions/strides")
        else:
            require(r["metadata"] == "unchecked", "non-F32 metadata unexpectedly checked")
        require(r["metadata"] == "true" or r["contiguous"] == "unchecked", "invalid metadata claims contiguity")
        if r["metadata"] == "true":
            require(r["contiguous"] == str(f32_contiguous(r["ne"], r["nb"])).lower(), "contiguity contradicts F32 dimensions/strides")
        if r["range"] != "unchecked":
            r["range"] = [integer(x) for x in r["range"].split(":")]
            require(len(r["range"]) == 2 and 0 < r["range"][0] < r["range"][1] <= (1 << 63)-1, "invalid half-open range")
        require((r["range_safe"] == "true") == (r["range"] != "unchecked"), "range safety contradicts interval")
        if r["range_safe"] == "true":
            require(r["metadata"] == r["buffer"] == "true" and r["type"] == 0 and min(r["ne"]) > 0, "unsafe metadata marked safe")
            size = 4 + sum((n-1)*s for n, s in zip(r["ne"], r["nb"]))
            require(r["range"] == [r["data"], r["data"]+size], "interval contradicts F32 data/strides")
    if event == "view":
        require(r["tensor"] != 0 and r["in_pattern"] >= -1, "invalid view identity")
    return r


def reason(c):
    failures = [(c["cc"] != 860 or c["compiled_cc"] != 860, "not_sm86"),
                (c["fusion_disabled"] == "true", "fusion_disabled")]
    failures += [(c[k] != "true", why) for k, why in [("plain", "not_plain_split_swiglu"), ("links", "broken_links_or_hint")]]
    failures += [(c["extra_sources"] == "true", "extra_sources"), (c["f32"] != "true", "not_f32_or_missing_tensor"),
                 (c["shape"] != "true", "different_shape"), (c["tokens"] not in (508,512), "other_token_width"),
                 (c["contiguous"] != "true", "noncontiguous_or_invalid_metadata"),
                 (c["allocated"] != "true", "unsupported_or_unallocated_buffer"),
                 (c["views_safe"] != "true", "invalid_view_chain"), (c["can_fuse"] != "true", "canonical_subgraph_rejected"),
                 (c["ranges"] == "unchecked", "unsafe_or_unchecked_range"), (c["ranges"] == "false", "canonical_memory_overlap")]
    overlaps = [c[r+"_overlap"] for r in ROLES[4:]]
    failures += [("unchecked" in overlaps, "unchecked_external_range")]
    failures += [(c[r+"_overlap"] == "true", r+"_output_overlap") for r in ROLES[4:]]
    return next((why for failed, why in failures if failed), "eligible")


def all_facts(values):
    return "false" if "false" in values else "unchecked" if "unchecked" in values else "true"


def validate_candidate(item):
    c, tensors, views = item["record"], item["tensors"], item["views"]
    require(set(tensors) == set(ROLES), "missing tensor roles")
    core = [tensors[r] for r in ROLES[:4]]
    require(all(t["tensor"] for t in core) and len({t["tensor"] for t in core}) == 4, "aliased/null graph-node identities")
    require([t["op"] for t in core] == [100,7,36,29], "op tags contradict four-node source pattern")
    for a in tensors.values():
        for b in tensors.values():
            if a["tensor"] and a["tensor"] == b["tensor"]:
                require(all(a[k] == b[k] for k in ("data","type","op","ne","nb","src0","src1","view_src","flags")), "same tensor identity has contradictory facts")
    require(c["tokens"] == tensors["glu"]["ne"][1], "token width contradicts GLU")
    shapes = {r:[17408,c["tokens"],1,1] for r in ("glu","mul","gate","up")}
    shapes.update(signs=[17408,1,1,1],hadamard=[1024,1024,1,1],reshape=[1024,17*c["tokens"],1,1],output=[1024,17*c["tokens"],1,1])
    shape = 0 < c["tokens"] <= ((1 << 63)-1)//17 and all(tensors[r].get("ne") == ne for r,ne in shapes.items())
    require(c["shape"] == str(shape).lower(), "shape flag contradicts tensors")
    require(c["reason"] == reason(c) and (c["eligible"] == "true") == (c["reason"] == "eligible"), "eligibility/reason contradicts facts")
    for key in ("f32", "contiguous", "allocated"):
        values = []
        for t in tensors.values():
            values.append("unchecked" if not t["tensor"] else
                          (str(t["type"] == 0).lower() if key == "f32" else t["contiguous"] if key == "contiguous" else
                           str(bool(t["data"]) and t["buffer"] == "true").lower()))
        require(c[key] == all_facts(values), "aggregate contradicts tensor facts: " + key)
    for role, t in tensors.items():
        chain = views.get(role, [])
        current = t.get("view_src", 0)
        seen = set()
        for depth, v in enumerate(chain):
            require(v["depth"] == depth and v["tensor"] == current and current not in seen, "broken/duplicated view chain")
            seen.add(current)
            indices = {tensors[r]["tensor"]: c["index"]+j for j,r in enumerate(ROLES[:4])}
            require(v["in_pattern"] == indices.get(current,-1), "view membership contradicts identity")
            if c["can_fuse"] == "true" and role in ROLES[:3]:
                require(v["in_pattern"] != -1 or v["constant"] == "true", "canonical view claim contradicts ancestry")
            current = v["parent"]
        require(current == 0, "missing/truncated view ancestry")
        if role in ROLES[:4]:
            require(isinstance(t["uses"],int), "unchecked graph-node consumer count")
            if c["can_fuse"] == "true":
                require(t["flags"] & 16 and (role == "output" or not t["flags"] & 2), "canonical output/compute flags contradict predicate")
    output = tensors["output"]
    for role in ROLES[4:]:
        t = tensors[role]
        observed = c[role+"_overlap"]
        if output.get("range_safe") == t.get("range_safe") == "true":
            a,b = output["range"],t["range"]
            require(observed == str(a[0] < b[1] and b[0] < a[1]).lower(), "wrong external overlap: " + role)
        else:
            require(observed == "unchecked", "unchecked interval claimed safe: " + role)
    if c["links"] == "true":
        require(tensors["mul"]["src0"] == tensors["glu"]["tensor"] and tensors["reshape"]["src0"] == tensors["mul"]["tensor"] and output["src1"] == tensors["reshape"]["tensor"], "broken linkage claimed true")
    require(tensors["glu"]["src0"] == tensors["gate"]["tensor"] and tensors["glu"]["src1"] == tensors["up"]["tensor"] and tensors["mul"]["src1"] == tensors["signs"]["tensor"] and output["src0"] == tensors["hadamard"]["tensor"], "external role identity mismatch")
    if c["eligible"] == "true":
        require(all(t["range_safe"] == "true" for t in tensors.values()), "eligible with unchecked tensor range")
        require(all(tensors[r]["uses"] == 1 for r in ROLES[:3]), "eligible intermediate has extra/missing consumers")


def validate_inspection_facts(g):
    # Tensor identities and graph indices belong to one inspection, not one candidate.
    known, nodes, ancestry = {}, {}, {}
    intrinsic = ("data", "type", "op", "ne", "nb", "src0", "src1", "view_src", "flags",
                 "metadata", "contiguous", "buffer", "range_safe", "range")
    for item in g["candidates"].values():
        for t in item["tensors"].values():
            ptr = t["tensor"]
            if ptr:
                if ptr in known:
                    require(all(t[k] == known[ptr][k] for k in intrinsic), "inspection tensor identity has contradictory facts")
                else:
                    known[ptr] = t
        for offset, role in enumerate(ROLES[:4]):
            t = item["tensors"][role]
            index = item["record"]["index"] + offset
            fact = (t["tensor"], t["op"])
            require(index not in nodes or nodes[index] == fact, "overlapping graph index has contradictory tensor/op")
            nodes[index] = fact
    for item in g["candidates"].values():
        for chain in item["views"].values():
            for v in chain:
                ptr, parent = v["tensor"], v["parent"]
                if ptr in known:
                    require(parent == known[ptr]["view_src"], "view parent contradicts known tensor view_src")
                fact = (parent, v["constant"])
                require(ptr not in ancestry or ancestry[ptr] == fact, "same view identity has contradictory ancestry")
                ancestry[ptr] = fact


def analyze(lines):
    stats = GRAPH["read_records"](lines)
    stat_iter = iter(stats)
    pending, inspections = [], {}
    for lineno, line in enumerate(lines,1):
        match = MARKER.search(line)
        if not match:
            continue
        try:
            raw = fields(line,match)
            if match[1] == "CUDA_GRAPH_STATS":
                s = next(stat_iter)
                if s["event"] == "execute":
                    for key in ("uid","device","tokens"):
                        require(integer(raw[key]) == s[key], "invalid execute identity")
                    s = dict(s,key=pointer(s["key"]),line=lineno)
                    pending.append(s)
                continue
            r = normalize(raw)
            inv = r["invocation"]
            if inv not in inspections:
                require(r["event"] in ("candidate","graph"), "orphan tensor/view")
                matches = [s for s in pending if all(s[k] == r[k] for k in ("uid","key","mode","device"))]
                require(len(matches) == 1, "missing/ambiguous matching execute")
                execution = matches[0]
                pending.remove(execution)
                inspections[inv] = dict(invocation=inv,uid=r["uid"],key=r["key"],mode=r["mode"],device=r["device"],execute=execution,candidates={},graph=None)
            g = inspections[inv]
            require(r["uid"] == g["uid"] and g["graph"] is None, "cross-UID or record after final graph")
            if r["event"] in ("candidate","graph"):
                require(all(r[k] == g[k] for k in ("key","mode","device")), "cross-key/mode/device")
            if r["event"] == "graph":
                g["graph"] = r
            elif r["event"] == "candidate":
                require(r["index"] not in g["candidates"], "duplicate final candidate")
                g["candidates"][r["index"]] = dict(record=r,tensors={},views={})
            else:
                require(r["index"] in g["candidates"], "orphan/cross-index tensor or view")
                item = g["candidates"][r["index"]]
                role = r["role"]
                if r["event"] == "tensor":
                    require(role not in item["tensors"], "duplicate tensor role")
                    item["tensors"][role] = r
                else:
                    require(role in item["tensors"], "view precedes tensor")
                    item["views"].setdefault(role,[]).append(r)
        except (ValueError,StopIteration) as error:
            raise ValueError(f"line {lineno}: {error}") from error
    require(inspections and not pending, "missing inspections or unmatched execute records")
    require(sorted(inspections) == list(range(len(inspections))), "missing startup/inspection invocation")
    counts = Counter()
    for g in inspections.values():
        require(g["graph"] is not None, "truncated inspection: no final graph")
        for item in g["candidates"].values():
            validate_candidate(item)
            c = item["record"]
            counts[(c["tokens"],c["mode"],c["reason"],c["eligible"])] += 1
        validate_inspection_facts(g)
        eligible = sum(c["record"]["eligible"] == "true" for c in g["candidates"].values())
        require((g["graph"]["candidates"],g["graph"]["eligible"],g["graph"]["rejected"]) == (len(g["candidates"]),eligible,len(g["candidates"])-eligible), "graph totals disagree")
    return dict(passed=True,inspection_count=len(inspections),candidate_count=sum(counts.values()),
                counts=[dict(tokens=t,mode=m,reason=r,eligible=e=="true",count=n) for (t,m,r,e),n in sorted(counts.items())],
                inspections=list(inspections.values()),graph_stats=GRAPH["summarize"](stats),limitations=LIMITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    try:
        with args.output.open("x",encoding="utf8") as output:
            provenance = dict(log=str(args.log.resolve()),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              graph_tool_sha256=hashlib.sha256(GRAPH_TOOL.read_bytes()).hexdigest(),
                              expected_diagnostic_source_sha256=SOURCE_SHA256,source_binding_verified=False)
            try:
                data = args.log.read_bytes()
                provenance["log_sha256"] = hashlib.sha256(data).hexdigest()
                result = analyze(data.decode("utf8").splitlines())
            except (ValueError,OSError,UnicodeError) as error:
                result = dict(passed=False,error=str(error),limitations=LIMITS)
            result["provenance"] = provenance
            json.dump(result,output,indent=2)
            output.write("\n")
        print(json.dumps({k:result[k] for k in ("passed","inspection_count","candidate_count","error") if k in result}))
        return 0 if result["passed"] else 2
    except OSError as error:
        print(str(error),file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
