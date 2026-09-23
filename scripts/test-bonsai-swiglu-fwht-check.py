#!/usr/bin/env python3
"""Explicitly synthetic044 logs; no CUDA or service requests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("bonsai-swiglu-fwht-check.py")
SPEC = importlib.util.spec_from_file_location("swiglu_check", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def fixture(invocation=0, mode="direct", tokens=508):
    """Synthetic identities/layouts are intentionally not an actual service capture."""
    uid, index, key = 17, 12, "000000000000ABCD"
    common = dict(invocation=invocation, uid=uid, index=index)
    candidate = dict(event="candidate",observation="inspection",**common,key=key,mode=mode,device=0,cc=860,compiled_cc=860,
                     tokens=tokens,eligible="true",reason="eligible",links="true",plain="true",shape="true",f32="true",
                     contiguous="true",allocated="true",views_safe="true",extra_sources="false",fusion_disabled="false",
                     can_fuse="true",ranges="true",gate_overlap="false",up_overlap="false",signs_overlap="false",hadamard_overlap="false")
    records = [("CUDA_GRAPH_STATS",dict(event="execute",time_us=invocation+1,device=0,key=key,uid=uid,host_us=0,idle_us=0,tokens=tokens,mode=mode,reason="stable",node=-1)),
               ("CUDA_SWIGLU_FWHT",candidate)]
    identities = {r:f"{0x100+i:016X}" for i,r in enumerate(CHECK.ROLES)}
    for n,role in enumerate(CHECK.ROLES):
        ne = [17408,tokens,1,1]
        if role in ("reshape","output"):
            ne = [1024,17*tokens,1,1]
        elif role == "signs":
            ne = [17408,1,1,1]
        elif role == "hadamard":
            ne = [1024,1024,1,1]
        nb = [4,4*ne[0],4*ne[0]*ne[1],4*ne[0]*ne[1]]
        address = 0x10000000+n*0x4000000
        t = dict(event="tensor",**common,role=role,tensor=identities[role],data=f"{address:016X}",type=0,
                 op=[100,7,36,29,29,29,0,0][n],ne=":".join(map(str,ne)),nb=":".join(map(str,nb)),
                 src0="0000000000000000",src1="0000000000000000",view_src="0000000000000000",flags=16 if n<4 else 0,
                 uses=1 if n<4 else "unchecked",metadata="true",contiguous="true",buffer="true",range_safe="true",
                 range=f"{address}:{address+4*ne[0]*ne[1]}")
        if role == "glu":
            t.update(src0=identities["gate"],src1=identities["up"])
        elif role == "mul":
            t.update(src0=identities["glu"],src1=identities["signs"])
        elif role == "reshape":
            t.update(src0=identities["mul"],view_src=identities["mul"])
        elif role == "output":
            t.update(src0=identities["hadamard"],src1=identities["reshape"])
        records.append(("CUDA_SWIGLU_FWHT",t))
        if role == "reshape":
            records.append(("CUDA_SWIGLU_FWHT",dict(event="view",**common,role=role,depth=0,tensor=identities["mul"],parent="0000000000000000",constant="false",in_pattern=index+1)))
    records.append(("CUDA_SWIGLU_FWHT",dict(event="graph",observation="inspection",invocation=invocation,uid=uid,key=key,mode=mode,device=0,candidates=1,eligible=1,rejected=0)))
    return records


def lines(records):
    return ["SYNTHETIC FIXTURE - NOT A SERVICE CAPTURE"] + ["[server prefix] "+m+","+",".join(f"{k}={v}" for k,v in r.items()) for m,r in records]


def tensor(records,role):
    return next(r for _,r in records if r["event"] == "tensor" and r["role"] == role)


def view(records):
    return next(r for _,r in records if r["event"] == "view")


def second_pattern(records, shift=4):
    """Distinct graph nodes may share external tensors and allocator storage."""
    extra = copy.deepcopy(records[1:-1])
    mapping = {tensor(records, r)["tensor"]: f"{0x200+i:016X}" for i,r in enumerate(CHECK.ROLES[:4])}
    for _, row in extra:
        row["index"] += shift
        if row["event"] == "view" and row["in_pattern"] != -1:
            row["in_pattern"] += shift
        for key in ("tensor", "src0", "src1", "view_src", "parent"):
            if row.get(key) in mapping:
                row[key] = mapping[row[key]]
    records[-1:-1] = extra
    records[-1][1].update(candidates=2, eligible=2, rejected=0)
    return records


class ValidatorTests(unittest.TestCase):
    def result(self,records):
        return CHECK.analyze(lines(records))

    def fails(self,records):
        with self.assertRaises(ValueError):
            self.result(records)

    def test_all_modes_include_replays(self):
        records = sum([fixture(i,m) for i,m in enumerate(["direct","capture","replay","replay"])],[])
        result = self.result(records)
        self.assertEqual(result["inspection_count"],4)
        self.assertEqual(result["candidate_count"],4)
        self.assertEqual(sum(x["count"] for x in result["counts"] if x["mode"]=="replay"),2)
        self.assertEqual(len(result["inspections"][0]["candidates"][12]["tensors"]),8)

    def test_512(self):
        self.assertTrue(self.result(fixture(tokens=512))["passed"])

    def test_stride_claim_matches_actual_f32_layout(self):
        r=fixture();t=tensor(r,"gate");nb=list(map(int,t["nb"].split(":")));nb[0]=8
        t["nb"]=":".join(map(str,nb));ne=list(map(int,t["ne"].split(":")));start=int(t["data"],16)
        t["range"]=f"{start}:{start+4+sum((n-1)*s for n,s in zip(ne,nb))}"
        with self.assertRaisesRegex(ValueError,"contiguity contradicts"):
            self.result(r)
        t["contiguous"]="false"
        r[1][1].update(contiguous="false",eligible="false",reason="noncontiguous_or_invalid_metadata")
        r[-1][1].update(eligible=0,rejected=1)
        self.assertTrue(self.result(r)["passed"])

    def test_singleton_strides_are_immaterial(self):
        r=fixture()
        for _,t in r:
            if t["event"] == "tensor":
                ne=list(map(int,t["ne"].split(":")));nb=list(map(int,t["nb"].split(":")))
                t["nb"]=":".join(str(123+j if n==1 else nb[j]) for j,n in enumerate(ne))
        self.assertTrue(self.result(r)["passed"])
        # Exercise the ne0==1 exception through the actual record normalizer.
        t=copy.deepcopy(tensor(fixture(),"gate"))
        t.update(ne="1:2:1:1",nb="99:4:77:88")
        start=int(t["data"],16);t["range"]=f"{start}:{start+8}"
        self.assertEqual(CHECK.normalize({k:str(v) for k,v in t.items()})["contiguous"],"true")
        t["nb"]="99:8:77:88";t["range"]=f"{start}:{start+12}"
        with self.assertRaisesRegex(ValueError,"contiguity contradicts"):
            CHECK.normalize({k:str(v) for k,v in t.items()})

    def test_view_parent_must_match_known_tensor(self):
        r=fixture();parent=tensor(r,"gate")["tensor"];view(r)["parent"]=parent
        v=copy.deepcopy(view(r));v.update(depth=1,tensor=parent,parent="0",constant="true",in_pattern=-1)
        pos=next(i for i,(_,row) in enumerate(r) if row["event"]=="view")
        r.insert(pos+1,("CUDA_SWIGLU_FWHT",v))
        with self.assertRaisesRegex(ValueError,"view parent contradicts"):
            self.result(r)

    def test_consistent_known_output_view(self):
        r=fixture();tensor(r,"output")["view_src"]=tensor(r,"gate")["tensor"]
        v=copy.deepcopy(view(r));v.update(role="output",tensor=tensor(r,"gate")["tensor"],in_pattern=-1)
        r.insert(-1,("CUDA_SWIGLU_FWHT",v))
        self.assertTrue(self.result(r)["passed"])

    def test_overlapping_graph_indices_conflict(self):
        r=fixture();extra=copy.deepcopy(r[1:-1])
        for _,row in extra:
            row["index"]+=1
            if row["event"]=="view":row["in_pattern"]+=1
        r[-1:-1]=extra;r[-1][1].update(candidates=2,eligible=2,rejected=0)
        with self.assertRaisesRegex(ValueError,"overlapping graph index"):
            self.result(r)
        # Also reject overlap when identities themselves were not reused.
        for shift in (1,2,3):
            with self.subTest(shift=shift):
                self.fails(second_pattern(fixture(),shift))

    def test_disjoint_patterns_share_leaves_and_reuse_storage(self):
        self.assertEqual(self.result(second_pattern(fixture()))["candidate_count"],2)

    def test_inspection_shared_tensor_facts_cannot_disagree(self):
        r=second_pattern(fixture())
        t=next(t for _,t in r if t["event"]=="tensor" and t["role"]=="gate" and t["index"]==16)
        t["flags"]=4  # Still locally valid; only the shared identity exposes this contradiction.
        with self.assertRaisesRegex(ValueError,"inspection tensor identity"):
            self.result(r)

    def test_read_only_operands_can_share_tensor_identity(self):
        r=fixture();gate=tensor(r,"gate");up=tensor(r,"up")
        up.update({k:v for k,v in gate.items() if k!="role"});tensor(r,"glu")["src1"]=gate["tensor"]
        self.assertTrue(self.result(r)["passed"])

    def test_known_view_parent_across_candidates(self):
        r=second_pattern(fixture())
        other=next(t["tensor"] for _,t in r if t["event"]=="tensor" and t["role"]=="glu" and t["index"]==16)
        output=tensor(r,"output");output["view_src"]=other
        v=copy.deepcopy(view(r));v.update(role="output",tensor=other,in_pattern=-1)
        r.insert(-1,("CUDA_SWIGLU_FWHT",v))
        self.assertTrue(self.result(r)["passed"])
        v["parent"]=tensor(r,"gate")["tensor"]
        tail=copy.deepcopy(v);tail.update(depth=1,tensor=v["parent"],parent="0")
        r.insert(-1,("CUDA_SWIGLU_FWHT",tail))
        with self.assertRaisesRegex(ValueError,"view parent contradicts"):
            self.result(r)

    def test_zero_eligible_is_valid(self):
        r=fixture();r[1][1].update(compiled_cc=800,eligible="false",reason="not_sm86");r[-1][1].update(eligible=0,rejected=1)
        self.assertEqual(self.result(r)["counts"][0]["eligible"],False)

    def test_zero_candidates_startup(self):
        r=fixture();r=[r[0],r[-1]];r[0][1]["tokens"]=0;r[-1][1].update(candidates=0,eligible=0,rejected=0)
        self.assertEqual(self.result(r)["candidate_count"],0)

    def test_null_rejected_input(self):
        r=fixture();gate=tensor(r,"gate");gate.clear();gate.update(event="tensor",invocation=0,uid=17,index=12,role="gate",tensor="null")
        tensor(r,"glu")["src0"]="0000000000000000"
        r[1][1].update(plain="false",shape="false",f32="unchecked",contiguous="unchecked",allocated="unchecked",gate_overlap="unchecked",eligible="false",reason="not_plain_split_swiglu")
        r[-1][1].update(eligible=0,rejected=1)
        self.assertTrue(self.result(r)["passed"])

    def test_missing_each_tensor_role(self):
        for role in CHECK.ROLES:
            with self.subTest(role=role):
                self.fails([(m,r) for m,r in fixture() if not(r["event"]=="tensor" and r["role"]==role)])

    def test_duplicate_each_record(self):
        for i in range(len(fixture())):
            with self.subTest(index=i):
                r=fixture();r.insert(i,copy.deepcopy(r[i]));self.fails(r)

    def test_foreign_uid_index(self):
        for field in ("uid","index"):
            r=fixture();tensor(r,"gate")[field]=999;self.fails(r)

    def test_cross_graph_identity(self):
        for field,value in [("key","BB"),("mode","replay"),("device",1),("uid",999)]:
            r=fixture();r[-1][1][field]=value;self.fails(r)

    def test_missing_execute(self):
        self.fails(fixture()[1:])

    def test_unmatched_execute(self):
        r=fixture();r.append(copy.deepcopy(r[0]));self.fails(r)

    def test_missing_startup_or_invocation(self):
        self.fails(fixture(invocation=1))
        self.fails(fixture()+fixture(invocation=2))

    def test_noncanonical_numeric_id(self):
        for value in ["00","+0","-0","1.0","18446744073709551616"]:
            r=fixture();r[1][1]["invocation"]=value;self.fails(r)

    def test_c_integer_widths(self):
        for field in ["index","device","cc","compiled_cc"]:
            r=fixture();r[1][1][field]=1<<31;self.fails(r)

    def test_wrong_overlap(self):
        r=fixture();r[1][1]["gate_overlap"]="true";self.fails(r)

    def test_fabricated_safe_range(self):
        r=fixture();tensor(r,"gate")["range"]="1:2";self.fails(r)

    def test_unchecked_claimed_eligible(self):
        for field in ["can_fuse","ranges","gate_overlap","contiguous"]:
            r=fixture();r[1][1][field]="unchecked";self.fails(r)

    def test_false_known_flags(self):
        for field in ["shape","links","plain","allocated","views_safe"]:
            r=fixture();r[1][1][field]="false";self.fails(r)

    def test_wrong_totals(self):
        for field in ["candidates","eligible","rejected"]:
            r=fixture();r[-1][1][field]+=1;self.fails(r)

    def test_bad_views(self):
        for field,value in [("depth",1),("tensor","FF"),("parent","FF"),("in_pattern",-1),("constant","unchecked")]:
            r=fixture();view(r)[field]=value;self.fails(r)

    def test_missing_view(self):
        self.fails([(m,r) for m,r in fixture() if r["event"]!="view"])

    def test_extra_consumer(self):
        r=fixture();tensor(r,"glu")["uses"]=2;self.fails(r)

    def test_observable_intermediate(self):
        r=fixture();tensor(r,"glu")["flags"]=18;self.fails(r)

    def test_tensor_identity_alias(self):
        r=fixture();tensor(r,"mul")["tensor"]=tensor(r,"glu")["tensor"];self.fails(r)

    def test_wrong_shape_with_valid_interval(self):
        r=fixture();t=tensor(r,"signs");t["ne"]="17408:1:1:2";start=int(t["data"],16);t["range"]=f"{start}:{start+139264}";self.fails(r)

    def test_truncation(self):
        self.fails(fixture()[:-1])
        with self.assertRaises(ValueError):
            CHECK.analyze(lines(fixture())+["CUDA_SWIGLU_FWHT,event=tensor,invocation="])

    def test_duplicate_field_and_range_delimiter(self):
        for suffix in [",uid=17",",bogus=x"]:
            r=lines(fixture());r[2]+=suffix
            with self.assertRaises(ValueError): CHECK.analyze(r)
        r=fixture();tensor(r,"gate")["range"]="10,20";self.fails(r)

    def test_cli_valid_error_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/"synthetic.log";out=root/"result.json"
            log.write_text("\n".join(lines(fixture())),encoding="utf8")
            args=[sys.executable,str(SCRIPT),"--log",str(log),"--output",str(out)]
            p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            before=out.read_bytes();data=json.loads(before);self.assertFalse(data["provenance"]["source_binding_verified"])
            self.assertEqual(len(data["provenance"]["log_sha256"]),64)
            p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertEqual(out.read_bytes(),before)
            bad=root/"bad.json";log.write_text("CUDA_SWIGLU_FWHT,event=",encoding="utf8")
            p=subprocess.run(args[:-1]+[str(bad)],capture_output=True,text=True);self.assertEqual(p.returncode,2)
            self.assertFalse(json.loads(bad.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
