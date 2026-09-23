#!/usr/bin/env python3
"""Synthetic process/HTTP facts only; never contacts a service or the GPU."""
import argparse
import copy
from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

SPEC = importlib.util.spec_from_file_location("provenance", Path(__file__).with_name("bonsai-server-provenance.py"))
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stable = self.root/"tools/llamacpp-prism"
        self.snapshot = self.root/"snapshot"
        self.stable.mkdir(parents=True); self.snapshot.mkdir()
        self.manifest = {}
        for name in sorted(P.RUNTIME):
            for directory in (self.stable, self.snapshot):
                (directory/name).write_bytes(name.encode())
            self.manifest[name] = P.digest(self.stable/name)
        model = self.root/"models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
        model.parent.mkdir(); model.write_bytes(b"synthetic model path only")
        self.deployment = dict(status="deployed", error=None, destination=str(self.stable),
                               files=[dict(name=n, sha256=h.upper()) for n,h in self.manifest.items()])
        self.process = dict(Pid=123, CimPid=123, ParentPid=99, CommandLine="synthetic llama-server.exe command",
                            StartTimeUtc="2026-09-23T05:52:40.4126348Z", VerifiedStartTimeUtc="2026-09-23T05:52:40.4126348Z",
                            AliveAfter=True, Executable=str(self.stable/"llama-server.exe"),
                            Modules=[dict(Name=n, Path=str(self.stable/n)) for n in sorted(P.LOADED)],
                            Listeners=[dict(LocalAddress="0.0.0.0", LocalPort=8090, OwningProcess=123)])
        self.http = {"/health":dict(status=200, body=dict(status="ok")),
                     "/props":dict(status=200, body=dict(model_alias="bonsai-2-27b", model_path=str(model), total_slots=4,
                                                        default_generation_settings=dict(n_ctx=188416))),
                     "/slots":dict(status=200, body=[dict(id=i, n_ctx=188416, is_processing=False) for i in range(4)])}
        self.args = argparse.Namespace(pid=123, root=self.root, snapshot_dir=self.snapshot,
                                      snapshot_manifest=self.root/"manifest.json", deployment_receipt=self.root/"deployment.json",
                                      output=self.root/"result.json")
        self.args.snapshot_manifest.write_text(json.dumps(self.manifest))
        self.args.deployment_receipt.write_text(json.dumps(self.deployment))

    def collect(self):
        evidence = {}
        def http(route, rows):
            rows[route] = copy.deepcopy(self.http[route])
        with patch.object(P, "inspect_process", return_value=self.process), patch.object(P, "read_http", side_effect=http):
            P.collect(self.args, evidence)
        return evidence

    def test_complete_synthetic_positive(self):
        evidence = self.collect()
        self.assertEqual(evidence["stable_runtime_hashes"], self.manifest)
        self.assertEqual(evidence["snapshot_runtime_hashes"], self.manifest)
        self.assertEqual(set(evidence["loaded_runtime_modules"]), P.LOADED)
        self.assertEqual(evidence["http"], self.http)
        self.assertEqual(evidence["snapshot_manifest_sha256"], P.digest(self.args.snapshot_manifest))

    def test_manifest_missing_extra_and_invalid_hash(self):
        for change in ("missing", "extra", "hash", "escape"):
            m=dict(self.manifest)
            if change=="missing":m.pop("ggml-cuda.dll")
            elif change=="extra":m["x.dll"]="0"*64
            elif change=="escape":m["../ggml-cuda.dll"]=m.pop("ggml-cuda.dll")
            else:m["ggml-cuda.dll"]="g"*64
            with self.subTest(change=change), self.assertRaises(ValueError):P.validate_manifest(m)

    def test_duplicate_json_key(self):
        with self.assertRaisesRegex(ValueError,"duplicate JSON key"):
            P.decode_json('{"ggml-cuda.dll":"a","ggml-cuda.dll":"b"}')

    def test_deployment_status_destination_count_duplicate_hash(self):
        for change in ("status", "destination", "count", "duplicate", "hash"):
            d=copy.deepcopy(self.deployment)
            if change=="status":d["status"]="failed"
            elif change=="destination":d["destination"]=str(self.snapshot)
            elif change=="count":d["files"].pop()
            elif change=="duplicate":d["files"][-1]=d["files"][0]
            else:d["files"][0]["sha256"]="0"*64
            with self.subTest(change=change), self.assertRaises(ValueError):P.validate_deployment(d,self.stable,self.manifest)

    def test_process_identity_races(self):
        for field,value in [("Pid",124),("CimPid",124),("AliveAfter",False),
                            ("VerifiedStartTimeUtc","2026-09-23T05:52:41Z"),("StartTimeUtc",""),
                            ("ParentPid",None),("CommandLine",None)]:
            p=copy.deepcopy(self.process);p[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):P.validate_process(p,123,self.stable)

    def test_wrong_executable(self):
        self.process["Executable"]=str(self.snapshot/"llama-server.exe")
        with self.assertRaisesRegex(ValueError,"executable"):P.validate_process(self.process,123,self.stable)

    def test_module_missing_duplicate_foreign_path(self):
        for change in ("missing", "duplicate", "path", "relative", "extra_runtime"):
            p=copy.deepcopy(self.process)
            if change=="missing":p["Modules"].pop()
            elif change=="duplicate":p["Modules"].append(p["Modules"][0])
            elif change=="path":p["Modules"][0]["Path"]=str(self.snapshot/p["Modules"][0]["Name"])
            elif change=="relative":p["Modules"][0]["Path"]=p["Modules"][0]["Name"]
            else:p["Modules"].append(dict(Name="llama-bench-impl.dll",Path=str(self.stable/"llama-bench-impl.dll")))
            with self.subTest(change=change), self.assertRaises(ValueError):P.validate_process(p,123,self.stable)

    def test_unrelated_system_modules_allowed(self):
        self.process["Modules"].append(dict(Name="ntdll.dll",Path="synthetic-system-module"))
        self.assertEqual(set(P.validate_process(self.process,123,self.stable)),P.LOADED)

    def test_listener_identity(self):
        for listeners in ([], [dict(LocalPort=8090,OwningProcess=124)], [dict(LocalPort=8091,OwningProcess=123)]):
            self.process["Listeners"]=listeners
            with self.assertRaisesRegex(ValueError,"listener"):P.validate_process(self.process,123,self.stable)

    def test_file_hash_mismatch_preserved(self):
        (self.snapshot/"ggml-cuda.dll").write_bytes(b"wrong")
        with patch.object(P,"inspect_process") as inspect, self.assertRaisesRegex(ValueError,"backing file hash mismatch"):
            P.collect(self.args,{})
        inspect.assert_not_called()

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,"unexpected runtime filename"):P.confined(self.stable,"../outside.dll")
        with patch.object(P,"absolute",return_value=self.snapshot/"ggml-cuda.dll"):
            with self.assertRaisesRegex(ValueError,"escapes directory"):P.confined(self.stable,"ggml-cuda.dll")
        with self.assertRaisesRegex(ValueError,"absolute path required"):P.absolute(Path("relative"))

    def test_http_wrong_model_capacity_busy_and_missing(self):
        for change in ("alias", "path", "slots", "context", "busy", "duplicate", "missing", "status"):
            h=copy.deepcopy(self.http)
            if change=="alias":h["/props"]["body"]["model_alias"]="other"
            elif change=="path":h["/props"]["body"]["model_path"]=str(self.args.snapshot_manifest)
            elif change=="slots":h["/props"]["body"]["total_slots"]=3
            elif change=="context":h["/slots"]["body"][2]["n_ctx"]=1024
            elif change=="busy":h["/slots"]["body"][2]["is_processing"]=True
            elif change=="duplicate":h["/slots"]["body"][2]["id"]=1
            elif change=="missing":del h["/slots"]
            else:h["/health"]["status"]=503
            with self.subTest(change=change), self.assertRaises(ValueError):P.validate_http(h,self.root)

    def test_process_query_environment_and_evidence(self):
        raw=json.dumps(self.process).encode(); e={}
        with patch.dict(P.os.environ,{"PSModulePath":"remove","pSmOdUlEpAtH":"remove2"}), patch.object(P.subprocess,"run",return_value=subprocess.CompletedProcess([],0,raw,b"warning")) as run:
            self.assertEqual(P.inspect_process(123,e),self.process)
            kw=run.call_args.kwargs
            self.assertFalse(any(k.upper()=="PSMODULEPATH" for k in kw["env"]))
            self.assertEqual(kw["stdin"],subprocess.DEVNULL)
            self.assertEqual(kw["timeout"],30)
        self.assertEqual(e["process_query"]["exit_code"],0)
        self.assertEqual(e["process_query"]["stderr"],"warning")

    def test_process_failure_and_timeout_preserve_evidence(self):
        e={}
        with patch.object(P.subprocess,"run",return_value=subprocess.CompletedProcess([],7,b"partial",b"denied")):
            with self.assertRaises(ValueError):P.inspect_process(123,e)
        self.assertEqual(e["process_query"]["exit_code"],7)
        self.assertEqual(e["process_query"]["stdout"],"partial")
        e={}
        with patch.object(P.subprocess,"run",side_effect=subprocess.TimeoutExpired([],30,output=b"partial",stderr=b"slow")):
            with self.assertRaises(subprocess.TimeoutExpired):P.inspect_process(123,e)
        self.assertIsNone(e["process_query"]["exit_code"])
        self.assertEqual(P.base64.b64decode(e["process_query"]["stdout_base64"]),b"partial")

    def test_http_error_retains_full_body(self):
        body=b'{"error":"busy","details":[1,2,3]}'
        error=urllib.error.HTTPError("http://127.0.0.1:8090/slots",503,"busy",Message(),io.BytesIO(body));e={}
        with patch.object(P.urllib.request,"urlopen",side_effect=error) as request:
            P.read_http("/slots",e)
        self.assertEqual(request.call_args.kwargs["timeout"],10)
        self.assertEqual(e["/slots"]["status"],503)
        self.assertEqual(e["/slots"]["body"]["details"],[1,2,3])
        self.assertEqual(P.base64.b64decode(e["/slots"]["raw_body_base64"]),body)

    def test_http_malformed_body_preserved(self):
        body=b"upstream diagnostic, not JSON"
        error=urllib.error.HTTPError("http://127.0.0.1:8090/slots",503,"busy",Message(),io.BytesIO(body));e={}
        with patch.object(P.urllib.request,"urlopen",side_effect=error):
            with self.assertRaises(ValueError):P.read_http("/slots",e)
        self.assertEqual(P.base64.b64decode(e["/slots"]["raw_body_base64"]),body)
        self.assertIn("error",e["/slots"])

    def test_cli_success_and_busy_preserve_complete_facts(self):
        for busy in (False,True):
            output=self.root/("busy.json" if busy else "healthy.json")
            self.http["/slots"]["body"][2]["is_processing"]=busy
            def http(route, rows):rows[route]=copy.deepcopy(self.http[route])
            argv=["--pid","123","--root",str(self.root),"--snapshot-manifest",str(self.args.snapshot_manifest),
                  "--snapshot-dir",str(self.snapshot),"--deployment-receipt",str(self.args.deployment_receipt),"--output",str(output)]
            with patch.object(P,"inspect_process",return_value=self.process),patch.object(P,"read_http",side_effect=http):
                self.assertEqual(P.main(argv),2 if busy else 0)
            result=json.loads(output.read_bytes())
            self.assertEqual(result["passed"],not busy)
            self.assertEqual(result["http"],self.http)
            self.assertEqual(result["process"],self.process)
            self.assertEqual(set(result["loaded_runtime_modules"]),P.LOADED)

    def test_cli_failure_json_and_exclusive_output(self):
        argv=["--pid","0","--root",str(self.root),"--snapshot-manifest",str(self.args.snapshot_manifest),
              "--snapshot-dir",str(self.snapshot),"--deployment-receipt",str(self.args.deployment_receipt),"--output",str(self.args.output)]
        with patch.object(P,"inspect_process") as inspect, patch.object(P,"read_http") as http:
            self.assertEqual(P.main(argv),2)
            before=self.args.output.read_bytes()
            self.assertEqual(P.main(argv),2)
            self.assertEqual(self.args.output.read_bytes(),before)
            inspect.assert_not_called();http.assert_not_called()
        result=json.loads(before)
        self.assertFalse(result["passed"]);self.assertIn("invalid Win32 PID",result["error"])
        self.assertEqual(result["tool_sha256"],P.digest(P.__file__))


if __name__ == "__main__":
    unittest.main()
