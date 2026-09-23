#!/usr/bin/env python3
"""Record and validate Bonsai's live process, deployed files and read-only HTTP state."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNTIME = frozenset(("llama-server.exe", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll",
                     "llama-bench-impl.dll", "llama-common.dll", "llama-perplexity-impl.dll",
                     "llama-server-impl.dll", "llama.dll", "mtmd.dll"))
LOADED = RUNTIME - {"llama-bench-impl.dll", "llama-perplexity-impl.dll"}
SCOPE = "Live module paths and hashes of backing files; not relocated in-memory code hashes. Process, file, and HTTP observations are sequential, not an atomic snapshot."
PROCESS_QUERY = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$targetId = [uint32]$env:BONSAI_PROVENANCE_PID
$p = Get-Process -Id $targetId
$c = Get-CimInstance Win32_Process -Filter "ProcessId = $targetId"
$started = $p.StartTime.ToUniversalTime().ToString('o')
$exe = $p.Path
$modules = @($p.Modules | ForEach-Object { [ordered]@{Name=$_.ModuleName;Path=$_.FileName} })
$listen = @(Get-NetTCPConnection -State Listen -LocalPort 8090 | Select-Object LocalAddress,LocalPort,OwningProcess)
$after = Get-Process -Id $targetId
[ordered]@{Pid=$p.Id;CimPid=$c.ProcessId;ParentPid=$c.ParentProcessId;CommandLine=$c.CommandLine;
 StartTimeUtc=$started;VerifiedStartTimeUtc=$after.StartTime.ToUniversalTime().ToString('o');
 AliveAfter=(-not $after.HasExited);Executable=$exe;Modules=$modules;Listeners=$listen} | ConvertTo-Json -Depth 5
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def decode_json(data):
    return json.loads(data, object_pairs_hook=unique_object)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def absolute(path):
    path = Path(path)
    require(path.is_absolute(), "absolute path required: " + str(path))
    return path.resolve(strict=True)


def confined(directory, name):
    require(name in RUNTIME, "unexpected runtime filename: " + name)
    path = absolute(directory / name)
    require(path.parent == directory, "runtime file escapes directory: " + name)
    require(path.is_file(), "runtime path is not a file: " + str(path))
    return path


def bound_path(path, expected, label):
    supplied = Path(path)
    require(supplied.is_absolute() and Path(os.path.abspath(supplied)) == expected and
            absolute(supplied) == expected, label + " path does not match expected location")


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == RUNTIME, "manifest must contain exactly eleven runtime filenames")
    require(all(isinstance(h, str) and re.fullmatch(r"[0-9a-fA-F]{64}", h) for h in manifest.values()), "invalid manifest SHA256")
    return {name: value.lower() for name, value in manifest.items()}


def validate_deployment(deployment, stable, manifest):
    require(deployment.get("status") == "deployed" and not deployment.get("error"), "deployment is not successfully completed")
    bound_path(deployment["destination"], stable, "deployment destination")
    files = deployment["files"]
    require(isinstance(files, list) and len(files) == 11, "deployment must list eleven files")
    require(all(isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(f.get("sha256"), str) for f in files), "invalid deployment file record")
    require(len({f["name"] for f in files}) == 11, "duplicate deployment filename")
    require({f["name"]: f["sha256"].lower() for f in files} == manifest, "deployment hashes differ from snapshot manifest")


def validate_process(process, pid, stable):
    require(type(process.get("Pid")) is int and process["Pid"] == pid and process.get("CimPid") == pid, "wrong inspected PID")
    require(process.get("AliveAfter") is True and isinstance(process.get("StartTimeUtc"), str) and
            re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", process["StartTimeUtc"]) and
            process["StartTimeUtc"] == process.get("VerifiedStartTimeUtc"), "process exited or PID/start time changed")
    require(type(process.get("ParentPid")) is int and process["ParentPid"] > 0 and
            isinstance(process.get("CommandLine"), str) and process["CommandLine"].strip(), "missing parent or command line")
    bound_path(process["Executable"], confined(stable, "llama-server.exe"), "server executable")
    listeners = process["Listeners"]
    require(isinstance(listeners, list) and listeners and all(
        row.get("LocalPort") == 8090 and row.get("OwningProcess") == pid for row in listeners), "port8090 listener is missing or owned by another PID")
    modules = process["Modules"]
    require(isinstance(modules, list), "missing module list")
    loaded = {}
    for module in modules:
        require(isinstance(module.get("Name"), str) and isinstance(module.get("Path"), str), "invalid module record")
        name = module["Name"].lower()
        if name in RUNTIME:
            require(name not in loaded, "duplicate runtime module: " + name)
            bound_path(module["Path"], confined(stable, name), "loaded module " + name)
            loaded[name] = module["Path"]
    require(set(loaded) == LOADED, "loaded runtime modules differ from required nine")
    return loaded


def validate_http(http, root):
    require(set(http) == {"/health", "/props", "/slots"}, "missing HTTP endpoint evidence")
    require(all(row.get("status") == 200 and "body" in row for row in http.values()), "HTTP endpoint failed")
    require(http["/health"]["body"].get("status") == "ok", "server health is not ok")
    props = http["/props"]["body"]
    require(props.get("model_alias") == "bonsai-2-27b", "wrong model alias")
    bound_path(props["model_path"], root/"models/Ternary-Bonsai-2-27B-PQ2_0.gguf", "model")
    require(props.get("total_slots") == 4 and props["default_generation_settings"].get("n_ctx") == 188416, "wrong capacity in props")
    slots = http["/slots"]["body"]
    require(isinstance(slots, list) and len(slots) == 4 and all(type(s.get("id")) is int for s in slots) and
            {s.get("id") for s in slots} == {0,1,2,3}, "wrong or duplicate slot identities")
    require(all(s.get("n_ctx") == 188416 and s.get("is_processing") is False for s in slots), "busy slots or wrong slot capacity")


def inspect_process(pid, evidence):
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", PROCESS_QUERY]
    env = {k:v for k,v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    env["BONSAI_PROVENANCE_PID"] = str(pid)
    row = dict(command=command, target_pid=pid, exit_code=None, environment_removed=["PSModulePath"])
    evidence["process_query"] = row
    started = time.monotonic()
    try:
        p = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, env=env, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        row.update(exit_code=p.returncode, stdout=p.stdout.decode("utf-8-sig", errors="replace"), stderr=p.stderr.decode("utf-8-sig", errors="replace"),
                   stdout_base64=base64.b64encode(p.stdout).decode(), stderr_base64=base64.b64encode(p.stderr).decode())
        require(p.returncode == 0, "process query failed with exit " + str(p.returncode))
        return decode_json(p.stdout.decode("utf-8-sig"))
    except subprocess.TimeoutExpired as error:
        row.update(error="process query timed out", stdout_base64=base64.b64encode(error.stdout or b"").decode(),
                   stderr_base64=base64.b64encode(error.stderr or b"").decode())
        raise
    finally:
        row["elapsed_seconds"] = time.monotonic()-started


def read_http(route, evidence):
    row = dict(url="http://127.0.0.1:8090"+route)
    evidence[route] = row
    try:
        try:
            response = urllib.request.urlopen(row["url"], timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            row.update(status=response.code, headers=list(response.headers.items()), raw_body_base64=base64.b64encode(data).decode())
            row["body"] = decode_json(data.decode("utf-8"))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        raise


def collect(args, evidence):
    require(type(args.pid) is int and 0 < args.pid <= 0xffffffff, "invalid Win32 PID")
    root = absolute(args.root)
    stable = absolute(root/"tools/llamacpp-prism")
    require(stable == root/"tools/llamacpp-prism", "stable runtime directory escapes root")
    snapshot = absolute(args.snapshot_dir)
    manifest_path, deployment_path = absolute(args.snapshot_manifest), absolute(args.deployment_receipt)
    evidence.update(snapshot_manifest=str(manifest_path), snapshot_manifest_sha256=digest(manifest_path),
                    deployment_receipt=str(deployment_path), deployment_receipt_sha256=digest(deployment_path), snapshot_dir=str(snapshot))
    manifest = validate_manifest(decode_json(manifest_path.read_text(encoding="utf-8-sig")))
    deployment = decode_json(deployment_path.read_text(encoding="utf-8-sig"))
    validate_deployment(deployment, stable, manifest)
    evidence["stable_runtime_hashes"], evidence["snapshot_runtime_hashes"] = {}, {}
    for name, expected in manifest.items():
        for directory, key in [(stable, "stable_runtime_hashes"), (snapshot, "snapshot_runtime_hashes")]:
            observed = digest(confined(directory, name))
            evidence[key][name] = observed
            require(observed == expected, "backing file hash mismatch: " + str(directory/name))
    process = inspect_process(args.pid, evidence)
    evidence["process"] = process
    loaded = validate_process(process, args.pid, stable)
    evidence["loaded_runtime_modules"] = {}
    for name, path in loaded.items():
        observed = digest(path)
        evidence["loaded_runtime_modules"][name] = dict(path=path, backing_file_sha256=observed)
        require(observed == manifest[name], "loaded backing file changed: " + name)
    evidence["http"] = {}
    for route in ("/health", "/props", "/slots"):
        read_http(route, evidence["http"])
    validate_http(evidence["http"], root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    for flag in ("root", "snapshot-manifest", "snapshot-dir", "deployment-receipt", "output"):
        parser.add_argument("--"+flag, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with args.output.open("x", encoding="utf8") as output:
            result = dict(recorded_unix=time.time(), server_pid=args.pid, scope=SCOPE,
                          tool_sha256=digest(__file__), passed=False)
            try:
                collect(args, result)
                result["passed"] = True
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            json.dump(result, output, indent=2)
            output.write("\n")
        print(json.dumps({k:result[k] for k in ("passed", "server_pid", "error") if k in result}))
        return 0 if result["passed"] else 2
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
