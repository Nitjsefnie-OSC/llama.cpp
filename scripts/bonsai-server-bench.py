#!/usr/bin/env python3
"""Measure uncached Bonsai ingestion and decode through a running llama server."""

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import json
import ntpath
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request


def affinity_mask(value):
    mask = int(value, 16 if value.lower().startswith("0x") else 10)
    if mask <= 0:
        raise argparse.ArgumentTypeError("affinity mask must be nonzero and positive")
    return mask


@contextmanager
def server_affinity(pid, requested_mask):
    if requested_mask is not None and (pid is None or sys.platform != "win32"):
        raise ValueError("affinity changes require --pid on 64-bit Windows")
    if pid is None or sys.platform != "win32":
        yield None
        return
    if not 1 <= pid <= 0xFFFFFFFF:
        raise ValueError("pid must fit a positive Win32 DWORD")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("affinity inspection requires 64-bit Python on Windows")

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    api.QueryFullProcessImageNameW.restype = wintypes.BOOL
    api.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.GetProcessAffinityMask.restype = wintypes.BOOL
    api.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    api.SetProcessAffinityMask.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL

    access = 0x1000 | (0x0200 if requested_mask is not None else 0)  # query limited information / set information
    handle = api.OpenProcess(access, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    def read_mask():
        process = ctypes.c_size_t()
        system = ctypes.c_size_t()
        if not api.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return process.value, system.value

    restore_needed = False
    try:
        name = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(name))
        if not api.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
            raise ctypes.WinError(ctypes.get_last_error())
        if ntpath.basename(name.value).lower() != "llama-server.exe":
            raise RuntimeError(f"PID {pid} is not llama-server.exe: {name.value}")
        original, system = read_mask()
        if not original or not system:
            raise RuntimeError("Process affinity is unavailable or spans unsupported processor groups")
        if requested_mask is not None and (requested_mask <= 0 or requested_mask & ~system):
            raise ValueError(f"Affinity mask must be a nonzero subset of system mask {system:#x}")
        metadata = {"original_mask": hex(original), "effective_mask": hex(original),
                    "system_mask": hex(system), "requested_mask": None if requested_mask is None else hex(requested_mask)}
        if requested_mask is not None:
            restore_needed = True
            if not api.SetProcessAffinityMask(handle, requested_mask):
                raise ctypes.WinError(ctypes.get_last_error())
            effective, _ = read_mask()
            if effective != requested_mask:
                raise RuntimeError(f"Requested affinity {requested_mask:#x}, read back {effective:#x}")
            metadata["effective_mask"] = hex(effective)
        yield metadata
    finally:
        try:
            if restore_needed:
                try:
                    if not api.SetProcessAffinityMask(handle, original):
                        raise ctypes.WinError(ctypes.get_last_error())
                    restored, _ = read_mask()
                    if restored != original:
                        raise RuntimeError(f"Expected {original:#x}, read back {restored:#x}")
                    metadata["restored_mask"] = hex(restored)
                except (OSError, RuntimeError) as error:
                    raise RuntimeError(f"Failed to restore PID {pid} affinity to {original:#x}: {error}") from error
                print(f"Restored PID {pid} affinity to {restored:#x}", file=sys.stderr, flush=True)
        finally:
            if not api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", default="512,4096")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--pid", type=int, help="Windows server PID for dedicated/shared GPU memory counters")
    parser.add_argument("--affinity-mask", type=affinity_mask, help="temporary Windows process affinity mask (decimal or 0x); requires --pid")
    args = parser.parse_args()
    sizes = [int(n) for n in args.prompts.split(",")]
    if min(sizes) < 1 or args.tokens < 1 or args.reps < 1:
        parser.error("Prompt lengths, tokens and repetitions must be positive")
    if args.pid is not None and not 1 <= args.pid <= 0xFFFFFFFF:
        parser.error("pid must be between 1 and 4294967295")
    if args.affinity_mask is not None and (args.pid is None or sys.platform != "win32"):
        parser.error("--affinity-mask requires --pid on 64-bit Windows")
    with server_affinity(args.pid, args.affinity_mask) as affinity:
        run_benchmark(args, sizes, affinity, parser)


def run_benchmark(args, sizes, affinity, parser):
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

    def process_memory():
        if args.pid is None:
            return None
        command = (
            "$ErrorActionPreference='Stop'; "
            f"(Get-Counter '\\GPU Process Memory(pid_{args.pid}_*)\\Dedicated Usage',"
            f"'\\GPU Process Memory(pid_{args.pid}_*)\\Shared Usage').CounterSamples "
            "| Select-Object Path,CookedValue | ConvertTo-Json -Compress"
        )
        return json.loads(subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], text=True))

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

        metadata = {"kind": "metadata", "url": args.url, "props": props,
                "gpu_fields": "temperature,sm_clock,memory_clock,power,memory_used,utilization",
                "gpu": gpu(), "server_pid": args.pid, "process_memory": process_memory(),
                "sizes": sizes, "tokens": args.tokens, "reps": args.reps}
        if affinity is not None:
            metadata["affinity"] = affinity.copy()
        record(metadata)
        rows = []
        for repetition in range(args.reps + 1):
            for size in sizes:
                if any(slot["is_processing"] for slot in request("/slots")):
                    raise RuntimeError("Server is busy; refusing a contaminated measurement")
                body = {"prompt": tokenized[:size], "n_predict": args.tokens,
                        "temperature": 0, "seed": 1234, "ignore_eos": True,
                        "cache_prompt": False, "id_slot": 0, "return_tokens": True}
                before = gpu()
                memory_before = process_memory()
                start = time.perf_counter()
                response = request("/completion", body)
                elapsed = time.perf_counter() - start
                timings = response["timings"]
                if timings["prompt_n"] != size or timings["predicted_n"] != args.tokens:
                    raise RuntimeError(f"Unexpected token counts: {timings}")
                row = {"kind": "trial", "warmup": repetition == 0, "repetition": repetition,
                       "prompt_tokens": size, "wall_seconds": elapsed, "gpu_before": before,
                       "gpu_after": gpu(), "request": body, "response": response,
                       "process_memory_before": memory_before, "process_memory_after": process_memory(),
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
