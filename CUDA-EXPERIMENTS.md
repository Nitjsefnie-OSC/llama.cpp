# Bonsai CUDA experiment log

Append-only: append new trials, decisions, and corrections at the end. Do not edit or remove older entries. An unsuccessful experiment remains part of the record.

## Scope and measurement conventions

- Target: Ternary-Bonsai-2-27B-PQ2_0.gguf on NVIDIA RTX 3060 12 GB (SM 8.6), Windows, Intel i7-13700, 32 GB RAM.
- Optimize both prompt ingestion and output generation. Other architectures are outside the optimization target.
- Source baseline: PrismML-Eng/llama.cpp `bdc23b56b4458b9f1655aec5287f3ab56ee8daaa`. Fork: https://github.com/Nitjsefnie-OSC/llama.cpp, branch `bonsai-cuda-rtx3060`.
- Reproducible build/benchmark setup commit: `265025b91fd81ff671d6dac6d75f9af7304df2d8`.
- Local baseline: CUDA 12.9.1, MSVC 19.44, Release, architecture `86-real`; binaries in `../tools/llamacpp-cuda-baseline`. The live server still uses Prism `b10709-9a9394a89`, not this local build.
- M/N/K below are output rows / token columns / reduction width. Times are microseconds per operation, from `test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'type_a=pq2_0'`. Speedup is baseline time divided by candidate time; greater than 1 is faster.
- The live server remains resident during operation benchmarks. No full-model A/B comparison or deployment of a candidate has occurred. Operation speedups must not be reported as model token-rate speedups.
- Raw artifacts are retained under `C:/Users/zmatek/llm/logs/` (outside git). Paths below are relative to that directory. Tests compare CUDA with the existing CPU reference and its numerical tolerances.

## Backfill recorded 2026-09-22T15:07:04.923185+00:00

These entries reconstruct completed trials from their retained artifacts. Their order follows the experiment sequence, not a fabricated per-trial timestamp.

### 001 - Baseline and benchmark setup

- Built `llama-bench`, `llama-server`, `test-backend-ops`, and `llama-perplexity` with the pinned CUDA toolchain: passed.
- Ran the PQ2_0 operation performance suite: 28 cases. Baseline times appear in entry 002's table.
- Tried CSV output first: it contained support/operation metadata but no usable timing values in this test tool. Rejected it as a timing source and switched to the console timing output. Artifacts: `cuda-ops-baseline.csv`, `cuda-ops-baseline.stderr.log`, `cuda-ops-baseline-console.txt`, `cuda-ops-baseline-console.stderr.log`.
- Exercised the server benchmark helper with a 128-token prompt, 8 output tokens, one warmup and one measured repetition: passed exact token-count checks. This is a harness smoke test, not a stable performance baseline. Artifact: `cuda-bench-harness-smoke.jsonl`.
- An earlier live completion trial produced 26.217 output tok/s for 128 output tokens after a 15-token prompt. Single trial only. Artifact: `cuda-baseline-live-20260922-163023.json`.

### 002 - Five warps per decode block: REJECTED

- Hypothesis: increasing PQ2_0 single-token blocks from four to five warps reduces reduction-loop trips for the model's 5120-wide projections.
- Change: generic `calc_nwarps` returns 5 for PQ2_0, one output column, non-small-K.
- Correctness: 32/32 targeted matrix cases passed (67 output rows; N=1..8; K=1024,5120,6144,17408).
- Result: the principal K=5120 projections got about 8% slower. Removed this candidate; no deployment.
- Artifacts: `cuda-mmvq5-correctness.txt`, `cuda-ops-mmvq5.txt`, `cuda-ops-mmvq5.stderr.log`.

### 003 - One warp per decode output row: PROVISIONALLY RETAINED

- Hypothesis: assigning one row to one warp removes shared-memory reduction between warps and improves the short reduction-width projections.
- Change: dedicated PQ2_0 single-token kernel, four independent row warps per 128-thread block, SM 8.6 dense single-channel/single-sample dispatch. Preserve bias/gate fusion and fall back for indexed, batched, or scaled cases.
- Correctness: 32/32 targeted matrix cases; then 36/36 existing fusion cases; then 40/40 fusion cases after adding full-width and row-tail coverage. All passed.
- First performance pass covered the same 28 PQ2_0 cases as baseline and the five-warp experiment. Full results below, including unchanged paths and noisy control cases.
- Artifacts: `cuda-warp-correctness.txt`, `cuda-warp-fusion.txt`, `cuda-warp-fusion-large.txt`, `cuda-ops-warp.txt` and matching stderr files. Binary snapshot: `../tools/llamacpp-cuda-warp`.

| M | N | K | Baseline us | Five warps us | One warp/row us |
|---:|---:|---:|---:|---:|---:|
| 17408 | 1 | 5120 | 90.94 | 98.30 | 79.01 |
| 5120 | 1 | 17408 | 81.18 | 82.23 | 80.39 |
| 10240 | 1 | 5120 | 54.98 | 59.50 | 47.86 |
| 17408 | 2 | 5120 | 90.36 | 90.53 | 89.73 |
| 17408 | 4 | 5120 | 153.67 | 153.97 | 152.68 |
| 17408 | 8 | 5120 | 252.27 | 253.02 | 247.08 |
| 17408 | 32 | 5120 | 232.85 | 234.05 | 229.61 |
| 5120 | 32 | 17408 | 288.87 | 291.00 | 283.67 |
| 10240 | 32 | 5120 | 144.75 | 145.05 | 141.59 |
| 6144 | 32 | 5120 | 106.58 | 104.23 | 101.70 |
| 5120 | 32 | 6144 | 109.77 | 105.34 | 101.70 |
| 17408 | 128 | 5120 | 637.10 | 602.43 | 585.96 |
| 5120 | 128 | 17408 | 640.56 | 630.07 | 618.70 |
| 10240 | 128 | 5120 | 373.62 | 378.85 | 357.29 |
| 6144 | 128 | 5120 | 237.34 | 236.20 | 232.26 |
| 5120 | 128 | 6144 | 239.11 | 238.32 | 235.35 |
| 17408 | 512 | 5120 | 2439.22 | 2357.55 | 2322.63 |
| 5120 | 512 | 17408 | 2507.70 | 2442.47 | 2409.08 |
| 10240 | 512 | 5120 | 1437.59 | 1455.07 | 1424.62 |
| 6144 | 512 | 5120 | 862.21 | 866.14 | 857.63 |
| 5120 | 512 | 6144 | 886.80 | 889.08 | 880.81 |
| 4096 | 1 | 14336 | 55.36 | 55.96 | 54.35 |
| 4096 | 2 | 14336 | 56.69 | 56.16 | 56.50 |
| 4096 | 3 | 14336 | 80.23 | 80.20 | 80.02 |
| 4096 | 4 | 14336 | 111.10 | 113.91 | 109.86 |
| 4096 | 5 | 14336 | 116.80 | 115.22 | 115.86 |
| 4096 | 8 | 14336 | 187.88 | 190.07 | 185.98 |
| 4096 | 512 | 14336 | 1698.89 | 1705.94 | 1715.95 |

Confirmation in ABBA order (baseline, candidate, candidate, baseline), restricted to N=1. Samples are listed in measurement order for each binary. This is stronger evidence than the first pass; the long-K benefit is small/noisy.

| M | K | Baseline samples us | Candidate samples us | Median speedup |
|---:|---:|---|---|---:|
| 17408 | 5120 | [96.69, 98.49] | [80.33, 81.64] | 1.205x |
| 5120 | 17408 | [86.45, 86.9] | [81.18, 86.31] | 1.035x |
| 10240 | 5120 | [57.12, 55.94] | [48.72, 48.92] | 1.158x |
| 4096 | 14336 | [55.24, 55.46] | [54.77, 54.75] | 1.011x |

Artifacts: `cuda-warp-abba-0.txt` through `cuda-warp-abba-3.txt`, with matching stderr files. End-to-end acceptance remains pending.

### 004 - CUDA event timing instrumentation: SMOKE PASSED

- Added `DEBUG_CUDA_TIMING=1` instrumentation around executed CUDA graph operations, including fused groups, plus graph totals. CUDA graph replay is disabled during this diagnostic mode so operation events are observable. No CUDA events are created when it is off.
- Ran the 67-row, N=1, K=5120 PQ2_0 correctness case with timing enabled: 1/1 passed and emitted an operation row plus totals.
- Observed approximately 9.59 ms on this cold instrumented launch. This includes cold-launch costs and is not a steady-state kernel benchmark. Do not compare it with operation-suite microseconds.
- Artifacts: `cuda-timing-smoke.txt`, `cuda-timing-smoke.stderr.txt`.
- Full-model attribution with this instrumentation has not run yet.

### 005 - Live server ingestion/output baseline: MEASURED, UNSTABLE

- Server context 188416, four slots, batch/microbatch 512, Q4_0 K/V cache, PQ2_0 model fully offloaded, alias `bonsai-2-27b`, port 8090.
- Command: `python scripts/bonsai-server-bench.py --output ../logs/cuda-live-baseline-512-4096.jsonl --prompts 512,4096 --tokens 256 --reps 3`.
- One warmup per prompt length, then three measured repetitions; temperature 0, seed 1234, ignore EOS, no prompt-cache reuse, exact prompt/output token-count checks. All eight requests completed.

| Repetition (0=warmup) | Prompt tokens | Ingest tok/s | Output tok/s | Wall seconds |
|---:|---:|---:|---:|---:|
| 0 | 512 | 350.69 | 30.40 | 9.88 |
| 0 | 4096 | 494.91 | 28.63 | 17.20 |
| 1 | 512 | 467.47 | 30.79 | 9.53 |
| 1 | 4096 | 462.41 | 23.96 | 19.53 |
| 2 | 512 | 412.74 | 23.45 | 12.13 |
| 2 | 4096 | 420.19 | 25.10 | 19.92 |
| 3 | 512 | 288.08 | 21.03 | 13.91 |
| 3 | 4096 | 53.03 | 12.90 | 97.01 |

- Strong drift appeared during this run, including a severe final long-prompt slowdown. Windows GPU process counters for server PID 10160 later reported 12041441280 bytes dedicated and 442499072 bytes shared usage (about 422 MiB shared); an earlier inspection had reported zero shared usage. This indicates memory pressure, but does not by itself prove the cause of every slowdown.
- Ingestion-candidate compilation overlapped part of the live baseline, so CPU contention is another confounder. Operation benchmarks did not overlap these live requests.
- This baseline is retained as evidence, not treated as a controlled A/B result. A maintenance-window request to stop/restore the existing server and wrapper is pending. No existing service process was stopped.
- Artifact: `cuda-live-baseline-512-4096.jsonl` contains full requests/responses, token outputs, timings, GPU telemetry, and medians.

### 006 - MMQ column tile capped at 64: REJECTED

- Hypothesis: reducing the PQ2_0 MMQ column tile from 128 to 64 on SM 8.6 reduces per-block shared memory and allows more resident blocks.
- Change: cap `J` at 64 in `mul_mat_q_switch_J`; leave the retained decode kernel intact.
- Correctness: 8/8 new prefill cases passed, covering M=128/129, N=65/512, K=5120/17408; 40/40 fusion cases also passed.
- Measurement: ABBA, six PQ2_0 N=512 shapes. Every measured shape regressed. Narrower tiles lose despite the proposed occupancy benefit. Rejected; this change is to be removed before the next candidate.

| M | K | Baseline samples us | Candidate samples us | Median speedup |
|---:|---:|---|---|---:|
| 10240 | 5120 | [1429.28, 1447.6] | [1684.76, 1691.93] | 0.852x |
| 17408 | 5120 | [2471.19, 2369.87] | [2797.19, 2816.26] | 0.862x |
| 4096 | 14336 | [1694.25, 1718.83] | [2068.04, 2070.8] | 0.825x |
| 5120 | 17408 | [2442.08, 2468.29] | [2966.32, 2965.08] | 0.828x |
| 5120 | 6144 | [875.54, 889.48] | [1070.29, 1074.56] | 0.823x |
| 6144 | 5120 | [851.6, 866.72] | [1045.59, 1046.2] | 0.821x |

- Artifacts: `cuda-mmq64-correctness.txt`, `cuda-mmq64-abba/metadata.json`, `cuda-mmq64-abba/trials.jsonl`, `cuda-mmq64-abba/comparison.json`, per-trial stdout/stderr, and `cuda-build-mmq64.txt`.
- The reusable `scripts/bonsai-cuda-ops-bench.py` runner was exercised by this ABBA run: four trials, six matching cases each, binary hashes and GPU telemetry recorded successfully.

### Next experiment announced, not yet run

A larger MMQ row tile is under source inspection. It has no measured result and is not an accepted change. Full-model candidate validation and deployment remain pending.

## 007 - Larger MMQ row tile: experiment started 2026-09-22T15:09:32.715128+00:00

- The rejected J=64 cap from entry 006 has been removed from source.
- Hypothesis: a 256-row x 128-column tile improves activation reuse over the baseline 128 x 128 tile. Use 512 threads (16 warps), because the MMA row mapping must cover all 256 rows; merely changing I with 256 threads would not cover them.
- Change: PQ2_0 Ampere J=128 configuration uses I=256 and 512 threads for both full/tail variants; fast-path row divisibility changes to 256 for PQ2_0. Smaller J configurations remain unchanged.
- Prediction: at least 5% speedup on the N=512 main projections, with passing CPU-reference checks. This prediction is not a result.
- Build is running. Planned gate: full/tail row tiles at the model's K widths, then the same N=512 ABBA suite as entry 006. Reject any numerical failure or repeatable regression.
- Build output: `cuda-build-mmq256x128.txt`. Source patch captured as `cuda-mmq256x128-source.patch`.

## 007 result - Larger MMQ row tile: REJECTED (2026-09-22T15:14:10.114062+00:00)

- Build passed. Expanded the correctness gate to M=128,129,256,257; N=65,512; K=5120,17408. All 16/16 cases passed.
- N=512 ABBA results below. Every measured shape regressed; the predicted gain was falsified. Removed both the tile changes and the altered row-divisibility condition.

| M | K | Baseline samples us | Candidate samples us | Median speedup |
|---:|---:|---|---|---:|
| 10240 | 5120 | [1445.1, 1429.18] | [1573.95, 1572.51] | 0.913x |
| 17408 | 5120 | [2613.53, 2335.76] | [2597.86, 2605.2] | 0.951x |
| 4096 | 14336 | [1674.08, 1705.04] | [2114.57, 2120.38] | 0.798x |
| 5120 | 17408 | [2482.6, 2435.6] | [2621.06, 2623.89] | 0.938x |
| 5120 | 6144 | [864.95, 881.75] | [956.04, 957.35] | 0.913x |
| 6144 | 5120 | [868.7, 856.99] | [1143.07, 1148.42] | 0.753x |

Artifacts: `cuda-mmq256x128-correctness.txt`, matching stderr, and the complete `cuda-mmq256x128-abba/` directory.

## 008 - Fuse PQ2_0 MMQ scale loading: experiment started

- Source inspection found a separate scale-loading loop in `ggml_cuda_mmq_load_tiles_pq2_0` after the packed-weight loop. On a 32-lane warp it repeats the eight scale positions across four lane groups and traverses rows again.
- Hypothesis: write each scale from the same lane already loading its 32 quantized weights, reusing the block pointer and eliminating the second row loop and duplicate scale loads/stores.
- Change: move scale writes into the quantized-weight loop, using `ksx = kbx*scale_entries_per_block + kqsx`; remove the separate scale loop. Restore baseline MMQ tiling. The retained decode candidate is unchanged.
- Prediction: at least 3% improvement on the principal N=512 projections, with the same numerical results within existing tolerances. This is not a measured result.
- Planned validation: all 16 prefill correctness cases and the N=512 ABBA suite, followed by smaller-batch coverage if promising.
- Source patch: `cuda-scale-fused-source.patch`; build output: `cuda-build-scale-fused.txt`.

## 008/009 update - Service acceptance gate (2026-09-22T15:23:08.557354+00:00)

- Experiment 008 built successfully; 48/48 PQ2_0 matrix correctness cases and 40/40 fusion cases passed.
- No isolated performance trial was run for 008. The user specified that actual service throughput is the acceptance benchmark; subsequent kernel timings are diagnostic only.
- The user approved stopping/restoring serving and confirmed no generation was active. Saved all four slot states (slot 0: 4351 tokens, 237196840 bytes; others empty) before stopping the inspected wrapper/launcher/server PIDs 18700/10872/10160 through the existing session-0 scheduled task.
- Original supervisor saved byte-for-byte at `logs/cuda-maintenance/llama-supervisor.original.ps1`, SHA256 `54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be`. The temporary maintenance hook suppresses automatic server restarts. Restore it before completing the task. Scheduled-task settings were not changed.
- After stopping, GPU memory use fell to 499 MiB. This is the clean starting condition for server comparisons.
- Service benchmark launch attempt 1 failed before launching a model: Windows PowerShell evaluated the default Root expression while PSScriptRoot was empty. Moved that default computation into the script body and supplied Root explicitly on the retry. This is a harness failure, not a CUDA result.
- Service trials retain context 188416, four slots, batch/microbatch 512, Q4_0 K/V cache, the same model and chat template, port 8090, and uncached deterministic requests. Added per-server-PID dedicated/shared GPU memory counters to the request benchmark.

## 009 configuration correction (2026-09-22T15:25:09.581980+00:00)

- The first successful benchmark-server launch explicitly passed `-np 4`. Startup inspection showed this changed the context layout to four fixed 47104-token slots, unlike the original automatic four-slot configuration with 188416 tokens available per slot.
- No throughput trial ran with that configuration. Stopped that inspected benchmark server, removed `-np 4`, and restarted with the original command-line flags. `/props` then confirmed context 188416, four slots, original model path, build `b10709-9a9394a89`.
- The original-build service baseline is now running: 512/4096 input tokens, 256 output tokens, one warmup and three measured repetitions. Artifact: `cuda-service-stock-a.jsonl`.
- Admission gate: actual service ingestion/output rates plus correctness. The earlier isolated-kernel gains remain provisional until this gate passes. No reduction of context or cache precision is part of the candidate.

## 009 first service comparison (2026-09-22T15:31:42.185657+00:00)

Original Prism build versus the combined one-warp decode + fused-scale loader candidate. Same serving flags and identical request bodies; three measured repetitions after warmup.

| Prompt tokens | Original ingest tok/s | Candidate ingest tok/s | Original output tok/s | Candidate output tok/s | Ingest ratio | Output ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 390.97 | 389.23 | 31.66 | 32.94 | 0.996x | 1.040x |
| 4096 | 482.93 | 502.07 | 27.71 | 30.91 | 1.040x | 1.116x |

- Exact generated-content hashes matched in 8/8 paired requests, including warmups.
- Short-prompt ingestion is effectively unchanged in this comparison; 4096-token ingestion and output rates improved. Output improvement is much smaller than the isolated projection-kernel improvement.
- Original output at 4096 tokens drifted from 29.73 to 26.34 tok/s, while candidate output stayed between 30.79 and 31.09. Temperature reached 86 C in the candidate run, so order/thermal effects remain a concern.
- This compares different packaged builds/toolchains as well as the kernel changes. Next compare against the unmodified local build to isolate the code changes. No final acceptance claim yet.
- Artifacts: `cuda-service-stock-a.jsonl`, `cuda-service-warp-scale-a.jsonl`, `cuda-warp-scale-provenance.json`, `cuda-warp-scale-source.patch`.

## 010 - Unmodified local build versus candidate (2026-09-22T15:35:24.220150+00:00)

Same CUDA toolchain, same serving flags, warmup plus three measured repetitions. Candidate was measured before this baseline, reversing the order of the first comparison.

| Prompt | Baseline ingest tok/s | Candidate ingest tok/s | Baseline output tok/s | Candidate output tok/s | Ingest ratio | Output ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 385.89 | 389.23 | 31.35 | 32.94 | 1.009x | 1.051x |
| 4096 | 483.72 | 502.07 | 29.53 | 30.91 | 1.038x | 1.047x |

- Identical request bodies and 8/8 exact generated-content matches across paired trials, including warmups.
- Evidence supports about 5% higher service output throughput and about 4% higher ingestion throughput at 4096 input tokens. The 512-token ingestion difference is too small/noisy to claim a meaningful gain.
- Started additional baseline validation at 16384 input tokens and 128 output tokens (warmup plus two measured repetitions), followed by a candidate repeat and the same long-prompt test. No context reduction.
- Artifacts: `cuda-service-built-baseline-a.jsonl`, `cuda-service-warp-scale-a.jsonl`; long-prompt baseline will be `cuda-service-built-baseline-16k.jsonl`.
