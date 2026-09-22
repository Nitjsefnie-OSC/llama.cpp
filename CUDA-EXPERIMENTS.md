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

## 011 - Repeat exposes ingestion instability (2026-09-22T15:47:08.513044+00:00)

| Artifact | Prompt tokens | Ingest tok/s | Output tok/s | Wall seconds |
|---|---:|---:|---:|---:|
| cuda-service-built-baseline-16k.jsonl | 16384 | 462.27 | 24.86 | 40.561 |
| cuda-service-warp-scale-b.jsonl | 512 | 314.78 | 33.67 | 9.248 |
| cuda-service-warp-scale-b.jsonl | 4096 | 381.21 | 31.68 | 18.797 |

- Candidate repeat B uses exactly the same binary as candidate A, with unchanged service flags. Short-prompt output remains faster, but 4096-token ingestion dropped from 502.07 to about 380 tok/s. This invalidates treating the first-pass ingestion result as a stable gain; the source optimization is retained provisionally while the environment sensitivity is diagnosed.
- Candidate B process memory: dedicated 12085395456 bytes, shared 398458880 bytes; candidate A: dedicated 12093767680, shared 390070272. This small inverse change is a possible residency signal, not proof of a cause. GPU temperatures/clocks and full request/output data are preserved in the artifacts.
- Baseline 16K completed; candidate 16K is next, then actual-service CUDA timing. Timing disables CUDA graph replay and therefore is diagnostic only, not acceptance throughput.
- User reported repeated network permission prompts. Read-only firewall inspection found persistent path-specific TCP/UDP allow rules for each tested executable directory on Domain/Public profiles, current network DomainAuthenticated, and no enabled inbound block rules. Future deployments will reuse the already-approved tools/llamacpp-prism/llama-server.exe path and preserve immutable snapshots, rather than launching each new directory.

## 011 continuation - 16K and stable firewall path (2026-09-22T15:50:52.186282+00:00)

- Long-context baseline: ingest 462.27 tok/s, output 24.86 tok/s, wall 40.561 s. Candidate repeat: ingest 368.48 tok/s, output 26.60 tok/s, wall 49.254 s. Exact request/output-content matches: 3/3 including warmup. Output improves about 7%; ingestion regression in this service instance remains unexplained and must not be hidden by the initial gain.
- Added persistent launcher deployment tooling. Every candidate is now copied to the already firewall-approved `tools/llamacpp-prism` path with verified file hashes, while preserving the entire previous directory. No firewall rules or profile notifications were changed. Existing grants cover the current Domain network and Public, not Private.
- First real deployment receipt: `bonsai-deploy-20260922T175003285-9cfe7cc023d741f19d39cf5dd684a338.json`. Original package preserved at `C:\Users\zmatek\llm\tools\llamacpp-deploy-backups\20260922T175003285-9cfe7cc023d741f19d39cf5dd684a338`. Production launcher now adds portable CUDA 12.9.1 to PATH; original launcher preserved at `logs/cuda-maintenance/serve-qwen38.before-cuda-deploy.ps1`.
- Started actual-service CUDA event profile at the same context/flags using DEBUG_CUDA_TIMING and the server's built-in log file. Profile artifacts: `cuda-service-profile-6295.log`, `cuda-service-profile-6295-requests.jsonl`. No throughput acceptance numbers will be taken with profiling enabled.

## 012 - Actual-service CUDA profile: first capture failed

- Deployed the existing candidate to the stable firewall-approved path and issued two real /completion requests (512 input, 8 output). Requests succeeded; these instrumented rates are not acceptance measurements.
- The built-in server log contained application messages but zero CUDA_TIMING records. The new parser correctly returned exit 2 with no graphs, so there is no usable attribution from this attempt. Investigating backend log filtering before retrying; preserve the failed log and JSON.
- Independent review of commit 6295cbe5 found no critical/important issue in the PQ2 kernels or timing ownership/graph gating. Minor uncovered case: the new single-column SWIGLU_OAI fusion branch lacks a focused correctness case; static operand/clipping behavior matches baseline.
- Stable-deployment tests passed under Windows PowerShell 5.1 and PowerShell 7, including hashes, complete backup, path/junction guards, running-process refusal, and injected swap rollback. Parser has 20 checks and runner 8 mocked checks. Real deployment also succeeded, with file hashes in its receipt.

## 012 capture correction

- Source proof: common/log.cpp maps GGML_LOG_LEVEL_INFO to TRACE (level 4) and filters before enqueueing file output; default verbosity is level 3. llama_log_set forwards that callback to ggml. The instrumentation was executing but its records were suppressed.
- Updated the diagnostic runner to add --log-verbosity 4 only when CudaTimingLog is requested. Normal runs keep their original verbosity. Mock checks confirmed both paths and environment restoration.
- Restarted the same candidate at the same stable path; fresh log `cuda-service-profile-6295-v4.log` contains two CUDA_TIMING_TOTAL records from startup warmup. Profiling requests are recorded in `cuda-service-profile-6295-v4-requests.jsonl`.

## 013 - SM86 asynchronous MMQ activation staging: preregistered experiment

- Actual-service profile is now captured. Across the first post-warmup selection, PQ2 MUL_MAT contributes 984.7/1499.5 ms (65.7%) of recorded operation time, GATED_DELTA_NET 212.8 ms (14.2%). These diagnostics include synchronization/instrumentation overhead and are not throughput acceptance. Correcting the parser phase heuristic: Hadamard's auxiliary BF16/F32 matrices have width >1 even during single-token decode, so phase detection must prefer quantized projection widths.
- Hypothesis: on SM86, PQ2 MMQ at J=128 can overlap activation-tile loads with computation using the existing asynchronous Y-buffer path. It already runs at one CTA per SM; adding the second Y buffer should fit shared memory without lowering CTA residency.
- Change to test: enable existing async_buffer_y for type PQ2_0, SM86, J=128 in both device selection and host shared-memory sizing. Keep the existing DGX Spark condition and all tile/occupancy selection unchanged.
- Prediction: at least 3% higher actual-service ingestion at 4096 input tokens; decode unchanged within 2%. Failure to exceed repeat variation, or material regressions, rejects the candidate. Context, Q4 KV, batching, model, prompt and output lengths stay unchanged.
- Validation: build, all 48 PQ2 matrix correctness cases and 40 fusion cases; then matched actual-service requests with profiling off and exact output comparison. Retain raw results regardless of outcome.

## 012 result and 013 sizing correction

- Corrected parser output (`cuda-service-profile-6295-v4-phases.json`, skip first 11 graphs): 20 complete graphs, no malformed/incomplete data; selected 2 prefill + 7 decode graphs. Prefill: graph 1174.567 ms, operation sum 1162.828 ms (11.739 ms gap). Decode: graph 325.369 ms, operation sum 293.262 ms (32.107 ms gap). Instrumentation and host submission gaps remain visible rather than being assigned to kernels.
- Prefill's 48 GATED_DELTA_NET calls total 201.450 ms (17.32% of operation sum). Main FFN PQ2 projections total 493.170 ms (42.41%). Decode's fused gate PQ2 projections total 63.918 ms (21.80%). The pipeline experiment targets the dominant prefill matrices; a separate read-only investigation is examining the recurrent attention cost.
- The SM86/J128 shared-memory calculation is 57,856 bytes (56.5 KiB) baseline and 76,288 bytes (74.5 KiB) with two Y buffers: X=38,912, ids=512, each Y=18,432. This corrects the approximate preregistration sizing; the one-CTA residency hypothesis is unchanged.
- Tooling received independent review with no blocking issue. The real verbosity-4 profile verifies the logging integration after mocked tests. Parser phase fixes additionally passed 11 synthetic checks including the auxiliary-matrix counterexample.

## 013 control run before asynchronous candidate

- Fresh normal (profiling off) service at the stable approved path, still the 6295 kernel binary: pp512 400.05 tok/s / output 33.67 tok/s; pp4096 509.36 tok/s / output 31.66 tok/s. Three measured repetitions after warmup, 256 output tokens. Artifact: cuda-service-warp-scale-stablepath-a.jsonl.
- Ingestion recovered from repeat B's 381.21 to 509.36 tok/s after a restart with unchanged kernel code. This is evidence of instance/environment sensitivity, not proof of its cause. The new run is the immediate control for experiment 013.
- Candidate source patch retained in cuda-mmq-async-source.patch. Host allocation predicate uses highest compiled CUDA architecture so a compatible newer device cannot execute the SM86 two-buffer specialization with one-buffer allocation.
- Build output: cuda-build-mmq-async.txt. No service throughput measurement overlaps this compilation.
- First measured trial process memory: [{"Path": "\\\\conpc53\\gpu process memory(pid_5388_luid_0x00000000_0x0000fa7a_phys_0)\\dedicated usage", "CookedValue": 12097953792}, {"Path": "\\\\conpc53\\gpu process memory(pid_5388_luid_0x00000000_0x0000fa7a_phys_0)\\shared usage", "CookedValue": 385875968}]

## 014 - Raw-gate precomputation: preregistered next experiment

- Actual-service graph 11 recurrent-attention timings: 48 nodes, 201.449 ms total, median 3.170 ms; 37/48 below 3.4 ms. First two nodes cost 19.468 and 23.201 ms. The outliers leave an estimated 152.175 ms steady floor and do not establish a residency cause.
- Source: gated_delta_net.cu launches SM86 at one state column per warp and processes tokens sequentially. RAW gates repeatedly evaluate beta sigmoid and decay softplus/exp in each column warp. Bonsai uses RAW gates, so the existing GB10 activated-gate precompute path does not apply.
- Hypothesis: for SM86, scalar RAW gates, state width 128 and at least 32 input tokens, compute sigmoid(beta) and final exp(raw_a*softplus(g+dt_bias)) once per token/head, then use the existing non-RAW/precomputed recurrence. Decode and shorter inputs retain their current paths.
- Prediction: 15-30% reduction in steady GATED_DELTA_NET time, approximately 2-4% overall prefill operation time. Actual-service gain must exceed variation; this is an unmeasured prediction and is not additive with experiment 013.
- Implementation is prepared in a separate temporary git worktree while experiment 013 compiles. It will not enter the main build or service until the current comparison completes. One temporary pool allocation for both activated beta and decay; test threshold, sequence/head indexing, and snapshot/final-state outputs against CPU reference.

## 013 build and correctness

- Build succeeded. 101/101 PQ2 matrix cases passed, including full/tail prefill shapes; 40/40 PQ2 fusion cases passed. This matrix filter is broader than the earlier 48-case subset.
- First fusion invocation accidentally used the matrix parameter name type_a=pq2_0; it matched 0/0 cases. Rejected that empty pass, corrected to type=pq2_0, and required a nonzero passed count. Both failed-filter and corrected artifacts are retained.
- Snapshot tools/llamacpp-cuda-mmq-async and cuda-mmq-async-binary-hashes.json preserve verified candidate files. Independent source review found matching host/device allocation, identical source ranges/alignment, and correctly ordered copy completion/barriers.
- Deploying through the stable firewall-approved path for the actual-service trial; original and previous packages remain preserved in deployment backups.

## 013 first service result - admission failed; restoring control

| Prompt | Control ingest | Async ingest | Control output | Async output |
|---:|---:|---:|---:|---:|
| 512 | 400.05 | 329.25 | 33.67 | 34.48 |
| 4096 | 509.36 | 390.95 | 31.66 | 32.37 |

- Artifact cuda-service-mmq-async-a.jsonl; identical request bodies and exact generated-content hashes in 8/8 trials. The candidate misses the ingestion threshold and increases 4K request wall time from 16.100 to 18.368 s.
- Output changed despite an ingestion-only patch, and previous unchanged-binary ingestion also varied between roughly 381 and 509. Therefore this trial does not isolate the source of the regression. Restoring the control for an order reversal before final disposition.
- Candidate end-of-run process memory: dedicated 12112605184 bytes, shared 371195904 bytes. Control: dedicated 12097953792, shared 385875968. The slow candidate actually had more dedicated and less shared memory, so raw shared-allocation count alone does not explain this trial.
- Reverted the two source predicates using the preserved exact patch; candidate binaries, source patch, logs and hashes remain available. No candidate change is retained from this failed trial.

## 014 source review and integration for the next build

- Independent source review passed: raw activation formulas match, shape guards preserve head/sequence indexing and retained strides, pool lifetime covers both kernels on the same stream, snapshots/cache outputs are unchanged, decode incurs no new allocation/device query, and existing GB10 activated-gate behavior remains.
- Added eight CPU-reference cases at T=31/32/33/508, two sequences and repeated value heads, covering final state and four snapshots. Build/runtime gates remain pending.
- Applied the preserved cuda-gdn-raw-source.patch to the main working tree after reverting experiment 013. The running service is still the prior control binary; no compilation or new GPU correctness run overlaps its comparison.

## 013 control reversal; 014 build started

- Control rerun pp512: ingestion 405.37 tok/s, output 35.74 tok/s, wall 8.411 s.
- Control rerun pp4096: ingestion 525.71 tok/s, output 33.77 tok/s, wall 15.324 s.
- Control artifact: cuda-service-warp-scale-stablepath-b.jsonl. Experiment 013 remains rejected for lack of demonstrated service ingestion gain; preserved artifacts allow revisiting with tighter environmental control.
- Started the raw-gate candidate build only after the control requests completed. Build log: cuda-build-gdn-raw.txt.

## 014 build and correctness passed

- Build passed; all 11 supported raw-gate CPU-reference cases passed, including all eight new threshold/multi-sequence/state-snapshot cases. One pre-existing rows-mode case remains explicitly unsupported on CUDA and is not counted as a pass.
- Candidate snapshot: tools/llamacpp-cuda-gdn-raw; hashes: cuda-gdn-raw-binary-hashes.json. Test artifacts: cuda-gdn-raw-correctness.stdout.txt and matching stderr.
- Starting service comparison with unchanged context188416, automatic four slots, Q4 KV, batch512, and exact benchmark requests. CUDA profiling is off.

## 015 - Decode unpack algebra investigation: rejected before implementation

- Read-only investigation of PQ2 unpack sought a cheaper bitplane dot plus Q8_1 sum correction without increasing model VRAM. PQ2 codes decode 00/01/10/11 as -1/0/+1/+2; code 11 is defined and cannot be discarded.
- Current arithmetic is d2*d8*sum((code-1)*q8), using eight DP4A operations per 32-value chunk. Q8_1 ds[1] contains FP16(sum(original floating activations)), not d8*sum(integer q8). Substituting it changes numerical behavior.
- Concrete arithmetic counterexample: activations [1,0.1,0,...], all-zero ternary weights: current result zero, proposed ds[1] correction 0.002685546875. Rejected this proposed shortcut; no CUDA implementation or service trial was performed.
- Exact integer bitplane correction requires additional sums/DP4A work and has no evident instruction-count advantage. Blanket 32-bit packed loads also fail alignment: chunk offset 34*block+2+8*part is only two-byte aligned in even blocks; aligned-down reconstruction needs extra work and careful end-of-buffer handling. Retain current LUT path for now.

## 016 - Eight-row PQ2 decode blocks: preregistered next experiment

- The actual-service decode profile is dominated by PQ2 projections. The retained fast path assigns one warp per output row and groups four rows into each 128-thread block.
- Hypothesis: grouping eight rows into a 256-thread block reduces block scheduling overhead while retaining exactly the same dot arithmetic and one-warp reduction. Register allocation/residency may instead negate the gain; this is an experiment, not an asserted optimization.
- Change: one shared rows-per-block constant controls launch bounds, row indexing, grid rounding and block dimensions. Restrict to the existing SM86/single-column fast path; all fusion eligibility and arithmetic remain unchanged.
- Prediction: at least 3% actual-service output gain, with ingestion unchanged. Reject changes within variation or with material regressions. Existing odd-row and fused-gate correctness cases cover the row tail. Prepare the source patch in isolation; no service or build overlap with experiment 014's benchmark.

## 014 service result - not retained

| Prompt | Control ingest | Raw-gate candidate ingest | Control output | Candidate output |
|---:|---:|---:|---:|---:|
| 512 | 405.37 | 441.00 | 35.74 | 34.52 |
| 4096 | 525.71 | 521.03 | 33.77 | 32.47 |

- Artifact cuda-service-gdn-raw-a.jsonl. Exact request bodies and generated-content hashes matched in 8/8 trials; all11supportedCPU-referencecasespassed before service testing.
- Short-prompt ingestion improved 8.8%, but 4K ingestion did not improve, and total wall times rose (512:8.411->8.584s;4K:15.324->15.717s). This does not meet the predicted useful service gain. Revert the candidate rather than retain extra allocation/launch work on inconclusive evidence.
- End-of-run dedicated/shared memory was exactly equal between this candidate and its immediate control (12116791296/367001600 bytes). Decode also changed despite the prefill-only source change, so small causal claims remain uncertain. Rejection is for lack of demonstrated service benefit, not a proven numerical or isolated-kernel failure.
- Source/binary/test artifacts remain preserved. The next experiment returns to the retained one-warp/fused-scale baseline and only changes decode block grouping.

## 016 build and correctness passed

- Source review confirmed the row-count constant controls both gate variants' launch bounds, row mapping, grid and block dimensions. For 67 rows, the final block has three active whole warps and five returning warps; no block barrier is bypassed.
- Build passed. 101/101 PQ2 matrix and 40/40 fusion cases passed with nonzero case-count assertions. Snapshot tools/llamacpp-cuda-mmvq-rows8 and cuda-mmvq-rows8-binary-hashes.json preserve the candidate; source patch cuda-mmvq-rows8-source.patch.
- Candidate contains only the eight-row change on top of retained kernels. Raw-gate preprocessing was reverted before building. Starting normal actual-service requests with unchanged model/context/batching/KV settings.

## 017 - Four-column recurrent-attention warps: preregistered next experiment

- Existing GATED_DELTA_NET code already reuses q/k loads and scalar gate values across four columns per warp on DGX Spark. SM86 currently owns one column per warp. Test that existing four-column specialization for SM86, S_v128, scalar gates, keeping formulas and state/snapshot addressing unchanged.
- Host grid geometry must match the compiled device specialization. This changes both prefill and decode but adds no allocation or kernel launch. Register pressure and reduced block count may hurt occupancy.
- Prediction: at least 5% actual-service ingestion gain, with output not regressing by more than 2%. Source preparation is isolated from experiment016's running binary. Retain the eight threshold/multi-sequence/snapshot CPU-reference cases to validate geometry.

## 016 service result - not retained; 017 build started

- Eight-row candidate: pp512 413.89 tok/s, output34.41; pp4096 516.71, output32.46. Immediate prior control B:405.37/35.74 and525.71/33.77. Request/output hashes matched8/8. Artifact cuda-service-mmvq-rows8-a.jsonl.
- The predicted >=3% output gain was not demonstrated; request wall times increased against control B. Reverted the exact source patch; candidate binaries and all correctness/results remain available. Environmental variation limits small causal conclusions, but does not justify retaining an unproven change.
- Applied only the four-column GDN patch on the retained four-row PQ2 decode baseline. Source review passed column coverage, host/device selection, final-state/snapshot/fused-cache addressing, and unchanged S16/32/64 plus KDA fallbacks. Runtime gate will include existing activated/decode/snapshot cases, not only newly added raw-prefill cases.
- Starting build log cuda-build-gdn-cols4.txt after the service trial completed.

### Experiment 017: build and correctness result

CUDA build succeeded (cuda-build-gdn-cols4.txt). Full GATED_DELTA_NET CPU-reference suite: 47/47 supported tests passed, including raw and activated scalar gates, decode, prefill, snapshots and added token-tail cases. Logs: cuda-gdn-cols4-correctness.stdout.txt / .stderr.txt. Fresh binary snapshot tools/llamacpp-cuda-gdn-cols4 verified against build outputs by SHA256 (cuda-gdn-cols4-binary-hashes.json). Next gate is profiling-off throughput on the stable-path production service.

### Experiment 016: offline resource follow-up

NVIDIA cuobjdump reports identical baseline and eight-row PQ2 resources: unfused 42 registers/thread, fused 37, zero local/stack bytes. SM86 allocation granularity gives unchanged theoretical occupancy: unfused 40 resident warps (83.3%), fused 48 (100%). Grouping eight rows changes block count but provides no occupancy gain. This is static resource analysis, not measured GPU utilization. Reports: cuda-resource-warp-scale-sm86.txt and cuda-resource-mmvq-rows8-sm86.txt.

### Experiment 018: same-process CPU-affinity preregistration

Hypothesis: Windows hybrid-core scheduling contributes to throughput variation between server processes. Native GetSystemCpuSetInformation on this i7-13700 reports one group, P-core logical CPUs 0-15 (mask 0xFFFF), E-core CPUs 16-23, and full original mask 0xFFFFFF. This is a scheduling hypothesis, not an established explanation. Compare identical warmed service requests in A-B-B-A order on one unchanged PID/binary: A original full affinity; B P-core-only. Each leg uses the standard 512/4096 prompts, 256 output, one warmup and three measured repetitions. Require unchanged request/output hashes and at least 3% repeatable throughput improvement without over 2% regression at either prompt length; restore original mask on every exit. No priority, power-plan, model, context, CUDA-kernel or sampling changes during the affinity comparison. Preserve each leg as an exclusive JSONL artifact with masks and process identity.

### Experiment 017: offline register report

Four-column GDN S128 scalar kernels use 72 registers/thread in all raw/activated and KEEP variants, with zero local memory, stack or shared memory. At 128 threads/block this gives 28 theoretical resident warps (58.3%), down from 36 raw-gate or 40 activated-gate warps in the one-column baseline. Reduced repeated work may still offset the lower occupancy; only service throughput decides. Artifact: cuda-resource-gdn-cols4-sm86.txt.

### Experiment 017: first service result — promising ingest, not yet retained

Production service PID23724, profiling off, normal context188416 and q4 KV: cuda-service-gdn-cols4-a.jsonl. Medians pp512 447.145 / tg256 34.801 tok/s, wall8.5555s; pp4096 555.169 / tg256 32.631, wall15.1963s. Versus saved retained-control B (405.367/35.742 and525.707/33.773), ingestion improves10.3%/5.6%, but decode regresses2.6%/3.4%, so this does NOT satisfy the preregistered gate. Requests and generated tokens match8/8. Third-round output recovers36.04/33.88, consistent with significant runtime variation; cause remains unproven. Next:16K/128output long-context trial, then same-process affinity investigation and fresh baseline reversal before any retention claim.

### Experiment 019: shared activation permutations for fused PQ2 decode — preregistration

Hypothesis from the actual-service decode profile: fused FFN up/gate projections can share the activation permutation needed by both integer dots. Instead of separately interleaving weight lookup results qe/qo into qx/qy for each matrix, form ue=byte_perm(u,v,0x6420) and uo=byte_perm(u,v,0x7531) once per eight activation values and dot both matrices against them. Expected source-level PRMT count falls8 to6 per group (8 fewer per32-value chunk), with unchanged weight traffic. Preserve16-bit loads for34-byte PQ2block alignment, allfour codes, each chunk scale expression andfloat accumulation order. Scope only retained SM86 fused PQ2 warp path. Gate: CPU-reference fused/unfused tests with nonzero counts, exact service output tokens, generated-resource review, and >=3% repeatable service decode gain without >2% ingest regression. No speed claim before measurement.

### Experiment 020: prefill-only four-column GDN — preregistration

Experiment017 improves ingestion but has not passed its decode gate. Test a narrower specialization: scalar S128 SM86 uses four columns per warp only when n_tokens>=32, retaining the original one-column kernel for decode and short tails. Keep existing GB10 behavior and mathematical operation order. Gate: full GATED_DELTA_NET reference suite, exact service token agreement, repeatable >=5% ingest improvement and no >2% decode regression. This is a new candidate, not a retroactive change to017 acceptance criteria.

### Experiment 017: long-context service result

At16,384 prompt /128generated tokens, two measured runs after warmup: median ingestion514.630 andoutput26.659 tok/s, wall36.6213s (cuda-service-gdn-cols4-16k.jsonl). Requests and generated tokens match the unmodifiedlocalbuild3/3. That older localbaseline measured462.270/24.858 and40.5611s; older retainedwarp-scale long trial was anomalously slow at368.484/26.595. These are historical comparisons across restarts, so a fresh retainedbaseline16K reversal is required next. No retention claim from the anomalous denominator.

### Service deployment observation during017 reversal

The first attempt to deploy the retained baseline immediately after stopping PID23724 was correctly refused because CIM still listed the terminating llama-server.exe with an unavailable path. No files were replaced by that failed attempt. After confirming zero llama-server processes, one new deployment succeeded; retainedbaseline is healthy as PID19364. Future stop sequences will wait on the already-inspected Process handle to finish before deploying. This was a lifecycle race, not a CUDA test failure.

### Experiment 017: fresh16K baseline reversal

Retainedwarp-scale production PID19364, same16K/128output requests, profilingoff: median489.621ingest/27.334output tok/s, wall38.1164s (cuda-service-warp-scale-stablepath-16k-c.jsonl). Candidate017 was514.630/26.659 and36.6213s: +5.1%ingest, -2.5%output, -3.9%totalwall. All3generatedsequencesmatch. Thus ingest improvement survives a fresh long-context control, but output still breaches the no>2%regressiongate. Original017 is not retained; experiment020 explicitly restores original decode geometry.

### Experiment018 tooling verification; experiment019 prepared

Affinity benchmark controls committed8eba03dd (reviewed original5d7dc366),27mocktests pass both independently andafterintegration; no blockingreviewfindings for this one-group24logicalCPUhost. Multi-group Windows machines remainunsupportedbythisexperiment; zeroaffinitymasksalone do not detect allmulti-groupdefaults. Affinity A-B-B-A now starts on unchangedPID19364/retainedwarp-scale binary. No builds orGPUtestjobs run during these measurements. Original017 sourceandaddedGDNcases reversed exactly againsttheirpreservedpatch. Independentlyreviewed019pairedPQ2patch applied to mainworkingtree for a later build, with originalGDNsource; no experimentalCUDAcode retainedorcommitted yet.

### Experiment018 result — P-core affinity rejected

Completed A-B-B-A on unchangedPID19364/retainedkernelbinary, no concurrentbuildsorGPUtests. Artifacts cuda-service-affinity-{a1,b1,b2,a2}.jsonl. All24 pairedcandidate/controlrequestsandgeneratedtokensequencesmatch. Pooledsixmeasuredsamplespercondition: pp512 A415.298/B416.680 (+0.33%), output A35.831/B35.816 (-0.04%); pp4096 A517.241/B514.641 (-0.50%), output A33.636/B33.686 (+0.15%). No repeatable3%benefit; rejectP-core-onlyaffinity. Both B legs verifiedrestorationto0xFFFFFF; independentPowerShellreadback=16777215. Same-processdata do not support hybrid-coreaffinity as a meaningful causeofobservedvariation. No permanentaffinity/power/prioritychange.

### Experiment019 build and independent source review

PairedPQ2helper andfusedcaller independentlyreviewed; all65,536packed16-bitwords checked againstCPUdecoding/existingLUT, preservingallfourcoefficients. Integerregroupingfitsint32(maximum8192magnitude), andchunkscale/floataccumulationorderunchanged. SourceonlyscopeSM86fusedPQ2CUDA; unfused/HIP/MUSAunchanged. Build started after018completed, logcuda-build-fused-pq2-permute.txt. FullreferenceGPUtests andactualserviceacceptance remain pending.

### Experiment020 independent source review

Prefill-onlyGDNpatchreviewpassed: COLSspecializationandhostgridagree; scalarS128SM86 selectsCOLS4onlyT>=32,GB10retainsoriginalCOLS4,otherpathsCOLS1. Arithmetic/state/snapshotbodyunchanged. T31/32/33anddecodereferencevalidationremainrequiredwhenbuilt. Patchpreservedascuda-gdn-cols4-prefill-source.patch; notyetappliedtomainwhile019isisolated.

### Experiment 019: baseline generated instructions

The first SASS extraction failed because cuobjdump needs the separate nvdisasm executable. That failure is preserved in `cuda-sass-pq2-fused-baseline.txt`. After installing only the SHA256-verified NVIDIA 12.9.1 nvdisasm component, extraction succeeded in `cuda-sass-pq2-fused-baseline-complete.txt`.

The fused baseline kernel uses 37 registers and no local memory or stack. Its loop at 0x150 through 0x9e0 processes 32 activations per lane for both matrices, with four unrolled groups. Each iteration contains 138 static instructions, 36 PRMTs and 16 IDP.4A instructions. PRMTs comprise 16 lookup operations, 16 interleaves and four compiler-generated sign extensions. Thus the source-level estimate of eight permutations per group corresponds to nine actual baseline PRMTs. Candidate instruction counts and register pressure still need comparison after compilation.

### Experiment 019: build and GPU correctness passed

Build completed successfully (`cuda-build-fused-pq2-permute.txt`). CPU-reference comparisons passed 101/101 PQ2 MUL_MAT cases and 40/40 fused MUL_MAT_VEC_FUSION cases; both commands exited zero with nonzero test counts. Artifacts: `cuda-fused-pq2-permute-{mulmat,fusion}.{stdout,stderr}.txt`. The 14-file runtime snapshot `tools/llamacpp-cuda-fused-pq2-permute` matches build output hashes in `cuda-fused-pq2-permute-binary-hashes.json`. The inspected idle control process was stopped and its handle reached terminal state before deployment. Service measurements follow with original production flags and full CPU affinity.

### Experiment 019: generated-code reduction confirmed

Offline SASS comparison confirms the intended reduction per 32-activation fused loop: PRMT 36 -> 28; IDP.4A remains 16; total static instructions 138 -> 128. Registers remain 37 per thread, with zero local memory and stack. Calculated SM86 residency remains 12 blocks / 48 warps / 100%; this is theoretical occupancy, not utilization. Logs: `cuda-sass-pq2-fused-permute.txt` and `cuda-resource-fused-pq2-permute-sm86.txt`, both successful with empty stderr. Actual-service throughput is still the acceptance gate.

### Experiment 021: CUDA graph cache diagnostics — preregistration

Hypothesis, currently low confidence: graph cache eviction or property resets contribute to variation between otherwise identical service runs. `common.cuh` sweeps every five seconds and removes graph entries idle for at least ten seconds. Backend graph properties must match for two calls before capture/replay; property changes reset warmup. The existing llama-context "graphs reused" metric counts topology reuse before backend execution, so it cannot establish CUDA replay.

Some slow 4K prefill phases took 10.71–10.77 seconds versus 7.75–7.90 in fast runs. Crossing the cache TTL may affect subsequent decode, but could be a consequence of slower prefill. Neither this observation nor nonzero shared GPU allocation proves the cause of the ingest gap.

Add `DEBUG_CUDA_GRAPH_STATS=1`, inert otherwise, with cache identity, misses/evictions and idle age, property/warmup reset reasons, direct/capture/replay decisions, host lifecycle durations, and token width from quantized matrix projections. Keep CUDA replay enabled; add no CUDA events or synchronization. Matching decisions and negligible lifecycle costs in fast/slow runs would reject this hypothesis. Implement separately from performance candidates; use normal profiling-off service trials for speed acceptance.

### Experiment 019: first service trial fails the speed gate

Production PID 25044, full affinity, original flags, profiling off. `cuda-service-fused-pq2-permute-a.jsonl` measured:

| Prompt | Ingest tok/s | Output tok/s | Total seconds |
|---|---:|---:|---:|
| 512 | 392.881 | 34.512 | 8.7062 |
| 4096 | 516.971 | 32.425 | 15.7909 |

The nearest full-affinity control was 430.144/35.827 at 512 and 516.909/33.639 at 4096. All eight requests and generated token sequences match. There is no demonstrated gain; fewer compiled instructions alone are insufficient. The third measured round recovered to 35.45/33.84 output tok/s, so one repeat on this same PID will check whether early versus later measurements account for the difference. The original acceptance threshold remains unchanged.

### Experiment 019: repeat rejected; source reverted

The same-process repeat (`cuda-service-fused-pq2-permute-b.jsonl`) completed successfully with eight matching requests and token sequences. Medians: 512-token ingest 403.722 / output 34.322 tok/s; 4096-token ingest 514.185 / output 32.056 tok/s. This again fails the >=3% output improvement gate despite lower compiled instruction count. The paired-dot patch was reversed exactly against its preserved source patch; neither its CUDA code nor its binary is accepted for permanent deployment.

Operational correction: the benchmark file completed around 19:04 on September 22. At 20:46, process inspection showed only the service wrapper and server still running; the benchmark exit status was zero. Earlier chat updates incorrectly called the benchmark still running. The wrapper was keeping the idle server alive. No throughput samples were collected during that idle interval.

### Experiment 020 / 021: next build prepared

Applied the independently reviewed prefill-only four-column GDN patch and its eight reference test cases. Also applied the reviewed graph diagnostics and runner support; both diagnostic flags stay off for throughput acceptance. Graph diagnostics source review found no added CUDA API calls, events or synchronization and no changed graph decisions. Runner validation passed 23 mock cases across PowerShell 5.1 and 7.6, including restoration and conflicting log-mode checks. The next build excludes rejected experiment 019.

### Experiment 020 / 021: build and reference tests passed

The combined build completed successfully (`cuda-build-gdn-cols4-prefill-graph-stats.txt`). With both diagnostics disabled, CPU-reference checks passed 47/47 GATED_DELTA_NET, 101/101 PQ2 MUL_MAT and 40/40 PQ2 fusion cases. Each command exited zero with nonzero counts. Logs: `cuda-gdn-cols4-prefill-{gdn,mulmat,fusion}.{stdout,stderr}.txt`. The 14-file snapshot `tools/llamacpp-cuda-gdn-cols4-prefill` was SHA256-verified against build output. The normal service benchmark follows with all diagnostic flags disabled; runtime validation of enabled graph logging is separate.

### Experiment 020: first service trial fails; attribution unresolved

`cuda-service-gdn-cols4-prefill-a.jsonl` completed with exit zero and eight matching requests/generated token sequences. Diagnostics were disabled. Medians:

| Prompt | Ingest tok/s | Output tok/s | Total seconds |
|---|---:|---:|---:|
| 512 | 228.415 | 24.469 | 12.8134 |
| 4096 | 296.339 | 26.949 | 23.6085 |

This fails acceptance by a wide margin. Offline comparison confirms that both PQ2 decode kernels and the scalar S128 RAW COLS1 GDN kernels have exactly the baseline instruction mnemonic/operand sequences and register/local/stack counts. Binary encodings and scheduling-control bits were not compared. Four-column prefill variants exist with 72 registers and no spills.

During the slow run, an nvidia-smi sample showed SM 1890 MHz, memory 7301 MHz, GPU utilization 99%, memory utilization 13%, power 104.67 W and only 87 MiB reported free. Previous fast runs drew about 169 W at lower SM clocks. Per-process allocation counters were about 12.081 GB dedicated / 402.65 MB shared, versus 12.117 GB / 367.00 MB in the earlier control. These observations suggest a stall but do not prove paging or identify a cause. No unrelated application was stopped or changed. The retained binary is restored for a fresh control before attributing this slowdown to the source change.

### Fresh retained-control D under current conditions

`cuda-service-warp-scale-stablepath-d.jsonl` completed successfully with eight matching requests/tokens. Medians: 512 ingest 413.987 / output 32.123 tok/s, wall 9.1940s; 4096 ingest 498.872 / output 30.505, wall 16.6362s. The same retained binary previously produced 35.827/33.639 output tok/s in affinity control A2. Current conditions therefore also affect the retained control; candidate 020 remains unaccepted and its larger slowdown is still unexplained.

A separate Windows counter snapshot reported dedicated allocations of 11491.6 MiB for llama-server, 1205.0 MiB for DWM and 1009.4 MiB for NVIDIA Overlay, plus smaller applications. These counters are not a measurement of uniquely resident physical pages and must not simply be summed to infer paging. No unrelated process was stopped.

### Experiment 022: lower microbatch to test memory headroom — preregistration

The retained-control startup log reports model 6539.67 MiB, KV 3312.00 MiB, recurrent state 598.50 MiB and a 1010.28 MiB CUDA compute arena, before additional scratch-pool capacity. Source inspection shows the arena does not shrink for smaller decode graphs, while the VMM scratch pool retains its mapped high-water capacity until context destruction. Actual unused bytes remain unknown.

Test the retained binary with only microbatch lowered from 512 to 256. Keep batch 512, context 188416, automatic four slots, q4 KV, model and sampling unchanged. This preserves advertised context and slot capacity. Hypothesis: smaller reserved workspace provides enough headroom to improve output and stabilize ingestion. Compare exact service requests/tokens and GPU allocation counters against fresh control D. Require at least 5% output improvement without more than 2% ingest regression before retaining this configuration; otherwise record the tradeoff and reject or test a separate setting. This is a configuration experiment, not a claimed kernel speedup.


### 022 result: microbatch 256 rejected (2026-09-22)

Actual service run `cuda-service-warp-scale-ub256-a.jsonl`, PID 13800, monitor 18389, exited 0. Started 21:09:48 CEST and completed 21:11:55, 127 seconds. Eight of eight full request/generated-token pairs match fresh retained control D.

- 512 tokens: ingest 384.502 tok/s (-7.12%); output 33.758 tok/s (+5.09%); wall 8.8766s.
- 4096 tokens: ingest 503.004 tok/s (+0.83%); output 31.679 tok/s (+3.85%); wall 16.1292s.

The >=5% output/no >2% ingest-regression gate failed. Default UBatch remains 512. GPU allocation counters ended about 244 MiB lower; allocation counters alone do not establish physical residency or paging.

### 023 preregistration: lazy compute reservation

Source attribution from CUDA FA allocation: full-context FP16 K conversion 368 MiB + V conversion 368 MiB + F16 mask 184 MiB account for 920 MiB of the logged 1010.28 MiB compute arena. These are separate from persistent KV memory. Actual short-context views already shrink, but allocator capacity only grows. Candidate: opt-in split-only worst-case reservations, then synchronized allocation of actual graphs, preserving context 188416, automatic four slots, batch/ubatch 512, quantized KV, and logits features. Expected benefit: lower short-context arena allocation and potentially >=5% service ingest or output improvement, with no >2% regression in either. No throughput gain is assumed from the memory estimate. Runtime gates include graph recapture after growth, deterministic token agreement, four-slot/logits coverage and allocation-error-path review. Default must remain unchanged.

### 021 runtime diagnostic check started

Launching the existing prefill-only candidate snapshot with DEBUG_CUDA_GRAPH_STATS enabled and DEBUG_CUDA_TIMING cleared. This run validates instrumentation, not speed acceptance. Use 512/4096 token prompts, 32 generated tokens, one warmup and one measured round; require actual graph replay, valid parser output, and token-prefix agreement.


### 021 runtime validation passed; 020 source reset

The enabled graph diagnostic service run `cuda-service-graph-stats-a-requests.jsonl` exited zero and all four 32-token outputs match retained-control prefixes. Parser output `cuda-service-graph-stats-a-summary.json`: 148 executions, including 116 decode replays, 4 decode captures, 4 direct decodes; 23 direct prefills and 1 captured prefill. Five captures cost 30.751 ms host time total, two instantiations 11.421 ms, five updates 2.991 ms and two evictions 2.108 ms. Observed evicted ages were 53.374 and 53.156 seconds. This validates real replay and eviction logging without the timing mode that disables replay. Eight parser tests passed.

One diagnostic 4K request decoded at 12.30 tok/s despite normal earlier requests. Summed host lifecycle time across the entire run is only 47.271 ms, so the measured graph lifecycle costs do not explain that seconds-scale slowdown. This is diagnostic evidence, not a controlled performance acceptance run.

Experiment 020 source was reversed exactly using its preserved patch, with git diff proving gated_delta_net.cu equals retained HEAD. Its candidate snapshot and source patch remain available. Additional GDN reference cases are retained for validation; experiment 023 will isolate allocation changes from the failed GDN kernel specialization.


### 023 implementation and review gate

The lazy reservation patch and separate allocation-failure correction passed independent review for the target CUDA-plus-CPU scheduler path. Real scheduler/allocator CPU regression reproduced stock initial allocation access violation and growth assertion termination, plus a guard-only retry crash. Recreating failed allocation metadata and resetting the context graph yielded six passing success/failure/retry checks. Artifacts: `lazy-reserve-cpu-7p_289gy/guarded-recovery-results.json`; source patches `ggml-scheduler-allocation-failure-source.patch` and `cuda-lazy-compute-reserve-source.patch`. Partial allocation before a later backend failure and full context recovery remain uncovered. No claim is made about the pre-existing single-backend growth synchronization path.

Added an actual-service growth checker (512,4096,16384,512 plus four concurrent slots; 16 output tokens and top-five log probabilities). Independent review found intermediate /slots exchanges were not retained; v2 fixes correlated full exchange/error logging and preserves completion artifacts once. Twelve mock tests pass. The original baseline run was already running; its full completion responses remain suitable as correctness reference if it passes.

User-proposed next hypothesis: compile out irrelevant quantization kernels. Existing build already uses only SM86 and disables most flash-attention quantization variants. The retained CUDA DLL is 129337344 bytes on disk, which is not a GPU residency measurement. Bounded discovery is underway to retain exactly the model/cache/activation types required before creating an isolated build comparison.


### 023 build/reference gates passed; enabled service test started

The canonical CUDA build completed exit 0 (`cuda-build-lazy-compute-reserve.txt`). Integrated CPU allocation regression passed six checks. CUDA correctness passed 47/47 GDN, 101/101 PQ2 MUL_MAT and 40/40 PQ2 fusion cases with nonzero counts, diagnostics disabled. The 11-file `tools/llamacpp-cuda-lazy-reserve` service snapshot was hash-verified; hashes in `cuda-lazy-reserve-binary-hashes.json`.

Retained-server growth reference completed all eight cases, including exact repeated-512 consistency and four concurrent slots, with full completion tokens/log probabilities preserved in `cuda-lazy-reserve-growth-baseline.jsonl`. During this correctness run, retained prefill fell near 60 tok/s and nvidia-smi showed 59 MiB free; no causal attribution is made.

Starting lazy-reservation-enabled service with graph lifecycle logging for functional comparison and actual growth/replay checks. This diagnostic run is not throughput acceptance. Upstream discovery also found the unchecked reserve result already reported in ggml-org/llama.cpp#27817 and guarded upstream by #26070; Prism lacks that backport. Retry invalidation is a separate local correction, reproduced on this Prism-derived source. No duplicate upstream issue was filed.


### 023 enabled growth trial: probability gate failed, candidate not accepted

`cuda-lazy-reserve-growth-enabled-a.jsonl` exited 1. All eight exact requests and generated-token sequences match the retained reference; all four sequential growth cases and concurrent slots 0/1 also pass the strict probability comparison. Concurrent slots 2/3 differ at first-token log probability: -0.04377519 vs -0.03808111, and -0.04053763 vs -0.04424564 (absolute tolerance 1e-4). Do not relax this gate without control evidence. Next compare concurrent reference repeats to distinguish schedule-dependent baseline variation from allocation effects.

Replay diagnostics parsed successfully: 128 executions with 52 decode replays and 12 prefill replays; six captures, three instantiations, six updates, two evictions. Source: `cuda-service-lazy-reserve-graph-a.log` and its summary. A post-start GPU snapshot showed 775 MiB free vs the earlier retained-run 59 MiB snapshot; these are separate observations, not a controlled residency measurement or throughput gain.


### 023 same-binary disabled control A

`cuda-service-lazy-reserve-off-a.jsonl` (PID596, monitor40067) completed exit0 with eight exact request/token matches against retained control D. LLAMA_LAZY_COMPUTE_RESERVE=0, both diagnostics off. Medians: 512 ingest307.688/output31.354 tok/s, wall9.8079s; 4096 ingest359.740/output30.350, wall19.7915s. These are a fresh control under current conditions, not a claimed improvement.

Independent analysis confirmed baseline concurrent admission order0,1,3 then2; enabled order0,1,2 then3. Only slots2/3 probability comparisons failed, while all sequential cases and slots0/1 had bit-identical full probability records. Baseline identical concurrent prompts already vary across slots by up to0.02662753 chosen-token log probability. This establishes a scheduling confound, not candidate acceptance. Native multi-prompt /completion queues all four tasks under one mutex before slot updates; a controlled atomic four-prompt check will retain the1e-4 gate.

The reviewed concurrent-only checker extension passed16mocktests and preserves full-suite behavior, full request/response audit, and four/eight-case reference validation.
