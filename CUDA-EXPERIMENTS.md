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


### 023 concurrent disabled-control replay also fails probability reference

`cuda-lazy-reserve-concurrent-off-a.jsonl`, monitor53293, exited1 with LLAMA_LAZY_COMPUTE_RESERVE=0. Slots0/3 passed; the remaining cases failed strict probability comparisons, including first-token -0.04076955 vs -0.03808111 and top-token id248046 -3.61306787 vs -3.59817123. This confirms the HTTP-admission probability comparison is confounded even with the allocation change disabled. It does not by itself pass the candidate.

Implementing an atomic four-prompt native /completion mode that enqueues all requests together, omits shared id_slot, verifies full index/slot sets, and compares by actual slot with the existing1e-4 probability tolerance. Capturing enabled throughput with diagnostics off while controlled correctness validation is prepared; no acceptance decision until both gates pass.


### 023 first enabled throughput trial passes numerical speed gate; acceptance pending

`cuda-service-lazy-reserve-on-a.jsonl` (PID24388, monitor10175) completed exit0. Samebinary/configasoffA, onlylazyreservationenabled; bothdiagnosticsoff. Eightrequest/generated-tokenpairsmatch. Medians:512ingest402.888/output33.869tok/s, wall8.8119s;4096ingest501.782/output32.012, wall16.1327s. AgainstoffA:+30.94%/+39.48%ingest and+8.02%/+5.47%output;wall-10.15%/-18.49%.

These gains exceedthepre-registeredthresholds, butthecandidateisnotyetaccepted: controlledfour-slotprobabilitiesandrepeatmeasurementsremain. Enabledingestspeedsareclose toearlierbestretainedruns,sotheobservedgainmayreflectavoidingcurrentmemory-relatedslowdownratherthanraisingthepreviouspeak. NextmeasurementorderisoffA,onA,onB,offBwiththesamebinary; noadditionalGPUworkrunsalongsideeachperformancebenchmark.


### 023 measurement details and controlled concurrency tooling

The first enabled run measured 402.888 ingest / 33.869 output tok/s at 512 tokens, and 501.782 / 32.012 at 4096. The disabled control measured 307.688 / 31.354 and 359.740 / 30.350 respectively. All eight request and token pairs match; controlled concurrency and repeat measurements remain pending.

At the final 4K sample, disabled control process allocation counters were 11503.57 MiB dedicated plus 402.00 MiB shared; enabled counters were 10744.99 plus 106.00 MiB, about 1054.59 MiB lower in total. Both samples reported GPU temperature 84 C, SM clock 1777 MHz, memory clock 7301 MHz and power about 169 W. Allocation counters still do not prove unique physical residency or paging.

The atomic four-prompt checker passed independent review and 22 CPU mock tests, including malformed indices/slots, index-to-slot remapping, exact bulk-request matching, probability differences, and atomic/HTTP reference isolation. It requires --concurrent-only and preserves all previous 16 tests.

### 024 preparation: trace-guided prefill FFN fusion eligibility

The recorded 508-token profile has 64 separate PQ2 gate/up/SwiGLU groups with K=5120 and 17408 output rows. A fused kernel could remove duplicate activation quantization and about 8.43 GiB of intermediate write/read traffic per 508-token graph. This is a logical traffic estimate, not a measured speed gain. Dual accumulators may increase registers and spill; nonlinear SwiGLU must follow complete accumulations, never partial split-K sums.

A separate DEBUG_CUDA_FFN_FUSION diagnostic patch is under independent review to verify actual structural eligibility and tensor consumers. It adds no CUDA calls, synchronization or execution changes. Actual MMQ dispatch, tile and stream-K are explicitly unknown rather than reconstructed. This patch is not part of the current allocation experiment binary.


### 023 enabled repeat B completed

`cuda-service-lazy-reserve-on-b.jsonl` (same PID24388, monitor7483) completed exit0 with eight matching request/token pairs. 512 tokens: ingest 399.436, output 34.889 tok/s, wall 8.6056s; 4096 tokens: ingest 508.399, output 32.886 tok/s, wall 15.8174s.

Repeat disabled control B follows to complete off/on/on/off measurement order. Controlled atomic four-slot probability checks remain required.


### 023 off/on/on/off throughput comparison completed

Disabled repeat B (`cuda-service-lazy-reserve-off-b.jsonl`, PID22696, monitor92621) exited0: 512 ingest280.687/output32.565 tok/s, wall9.7073s; 4096 ingest328.317/output30.778, wall20.7975s. All32 request/generated-token pairs across the four runs match.

Pooled medians over six measured repetitions per condition and prompt length:

| Prompt | Disabled ingest | Enabled ingest | Change | Disabled output | Enabled output | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 294.551 | 401.162 | +36.19% | 32.438 | 34.318 | +5.80% |
| 4096 | 340.892 | 503.881 | +47.81% | 30.665 | 32.440 | +5.79% |

The speed gate passes in this current environment. These rates recover ingestion toward earlier best measurements; do not claim a 48% increase over the historical best. Final acceptance still requires the controlled atomic four-slot probability check. Diagnostic replay logging is enabled only for that correctness comparison, separately from the completed performance trials.

### 025 preregistration: compile fixed FFN dimensions into decode

The user suggested compilation guided by actual execution. Existing fused PQ2 SASS retains runtime K/32 setup, five loop increments/comparisons/branches for K5120, and signed division/remainder address work. The inner four-element dot loop is already unrolled. Test a guarded SM86 fused-decode specialization only for K5120 and17408 rows, keeping four warps/four rows per CTA, sequential floating-point accumulation, existing permutations and exact math. Other shapes keep the current path. Avoid the larger seventeen-iteration K17408 unroll initially because register and code-size growth may regress performance. Require at least3% service output gain with no more than2% ingest regression, matching tokens, CPU-reference CUDA checks, and SASS/register inspection. Source preparation runs separately; no candidate build or GPU work overlaps performance trials.


### 023 accepted: controlled atomic correctness passed

Both atomic four-prompt runs completed exit 0: `cuda-lazy-atomic-off-a.jsonl` and `cuda-lazy-atomic-on-a.jsonl`. Same candidate binary, lazy flag 0 versus 1, context 188416, automatic four slots, batch/ubatch 512, Q4 KV. Exact requests, generated tokens, content and top-five log probabilities pass the unchanged absolute 1e-4 tolerance for all four actual slots. Graph execute widths after saved pre-request offsets are identical: 496, 16, then fifteen width-4 executions. Evidence: `cuda-lazy-atomic-graph-comparison.json` and the paired graph logs. This removes the prior HTTP admission confound without relaxing the probability gate. The checker has 22 passing mock tests and independent review.

One analysis attempt used the wrong baseline offset key and accidentally included two startup executions. It failed the sequence assertion; correcting the key from byte_offset to the recorded offset_bytes yielded the identical request-only sequences above. No service trial was rerun or altered for this analysis correction.

Combined with the sequential growth/repeated-prompt checks, CUDA reference tests, allocator failure/retry regression and completed off/on/on/off throughput trials, experiment 023 is retained as the second validated service win. Pooled current rates: 512 tokens 401.162 ingest / 34.318 output tok/s, 4096 tokens 503.881 / 32.440. Versus paired current disabled controls, +36.19% / +47.81% ingest and +5.80% / +5.79% output. Ingest recovers earlier peak territory rather than exceeding the historical best by those percentages. Workspace grows on demand and remains grown after large requests; full-context memory demand is not eliminated. The persistent Prism launcher will opt in to LLAMA_LAZY_COMPUTE_RESERVE=1; the library default remains off.

### 024/025 implementation review before compilation

The FFN diagnostic v2 passed independent source review after canonical operand-order eligibility was added. It is inert unless DEBUG_CUDA_FFN_FUSION is exactly 1. Runtime eligibility is still unproven.

The exact K5120/17408-row fused PQ2 decode unroll passed independent spec and quality review. The generic loop, eligibility guards, ordered accumulations, four-warps/four-rows layout and epilogue are preserved. Four exact-shape reference tests cover SWIGLU/GEGLU with biases off/on. Source and static proof are preserved as `cuda-pq2-k5120-unroll-source.patch` and `cuda-pq2-k5120-unroll-static-proof.json`. No speed claim before compilation, CUDA correctness, SASS inspection and actual-service comparison.


### 025 build and CUDA reference checks passed

Build completed exit 0 in `cuda-build-pq2-k5120-unroll.txt`. CUDA fusion passed 44/44 cases, including all four exact 5120 x 17408 specializations; PQ2 MUL_MAT passed 101/101. The allocator failure/retry suite passed 6/6. Full stdout/stderr artifacts are `cuda-k5120-{fusion,mulmat,alloc}.*.txt`. The 11-file candidate snapshot `tools/llamacpp-cuda-k5120-unroll` was hash-verified against build output (`cuda-k5120-unroll-binary-hashes.json`). The candidate includes the separately reviewed FFN eligibility diagnostic, disabled during performance tests.

The persistent Prism launcher opt-in was applied after verifying its exact pre-change backup SHA256, and its PowerShell AST parses without errors. Lazy allocation code/checker/log integration was committed and pushed as 2110377e. Supervisor maintenance remains active during the next controlled trial window.


### 024 diagnostic runtime result: partial fusion eligibility

`cuda-ffn-fusion-k5120-diagnostic-requests.jsonl` completed exit 0 on the candidate service with diagnostics enabled. Both 32-token output prefixes match the retained service. In each of two 508-token graphs, all 64 FFN groups were found: 40 pass structural/memory eligibility, 24 fail memory_overlap. Width 1/2/4 groups are excluded. Evidence: `cuda-ffn-fusion-k5120-diagnostic.log` and `cuda-ffn-fusion-k5120-diagnostic-summary.json`. The original all-64 logical traffic-saving estimate is therefore an upper bound, not the immediately eligible scope. Diagnostic rates are not throughput acceptance.

### 025 generated-code risk and service comparison start

cuobjdump reports 100 registers for the specialized fused kernel versus 37 for the retained/generic fused kernel; both have zero stack/local/shared usage in resource metadata. Unrolling can increase live values and reduce occupancy; no performance conclusion follows without service measurement. Fresh retained control A starts with lazy allocation enabled, diagnostics off and unchanged production arguments. Candidate comparison will use the same conditions and preserved snapshot.


### 025 generated-code analysis completed

`cuda-pq2-k5120-sass-comparison.json` records exact binary hashes, resource counts and FFMA chains. The specialized fused kernel removes runtime K loading and the loop back-edge while preserving five sequential dependent accumulator updates. Both generic fused and unfused SASS instruction/address text remain identical to the retained snapshot. Static instruction slots grow from 248 to 768 and code size from 3968 to 12288 bytes. There are no local-memory load/store instructions or reported spills.

The compiler hoisted all 95 dot-product global loads (five groups of 19) before the first accumulation FFMA at 0x1230, increasing registers from 37 to 100. Using local cuda_occupancy.h register rounding and SM86 resource limits, the static ceiling falls from 12 CTAs / 48 warps to 4 CTAs / 16 warps for the 128-thread launch. This is inferred occupancy, not a measured runtime count. The initial SASS command failed because nvdisasm was absent from PATH; the corrected run used the tool-local NVDISASM_PATH and retained the failed artifact. All such analysis is separate from throughput acceptance.


### 025 fresh control A completed

`cuda-service-k5120-control-a.jsonl` completed exit 0 (PID 2584, monitor 54475). All eight exact request/token/content records match the retained lazy-enabled reference. Medians: 512 ingest 405.586 / output 35.025 tok/s, wall 8.5503 s; 4096 ingest 516.275 / output 32.797, wall 15.7142 s. Candidate A follows with the same launcher flags, lazy allocation enabled and all diagnostics off.


### 026 preregistration: staged prefill FFN fusion

The 24 excluded T508 groups reuse the activation address as the GLU output (10,403,840 overlapping activation bytes); preserve the canonical memory rejection. The 40 eligible groups also have mutually disjoint gate/up/GLU outputs. First test a staged fusion: quantize shared activation once, retain ordinary gate MMQ, and compute up with a dedicated complete-K tile whose final store applies SiLU(gate) * up. This removes one activation quantization, the standalone GLU launch, and up intermediate write/read, approximately 2.635 GiB logical traffic over 40 groups. Gate traffic and graph-reserved buffers remain.

Restrict to SM86, contiguous PQ2 K5120/17408-row weights, shared contiguous F32 activation, T508 or512, plain SwiGLU without bias/scales, singleton higher dimensions, exclusive intermediates and validated disjoint ranges. The proposed up tile uses I128/J128, 256 threads and a136 x4 grid, consuming all40 PQ2 K-blocks before nonlinear writeback. This changes up scheduling from generic stream-K to one complete-K CTA per output tile; never apply nonlinear activation before partial-sum fixup. Gate dispatch remains unchanged. Independent design review precedes implementation.

Planning prediction is 1-3% prefill improvement, no decode improvement, with possible regression from scheduling/register effects. Acceptance requires repeated actual-service comparisons with at least2% improvement in both512 and4096 ingest medians and no more than2% output regression, exact generated tokens/content, CPU-reference fused-graph checks and service growth/concurrency checks. All diagnostics must be off for timing. A fully paired gate/up accumulator design is deferred because its doubled accumulator/shared-memory cost is substantially higher.


### 025 rejected: no service output gain

`cuda-service-k5120-candidate-a.jsonl` completed exit 0 (PID23592, monitor37680). All eight exact requests, generated tokens and content match fresh control A. Measured medians:

| Prompt | Control ingest | Candidate ingest | Change | Control output | Candidate output | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 405.586 | 412.774 | +1.772% | 35.025 | 34.904 | -0.347% |
| 4096 | 516.275 | 508.152 | -1.573% | 32.797 | 32.830 | +0.101% |

Wall medians regressed 0.532% and0.871%. The preregistered >=3% output gain failed by a wide margin; do not claim a win or keep the specialization. Source mmvq.cu was reversed with its exact saved patch and verified identical to retained HEAD. Exact-shape correctness cases remain useful coverage. Candidate source, static proofs, disassemblies, build output, binary snapshot and all trials are preserved. No further candidate timing repeats are needed to retain a change that has no demonstrated gain.

The stable executable path was redeployed from the accepted lazy-reserve snapshot; deployment receipt `bonsai-deploy-20260922T221853575-a4199de488374a2194e3569d155b3569.json`. The original supervisor was restored byte-for-byte to SHA25654af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be and its scheduled task started. The permanent Prism launcher now enables lazy reservation. Ordinary service verification follows while next-candidate source work continues.


### 025 rollback verification and retained coverage

The ordinary supervised server is healthy on port8090 (PID22400); LlamaSupervisor returned0 and logs confirm four slots with188416 context per slot. All11 files at the stable deployment path match the accepted lazy-reserve snapshot hashes. The scheduled process command line is inaccessible from this user token, so binary provenance is established by launcher output and deployed hashes instead. The launcher explicitly sets the lazy opt-in before invoking Prism.

The new44-case fusion suite was additionally run from an isolated test-executable directory with retained lazy-reserve DLLs on PATH. All44/44 pass, including the exact-shape cases, after reverting the unroll. Artifacts: `cuda-k5120-retained-fusion.stdout.txt` and stderr. This validates retained-kernel coverage independently of the rejected specialization.

Experiment026 independent design review confirms complete-K ordering and fragment addressing, requires a57344-plus-ID shared allocation (57856 bytes from mmq_get_nbytes_shared), explicit T508 tail guards and a shared-memory attribute on the new kernel symbol, same-stream Q8 lifetime, and pairwise output-disjointness. The recorded T508 graphs do not prove T512 eligibility; both widths require runtime/reference checks. On this28-SM GPU, the current544-tile stream-K configuration already assigns544 full-K blocks at97% tile efficiency, so the proposed up wrapper need not change accumulation order for this geometry.


### Supervised service correctness confirmed

`cuda-lazy-supervised-atomic.jsonl` passed all four slots against the controlled atomic reference with exact requests/tokens/content and unchanged probability tolerance. This verifies the ordinary supervised launcher after rollback, separately from the benchmark runner. Service health is OK; no candidate unroll is deployed. Experiment026 implementation continues in an isolated worktree while the accepted service remains available.


### Reusable strict service comparison tooling

Added `scripts/bonsai-server-compare.py` for complete JSONL validation, exact cross-condition requests/tokens/content, pooled measured medians and optional per-prompt speed/regression gates. It excludes warmups, recomputes rates from counts/timing durations, rejects incomplete or duplicate artifacts, records source hashes and preserves failure details in an exclusive JSON result. Eleven CPU test methods passed both implementation and independent review, then again after integration. The independent four-run023 smoke recomputed all recorded gains with six measured samples per condition/size.

Applied the same tool to025 fresh control/candidate A. It passed correctness and returned exit1 specifically for the3% decode performance gate; both prompt sizes failed. Full comparison is `cuda-service-k5120-comparison.json`. No benchmark was rerun to generate these analysis artifacts.


### Ingestion acceptance gates added to comparison tool

The comparator now also supports paired --min-ingest-gain-pct and --max-decode-regression-pct, mutually exclusive with the existing decode-focused pair. Existing positional API and JSON fields remain available; results identify the selected axis. Incomplete/mixed/nonfinite/negative thresholds are rejected, and every prompt must pass both bounds. Fifteen CPU tests pass, including the original11, boundary/failure/undefined-decode cases and output artifacts. Independent review passed; the023 four-artifact smoke passes the2% ingest /2% output-regression gate. Experiment026 will use this same tool with its preregistered thresholds.


### 026 source review: CUDA passed, test-construction blocker found

Independent review of v2 passed the target CUDA implementation: Q8 layout/padding, complete40-block accumulation,57856-byte shared configuration, existing fragment mapping/tail bounds, same-stream buffer lifetime, narrow eligibility and generic fallback. Added successful staged_launch records under existing DEBUG_CUDA_FFN_FUSION to prove dispatch during host submission/capture; replay does not re-enter that host path.

Review found a test-only null graph hazard before any candidate build: the reversed-input test called ggml_build_forward_expand on member gf, while eval_perf constructs the graph before initializing its separate local graph. Filtering follows construction, so an unrelated perf selection could reach this null dereference. This is a source finding, not a reproduced runtime crash. A minimal initialized-member guard is required before compilation; original v1/v2 patches remain preserved. Eight CPU-reference cases have been added but are not yet executed.


### 026 v3 source gate passed; build started

The reversed-input test now pre-expands only when mode is MODE_TEST and member gf is non-null. Independent delta review confirmed that correctness evaluation initializes both before construction and retains the required node ordering, while performance mode skips the unsafe pre-expansion. Final source spec/quality review passed. V3 was applied after a clean patch check; v1/v2/v3 remain preserved. The canonical CUDA build is running with complete output reserved exclusively at `cuda-build-prefill-staged-ffn.txt`. The accepted supervised service remains available during compilation; no performance measurement runs alongside the build.


### 026 build and GPU correctness passed with observed dispatch

Canonical build exited0 (`cuda-build-prefill-staged-ffn.txt`). All8 PQ2_STAGED_FFN whole-graph CPU-reference cases passed on CUDA0. The diagnostic emitted exactly two staged_launch records, one each for508 and512 tokens; the six fallback cases emitted none. Existing PQ2 fusion passed44/44 and MUL_MAT101/101; allocator success/failure/retry passed6/6. A deliberately nonmatching perf selector completed graph construction and exited0 without executing throughput cases, checking the reviewed null-graph fix. Artifacts are `cuda-staged-ffn-{staged,fusion,mulmat,alloc,perf-construction}.{stdout,stderr}.txt`.

The11-file `tools/llamacpp-cuda-prefill-staged-ffn` snapshot matches build hashes in `cuda-prefill-staged-ffn-binary-hashes.json`. The authorized maintenance window was reopened after inspecting the supervised wrapper/launcher/server command lines in session0 and verifying all slots idle; receipt `cuda-maintenance/stopped026.json`. Fresh accepted-build control A is now measured on the actual service with lazy allocation enabled and all diagnostics off. No builds or GPU-reference tests overlap these performance requests.


### 026 generated code passed inspection

`cuda-prefill-staged-ffn-sass-comparison.json` and related control-flow/kernel artifacts identify the new staged symbol. It uses250 registers versus254 in retained generic PQ2 J128, with zero reported stack/local allocation and no LDL/STL spills. Both round to one256-thread CTA per SM under the register budget; this is inferred occupancy. Static instructions/code bytes are3584/57344 versus5520/88320, which is not a dynamic instruction or speed measurement. Candidate generic J128 instruction text is identical to retained.

The full-K loop starts0, increments2 and exits40; SiLU/gate multiplication follows loop completion, and tail predicates guard gate loads/output stores. Dynamic shared memory remains57856 bytes. Generic stream-K can in general use a different reduction order, so strict actual-service output/probability checks remain mandatory even though the target geometry is expected to use complete-K tiles.


### 026 fresh control A completed

`cuda-service-staged-ffn-control-a.jsonl` completed exit0 (PID21524, monitor23665). Medians:512 ingest425.149/output34.872 tok/s, wall8.5198s;4096 ingest512.417/output32.994, wall15.7160s. Candidate A starts with the identical production arguments, lazy allocation enabled and diagnostics off. Timing is preliminary until strict service correctness and repeat comparisons pass.


### Sequential growth correctness mode added

The canonical checker now supports --growth-only for the unchanged512/4096/16384/512 sequence. This separates allocator/growth correctness from the previously demonstrated HTTP-admission confound; atomic four-slot checks remain a separate gate. It accepts passing full8-case or explicitly marked4-case growth references, rejects incomplete/incompatible references and conflicting modes, and keeps exact request/token/content comparisons plus the1e-4 probability tolerance. All28 CPU tests pass, including22 existing cases; independent spec/quality review passed. No HTTP or GPU calls occurred during the tool tests.


### 026 first candidate throughput trial did not pass

Candidate A (`cuda-service-staged-ffn-candidate-a.jsonl`, PID25564, monitor91419) completed exit0. Comparator passed all8 exact request/token/content pairs and returned exit1 for the performance gate. 512 ingest400.237/output34.584 tok/s, wall8.6990s: -5.860% ingest/-0.827% output versus control A. 4096 ingest509.359/output32.595, wall15.8685s: -0.597%/-1.211%. Evidence: `cuda-service-staged-ffn-comparison-a.json`. No win is claimed.

Thermal conditions differed: control A began at39 C and ended83 C; candidate A began55 C and ended85 C, with measured candidate samples at somewhat lower SM clocks. This is a recorded confound, not proof that temperature caused the difference. Strict diagnostic growth/atomic correctness comes next. Any close performance conclusion requires repeat controls under comparable warmed conditions; thresholds remain unchanged.


### 026 strict service correctness and live dispatch passed

`cuda-staged-ffn-growth-correctness.jsonl` passed512/4096/16384/512 against retained reference with exact requests/tokens/content and unchanged1e-4 probability tolerance, including repeated512 consistency. `cuda-staged-ffn-atomic-correctness.jsonl` passed allfour slots; its execute-width sequence exactly matches the controlled atomic reference. Logs show1680 successful staged host submissions:160 at508 and1520 at512. Every submission maps to an eligible non-overlapping candidate. Evidence: `cuda-staged-ffn-dispatch-and-atomic-proof.json`.

Graph diagnostics show125 executions, including13 prefill and52 decode replays, six captures, three instantiations, six updates and three evictions. This is functional/capture evidence, not speed acceptance (`cuda-staged-ffn-growth-graph-summary.json`).

### 026 warmed repeat protocol before further timing

To resolve the first pair's temperature difference, each fresh process gets the same additional unscored four-request conditioning pass: existing benchmark,512/4096 prompts,256 outputs,reps1. Immediately afterward run the normal eight-request benchmark with its own warmup and three measured repetitions. Preserve conditioning artifacts separately and exclude them from acceptance medians. First order is candidate B then retained control B. If that pair is promising, mirror the order for another pair; otherwise reject without claiming the initial difference was temperature-caused. Thresholds remain >=2% ingest at both lengths and <=2% output regression. Both diagnostics and other GPU work remain off during timing.


### MMQ register-pressure follow-up rejected before implementation

Read-only source/SASS analysis attributes part of retained254-register pressure to64 persistent FP32 accumulators,32 A-fragment registers plus16 scales, and18 registers prefetching the next Q8 half-tile while computing the current half. Concrete retained SASS lifetimes include A fragment R96-R99 from0x2160 through0x53f0, and prefetched R158 from0x22e0 through0x5770. This is not an exact partition of all registers.

A K-major source loop might shrink A/scale live storage while preserving accumulation order, but would constrain compiler scheduling/add loop control. With no spills and57856 bytes shared per CTA, register reduction alone still leaves one CTA per SM. No credible positive service-gain prediction was established, so no code/build/performance trial is scheduled for this hypothesis. Existing retained SASS artifact: `cuda-prefill-staged-ffn-sass-retained-selected-complete.txt`.


### 026 warmed candidate B completed; retained control B running

Candidate B completed exit 0 with conditioning and measured artifacts kept separately: `cuda-service-staged-ffn-candidate-conditioning-b.jsonl` and `cuda-service-staged-ffn-candidate-b.jsonl`. Measured medians: 512 ingest 389.256 / output 34.867 tok/s, wall 8.6316 s; 4096 ingest 505.964 / output 32.718, wall 15.8931 s. These are candidate observations, not an accepted gain. The matching warmed retained control B is now running on actual service PID 23724 with the identical request protocol and diagnostics off. The accepted snapshot's CUDA DLL hash was verified before deployment; runner deployment preserves the previous candidate folder for rollback.

### 027 preregistration: prune unused CUDA weight-format instantiations

Read-only model inventory found 353 F32, 96 BF16 and 402 PQ2_0 tensors, with Q4_0 K/V caches. Prepare optional GGML_CUDA_BONSAI_ONLY, default OFF, retaining PQ2_0/Q4_0 MMQ and MMVQ weight instantiations plus all F32/F16/BF16, Q8_1 activation, attention, conversion, GDN and control paths. Exclude 22 unused MMQ translation units and corresponding MMQ/MMVQ dispatch branches; advertise only supported matrix weight types when opted in. This is a bounded test of the user's code-pruning hypothesis.

Prediction: lower build time and CUDA binary/device-code section size; low confidence in VRAM or throughput benefit because CUDA lazy loading may already omit unused kernels from resident device code. A smaller DLL alone will not count as a memory or speed win. Before service timing, require omitted functions absent from SM86 device code and retained PQ2 SASS/resources unchanged, then CPU-reference and strict service growth/atomic correctness. Compare memory only after matched initialization and shape warmup with equal tensor/pool allocations. Speed acceptance requires repeated actual-service pairs showing at least 2% improvement on the target phase at both prompt sizes with no more than 2% regression in the other phase; exact outputs remain mandatory. Source preparation runs in an isolated worktree without compilation or GPU activity during experiment 026 timing.


### 026 rejected after warmed service comparison

Retained control B completed exit 0; all eight corresponding requests, generated tokens and content match candidate B. The comparator returned exit 1 specifically for the performance gate (`cuda-service-staged-ffn-comparison-b.json`). Results exclude both conditioning and normal warmups:

| Prompt | Retained ingest | Candidate ingest | Change | Retained output | Candidate output | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 406.756 | 389.256 | -4.302% | 34.727 | 34.867 | +0.402% |
| 4096 | 510.065 | 505.964 | -0.804% | 32.575 | 32.718 | +0.438% |

Wall medians regressed 0.316% and 0.403%. Candidate measured samples began at 73-74 C and ended at 83-86 C; control began at 71-75 C and ended at 82-86 C. Conditions are closer than pair A, but this is not proof of equal instantaneous clocks throughout each request. Neither pair meets the preregistered ingest gate. Reject without further repeats or a win claim.

The exact v3 patch was reversed for the three CUDA files only, then git diff confirmed they equal retained HEAD. Candidate source, binaries, SASS, diagnostics and all benchmark artifacts remain preserved. The eight whole-graph PQ2 reference cases remain useful coverage: all 8/8 passed again using the candidate test executable isolated from candidate DLLs and the accepted lazy-reserve DLLs on PATH (`cuda-staged-ffn-retained-reference.stdout.txt` and stderr). The first attempt asserted an incorrect bin/Release executable path before creating artifacts or starting a test; the canonical Ninja build script established bin/test-backend-ops.exe, and the corrected run exited 0.

All 11 stable deployed files match the accepted snapshot hashes. The original supervisor was restored byte-for-byte to SHA256 54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be; the maintenance prefix remains in `cuda-maintenance/llama-supervisor.maintenance-before-restore-026.ps1`. The ordinary scheduled service is healthy on port 8090, PID 5320, with four slots and context 188416. Its permanent Prism launcher retains LLAMA_LAZY_COMPUTE_RESERVE=1. There are still two validated optimization wins.

### Partial decode unroll follow-up rejected before implementation

Retained fused K5120 SASS executes five iterations of 138 instructions; only three per iteration are explicit loop control (increment 0x3e0, compare 0x3f0, backedge 0x9e0). An ideal factor-two loop with a peeled tail saves nine of 690 loop instructions, about 1.3% of loop issue count before unaffected reduction, epilogue and other decode work. This is not a latency bound. All 19 loads per iteration already precede dot arithmetic, and partial unrolling exposes more values to the load-hoisting behavior observed in 025. Retained 37 registers leave three before crossing the 40-register allocation tier; 41-48 registers infer 40 rather than 48 resident SM86 warps. No credible >=2% service-output prediction was established, so no implementation/build/runtime test is scheduled. Evidence: `cuda-pq2-k5120-retained-sass-resources-complete.txt`, `cuda-pq2-k5120-sass-comparison.json`, preserved 025 patch, and mmvq.cu/vecdotq.cuh source.

### 027 source prepared for independent review

The five-file optional Bonsai-only candidate is preserved at `cuda-bonsai-only-source.patch`, SHA256 5b8e54691f63603a6eda895602f9ff9cacb87c70d998a47086405fade0ced89f. Author source checks report exactly two retained MMQ translation units, aligned MMQ/MMVQ guards and default-OFF token equivalence. These are source checks, not a successful build. Independent review is running before integration or compilation. For clarity before measurement, the primary service gate for this memory-headroom hypothesis is >=2% output gain at both prompt lengths with <=2% ingest regression; gains in DLL size alone cannot satisfy it.


### 027 source review correction and reproducible build control

Initial independent review found the MUL_MAT_ID path calls get_mmvq_mmid_max_batch directly, bypassing should_use_mmvq. Scheduler support rejection protects normal placement, but the helper still returned a positive batch limit for removed types. The revised source returns zero for non-PQ2_0/non-Q4_0 types under the option, also covering synchronization prediction. This is a source-review finding, not a reproduced Bonsai runtime failure.

The canonical build script now has a trailing -BonsaiOnly switch and explicitly configures GGML_CUDA_BONSAI_ONLY=ON or OFF every time, preventing stale cache state from changing later controls. Author parsing, OFF/ON/OFF mock configuration, default-OFF source-equivalence and ON-whitelist checks passed; independent delta review follows. V1 and V2 remain preserved; V3 is `cuda-bonsai-only-source-v3.patch`, SHA256 28969c58845e45782a226bced6d80aa33be5d2aab2f1c1e4b0482ae6bec6f602. Its patch check against current retained source passed. Separate clean OFF/ON build directories and exclusive log paths are unused, with no compiler processes already running.


### 027 final source review passed; clean baseline build started

Independent spec and quality review passed V3, including the MMID helper fix, default-OFF equivalence, PowerShell parsing, explicit false, and mocked OFF/ON/OFF configurations. Applied the reviewed six-file patch to root after its clean check; whitespace validation passed. Clean OFF compilation is running through the canonical script in build-bonsai-only-off (monitor 21155); ON will use a separate clean build-bonsai-only-on directory. Complete monitor output will be retained as build artifacts when each build finishes. The accepted ordinary service stays available during compilation, and no throughput benchmark overlaps compilation.

The retained-format runtime matrix is source-derived and must be confirmed by nonzero executed counts: PQ2 matmul 101, PQ2 fused decode 44, whole-graph PQ2 FFN 8, small Q4_0 matmul 3, F32/F16/BF16 matmul 9, and Q4_0 attention 2 cases per build. The two attention cases cover vector Q4 attention and the F16 MMA path with Q4 conversion. Counts and selectors have not yet been runtime-verified for this candidate. Strict service growth/atomic checks and primary output-speed acceptance follow only after these gates.


### PQ2 separate-plane repack rejected before implementation

Read-only source/SASS review found naturally aligned U16/S16 weight loads at retained fused-loop addresses 0x01f0-0x0350. Its 19 loads comprise eight quant-payload halfwords, two weight scales and nine Q8 activation loads; existing permutations decode 2-bit symbols rather than repair misaligned loads. Separate payload planes might permit fewer wider loads, but require extraction and different register scheduling. CPU enumeration of the target lane addresses found nine unique 32-byte sectors per weight/warp iteration in either layout (current interleaving nine; separate planes eight quant plus one scale). This does not establish a reduction in DRAM bytes or measured stalls.

A same-size repack would require coordinated translation for arbitrary chunk and 2D uploads/readbacks, D2D copies, parent-offset views, CPU copies and GGUF tensor_get serialization, as well as MMQ/MMVQ/dequant/getrows consumers. Temporary conversion storage can avoid permanent duplicated weights, but does not remove those format contracts for the served model. No bounded change with a credible service gain was established; no implementation, build or runtime test is scheduled. Evidence: retained SASS resource artifact, vecdotq.cuh, ggml-cuda.cu buffer callbacks, ggml-backend.cpp views/copies, llama-model-loader.cpp chunking and gguf.cpp writer.

### Need for graph-preserving profiling recorded

The existing diagnostic decode profile contains seven graphs at 325.369 ms total, with 293.262 ms attributed to operations and 32.107 ms outside that sum. Its own metadata states DEBUG_CUDA_TIMING disables replay and adds synchronization, so those approximately 46 ms per graph cannot represent the current approximately 29 ms service decode. A read-only discovery task is checking local NVIDIA tracing tools and a graph-preserving service capture route before recommending another kernel change from that profile. No profiler has been launched or installed.


### 027 build-output retention limitation

The clean OFF build emits enough compiler-template warnings to exceed the monitor read limit (truncated=true at roughly 1 MiB). This corrects the earlier promise of a complete monitor-output artifact: preserve the retained output and terminal status, but do not label it the complete compiler log. Compilation is still progressing. Future builds will use an exclusive full log with the command exit code preserved, as earlier experiments did, instead of relying on this capped capture. No successful build will be repeated merely to regenerate warning text.


### 027 clean OFF build passed; ON build running

The clean OFF build completed all 485 steps and exited 0 (monitor 21155, approximately 594 seconds observed). Its native capture retained 1 MiB of 19,084,338 emitted bytes; the explicitly marked partial artifact is `cuda-build-bonsai-only-off.retained.txt`. The compiler warning text was not regenerated. All 11 OFF snapshot files were copied exclusively to tools/llamacpp-cuda-bonsai-only-off and hash-verified (`cuda-bonsai-only-off-binary-hashes.json`); the OFF CMake cache was checked.

Clean ON compilation is now running through the canonical -BonsaiOnly switch in a separate directory, with complete stdout/stderr reserved at `cuda-build-bonsai-only-on.txt` and its real exit code propagated by the wrapper. Offline OFF symbol/resource inspection is independent of this build. No throughput measurement or GPU correctness workload is running during compilation.


### 027 generated build topology verified

Generated CMake caches and Ninja compile rules confirm OFF has 24 MMQ translation units and no Bonsai-only compiler define; ON has exactly pq2_0 and q4_0 translation units with the define enabled. Evidence: `cuda-bonsai-only-build-topology.json`. ON compilation remains in progress; no runtime conclusion follows from this source/build graph check.

### External graph-preserving profiler preparation

Local discovery found no Nsight/CUPTI profiler in checked locations. NVIDIA's general Windows CLI documentation requires administrator execution, while both this shell and LlamaSupervisor use Limited tokens; CUDA-only behavior with CPU sampling/context-switch/counter collection disabled remains untested. The official standalone Nsight Systems 2026.5.1.161 MSI is 663,678,976 bytes. Workspace-only download, NVIDIA signature verification and pure archive extraction are being prepared; no system installation, task privilege change, UAC launch or profiler capture is authorized within that preparation. Actual capture must explicitly enable LLAMA_LAZY_COMPUTE_RESERVE=1 to match the retained service, since bonsai-server-run.ps1 itself does not set it.


### 027 OFF kernel inventory preserved

Offline inspection of the 129,346,048-byte OFF CUDA DLL found 145 SM86 modules, 6,990 resource records and 6,514 unique function symbols. PQ2 is enum 142. Targeted dumps cover all 19 PQ2 MMVQ and 64 PQ2 MMQ functions. Retained fused warp true uses 37 registers / 3,968 text bytes; warp false uses 42 / 5,120; PQ2 J128 MMQ uses 254 registers with 88,320 text bytes (fallback false) or 96,768 (fallback true). Those kernels report zero local/stack storage. The MMVQ cubin contains 3,828,736 total text bytes and the PQ2 MMQ cubin 1,221,760; these are compiled sections, not resident VRAM.

Artifacts have the cuda-bonsai-only-off- prefix: resource.txt, symbols.txt, elf-list.txt, functions.json, pq2-mmvq-sass.txt, pq2-mmq-sass.txt, cubin-sections.json, pq2-comparison-baseline.json, inventory-provenance.json, targeted-provenance.json and extracted cubins. Full native outputs and command exit codes are preserved in exclusive files. ON comparison will check absent functions/modules and all 83 retained kernels' resources and normalized instructions, preserving predicates/mnemonics/operands while excluding address annotations and encoding comments.


### 027 clean ON build and binary snapshot passed

The clean ON build completed all 463 steps and exited 0 (session 53286); full compiler output is `cuda-build-bonsai-only-on.txt`. All 11 ON snapshot files were copied exclusively to tools/llamacpp-cuda-bonsai-only-on and hash-verified (`cuda-bonsai-only-on-binary-hashes.json`). CUDA DLL size changed from 129,346,048 to 83,599,360 bytes, a compiled-file reduction only, not evidence of resident-memory or throughput improvement.

The authorized maintenance window was opened after inspecting the ordinary wrapper/launcher/server command lines from session 0 and checking all four slots idle. Only inspected PIDs 10348, 23284 and 5320 were stopped, with receipt `cuda-maintenance/stopped027.json`; the tunnel remains untouched. The original supervisor is still preserved for byte-for-byte restoration. CPU-reference correctness for both fresh builds is running with exact expected nonzero counts and exclusive stdout/stderr/result artifacts, while the independent offline ON kernel comparison proceeds. No throughput benchmark overlaps those checks.


### 027 static code gate passed; runtime-launch failure recorded

Independent OFF/ON comparison passed for all 83 retained PQ2 kernels: identical normalized instructions/operands, raw ELF machine-code bytes, code sizes, registers, local/stack memory and other resources. No mismatches. SM86 modules decrease 145 to 123; resource records 6,990 to 5,236; ELF text 91,159,936 to 54,962,944 bytes; ELF global sections 3,900,875 to 3,306,919 bytes. Removed symbols comprise 1,408 MMQ kernels across 22 types and 346 MMVQ kernels across 23 types. Both families retain only PQ2_0 and Q4_0. Full evidence is `cuda-bonsai-only-comparison.json` and the exclusive OFF/ON inventories, SASS, section reports, cubins and provenance. These are compiled-artifact results, not physical-memory or speed gains.

The first runtime orchestration attempt (monitor 11637) failed with Python SyntaxError at an inline regular-expression argument before any test or result artifact started. The same finite matrix was saved to `cuda-bonsai-only-runtime-check.py`, parsed successfully with py_compile and relaunched as a file (monitor 89988). The failure did not relax any expected counts or correctness criteria.

### Standalone profiler is prepared, capture still untested

Nsight Systems 2026.5.1.161 was downloaded and NVIDIA Authenticode-verified, then extracted without running installer actions. All 6,654 reconstructed files were hash-checked. MSI SHA256 is 379c0a15a9cf7b8028081073fdd1d9798b9aaa618fb46c6a01785b69ed01d5f2; CLI SHA256 is e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291. Version and launch/start/stop help exited 0. Installed help corrects web-derived examples: --sample and --cpuctxsw belong on start, not launch. No profiler capture, installation, elevation or permission change has occurred. Full preparation evidence and commands are in `nsight-preparation-ready.md`.


### 027 all retained-format backend references passed

The file-based runner completed exit 0 (monitor 89988). Both clean builds passed 101/101 PQ2 matmul, 44/44 PQ2 fused decode, 8/8 whole-graph PQ2 FFN, 3/3 Q4_0 matmul, 9/9 F32/F16/BF16 matmul, and 2/2 Q4_0 attention cases: 167 per build, 334 total. These are CPU-reference correctness checks; their durations are not service throughput results. Full per-command outputs, exact arguments, exits and count assertions remain in cuda-bonsai-only-{off,on}-*.stdout/stderr.txt and `cuda-bonsai-only-runtime-results.jsonl`.

The pruned build is now healthy at the stable service path, PID 9040, with original production arguments and lazy reservation enabled. Strict 512/4096/16384/512 service growth/reference checking is running before four-slot atomic checking and uninstrumented OFF/ON speed comparison. All profiling flags remain off for those checks.


### 027 strict actual-service correctness passed

The pruned build passed all four 512/4096/16384/512 growth cases, repeated-prompt consistency, and all four slots in one atomic concurrent request. Both commands exited 0, matching exact reference requests, generated tokens/content, and top-five probabilities within the existing absolute 1e-4 gate. Evidence: cuda-bonsai-only-on-growth-correctness.jsonl and cuda-bonsai-only-on-atomic-correctness.jsonl. No timing instrumentation was enabled. The inspected idle candidate server PID 9040 was stopped for fresh OFF/ON conditioned speed comparisons; accepted binaries and original supervisor remain preserved. Still two validated throughput wins.


### 027 rejected: smaller binary did not improve service output

The fresh OFF/ON runs used identical production settings, lazy reservation, separate conditioning (four requests), normal warmups and three measured requests at each length. Both benchmark commands exited 0. All eight cross-condition requests/tokens/content match; the comparator exited 1 solely for the preregistered >=2% output gate. Conditioning and warmups are excluded below.

| Prompt | OFF ingest | ON ingest | Change | OFF output | ON output | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 400.344 | 419.306 | +4.736% | 35.069 | 35.149 | +0.230% |
| 4096 | 504.719 | 510.025 | +1.051% | 33.163 | 33.038 | -0.377% |

No target output gain was established. The 512-token ingest signal was not consistent across both lengths, and neither length clears a 2% ingest gate; no post-hoc win is counted. Stop this candidate without more pairs. All six measured before/after Windows process GPU-memory samples in both conditions were identical: dedicated 11,266,936,832 bytes and shared 111,149,056 bytes. These are allocation counters, not proof of physical residency. Matched shape warmup and byte-identical retained kernels leave no observed service memory benefit despite the 35.37% smaller CUDA DLL.

Evidence: cuda-service-bonsai-only-{off,on}-a.jsonl, separate conditioning files, cuda-service-bonsai-only-comparison-a.json and cuda-bonsai-only-service-memory-comparison.json. Warmed OFF samples began at 71-75 C and ended at 82-86 C; ON began at 73-74 C and ended at 84-85 C. No claim of identical clocks throughout a request is made.

Reversed the exact V3 six-file patch and verified all six files equal retained HEAD. Preserved source patches, snapshots, inventories, native test outputs and service artifacts. Stopped inspected idle PID 8684 and redeployed the accepted lazy-reserve snapshot through the canonical helper. The original supervisor remains temporarily guarded for the immediately following profiler maintenance; it has not yet been restored. Still two validated speed wins.

### 028 graph-preserving trace: minimal privilege probe

Purpose: attribute current service prefill and decode while CUDA graph replay remains enabled. The older DEBUG_CUDA_TIMING profile disables replay and is not suitable for this attribution. First test workspace-extracted Nsight Systems with CUDA tracing only, host-launched graph node activities, CPU sampling/context switches/GPU counters disabled and lazy reservation enabled. A trace is diagnostic evidence, not a throughput win; profile overhead must be checked against an untraced service request before using durations to predict a change.

The accepted service is being launched through Nsight session bonsai028decode with DEBUG_CUDA_* flags cleared and no elevation or system installation. Full launch output and real exit status are reserved at cuda-nsys028-launch.*. Runtime privilege support and useful device events are still unverified.

027 wording correction: the 512-token ingest delta (+4.736%) exceeds 2%; the 4096-token delta (+1.051%) does not. Thus an ingest gate requiring both lengths would fail, while the actual preregistered output gate fails at both lengths. The previous sentence saying neither length clears a 2% ingest gate was incorrect; the table and rejection decision are unchanged.


### 028 non-admin CUDA-only capture succeeded; ordinary service restored

Nsight start and stop both exited 0 with CPU sampling, CPU context switches and GPU metrics disabled. Two actual-service 512-prompt/32-output requests completed and produced cuda-service-nsys028-decode.nsys-rep (4,881,889 bytes). This establishes non-admin CUDA-only capture for this machine; it does not establish access to CPU sampling or GPU hardware counters. Graph node device events and replay correlation are undergoing independent verification before any bottleneck claim.

After capture conversion completed, the inspected idle profiled server PID 3548 was intentionally stopped. The launch command, which remained attached to the application lifetime, consequently returned -1; its stdout says Collecting data and stderr is empty. This is preserved in cuda-nsys028-launch.* and is distinct from successful capture start/stop/benchmark exits. No Nsight processes remain in the subsequent CIM inspection.

The original supervisor was restored byte-for-byte to SHA256 54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be, with the maintenance version preserved. All 11 deployed files match the accepted lazy-reserve snapshot. The ordinary scheduled service is healthy, PID 1944. An identical untraced 512/32 service request pair is running to bound trace perturbation; profile timings are not counted as optimization results.

028 control-launch correction: passing --pid 1944 failed immediately in OpenProcess with WinError 5 because the ordinary scheduled service runs in another Windows session. No HTTP request or result file was created. The canonical optional-PID path is now being used for the same HTTP benchmark, omitting affinity/process-memory inspection; no privileges, affinity or service settings were changed. The previous entry described the intended running control before this launch result was inspected.


### 028 trace perturbation smoke comparison

The identical untraced 512-prompt/32-output HTTP benchmark completed exit 0. Both warmup and measured requests have exact cross-condition request/token/content equality. The measured untraced result is 513.733 ingest / 34.977 output tok/s versus traced 410.537 / 34.130: -20.087% ingest, -2.421% output, +13.890% wall time. This single sample uses distinct restarted processes and different thermal/Windows-session conditions, so it is not a precise causal overhead estimate. The output trace is much closer to current service cadence than the earlier replay-disabling instrumentation; prefill timing is visibly perturbed and must not be treated as native throughput. Comparator artifact: cuda-service-nsys028-overhead-comparison.json; no performance acceptance gate was requested.

Independent export inspection found 132,598 positive-duration kernels, 118,980 graph-node kernels and 60 cudaGraphLaunch calls. Node mode intentionally has no whole-graph activity rows. DIAGNOSTIC_EVENT warns that not all CUDA events might have been collected; per-replay coverage and correlation are still being checked. No claim that summed event times account for every service delay is made yet.


### 028 replay attribution and coverage audit completed

All 60 graph launches contain the same 1,983 node identities, names and geometry. All 118,980 graph-node records correlate to those launches. The 13,618 directly executed kernels have matching unique launch APIs; 3,966 other launch APIs occur inside the two 1,983-node capture intervals. All 848 copies and 192 memsets match recorded submissions one-to-one, with no CUDA API failures. The generic Nsight warning remains unexplained; the audit establishes no identified gaps among recorded execution submissions, not universal collection completeness.

Measured steady decode (last 30 replays): summed kernels 26.983 ms/token, device span 27.349 ms, average inter-replay gap 1.373 ms. Ordinary PQ2 warp kernels account for 15.698 ms/token and fused PQ2 gate/up 5.773 ms/token: combined 21.471 ms, 79.57% of kernel time. GDN is 0.935 ms and get_rows_float_vec is 0.899 ms. cudaGraphLaunch host duration overlaps device work and must not be added to GPU duration. No hardware bandwidth counters were collected, so a bandwidth bottleneck is unproven.

The traced measured 508-token prefill has PQ2 MMQ 737.303 ms (64.53%) and GDN 214.330 ms (18.76%); the previously recorded prefill perturbation limits applying these proportions to untraced service timing. First-three GDN outliers remain observed without a paging explanation. Complete evidence: cuda-nsys028-analysis-report.md, results.json, coverage.json, filtered-count-check.json, original SQLite, CSVs and native command outputs. A first filtered report command failed with Time format overflow for a bare large nanosecond component; split seconds/nanoseconds succeeded and both artifacts were preserved.

Next read-only candidate work is narrowly scoped to (a) the active ordinary PQ2 warp register tier and shape specialization, and (b) the source/geometry behind the get_rows_float_vec cost. No new kernel change or speed win is claimed from profiling alone.


### 028 recurrent-state gather follow-up scoped, not implemented

Independent source/trace attribution identified exactly 48 vector gathers per token, each copying 786,432 F32 values (3 MiB) from recurrent state. Measured cost is 0.898677 ms/token. Origins are state_predelta in qwen35.cpp and get_state_rows in llama-graph.cpp; GDN subsequently reads that materialized state. The unrelated scalar gathers for convolution history/output selection remain outside this finding.

A direct indexed GDN input could remove up to 288 MiB/token of gather read/write traffic, but existing rows mode requires ring snapshots and changes writeback to SET_ROWS, losing the current CUDA GDN-to-CPY fusion. Any future candidate must keep dynamic device row indices across replay, preserve write fusion, and limit to single-token/single-sequence scalar S128/H48/K1 with no extra-state relocation; graph-builder comments document an overwrite hazard otherwise. Complete elimination would save 3.07% of the traced 29.300 ms latency, an ideal 3.16% throughput ceiling. Actual gain would be smaller. No implementation is scheduled now; prioritize the simpler active ordinary-PQ2 register hypothesis before this state-sensitive multi-file change.


## 029 - disable automatic ordinary PQ2 decode unrolling

The actual 028 DLL SHA matches the retained SASS metadata, and ordinary/fused instruction text matches clean 027 OFF output. The active ordinary kernel uses 42 registers and zero local memory, with a compiler-generated peeled first iteration then paired chunk loop: increment 64, 28 global loads before the first DP4A, two sequential accumulation FFMAs. The fused kernel uses 37 registers and one chunk per iteration. This is not the previously rejected explicit unroll proposal; the hypothesis is that compiler-created second-chunk live state crosses a register allocation tier.

Candidate: split the compile-time ordinary/fused loop branches, use literal pragma unroll 1 only on the ordinary loop, and retain fused semantics, reduction/epilogue, dot helper, per-lane arithmetic order, and 128-thread launch geometry. No K specialization or format change. Static gates before GPU correctness: ordinary <=40 registers, zero stack/local/spill instructions, one chunk per iteration with increment32 and unchanged tail/accumulation order, smaller approximately14-load group; fused normalized instructions and37-register resources unchanged. Reject if those gates fail.

SM86 occupancy improvement from40 to48 resident warps is inferred from register allocation, not measured. Conditional service prediction: a4-8% reduction in the15.698ms ordinary kernel share would save0.628-1.256ms/token, about2.2-4.5% output throughput against the traced29.3ms cadence. A2% output gain needs about3.7% improvement in this kernel family. Extra loop instructions or reduced instruction/memory parallelism may cancel the benefit. Acceptance remains repeated uninstrumented actual-service pairs with >=2% output gain at both512/4096 and <=2% ingest regression, exact outputs, plus retained CPU-reference and service growth/atomic correctness.

Source preparation is isolated in experiment/pq2-no-auto-unroll. Normal accepted service remains healthy; no compilation or service test for029 has started.

029 source candidate prepared: one mmvq.cu change, +13/-5 lines, patch cuda-pq2-rolled-source.patch SHA25665071704a5548554161ab6b04d408ebbcf4953eb854602806479add40fe2da20. Author source-expansion/diff checks passed; independent source review is pending before integration/build. The existing clean full-format OFF build cache uses Release/86-real and root source; no compiler processes or candidate log/snapshot artifacts already exist. Its preserved OFF snapshot will remain immutable if that build directory is reused.


### 029 reviewed source compiled successfully

Independent spec/source-quality review passed, then the exact patch was applied to root and whitespace-checked. Canonical CUDA12.9.1/SM86 Release compilation in the reusable full-format cache completed all210 build steps, exit0 in58.5s. Full202,457-byte compiler log and command/exit/timing receipt are cuda-build-pq2-rolled.txt and cuda-build-pq2-rolled.exit.json. The prior clean OFF snapshot remains separate and unchanged; the build directory now contains029 and must not be treated as the old baseline.

All11 candidate files were copied exclusively and hash-verified in tools/llamacpp-cuda-pq2-rolled. CUDA DLL SHA256f85e33eef55dfb6ae5b4f126a3356dcf7318ff0a902e4ce0aa4fd12f3f91b060; manifest cuda-pq2-rolled-binary-hashes.json. Independent disassembly/resource verification is pending; no029 GPU correctness workload or service comparison has started.


### 029 static gates passed

Independent disassembly of the hash-verified candidate reports ordinary registers42->28, zero stack/local memory and no LDL/STL spill instructions. Ordinary code size5120->2176 bytes; the paired loop is replaced by one chunk/increment32, with14 rather than28 loads before the first DP4A. Static inspection preserves eight ordered DP4As, scale multiplication, one sequential accumulator FFMA, tail/row guards, reduction, bias and32x4 launch geometry. Fused normalized instruction text and full resource record are identical to accepted:37 registers,3968 code bytes.

Evidence: cuda-pq2-rolled-sass-candidate.txt, its metadata.json, cuda-pq2-rolled-sass-comparison.json and cuda-pq2-rolled-static-gates.json. This crosses the preregistered register gate but is not a speed result; reduced load overlap remains a risk. Runtime references and strict actual-service correctness follow before timing.


### 029 CPU-reference runtime gates passed

After inspected command lines and idle-slot checks, maintenance stopped only the ordinary wrapper/launcher/server PIDs25388/22132/1944; receipt cuda-maintenance/stopped029.json. The tunnel/task definition remain unchanged, and original supervisor plus accepted snapshots are preserved for restoration.

The candidate passed101/101 PQ2 matmul,44/44 PQ2 fused decode and8/8 whole-graph PQ2 FFN cases,153 total. All native commands and expected nonzero counts passed; full stdout/stderr and command/exit/count assertions are cuda-pq2-rolled-{mulmat,fusion,ffn}.*.txt and cuda-pq2-rolled-runtime-results.jsonl. No acceptance throughput is inferred from reference-test durations. Candidate service startup uses the stable path, full production context/batch/slots, lazy reservation and no debug instrumentation; strict growth/atomic gates follow.


### 029 strict service correctness passed

Both growth and atomic commands exited0: all512/4096/16384/512 requests, repeated512 after buffer growth, and all four slots in one atomic concurrent request match accepted references under exact requests/tokens/content plus the unchanged1e-4 probability gate. Evidence: cuda-pq2-rolled-growth-correctness.jsonl and cuda-pq2-rolled-atomic-correctness.jsonl. The inspected idle candidate correctness process was stopped; throughput conditions each start fresh so the16K growth allocation is not carried into candidate timing. Warming/measurement use the same canonical service benchmark with profiling flags off.


### 029 rejected: register reduction did not produce a service gain

Matched fresh processes each completed separate conditioning and three measured requests at both lengths, with profiling flags off. All eight corresponding requests/tokens/content match. The comparator exited1 for the preregistered output performance gate:

| Prompt | Retained ingest | Candidate ingest | Change | Retained output | Candidate output | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 400.404 | 401.689 | +0.321% | 35.485 | 35.360 | -0.353% |
| 4096 | 515.580 | 516.721 | +0.221% | 33.397 | 33.280 | -0.353% |

The42->28 register reduction, zero spills and smaller generated code did not meet the predicted service gain. Reject without reverse-order repeats; do not infer why throughput stayed flat from occupancy alone. Control measured samples began71-75C/ended82-86C, candidate73-74C/84-85C; instantaneous clock equality is unproven. Artifacts: cuda-service-pq2-rolled-{control,candidate}-a.jsonl, separate conditioning files and cuda-service-pq2-rolled-comparison-a.json.

Reversed the exact source patch and verified mmvq.cu equals retained HEAD. Preserved patch, candidate snapshot, SASS/resources and all runtime/service evidence. After stopping inspected idle PID14152, the canonical deploy helper restored accepted lazy-reserve binaries. All11 hashes match; original supervisor restored byte-for-byte to54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be; normal scheduled service healthy PID24124. Still two validated optimization wins.

One combined health-check/log-update tool command was blocked by policy with no stated reason before execution; separate health and append operations succeeded without changing permissions. This affected orchestration only, not any correctness/performance gate.

## 030 - direct indexed recurrent-state input during single-sequence decode

After029 failed, proceed with the previously scoped gather candidate. Mechanism: remove48 materialized3MiB F32 state gathers/token and load the selected cached row directly in GDN, retaining existing cache-write fusion and exact arithmetic. Guard to scalar S128/H48, one token/sequence/K1, no extra-state relocation and supported layouts; retain normal path otherwise. Row indices stay dynamic device inputs across replay. Existing ring-mode activation/writeback must not be enabled blindly.

Measured gather cost0.898677ms/token implies an ideal3.16% throughput ceiling at29.3ms/token. Conditional prediction is roughly2-3% output gain if most of this copy cost is removed with negligible indexing overhead; no prefill gain is expected. Static/source gates require correct input lifetimes, relocation exclusions and preserved write fusion. Runtime gates include nonzero/changing state row IDs, untouched-slot checks, all existing relevant GDN references, actual-service growth/atomic equivalence, and trace confirmation that the48 target gathers disappear. Acceptance requires repeated uninstrumented service pairs with>=2% output gain at both512/4096 and<=2% ingest regression, exact requests/outputs.

Implementation is isolated in experiment/gdn-indexed-state. A separate workspace-only Nsight Compute preparation task will determine a safe non-admin targeted-counter probe for later PQ2 diagnosis; no counter permission change, installation/elevation or additional profiler capture has occurred.


## 031 - hardware-counter permission probe failed

Prepared portable Nsight Compute package2026.3.0.13 (CLI2026.3.0.0 build38525999), official archive486,746,334 bytes with NVIDIA-manifest SHA2569467f08354c93868aed67c32acad6f5811af46b769f4c5117b3e9210a76f9837. Executable SHA25681b4977676f813aa29878d6d94148c8010476554e42dfb907e3e6e1c5fe3c147 and valid NVIDIA Authenticode; version/help/offline GA106 metric queries passed. Preparation evidence: cuda-ncu-preparation-ready.md/json and native outputs. Section listing unexpectedly materialized stock files under Documents/NVIDIA Nsight Compute/2026.3.0; they are preserved. The probe explicitly selected packaged sections. No installer action or system/driver setting was changed.

After inspecting ordinary wrapper/launcher/server PIDs5944/12504/24124 and all idle slots, a short maintenance interval launched accepted service binaries under NCU, selecting one ordinary PQ2 launch and one real counter, dram__bytes_read.sum. Graph node/kernel replay mode was used with clock-control none/cache-control none and debug timing off. Launch-skip0 was a capability probe that could select startup warmup, not a steady-request performance sample.

NCU connected to service PID5776 and explicitly returned ERR_NVGPUCTRPERM: this user cannot access NVIDIA GPU Performance Counters on device0. No numeric counter or ncu-rep report was produced. The server itself loaded and became healthy. No HTTP benchmark or second/fused counter attempt was made after this decisive permission failure. Hardware counters remain unavailable under current permissions; no privilege/counter-policy change is scheduled. The existing Systems trace and030 state-copy experiment remain usable independently.

After that error, the inspected idle diagnostic server was intentionally stopped. NCU parent then reported application exit4294967295 (tool status-1); the command/real exit/timing and full stdout/stderr are preserved in cuda-ncu031-launch.*. This terminal application error is separate from the earlier counter-permission error. No NCU processes remain. Original supervisor restored byte-for-byte, all11 accepted binary hashes match, and ordinary scheduled service is healthy PID7636. Still two validated optimization wins.


### 030 source/test candidate prepared for review

Seven-file candidate (+199/-27) preserved at cuda-gdn-indexed-state-source.patch, SHA256a8e53ec88f24fff31ab2acec95afb3454c5b417e7ba48401a78a548021ef02ae. It adds a narrow dynamic cached-row input to CUDA GDN after PDL synchronization while retaining K1 CPY graph output; ring behavior and the GGML_GDN_STATE_GATHER fallback remain. Author also guards partial/non-row-aligned cache overlap from write fusion. Patch application check against retained root passed; root code is still unchanged pending independent review.

Prepared focused selector GATED_DELTA_NET_INDEXED_CACHE expects10 whole-graph cases covering nonzero/different rows, in-place/cross-slot writes, changing device index inputs and untouched-slot canaries/partial-overlap fallback. Existing GDN selector gains two supported indexed and four unsupported-shape cases. These are source preparations, not executed tests. Review specifically checks index lifetime, alias/order hazards, guard coverage and whether the tests genuinely preserve intended fusion. Real CUDA graph replay across service slots remains a required runtime gate.


### 030 independent review: production pass, test-construction fix required

Review found a concrete source-level test bug: indexed-cache build_graph unconditionally expanded member gf, which is null when eval_perf constructs cases before allocating its own graph. This could crash performance enumeration even for unrelated selectors. No crash was executed; the construction call chain establishes the defect. Author is restricting ordered multi-node graph expansion to correctness mode with a valid member graph, following retained PQ2_STAGED_FFN practice. A native perf/NO_MATCH construction smoke is required after build.

Production/alias review passed within the served scope: all changed builder callers and four CUDA launches checked; explicit cache/index dependencies survive, row reads follow PDL synchronization, relocation guard is present, whole-row alias stores are column-disjoint, and shifted overlap disables write fusion. Ten prepared cases structurally cover eight eligible write-fusion and two partial-overlap fallback graphs. Compiled behavior and positive fusion evidence remain pending.

Review also identified a coverage gap: two indexed GDN nodes in one compute do not prove cache replay across repeated computes. Acceptance now explicitly requires the same graph/tensor pointers across at least three computes, changing device row-index contents, checking output/state/all-slot canaries against CPU at each step, and positive evidence that CUDA graph replay occurred. Existing service growth/atomic checks are retained but do not substitute for this gate.


### 030 V2 source review/build passed; replay harness being completed

Independent review confirmed the early mode/member-graph guard closes the construction defect; production sections are byte-identical to V1. V2 patch SHA256308f6bfcaf6d9ff0eb53799b3042d50173680290119eda9beb802db39b74eec1 was applied after its clean check. Canonical CUDA12.9.1/SM86 Release build completed exit0; full compiler log/command/exit/timing are cuda-build-gdn-indexed-state-v2.txt and .exit.json. The deployed ordinary service remains on accepted binaries.

The replay test is being added through a narrow virtual comparison hook whose default preserves the existing helper. Only indexed-cache tests override it, using one persistent CPU graph copy and four computes with unchanged graph/node/data/buffer addresses, changing existing row tensors before each compute and comparing all ten attention/cache snapshots after each. One exact case will run in a fresh process for unambiguous direct/capture/replay/replay logging; allocator/key reuse across the ten-case suite could otherwise inherit history. No new runtime test has run yet.


### 030 V3 repeated-graph test review passed

Independent spec/quality review passed the V3 test delta; production patch sections are byte-identical to V2. V3 SHA256f715994a1e20de2c1f3021c06264c091a1b872eba862d1bb062691a354e144f8. Applied only the reviewed test delta, preserving its own patch for rollback. The test uses one persistent CPU graph copy, four computes with device indices1/3 alternating, accumulated cache contents, unchanged graph/node/data/buffer addresses and all ten snapshots compared after every compute. Expected runtime counts are10/10 indexed-cache cases and49 executed GDN cases with14 unsupported exclusions. The exact single-case graph-stats run must prove direct/capture/replay/replay. These remain pending runtime gates.

Trace proof is also preregistered: actual-service decode should replace48 ordinary GDNs with48 indexed GDNs, remove all48 target3MiB float gathers and reduce1983 to1935 kernels/replay if other work is unchanged. Contiguous write fusion is checked through D2D copy size/order/correlation, not kernel names: eight eligible focused cases retain eight3MiB snapshot copies per compute; two shifted-overlap controls retain ten including the two unfused state writes. Expected totals across four computes per case are80 indexed GDNs and3363MiB D2D copies. Reference and trace gates precede uninstrumented throughput acceptance.


### 030 V3 build and maintenance preparation

The test-only canonical rebuild passed both steps, exit0 in7.8s; cuda-build-gdn-indexed-state-v3.txt and .exit.json preserve full output and command. All11 candidate service files were copied exclusively into tools/llamacpp-cuda-gdn-indexed-state and hash-verified; manifest cuda-gdn-indexed-state-binary-hashes.json, CUDA SHA256a81640f1579b2d8a862747958c021f3505bc7e12b4bebd31466283a3aeb9ccc4.

Inspected ordinary wrapper/launcher/server1696/17132/7636 and all four idle slots, then stopped only those processes for the authorized030 runtime interval; inspection030.json and stopped030.json preserve evidence. The maintenance supervisor prefix holds restarts, with byte-exact original and accepted snapshot retained for restoration. No candidate runtime result is claimed yet.

Coverage clarification from runner preparation: perf builds make_test_cases_perf(), while the focused indexed-cache class is registered only in make_test_cases_eval(). Therefore perf/NO_MATCH is general construction smoke, not direct dynamic coverage of that class guard. The earlier review demonstrated the null-member hazard if constructed in perf mode, but did not establish that current perf registration reaches it. The guard remains defensive; focused correctness execution and fresh-process replay are the decisive runtime tests.


### 030 native reference and graph-replay gates passed

The exclusive runtime runner completed all four commands successfully: general perf/NO_MATCH construction smoke; 49/49 GDN references with 14 unsupported exclusions; 10/10 indexed-cache cases; and the exact raw-gate/in-place case in a fresh process. The focused class performs four computes per backend and ten output comparisons after each compute. Fresh-process graph statistics show one stable key/UID through direct(properties_changed), capture(warmup_complete), replay(stable), replay(stable), while device row inputs alternate1/3. DEBUG_CUDA_TIMING and GGML_CUDA_DISABLE_GRAPHS were absent. Full native outputs, commands, hashes, counts and graph events are cuda-gdn-indexed-state-runtime*. No test-duration speed claim is made.

### 030 focused trace confirms preserved write fusion

Nsight Systems profile completed exit0 in4.7s; cuda-nsys030-indexed-tests.nsys-rep plus full command/exit/outputs are preserved. Offline analysis passed the unchanged preregistered counts: 80 indexed GDN executions, 40 computes, eight eligible cases with eight state-sized D2D snapshot copies per compute and two shifted-overlap fallbacks with ten copies. Total336 copies of3MiB, with zero unassigned D2D. The compact spelling3363MiB in the earlier prediction means336 copies of3MiB, not a3363MiB transfer. Captured target stdout was recovered from SQLite and confirms10/10 tests passed.

All30 graph executions have stable nonzero node identities within each case:15 nodes for eligible cases (four kernels/11 copies) and17 for fallback (four/13). Twenty-four CUPTI originalGraph/originalNode INVALID_PARAMETER lookup diagnostics remain unexplained; no missing execution record was identified by these checks. Three cudaGraphExecUpdate status910 events each immediately reinstantiate and launch successfully, matching the handled source fallback. This establishes execution-level fusion evidence within the trace coverage checked, not universal profiler completeness.

The candidate actual-service512/32 capture also completed with launch/start/benchmark/stop evidence under cuda-nsys030-* and cuda-service-nsys030-decode.*; service analysis and strict context-growth/atomic checks are still pending.


### 030 actual-service mechanism and correctness passed

Offline service analysis passed all final30 measured replays:48 indexed GDN executions, zero target F32 state gathers, zero3MiB state writes and exactly1935 kernels/replay. After normalizing GDN template names, the inventory difference from028 is exactly48 removed gathers/token, with no other additions/removals. Evidence: cuda-nsys030-service-analysis.json, cuda-nsys030-service-vs028.json and full export/analysis receipts. Generic collection warning retained.

The separate traced run does not show a speed win: summed kernel time29.160ms/token versus26.983 in028, with ordinary PQ2 increasing18.117 versus15.698ms while GDN stayed0.943 versus0.935ms. These runs do not provide a matched uninstrumented comparison; do not attribute the variation to the candidate or claim a gain from removed kernel counts alone.

Strict actual-service growth512/4096/16384/512 and atomic four-slot checks both passed exact request/token/content and the unchanged probability tolerance. Artifacts: cuda-gdn-indexed-state-growth-correctness.jsonl and cuda-gdn-indexed-state-atomic-correctness.jsonl. Inspected idle diagnostic PID14644 was then stopped, ending the Nsight launch parent with4294967295/tool1 as expected for intentional target termination; capture start/benchmark/stop had already exited0. No profiler process remains.

Restored all11 accepted snapshot hashes and started fresh uninstrumented baseline PID19440 under the same production parameters. Separate conditioning plus three measured requests per length are running. Neither candidate acceptance nor a third win is claimed.


## 032 - direct PQ2 block indexing hypothesis

Independent read-only discovery identified repeated signed chunk division/remainder in retained ordinary and fused PQ2 SASS. Proposed narrow change: constant part=threadIdx.x%4; iterate block=threadIdx.x/4 with stride8 up to ncols/QK_PQ2_0, preserving dot calls and accumulation order. Discovery checked identical (block,part) sequences for every lane across whole-block K128..65536. No weight layout, codec, row grouping, load policy or unroll-policy change is intended.

Static gate before runtime: at least six fewer dynamic address-generation instructions per logical chunk; unchanged load/DP4A counts and arithmetic order; zero spills; registers no higher than retained ordinary42/fused37; unchanged loop expansion and launch geometry. Reject if those requirements fail. Family share21.471ms/token means a2% end-to-end gain needs approximately0.575ms, or2.7% of that family. A nominal4-7% loop-instruction reduction makes that plausible only if instruction/address issue matters; traffic is unchanged, confidence in an actual gain is low. Prior019/029 failures explicitly argue against equating smaller SASS with throughput.

Source preparation will be isolated while030 service testing runs. No032 compilation or GPU workload may overlap throughput measurements. Runtime acceptance remains strict PQ2 CPU references, service correctness and repeated uninstrumented service pairs with>=2% output gain at both512/4096 and<=2% ingest regression. If030 is accepted,032 must be compared against the newly retained030 binary, not the older baseline.


### 032 source prepared and independently reviewed

The isolated patch cuda-pq2-block-index-source.patch has SHA256fae10244959201334c0041626e42d73f822376a37e9987f560c4cf6dc2d96823 and changes only the requested loop (+2/-3 lines). Author and independent reviewer each verified16,384 lane/K combinations, including tails, and unchanged remaining source bytes. Source gate passed. The compiled gate remains unchanged: at least six fewer dynamic address instructions per chunk (12 per paired ordinary iteration), retained loop expansion/load widths/ordered arithmetic, registers<=42/37, zero stack/local/spills and unchanged32x4 geometry. No032 compilation/GPU work occurred during030 service measurements; patch integration waits for030 outcome.


### 030 first uninstrumented service pair passed; reverse pair pending

Fresh baseline19440 and candidate25584 each completed separate conditioning and three measured requests per prompt, with no profiler/debug instrumentation. All corresponding requests/tokens/content matched. The comparator passed both preregistered gates:

| Prompt | Retained ingest | Candidate ingest | Ingest change | Retained output | Candidate output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 428.152 | 436.383 | +1.923% | 35.153 | 36.860 | +4.855% |
| 4096 | 509.557 | 514.644 | +0.998% | 33.071 | 34.637 | +4.735% |

Evidence: cuda-service-gdn-indexed-state-{control,candidate}-a.jsonl, their separate conditioning files and comparison-a.json. This is promising but not yet an accepted third win. The reverse pair uses fresh candidate19024 followed by fresh retained control, with identical conditioning/measurement. The earlier traced gather-share ceiling is approximate and from different conditions; do not attribute the full observed percentage to its kernel-time estimate.


### 030 accepted: third validated optimization win

The reverse pair passed unchanged gates: output+3.751%/+3.108% at512/4096, ingest+2.138%/-0.700%. Pooled ABBA has six measured samples per condition at each length, all exact request/token/content comparisons passing:

| Prompt | Retained ingest | Candidate ingest | Ingest change | Retained output | Candidate output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 414.723 | 425.621 | +2.628% | 35.452 | 36.868 | +3.994% |
| 4096 | 515.155 | 514.956 | -0.039% | 33.393 | 34.617 | +3.666% |

Evidence: both comparison-b.json and comparison-abba.json, four measured artifacts and separate conditioning artifacts under cuda-service-gdn-indexed-state-*. Retain for repeatable output improvement with no material ingest regression; the512 ingest increase is observed, not a demonstrated prefill-kernel optimization. Independent final review passed the exact V3 diff, runtime hashes/counts, independently queried SQLite mechanism evidence, and all768 checked growth/atomic logprob values matching exactly. Its pending ABBA condition is now satisfied.

After inspecting idle baseline18904, stopped it and deployed tools/llamacpp-cuda-gdn-indexed-state through the canonical helper. All11 stable-path hashes match cuda-gdn-indexed-state-binary-hashes.json. Restored the original supervisor byte-for-byte (SHA25654af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be) and restarted the normal scheduled service, healthy PID23780. Maintenance artifacts remain preserved. This candidate is the new rollback/performance baseline for032; old lazy-reserve snapshot remains available. Three validated wins total.


### 032 build passed; static gate pending

Applied the exact reviewed032patch on top of accepted030commit10bd19e6d8d709d4750bfb8da86c1e9046fbf53f. Canonical CUDA12.9.1/SM86 Release build completed exit0 in58.98s; full logs and command/exit/timing are cuda-build-pq2-block-index.txt and .exit.json. All11 files copied exclusively and hash-verified in tools/llamacpp-cuda-pq2-block-index; manifest cuda-pq2-block-index-binary-hashes.json, CUDA SHA25672d45521bb02021bce2967660e96e47c98d6dcfef4a095f02f200780f8b353d0. Independent assembly/resource checks against the exact newly retained030DLL are pending before any GPU testing. The normal service remains on all11 verified030hashes.

After030landed, its author worktree was removed only after verifying all seven modified files matched committed10bd19e6 byte-for-byte, no untracked/ignored files existed, and the preservedV3patch hash matched. Its branch/logs/patch and the separate032worktree remain.


### 032 rejected at the preregistered static gate

Independent assembly review found ordinary registers42->46 and fused37->43, both above the unchanged ceilings. Address-generation savings were12 instructions/chunk in ordinary but only3 in fused (required>=6). Ordinary paired loop instructions172->145 and code5120->4736 bytes; fused single-loop138->136 and code3968->4224 bytes. Zero stack/local/spills, preserved loop expansion/load counts/widths/ordered dot/FP32 arithmetic/tails/geometry passed inspection.

The compiler retained weight pointers across iterations; fused maintenance of four scale/payload pointers offset most address savings and increased live state. Fused43 registers cross the40-register allocation tier. This is a static resource failure, not a measured throughput regression. Reject without any032GPU/reference/service run, preserving the predeclared gate. Full baseline/candidate SASS with command/hash metadata, comparison and instruction-level gate evidence are cuda-pq2-block-index-sass-*.txt/json and cuda-pq2-block-index-static-gates.json.

Reversed the exact032source patch and verified mmvq.cu equals retainedHEAD. Preserved patch, snapshot, build logs and disassembly evidence. Normal service remains on accepted030; still three validated wins.


## 033 - unsigned PQ2 chunk arithmetic, compile-only hypothesis first

Keep the original chunk loop and index structure, but use uint32_t for chunks/chunk/block/part, unsigned division/remainder constants and stride32u. Leave helper signatures, dot calls, accumulation, unroll policy and launch geometry unchanged. Positive whole-block K guarantees identical chunk order and addresses; active helper inputs must remain within their existing signed-int domain.

Independent baselineSASS review identifies potentially6-7 net address-instruction savings/chunk in both kernels: signed quotient/remainder correction and sign-extension/carry/high-half reconstruction for the Q8 address, subtracting necessary unsigned replacement operations. Keeping block unsigned is part of the hypothesis so y+block*4 does not immediately reintroduce signed offset handling. Unlike032, this preserves original chunk induction, but the compiler may still strength-reduce into persistent pointers and raise live state. Confidence remains low; no speed prediction beyond the prior roughly2.7% PQ2-family improvement needed for2% service throughput.

Unchanged static gates: at least six net dynamic address instructions saved per logical chunk in BOTH ordinary/fused, registers<=42/37, zero stack/local/spills, same ordinarypaired/fusedsingle expansion, loadcounts/widths, ordered dot/FP32 arithmetic, tails and32x4 geometry. No register cap or auxiliary tuning. Any failure rejects beforeGPU tests. Only a static pass authorizes runtime references, strict service correctness and repeated uninstrumented service comparisons against accepted030, requiring>=2% output at512/4096 and<=2% ingest regression.032 remains rejected.


### 033 source candidate prepared

Patch cuda-pq2-unsigned-index-source.patch SHA256192069d4ad5b3c7737f5f27578bad5d2a950be1021fdcc694f4f8b421cbdeabd changes only four loop-index lines (+4/-3), with all other mmvq.cu bytes unchanged. Author verified16,384 lane/K ordered sequences and6,303,744 pointer/index comparisons including padded strides and boundary rows within the existing signed helper-index domain. Existing dispatch whole-block assertion and only two kernel launches remain. Diffcheck passed; independent source review pending. No033compilation orGPUtest yet.


### 033 source review and build passed; static gate pending

Independent source review passed the four-line patch, including unsigned promotions and unchanged valid signed helper-index domain. Applied after its clean check. Canonical CUDA12.9.1/SM86 Release build completed exit0 in56.50s, preserved in cuda-build-pq2-unsigned-index.txt and .exit.json. Snapshot tools/llamacpp-cuda-pq2-unsigned-index contains11 exclusively copied, hash-verified files; manifest cuda-pq2-unsigned-index-binary-hashes.json, CUDA SHA256d615f13b8bffb0aa9b7bc8ac6e508690c9700cf860320817dc12160358c3f1d0. The exact030baselineSASS is reused unchanged for independent static comparison. No033GPU/runtime test has started. The normal service still matches all11 accepted030hashes.


### 033 rejected at the unchanged static gate

Independent SASS review found ordinary registers42->47 and fused37->40, both exceeding the preregistered ceilings. Net address savings were6.5 instructions/chunk ordinary (13 per pair) and3 fused, the latter below six. Loop instructions172->153 per ordinary pair and138->135 fused; code5120->4608 and3968->4096 bytes. Zero stack/local/spills and preserved loop expansion, load counts/widths/cache policy, ordered DP4A/FP32, tails and32x4 geometry passed inspection.

Unsigned arithmetic removed signed corrections, but fused generated code again maintained four64-bit weight pointers with eight increment/carry instructions each iteration, offsetting the intended savings and increasing live state. Reject beforeGPU/service testing; this is not a measured throughput regression. Preserved candidateSASS, command/hash metadata, comparison and gate evidence are cuda-pq2-unsigned-index-sass-candidate.*, cuda-pq2-unsigned-index-sass-comparison.json and cuda-pq2-unsigned-index-static-gates.json.

Reversed the exact033patch and verified mmvq.cu equals retainedHEAD. All source/build/snapshot/assembly artifacts remain preserved. Three wins remain retained; the normal service stays on030. Next discovery focuses on prefill, first checking prior017/020evidence to avoid repeating an already-falsified GDN change.


## 034 - two-column GDN restricted to full raw-gate prefill

Read-only review of prior017/020 confirmed020was correctly rejected: medians512 ingest/output228.415/24.469 versus fresh retained controlD413.987/32.123, and4096 296.339/26.949 versus498.872/30.505. It passed correctness but failed its>=5% ingest/no>2% decode-regression gate. Retained decode instructions/resources matched; the slowdown cause was unresolved, not proven to be arithmetic or paging.017had shown some ingestion gains, including+5.1% in a fresh16K reversal, but crossed its decode limit by-2.5%. Do not retry020unchanged on the basis of later allocation changes.

Distinct034hypothesis: compile a two-column-per-warp specialization only for runtimeSM86/highest compiledarch860, scalarS128/H48 raw gates, noKDA/history/precomputed/indexedstate, one sequence/K1, and n_tokens exactly508 or512. Block32x4, grid48x1x16 gives768 CTAs, between retained single-column1536 and prior four-column384. Share each token's q/k and gate work across two columns while preserving each column's token order, four-term lane accumulation, reduction, FMA expressions and final/cache addressing. Decode, short-tail and unsupported paths remain the exact retained selection.

Trace028GDN214.330ms in1247.146ms service prefill gives roughly17.2% share and an impossible-elimination ceiling near20.8% throughput. A2% service improvement requires roughly24.5ms or11.4% of observedGDN time; excluding the unexplained58ms early-layer excess requires~15.7% of regularGDN work. Do not predict a gain from removing those unexplained outliers.

Pre-registered compile gate: candidate<=64 registers, zero stack/local/spills, correctcolumncoverage, unchanged accepted decode/PQ2 instruction paths. Runtime gates: CPU references covering eligible508/512, neighboring-width fallbacks and final/cache outputs; retained indexed replay tests; strict service request/token/content/log-probability equality. Acceptance: repeated uninstrumented ABBA versus accepted030 with>=2% ingest at BOTH512/4096 and<=2% output regression, identical capacity/settings. Any failed gate rejects. Source preparation is isolated; normal service stays on030.


### 034 baseline and source/test candidate prepared

Exact retained030DLL resource extraction found rawS128 fallback/prefill53 registers and3456 code bytes, raw indexed54/3712 and activated indexed46/3584, all zero stack/local/spills. All34 SM86GDN variants and full provenance are preserved in cuda-gdn-cols2-prefill-sass-baseline.*. Prior verified PQ2 resources remain42/37.

Two-file candidate (+36/-1) is preserved at cuda-gdn-cols2-prefill-source.patch, SHA2567875d657f15b0597ca673e8d5f6238b8a2ac59cacca61fbf4317a81375659df4. Root application check passed without applying it. Author reports the exact guarded two-column specialization with unchanged original launches and recurrence body. Default symbols now include COLS_OVERRIDE=0, requiring that suffix mapping in compiled invariance checks. New GATED_DELTA_NET_COLS2_PREFILL selector expects9 passes: raw widths507/508/511/512/513, activated512, permutedV508, B2/508 and K2/512. Existing GDN49/14 and indexed10 replay cases remain. Independent review pending; no034build/GPUtest yet.


### 034 source review and build passed

Independent source/spec/quality review passed. Coverage clarification: permuted V remains eligible and exercises stride handling, so the nine new cases contain three COLS2 cases and six fallbacks. They compare full attention and standalone state outputs; actual-service checks must additionally verify cache-write behavior and dedicated-kernel dispatch. Existing GDN and indexed-replay test counts remain unchanged.

Applied the exact reviewed patch after a clean check. Canonical CUDA 12.9.1 / SM86 Release compilation completed with exit 0 in 59.397 seconds. Full compiler output and command/exit/timing are cuda-build-gdn-cols2-prefill.txt and .exit.json. All 11 files were copied exclusively and hash-verified in tools/llamacpp-cuda-gdn-cols2-prefill; manifest cuda-gdn-cols2-prefill-binary-hashes.json. CUDA DLL SHA256: 90db3738e8312e6f204dd7a7a7a7f9f74353bdb1001e40b2cbd94c31a2f8215a. Independent compiled resource and unchanged-path checks are pending; no 034 GPU test has run.


### 034 static and native reference gates passed

Independent compiled inspection passed: new COLS2 uses 56 registers (limit 64), 4480 code bytes, and zero stack/local/spills. All 34 retained GDN variants have identical normalized instructions and resources after mapping the default template suffix; both PQ2 variants also match exactly at 42/37 registers. Column coverage, state offsets, ordered per-column recurrence/reductions, token strides and lane-zero output stores passed inspection. The token loop has 14 loads for two columns versus 26 across two retained column-warps. This confirms the intended sharing, not a throughput gain. Full evidence: cuda-gdn-cols2-prefill-sass-candidate.*, ...-sass-comparison.json and ...-static-gates.json.

After inspection of ordinary wrapper/launcher/server 10160/5620/23780 and four idle slots, authorized maintenance stopped only those processes; inspection034.json and stopped034.json preserve the evidence. The original supervisor and accepted030 snapshot remain available for rollback.

The prepared runner completed all five commands with exit 0: construction smoke, 9/9 new prefill references, 49/49 retained GDN references with 14 explicit unsupported cases, 10/10 indexed-cache cases, and the exact fresh-process replay case. The latter retained direct/capture/replay/replay with a stable graph key/UID. Full native output, hashes, commands, identities and graph events are cuda-gdn-cols2-prefill-runtime*. Candidate service PID14176 is healthy under Nsight for dedicated-kernel dispatch verification; strict service correctness and uninstrumented speed gates remain pending.


### 034 strict service correctness passed; first timing pair started

The service capture completed with start/benchmark/stop exit 0; artifacts cuda-service-nsys034-prefill.* and cuda-nsys034-* preserve the full evidence. Offline mechanism analysis remains pending. Growth checks at 512/4096/16384/512 and the atomic four-slot request both passed exact requests/tokens/content and the unchanged logprob tolerance against accepted030 references. Artifacts: cuda-gdn-cols2-prefill-growth-correctness.jsonl and cuda-gdn-cols2-prefill-atomic-correctness.jsonl.

After inspecting idle diagnostic PID14176, stopped it; the Nsight launch parent then returned 4294967295/tool1 due to intentional target termination, distinct from the successful capture. No profiler process remains. Deployed and hash-verified all 11 accepted030 files, started fresh baseline PID10132, and began separate conditioning plus three measured requests per prompt. The supervisor maintenance prefix still holds ordinary restarts; original supervisor and accepted snapshot remain preserved for restoration after the experiment.


### 034 actual-service dispatch and unchanged decode confirmed

Offline export and mechanism analysis passed. Both 508-token prefills execute exactly 48 new COLS2 kernels at grid48x1x16/block32x4; both four-token tails retain the default column specialization. All final 30 decode replays retain 48 indexed/default GDN kernels, 1935 total kernels, zero target state gathers and zero 3MiB D2D state writes. Relative to accepted030, only the intended 48 prefill GDN substitutions change per full prefill. Copy size/direction inventories match exactly, with no D2D activity.

Descriptive GDN timing from separate traces: warmup034 total162.523ms/median2.430ms versus030 total236.812/3.578; measured034 total146.581/2.430 versus030 total238.005/3.345. These are not acceptance gains. Early-node outliers remain unexplained. Recorded execution submissions reconcile, with3870 capture-only launches accounting for two1935-node captures and no CUDA API failures; the generic collection warning remains preserved. Evidence: cuda-nsys034-service-analysis.json and cuda-service-nsys034-prefill.sqlite plus export receipts.


### 034 first uninstrumented pair passed; reverse pair pending

Independent final review passed the exact two-file diff, runtime hashes/counts, separate SQLite verification and all eight growth/atomic responses. All 768 checked probability values match accepted030 exactly (maximum absolute difference 0.0). Final retention remains conditional on throughput.

Fresh baseline10132 and candidate24992 each completed separate conditioning and three measured requests per prompt, with profiling/debug flags off. Exact requests/tokens/content match. Pair A passed the original ingest/output gates:

| Prompt | Retained ingest | Candidate ingest | Ingest change | Retained output | Candidate output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 412.133 | 447.546 | +8.593% | 36.332 | 36.566 | +0.644% |
| 4096 | 513.013 | 535.905 | +4.462% | 34.371 | 34.435 | +0.188% |

Artifacts: cuda-service-gdn-cols2-prefill-{control,candidate}-a.jsonl, separate conditioning files and comparison-a.json. Candidate11184 starts the reverse pair from a fresh process using the already deployed, reverified stable-path candidate files. No fourth win is accepted until the reverse/pooled checks pass.


### 035 pre-registered compile-only hypothesis: PQ2 I64/J128 MMQ

Read-only follow-up found that experiment 006 tested I128/J64 and 007 tested I256/J128 with 512 threads; neither falsifies I64/J128 with 256 threads. The accepted service trace contains 400 PQ2 MMQ kernels totaling 645.112 ms (63.64% of the measured 508-token prefill kernel sum). Existing I128/J128 uses 254 registers and 57,856 shared bytes per block, preventing two such resident blocks on SM86. Evidence: cuda-next-mmq-discovery-trace.py/.json and the preserved 006/007 patches.

Hypothesis: I64/J128 with four 16-row groups and two warps per group can reduce accumulators from 64 to 32 per thread and shared storage to 38,400 bytes. Map row base to (warp/2)*16 and column offset to (warp%2)*8, retaining the 16x8 MMA primitive, K256, quantization and ascending per-output K/FP32 operation order. Changing only the tile configuration is invalid: the existing eight-warp accumulation/writeback mapping covers 128 rows. Both mappings and matching host/device tile selection must change together.

Scope: a separate worktree, SM86 PQ2 J128 and the observed full-K service shapes only; preserve fallback specializations. The proposed first compile gate is at most 128 registers per thread, exactly 38,400 shared bytes, zero stack/local/spills, complete unique output coverage, and unchanged per-output arithmetic order. Any failure rejects before GPU testing. These constraints permit two-block residency but do not prove actual residency or a speed improvement. Costs include twice as many row tiles/CTAs and approximately twice the logical activation staging; weight staging and arithmetic are approximately unchanged. No hardware-counter claim is possible under the recorded 031 counter-permission failure.

Source preparation and CPU layout proof may proceed during 034 measurements; compilation and GPU tests must wait. After independent source/static approval, require CPU references and strict service correctness before an uninstrumented ABBA against the then-retained build. Acceptance remains at least 2% ingest gain at BOTH 512/4096 and no more than 2% output regression, with exact requests/tokens/content and unchanged capacity/settings. No throughput win is claimed for 035.


### 034 rejected: reverse-pair 512-token ingest failed repeatability gate

The fresh candidate B and retained030 control B completed conditioning plus three measured requests per prompt, with all exact request/token/content checks passing. Reverse-pair results:

| Prompt | Retained ingest | Candidate ingest | Ingest change | Retained output | Candidate output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 420.396 | 416.744 | -0.869% | 36.744 | 36.816 | +0.196% |
| 4096 | 517.103 | 536.366 | +3.725% | 34.673 | 34.078 | -1.716% |

The pooled ABBA comparison (six measured requests per prompt and condition) passed: 512-token ingest 415.289 -> 436.109 tok/s (+5.013%), 4K ingest 516.250 -> 536.135 (+3.852%); output changed +0.321% and -0.145%. This does not override the failed reverse-pair gate: 512-token ingest changed -0.869%, below the pre-registered +2% requirement. The candidate is rejected and does not count as a fourth retained win. The possible average gain remains unresolved under the observed run variation; no claim of zero intrinsic kernel benefit is made.

Artifacts: cuda-service-gdn-cols2-prefill-{control,candidate}-b.jsonl, corresponding conditioning files, comparison-b.json (exit 1), comparison-abba.json (exit 0), alongside preserved pair A. No selective rerun or relaxed threshold was used to turn this failure into acceptance.

After inspecting idle control PID24608, stopped it and verified that all 11 stable-path files already match retained030. Reversed only the exact SHA-verified 034 source patch; both source files now match HEAD. Restored the exact original supervisor (SHA256 54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be) and started LlamaSupervisor. The maintenance prefix, isolated candidate worktree, binary snapshot, patch, build output, native tests, service traces and every benchmark remain preserved. Normal-service health verification follows. The retained win count remains three.

Normal-service restoration verified: HTTP health ok, correct model, four idle slots each with context 188416, all 11 retained030 stable-file hashes exact, and original supervisor hash exact. Receipt: cuda-gdn-cols2-prefill-restored-service.json. New normal server PID23780 has parent20532 (Session0 cmdline unavailable to the interactive account); PID reuse alone is not treated as process identity evidence.


### 035 isolated source candidate and CPU mapping proof prepared

Five-file candidate (+133/-16) is preserved as cuda-pq2-i64-j128-source.patch, SHA256 a41eeff96410d12516b77ba9c0ccd54018b3ae3e3c35b1acd6712f2787eb93ae. Clean root application check passed without applying it. The author chose a dedicated full-K mul_mat_q_pq2_i64_j128 entry point with defaulted helper layout overrides, leaving generic tile configuration and stream-K selection intact. Dispatch targets six (K,M) pairs: (5120,17408), (17408,5120), (5120,10240), (5120,6144), (6144,5120), (5120,12288), at T508/512 with original full-K eligibility and packed singleton layouts. The 1024-row split-K projection remains generic.

Author CPU proof reports 8192 unique outputs/tile, 32 accumulator slots/thread, 4096 unique payload writes plus 512 scale writes, 38,400 shared bytes, tail/global bounds and identical ordered K terms for all twelve eligible shape/width combinations. Default mappings were checked at eleven widths; an unsafe config-only mapping fails. Artifacts: cuda-pq2-i64-j128-enumeration.json, -proof.py/.json and -source-receipt.json. New native selector PQ2_I64_J128 expects 21 cases (12 eligible, 9 fallback). Independent source review and exact retained030 compiled-baseline extraction are running. No 035 compilation or GPU execution has occurred, and the normal service remains on accepted030.


### 035 independent source review and canonical build passed

Independent specification and quality review passed with no source blockers. The reviewer separately verified unique output ownership, matching dot/writeback mappings, ordered K arithmetic, 38,400 dynamic shared bytes, initialized payload/scale accesses, synchronization, padded T508 reads and guarded output stores. Default mappings remain algebraically equivalent across eleven configured J widths; compiled invariance is still a separate gate.

Applied the exact reviewed patch and built through scripts/bonsai-cuda-build.ps1 with CUDA12.9.1, SM86 Release and the unchanged build settings. Build exit 0 after 244.813 seconds; full output and command/timing receipt are cuda-build-pq2-i64-j128.txt and .exit.json. All eleven snapshot files were exclusively copied and hash-verified in tools/llamacpp-cuda-pq2-i64-j128. Manifest: cuda-pq2-i64-j128-binary-hashes.json; CUDA DLL SHA256 021012a901fad63b54cad3f4cf413f5125ca374696ab3321c845603aeb047e80.

Independent compiled checking now compares all 64 retained generic PQ2 MMQ kernels and both ordinary/fused PQ2 decode kernels against exact accepted030. The new kernel must still meet <=128 registers and zero stack/local/spills. Dynamic shared memory is established from host-launch/source layout evidence, separately from cuobjdump's static SHARED field. No 035 GPU/service test has run; accepted030 remains served normally.


### 035 rejected at the compiled resource gate; no GPU or service test

The dedicated I64/J128 kernel uses 160 registers per thread, exceeding the pre-registered 128 limit for the two-resident-block hypothesis. Stack, local storage and static shared storage are zero; no local load/store spill instructions appear. The reviewed host launch supplies 38,400 dynamic shared bytes and 256 threads. Passing the shared-memory budget alone does not establish the proposed residency.

The separate retained-path gate also failed: 24 of 64 generic PQ2 MMQ bodies differ under the checker's strict normalization. Some differences are relocated call operands, but others are substantive: J128 instruction slots 5520 -> 5504, J112 registers 252 -> 254, and J96 registers 224 -> 222. Both ordinary/fused PQ2 decode kernels remain instruction/resource-identical at 42/37 registers. No claim of fallback throughput regression is made without measurement; compiled invariance itself was the pre-registered requirement.

Full baseline/candidate SASS, parsed inventories and provenance are cuda-pq2-i64-j128-static-*. The checker (SHA256 1ee60dbbc1ccd6e6a89c51c3895dfb6f705a20a71e0ea9fcdfe6039d2c2b6ff9) passed 15 synthetic checks; all extraction commands exited 0 and the gate correctly exited 1. Verdict: cuda-pq2-i64-j128-static-gates.json. No register cap was forced and no GPU/native/service benchmark was run after this failure.

Reversed the exact preserved patch and verified all root source files match HEAD. Accepted030 remains healthy at the unchanged stable service path, with all eleven binary hashes reverified. The failed candidate source worktree, patch, snapshot and all proof/build/disassembly artifacts remain preserved. Retained win count remains three. Further work must address actual register lifetime and avoid changing generic helper instantiations before this mechanism can be reconsidered.


### 036 pre-registered hypothesis: shorten future Q8 staging lifetime

Read-only disassembly of the exact failed035 kernel reports 160 allocated registers but 132 peak live GPRs at IMMA offset 0x2470. Eighteen future second-half Q8 values are loaded by LDG.E.CONSTANT at 0x13f0-0x1500 and first used by shared stores at 0x2db0-0x2ec0, after barrier 0x2da0. Each survives 412 instruction slots without a use/redefinition. Source correspondence is the fully unrolled second staging loop at failed035 mmq.cu:40-44. Evidence: cuda-pq2-i64-j128-liveness-findings.json, native nvdisasm output, extraction receipts and the exact cubin.

Hypothesis: use pragma unroll 1 only on that eighteen-word second-half staging copy, keeping each load close to its shared store after the existing barrier. Preserve the first copy, MMA loops, K traversal, arithmetic and geometry. Removing those eighteen observed lifetimes would hypothetically lower that peak to 114 live GPRs; this is not a prediction of allocated registers. Costs include lost load/MMA overlap and eighteen loop/address/control iterations per K256 step (360 for K5120 or 1224 for K17408).

A separate candidate worktree will keep the retained generic helper headers byte-identical and put fixed-I64 helpers/kernel under private names in a dedicated CUDA translation unit, with only a narrow guarded host dispatch bridge. Preserve the six full-K shape pairs, T508/512 and original 035 guards. Reuse the focused 21-case native reference selector. This addresses the actual 035 lifetime evidence and default-template code changes; it is not a tile/unroll-factor sweep.

Compile gates remain <=128 registers per thread, zero stack/local/spills, 38,400 dynamic shared bytes, unchanged per-output ordered arithmetic, and retained generic PQ2 MMQ plus ordinary/fused decode compiled invariance. Reject before GPU testing on failure. After static approval require native CPU references, actual-service dispatch verification and strict service correctness. Acceptance remains repeated uninstrumented ABBA with >=2% ingest at both 512/4096 and <=2% output regression at unchanged capacity/settings. No register cap or relaxed gate is authorized by this hypothesis.


### Compiler-profile discovery: no justified PGO/device-LTO experiment yet

Read-only inspection of the actual CUDA12.9.1/MSVC14.44 build found SM86 native code, use_fast_math and host /O2 /Ob2, with GGML_LTO=OFF and no rdc/dlto/device-link, /GL or PGO flags in generated Ninja rules. CUDA supports device LTO through dlto/lto_86 intermediate code and a device-link stage. However, the current whole-program compilation already sees the header-defined helpers for these kernels: retained ordinary/fused PQ2 decode and examined S128 GDN kernels have no CALL instructions; J128 MMQ calls target code within its emitted function. No missing cross-translation-unit optimization boundary was identified.

MSVC host PGO is available through /GL, /LTCG /GENPROFILE, representative training and /USEPROFILE. It does not optimize embedded CUDA SASS. A measured CPU critical path and instrumented training data are missing; the 1.373 ms observed inter-replay gap does not establish CPU overhead, and overlapping cudaGraphLaunch durations cannot be added to GPU time. No supported device execution-profile feedback option was found in the pinned nvcc/nvlink/ptxas help or CUDA12.9.1 documentation; nvcc --profile concerns Linux gprof and nvcc.profile is compiler configuration.

Evidence: cuda-profile-guided-discovery-evidence.json and preserved command/help outputs and hashes. Official references: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#optimization-of-separate-compilation and https://learn.microsoft.com/en-us/cpp/build/profile-guided-optimizations?view=msvc-170 . This was discovery only: no PGO/LTO build, GPU test or service speed comparison occurred, and no claim that PGO can never help is made.


### 036 isolated source prepared; initial whitespace failure preserved

The four-file candidate adds a private mmq-pq2-i64.cu/.cuh translation unit, a six-line guarded host bridge and the same 21 native reference cases as035. Generic helper headers and CMake remain unchanged. CPU coverage/order and private-helper equivalence checks passed. Reviewed replacement patch: cuda-pq2-i64-stage-lifetime-source-v2.patch, SHA256 53278e69160cf76abeee4b26dbc7c6e796cb59e6df91769100a30a8d57080dbd; clean root application check passed without applying. Proof/source-check-v2 JSON, scripts, enumeration and source receipt preserve the evidence.

The initial export's two new files used CRLF, and git diff --check reported trailing-whitespace errors. The initial combined shell command incorrectly ended on git diff --stat and returned 0, so its inner Git exit status was not retained. The subsequent Python export returned 1 after asserting the failed check; its numeric inner Git status was not printed and no receipt was written. It had already preserved cuda-pq2-i64-stage-lifetime-source.patch. V2 normalizes only those files to LF; behavior, guards and tests are unchanged. Initial and V2 CPU proof/source-check commands all exited 0, and final standalone diff check/export exited 0. Both patches and all proofs remain preserved.

Independent review is checking the source. Existing CMake GLOB includes the new CUDA file only after configuration; canonical build must rerun configuration and confirm the translation unit appears in generated Ninja rules. Resource, compiled-order/invariance, native and service gates remain untested. No 036 build or GPU work has occurred.


### 036 independent source review and canonical build passed

Independent specification/quality review passed. All three private helper bodies match the035 SM86 specializations token-for-token after fixing constants; the only kernel-body change is the intended second-copy pragma. First staging, MMA/K order, indexing, barriers, writeback, guards and allocation behavior are unchanged. Declaration/definition/call agree, non-SM86 execution falls back, and the test file matches reviewed035's 21 cases.

The canonical wrapper reconfigured CMake and built successfully in 241.188 seconds. Generated Ninja and the compiler's [12/240] mmq-pq2-i64.cu.obj step confirm the new translation unit was included; receipt cuda-pq2-i64-stage-lifetime-build-inclusion.json. Three unused-variable warnings (I at line72, nwarps/I at lines159/160) in the fixed private helpers remain preserved in the build log. Full command/timing/exit0: cuda-build-pq2-i64-stage-lifetime.txt and .exit.json.

All eleven files were exclusively copied and hash-verified in tools/llamacpp-cuda-pq2-i64-stage-lifetime, manifest cuda-pq2-i64-stage-lifetime-binary-hashes.json. CUDA DLL SHA256 c73c928fd73b7fa921e68682bda145456c07684c9b0b2c653bfac36f30e2ad7b. Independent static resource/invariance and compiled staging/order analysis is running using the preserved030 baseline. No 036 GPU/service test has run; accepted030 remains served.


### 036 static gate passed; native correctness phase authorized

The new kernel uses exactly 128 registers, zero stack/local storage and no spill instructions. All 64 generic PQ2 MMQ plus both decode bodies/resources match retained030 under the established normalization. The future second-tile values no longer span first-half MMA: the first barrier is at 0x3010, and an eighteen-iteration copy loop loads at 0x3070 and stores at 0x30b0, four instruction slots apart versus412 in035. The next barrier is at 0x30d0. Independent compiled dependency inspection found 64 IMMA and 256 I2FP/FMUL/FFMA updates, with32 outputs at each accumulation depth1-8 in both binaries;036 splits depths1-4 before staging and5-8 after. Reviewed source establishes the mapping and38,400 dynamic shared bytes/256threads. These facts permit the proposed resource budget; they do not establish service speed or actual occupancy.

Evidence: cuda-pq2-i64-stage-lifetime-static-gates.json, candidate provenance, and static-staging-math.json (SHA25611f91beeed2b244efe85c3a2768dc4b5bcccea253ba9e9e059bf83794e548dc5). The reused baseline provenance was verified without another extraction; ten synthetic checker tests passed, including rejection of035.

After fresh Session0 inspection of wrapper24100/launcher20532/server23780 and four idle slots, maintenance stopped those exact processes. Receipts: cuda-maintenance/inspection036.json and stopped036.json. Original supervisor and retained030 binary snapshot remain preserved; no server/native-test process remained before authorizing native tests. Expected coverage is21 new references,101 retained MUL_MAT passes plus33 known F16-activation unsupported identities,44 fused references and8 staged-FFN references. All test outputs/counts and strict service checks still must pass before any speed acceptance.


### 036 native correctness and independent evidence review passed

All four native commands exited 0: 21/21 I64 references, 101/101 retained MUL_MAT references plus the exact 33 expected F16-activation unsupported identities, 44/44 fusion and 8/8 staged-FFN references. Total174 passes. Complete identities, multisets and footers were validated; no timeout or truncated output. Thirteen CPU-only validator checks passed. Evidence: cuda-pq2-i64-stage-lifetime-runtime-check.py, runtime-run1.jsonl, runtime-verdict.json and separate complete stdout/stderr files.

Independent review recomputed all33 recorded source/patch/executable/manifest/snapshot/build hashes, checked all eight raw output files, operation/parameter filters and actual CUDA-versus-CPU reference setup, and reran13/13 validator tests. The retained MUL_MAT reference multiset intentionally contains two duplicate executions. Printed CPU backend Skipping does not skip the numerical CPU reference. Native correctness is established within existing test tolerances; it is not an exact service-output or throughput verdict.

Deployed and reverified all11 candidate files at the stable service path. Candidate PID596 (parent6432) is healthy under Nsight with unchanged model/capacity/settings. The canonical512-token/32-output-token capture completed start/benchmark/stop with exit0; artifacts cuda-service-nsys036-prefill.* and cuda-nsys036-{start,bench,stop}.*. Offline mechanism analysis is pending. Growth and atomic four-slot strict comparisons against accepted030 references are running separately; no uninstrumented speed claim is made.


### 036 service mechanism and strict correctness passed; timing started

Actual-service analysis confirms736 new launches across the two508-token prefills,368 each, with exact expected grids,32x8 blocks,128 registers and38,400 dynamic shared bytes. All66 evaluations match expected substitution inventories:2419 kernels per508 prefill,2407 per4-token tail and1935 per decode. All30 indexed-GDN replay checks and complete copy-size/direction inventories match030; no state-sized D2D copies occur. Recorded execution reconciles with no unmatched activity, unexplained launches or CUDA API failures. The generic collection warning remains recorded.

The initial validator failed with exit2 because it incorrectly required localMemoryTotal==0. Both030 and036 report identical nonzero totals across129,622 kernels:129,558 at23,855,104 bytes and64 at24,772,608, with zero localMemoryPerThread throughout. The corrected check requires exact baseline-total equality and retains the zero per-thread and compiled no-spill gates. Original failure and v1 sources remain preserved. Corrected helper SHA2563b622087f548aa23f2d008f9cc6dccc5aee27316434ee39fa843ab5f492f9013 passed19 CPU tests and actual analysis exit0. Evidence: cuda-nsys036-service-analysis-v2.json, analysis-v2.exit.json, tests-results-v2.json and export receipt.

Growth at512/4096/16384/512 and atomic four-slot checks both exited0 against accepted030. Independent final review verified all8 responses: exact requests, tokens, content, probability identities and all768 log-probability values (maximum difference0.0). Model,188416 context and four slots remain unchanged. All four source files match the reviewedV2 patch; no drift. Three unused constants are a minor cleanliness issue, not a correctness blocker. Review passes conditional on the original repeated throughput gates.

After inspecting idle diagnostic PID596, stopped it. Nsight launch parent returned4294967295/tool1 due to intentional target termination, separately from successful capture/export. Deployed and verified all11 retained030 files; fresh uninstrumented control PID22432 began separate conditioning plus three measured requests per prompt. No fourth win is accepted yet.


### 036 rejected: large actual-service ingest regression

Fresh retained030 and036 candidate processes each completed conditioning plus three measured requests per prompt. All exact request/token/content checks passed, but the first uninstrumented comparison failed the original ingest gate:

| Prompt | Retained ingest | Candidate ingest | Ingest change | Retained output | Candidate output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 424.944 | 297.999 | -29.873% | 36.249 | 36.848 | +1.652% |
| 4096 | 509.682 | 346.647 | -31.988% | 33.952 | 34.556 | +1.779% |

The 128-register resource result did not translate into faster service ingestion. These measurements do not isolate how much came from staging-loop overhead, lost overlap, activation rereads or actual residency. Output changes are observed pair differences, not retained decode wins. The first-pair failure rejects036; no reverse pair or selective rerun was used to rescue it. Evidence: cuda-service-pq2-i64-stage-lifetime-{control,candidate}-a.jsonl, separate conditioning files and comparison-a.json (exit1).

After separate inspection of idle candidate PID2548, stopped it. Verified the two new files match the preserved isolated source and resolve inside the repository, reversed the exact V2 patch, and verified all root source files match HEAD. Candidate worktree, source patches, snapshot and all proofs/build/native/trace/benchmark artifacts remain preserved. Canonical deployment restored all11 accepted030 hashes; exact original supervisor was restored and normal restart initiated. Retained win count remains three.

Normal-service restoration passed health/model checks, four idle188416-context slots, eleven retained030 binary hashes and the original supervisor hash. Receipt: cuda-pq2-i64-stage-lifetime-restored-service.json.


### 037 pre-registered new scope: GDN COLS2 for full512 batches only

Independent read-only assessment supports a distinct narrower candidate. The recorded long-request graph log cuda-service-lazy-reserve-graph-a.log shows4096 as seven512 batches followed by508+4, and16K as31x512 followed by508+4. Repeated512 requests use508+4, independently confirmed by030/034/036 Nsight captures. The checkpoint split is implemented in tools/server/server-context.cpp. Fresh long-request dispatch verification is required because the explicit long-request graph record predates034.

Change034's guard to n_tokens==512, keeping actual508 batches on the retained kernel. Expected new dispatch is7x48=336 COLS2 launches per4K prefill and zero for the current512-token benchmark. This is a batch-width optimization, not a general long-request detector: other lengths, concurrency, caching, checkpoints and batch settings can change eligibility. The prior034 failure remains a rejection. Both0344K pairs improved (+4.462% and+3.725%), which motivates testing this scope but does not prove the gain survives removing508 specialization.

Static gate: at most64 registers (prior56 expected), zero stack/local/spills, unchanged default GDN/PQ2 compiled paths and ordered recurrence. Nine focused native references retain widths507/508/511/512/513; move permuted-V positive and B2 fallback to512 so width alone does not exclude their intended paths. Expected two eligible and seven fallback cases. Retain indexed replay, strict growth/atomic comparisons and fresh4K dispatch/cache-write verification.

Prospective service gate: BOTH fresh pairs and pooled ABBA must show >=2%4K ingest gain, <=2%512 ingest regression, and <=2% output regression at both sizes, with exact correctness/capacity/settings. Use separate conditioning and five measured requests per prompt in every fresh process, preselected to reduce the short-request variation seen in034. The shorter benchmark is now a non-regression control because its kernel dispatch is deliberately unchanged; this criterion is prospective and does not accept the rejected034 candidate. Extend the canonical comparator with an explicit per-prompt ingest threshold override, strict validation and complete threshold provenance, preserving existing defaults and all correctness checks. No new throughput result exists yet.


### 037 isolated source prepared

Two-file candidate (+36/-1) is preserved at cuda-gdn-cols2-full-batch-source.patch, SHA25698c44c81f062943f39c858c4f18e57e024ce59b65d7cf9e32b02d2ac7a6974a1. Root application check passed without applying. Author source proof verifies that CUDA source equals034 except for narrowing the guard to512; the recurrence/default/indexed paths remain unchanged. The native selector is GATED_DELTA_NET_COLS2_FULL_BATCH, nine cases with two eligible and seven fallback. Permuted-V and B2 checks use512 as pre-registered. Proof: cuda-gdn-cols2-full-batch-proof.json.

Preparation had one guide-display failure (exit1): cp1250 stdout could not encode the guide text. UTF-8 display corrected it; evidence is cuda-gdn-cols2-full-batch-preparation-failure.json. No source or CPU proof failed. Independent source review, runtime-tool preparation and canonical per-prompt comparator implementation are running separately. No037 compilation or GPU test has occurred; normal service remains accepted030.


### 037 build, static checks and comparator extension passed

Independent source/specification and quality review passed the exact two-file patch. The canonical build exited 0 in 47.078 seconds; full output and receipt are cuda-build-gdn-cols2-full-batch.txt and .exit.json. All eleven files were exclusively copied and hash-verified in tools/llamacpp-cuda-gdn-cols2-full-batch. CUDA SHA256 055a0177f3d1b276a42c30ba9a642272a817f5f67045eb34f507daa8296938be; test-backend-ops.exe SHA256 f373c62ea2922ed62a7c80782fea4878abeddf5172fd279bc5b49e23d821a36b.

Independent static checks passed: new COLS2 uses 56 registers, zero stack/local storage, no local load/store instructions, and 280 instruction slots. Its instructions/resources exactly match034. All34 retained GDN variants match030 after only the default COLS_OVERRIDE=0 name mapping; both PQ2 decode variants remain exact at42/37 registers. Nine synthetic checker tests passed. Evidence: cuda-gdn-cols2-full-batch-static-run2-{provenance,gates,candidate-functions,synthetic}.json and full extraction outputs. The first static attempt stopped before extraction because of a transcribed034 SHA typo; the failure was preserved and the value corrected against recorded metadata and the actual DLL. No baseline extraction was repeated. Runtime geometry, correctness and throughput remain untested.

The canonical comparator now supports repeatable --min-ingest-gain-pct-for PROMPT=PCT. The prospective037 gate is --min-ingest-gain-pct -2 --min-ingest-gain-pct-for 4096=2 --max-decode-regression-pct 2, applied to each pair and pooled ABBA. Missing overrides preserve prior validation/output. Malformed, nonfinite, duplicate, unknown-size, partial or mixed gates are rejected; every prompt retains exact correctness checks. Complete global/override/effective thresholds survive later correctness failures; earlier unresolved failures explicitly say so. Independent V1 review found that failure-provenance gap, closed in V2 before integration. V2 patch SHA256 4cb9b59448935c7bfc38cd7d87c5fe66c5dab2d4f17a44df1e299c2965b5b7b4; all22 CPU tests and independent delta review passed. Root verification also passed22 tests (bonsai-compare-tests-vm3giyf_). No throughput threshold changed after measuring037; no037 throughput measurement exists yet.


### 037 native correctness independently verified; matched service traces started

After separate Session0 inspection confirmed wrapper17404, launcher19728 and idle server14552, maintenance stopped those exact processes. Receipts: cuda-maintenance/inspection037.json and stopped037.json. The original supervisor and accepted030 snapshot remain preserved for restoration.

All five native commands exited0 without timeout: zero-case construction smoke, nine full-batch references,49 retained GDN plus14 exact expected unsupported identities,10 indexed cases and one fresh replay case. The latter verified direct, capture, replay, replay with one graph key/UID. Evidence: cuda-gdn-cols2-full-batch-runtime-verdict.json, runtime-run1.jsonl and ten complete raw outputs. Independent review recomputed26 native preflight hashes and all output hashes. It also verified24 static provenance artifacts, independently reparsed all37 selected bodies/resources and reran nine static mutation checks. Specification, quality and evidence passed; generic PQ2 MMQ is outside this selected static inventory, and native shape labels do not prove service dispatch.

Stable service files were reverified against all eleven accepted030 hashes. Fresh diagnostic baseline PID19216 is healthy with four idle188416-context slots. Matched baseline and candidate captures use the canonical benchmark with --prompts 512,4096 --tokens32 --reps1 (warmup512, warmup4096, measured512, measured4096). The offline validator passed16 CPU fault-injection checks and requires146 evaluations, with expected candidate COLS2 counts0/336/0/336. No uninstrumented037 throughput result exists yet.


### 037 service correctness passed; initial trace-validator assumption failed

Matched accepted030 PID19216 and037 PID24476 captures completed start, canonical benchmark, stop and SQLite export with exit0. Complete artifacts are cuda-service-nsys037-{baseline,candidate}.{nsys-rep,jsonl,sqlite}, with cuda-nsys037-{baseline,candidate}-* command receipts/full outputs. Both diagnostic launch parents later returned4294967295 (tool1) after intentional inspected idle-server termination; these are separate from successful collection. Growth512/4096/16384/512 and atomic four-slot correctness commands both exited0 against accepted030 references; artifacts cuda-gdn-cols2-full-batch-{growth,atomic}-correctness.jsonl.

The initial offline validator exited2 at the two long-request508 evaluations40/113: it reused the short-request2419-kernel count, while BOTH baseline and candidate have2323, with exact equal inventories. Investigation identified48 scale_f32 clears of30,720 conv-history values plus48 clears of786,432 recurrent-state values present only at request-first evaluations0/33/73/106. qwen35.cpp state reads and llama-graph.cpp's rs_zero-dependent scale_inplace explain those96 calls. A context-dependent32-call Q4 dequant grid also grows from2048 to16384 without changing count. This is a validator assumption error, not a candidate discrepancy. Original helper/output/exit2 are preserved. V2 will retain fixed per-position counts, explicit clear presence/absence and full matched inventories rather than dropping coverage. No throughput result exists yet.

### 038 pre-registered compile-first decode hypothesis: unsigned packed-weight reads

Read-only source/SASS discovery identified redundant sign-extension PRMT instructions from int16_t packed PQ2 reads in vecdotq.cuh. The retained ordinary warp kernel has eight PRMT 0x9910 extensions per paired iteration (four per logical chunk); the fused kernel has four per chunk. Evidence is cuda-pq2-block-index-sass-baseline.txt, checked against accepted030 CUDA a81640f1579b2d8a862747958c021f3505bc7e12b4bebd31466283a3aeb9ccc4. A CPU exhaustive emulation covered65,536 packed words and524,288 coefficients: signed/unsigned lookup results match all four codec values because the sign-extension bits select duplicated lookup-table entries. This is distinct from rejected019 activation permutations and from025/029 unrolling or032/033 address changes.

Propose a private helper used only by the two specialized calls in mmvq.cu, with uint16_t payload reads into uint32_t values. Preserve the existing SM86/single-column/no-IDs/no-scales guard,16-bit load widths, memory layout, loop expansion, lookup permutations, DP4A/scaling/FP32 accumulation order and epilogue. All generic paths remain unchanged. No source implementation, build or GPU test has occurred. Memory traffic is unchanged; scheduling/register pressure may erase the instruction saving, and019's larger instruction reduction failed. No speed prediction is accepted.

Compile gate: remove the identified extensions and save at least four net instructions per logical chunk in BOTH kernels, registers no higher than42 ordinary/37 fused, zero stack/local/spills, same loads/loop expansion/ordered math and all non-target kernels unchanged. Reject before GPU use on failure. Then require exhaustive codec proof,101 PQ2 references plus exact33 known unsupported identities,44 fusion and8 FFN references, strict service growth/atomic correctness and actual dispatch verification. Service acceptance: BOTH fresh pairs and pooled ABBA show >=2% output gain at512/4096 and <=2% ingest regression, unchanged capacity/settings, separate conditioning and five measured requests per prompt. Use the current retained build as baseline;037 is still undecided. Source-only preparation may overlap037 review/timing, but no compilation or GPU work may overlap service measurements.


### 037 corrected mechanism check and strict service review passed; uninstrumented comparison started

Independent V2 analysis verified all146 evaluations,672 COLS2 launches distributed0/336/0/336, exact per-position kernel counts, state-clear presence/absence, retained inventories/resources, copy inventories and indexed decode/replay. No state-sized D2D copies, CUDA runtime API failures or unmatched recorded activities were found. Eighteen CPU regression checks passed. Helper SHA256838bd56947c18dfaeac8e997dedc9d38b9cea1fe96e18a297568e9e5dfbece11; results cuda-nsys037-service-analysis-v2.json and analysis-v2.exit.json. The initial failed helper and results remain preserved.

Strict independent review verified all eight growth/atomic responses,128 tokens and768 probability values exactly match030 (maximum difference0); all four capture benchmark requests and128 output tokens also match. Proof: cuda-nsys037-correctness-review.json. Model/context/four-slot capacity remain unchanged.

Profiler limitation: each fresh trace contains5,571 CUPTI GetGraphId and5,571 GetGraphNodeId severity3 errors, all inside successful cudaGraphExecUpdate intervals, plus the generic collection warning. Both traces have the same count, all runtime APIs succeed and final30 replay signatures contain all1935 expected kernels. Recorded submissions/inventories reconcile, but complete profiler collection cannot be asserted. Diagnostic evidence: cuda-nsys037-profiler-diagnostic-review.json. This matched profiler metadata problem is preserved separately from actual CUDA runtime failures.

Stopped inspected idle diagnostic PID24476, deployed and verified all eleven030 files, and started fresh uninstrumented baseline PID22016. The prospective separate conditioning plus five measured requests per size has started. No fourth retained win is claimed.


### 038 isolated source prepared; exhaustive packed-word proof passed

Candidate source is preserved in cuda-pq2-unsigned-payload-source.patch, SHA2568591b395d053885333c6223ce452bcbb3035187d1f54ab6e04aee07514b4a456, based on b1a9528578baee4da2ab6856ecc5f26b60cad1c9. Only mmvq.cu changes (+42/-2): a private helper and the two specialized calls. The generic vecdotq helper and remaining mmvq source are byte-identical. No new native cases were added because existing101 PQ2/33 unsupported,44 fusion and8 FFN cases cover the intended dispatch, alongside the new exhaustive codec proof.

Reproducible CPU proof covered65,536 packed words and524,288 coefficients twice, with the same patch hash and no preparation failures. Artifacts: cuda-pq2-unsigned-payload-proof.py, proof.json, verify2-proof.json, discovery-sass.{txt,json}, enumeration and final-check receipts. Independent source review and compiled-checker preparation are running. No038 build, GPU execution, source integration or throughput measurement has occurred.


### 038 compiled-checker preparation: synthetic failure corrected before use

The CPU-only checker now requires an explicit exact baseline/candidate binary binding and inventories90 PQ2-related kernels:64 MMQ/fixup,16 generic MMVQ,one MoE,four get_rows,three dequantize and two target warp kernels. All88 non-target instruction bodies/resources must match. Verified030 cached disassembly covers66 bodies, leaving24 to extract later; verified037 cache covers two PQ2 bodies, leaving88. No extraction, compilation or GPU command ran during037 service timing.

The first fault-injection test caught missing tail-prefix lookup screening. V1 and its failed result remain preserved; V2 corrected it and passed25 CPU fault tests. Helper cuda-pq2-unsigned-payload-static-check-v2.py SHA256669fe93c00e4dda498ee3ad2fe23f1d5760f2f362a8aa9bde6c24b630cdec062; preparation evidence cuda-pq2-unsigned-payload-static-prepare2-{provenance,tests}.json. Passing target instruction/register/spill/load/lookup/math/loop screens produces REQUIRES_SEMANTIC_REVIEW, not automatic semantic approval: physical-register dependencies, pointer identity and ordered accumulation require independent operand review of the actual compiled candidate.


### 037 first uninstrumented pair passed; reverse pair required

Fresh030 PID22016 and037 PID6508 each completed separate conditioning and five measured requests per prompt with exit0. Exact request/token/content comparison passed. First pair:512 ingest411.013558 to419.477177 (+2.059207%), output36.797678 to36.835818 (+0.103646%);4096 ingest515.524382 to535.526762 (+3.880006%), output34.530260 to34.502275 (-0.081047%). Both prospective per-prompt gates passed. The short-request difference is an observation, not evidence of a changed short-prompt kernel. Artifacts: cuda-service-gdn-cols2-full-batch-{control,candidate}-a.jsonl, their separate conditioning files/receipts, and comparison-a.json (exit0).

After separate inspection of idle PID6508, stopped it and started a fresh037 process PID12580 with all eleven snapshot hashes verified. Candidate-B conditioning/measurement is running; fresh control-B and both reverse/pooled gates remain required. No fourth win is accepted yet.

### 038 independent source and exhaustive codec review passed

Independent specification/quality review verified the exact one-file patch and baseline reconstruction. It independently exhausted all65,536 words/524,288 coefficients, including32,768 negative signed interpretations, against the direct codec oracle {-1,0,1,2}. Every coefficient matches. All sixteen two-byte payload offsets remain aligned within each34-byte block; the existing type-punning and endianness assumptions are unchanged. Lookup permutations, signed DP4A order and floating-point scaling are preserved; maximum chunk magnitude is8192. Outer accumulation, reduction, fusion, guards and geometry remain exact.

Review rehashed accepted030 DLL/SASS/metadata and confirmed eight ordinary paired-loop and four fused-loop sign extensions, four per logical chunk each. The subsequent0x7777 selector masks support the duplicated-table equivalence. No source blocker was found; private helper duplication isolates the experiment. Approval is safe-to-compile only, after037 timing ends. All compiled/native/service gates remain unresolved.


### 038 native-runner preparation passed CPU checks

The prepared runner requires future actual038 CUDA DLL, test executable and test-source hashes plus the actual build-base ancestor, allowing037 if retained. It runs exactly101 PQ2 passes with33 known unsupported identities,44 fusion and8 FFN passes; no I64 selector or zero-match smoke is counted. Fourteen CPU-only regression checks passed and its validation function is AST-identical to the verified036 validator. No native command or compiled hash binding occurred. Artifacts: cuda-pq2-unsigned-payload-runtime-ready.json, runtime-check.py and runtime-synthetic.json. Runner SHA256a133a5101308cf8e7c527f8b1fbaf04537c1f314c8a8e13f01c5b38b1a26a72b.


### 037 accepted: fourth retained win, full-batch GDN ingestion

Fresh reverse pair passed:512 ingest414.906824 to416.014951 (+0.267078%), output36.848289 to36.849312 (+0.002775%);4096 ingest514.210398 to536.484973 (+4.331802%), output34.537767 to34.591928 (+0.156817%). Both individual comparisons and pooled ABBA pass every prospective threshold. Each process had separate conditioning and five measured requests per prompt, with no excluded or selectively repeated measured trials.

| Prompt | Retained030 ingest | Accepted037 ingest | Ingest change | Retained030 output | Accepted037 output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 412.960191 | 417.746064 | +1.158919% | 36.801566 | 36.842565 | +0.111404% |
| 4096 | 514.859053 | 536.198446 | +4.144706% | 34.534014 | 34.562791 | +0.083331% |

Independent final source/specification, quality and retention review passed. It recomputed all40 measured trials, verified all64 measured/warmup/conditioning responses and16,384 output tokens exactly, and confirmed unchanged188416 context, four slots,256 outputs and all-core affinity. All three deployment receipts match all eleven snapshot files (33 hash comparisons), and timestamps follow ABBA order. Exact two-file source patch matches SHA25698c44c81f062943f39c858c4f18e57e024ce59b65d7cf9e32b02d2ac7a6974a1.

Retain for long-prompt ingestion with actual full512 batches; do not claim a short-prompt kernel improvement or decode gain. Both4K pairs improved (+3.880006% and+4.331802%). Snapshot tools/llamacpp-cuda-gdn-cols2-full-batch, CUDA SHA256055a0177f3d1b276a42c30ba9a642272a817f5f67045eb34f507daa8296938be, becomes the next experiment baseline. Artifacts: cuda-service-gdn-cols2-full-batch-comparison-{a,b,abba}.json and original command receipts/JSONLs.

Evidence limitations remain: per-process executable/hash checks were contemporaneous root console evidence, while the original deployment receipts persist source/target/file hashes. CandidateB restarted the unchanged candidate deployment. cuda-gdn-cols2-full-batch-timing-provenance.json is an explicitly retrospective evidence index, not an invented historical hash measurement. Matched CUPTI graph-metadata warnings remain recorded and complete profiler collection is not asserted. These limits do not contradict the exact recorded mechanism, correctness or repeated service gain.

Retained win count is now four. Normal-service deployment/restoration is the next operation;038 remains source-only until this integration completes.


037 integrated and pushed as e801f710edbdffe422c638a650129758c30939c8. Normal service is restored on accepted037: health ok, correct model, four idle188416-context slots, all eleven runtime hashes and original supervisor hash verified. Server PID10128; receipt cuda-gdn-cols2-full-batch-restored-service.json. After verifying exact committed source equality, no extra files and preserved patch/branch/snapshot, removed the accepted candidate temporary worktree. Rejected candidates and all experiment artifacts remain preserved.


### 038 canonical build passed on retained037 base

Applied the reviewed one-file patch on clean retained base6d2111fd9fb618b3c28884c5976cebc4b111cfb7; only mmvq.cu changed. Build/source/test binding is preserved in cuda-pq2-unsigned-payload-build-source.json. Canonical configuration/build exited0 in57.472 seconds, full output cuda-build-pq2-unsigned-payload.txt and .exit.json.

All eleven files were exclusively copied and hash-verified in tools/llamacpp-cuda-pq2-unsigned-payload, manifest cuda-pq2-unsigned-payload-binary-hashes.json. CUDA SHA25615284ff3361890e1ac11de31c7ed3290e68f8287b37c3d6646d1052bbef88e5d; test executable SHA256fda38f489f417b354e665e1dd5681a6d7aba6c323ae83f844f3810e8453e8d21; test source SHA256e9e01e08ab7446025244b0f540d43a18d85efdad7316e2f8b97e6c557716eb04. Offline static comparison and independent target dataflow review are running against exact accepted037. Runtime preparation may bind hashes only; no038 GPU/native/service test is authorized before static approval. Normal service remains accepted037.


### 038 preflight passed; static register gate failed and candidate rejected

Runtime preflight exited0, verifying27 file hashes and the actual retained037 build base. It launched zero native commands. Evidence: cuda-pq2-unsigned-payload-runtime-preflight1.{jsonl,exit.json,stdout.txt,stderr.txt}. The initial attempt to read a ledger heading failed with StopIteration; runtime-preflight-read-failure.json preserves that failure, and the corrected read succeeded. This was a preparation lookup failure, not a numerical test failure.

All four disassembly commands exited0, but the compiled checker exited1: ordinary decode registers increased42 to43, exceeding the prospective42 limit. Fused remains37. Both have zero stack/local storage or local spill instructions. Ordinary loop instructions fell172 to159 per paired iteration (6.5 per logical chunk), and fused138 to131 (seven per chunk), with the identified sign extensions removed. Load counts remain28/19; fused packed payload accesses change signed16 to unsigned16 as intended. All88 non-target PQ2 bodies/resources match accepted037 exactly. Instruction reduction is not a measured speed result.

The register failure alone rejects038 under its original gate; no native GPU, service correctness or throughput test ran, and the gate was not relaxed. Evidence: cuda-pq2-unsigned-payload-static-compiled1-{provenance,screening,baseline-functions,candidate-functions,build-binding}.json and full baseline/candidate resource/SASS outputs. Independent follow-up may inspect the concrete lifetime change for a new source hypothesis; it does not revive038.

Verified the root diff exactly matches the preserved patch, reversed it, and confirmed root source matches retained037. The isolated038 worktree, patch, compiled snapshot and all outputs remain preserved. Normal service was never changed during038 and remains accepted037; retained win count stays four.


### 038 independent failure confirmation and 039 prospective lookup-lifetime hypothesis

Independent raw-resource review confirmed038 ordinary REG43 versus accepted42, fused37, stack/local0 and exact88 non-target kernels. Ordinary still has28 loads (12 unsigned16-bit and16 regular), with changed scheduling. Fused still has19 loads; four signed16-bit payload loads become unsigned16-bit as intended, without added traffic.

Candidate038 ordinary SASS materializes three lookup-table copies at0xa60/0xa70/0xa80 before forming the first selectors, versus one before baseline sign-extension/selector processing. Its selector usesR40 versus baselineR39; there are14 lookup-table materializations per paired iteration. This supports a source-lifetime hypothesis, not a proof of the compiler allocator's internal decision.

039 proposes changing only the ordinary private helper to select exclusively from the second byte-permute operand: __byte_perm(0, 0x020100FF, (q & 0x3333u) | 0x4444u) and its q>>2 counterpart. Keep038's fused helper unchanged, and retain unsigned16-bit payload reads. Independent CPU emulation verified131,072 proposed permutations across all65,536 words. The zero first operand may remove the live vector-register lookup copies; extra selector masking may add an instruction per lookup. Removing14 materializations while adding16 selector instructions would still leave about5.5 saved instructions per chunk if other code generation stayed unchanged; this is conditional instruction accounting, not a speed prediction.

Prospective039 compile gate: remove the identified lookup-copy live ranges, ordinary registers<=42/fused<=37, zero stack/local/spills, >=4 net instructions saved per logical chunk in BOTH kernels versus accepted037, unchanged loads/loop expansion/ordered arithmetic/geometry, fused instructions/resources exact038, and all88 non-target PQ2 kernels exact accepted037. Reject before GPU execution on failure, without relaxing038. Preserve exhaustive codec proof and the same153 native passes/33 expected unsupported identities, strict growth/atomic service checks and actual dispatch verification. Service gate remains both fresh pairs and pooled ABBA with>=2% output at512/4096,<=2% ingest regression, unchanged capacity/settings, separate conditioning and five measured requests per prompt. No039 source, build or runtime experiment has occurred yet.


### 039 isolated source and exhaustive proof prepared

The one-file candidate (+46/-2 in mmvq.cu) is preserved in cuda-pq2-zero-lut-source.patch, SHA256 58496ad1eee2bf5fb01215186a428b358638417b2362c2476ffc76a6abab865e, based on 3bbb9983291ee0966df734b4172a854e8ad79d6b. A private zero_lut template parameter selects the new ordinary expression; fused calls retain the exact 038 expression. Root application check passed without applying the patch.

Author CPU proof passed all 65,536 packed words, 524,288 coefficients and 131,072 proposed lookup calls. Ordinary 039, fused 038, signed 037 and the direct four-value codec agree. Generic helper, test source, arithmetic order, guards and geometry are unchanged in the source proof. Evidence: cuda-pq2-zero-lut-{enumeration,proof,final-checks}.json and reproducible proof.py. The initial ledger-heading lookup failed with StopIteration (exit1), preserved in preparation-failure1.json; the corrected read succeeded. Independent source review and compiled-checker preparation are running. No 039 build or GPU execution has occurred.


### 039 independent source review and static-checker preparation passed

Independent specification and quality review passed. The actual one-file diff matches the recorded patch. Helper definition, both calls and both launches were enumerated: ordinary instantiates true through !has_gate, and fused instantiates false. Independent exhaustive comparison covered all 65,536 words and 524,288 coefficients against the direct codec. Selector masks choose only bytes4-7 of the second operand with bit3 cleared; no shift/signedness ambiguity was found. Two-byte access bounds/alignment and existing type-punning/endianness assumptions remain unchanged. Resolving false reproduces 038 helper expressions exactly, and removing the helper/restoring call names reproduces retained source. Approval is safe-to-compile only.

The prepared static checker passed 20 CPU fault checks without native extraction. It reuses all 90 verified accepted037 bodies/resources and pins fused output to the exact038 body/resources. It requires removal of ordinary LUT materializations, recognizes zero-first-operand lookups, permits the intended selector-mask change and retains load-width/offset, math/control, register/spill and non-target checks. Successful screening still requires independent operand/dataflow review. Helper cuda-pq2-zero-lut-static-check.py SHA256 9e2abd1278e61759ca75c1887dd7a94140b5b123650264e603b7bfbd3b1f07f3; evidence cuda-pq2-zero-lut-static-prepare-{provenance,tests}.json.


### 039 canonical build and snapshot completed

The reviewed patch was applied on clean retained base 9dca3b1fd15092c5907b133736348fb1ec1b520d, with only mmvq.cu changed. Canonical configuration/build exited0 in 57.046 seconds. Full output and receipt: cuda-build-pq2-zero-lut.txt and .exit.json; source/test/base binding: cuda-pq2-zero-lut-build-source.json.

All eleven files were exclusively copied and verified in tools/llamacpp-cuda-pq2-zero-lut. Manifest: cuda-pq2-zero-lut-binary-hashes.json. CUDA SHA256 bd2cf32c161c7b8006c6b75b737cb2b7f7b678d4edde5721a4832c09824d8909; test executable SHA256 0f789eccd619f139205138099d894c80593d6d21d701a238a7299a3d9540de34. Source/test hashes still match the pre-build binding. Offline compiled checks are running against cached exact037 and the exact038 fused reference. No 039 native/GPU/service test has occurred; normal service remains accepted037.


### 039 resource and instruction targets met; ordered-math screening needs dependency review

Both candidate extraction commands exited0. Ordinary uses 40 registers versus accepted42; fused remains37. Stack/local/spill checks pass. Ordinary paired-loop instructions fall172 to160 (six saved per logical chunk); fused138 to131 (seven per chunk). All fourteen identified ordinary loop LUT materializations are removed, and zero-first-operand lookup/mixing screens pass. All88 non-target PQ2 bodies/resources match accepted037; fused instructions/resources exactly match038.

The checker nevertheless exited1 because its ordered_math_shapes signature differs for ordinary decode: DP4A instructions interleave differently between integer accumulation chains. This screening difference does not by itself establish changed per-output arithmetic. Original failure is preserved in cuda-pq2-zero-lut-static-compiled1-{screening,provenance,candidate-functions,build-binding}.json and complete candidate resource/SASS outputs. Independent operand/dependency review must prove or refute the ordering, including the exact fused038 body versus037; the math gate has not been waived. CPU-only native/trace tooling preparation can proceed, but no039 GPU or service trial is authorized yet.


### 039 runtime preflight and service-trace validator prepared

Native-runner preparation passed 14 CPU tests and preflight exit0 verified27 file hashes, with no preparation failures or native executions. Validation retains exactly101 PQ2 passes plus33 expected unsupported identities,44 fusion and8 FFN cases. Runner cuda-pq2-zero-lut-runtime-check.py SHA256 bc263e0ea46a6f4af80bd90c78281247e893cefbe0ef50041e75655bb57d1412; ready/synthetic/preflight1 artifacts are preserved under cuda-pq2-zero-lut-runtime-*.

The future service-trace validator passed21 CPU tests in10.913 seconds using preserved accepted037 evidence. It pins baseline PID24476/trace hashes and the actual039 DLL, requires canonical benchmark/exit provenance, and checks146 evaluations with exact inventories allowing only ordinary registers42 to40. Expected target presence is39,808 ordinary and4,960 fused launches;672 full-batch GDN launches remain. Copy inventories, replay signatures,1935-kernel decode/48 indexed nodes, recorded submission coverage, API errors and baseline-relative local-memory reservation are retained. Full profiler warnings are recorded, including the baseline11,142 CUPTI metadata errors and generic collection warning; changed diagnostics require review. SQLite cannot independently attest loaded DLL bytes, and complete collection is not asserted. Helper cuda-nsys039-analysis.py SHA256 548455610851a58a4a0b940325832cd0a5c8eb107f62a603901805434bbb7bb4; proof cuda-nsys039-tests-results.json. No039 candidate trace or service action has occurred. Independent compiled dependency review remains pending.


### 039 independent compiled dependency gate passed

The secondary operand-aware review passed for ordinary tail/paired loop and fused038 versus accepted037. It preserves each nested DP4A dependency and FP32 operand order; both ordinary chunks still accumulate sequentially into the same output. Exact load-address expressions, widths/cache policy, packed selectors, preheaders, branches, reductions, biases and all four GLU cases match. Direct ELF jump-table reads confirm corresponding relocated GLU bodies. Resource/instruction gates remain40/37 registers, zero stack/local/spills, six/seven saved instructions per chunk, exact038 fused body and exact88 non-target kernels.

Twenty-four CPU regression checks passed, including deliberate real-SASS mutations of same-chain ordering, operands, accumulators, pointers, selectors, scales, signed zero and control flow. The initial global ordered_math_shapes failure remains untouched; the correction compares dependency trees per output instead of globally interleaved instructions. Evidence: cuda-pq2-zero-lut-semantic-{check.py,tests.py,verdict.json} and complete test/review outputs/exit receipts. Checker SHA256 c55ea6e0eb4094d90321e112be2db5c96834c12505dd93d60bb969b4830a399e. Scope is the exact pinned SASS subset, not a general CUDA emulator.

Independent verdict permits native correctness testing only. Service correctness/dispatch and both paired/pooled throughput gates remain required. Maintenance inspection of the currently idle accepted037 service has started; no039 GPU test has run yet.


### 039 native correctness independently passed; service capture started

After separate Session0 inspection and idle checks, maintenance stopped exact wrapper24812, launcher14828 and server10128. Receipts: cuda-maintenance/inspection039.json and stopped039.json. All three native commands then ran once and exited0:101 MUL_MAT passes plus33 exact expected unsupported identities,44 fusion passes and8 FFN passes. No timeouts or selective reruns. Runtime-run1 JSONL, monitor95961 completion, verification JSON and six complete raw outputs are preserved under cuda-pq2-zero-lut-runtime-run1*.

Independent review recomputed all27 source/binary bindings and six raw-output hashes, verified exact identity multisets and CUDA0/backend/process footers, and reran14 CPU validator tests. It found no failure messages in the retained initialization/graph-warmup stderr. Independent trace-analyzer review also passed all146 baseline-derived evaluations and rejected seven injected faults; actual candidate trace is still required.

Canonical deployment installed all eleven039 files at the stable path. Fresh diagnostic PID3432 is healthy with four idle188416-context slots. Live process enumeration verified nine loaded runtime module paths and their backing-file hashes, including ggml-cuda.dll, plus the full eleven-file snapshot. Receipt cuda-nsys039-process-provenance.json records this contemporaneous evidence and explicitly does not claim a hash of relocated module memory. The canonical short/long32-output capture has started. No uninstrumented performance result exists yet.


### 039 actual-service correctness and independent trace review passed; timing started

The actual candidate trace passed without checker corrections: 146 evaluations, 39,808 ordinary launches at 40 registers and 4,960 fused launches at 37. All other recorded resources, kernel inventories and copies match accepted037; full-batch COLS2 counts are 0/336/0/336 and the final 30 stable decode replays retain 1,935 kernels and 48 indexed GDN nodes. Four captured requests and 128 output tokens/content match. Start, benchmark, stop, export and analysis receipts all exit0. Artifacts: cuda-service-nsys039-decode.{nsys-rep,jsonl,sqlite}, cuda-nsys039-service-analysis.json and cuda-nsys039-*.exit.json.

Strict growth and atomic checks each passed four cases against accepted037. Independent review verified 128 output tokens/content and all 768 probability values and IDs exactly, maximum float difference0. It independently reproduced the complete saved trace analysis, all eleven snapshot/deployment/backing-file hashes, nine loaded module paths and PID3432 chronology. Matched CUPTI metadata warnings remain a collection limitation; backing-file hashes do not attest relocated memory. Source, native and service correctness permit throughput testing, not retention.

After inspecting idle diagnostic PID3432 and stopping it, deployed accepted037 and started fresh control-A PID12856. Saved cuda-service-pq2-zero-lut-control-a-provenance.json before timing, including all eleven backing-file hashes and loaded module paths. Separate conditioning (one repetition per size) and measurement (five per size) have started through the actual service. The original paired and pooled >=2% output gain at both512/4096 and <=2% ingestion regression gates remain unchanged. No fifth win is claimed.

### 040 prospective ingestion hypothesis: direct second-half staging into the existing I64 Y tile

Read-only follow-up of rejected036 identified its second-half staging loop as 18 iterations per thread, each with 11 compiled instructions including LDG followed by dependent STS. New040 will replace only that staging operation with 18 unrolled four-byte cp.async.ca.shared.global copies into the same shared Y tile. Preserve the pre-copy CTA barrier, commit the thread's copies, wait_group0, then preserve the post-copy CTA barrier and exact second-half matrix arithmetic. Keep the first staging copy, six exact K/M shape pairs, T508/512 eligibility, SM86 guards, shared layout and output ownership unchanged. Use the isolated private kernel/host bridge and existing 21 focused references from036; do not change generic MMQ headers.

The candidate targets register bypass and repeated scalar-loop overhead. Each of 256 threads copies l=256*n+tid for n0..17: 4,608 distinct words, 18,432 bytes, naturally four-byte aligned. Total shared allocation stays 38,400 bytes. The existing508 tail uses allocation padding already required by036. Prove exact bounds, address conversion, disjoint destinations, uniform synchronization and supported instruction/group semantics before compilation. Do not issue copies before the first barrier, because that would overwrite Y still used by the first-half matrix operation. This proposal does not recover that overlap and retains I64's doubled row-tile count and activation rereads. Rejected036's roughly30% regression does not establish which cost dominated, and there is no predicted throughput gain.

This differs from rejected013's sixteen-byte copies with a second shared Y buffer (57,856 to76,288 bytes). Official CUDA12.9.1 PTX documentation describes four-byte .ca copies on SM80+, per-thread commit/wait groups, and undefined behavior for overlapping destinations in one group: https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async . The CTA barriers remain necessary for cross-thread consumers; wait_group orders these async copies only.

Prospective compile gate: <=128 registers, zero stack/local/spills, 38,400 shared bytes, and actual direct async staging replacing the second LDG-to-STS loop. Independently verify first-copy behavior, every per-output ordered arithmetic dependency, addressing, barriers and dispatch. All retained MMQ, PQ2 decode and GDN kernel instructions/resources must remain identical to the current accepted binary; reject before GPU execution on a static failure. Then require 21 focused native passes plus101 PQ2/33 expected unsupported,44 fusion and8 FFN, strict service growth/atomic comparisons, and actual-service dispatch evidence.

Prospective performance gate: fresh accepted-baseline/candidate ABBA, each process with separate conditioning reps1 and measurement reps5 at512/4096 with256 outputs. BOTH individual pairs and pooled results must improve ingestion >=2% at BOTH sizes with output regression <=2%; all original request, context, slots, affinity and correctness constraints apply. Reject at the first failed pair, with no selective reruns or relaxed thresholds. Beating rejected036 is insufficient. Baseline is accepted037 unless039 independently passes and is retained first; preserve any retained039 decode patch unchanged. Only source/CPU preparation is permitted during039 timing; no build, GPU test or service change.


### 040 source prepared and CPU mapping proof passed

The isolated source candidate is based on22d3b3e1cacd8299b274a667de2fd6c992e98bbd. Patch cuda-pq2-i64-async-stage-source.patch, SHA2567753ce0bba8b4f1dff488daf5bf23beef2e8b4f0d2c9417af1bf5224d101ed99, adds329 lines across four files: private CUDA translation unit/header, the unchanged036 host bridge and21 focused native references. Its only private-kernel delta from036 is the second copy:18 unrolled four-byte async operations, explicit shared-space address conversion, compiler memory clobbers and commit/wait0 between the original CTA barriers. Generic headers, CMake, PQ2 decode and GDN remain untouched.

CPU proofs passed12 shape combinations,4,608 unique staging words and8,192 output owners, with padding, alignment, ordering and negative-control checks. Shared allocation remains38,400 bytes. Pinned PTX defines no per-group copy-count limit for this interface; this does not establish that18 copies can be physically outstanding without stalls. No preparation failure occurred. Artifacts: cuda-pq2-i64-async-stage-{enumeration,replacement,proof,source-check,final-checks}.json and reusable proof scripts.

Independent source review and offline static-checker preparation are running. No040 compilation, GPU execution or service change has occurred.039 baseline conditioning/measurement finished with exit0; candidate-A now runs through the service on fresh PID10132, with all eleven backing-file hashes and nine loaded module paths saved before timing. Performance conclusions remain pending the original gates.


### 039 rejected: instruction/register improvement did not produce the required service gain

Fresh baseline PID12856 and candidate PID10132 each completed separate conditioning reps1 and measurement reps5 at512/4096 with256 output tokens; all four benchmark commands exited0. Exact request/token/content comparison passed, but canonical comparison-a exited1 on the original performance gate.

| Prompt | Accepted037 ingest | Candidate039 ingest | Ingest change | Accepted037 output | Candidate039 output | Output change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 401.627218 | 406.258604 | +1.153156% | 36.810682 | 36.871066 | +0.164040% |
| 4096 | 534.895325 | 534.317995 | -0.107933% | 34.605715 | 34.685294 | +0.229957% |

Both output gains miss the prospective >=2% threshold. Reject immediately; no reverse pair, pooled acceptance, excluded trials or selective rerun. Fewer compiled instructions and40 versus42 ordinary registers are verified mechanisms, not a retained speed win. These results do not isolate the remaining hardware bottleneck. Artifacts: cuda-service-pq2-zero-lut-{control,candidate}-a.jsonl, separate conditioning files, contemporaneous per-process provenance, complete stdout/stderr/exit receipts and comparison-a.{json,exit.json}. Independent evidence audit remains pending.

After separately inspecting idle candidate PID10132, stopped it and restored accepted037 through the canonical deployment script. Receipt bonsai-deploy-20260923T060721005-48e3244da8624ae2bc1d61d760eec940.json. Verified exact root patch equality, reversed039 and confirmed clean source. Preserved the isolated039 worktree, patch, compiled snapshot and every successful/failed artifact. Restored the original supervisor bytes and restarted the scheduled service. Fresh PID23820 is healthy on the correct model with four idle188416-context slots; all eleven backing-file hashes and supervisor SHA match accepted references. Receipt cuda-pq2-zero-lut-restored-service.json explicitly records Session0 inspection limits.

Retained win count remains four.040 baseline is now definitively accepted037. Source review and CPU-only static/native tooling preparation continue; no040 build or GPU execution has occurred.


### 039 rejection audit passed; 040 source/tooling review and canonical build passed

Independent039 audit recomputed all20 measured samples and the exact failed medians. All32 conditioning/warmup/measured responses and8,192 tokens match, with no omitted repetitions. Four benchmark receipts exit0; the comparator's exit1 is the expected performance failure. Fresh-process chronology,22 snapshot hashes and18 loaded-module path/backing-file bindings match the deployments. Rejection stands without a reverse pair.

Independent040 source review passed specification and quality for compilation only. It enumerated6,340,608 copy addresses,4,608 disjoint aligned words per CTA,8,192 output owners and the X payload/scales. The508 tail reaches at most576 bytes into existing18,432-byte allocation padding; invalid columns cannot reach output stores. Source and control-flow comparisons confirm only the second staging operation changes. Proof: cuda-pq2-i64-async-stage-review-proof-v1.json.

The static checker passed22 CPU fault tests and independent review, binding125 retained accepted037 bodies. It explicitly requires independent semantic approval rather than equating opcode counts with correct operands/order. Helper SHA256346c70fdbc4704dfa9e980b385f21eced5d0dc5a9959b3c8b96794762040a2fe. Native-runner preparation separately passed22 CPU tests and independent review; exact174 pass and33 unsupported identities, validator and native execution loop match036. Runner SHA25658fab856a623ea5b5630bb6529446040e979065f93045e34442d47708b561cfc. Full stderr remains available and must be inspected after actual native execution; arbitrary stderr text alone is not automatically classified as failure. No preparation failure occurred.

Applied the exact reviewed040 patch on clean root base658b3c2efffee7bd526fe405a98ad70ef6a8fe94 and bound all four source hashes before compiling. Build-source receipt SHA25627606ca9dc8f8e5c5c7be8053b66a237d04fe06a076a7bbdde060807cf435029. The canonical build exited0 in61.471743 seconds, all211 steps complete. Full output/receipt: cuda-build-pq2-i64-async-stage.txt and .exit.json. Root collected the successful terminal output after inspecting the saved receipt; no build was relaunched.

Exclusively copied and hash-verified all eleven files into tools/llamacpp-cuda-pq2-i64-async-stage. CUDA SHA25688ad31a63206f895fddf8f9ef30117ded0e488662608c3d7fb61fe3aedbf86d7; native executable SHA256c964381859ebdcab8a5b069606aa4513a6321e41914eba199726068b9da341b7; test source SHA25641545ca51a7fa65699e4839f948b85fc2617c6e3ddcc81f0f7cc1adbf3846a80. Manifest cuda-pq2-i64-async-stage-binary-hashes.json. Offline compiled comparison and independent semantics review are running; native preflight is authorized only with zero GPU executions. Normal service remains accepted037.

Read-only discovery also identified a possible signed-FWHT-to-MMQ-Q8 fusion for the current down-projection shape. Existing graph consumer/alias matching and external-Q8 APIs appear usable, but actual eligibility is not proved by kernel adjacency; no new experiment or speed result is claimed. Two preparation read failures occurred: default cp1250 source decoding raised UnicodeDecodeError (corrected with UTF-8), and a scoped quantize.cuh search returned no match before the layout was found in mmq.cuh. Both exit1 results remain in tool output; no files were authorized in that read-only discovery. Subsequent reads exited0. No source edits, build or GPU work resulted from that investigation.


### 040 rejected at compiled register gate; no GPU execution

Both candidate cuobjdump commands exited0, but the static checker exited1. Raw resources show130 registers, exceeding the prospective <=128 limit. Stack, local and static shared allocation are zero, with no spill instructions; source launch remains256 threads and38,400 bytes dynamic shared. All125 retained kernel bodies/resources match accepted037. The new second stage contains18 unpredicated four-byte LDGSTS operations with commit/wait0, and the scalar second-copy loop is removed. A separate first-copy screen also fails:18 LDG.E.CONSTANT loads became LDG.E. These code-generation changes do not override the original register gate.

Independent review verified the exact snapshot hash, raw resources, extraction exit codes and checker failure. REG130 alone decisively rejects040; full semantic approval stopped because no GPU gate can proceed. Preserve the untouched failure in cuda-pq2-i64-async-stage-static-compiled1-{candidate-resources.txt,candidate-sass.txt,candidate-functions.json,screening.json,provenance.json,exit.json}. The caller wrapper exited0 while recording the checker's actual exit1; the child receipt and failed result, not the wrapper, determine this outcome.

The independently reviewed native runner's actual preflight separately exited0 and verified36 bindings, with zero native commands. No correctness or speed inference follows from that preparation. Artifacts: cuda-pq2-i64-async-stage-runtime-preflight1.{jsonl,stdout.txt,stderr.txt,exit.json}. One preceding console preparation read exited1: printing a UTF-8 skill through cp1250 raised UnicodeEncodeError at U+2192 before the runner-tail read. The failure is preserved in runtime-preflight1-preparation-failure.json; it was not a native-test failure.

Before reversal, verified all four root source files exactly match the isolated worktree, the preserved patch hash is7753ce0bba8b4f1dff488daf5bf23beef2e8b4f0d2c9417af1bf5224d101ed99 and the snapshot hash matches the reviewed DLL. Reverse-apply check passed; reversed the exact patch and verified a clean root tree. Worktree, patch, compiled snapshot and every artifact remain preserved. No040 native GPU test, service deployment or throughput run occurred. Normal service remains accepted037; retained win count stays four.


### 041 prospective ingestion hypothesis: signed FWHT directly into MMQ Q8 storage

Read-only accepted037 trace analysis identified1,152 signed-FWHT/quantizer pairs on K17408 and T508/512,64 per full prefill graph. Their recorded totals are262.39 ms transform and168.40 ms quantization over18 such graphs. These partial-profiler totals do not establish a speedup or prove graph eligibility. The proposed single-consumer FFN-down path removes one34 MiB F32 write and its subsequent read at T512, plus one quantizer launch per eligible pair. Transform arithmetic and ordinary MMQ remain; extra shared rearrangement, one CTA barrier and quantization registers are costs. This differs from026's matrix-kernel/FFN scheduling change.

Match exactly MUL(x,signs) -> RESHAPE[1024,17*T] -> HADAMARD MUL_MAT -> RESHAPE[17408,T] -> PQ2 MUL_MAT producing5120 rows. Initially require SM86, signed N1024, contiguous F32 activation[17408,T,1,1], T508/512 and plain PQ2 down projection. Use canonical five-node subgraph consumer/observable-output/view checks, explicit edge/type/shape/hint checks, canonical allocated-memory range checks and leaf-alias guards. Fail closed for sharing, overlapping buffers, unsupported shapes/types and disabled fusion. Source construction alone does not prove64 eligible paths: memoization, scheduling and allocated aliases require actual-service diagnosis.

Add default-inert DEBUG_CUDA_FWHT_Q8 tooling, using the same pure eligibility predicate as dispatch, to report eligible/rejected graph/token counts and reasons. Distinguish capture-time submission from replay; diagnostics cannot supply uninstrumented throughput evidence. Keep this real diagnostic tooling with any retained fix. Narrow production surface is ggml-cuda.cu, fwht.cu and fwht.cuh; retain ordinary quantize/MMQ implementations. Enumerate consumers before API changes.

Preserve signed FWHT butterflies and normalization exactly. Replace its final F32 global stores with the existing1024-float shared array, add a CTA barrier, then read contiguous float4 groups and reproduce the existing MMQ Q8 D4 reduction, scaling, roundf and layout exactly. Destination block is (chunk*8+local128block)*T+token,136 blocks per token. Use independent pool storage sized by ggml_cuda_mul_mat_q_q8_size(weight,activation_view), retain its safety tail, and keep its allocation alive across both launches on the same current stream. Pass the Q8 allocation base as external_q8, quantize_external=false; the F32 reshape supplies metadata, not the Q8 pointer.

Prospective compile gate: dedicated new specialization,256 threads, static shared exactly4096 bytes/dynamic0, <=40 registers, zero stack/local/spills. Accepted signedFWHT is20 registers/4096 bytes; <=40 reserves enough registers for six256-thread blocks within SM86's65,536-register/48-warp limits, without claiming achieved occupancy or speed. Primary limits: https://docs.nvidia.com/cuda/archive/12.9.1/ampere-tuning-guide/index.html#occupancy . All retained PQ2/MMQ/GDN and original FWHT instruction bodies/resources must remain identical. Independently prove ordered transform/quantizer arithmetic, reduction lanes, indexing, buffer bounds, synchronization and graph/pool lifetime; reject before GPU tests on a failed static gate.

Native correctness must compare every initialized Q8 payload byte and all four F32 D4 scales against the existing signedFWHT-plus-quantizer path, with guard sentinels and T508/512 cases covering random, zero/signed-zero, extrema and rounding boundaries. A versioned CUDA registry test hook invoked only by a dedicated backend-test selector is permitted; it must not add a public GGML operator or run automatically in the service. Deliberate byte/scale/index/sentinel corruption must fail validation. Also require whole-graph CPU-reference output cases, negative eligibility/consumer/alias cases, direct/capture/replay with changed inputs, existing27 Hadamard cases and101 PQ2/33 unsupported,44 fusion and8 FFN cases. Source preparation must enumerate exact identities and nonzero expected counts before any runtime execution.

Service gates: diagnostic proof of actual eligible replacements and retained fallbacks, strict growth/atomic tokens/probabilities versus accepted037, then fresh accepted037/candidate ABBA. Each process has separate conditioning reps1 and measurement reps5 at512/4096 with256 outputs. BOTH individual pairs and pooled results must improve ingestion >=2% at BOTH prompt sizes, with output regression <=2%; context188416,auto-four slots,KV formats,batch512,affinity and request semantics remain unchanged. First failed pair rejects without selective reruns or weakened thresholds. Accepted037 remains the baseline and served build.041 is now source-only; no compilation or GPU/service action is authorized before source review.


### 041 baseline compiled inventory prepared

The accepted037 baseline now has a complete177-function cache:125 previously verified PQ2/MMQ/GDN bodies plus all52 original FWHT variants. Read-only cache discovery found no existing FWHT bodies in10 function-cache JSONs or39 SASS files. A single targeted cuobjdump extraction of the missing52 bodies exited0 in9.77 seconds with empty stderr. Existing hash-verified resource output was reused; prior resource warnings remain in provenance.

FWHT inventory is24 register kernels,8 shared-memory kernels and20 block kernels, with26 F32/26 F16 and26 signed/26 unsigned variants. The served signed F32 N1024 block has20 registers,4096 static shared bytes,zero stack/local/spills and280 instruction slots. Twelve legacy FWHT variants already have nonzero stack; their acceptance check is exact baseline invariance, not the new kernel's zero-stack gate.

Cache cuda-fwht-mmq-q8-static-baseline-functions.json SHA256b89b1351750f2600ff48cfe71f5b664995da2070f99d7501f9ddd2e944bf2833; FWHT-only cache SHA25628af086dcd676c98cea9ced76258d6c2092b36ec4af27d72d38f32325793558c; raw FWHT SASS SHA256611ebb1f39902a8d355d84ac8e9b10c31de8485256447943c67aac06a4dc82f2. Full command/tool/binary/cache bindings and receipts: cuda-fwht-mmq-q8-static-baseline-{provenance,ready}.json. Independent cache review is pending. No041 target symbol has been guessed, and no candidate compilation, GPU, native or service operation has occurred.

Source implementation is in an isolated worktree. One minimal internal fwht-test.h POD/typedef contract was included in the authorized scope to share the versioned CUDA test-hook ABI without CUDA headers or a public GGML operator. Its correctness and graph/alias guards remain subject to independent source review. Separately confirmed040 preflight ran exactly once, with no native commands; the earlier encoding failure preceded execution rather than causing a preflight rerun.


### 041 baseline independently approved; source V1 under review and packaging corrected next

Independent baseline review verified the exact037 DLL and19 artifact hashes, all52 FWHT bodies/resources with no duplicates or omissions, and exact preservation of125 earlier cache entries in the combined177-function cache. Active N1024 signed F32 launch/resource claims are confirmed. Evidence: cuda-fwht-mmq-q8-static-baseline-review-v1.json. This approves baseline evidence only.

Source V1 preparation produced patch cuda-fwht-mmq-q8-source.patch, SHA256a281de66ff607e45f7c0ffc420c2dabf6d5ee5a25d31b0947a9808cc1741df36, based on513eea13a7c4feabed84f22dce5a3155269c418f. CPU proof run2 passes30 checks. It prepared18 native byte/sentinel/fault cases and13 whole-graph cases, with native execution still zero. Two preparation failures were preserved: initial Windows CRLF translation made the whitespace gate exit2 (corrected to LF), and proof-v1 used an indentation-sensitive comment marker and exited1 (corrected in V2). Full original patches/outputs/receipts remain in cuda-fwht-mmq-q8-draft-crlf-*, preparation-failure1.json and proof-run1*; corrected proof is proof-run2.json.

Root's review of the ready receipt found that its five source hashes include fwht-test.h, but its patch/stat/status contain only four tracked files and omit that new header. Reverse-checking in the same worktree did not prove a complete portable patch. It also found the native fallback list omitted the explicitly pre-registered T1 case. Neither issue is accepted. The author is preparing a preserved V2 patch with the exact header whitelist in the deny-default ignore file, complete fresh-base reconstruction proof and T1 coverage; counts and hashes must be updated. Independent source/alias/arithmetic review and static-helper preparation continue. No041 compile, native/GPU test or service deployment has occurred.


Correction to the preceding041 packaging finding: direct inspection of the exact a281de66ff607e45f7c0ffc420c2dabf6d5ee5a25d31b0947a9808cc1741df36 patch confirms FIVE diff entries, including a complete new-file fwht-test.h diff. The author had manually appended it after the four-file git diff recorded in the ready receipt. Root's claim that the exported patch omitted the header was wrong; it inferred patch contents from the incomplete stat/receipt instead of inspecting the patch. The verified defect is missing normal tracking/allow-rule: git check-ignore identifies .gitignore:675 and git ls-files has no header entry. V2 will fix that packaging inconsistency and add fresh-base reconstruction evidence. The separate missing T1 fallback finding remains valid. Original artifacts and this correction remain append-only.


### 041 V2 packaging and T1 fallback preparation passed

Preserved V2 patch cuda-fwht-mmq-q8-source-v2.patch has SHA256a4debed795bcb0813d5cd7cc769a280ccf905db723bff2f40f3c8a77dad39428. It adds the exact fwht-test.h allow-rule and tracking, includes all six changed files, and adds the missing T1 case. Applying it to a fresh temporary Git index initialized from513eea13a7c4feabed84f22dce5a3155269c418f reconstructed every changed file byte-for-byte, including the header and ignore file. Full receipt: cuda-fwht-mmq-q8-v2-packaging-commands.jsonl; ready-v2.json. Original V1 artifacts remain intact.

CPU proof-run3 passes30 checks; production implementation bytes remain unchanged from V1. Prepared native cases are18 byte tests and14 whole-graph tests (two eligible/twelve fallback), with20 compute calls and21 output comparisons in the normal whole-graph invocation. A separate disabled-fusion invocation expects14 fallbacks. No new preparation failures, native execution, build or GPU/service action occurred. Independent source review remains pending. Runtime-runner preparation will bind the exact final reviewed patch, compiled files and source receipt and must verify actual capture/replay evidence rather than infer it from four graph-compute calls.


### 041 production source review passed; V3 adds observed native evidence

Independent V2 specification/quality review found no production blocker. It reconstructed all six files independently, verified exact copied butterfly and D4 reduction/rounding expressions, and exhaustively checked all initialized Q8 words:2,487,168 at T508 and2,506,752 at T512. Consumer/output/view and external-leaf alias checks, transposed layout and same-stream pool ownership passed source review. Evidence: cuda-fwht-mmq-q8-review-source-v2.json SHA256660d0bcf4835c9296a2c605d512e32a2a1caa32e9b215c85b7ba3927324c89b8. This does not prove compiled behavior, zero-group conversion, native equality or replay.

Runtime preparation found an evidence gap: generic OK lines did not expose byte-test mismatch/ABI flags or whole-graph execution counts. V3 adds structured stderr records from actual hook results and completed computations/comparisons. Byte records include hook_ok, baseline_equal, equal, payload_bytes and actual ABI checks/rejections. Graph records preserve the initially observed predicate result before negative-control mutations and count synchronized GPU computations and successful output comparisons. Production and POD ABI bytes are unchanged. V3 patch SHA256ccac29a6145d4f95e3b3e48a4a141d38ca4647bc4033afeeb5af530262b9d4e9; source-ready-v3.json and v3-packaging-commands.jsonl prove fresh-base reconstruction of six files. CPU proof-run4 passes30 checks; test-source SHA256d6ad0f84efe78aba0111c73c6fcebc6995cce523889dbf5ba3405e0c42c73f99. Independent V3 delta review remains required before compilation.

Static-checker V2 preparation passed22 CPU fault tests, enforcing177 retained kernels and the new resource gate without claiming semantic approval; it is being rebound to V3. Runtime runner preparation passed29 CPU tests and preserves the earlier27-test result. Runner SHA2562a131d4b402cf5fd02152560db7ff7ec6a0c2a522932f658e2302116142a208c, ready.json and synthetic2.json. Seven core suites require226 passes plus33 exact expected unsupported identities:18 byte,14 graph,14 separately disabled graph,27 Hadamard,101 PQ2,44 fusion and8 FFN. All18 byte records must report exact payload sizes(9,948,672 for508;10,027,008 for512), four actual ABI checks/rejections, and the expected observed equality flags for positive/corrupted cases.

Two additional fresh-process replay cases are pre-registered now, before any runtime execution: selector FWHT_MMQ_Q8 with exact parameter patterns ^blk=1024,width=17408,n_tokens=508,type_x=f32,kind=0$ and the corresponding512 pattern. Each must contain exactly one passing case, zero unsupported, four actual graph modes direct/capture/replay/replay on one stable UID/key, two matching fused host submissions, changed inputs and four successful numerical comparisons. These are two additional passes, for228 total plus33 unsupported when all phases complete. Graph counters alone cannot establish these modes; use DEBUG_CUDA_GRAPH_STATS while leaving graphs enabled, and verify diagnostic visibility. No selective retry or weakened lifecycle gate.

Two read-preparation failures remain preserved in runtime-preparation-failure1.json and failure2.json: rg was unavailable and a subsequent Get-Content masked that shell failure; an assumed graph-stats.cuh path did not exist and the definition was then found in common.cuh. No native/GPU/build execution occurred. Future preflight must bind the final reviewed source/patch, actual build receipt and exact DLL/executable hashes; independent runtime/checker reviews are pending.


### 041 V3 review and build passed; replay-validator defect corrected before use

Independent V3 specification/quality review passed for compilation. It verified all six reconstructed files and unchanged production/POD ABI bytes, observed test counters and failure propagation. Native INFO records require no extra verbosity: test-backend-ops uses the default GGML callback that writes stderr. Static-checker V3 independently passed22 CPU tests and review, requiring exact whole-DLL symbol multiplicities plus one target,177 retained kernels, the original resources/barrier gates and independent compiled semantics. Versioned helper SHA25623bca1f0dbc5be0f2dd7a9ea019338215801c1c4aef441f6a369c750077a4719; V2 remains preserved.

Independent runtime review found a real false-approval defect before any native execution: both508/512 synthetic cases accepted two fused submissions from foreign graph UID999/index77 while the four execute records belonged to UID1/key000000001. The original29 tests passed despite this gap. Preserved reproduction: cuda-fwht-mmq-q8-runtime-review-foreign-uid.json. Runtime V2 now requires submission UID to match the unique execution UID and a consistent valid node index, alongside existing tokens/capture/mode/key checks. Helper SHA256f2ac74485c2d1bf9430436133fd872cf0620b8f52a3db04bd73d682162b8aa04. All32 author tests passed; independent review rejected both saved reproductions plus ten additional correlation faults, accepted both valid fixtures, and verified every function except validate_evidence remains AST-identical. Review-v2-verdict.json SHA25688713782157f9e78c6aef3e2d0c0c767e4932b61f74a4b0fb33c8b9d8edda9d0. No correctness/performance evidence was accepted through the defective validator.

Applied V3 to clean basece379b6ad17d0bdd36970f7b9446985f859c6069 and verified all six bytes/hash bindings before compilation. Build-source receipt SHA256ed8744dcd48296c2586a1a4dd529876a2c1064fe814c492dfe6621cb7c4a0cfe. First monitor launch58883 failed before Python/compiler execution because the quoted absolute executable was parsed as a PowerShell expression; no build artifacts were created. Full error is preserved in cuda-build-fwht-mmq-q8-launch-failure1.json. Corrected launch51797 uses the verified Python command and a preserved progress wrapper around the canonical build script.

The actual canonical build ran once, exited0 in45.002856 seconds, and its terminal result was collected. Full combined output and real receipt: cuda-build-fwht-mmq-q8.txt/.exit.json. Existing compiler warnings and a nonfatal UI-asset download warning are preserved rather than called a warning-free build. All eleven files were exclusively copied and hash-verified in tools/llamacpp-cuda-fwht-mmq-q8. CUDA SHA25632e0e694e4bb2d9a982905210b9445e24ed16041e0c429b6851c761bfd59bc9e; test executable SHA256abb891c23aa9e7af2a84cd02e552b34410252a86407599515182e954c2f81d6c; test source SHA256d6ad0f84efe78aba0111c73c6fcebc6995cce523889dbf5ba3405e0c42c73f99. Snapshot manifest cuda-fwht-mmq-q8-binary-hashes.json.

Actual offline static comparison and native preflight are running. Native preflight must launch zero GPU commands; compiled semantics approval remains required before GPU execution. Normal service remains accepted037 and retained wins remain four.


### 041 compiled static and semantic gates passed; native preflight passed

Actual static extraction/comparison exited0 in37.282 seconds, with both cuobjdump commands successful and empty stderr. Target fwht_mmq_q8_17408 has22 registers,4096 static shared bytes,zero dynamic shared/stack/local/spills,344 instruction slots and seven barriers. Host launch is256 threads. All177 retained kernels match accepted037 exactly; whole-DLL symbol inventory differs by this one addition only. Raw SASS SHA2567d138a2e7f072824b9d04a95b21d47ec6cab9646d73c5f7d2d3c31d8584b95e0. Evidence: cuda-fwht-mmq-q8-static-compiled1-*.

Independent compiled semantic review passed. Actual256-lane instruction operand graphs match1024 ordered FWHT outputs,1024 quantized bytes and32 scales per1024 block against accepted FWHT and the exact D4 quantizer. Nine operand/reduction/barrier/rounding/packing mutations were rejected. SASS-derived address enumeration covers every initialized Q8 word for both508/512 tokens,136 blocks per token, with no padding writes. Only the missing exact accepted037 D4 reference was extracted once; full provenance remains. A proof-harness EXIT parser failure is preserved in semantic-proof-failure1.json; corrected proof is semantic-proof-run2.json. Review receipt cuda-fwht-mmq-q8-semantic-address-resource-v1.json SHA25601533af4649159be4202b54238822c5847d32a28a5fa1d2da6d2fd7f00fa4ffb. This is not native zero/rounding, graph replay, service correctness or throughput evidence.

Actual runtime V2 preflight exited0 in0.312 seconds with native_commands0. It binds actual041 DLL/test executable, reviewed test source and build receipt/base. Evidence: cuda-fwht-mmq-q8-runtime-preflight1.{jsonl,stdout.txt,stderr.txt,exit.json}. Core226-pass/33-unsupported native phase and two separately preregistered fresh replay cases may now proceed after idle-service maintenance. Service remains accepted037 before maintenance; four wins retained.

Service validator preparation passed50 CPU tests in22.826 seconds, including an exact037-derived synthetic full trace and diagnostic/correlation/inventory/provenance faults. Helper cuda-nsys041-analysis.py SHA256e7ad86a779318786ae92bb9145d6065ce7ceba235a7d270299f05c8975f4d62f; ready receipt SHA2560e7e6144d6f4601b597c4a14ca99151459fd07bebd2bc4587e09bb1fae279c12. Independent review remains pending; no actual041 service eligibility or speed result yet.


### 041 native correctness passed and independently reviewed; service-validator V1 rejected

Idle normal accepted037 service was stopped through its inspected Session0 chain6120/10924/23820; inspection041.json and stopped041.json preserve actual identities and maintenance receipt. Original supervisor bytes remain verified for restoration. Exact reviewed runtime V2 then ran once: core226 passes plus33 expected unsupported in41.859 seconds, and two separately preregistered fresh replay cases in5.938 seconds. All nine native commands exited0. Artifacts cuda-fwht-mmq-q8-runtime-{run1,replay1} retain command/PID/binary/source provenance, raw stdout/stderr and true exit receipts; runtime-execution1-verdict.json records the runner result. No source/helper edits or reruns occurred.

Independent review rehashed all bindings and18 raw streams, checked nine distinct native PIDs/commands with no timeouts, exact case identity/status multisets and footers. Both508/512 fresh processes show direct/capture/replay/replay with stable UID/key, two index0 fused submissions and four changed-input GPU/CPU output comparisons. All18 byte cases have baseline_equal1: ten exact positives and eight deliberately detected corruptions, each with four ABI checks/rejections and exact9948672/10027008-byte payload. Enabled graph tests show20 computes/21 comparisons; disabled show14/15 with zero eligible. Native review evidence SHA256a2b1e5f897d9147e26606a0270ee954213aac74bfd1cdf299a9ed7a870fd6f89; receipt review SHA2568e44aeeecf47d930725f8264a3787c2e2162357a7370b5db25525fd3a08ecd1a. Exact candidate is cleared for diagnostic service testing, not accepted for speed.

Independent CPU probes rejected service-validator V1 despite its50 passing tests: it falsely accepted a target on device1 instead of device0, a kernel correlation reassigned to cudaMemcpyAsync after deleting its launch API, and duplicate numeric graph node indices spelled0/00. Evidence cuda-nsys041-review-fault-probes-v1.json preserves all three reproductions. A nonfinite live-receipt timestamp was also accepted, outside the helper's former chronology checks and requiring explicit correction/provenance review. No actual service result was accepted through V1. Versioned fixes and regression tests are in preparation; all original files remain.

The candidate's eleven files were deployed to the stable path with the canonical stopped-service deploy script, exit0. Receipt bonsai-deploy-20260923T073818197-e24e944f60c841588e7c533c1e026fbf.json preserves the accepted037 rollback. No041 service process has been launched yet; no fifth win is claimed.


### 041 service validator V2 approved; diagnostic capture started

Versioned trace validator cuda-nsys041-analysis-v2.py SHA256ee58d5c5f44b95a5673fea2431051635ca6303db5d7f53cbfaf8a75b0c0860c2 closes wrong-device, wrong-CUDA-API ownership, numeric-node-alias and nonfinite-receipt defects. Independent55/55 tests passed in30.594 seconds, exit0; evidence cuda-nsys041-review-synthetic-v2.json SHA256a5525f2abb39bdf7d9c43283855c70916b94e555852053d5948e8859be6d1532. V1 remains rejected and preserved. Partial-profiler coverage and externally bound same-process diagnostic provenance remain explicit limitations.

A preserved wrapper composes existing canonical runner/bench commands because no persisted orchestration receipt helper existed. cuda-nsys041-orchestrate-v1.py SHA2562792f6f527dda08e25051129984bc28bc56f11063d314f2c72a48d7957eb0eda passed20 CPU checks, including a real small command exit7 with preserved output/receipt. It pins scripts/manifests/validator, clears inherited debug and disable-graphs/disable-fusion variables, enables FWHT/graph diagnostics and lazy reserve, preserves all streams/actual exits, binds server ancestry to its launch PID, stops profiling in finally and exports only on success. It does not change capacity or accept performance.

Diagnostic service PID24276 is healthy at the stable executable path. Contemporaneous cuda-nsys041-process-provenance.json verifies all eleven candidate files, nine loaded runtime paths/backing hashes, and four idle slots each188416 context. Actual process arguments preserve batch512/ubatch512/model/KV settings, with verbosity4 and a new graph-stats log. The canonical512/4096,32-output,reps1 capture is now running under cuda-nsys041-capture1*. No actual dispatch or throughput result is yet claimed.

### 042 pre-registration: aligned private Q8 activation storage for output

Prepare a separate compile-first experiment while041 service gates run. Restrict to ordinary PQ2 down projection on SM86, K17408/M5120/T1, contiguous F32 activation, singleton higher dimensions and no IDs/bias/gate/scales. Keep existing128-thread32x4 launch, four rows per CTA, lane/chunk progression, signed weight codec, DP4A chains, FP32 arithmetic and reduction. Replace only the private activation scratch layout:17408 payload bytes followed by544 unchanged half2 scale/sum values(2176 bytes), total19584 bytes. Original PAD(17408,512) is17408 and544*36 is19584; no extra buffer/pass/launch or public quantization-format change. Payload at base+32*chunk permits two aligned read-only128-bit loads in place of eight scalar32-bit loads. Scale/sum lives at base+17408+4*chunk.

Accepted037 cached ordinary SASS has42 registers and eight scalar payload loads per chunk; paired loop28 total global loads could become16, and peeled14 could become8, if actual compiler output realizes the proposal. The recorded final30 service replays contain64 matching down-projection launches per token,4.680 ms/replay,17.14% of recorded summed kernel time. Shape attribution includes quantizer geometry/sequencing and is not pointer-level proof; partial-collection caveats remain. Traffic bytes are unchanged, load issue pressure is unproven, and vector grouping/address work/cache behavior may cancel any benefit. This does not repeat earlier LUT, signedness, unroll, index or pruning candidates. No tok/s prediction.

Production scope should be private producer/dot/ordinary-kernel implementation in mmvq.cu plus focused tests and, if needed, minimal versioned internal registry hook contract/ignore allow-rule. Enumerate all producer/consumer/lifetime sites before editing. Retain existing generic kernels exactly. CPU proof must show aligned address coverage, unchanged payload/half2 semantics, bounded ownership and pool lifetime. Prospective compile gate: two aligned128-bit payload loads per logical chunk, original cache policy, exact ordered arithmetic and paired/peeled expansion; new ordinary<=42 registers and producer<=16 registers, zero stack/local/spills/shared. The producer ceiling is accepted037's actual _Z13quantize_q8_1PKfPvxxxxxj5uint3 REG16 resource record, not an occupancy estimate. Reject scalarization, changed arithmetic, resource failure or address instructions erasing the proposed saving, with no unroll/register-cap adjustment. All original bodies/resources must remain exact. Missing original producer SASS may be extracted once later with full provenance; cached resources and other bodies must be reused.

Native gates: explicit eligible shape with four changed-input direct/capture/replay/replay computations and CPU output comparisons; width2/K6144/M5121/noncontiguous/bias/fused fallbacks. Compare every normalized payload byte and both half2 components against original native Q8 quantization for random, zero, signed-zero, extrema and rounding boundaries, guarded sentinels and deliberate payload/scale/plane-offset corruptions. Enumerate exact case identities/counts before runtime; retain101 PQ2 passes/33 unsupported,44 fusion and8 FFN. Actual-service diagnostics must prove64 replacements/token, preserved1935-kernel decode replay count and all other inventories, then strict growth/atomic correctness. Use the latest accepted baseline consistently (041 if retained); fresh ABBA with conditioning1 and measurement5 at512/4096/256 outputs. BOTH pairs and pooled output must improve>=2%, ingestion may regress<=2%; requests/tokens/content/probabilities/capacity remain exact. First failed pair rejects without selective reruns or relaxed gates. Source preparation only until independent review; no042 compilation/GPU action yet.

Read-only discovery notes and command failures are preserved in logs/implementer-next-pq2-output.md: an overbroad ledger section read produced truncated output(exit0), and a docs-hub captured UTF8 print failed under cp1250(exit1), corrected by UTF8 read(exit0). Neither performed source edits, extraction or GPU work.

Preparation failure: the command that appended the042 pre-registration completed its append, then an accidental trailing PowerShell token `a` caused the outer shell to exit1. No source/build/service operation was attempted by that token. The original append is retained; it is not rerun. Evidence is preserved in cuda-042-ledger-preparation-failure1.json; a separate whitespace check follows.


042 source-scope refinement: the private mmvq implementation lacks graph UID/node context and registry export sites. Minimal ggml-cuda.cu registry/graph-diagnostic wiring and mmvq.cuh internal declarations are authorized in the isolated042 worktree, alongside its POD hook header/allow-rule. Prefer a shared pure eligibility predicate at the existing graph context, distinguish observed submission from eligibility/replay, and keep diagnostics default-inert. common.cuh state changes remain excluded without a concrete further design review. No042 compile/GPU work is authorized by this refinement.

041 actual capture completed once: start0/0.397s, canonical bench0/33.828s, stop0/3.347s, export0/1.506s. The tracked wrapper session68412 terminal exit0 was collected. Full artifacts remain under cuda-nsys041-capture1*. Canonical --log-file receives INFO diagnostics; launch stdout/stderr are empty. Before further requests, an exclusive969685-byte diagnostic snapshot was preserved with SHA2563ce92a9b3c2260051270b73e38392523808da6f23ec865a961fa98cfaffe6731 and launch/PID ancestry bindings. It contains148 execute records including startup prefix,8206 candidate records and1152 host submissions; these preliminary counts are not yet actual-trace acceptance. Approved V2 trace analysis is running once; strict growth and atomic correctness checks follow on the same diagnostic service.


### 041 actual service dispatch and strict numerical gates passed

Approved V2 actual trace validation ran once, exited0 in16.543 seconds: cuda-nsys041-service-analysis-run1.json SHA256c086af70154c93f69a3d6838efc27b842cd3d24327cfb6d8c02dcd2399cd90df. It found exactly1152 replacements,64 per fourT508 and fourteenT512 evaluations, zero on fourT4 and124T1. Canonical146 executions follow two startup records:26direct/4capture/116replay. Four requests and128 output tokens/content match accepted037 exactly. All retained per-evaluation kernel/resource and complete copy inventories match; COLS2 counts remain0/336/0/336. Final30 decode replays preserve1935 kernels and48 indexedGDN with no gathers/state-D2D. No CUDA API errors, unmatched activities or unexplained launches;7740 capture-only submissions are explicitly accounted.

Profiler warnings remain exactly accepted037:5571 GetGraphId and5571 GetGraphNodeId errors plus one generic incomplete-collection warning. No warning waiver or complete-collection claim is made. Process/log ancestry and live backing-file hashes are preserved; pointer-level graph-node attribution is not asserted. Independent actual-evidence review is pending separately.

Canonical strict growth comparison passed once against accepted037 in42.485 seconds; atomic four-slot comparison passed in2.428 seconds, both exit0. Four growth responses cover512/4096/16384/512, and four atomic responses cover distinct slots0..3. Full requests/responses/probability comparisons and command streams/receipts remain in cuda-fwht-mmq-q8-{growth,atomic}-correctness.*. Diagnostic server24276 was then inspected idle and stopped deliberately. Nsight launch child exited4294967295 and its wrapper monitor35647 exited1 as expected for that stop; launch failure is not rewritten as successful termination. All capture/benchmark/stop/export checks completed before this termination.

Fresh accepted037 controlA is now healthy, PID19472 under watcher89749, after canonical deployment receipt bonsai-deploy-20260923T075210643-92a973aa5a9e46c5aca4793f145805c5.json. All DEBUG variables and graph/fusion disable flags were cleared before launch, with lazy reserve enabled. Original model/capacity/batch/KV arguments are preserved. Performance execution is delegated and remains gated on independent actual-evidence approval; no untraced041 speed result yet.


Independent041 actual-service review passed specification, quality and evidence, clearing uninstrumented timing. Receipt cuda-fwht-mmq-q8-service-review-verdict-v1.json SHA25630029ca9a8b3fc677b9f81f32bb1d0f4d8cc3b8cbce7548a3f0688c0b2da5024. All eight strict growth/atomic responses,128 tokens and768 sampled/top-logprob values equal accepted037 exactly; maximum absolute logprob difference0.0. Independent raw-SQLite review verified all146 evaluation inventories,1152 substitutions, device0 and correct CUDA API ownership, retained copies/resources, diagnostics and original warning inventory. Actual PID24276/module/deployment/log bindings were rehashed. No throughput conclusion follows from this correctness approval.

Performance controlA19472 has now passed its contemporaneous provenance/capacity checks and started separate conditioning then measurement. Candidate transition is authorized after controlA completion; all original041 thresholds and first-failure rejection remain in force. Source-only042 preparation continues separately without compilation/extraction/GPU activity during timing.


041 first uninstrumented controlA completed without reruns: conditioning exit0/58.262s, measurement exit0/175.075s. Reported medians are512 ingestion420.3906/output36.7906 tok/s and4096 ingestion533.1483/output34.5727 tok/s. These are baseline observations, not candidate gains. All original trials remain under cuda-service-fwht-mmq-q8-control-a[-conditioning].jsonl with process19472 provenance and full command streams/receipts. Two attempted combined read-only inspection commands were blocked automatically before subprocess creation; direct literal-PID CIM inspection succeeded, and exact notices are being preserved as preparation failures. No benchmark rerun or threshold change followed. CandidateA is next under the unchanged first-pair gate.


### 041 candidate deployment preparation failure: inherited PowerShell module path

After controlA completed and its inspected idle server19472 was stopped, the first candidate deployment command exited1 before swapping files. Canonical deploy.ps1 line101 could not resolve Get-FileHash. Full command,stdout/stderr and true exit are preserved in cuda-service-fwht-mmq-q8-candidate-a-deploy.*. Independent stable-file check confirms all eleven accepted037 files remained unchanged; no candidate benchmark ran. A recovery accepted037 server was launched before root's hold message arrived; no benchmark/request was made on that recovery process. It will be inspected/stopped separately before corrected deployment. Original controlA measurements are retained without rerun.

Root reproduced the exact failure with a read-only WindowsPowerShell5 child launched through Python using inherited PowerShell7 PSModulePath: Get-Command Get-FileHash exits1. The same command with ONLY PSModulePath omitted from the child environment exits0 with empty stderr, WindowsPowerShell5.1.26100.9278 native module paths and Microsoft.PowerShell.Utility providing Get-FileHash. Full argv/environment-path-only outputs and exits: cuda-fwht-mmq-q8-windowsps-module-inherited1.json and -native-defaults1.json. No global configuration/module edit was made. Corrected deployment attempt2 and subsequent WindowsPowerShell launches may use native child defaults; this repairs launch environment before the candidate's first timing, not a selective timing rerun or a performance rejection. Original failed staging/artifacts remain preserved.


### 041 rejected on its first service performance pair

Corrected deployment attempt2 succeeded in1.230 seconds using the same canonical WindowsPowerShell deployment script with only child PSModulePath removed. Deployment receipt bonsai-deploy-20260923T080720507-d69df0ea573c403c8b571a65f5fb2d71.json. Recovery accepted037 process3080 was separately inspected idle and stopped, with watcher89345 terminalexit1 and child4294967295 preserved as intentional termination. CandidateA then started fresh as PID24116,parent9728,watcher29994, with nine loaded module paths/eleven runtime hashes and four188416-context idle slots verified; all diagnostics/disable flags were cleared and lazy reserve enabled. Preparation notices and the empty failed staging directory are preserved in cuda-service-fwht-mmq-q8-preparation-failures.json and -failed-staging-inventory.json. The deployment/recovery gap is retained in provenance; no benchmark sample was discarded or repeated.

Both original candidate conditioning and measurement commands exited0. Canonical comparison-a exited1 with correctness passed and the original performance thresholds failed. For512 tokens, ingestion420.39058555 ->410.01493511 tok/s (-2.46809771%); output36.79058553 ->36.79812448 (+0.02049150%). For4096 tokens, ingestion533.14832809 ->541.87456054 (+1.63673634%); output34.57272719 ->34.61140822 (+0.11188307%). Neither prompt size meets the pre-registered>=2% ingestion gain. Output passes its regression threshold but supplies no separate win. All exact request/output checks passed.

041 is rejected. No reverse/B/pool run, selective rerun or threshold relaxation will follow. Comparator artifact cuda-service-fwht-mmq-q8-comparison-a.json binds baseline SHA25665603e1404bd396e601c40ef453ce3c460956888984deec46be5d42e1910e9cd and candidate SHA256adaba5a75dbb594888106a4bc87ce3cc88e32e27c3504cf5d28070af902990b2. Independent performance evidence review is pending; the failed gate is already decisive for rejection.

Root verified all six source hashes against the reviewed V3 build receipt, portable patch and preserved041 worktree, then reverse-checked and reversed exactly that patch. Verification receipt cuda-fwht-mmq-q8-rejection-source-verification.json; git status and whitespace check are clean afterward. Rejected source worktree, patch, binary snapshot, build output, static/native/service evidence and failure artifacts remain intact. Service owner is restoring accepted037; four retained wins remain.

### 042 source preparation ready for independent review

Source-only candidate042 has six files,+484/-2 at base6f2dbabd581a2d790106672b31ba086f828457c3. Portable patch cuda-pq2-aligned-q8-source.patch SHA256add4491ea3a2968789d845bce5fb8c7b05641852c002d8dbdfdc0aa37c85e5c6; exact per-file hashes, caller enumeration, reconstructed Git tree and command receipts are in source-ready.json/enumeration.json/packaging-receipts.json. Fresh-base index reconstruction reproduces every byte, including the new POD header and explicit deny-default allow-rule. CPU proof-run3 passes67 checks covering19584-byte ownership,1088 aligned vector reads,544 scale entries, order and corruption rejection. The graph shortcut retains original MMVQ callers, generic helpers and common.cuh, and now explicitly preserves compute_forward's launch-error check.

Prepared native identities are nine byte cases, eight enabled graph cases, eight separately disabled graph cases and153 retained cases:178 expected passes plus33 unsupported. Actual replay must be proved by correlated runtime records, not inferred from four calls. No compilation/native/GPU/service action has run for042. Self-review corrected an uncompiled draft's nonexistent hint getter to existing op_params index1; draft-correction1.json and earlier proof remain. Final read-only display quoting failed exit1, preserved in final-display-failure1.json; corrected display exited0. Independent source review is running. Baseline remains accepted037 after041 rejection; one missing original-Q8 producer SASS extraction and offline checker preparation may now proceed, reusing all cached evidence.


Final041 restoration completed: root separately inspected/stopped temporary accepted037 PID25096, and its owner collected watcher97200 exit1/child4294967295 with complete streams. The exact original4250-byte supervisor was restored with SHA25654af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be; the041 maintenance prefix remains preserved separately. Normal scheduled service is healthy as PID23196,parent23616, correct bonsai-2-27b model, four idle188416-context slots and all eleven accepted037 file hashes verified. Receipt cuda-fwht-mmq-q8-restored-service.json states the Session0 module-inspection limitation explicitly. No benchmark runs on this final restoration. Four wins retained.

042 independent source specification/quality review passed for compilation only. It verified all six reconstructed blobs, exact guarded graph/stream/pool/error paths, original caller/generic-helper preservation, aligned bounds for544 chunks and exhaustive65536 packed-weight words/524288 coefficients. Evidence cuda-pq2-aligned-q8-review-source-proof-v2.json SHA2560c9ad32881efff1274be6c5cb6b0a84f2ad05fdd893c57748ebb03da3d8795b9. An initial reviewer simulator incorrectly applied a PTX sign nibble to the CUDA intrinsic; that failed proof was preserved and corrected against the documented three-bit intrinsic selectors. No candidate source change was needed. Resource/vectorization/cache-policy/arithmetic/invariance/native/service gates remain untested. Compilation may proceed while accepted037 serves normally; no042 GPU execution before static/semantic approval.


042 fresh-replay subprotocol is preregistered before native execution: in addition to the178 core passes/33 unsupported, run one separate fresh process selecting only the eligible17408x5120/T1 kind0 graph. Require actual direct/capture/replay/replay for one UID/key, matching unique numeric node index and token count, exactly two host submissions and four successful changed-input comparisons. Total expected native coverage becomes179 passes/33 unsupported across core plus this separate proof. A helper must reject missing/cross-UID/cross-index/duplicate evidence and preserve all bytes/flags; it cannot treat four compute calls alone as replay proof. This adds an explicit standalone witness without changing production source or performance gates.


Independent041 performance rejection review is complete: cuda-service-fwht-mmq-q8-review-pair-a-v1.json SHA256fbd0b713efdb4f2cdaa2c7e439683f07475f756798503cc32ca18242d37703a7 and review-provenance-v2.json SHA25653d922f1a80174ca33330ac49374c3029426c4f7520094cbb1df337f36efbed0. It recomputed all five measured samples per prompt and both failed original ingestion gates, verified all32 responses/8192 emitted tokens(16 paired responses/4096 paired positions), exact settings/requests/content, four benchmark exits0 and expected comparator1. Both PID/module/eleven-file deployment bindings and deployment-before-process-before-provenance-before-conditioning chronology match. The preserved preparation gap preceded candidate conditioning and supplies no reason to rerun or relax rejection.

### 042 first compilation failed on host buffer API use

Applied the exact reviewed V1 patch to clean accepted037 source at actual build base220b5983b76425f04012676c54e19ef2f6eec66c and verified all six source hashes. Build-source receipt SHA25612133002554336974778a358b150c92ef178f1560421428dd1e61388672765af. The preserved canonical progress wrapper SHA25666c661d48150a2b16dc0053bf6d7edde1fd33874304a54d41a3951829687e0fe uses native WindowsPowerShell child module defaults, the same build script/flags and exclusive output. Duplicate artifact/compiler checks passed before launch.

Actual first042 build monitor8535 exited1 in12.395658 seconds, terminal collected. Full combined output and real command receipt remain in cuda-build-pq2-aligned-q8.txt/.exit.json. CUDA compilation rejects mmvq.cu:1093 because direct t->buffer->buft access requires a complete ggml_backend_buffer type unavailable in that translation unit. This escaped V1 source review. No candidate snapshot was created and no042 GPU/native/service execution occurred; accepted037 normal service remains unchanged. Author is preparing a minimal existing-accessor correction in versioned V2, with full source reconstruction and independent delta review before an exclusive second build. Original source/patch/build failures remain intact; no resource/arithmetic/performance threshold changes are authorized.

Separately, the one missing accepted037 original-Q8 producer body was extracted once, exit0/0.269s and empty stderr. It has104 instruction slots and16 registers with zero stack/local/shared. The new178-function baseline cache preserves all prior177 unchanged and adds this producer; SHA2560a72e18edea65e498f63cd6e5388b2607e1d961c3c264057b44f88cc7b7f8bbb. Provenance is cuda-pq2-aligned-q8-static-baseline1-provenance.json. Offline checker and future native-evidence tooling remain in preparation.


042 source V2 is prepared: full patch SHA256f9ebdd5dacf17d0c9369340aeaddd500cca7d9e6cfb87df0014acfb4087199cf. Its sole delta replaces direct buffer->buft access with ggml_backend_buffer_get_type(buffer); the accessor returns the same pointer and the existing null short-circuit remains. Other five sources are unchanged. Author re-ran67 CPU checks and fresh-base reconstruction for all six files; source-ready-v2.json, source-v2-delta.patch and v2-correction-proof.json preserve the exact change. Independent delta approval is pending before the second compile.

Offline static checker V3 is also prepared and pinned to source V2: cuda-pq2-aligned-q8-static-check-v3.py SHA256d9604232f9b393b8de1f51018fdb2aea468e496cb8939a1115b444c7e16e3bf2,51 CPU tests passed. Its actual prepare step exits0 with zero additional extraction commands; it reuses the178-function baseline. A numeric pass cannot set static_gate_passed, because dynamic vector-load mapping, ordered arithmetic and address-cost proof require independent compiled review. Full future command/bindings are in static-ready-v3.json. Independent helper/baseline review is pending; no candidate binary extraction/native test has occurred.


### 042 accessor correction independently approved; second build passed

Independent V2 delta review passed and verified the accessor definition, preceding null rejection and unchanged other five files/kernel/test/dispatch source. Evidence cuda-pq2-aligned-q8-review-accessor-v2.json SHA256ce791cbfa07d8c1d069b0a37186006553530f1c6c68aea672a435753b8f06206. Root verified V1 source hashes, applied only the reviewed one-line delta, and verified all six V2 hashes. Actual second-build base isd484b428144655a8cb1f587604fce7c79d37ea29; build2-source.json SHA256dda40e8dd34bae9b1bfa907fb79f145edf127fc3851240848471253c51b4a5b7. The distinct build2 progress wrapper SHA2568eb67647c9ca5b29511ccc78eb13d1f19b9c439489d0de2d30f570c6c7205a09 preserves the original failed wrapper/output and uses the same canonical build command/native child environment.

Second canonical build monitor92391 exited0 in59.124019 seconds; final result collected. Full output and real command receipt are cuda-build-pq2-aligned-q8-build2.txt/.exit.json. Existing compiler warnings and the UI-asset download warning are retained; this was not a warning-free build. All eleven runtime files were exclusively copied/hash-verified in tools/llamacpp-cuda-pq2-aligned-q8. CUDA SHA2562903dc8c57be6b0a35b17c7271ea6436a8b181075a38b366f07151487e349db2; test executable SHA25670c552b0c5d0032129071feab36bf52a6bf3b0332a668dc747d9a068cd86a9dd; test source SHA256e735c0d8436b3d811a692391c492ac5d0f8ae73afd8c2a020b3f2212e6db35a8. Snapshot manifest cuda-pq2-aligned-q8-binary-hashes.json. No042 native/GPU/service execution has occurred; normal accepted037 remains PID23196. Actual compiled comparison awaits independent helper/baseline approval, and numeric success still requires compiled semantics review.


### 042 rejected by compiled register gate before GPU execution

Independent static-helper/baseline review passed51 CPU tests and rehashed13 provenance bindings. All177 reused bodies remained exact; the added original producer reparsed to104 slots/16 registers. Review cuda-pq2-aligned-q8-static-review-baseline-v3.json SHA256f112859851891e1749adee58a8f5c2a4111e1ad4138e3e3188e4776f5caef907 authorized actual comparison, not semantic/performance acceptance.

Actual comparison ran once under session44580 and exited1 in33.976434 seconds; terminal collected. Both cuobjdump commands exited0 with empty stderr. Raw resources,SASS,functions,screening and full provenance remain under cuda-pq2-aligned-q8-static-compiled1-*, with separate wrapper streams/exit receipt. Candidate consumer uses48 registers, exceeding the pre-registered42 ceiling. It has296 instruction slots, six LDG.E.128.CONSTANT instructions and24 total global loads; these vector/cache/load-count screens passed. Producer uses14 registers/64 slots, under its16 ceiling. Both have zero stack/local/shared/spills; all178 retained bodies/resources match accepted037. Raw SASS SHA2567d7ac562adb59508098bf36fc769f2912ede5b5e08e1e6829b5f4626059e4e4d; raw resource SHA2564aa010dcb85045031da3aa4a351cb6257fcbf224630094f6caa61cf3d4f187df.

042 is rejected on its original static gate. This is not a measured throughput regression or a claim that48 registers changes occupancy. No ceiling relaxation, native/GPU/service test or selective rerun follows. Full arithmetic/dynamic-address proof stopped behind the failed gate; independent raw failure confirmation is pending. All six current source bytes and patch were checked against preserved V2/worktree, reverse-check passed, and root reversed only that exact patch. Root code returned clean to accepted037. Verification receipt cuda-pq2-aligned-q8-rejection-source-verification.json; rejected source worktree, both patches/builds, snapshot and all outputs remain preserved. Normal accepted037 service was never replaced for042.

Unused042 native-tool preparation is also retained. Runtime runner SHA256dcf1747486743bf6ccb41d6d5adb79f648834778d278c915b1b9b5b81c07bbfb; ready SHA2564dbceea7fafb961bc8648e96a0c06edd787495a44a2706663942183f107a6289. Author29 CPU tests passed after a test-only expected-file-inventory count39/40 assertion was corrected in selftest-v2; original test/synthetic1 failure and synthetic2 success remain, runner unchanged. Independent29 tests plus eight additional UID/key/token/mode/duplicate/index-alias faults passed. Review cuda-pq2-aligned-q8-runtime-review-v1.json SHA2569c177879c39972e3cbc667495fd039224f7b18d4ea8341b87db2ac09c70b3a1a approved preflight only; no actual preflight/native command ran. Prepared179-pass/33-unsupported gates remain unused.

User-facing current-speed report uses the latest accepted037 service measurements from041 controlA:512 ingestion420.39058555/output36.79058553 tok/s;4096 ingestion533.14832809/output34.57272719. Earlier accepted037 ABBA pooled4K ingestion536.198446 is a separate retained measurement, not a new build. Four wins remain. Retained changes stack, but percentages across different matched rounds, including lazy-reserve recovery of a slowdown, are not an exact end-to-end fork comparison.

Independent042 rejection confirmation: cuda-pq2-aligned-q8-static-review-rejection-v1.json SHA25637afc5bc31bc9c5ca583469b53e080a27b5f62c93d4c5e3fa670800e23f8db97 rehashed16 provenance bindings and the actual DLL/raw dumps, confirmed48>42 registers,178 retained kernels exact, and exactly two added symbols with none missing. Both extractions exited0, checker1 remains preserved; no semantic/native/service approval follows.


### 042 postmortem: register allocation does not establish an occupancy regression

Read-only author and independent reviewer inspected actual042 SASS slots107-159: separate scale-plane base R2:R3, scale indices R37/R39 and payload index R45 remain live. Both baseline and042 preload sixteen payload words;042 additionally requires contiguous register quartets for128-bit loads. These are plausible pressure sources, not a proof attributing each of the six extra allocated registers.

Both reviewers checked local CUDA12.9.1 cuda_occupancy.h lines671-695,698-718 and1486-1541. Its256-register warp allocation rounds both42 and48 registers to1536 registers/warp and6144 per128-thread CTA. With SM86 register capacity and four subpartitions, both have the same calculated register limit of10 CTAs/40 warps. This is an analytical allocation bound, not measured occupancy or performance. The frozen042 <=42 register rejection stands; no retrospective gate change or native/service run follows.

### 043 private 40-byte Q8 records: preregistered source preparation

Hypothesis: replace042's separate payload/scale planes with private40-byte interleaved records for the same ordinary PQ2 down-decode shape K17408/M5120/T1 on SM86. Retain the original half2 scale/sum at bytes0-3, leave padding4-7 untouched and put the same32 signed quantized values at bytes8-39. Four aligned int2 loads per chunk may reduce register grouping/address lifetimes relative to042. This is a different layout, not a rerun of042. Independent feasibility review permits isolated source preparation only, with low confidence of reaching the service gain.

Costs:544 records occupy21760 bytes versus19584, an11.1% scratch increase. Expected peeled/paired static loads are12 payload64-bit instructions plus18 weight/scale loads,30 total versus042's24 and retained scalar code's42 before optional bias. Neither static load counts nor footprint predicts traffic or throughput. The compiler may still preload both chunks.

Source requirements: assert record size40, alignment at least8 and payload offset8; preserve original quantization, signed codec, DP4A and FP32 accumulation order. Preserve all original dispatch/alias/contiguity/single-device/fusion guards, stream and private pool lifetime, geometry, default-inert diagnostics, CUDA error checks and retained paths. Production must neither read padding nor add an initialization/clearing operation. Update versioned byte-test ABI and its consumers consistently. Native preparation retains byte/layout/ABI/fallback/changed-input graph/direct-capture-replay-replay coverage and adds every-padding-byte verification with an observable padding-corruption negative control. Exact selectors/counts must be registered before execution.

Frozen compiled gates: private producer <=16 registers; consumer <=42; zero spills/shared/stack/local; all178 retained accepted037 function bodies/resources exact and exactly the intended new symbols. Require12 static LDG.E.64.CONSTANT payload loads, independently mapped to four per chunk across peeled and paired execution, plus the18 retained weight/scale loads. Preserve ordered arithmetic and geometry. Prove scale and payload derive from a shared record address, paired chunks differ by1280 bytes and successive paired iterations by2560, with no independent persistent scale-plane indices/base. Check operands and live ranges; register counts alone do not prove address simplification.

Runtime/service gates remain sequential: reviewed compiled semantics, exact byte/native/fallback/standalone replay proofs, actual canonical-service trace and strict growth/atomic correctness, then fresh uninstrumented service ABBA with separate conditioning. Require output gain >=2% for both512 and4096 token prompts in each matched pair and pooled result; ingestion regression <=2% for both. Use canonical256-output-token requests, five measured repetitions per prompt, original model/capacity/settings, accepted037 control. Stop on the first failed gate; no selective rerun or threshold relaxation. All failures and artifacts stay preserved. No043 compilation, extraction, GPU execution or service replacement is authorized by source preparation alone. The accepted037 service and four retained wins are unchanged.


043 source preparation is ready at base21e3bc3107120d617d63bbdc20587624abeb465d, six files,+517/-2. Patch cuda-pq2-record40-q8-source.patch SHA25643241bf9c125398c6a12a7fb99675d9372bc46830b8af71a1375eca1853a6d84; source-ready.json SHA25682ce7373041f3180a37201e47902e8fdcd96345759c9f2fae7b9e19820f220d9 binds all six reconstructed blobs. Root independently rehashed the patch, ready receipt and all six worktree files. Author reports71 CPU checks, including65536 packed words/524288 decoded coefficients and individual fault injection across2176 padding bytes. Independent source review is pending; these are not native GPU results.

Prepared native inventory is now explicit before execution:10 byte cases (five patterns/five corruptions),8 enabled graph cases,8 disabled cases and153 retained passes/33 unsupported, totaling179 core passes/33 unsupported; one separate exact eligible fresh replay case adds1 pass, for180 total. Byte ABI v2 reports actual padding equality/counts and full21760-byte record size. The native padding fault flips the final record's final padding byte after D2H and invokes the same full comparator; CPU proof separately covers every padding offset. Existing replay UID/key/index/host-submission requirements remain mandatory.

Three author preparation failures are preserved as cuda-pq2-record40-q8-preparation-failure1/2/3.json: non-quiet git show-ref for an absent ref returned128 instead of the draft's expected1; a compound source patch had a mismatched header-comment context; a Python -c packaging draft failed with unterminated string syntax after PowerShell quote processing. They were corrected with quiet absent-ref inspection, actual header-context inspection and explicit packaging edits. No build/extraction/GPU command occurred. Independent compiled-helper preparation reuses the178-function baseline without new extraction. Root prepared an exclusive canonical build wrapper but has not run it; compilation remains gated on source review.


### 043 independently reviewed source compiled successfully

Independent source specification/quality review passed for compilation only. Proof cuda-pq2-record40-q8-review-source-v1.json SHA25667bb389356a62bb64f84349a811967787385dc20c5eec724257b43ba3a8f9c09 reconstructs all six blobs and independently checks65536 words/524288 coefficients and2176 padding-corruption positions. It verifies the public buffer accessor, ordered arithmetic, stream/pool/fusion placement, untouched production padding, ABI2 diagnostics and180 total prepared native passes/33 unsupported. Retained MMVQ suffix and generic common/vecdotq/quantize files match baseline.

Root applied the exact reviewed patch after clean-source and duplicate compiler/output checks. Actual build baseff4713e99e94ea7ea85e09f5260f4a7bc6f16fc3; build-source.json SHA2565e68accfc10b6d3fb2a19039cc06df84d79dae757311faafdc8786df3dd194c2. All six source hashes match before/after compilation. Canonical wrapper SHA2561ee0ac4e94b21963d44b12faf1de0722168f25ddf2bfe005f8e10fde5dcdc653 preserves native WindowsPowerShell child module defaults and exclusive full output.

Build monitor51682 exited0 in61.491829 seconds; terminal collected. Full output and real command/exit/timing remain in cuda-build-pq2-record40-q8-build.txt/.exit.json. CMake/host/CUDA warnings remain preserved, including unused variables and UI-asset warning; this is not a warning-free build. Eleven files were exclusively snapshotted and rehashed at tools/llamacpp-cuda-pq2-record40-q8, manifest cuda-pq2-record40-q8-binary-hashes.json. CUDA SHA2565c9cf4adf2b007cddfee3dedc944fbbf4424e100847c7130759d3495e9f2ec0c; test executable SHA256abb5274ffed5f750fedccbde37b5d3fc1088f38862e3e5ddfb128496b0bbab66; test source SHA256d9358ee7c081e593880f1fff889c48c199ebf886fd87bb4a8082fd45b9700fc6. No043 GPU/native/service execution occurred; accepted037 remains served.

Offline checker preparation is complete under Temp/llama-pq2-record40-q8-static-20260923. static-check-v1.py SHA2569bc3e40261e9677d730ff3bb3ebd96c1b327c589fada96699ab9f142e849db57; ready-v1.json SHA2561f00c54e0de983dc77efd6bede627fd08019783a22738b753b077b04d85a650c. Author57 CPU tests passed with full streams/exit receipt; all178 baseline bodies were reused with zero extraction. The initial source-ready read used a nonexistent worktree-root path; its failure and correction to the exact existing logs path are preserved in preparation-observations-v1.json. Independent helper/baseline review is pending before actual compiled comparison. Numeric success cannot waive separate compiled semantic review.


043 static helper V1 failed independent review before any extraction. Review cuda-pq2-record40-q8-static-review-v1.json SHA25614609034af74fee29ec7aeff1547b3676ae976f60d7f54d6ae526dc7b4050fe7 reproduces actual-receipt rejection: helper lines137-138/227 require build_base/candidate_patch_sha256, whereas the unchanged real build receipt uses base/patch_sha256; direct validation reports Missing build base. All57 synthetic tests passed but did not cover that real-schema mismatch. Existing178 baseline provenance and frozen numerical gates checked correctly; no numeric result could approve semantics.

The review also found tests-v1 line28 writes fault fixtures under helper OUT even with a different result path. The reviewer's first57-test rerun created three directories in that author root; all are preserved and this scope error is disclosed, not removed. Author is producing versionedV2 to validate the actual schema strictly and route every test fixture through an explicit output root. V1 scripts, proof failures, test outputs, candidate source/binaries and original build receipt stay unchanged. No candidate extraction/preflight/native/service command occurred.


043 static checker V2 is prepared, pending independent delta approval. In the same isolated checker directory, static-check-v2.py SHA2565626637126083b190a872ed76340b01957cad2299ac0f4548c19a3f35aeaa1f1 and ready-v2.json SHA2562f31cd79366ba5799ef058dfd8ff5b27880ea186b207af0615f4f191109ba3ad bind the unchanged real build receipt/source. Author65 CPU tests pass; actual-receipt-cpu-proof-v2.json SHA256eef68a225e90f8fd472a4b48b904680000e03ec658415dc08e786797b8e6fe3e verifies actual base/patch_sha256/source_base fields, exact receipt hash and all six current source bytes. Wrong/missing schema keys and foreign aliases remain rejected. An explicit new absolute test output root owns all fixtures/results/streams; collisions are rejected. Author AST comparison reports compiled screening, symbol selection, inventory, extraction, baseline loading and source checks unchanged. Original numerical/semantic gates are intact. No candidate comparison/extraction occurred.

The V1 review artifact also preserves two failed ready-file lookups (workspace logs and a nested helper-root/logs path) before using helper-root/ready-v1.json successfully. These were read-only preparation errors and did not rerun a candidate measurement.


### 043 rejected by its original compiled register gate

Independent V2 helper specification/quality review passed65 CPU tests and ten additional invalid-receipt probes. All four mutation fixtures stayed under the explicitly supplied reviewer output root; originalV1 fixtures/artifacts remain unchanged. Actual receipt and all six source hashes bind exactly. Compiled screening/extraction functions remained AST-identical. Proof cuda-pq2-record40-q8-static-review-v2-verdict.json SHA25608b25dd5a983b449f332b33910908cf9ae4415417934e801516c9034a8558595 authorized one actual comparison, not semantic or runtime acceptance.

Actual comparison session84159 ran once and exited1 in38.962127 seconds; terminal collected. Both cuobjdump commands exited0 with empty stderr; checker/wrapper failed Numeric screening rejected. Raw output/functions/screening/provenance remain in the isolated checker directory with prefix cuda-pq2-record40-q8-static-compiled1-, and wrapper stdout/stderr/exit remain in workspace logs under the same prefix. Raw SASS SHA25689c3789c04521351de1cbbb88e7e5cf04b4d3b69169862a96f6a871a0027c438; raw resources SHA25691dd8b1f79c3e1b6d5e6fc93f5eb08f30033be88436721d8b913112819a37dfb.

The consumer uses46 registers, exceeding the frozen42 ceiling. It has288 instruction slots and exactly12 LDG.E.64.CONSTANT plus18 LDG.E.U16.CONSTANT loads; those load/cache gates passed. Producer uses14 registers and72 slots, below its16 ceiling. Both targets have zero stack/local/shared/spills. All178 retained bodies/resources match, and the whole resource inventory adds exactly the intended two kernels. The failed register gate rejects043; it is not a measured throughput regression or proof of reduced occupancy. No semantic approval, threshold relaxation, selective rerun, native/GPU or actual-service test follows.

Independent rejection proof cuda-pq2-record40-q8-static-review-rejection-v1.json SHA256729a90669b2ab20a066be95f096c04b89273f1d7eb85e7901b1b9c6c3cf0ba7c rehashed16 artifact bindings, all six preserved source files, actual binary/helper/build receipt, and independently reparsed all180 raw SASS bodies. It confirms46>42,178 retained kernels exact, exact two-symbol addition and both successful extractions; failed checker status remains intact.

Root verified all six current source hashes against reviewed source and preserved worktree, reverse-checked and reversed only the exact043 patch, and verified clean accepted source. Receipt cuda-pq2-record40-q8-rejection-source-verification.json. The source worktree/patch/binary snapshot and every failed/successful preparation/build/check artifact remain preserved. All eleven stable served runtime files still match accepted037. Four wins remain; next work is read-only postmortem and a materially changed candidate hypothesis.


043 read-only postmortem found that address simplification compiled, but payload lifetimes did not shorten. In cached043 SASS (zero-based slots), the record pointer R38:R39 is initialized at103-104 and advanced by2560 bytes at147-148; scale loads use0/1280 offsets and payload loads8..32/1288..1312. Versus042, IMAD.WIDE count falls10 to9 and IADD3 falls12 to5. Nevertheless all sixteen paired-chunk payload words are loaded before the first DP4A at165. Tail pairs R24:R25 and R26:R27 loaded at131/133 remain live until242/247. Four scale values and the persistent pointer overlap codec temporaries. Baseline037 also preloads sixteen words, but scalar loading does not require adjacent pairs and its activation address temporaries die after loading. These observations do not prove the allocator's exact48-to46 change or explain every register versus baseline42.

The same local occupancy formulas place42/46/48 registers in one allocation tier;40 would cross the next boundary. This does not reopen043 or imply a service result. The author found no sufficiently supported further source candidate from this payload corridor: ordinary source reordering need not delay SASS loads, while enforcing dependencies or replacing vector pairs adds work/reduces overlap without evidence of a net gain. Experiment029 already demonstrated that lower register count alone did not improve actual output. No files were changed and no extraction/build/GPU command ran for this postmortem. Next read-only discovery is bounded to the strongest remaining ingestion/output hot paths in the existing accepted037 service trace, with previous failed candidates excluded.


### 044 candidate discovery: SwiGLU followed by signed FWHT

Read-only analysis of the hash-verified accepted037 SQLite identifies a different possible fusion before FWHT. Across measured prefill evaluations, the author reports T508 (two evaluations): MMQ1212.55ms/800 launches, GDN334.66ms/96 and gated activations61.81ms/256; T512 (seven evaluations): MMQ4070.21ms/2800, GDN861.94ms/336 and gated activations199.85ms/896. The particular17408-wide SwiGLU contributes46.10ms/128 launches at508 and146.28ms/448 at512. Each observed target activation is immediately followed by its signed1024-point FWHT, contributing29.75ms/128 and99.10ms/448 respectively:64 adjacent pairs per evaluation. The activation's3.89-3.94% share of collected prefill GPU time is a cost envelope, not a predicted speedup. Existing5571 GetGraphId and5571 GetGraphNodeId severity3 warnings plus incomplete-collection warning remain; complete capture is not claimed.

Hypothesis under review: a dedicated FWHT input load computes the same rounded-F32 plain SwiGLU, followed by the same sign/normalization arithmetic and butterflies, for SM86/F32/K17408/T508 or512. This could remove the separate activation launch and intermediate write/read for eligible pairs. The ordinary down-MMQ, its Q8 quantizer and GDN stay outside this change. This differs from026's MMQ writeback and041's FWHT-output-to-Q8 fusion. Source anchors are llama-graph.cpp1890/1965/1569, llama-impl.h57-72, ggml-cuda.cu3513-3535, unary.cu263-276 and fwht.cu139-217 in accepted source.

Actual eligibility remains unknown: adjacency does not prove canonical consumer/output/view constraints or safe overlap of external gate/up inputs with the FWHT output. Next read-only work checks whether existing024 diagnostics already expose the needed predicates; independent review checks arithmetic/aliasing and prospective resource/service gates. If missing, the smallest proposed diagnostic is inert-by-default host predicate logging of UID/index/T and structural/consumer/range rejection reasons, without CUDA calls or replay changes. No044 source edit, instrumentation, build or GPU command has occurred; no speed claim or kernel acceptance follows from discovery.


### 044 eligibility diagnostic: source-preparation contract

Existing DEBUG_CUDA_FFN_FUSION only inspects [MUL_MAT,MUL_MAT,GLU] at ggml-cuda.cu4608-4641;024's40 eligible projection groups cannot establish downstream SwiGLU/signs/reshape/FWHT eligibility. DEBUG_CUDA_GRAPH_STATS supplies UID/key/token width and direct/capture/replay mode but no required consumer/alias observations; its parser has a fixed graph-event schema. Discovery therefore calls for a separate default-inert marker, not an inferred eligible count.

Prepare a diagnostic in ggml-cuda.cu only, enabled by cached exact-value DEBUG_CUDA_SWIGLU_FWHT=1. One shared pure predicate checks [GLU,MUL,RESHAPE,MUL_MAT], final output i+3, plain split SwiGLU, exact links/Hadamard hint, F32 contiguous K17408/T508 or512,1024-point transform, signs shape/singleton dimensions, supported buffers and runtime/compiledSM86. Canonical ggml_can_fuse_subgraph is authoritative for consumers/observable outputs/view ancestry. Validate buffers before canonical ggml_cuda_check_fusion_memory_ranges, then explicitly test final output against external gate/up/signs/Hadamard leaf ranges because the canonical helper skips GGML_OP_NONE sources. No unsupported pointer arithmetic or unchecked metadata dereference may be used merely for logging.

Call the inspection after existing graph mode selection and before capture starts. Emit separate CUDA_SWIGLU_FWHT records with invocation sequence, existing UID/key/mode, device/index/T, linkage/shape/contiguity/allocation flags, canonical subgraph/range result, intermediate consumer/output/view facts, individual external-leaf overlap flags, deterministic rejection reason and per-graph totals. These are graph inspections, including inspections preceding replay, not CUDA kernel submissions. Disabled cost is a cached flag branch. The diagnostic must make no CUDA API call, synchronization, tensor/device/scratch allocation, graph mutation, kernel launch or replay change. Existing host log formatting may allocate host memory. Keep graphstats schema unchanged and reuse the canonical runner's CudaGraphStatsLog option plus a separate launch flag.

Before actual service use require independent source review, unchanged compiled retained GPU paths/resources and full source/binary provenance. Then one canonical diagnostic service run may observe512/4096 prompts with32 generated tokens and one repetition, retaining every request/output and matching accepted037 reference tokens/content. Its timing is diagnostic only. Observe actual candidate/eligible/rejected counts by invocation/UID/index/T, including warmups, and report every rejection reason;64 adjacent pairs do not imply64 eligible pairs. No eligible-count threshold is invented to turn discovery into a win. Instrumentation errors or inconsistent identity/counts block kernel preparation until explained. Source preparation alone authorizes no compilation/extraction/GPU/service action. Kernel implementation and its final resource/service gates remain a later reviewed stage after observed eligibility.


044 independent feasibility review passes diagnostic preparation only; kernel preparation remains conditional on actual allocation eligibility. It confirms the canonical memory checker skips OP_NONE leaves and requires explicit final-output overlap checks against gate/up/signs/Hadamard after valid buffer/range checks, plus canonical consumer/output/view handling for memoized transform reuse. Current source preparation remains one host-only diagnostic file, with no computation changes.

For a later fused kernel, the reviewer identifies the accepted execution order as rounded F32 silu(gate)*up, then normalization multiplication, then sign multiplication, then existing butterflies. Fast math means a local float alone cannot prove that the removed store's rounding boundary remains; compiled dependency proof and bitwise comparison against the unfused CUDA path, including sensitive inputs, would be required. K17408 contains seventeen1024-point transforms: signs use r%17, grid17*T,256 threads and four elements/thread. A prospective <=40-register/4096-static-shared/zero-dynamic-shared-stack-local-spill gate preserves the calculated48-warp tier, not a performance result. At64 eligible pairs the eliminated intermediate write/read would total about4.22GiB forT508 or4.25GiB forT512 of logical activation traffic; this is not measured off-chip traffic or predicted savings, and activation arithmetic remains. The proposed later service gate stays >=2% ingestion for both prompt sizes in both matched pairs and pooled results, with <=2% output regression. No build/extraction/GPU/service action occurred in feasibility review.


044 diagnostic sourceV1 is prepared at base37a98073eedd897e122730c4820d1d813d8f8000: only ggml-cuda.cu changes,+226 lines. Patch cuda-swiglu-fwht-eligibility-source.patch SHA2566675968109330459e414e60b1f6f59f0a9ddac95d41fb903ff21af86fded1665; source-ready.json SHA2568cd8c9874282303e513b96ad0be23d54f54cb98070ec9bbc42d491c301b7b78a; source SHA25614fbddf21f027b5b1441ff49b5c6dbcca832874eb486261cfa9e634ec875dba9. Root rehashed these bindings. Author37 source/CPU checks passed, including4014 bounded-size cases,4096 independent alias cases, format arity, fresh patch reconstruction and byte-identical original execution source after removing additions. No author preparation failure was reported.

The schema emits one final candidate reason, eight tensor records, applicable view ancestry and per-graph totals, correlated by invocation/UID/index; candidate records additionally bind key/mode/T. Unsafe/untested facts are unchecked, absent tensors are null, and fusion-disabled state is explicit. This source evidence does not establish actual service eligibility.

Independent review has identified a compile blocker before building: the diagnostic calls ggml_is_constant(view), which is file-static in ggml.c and undeclared in the CUDA translation unit. Review continues before a versioned author correction; canonical helper export/scope will not be changed for logging. Offline retained-kernel checker preparation continues separately, with sourceV1 unapproved. Root prepared an exclusive canonical build wrapper but did not execute it. No044 compilation, extraction, GPU or service action has occurred.


044 diagnostic sourceV1 independent review failed specification/quality before compilation. Proof cuda-swiglu-fwht-eligibility-review-v1.json SHA256b6004f8cdcb5605c2d9996dd8c35390889fd99df44f86f13885a61f1cccfccf6 confirms two blockers. The private ggml_is_constant call has no CUDA-visible declaration. Separately, the canonical range guard validates final output and actual sources but does not guarantee first-node metadata when links are broken. With GLU dimensions [17408,INT64_MAX,2,1] and broken mul->src0, all actual source ranges can be valid while canonical ggml_nrows(first) evaluates INT64_MAX*2 before its is_topk_moe check. This signed-overflow counterexample is a source/guard model using exact Python integers, not execution of C++ undefined behavior.

Independent4014 span and4096 alias rechecks passed, as did exact patch reconstruction and unchanged original execution bytes. The37 author checks did not execute the C++ predicate and missed both defects; they are not relabeled native correctness. A reviewer encoding failure is preserved in cuda-swiglu-fwht-eligibility-review-failure1.json before the successful UTF-8 rerun. AuthorV2 is requested to use a diagnostic-local public-API equivalent for constant classification and validate first-node metadata before the canonical call, with a focused broken-link regression and explicit unchecked facts. OriginalV1 source patch/receipts/proofs remain preserved; no compile/extraction/GPU/service action occurred.


044 diagnostic sourceV2 is prepared, pending independent delta approval. Full patch cuda-swiglu-fwht-eligibility-v2-source.patch SHA256fd5a325fe6d98f918fa01f9ac214ea776798d46edbb33e8809e73efb09f1f223; V1-to-V2 delta SHA256f869dd046b06eced750180dd289e354d15fbce464e59d2d62411b8fcf17ed849; ready SHA2561a76c5fbdb6e60d138f4e840d230237900eb840c138c01cae71c2a1f9bf8314c. Only the same CUDA source file changes,+235 lines overall, SHA256b054bcbcfa51d69e34f28f72c0c9c1147e4aa0190c2d0a812157617c03c998f4. The delta introduces a diagnostic-local constant predicate with canonical null-buffer/WEIGHTS/non-PARAM semantics and validates first-node metadata before the canonical range call, leaving skipped range facts unchecked. Author55 checks pass, including public API availability, expression equivalence, broken-link overflow and both full/delta reconstruction. These remain Python/source checks, not native C++ execution. No new preparation failure was reported; V1 artifacts remain intact.

044 offline invariance checker logic is prepared separately in Temp/llama-swiglu-fwht-eligibility-static-20260923, pending corrected-source rebinding and review. static-check-v1.py SHA2568096fdcd6a40810a0a9e2df25552b3754eebc2ccd59aea21cec09c1aa280401c,31 CPU tests passed. Besides178 retained bodies, it checks all6993 baseline resource occurrences/6517 unique symbols, including metadata of213 legitimately repeated support symbols; no added/lost symbols are allowed for host-only diagnostics. Its original28-test pass used sourceV1 before volatility notice; later tests mock only the positive sourcefile lookup while checking immutableV1 ready/patch data. No successful actual receipt or corrected-source validation is claimed. The future receipt contract uses base/source_base/patch_sha256/exact one-file source_hashes; actual build SHA and receipt hash must be bound when available. No extraction/build/GPU/service command ran.


### 044 corrected diagnostic compiled successfully

IndependentV2 delta review passes specification/quality for compilation only. Proof cuda-swiglu-fwht-eligibility-review-v2-verdict.json SHA256615e9cc0939ffe845d49ee88531383d04281c80cd0857c4952cad4dde487c343 verifies exact full/V1-delta reconstruction, canonical constant semantics through the public getter, unchanged remaining source and all18 guard combinations. The previous invalid-first-node witness now leaves the canonical range call unchecked.

Root applied the exact reviewedV2 patch after clean-source and duplicate compiler/output checks. Actual build basec014770dab46eb3e741586638baab20f04477e76; build-source receipt SHA2561e5d3a0727dcf3c7d42d532cf87b9332f3a298f1e9fdb722324d1b0a4a94b802. Wrapper SHA256cd560cd64255f4318aa2a506e4753567dbf61789e7efde8df408a189e574ad8f uses the canonical build script, exclusive outputs and native WindowsPowerShell child module defaults. Build monitor73077 exited0 in61.028131 seconds; terminal result collected. Full output/real exit are cuda-build-swiglu-fwht-eligibility-build.txt/.exit.json. Existing host/CUDA/CMake warnings and UI-asset download warning are preserved; this was not warning-free.

Eleven runtime files were exclusively snapshotted/rehashed under tools/llamacpp-cuda-swiglu-fwht-eligibility, manifest cuda-swiglu-fwht-eligibility-binary-hashes.json. CUDA SHA256cea4aeda15c1160ab558d72ffc7ab089ab97fc957f7cf37497f0473007ee5e71; test executable SHA256c60bff9cbe16effb1a1e7e32f7fd878b3e3fe9fd773128c44d7cda6d4a6e9ccb; unchanged native-test source SHA256e9e01e08ab7446025244b0f540d43a18d85efdad7316e2f8b97e6c557716eb04. A root status-display wrapper then raised TypeError while treating the monitor's JSON string as an object. Decoding the retained same result fixed display; build was not repeated. Failure artifact cuda-swiglu-fwht-eligibility-build-status-format-failure1.json distinguishes this display error from build exit0.

Offline checkerV2 now pins correctedsource: static-check-v2.py SHA256f24bf5c8dd8afedf7a2e292b7165d259518c6b347597784b8be61f1b4b5cfe7b, ready-v2.json SHA256fd213c1e4fa1f715cfca6ebe95c45f6bc036014aa48e379d9c4c616c07bfabf0. Author32 CPU tests passed against actual stableV2source, without source-hash mocking; only three source pins changed and helper functions stayed identical. Independent checker/baseline review and actual receipt CPU validation are pending before extraction. A separate durable log-validator preparation is underway to reconcile actual inspection records and report observed eligibility, including zero eligibility honestly. No044 GPU/native/actual-service request has run; accepted037 remains served.


### 044 compiled diagnostic comparison completed; independent review pending

Independent checker V2 specification/quality review passed 32 CPU tests and seven additional fault injections. Proof cuda-swiglu-fwht-eligibility-static-review-v2-verdict.json SHA256 93875b72f6453a3655c8212a0d4f3d53c28341d5d69e41bd66c8f38a6f15c39d rehashes 20 baseline provenance artifacts and all eleven candidate snapshot files. It verifies 6,993 resource occurrences, 6,517 unique symbols, 213 repeated support symbols and 178 retained baseline bodies. The actual canonical build receipt passed CPU preflight with zero subprocesses or extraction. Only the three reviewed source pins changed between checker versions.

Root ran the one authorized actual offline comparison after checking for existing outputs and active extraction/native-benchmark processes. Session 41746 exited 0 in 37.9841919 seconds, terminal collected. Its two extraction commands completed and the checker reported REQUIRES_HOST_OBSERVATIONAL_REVIEW. Wrapper streams and actual argv/exit/timing are cuda-swiglu-fwht-eligibility-static-compiled1.stdout.txt, .stderr.txt and .exit.json in workspace logs; raw comparison/provenance artifacts remain under the same prefix in Temp/llama-swiglu-fwht-eligibility-static-20260923. Independent review of those actual outputs is pending. Normalized instruction-body comparison covers the pinned 178 kernels; whole-DLL resource/symbol checks do not establish all-symbol instruction-body equality. No new baseline extraction or service measurement was performed.

The durable host-log validator is ready separately in Temp/llama-swiglu-fwht-validator-20260923, base c014770dab46eb3e741586638baab20f04477e76. Its patch SHA256 is 6bd62f1f019b893cf1b3b2ca7b90d85d3f182d84ed1d3b7f6c33cb34002f4417; ready receipt SHA256 bfc363d9a40eb659a6f15978f3ebec4b0066f2976b710327b6e07c6869f92e0c. Author reports 28 passing CPU tests, a durable synthetic direct/capture/replay/replay positive CLI run with exit 0 and a truncated negative with exit 2 and retained failure JSON. No unexpected preparation failures were reported. Independent tool review and actual-service validation remain pending; these synthetic records are not observed eligibility.

Fresh service inspection still finds accepted037 PID 23196, launcher 23616 and wrapper 22604, with all four slots idle at 188416 context each. Original supervisor and its preserved backup both hash to 54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be. No 044 deployment or GPU request has occurred. Four retained wins and the last accepted service speeds remain unchanged.


044 actual compiled comparison now passes independent specification and quality review. Proof cuda-swiglu-fwht-eligibility-compiled-review-v1.json SHA256 44f623974a887810b5a7a3cb2a06cadbd60b6ba5fd9396b5531eaad5fff50ff7 independently reparses all 178 selected bodies (126456 instruction slots), verifies ordered instructions/resources unchanged, and confirms all 6993 resource occurrences, 6517 unique symbols and 145 Common records byte-identical to accepted037. There are zero symbol additions/losses; both extraction commands and wrapper exited 0 with empty stderr. Original execution source remains unchanged around the host-only additions; source/build/binary hashes bind exactly. This authorizes only the preregistered host service observation after validator review, not a kernel implementation, speed trial or retained win.

The durable validator V1 failed independent specification/quality review. Its 28 existing tests pass, but seven independent CLI probes expose three accepted contradictions: F32 nb0=8 with contiguous=true and eligible=true; a view-chain parent disagreeing with the same known tensor's logged view_src; and overlapping four-node patterns assigning contradictory identities/operations to a shared graph-node index. These are consistency checks supported by emitted facts, not attempts to infer absent external edges. Proof cuda-swiglu-fwht-validator-review-v1.json SHA256 be6df2fa4ecbd92662c7c3a380313345452d5df360e93237f6c8fd41b8eb5579 contains exact line references, CLI commands, exits and artifact hashes. All three corruptions incorrectly exited 0; legitimate positive and zero-eligible cases passed, and truncation correctly exited 2. Fixtures and full streams remain in Temp/llama-swiglu-fwht-validator-review-20260923. V2 corrections with exact regression cases have been requested; original V1 artifacts stay intact.

Root prepared, but did not activate, new control044 inspection/stop scripts and a canonical service-launch wrapper. Preparation receipt cuda-swiglu-fwht-eligibility-maintenance-prepared.json binds their hashes and the accepted037/original-supervisor rollback. No supervisor bytes, stable runtime files or running service were changed. Actual-service observation remains unrun pending corrected validator review.


### 044 validator V2 approved; diagnostic service prepared

Validator V2 fixes all three reviewed contradictions with canonical F32 singleton-stride handling, inspection-wide known tensor/view facts and graph-node-index consistency. Full patch SHA256 c406fd1dc2c9e6cd916cf23c5c4194a508c97b2ee651b64ba418c04159f903ef; V1 delta SHA256 92e281ab9f72456950885efe0fe14049d9356b07cb0e5ad1467b8fc52ac3b21a. Both full-base and V1-plus-delta reconstruct both files exactly. Author 37 CPU tests and all seven original reviewer CLI fixtures pass; the three previous false approvals now exit 2. Independent V2 specification/quality review passes 37 tests, 14 CLI probes and 48 stride-layout checks, including legitimate singleton/view/shared-tensor/reused-storage cases. Proof cuda-swiglu-fwht-validator-review-v2.json SHA256 2707384844c1b71fd053be56ad2755d1d57d00780d2121c870fcd7655c1d7e22. Logger schema and existing graphstats parser remain unchanged. Root applied the exact reviewed two-file patch and rehashed both installed sources. No actual log has been validated yet.

The first maintenance preflight exited 1 before any patch application or supervisor change: wrapping Invoke-RestMethod in a PowerShell array subexpression counted its returned JSON array as one pipeline object. Direct assignment independently showed the four idle 188416-context slots. Receipt cuda-swiglu-fwht-eligibility-maintenance-failure1.json preserves the failure and verifies original supervisor hash and absent validator source. The corrected preflight succeeded without repeating any benchmark.

Root activated the inspection-only supervisor prefix, obtained Session0 command lines for exact wrapper/launcher/server PIDs 22604/23616/23196, then separately activated the identity-checked and idle-gated literal stop. inspection044.json and stopped044.json preserve both steps; scheduled task completed 0. The original supervisor bytes and accepted037 snapshot remain the rollback. Canonical deploy exited 0: bonsai-deploy-20260923T104317054-98fb5a5b57744267a967645c47cb21e4.json. Tracked diagnostic service monitor 70263 launched actual server PID 12712, parent 22216, with the exact new log path and graphstats verbosity. Health is ok and all four slots are idle at context 188416. No benchmark or generation request has been sent to this diagnostic process yet.

Bounded discovery found no complete reusable process-provenance assembler in the 195 scoped script files. Exact prior read-only process/module commands survive in their exit receipts; the 041 helper has reusable exclusive command receipt functions, but its CLI and deployment validation are candidate-specific. A small parameterized repository provenance script is therefore being prepared before the run, using those observed commands and existing receipt structure. Discovery preserved a failed search from an utf8-sig codec spelling, corrected to utf-8-sig; a subsequent search emitted an invalid-escape SyntaxWarning and completed 0. These are preparation issues, not service or GPU failures. The running diagnostic has no changed CUDA kernel; it has not been counted as a retained speed win.


The reusable scripts/bonsai-server-provenance.py and focused tests are prepared in the isolated tooling worktree. Author 18 CPU tests and clean-base reconstruction passed with no live calls. Independent review repeated all 18 tests and eight additional faults covering PID reuse, foreign listener, duplicate modules, changed backing files, duplicate JSON keys and invalid slot IDs. All behavior checks passed; review cuda-server-provenance-review-v1.json SHA256 de01a870ebdd8e2fc1a4b47e60f6c6a9e610b370e31f223a9d76d23f5b998485 requested one specification correction: the actual persisted scope must explicitly say process/file/HTTP observations are sequential rather than an atomic snapshot. V1 only disclosed backing-file versus relocated-memory hashes in that output.

V2 changes only that scope string, appending the requested sentence. Full patch SHA256 f95bb70eb57117809c5f4637e799ddad3fe475a2aac4d22c3213657ddccafb96; delta SHA256 cdd52562c407f8123646fef7e93d2901911cc92f1b2df13aeb01f8eeb224d2e5; tool SHA256 a24b409208c7383678eb755797b6a26f9c297c38ab1246d62159cc5108bd80d7; unchanged tests SHA256 57427c5f9d040736dfd2a8ec6ea80aa2c126ccbf8071a1b7f264639dac4937ce. Byte/AST-only and both reconstruction proofs passed. No tests were rerun merely for the wording constant; the V1 behavioral evidence remains explicit. Mechanical delta review is pending. The reviewed validator files remain unchanged. Diagnostic PID12712 and parent22216 are still the observed processes; no benchmark has run.


### 044 actual service observation: zero eligible targets

Provenance V2 mechanical delta independently passes specification/quality: cuda-server-provenance-review-v2.json SHA256 48bfbd3b9bbe4141d59fa8acf5580226f73732dc6d1a2804ed9eb2fada59f899. Root applied and rehashed the exact two-file patch. The actual read-only provenance command exited 0 in 2.234616 seconds and passed PID12712/start/parent22216, stable executable/listener, all eleven snapshot/deployment/stable file hashes, nine loaded runtime modules and four idle188416-context slots. Receipt cuda-swiglu-fwht-eligibility-process-provenance.json explicitly limits these to sequential observations/backing-file hashes. Full command/streams/exit are preserved under -provenance-command1. The separate service-run-binding.json SHA256 be5cb3bbc945046e50ca9f08271e434effe4370b4b186c57b628b83da3e6a9da binds that receipt, source, launch wrapper, log path, benchmark, validator and accepted037 reference.

The one preregistered canonical benchmark (512/4096 prompts, 32 output tokens, one repetition plus warmups) ran once, session95964, exit0 in35.479698 seconds; terminal collected. Four complete requests/responses are cuda-service-swiglu-fwht-eligibility.jsonl SHA256 39827e20d47c01a4f94b2b47ff29cef96279ea72e1fec3233fccf5c19944258c. No-gate canonical comparison against cuda-service-nsys037-candidate.jsonl SHA256 0ce578c6b5b622eafde3925bd0257dd5b3ba295580331847066316a7870411e9 exited0 with correctness passed. The 128 output tokens/content, requests, settings and capacity match including warmups. This is correctness-only: the reported instrumented medians385.790/30.8425 tok/s at512 and532.910/28.5884 at4096 are not an uninstrumented performance acceptance or new speed win.

After benchmark completion and observing all four slots idle, root copied the size/mtime-stable log to cuda-swiglu-fwht-eligibility-service-snapshot.log: 33826754 bytes, SHA256 c310609d8d14a4833424a07c03e7f593e0d35aabedbd4babc88aa248cbbb2d7e. Snapshot receipt binds PID/start run evidence. The independently reviewed durable validator exited0 in8.667338 seconds:148 graph inspections and9472 candidate observations, including startup and replay. Every eligible flag is false. Target T508/direct256 and T512/direct896 observations all reject canonical_memory_overlap. Other-width rejections are T1 capture256/direct256/replay7424, T2 capture64/direct64 and T4 direct256. These are inspection observations, not kernel submissions. The parser intentionally leaves source_binding_verified=false because actual source/binary/process binding is an external evidence-review responsibility. Independent review of all actual artifacts is pending; no kernel implementation or alias-guard relaxation is authorized by these observations.

Root separately inspected then intentionally stopped diagnostic PID12712, collected monitor70263 terminal exit1 (wrapper outcome of deliberate child termination, not a request failure), and deployed accepted037 through the canonical script, exit0. Restoration receipt cuda-swiglu-fwht-eligibility-restored-service.json verifies all eleven stable hashes, exact original supervisor54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be, healthy normal server25844/parent12356 and four idle188416-context slots. Deployment receipt bonsai-deploy-20260923T110109754-0c79d74d44054f659b6d1337da72af5a.json. All diagnostic/source/proof artifacts remain intact. Four retained wins remain. Read-only allocation postmortem is examining the actual overlap classes and whether a materially changed graph-visible strategy can avoid them safely; original late fusion is not proceeding.


### 044 original guarded fusion rejected; reusable diagnostics retained

Independent actual diagnostic specification/evidence-quality review passed: cuda-swiglu-fwht-eligibility-service-review-v1.json SHA256 a011a2b2f601597c315260d6696c23709354ab1729cc4a0c6f67af4be2079685. It independently verifies all four requests and128 exact output tokens/content, including warmups; source/build/deployment/live-process/module bindings;148 inspections (2 startup plus146 benchmark), 27 direct/5 capture/116 replay; and9472 candidates with all eight tensor roles. Every one of1152 target observations has can_fuse=true, ranges=false, gate_overlap=true and no up/signs/Hadamard overlap. Two idle evictions approximately969 seconds after startup match the tooling-preparation gap. Informational comparator stderr contains two rate lines and PASSED; a reviewer assumption that it must be empty failed and was preserved, then corrected without rerunning measurements. Intentional diagnostic termination remains separate from successful requests and validation.

The original late four-node fusion is rejected on this actual workload because its required safe-memory predicate never holds. This rejects that guarded proposal; it is neither a measured kernel-speed regression nor proof that every alias-aware design must fail. Inert host diagnostics and the log/provenance tools are retained as reusable observability, not a fifth throughput win. The accepted037 binary remains served. Root integration checks reran the exact reviewed Python sources from their final repository paths:37 validator plus18 provenance tests passed, with complete streams/real exits in cuda-swiglu-fwht-integration-tests1.json and its referenced receipts. The CUDA source matches its successfully built/reviewed hash exactly; no computational kernel changed.

Allocation postmortem cuda-swiglu-fwht-allocation-postmortem-v1.json SHA256 d0426de72ef4450acf2fd05146d1ac1f009d51cf41e4b10704607bbb9cb8543d records18 target graph inspections of64 pairs each. Exact output/gate alias occurs494 times (T508110/T512384). Output begins earlier by5*T*4096 bytes in82 cases (18/64), or10*T*4096 bytes in576 cases (128/448), yielding658 shifted overlaps. All outputs are disjoint from up/signs/Hadamard/GLU; GLU/MUL/RESHAPE share one interval. CPU enumeration checked10009600 destination blocks. The accepted separate-kernel execution finishes reading gate/up before the transform; a hypothetical combined kernel would overwrite another block's unread gate input in shifted cases. Source lifetime rules permit freed/coalesced gate reuse, but the exact allocator free-list history was not logged and is not claimed. Documentation discovery had an inaccessible web-open result and a cp1250 UnicodeEncodeError while displaying the guide; corrected UTF-8 reading exited0. No source/GPU/service changes occurred in this analysis.

Two materially changed approaches are now under read-only feasibility review, with no implementation authorization yet: graph-visible fusion before allocation to keep explicit external inputs live, and a narrower dedicated exact-gate-alias implementation with per-block load-before-write and disjoint-footprint proofs. Neither may weaken the general canonical alias guard or admit shifted overlap. The former has the same local three-activation live set (about101.203MiB at508/102MiB at512), but global capacity/fragmentation remain unproved. The latter covers only the observed exact-alias subset; retained total activation time is not automatically removable time. Cached accepted037 costs must support the next bounded experiment against the unchanged >=2% ingestion/both-prompts service gate. No new speed claim follows.


044 alternative feasibility is complete: cuda-swiglu-fwht-next-feasibility-v1.json SHA256 e3fcf10b5c637ac310262971bd820d5e9b1aff73c4b17afc586c621d3f9a6eb3 recommends rejecting the exact-alias subset B as the next source prototype and deferring broader preallocation redesign A. Read-only recomputation of pinned accepted037 measured evaluations confirms T508 SwiGLU46.101458ms/128 plus FWHT29.745847ms/128, and T512146.283689ms/448 plus99.099504ms/448. Matching measured requests have24 exact-alias opportunities for the short prompt and223 across the long prompt's eight evaluations. Even an intentionally optimistic calculation that deletes the slowest eligible number of complete adjacent pair spans per evaluation saves only20.760612ms/125.383143ms, corresponding to1.8913%/1.7411% relative to those observed ingest times. The >=2% gate would require21.929922ms/143.664216ms. This is a trace-relative cost envelope, not a universal speed bound; real fusion retains the activation and transform arithmetic, and trace limitations remain.

The whole-pair erasure envelope for broader A is only about3.90%/3.98%, while its actual mechanism removes intermediate traffic/launches and adds graph/scheduler/fallback/global-allocation proof scope. Equal local buffer sizes do not establish equal global peak allocation. Exact-alias B remains a potentially safe design only after full-block load/barrier, linear-index and disjoint-footprint proofs; shifted overlaps remain excluded and the general canonical guard remains untouched. Neither alternative proceeds to source compilation or GPU testing now. The loop pivots to bounded read-only discovery of currently unoptimized output kernel/adjacency costs, including recurrent-state outputs and pointwise/norm consumers, in the existing accepted037 trace. No fresh service measurement, new kernel or fifth win has occurred.


### Next output discovery: small kernel fusion cost is insufficient; attribute host gaps

Read-only pinned037 discovery cuda-next-decode-discovery-v1.json SHA256 563bc08b101a0e6b857dd4a15f1bc0059bcb8b532b97572770490940da1bb04b selects30 measured steady replay evaluations per prompt, excluding warmups/prefill/first decode. Each contains1935 kernels. Per-token observed means for512/4096 are indexed GDN48 calls0.890/0.891ms, flash attention16 calls0.449/2.092ms, and the1024-thread weighted RMS variant129 calls0.584/0.588ms. These costs identify remaining work, not removable savings.

The strongest examined small boundary, GDN output through weighted RMS/SiLU gate/layout copy, spans only0.259/0.261ms per token across48 groups including recorded intervening gaps. Even total erasure is below the roughly0.532/0.568ms saved-time requirement for the frozen2% output gain at both prompt lengths. Actual fusion would retain reduction/SiLU arithmetic. It also is not a simple epilogue: GDN distributes a head's128 output columns across32 CTAs (grid48x1x32, block32x4), while RMS needs the complete head; an independent z-projection quantizer and PQ2 kernel intervene. Existing RMS/weight fusion already handles that part of the chain. No source prototype or GPU test follows this unsupported small-fusion candidate.

Across29 measured inter-token gaps per request, discovery reports mean device gaps0.783/0.893ms and pre-next-cudaGraphLaunch intervals0.726/0.834ms, each containing29 stream-synchronization API calls and nine asynchronous-copy API calls. Runtime callchains are absent. These intervals are not yet attributed to avoidable work; required readiness, input handling, scheduler actions and sampling may contribute. Next read-only review verifies timing/thread attribution and whether existing instrumentation suffices; independent local Nsight capability research checks CUDA API backtraces before adding code. If necessary, a default-inert host timestamp diagnostic would use existing phase boundaries and add no CUDA synchronization. No host PGO speed claim, fresh profiler run or optimization is implied.

The discovery proof preserves one corrected Python export syntax failure. Existing5571 GetGraphId plus5571 GetGraphNodeId warnings and incomplete-collection caveat remain; full capture is not claimed. The served accepted037 build, four retained wins and throughput measurements are unchanged.


Host-gap profiler capability investigation found no supported non-elevated runtime-callstack delta for the installed Windows Nsight Systems2026.5.1.161. Proof cuda-host-gap-nsys-backtrace-capability-v1.json SHA256 1c19f61091eeb93177f4d8c0be29c5ba1153897ae9832636d84ac5f259a419f5 retains the executable hash and exact version/launch-help/start-help/profile-help commands with complete outputs and exit0. None exposes --cudabacktrace. The Windows --backtrace=auto option controls CPU sampling, and local help requires administrative privileges for non-disabled process-tree/system-wide sampling. Generic NVIDIA documentation also ties CUDA API backtraces to CPU sampling; its generic option is not transplanted into the installed Windows CLI. --resolve-symbols only resolves frames that were captured. No connection to the earlier NCU counter-permission failure was established.

One combined Python evidence-capture command was blocked before process creation and is preserved as a preparation failure. Direct help-only calls and evidence writes succeeded. There was no profiling run, GPU-counter retry, elevation/permission experiment, install or service change. Prospective profiler delta is null. Independent review of the measured gaps and a possible minimal default-inert host timing diagnostic continues; source instrumentation is not yet prepared.


### 045 host-phase attribution diagnostic: source-preparation contract

Independent gap review cuda-host-gap-feasibility-v1.json SHA256 1d9d748fae1a45380a429a5ea0bfca21830cfa1a498437f10190f1603c52fd45 reproduces all58 measured gaps and1935-kernel replays. Pre-next-launch means0.726/0.834ms contain only0.258/0.291ms of CUDA API interval union, leaving0.468/0.543ms unattributed host wall time. All recorded runtime calls use one thread without mutual overlap, but the first synchronization starts during the previous GPU evaluation, and the graph-launch API continues after the next GPU work starts. Full launch API durations1.905/2.002ms are not idle overhead: only about0.0567/0.0585ms precede device work. No bottleneck or removable-time claim is established. The reviewer initially expected an unversioned cudaGraphLaunch name; exact row inspection found cudaGraphLaunch_v10000, after which explicitly normalizing only the version suffix allowed the preserved read-only recomputation to pass.

Prepare a default-inert host diagnostic under cached exact-value DEBUG_CUDA_HOST_PHASES=1, separate from DEBUG_CUDA_TIMING. The latter adds synchronization/disables replay and must remain off. Follow the full prospective_contract in the hash-bound review. Disabled instrumentation performs no clocks, counters, trace allocations or logging. Enabled instrumentation brackets existing context input setup, per-input/tensor uploads, scheduler graph enqueue, logit D2H enqueue, existing output-readiness/getter synchronizations, backend/buffer-set waits and graph direct/capture/replay submission. It must add no CUDA API, stream evaluation/query, event, wait, synchronization, device/scratch allocation, graph/cache/lifetime change or execution/error-handling change.

Use the existing exported monotonic ggml_time_us clock with explicit identity/units, PID/native thread, context pointer plus lifetime identity, decode attempt/ubatch sequence, tokens/n_outputs, span/parent identifiers and graph bridge/UID/key/mode/invocation where available without additional CUDA queries. Unknown or out-of-scope contexts must be explicit. Never identify a graph only by a reusable pointer or silently equate host and CUPTI clock epochs. Record bytes/direction/reason for existing transfers and waits. Distinguish enqueue from readiness and wall duration from CPU-active or GPU time; leave uninstrumented sampler/server time unattributed.

Enabled-only fixed-capacity per-thread recording must expose capture budget and dropped/truncated counts, batch flushing outside measured spans, and aggregate counts/bytes/sum/max plus enough sampled timeline to compute interval unions/exclusive spans. No per-kernel records, tensor values or unbounded names. Overflowed evidence fails validation. Nested parent/child synchronization durations and full graph-launch durations cannot be double-counted as idle gaps. Logging and clocks perturb execution; diagnostic rates never satisfy a throughput gate.

First prepare minimal context and CUDA copy/sync/submit coverage; server sampling instrumentation is a later decision only if unbound time remains dominant. Source review, successful compile, exact source/binary binding and unchanged retained device bodies/resources precede any use. A separate reviewed CPU validator must reject identity/correlation, duplicate, truncation, missing-span, nesting and clock inconsistencies. After those gates, one canonical matched service diagnostic sequence uses512/4096 prompts,32 output tokens and one repetition plus warmups, with all requests/outputs/settings/capacity matching accepted037 and nonzero expected phases/replays. No optimization or coalescing/deletion of waits is authorized by diagnostic preparation; a later change keeps original actual-service correctness/performance gates. Source preparation begins in an isolated worktree; no045 build or runtime action has occurred.


045 early immutable schema is cuda-host-phases-schema-v1.json SHA256 2e5c46e6de022df619f7756487c9474f31a5f5779bba889bdae3c6b4d5c9ad0c. The proposed nine-file source surface uses one internal ggml-base helper so llama/CUDA DLLs share thread-local state. It defines batch/span/aggregate/end/exhausted records,4096-record capacity and262144 per-thread capture budget, explicit identities and overflow sentinel, and completed-span flushing only after the outermost span. Independent schema review is pending before validator implementation. Source work is isolated at base2cd72c0b151aa85f98315d5d380ad7211bb3d949 in Temp/llama-cuda-host-phases-20260923; no045 compilation or runtime action has occurred.

045 offline checker logic is prepared in Temp/llama-cuda-host-phases-static-20260923: static-check-logic-v1.py SHA256 ec84a7cc326ebce9eb67a3bbe39478eac828316c2da4345a4121022a4134b8c1, tests SHA256 1d2e2343ba9e007e553af4eae6f383cc43d679f87b237fa76a28dcc06414f11f, ready-logic-v1.json SHA256 c1354292dd3354a72259617fff89b039bfdfe516520d1706d606668366b8a4ca. Author38 CPU tests and three CLI checks passed: preparation exits0 with explicitly unbound source; receipt and compare modes exit1 as expected while BINDING=None. All fixtures are confined to exclusive output roots; no unexpected failures were reported. Six baseline/resource/body/extraction functions remain AST-identical to the reviewed044 helper, reusing178 bodies and all6993 resource occurrences/6517 symbols/145 Common records without extraction. Immutable author source ready/patch/exact multi-file hashes and independent review remain required before a versioned bound checker or actual comparison.

Root prepared, but did not execute, the canonical045 build wrapper cuda-build-host-phases-progress.py SHA256 6b37684b2177b6d624934bec632088da25e752a849cbcad9ed56e8a0b4ee83de. It verifies every future receipt source hash before/after building and preserves full output/real exit plus eleven-file snapshot. Preparation receipt cuda-host-phases-build-wrapper-prepared.json. Accepted037 continues serving; no new throughput measurement or win.


045 independent early-schema review finds the design viable but requires immutableV2 clarification before validator implementation. V1's generic unknown-ID convention conflicts with valid graph UID0; graph boundness must use a non-null graph pointer. Budget accounting must define used/stored/dropped spans and aggregates precisely, with every suppressed span invalidating evidence at its root flush or an exhaustion marker emitted before that root executes. Scheduler full-graph versus CUDA split-graph identity introduction and invocation-origin/inheritance rules must be explicit. flush_end_us excludes the cost of logging its own end record, which remains perturbation. A wholly buffered aborted final root may disappear without any batch marker; parser success alone cannot establish completeness without successful external command receipts and expected graph coverage. These are schema/evidence corrections, not runtime failures or approval of unfinished C++ code. Author is preparing a versioned schema and preservesV1. No045 build/extraction/service action has run.


045 schema V2 is independently approved for validator implementation only. Schema cuda-host-phases-schema-v2.json SHA256 717e2d087e52b28acd27b1cb23ba6d739f5937d6a9a32295f1915a90f262eb98 and review cuda-host-phases-schema-review-v2.json SHA256 1f5b1d5dfb80cf87aa4afcc28f1cafe990713b229c8f444c89ebc29396f79878 close all five V1 ambiguities. Wire version remains1; allocation_failure is unconditional fatal evidence. Valid graph UID0, full/split graph origins, early unbound children, exact cumulative reserved-span/aggregate accounting and end-record residual logging cost are explicit. A wholly missing aborted final root remains an external successful-run/expected-coverage gate, not a parser-solvable condition. No C++ or runtime approval follows this schema review.

045 static-helper V1 independent review is BLOCKED by two preparation defects: static-tests-logic-v1.py rejects a new reviewer-owned output directory outside the author's root, so its prescribed suite invocation exited1 before running any tests; static-check-logic-v1.py rejects safe root-level .gitignore, one of the actual nine source paths. Review cuda-host-phases-static-logic-review-v1.json SHA256 3b54cce4c8dc0b533d76bda67b9e2dad2a5df275466d7608cfdb4c3f2d3a8c37 preserves the real exit and complete output. Independent three main-mode probes still verified prepare0 and unbound receipt/compare1 without extraction; eight unsafe-path probes rejected correctly. Six comparison functions are AST-identical to reviewed044, and the cached178 bodies/6993 resource occurrences/6517 symbols/145 Common records verify. The author's38 tests are not represented as an independently rerun suite. Versioned corrections must accept an explicit new absolute output root, confine all fixtures there, and admit safe single-component files while preserving unsafe-path rejection. V1 remains preserved; no candidate comparison has run.

045 source V1 is prepared at base2cd72c0b151aa85f98315d5d380ad7211bb3d949. Patch cuda-host-phases-source-v1.patch SHA256 21b4b96a4ddf2e83e4413accf696d4ba138559681e37c216a0ff180ce1dfba94 and ready cuda-host-phases-source-ready-v1.json SHA256 6bed801ac3a62fcb72a32619d4f1099c0bb5f5e6320b9bc6dc182375643d49e7 bind all nine source files and byte-exact reconstruction through a separate index. Author25 CPU model/source checks passed; these do not compile or execute the C++ recorder. Existing CUDA_CHECK/stream expressions, graph-statistics calls and15 decode return expressions were preserved by those source checks. Preparation records include three unavailable-rg failures (exit1, corrected with scoped Python reads) and overbroad reads whose displayed output was truncated (reread separately). Logger callback reentry and whole-root abort completeness are explicit limitations for independent review. Source review, versioned static-helper correction/binding and durable CPU validator implementation are now separate assigned tasks. No045 build, extraction, GPU or service run has occurred. Accepted037, its four wins, and the latest five-run service medians remain unchanged:512 ingestion420.3906/output36.7906tok/s;4096 ingestion533.1483/output34.5727tok/s.


045 corrected static-helper preparation preserves unbound logicV2 and a separate source-V1-bound helper. Ready-bound-source-v1.json in Temp/llama-cuda-host-phases-static-20260923 has SHA256 a115a98c0aa7deaa88703876fe363ebf9f4e43f4bf287d3ae30eb99b7f063297; bound helper c879d387aa22a03675eb6026fa5c502ac4298562b93c53d970221872c1d12671, unbound logicV2 874c3661fa6583f84a98323e65e158803acdaef99eef6ea04dbcf019d58f1a50, testsV2a126717e377025698bb4cfbeb4d41e50b4b2b6dd16637fd14c69ab132544009ba. The first corrected-suite run exercised external output roots and .gitignore successfully but had one fixture setup error among39 tests: creating an already existing root-file parent lacked exist_ok=True. That failed test version, outputs and exit1 remain preserved. A separate test-only V2a correction passes39/39. Six comparison functions remain AST-identical to044; cached baseline unchanged; bound-source preparation exits0 and verifies all nine files/ready/patch hashes without extraction. Independent corrected-helper review and actual build-receipt preflight remain pending.

045 independent source review cuda-host-phases-source-review-v1.json SHA256 2939772485fb2d8f939a8be188923de29365887e79a42ed8e4b4695511108823 passes specification and quality for the canonical common logger, permitting compilation only. It independently reconstructs all nine files, reruns25 CPU source/model checks and inspects10 helper exports/60 declaration-call sites/45 scopes/six wire events. No added CUDA call, stream evaluation, wait, ownership/cache decision or execution/result change was found. The actual permitted early-child phase table is empty: pre-binding calls including cache destruction do not reach any newly traced child hook. Custom reentrant logging callbacks remain unsupported. Flush spans measure producer formatting/callback/enqueue/backpressure, not asynchronous sink completion. The independent proof first failed on cp1250 decoding of UTF-8 source byte0x88; corrected explicit UTF-8 reads passed with both receipts preserved. Root also prematurely attempted reading the not-yet-published verdict once (missing-file exit1), then consumed the completed review; that failed read was not review approval.

Root applied exactly the reviewed source patch from clean build HEAD6c85ef5bb025f8df3a0662b796c8915b483caefd after git apply --check. Receipt cuda-host-phases-build-source.json SHA256 caacc64708a5d6466050b58eeacabc85ba33a8ed4b910432632c8811370fa658 distinguishes this build HEAD from author sourcebase2cd72c0b151aa85f98315d5d380ad7211bb3d949 and binds all nine source hashes, review and canonical wrapper. Original committed files plus the immutable reverse patch provide source rollback; served binaries were untouched. The new ignore entries admit only the two recorder files; git check-ignore -v confirms an actual seeded ggml/src/*.tmp remains ignored. The derivative runtime launcher is prepared only: cuda-host-phases-service.ps1 SHA256 d7120a0e65000e2778e60bd58155e28586f7f429c845f3919e3f60605d001b23 clears diagnostic flags, enables only host phases plus canonical graph statistics and retains lazy reserve. No service launch has run.

045 canonical build monitor84593 completed0 in47.070592s; its terminal output was collected. Full output and actual exit are cuda-build-host-phases-build.txt/.exit.json. Warnings include existing-source conversion/exception/nodiscard messages and a failed optional UI asset HTTP download; the final server link and build exit succeeded. The wrapper verified all source hashes both before and after compilation and saved an exclusive eleven-file snapshot tools/llamacpp-cuda-host-phases with manifest cuda-host-phases-binary-hashes.json. CUDA DLL SHA256 bfdc6afeca934f39c86a13f6e0845d5eca772eb33cbaf0022c42e0187c1c4ea7; test-backend-ops.exe619c927267acc115f87f2e99bae4e22fc369b2d4b5b13d0ab58cc9ec71d397f1; unchanged test source e9e01e08ab7446025244b0f540d43a18d85efdad7316e2f8b97e6c557716eb04. Compilation is not retained-device-code or service correctness evidence. Corrected-checker/receipt review, offline compiled comparison and reviewed CPU parser remain gates before the one planned diagnostic service sequence. Accepted037 still serves; no fifth win or new throughput claim.


045 corrected-helper independent review now passes. cuda-host-phases-static-review-v2.json SHA256 631db406172f0ea4a6030d5fc6a796e4a5520a2471ef26310dc4157b911ee275 independently runs39/39 CPU tests plus five negative probes, closes both original blockers and verifies nine source bindings with unchanged comparison functions. Separate cuda-host-phases-static-review-v2-actual-review.json SHA256 9d8702a5149aa5d7482a43f56fd277a4e60c0eba2eb985486d4d41ca3e28fd09 validates the actual receipt with exit0 and zero extraction, retaining the distinction between actual build HEAD and author sourcebase. The reviewer independently verifies candidate DLL identity and publishes the exact next comparison command in cuda-host-phases-static-review-v2-compare-argv.json SHA256 185d20bb6fa91e113a88f813e5c7dd440a37de1adff268caf74406b5bf3c88ac. Root launched that one offline comparison as session94063; compiled invariance remains pending until its real exit and retained artifacts are inspected. No GPU/service execution is implied by this authorization.


045 offline compiled comparison session94063 completed0 in38.135774s and its terminal output was collected; cuda-host-phases-static-compiled1-command.exit.json retains the actual command/result. Bound helper provenance cuda-host-phases-static-compiled1-provenance.json in the static helper Temp root has SHA256 f282a6c368e1189b47629bc936f5870ebf4bd42bd6343e59651cea5fe5f551bf. Both cuobjdump invocations exit0 with empty stderr. All6993 resource occurrences/6517 unique symbols/145 Common records match, with zero additions/losses. All178 retained bodies compare without mismatch. Resource output SHA256 cd124814ce81afcbf4cf19372ae02bc40832909955012444ac3189e72e38aca1 and selected SASS output2ccb3f473c15f3574ad5156abd6fc0b757b56217657c2ef4b53d6b9cbbd16507 are identical to the cached baseline outputs. This comparison covers retained normalized ordered opcode/operand bodies and full resource metadata, not every symbol's SASS or host behavior. Independent actual-evidence review is assigned; the reviewed CPU parser is still outstanding, and no045 GPU/service action has occurred.


045 durable validator V1 is prepared for independent review in Temp/llama-cuda-host-phases-validator-20260923 at base64357072e061d6737c252b19eb6793c9d5ea4993. Ready cuda-host-phases-validator-v1-ready.json SHA256 3f814edc406cbc6190ad34e74092b268867c10f392075d90ba3866182580856d and full patch cuda-host-phases-validator-v1-source.patch SHA256 2261e167eaae49c2ff7e85764980651cfd0bfd83e77f997c5cc307a5e32b548c bind two new Python files with exact fresh-base reconstruction. Author70 CPU tests pass;44 phase/reason pairs match45 source hooks, including the independently confirmed empty early-child table. Synthetic CLI evidence passes structural checks while explicitly leaving external coverage not established. The parser distinguishes host wall unions/exclusive spans from CPU/GPU time and logging producer emission from asynchronous sink completion. No actual log has been validated yet.

Validator preparation preserves expected test-first missing-implementation import failure (red1 exit1) and a later direct-mode fixture failure (test1 exit1): removing the launch span initially left a child parent ID unmapped, which the validator correctly rejected. Corrected fixtures and expanded checks subsequently pass64 then70 tests. The first failed test's CLI child streams were held only in memory and cannot be reconstructed as original evidence; its outer stdout/stderr/exit remain, and test2/test3 explicitly retain complete child streams and exits. This retention gap is disclosed rather than backfilled. Independent parser review and fault probing are assigned before root integration or service use. No045 GPU/service run or fifth win has occurred.


045 independent actual compiled-evidence review passes specification and quality: cuda-host-phases-compiled-review-v1.json SHA256 a415cc6b25bb8bafbb9b7b9e1d820eb480f61bd89f7510f086397a4b098aa891. Read-only raw reparse confirms178 retained bodies/126456 instruction slots plus6993 resource occurrences/6517 symbols/145 Common records, all unchanged, with no additions/losses. It rehashes all eleven snapshot files and nine source files, verifies both extraction exits0/empty stderr and the real wrapper exit0, and keeps actual build HEAD6c85ef5b distinct from later ledger commits. No extraction is rerun. Parser review remains the runtime gate; an independent read-only coverage plan is also checking expected requests,146 benchmark evaluations and whole-root logging completeness limits before the planned observation. No service test or new speed result.


045 validator V1 independent review requires changes: cuda-host-phases-validator-review-v1.json SHA256 7f665ed9326ebf7d62543ee66c8a7e92fea1a6168be4322a625fed4f0882f4e8. The independent70-test suite passes, but nine additional CLI probes (three positives, four confirmed corruptions and two exploratory cases) reveal four false approvals with real exit0. A D2H logits request of993280 bytes accepts a direct CUDA child of4 bytes; the same request accepts an H2D set_tensor_async child. A successful decode accepts zero input tokens with a positive-token ubatch, and another accepts outputs2/tokens1. Consistently regenerated aggregates hide none of these source-level contradictions; the parser lacked immediate request/child transfer semantics and decode-level result/metadata accounting. Reproduction logs, results and complete streams/exits remain in Temp/llama-host-phases-validator-review-20260923. Two other probes are not treated as proven blockers because padding and synchronous fallback require explicit source-specific analysis.

The reviewer verifies exact two-file patch reconstruction, all nine immutable C++ source hashes and45 hooks/44 phase-reason pairs. Its initial source-reader setup failed on Windows cp1250 decoding after the completed70-test run; versioned explicit UTF-8 reading corrected the proof without rerunning the suite or changing the parser. Author preparation failures and the original child-stream retention gap match their disclosures. A versioned two-file validator fix is assigned, with the four actual corruptions and valid multi-ubatch/failure/unwind/encode-fallback/zero-size/synchronous-transfer cases required. C++ source and built snapshot remain unchanged; parser approval and actual service observation remain pending. No new win or benchmark result.


045 external coverage plan V2 is cuda-host-phases-external-coverage-plan-v2.json SHA256 532e5bf5304168c45613c9306195b932e4c42ae00e0e92bdb0926f6b50a058bc. Read-only pinned037 rechecks confirm four responses/128 generated tokens,146 ordered evaluations (T512x14,T508x4,T4x4,T1x124),128 logits D2H copies of993280 bytes, and1935 kernels in every recorded T1 evaluation. These are historical anchors, not a new045 runtime measurement. The actual run must reconcile every benchmark graphstats execute with complete host compute/submit/ubatch/input/scheduler ancestry and128 logits transfers, exact requests/output, process/binary/source provenance and unchanged four-slot188416 context. Startup must be separately evidenced; prior148 total executes is not a lifecycle constant. Whole final computation omission must fail external coverage even if structural parse passes. Getter multiplicity and asynchronous sink drainage remain explicitly unproven without additional direct evidence. Preserved V1 SHA256 ce91a89a81387169598600335f7e1f1bd889adffce5cc15b01c9a5bbd91fd638 was superseded to remove an unproven decode-parent/ubatch token-sum equality; V2 imposes neither that equality nor146 decode roots. Corrected validator review remains required before runtime.


045 validator V2 is ready for independent delta review: cuda-host-phases-validator-v2-ready.json SHA256 e643d2c8bee1dc2ee881a709778c52e49f9b47d3cdcde52fc1f24641fd3b785b, full patch9bb93d6bc10a280e5a5368e63f137bf734d23b68ebc3bd61f185a269d57254a7 and V1-to-V2 delta6cf465df9c682bab1ad69e930fdccbb745f9ce61c4f1bdec40dca86c2c2b8650. Both original-base reconstruction routes reproduce the exact two files; immutable V1 remains reconstructable. Author107 CPU tests pass. Exact four reviewer corruption logs now exit2, three reviewer positives exit0. The author additionally audits original token-index/output-flag partitioning and unchanged request-size forwarding; valid synchronous copy/wait, CPU/no-CUDA, zero-size, failure/unwind and encode-fallback fixtures are retained. Two previously exploratory corruptions reject under these source-specific guards. Proof cuda-host-phases-validator-v2-proof.json SHA256 37deea27808e5f271c7e3fe918c1b717cee81a582c113ffe10d2ab37c8883db5 records the source anchors; these new rules remain subject to independent review.

V2 preparation failures are preserved: a read-only range overran a source file (IndexError/exit1, corrected bounded read); pre-fix red suite exited1 with10 assertion failures and7 errors; the first post-fix run still had7 fixture errors because result_known emitted True/False instead of canonical wire integers. Explicit integer fixtures then passed105 tests, and final expanded suite passed107. All outer/child streams and real exits are retained. No C++ source or built diagnostic binary changed. Read-only normal-service preflight still observes PID25844/parent12356, wrapper PID10992, four idle188416 slots and original supervisor hash54af3284fda7d5a446f5df8c7a82121444576fff96283d7eea21dbf7362142be. Versioned control045 inspect/stop scripts are prepared only in cuda-maintenance, recorded in cuda-host-phases-maintenance-prepared.json; no supervisor mutation, stop, deployment or045 observation has occurred. Fresh inspection will be required before their eventual execution after parser approval.


045 validator V2 independent review passes for pinned-source structural validation: cuda-host-phases-validator-review-v2.json SHA256 d277bc51179dad2c5656338591a5d053b11cca5da3432cc6ffe110eca27d268e. Independently107 tests, original nine CLI probes and six boundary probes return expected exits; all70 original test names remain. Root applies the exact two-file V2 patch, verifies its hashes and unchanged nine C++/eleven snapshot hashes, and reruns107 tests successfully in0.238s. Integration receipts use cuda-host-phases-validator-root-*; source/runtime completeness remains a separate gate.

045 actual diagnostic maintenance uses inspected normal wrapper10992/launcher12356/server25844, all four slots idle, then guarded literal stop via control045. Test deployment receipt bonsai-deploy-20260923T124717482-5d007654e4b449b0969cb9f14b4ee405.json verifies the stable path. The first monitor-launch tool expression fails with SyntaxError before a host-phases service is registered; monitor inventory confirms no such launch, though its displayed inventory is truncated. Corrected direct launch creates monitor40466 and actual server5456/parent3416. Failure receipt cuda-host-phases-launch-preparation-failure1.json preserves this preparation error; no benchmark rerun occurred. Durable pre/post provenance commands both exit0 and bind the live process, eleven deployment files, loaded core backing files and four idle188416 slots. Provenance SHA256 values are2ff8bd7581af3315891b11cde047ef79bdca0dba95070b87294f355e3ece852e and5432965d5f411dfc8c3141c76799f436f3ad84a94fbe0c691bc9cf3562f471f3.

The single planned benchmark session45663 completes0 in34.496195s and is collected. cuda-service-host-phases.jsonl SHA256 a1bdf6acf237b4ae21d42784b8998bee2548e28109417f1990db255b776d2886 records all four512/4096 requests,32 output tokens each. Canonical no-performance-gate comparison passes exact requests/content/token IDs against037, including warmups:128 tokens. Correctness artifact SHA2561669a1335b5b2987e82c2d68ae20ac4dc24bc0d4e32c56a31336966f32674fcc. Diagnostic-only measured rates are512 ingest399.1895/output35.2553 and4096 ingest537.2448/output33.1101tok/s; tracing perturbs execution and these are not accepted throughput gains or normal-service speed reports.

The prebenchmark log prefix is1942019 bytes/6999 host marker lines. After benchmark success and idle provenance, root saves a9751378-byte snapshot with32025 host marker lines, exact prefix preserved: cuda-host-phases-service-snapshot.log SHA25626c051d40bbd3387a55de9c89118f6d139a8974ac23b106dff1e413d26b846e9. Acquisition receipt SHA2567defc118dcc9ac5f6bde7670b76c0ffa6d78245afd3690503f681999f8f42b2d; run binding SHA2560952efdb854ceb94f08a29dcdf39529f042bf7c9c2b87a7a31aca2316c47df93. Structural validation FAILS: actual child exit2 (outer PowerShell exit1), line171 thread identity/batch sequence. Failed observations SHA256d85a2953e18cf48e4732c70e50eb7ff7afac9d36f3b9dad1b026aef334c7ac31 and full command streams remain. This is not complete timing evidence. Initial read-only diagnosis finds the first logged frame is thread2/batch4/used9/spans7-9 and investigates startup fit log filtering; strict parser rules remain unchanged. Independent raw suffix counts already find146 evaluations and128 logits transfers, but structural/attribution approval is withheld.

Root intentionally stops inspected diagnosticPID5456 only after snapshot, collects monitor40466 exit1 as intentional termination, redeploys accepted037 through bonsai-deploy-20260923T125133617-04b586637e3d4cd7869fa9c66d036702.json, and restores the exact original supervisor. Normal service nowPID24668/parent17128/wrapper25244; health/model identity/four idle188416 slots and all eleven accepted deployed hashes verify. Restoration receipt cuda-host-phases-restored-service.json SHA256707f399a2453c9630d7cd2b59799ae91af11b56b0f1d6fd82552173d62f51459 explicitly limits Session0 loaded-module/command-line claims. The9.75MB log remains immutable for diagnosis; no second capture is yet authorized by completed preparation. Four retained wins and accepted speeds remain unchanged.


045 capture 1 independent external review passes request/provenance/raw coverage checks but remains structurally incomplete: cuda-host-phases-service-external-review-v1.json SHA256 a8caa89ac0318e88095ff0466caadfe4875b6d1448ad27b2ab63ae5ccb09633d. The reviewer verifies 51 evidence files, all 11 diagnostic binaries, nine loaded module backing files, unchanged process identity and four idle 188416-token slots. All four requests and 128 output tokens match reference037 exactly. The immutable prefix contains two separately evidenced startup evaluations; the benchmark suffix has the expected ordered 146 evaluations and 128 logits transfers of 993280 bytes. Final getter/end records precede request release and idle messages. These raw inventories do not establish valid ancestry, timing attribution, whole-capture completeness or a speed improvement. Two reviewer read commands failed on PowerShell JSON UTF-8 BOMs (chunk0bfd70 and session30903/chunkd8b223); utf-8-sig decoding corrected the read without changing evidence or rerunning the service.

Read-only diagnosis is complete: cuda-host-phases-runtime-diagnosis-v1-final.json SHA256 ca188aa6465a0cf0c471e73701376ab5388baf112bd3048c55fdde4d4a731ff0 and independent cuda-host-phases-log-filter-review-v1.json SHA256 8ec5c28622f2c99ff800ce79fd0620ee4ac4ca4e0f174b782643587a5c8def3e agree. The first retained frame is thread2/batch4, used9, spans7-9: three startup batches and six spans are absent. All 3316 retained batch headers have no interior sequence gap, but that cannot repair the missing prefix. Fit converts INFO messages to DEBUG at the current verbosity4, and the common logger discards them after the producer has consumed IDs. Verbosity5 admits all tracer records affected by this specific filter. The original strict parser still rejects the original log; no prefix skipping, counter reset, synthetic records or parser weakening is allowed. Two diagnosis console reads failed (encoding alias and cp1250 output) and remain in cuda-host-phases-runtime-diagnosis-v1-read-failure*.json.

Capture 2 is preregistered as a logging-configuration correction, not a performance retry. The canonical runner will gain an appended optional diagnostic verbosity parameter defaulting to4, with explicit5 only for a fresh capture2 wrapper/log. Existing C++ instrumentation, diagnostic binary snapshot, strict validator and external coverage plan V2 remain unchanged. After independent runner review, use the same single four-request 512/4096, 32-output-token benchmark, exact reference037 comparison, pre/post live provenance, immutable prebenchmark prefix, full snapshot parser and external ancestry/coverage review. Restore accepted037 immediately after retaining the capture. Source proof of the filter correction does not establish sink drainage or future capture completeness; those limits remain. The normal service still runs accepted037, and the retained win count remains four.


045 capture 2 launcher correction is implemented and independently approved. scripts/bonsai-server-run.ps1 appends CudaLogVerbosity, integer range0-5/default4, requires an explicit diagnostic log when supplied, and substitutes that value only in the existing diagnostic logger argument. Patch SHA256 5ff328e3587248f18882f03fc02ef0271007509ce4531f0027b991bd8a43f2a3; integrated source SHA256 826309237fa01e5eb65c3884c7d1a1aa7289630d06ad694d532d3bb243309c00. Author and independent reviewer each pass 30 CPU mock cases across PowerShell5.1 and7.6.5. Defaults, explicit5/0, invalid values, early no-log rejection, existing-log preservation and environment restoration on normal/failure paths pass. Independent review cuda-host-log-verbosity-review-v1.json SHA256 9e8846e1c651ca087539ff6263d92e6a4c32068fb355ccd25e36deb6dd8c6a11 verifies exact clean-base reconstruction and only the three intended edits. Original author harness runs failed at unavailable Get-FileHash reporting and PS7 null/empty environment-fixture semantics; versioned helper corrections retain both failures and do not alter runner semantics.

Fresh control045b inspection/stop scripts bind last-observed normal wrapper25244/launcher17128/server24668 and require fresh identity/idle verification. The new cuda-host-phases-capture2-service.ps1 SHA256 d5023bf075a07811a062d525e77986c71631c8d10f454abe4a9168a658b20a44 adds only explicit verbosity5 and a new log path to the previous diagnostic invocation. Preparation receipt cuda-host-phases-capture2-prepared-v1.json records the exact files; no C++ rebuild or parser change is needed. This tool correction is not a fifth throughput win.


045 capture 2 completes with the reviewed launcher verbosity5 correction. Fresh control045b inspection verifies normal wrapper25244/launcher17128/server24668 and idle slots before guarded stop. Diagnostic deployment bonsai-deploy-20260923T131319448-be5704d9a520438b995938226c28e300.json launches monitor70886, server14484/parent3128 at the stable path. The live command line confirms --log-verbosity5. Durable pre/post provenance exits0 and verifies identical process identity, 11 diagnostic files, nine loaded core backing files and four idle 188416-token slots; hashes 584647550c2671badc45c1a1e8d083fb2e560edb3281f5aa3e3c33f7389e35fe and 876f4fd3cd0a40c6264b3953850439e394a73dfe5c1c86c30081175713693591. Run binding cuda-host-phases-capture2-run-binding.json SHA256 554dfdf2286ae2f3be28977ddc1eba5929ce7f39874489057ee672b368ae881b pins the unchanged source/parser/reference/coverage plan and corrected runner.

The 2267448-byte immutable startup prefix passes the unchanged strict parser (exit0). The single benchmark session12529 completes exit0 in34.210610s, with all command streams and terminal output collected. cuda-service-host-phases-capture2.jsonl SHA256 3f6c68dce953a885ec377d163b0c2a657165eb4ee69b6dbdb58602f7ecf393cb records the same four requests/128 output tokens, including warmups. Canonical comparison passes exact requests, content and token IDs: cuda-host-phases-capture2-correctness.json SHA256 ab140b2f36c9e6b54fef86e072c43d2a2fcc9cf0125cbaebd37c3686d2caef59. Diagnostic-only measured rates are512 ingest389.2814/output35.3396 and4096 ingest545.3506/output33.6289tok/s. The comparison requests no performance gate; logging perturbs execution, so these rates neither replace normal-service speeds nor establish a regression/win.

After idle postbenchmark provenance, the retained full snapshot is10318433 bytes: cuda-host-phases-capture2-snapshot.log SHA256 a50e3ebda4a8e124cf813ffbbae3f098bed5a497ae6b02c414158b9b9b531a2f. It preserves the exact startup prefix. Acquisition receipt SHA256 a44f9b524316da1bf3da33d783f06e138f50201f93b650094a0b2dba9a2a7710. Full unchanged structural validation exits0 in1.342277s, reports3319 batches/15806 spans, including the startup frames absent in capture1. Observations SHA256 330f19768f89ab1d431a3f087eb3718d0de80c6b8a0667b3f84af783eca61061. Structural success alone does not prove external coverage, sink drainage, CPU-active time or GPU-idle attribution; independent external review and read-only host-cost analysis are now assigned. Capture1 remains a preserved failed experiment.

Inspected diagnosticPID14484 is stopped only after retention and idle check. Monitor70886 exit1 is intentional termination and its terminal output is collected; monitor retention truncates, but the full canonical snapshot remains. Accepted037 is redeployed through bonsai-deploy-20260923T131638299-cdf7b7bf18a6456eb57a5bd18319e001.json, and the exact original supervisor is restored. Normal servicePID5596/parent16468/wrapper25836 verifies health/model identity, four idle188416 slots and all11 accepted stable-file hashes. Restoration receipt cuda-host-phases-capture2-restored-service.json SHA256 3975a6cd8b42ff0f48ec477aca61e11c2bd48a2854c6a70a8b0ab153753fb0bb retains the sequential/Session0 provenance limitations. No fifth retained throughput win.


045 capture 2 independent external and structural review passes for the frozen workload: cuda-host-phases-capture2-external-review-v1.json SHA256 f6d8ae96bbe9b6c5d163618ae99880723c0446f07d1efe473f8c9f3b1c1a53ab. An independent full-log parser run exits0 and exactly matches root observations. Batches1-3319 and spans1-15806 are contiguous; startup occupies862 batches/2689 spans, with benchmark13117 spans partitioned by actual request boundaries. Every146 benchmark compute has matching decode/ubatch/scheduler ancestry, submit/launch identity and ordered graphstats correspondence. Modes are26 direct,4 capture and116 replay, with two startup evaluations evidenced separately. All128 logits transfers have the expected993280-byte child, and all four responses/128 generated tokens remain exact. No recorded drops, exhaustion or allocation failure. Restoration supplement SHA256 c6af6cc8e4c07415781a919699890145a28880ba687d3c61ea14cca76e3e3311 verifies accepted037 restoration. Scope remains bounded: trailing getter multiplicity, asynchronous sink drain, relocated module memory, CPU-active time and GPU-idle attribution are not established; no diagnostic throughput win is claimed.

Root verifies all nine C++/build/ignore source hashes and both validator source hashes still exactly match their independently approved manifests before retaining the inert diagnostic and strict validator in git. No intervening source changes justify another build or duplicate test pass: the existing compiled-kernel equality check,107-test validator gate and actual service capture bind these same bytes. DEBUG_CUDA_HOST_PHASES remains off by default; accepted037 remains the deployed service build. The next read-only analysis uses only validated host-wall intervals and request partitions to identify a specific optimization hypothesis. Four retained throughput wins remain unchanged.


045 host-cost attribution finds no supported >=2% optimization at both prompt lengths. Provisional author verdict cuda-host-phases-capture2-attribution-v1-verdict.json SHA256 a4fd29fc11a9fed99e4cf92b4f679c312772e1374e00ddec1d2433bd3daf3db1 is under independent arithmetic/source review. Across the31 output evaluations per measured request, repeated getters excluding each step's longest required readiness wait have optimistic complete-erasure envelopes0.108%/0.127%; input-copy synchronization0.162%/0.166%. Required queued GPU completion cannot be counted as removable work. One/two recurrent checkpoint serializations measure42.259/66.344ms of D2H API wall intervals,96/192 copies and149.625MiB per checkpoint. Erasing even the entire copy interval covers only3.354%/0.892% of respective measured prefill windows; coalescing waits alone covers0.186/0.465ms. These are diagnostic observed-cost bounds, not causal speed predictions.

The author corrects its first prefill residual calculation: ctx0/attempt0 roots already recorded checkpoint copies and must count in the root union. Original artifacts remain; versioned supplement SHA25636f152f7b17c24049451035fc6021203bfe1372d47fd618e210b2db62029f33a fixes the partition. Corrected measured512/4096 prefill windows are1302.095/7506.084ms, root unions1251.613/7296.714ms, logger intervals18.407/34.452ms and uninstrumented residuals32.075/174.918ms. Output windows876.984/921.433ms partition into root unions821.361/869.485ms, logger intervals36.520/31.922ms and residuals19.103/20.026ms. These host-wall intervals are not CPU activity or GPU idle time; no clock mapping to Nsight is used.

common/common.cpp:2307 initializes fresh checkpoint storage before serialization overwrites it. Allocation/initialization is unmeasured, and the residual includes unrelated costs, so no saving is assigned and no source optimization follows. Instead the next bounded discovery targets existing fused PQ2 MMVQ: cached accepted037 records40 calls/~5.702ms per4K output step. That measured category has enough cost to investigate, but is not itself a proposed speedup. The next source-specific hypothesis must exclude previously rejected variants before compilation.

Attribution preparation failures remain recorded: a successful guide fetch failed when a Unicode arrow was printed through cp1250; corrected escaped output succeeded. A web guide open was unavailable, followed by successful direct read. Five exploratory console reads exceeded display limits without changing authoritative saved inputs. One read of nonexistent src/llama-sampling.cpp failed; bounded tracked-path discovery located src/llama-sampler.cpp. Both final offline analysis commands exit0. No GPU run, service mutation or fifth win follows this analysis.
