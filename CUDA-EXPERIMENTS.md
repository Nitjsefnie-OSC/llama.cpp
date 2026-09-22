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
