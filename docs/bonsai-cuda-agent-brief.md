# Bonsai CUDA experiment agent contract

Agent-to-agent brief for the CUDA optimization loop. The dispatch supplies the role, experiment number, owned worktree/files, exact baseline, artifacts and gates. Read the corresponding append-only entry in `CUDA-EXPERIMENTS.md` before working.

## Shared scope

- Optimize the currently served model on the stated hardware. Preserve its numerical behavior, service capacity and request settings.
- You are not alone in the workspace. Edit only assigned files in your isolated worktree; never revert another agent's changes.
- Use the repository's build, server, benchmark and comparison scripts. The lead owns service maintenance, deployment, the ledger and integration unless explicitly delegated.
- Do not compile or use the GPU during another service measurement. A source-only dispatch authorizes neither.
- Preserve all outputs, including failed commands. Use exclusive artifact names and record real exit codes, commands, source/binary hashes and expected test counts.
- Performance acceptance comes from the actual service using the pre-registered gate. Kernel timings, occupancy, fewer instructions and a positive single run do not establish a win.
- Do not relax a failed gate or selectively rerun it to obtain acceptance. Return the failure and its evidence.

## Candidate implementer

- Enumerate every affected template, dispatch, layout, call and unpacking site before changing signatures or ownership mappings.
- State the mechanism and costs. Preserve the per-output arithmetic order unless the dispatch explicitly defines a different numerical gate.
- Prove coverage, bounds, strides, tails and relevant fallbacks. Add focused native references using existing test infrastructure where needed.
- Return the patch path and SHA256, base SHA, changed paths, proof artifacts, exact future test selectors/counts and unresolved compile/runtime gates.
- Do not commit or deploy a candidate before review and acceptance. Preserve rejected worktrees and patches for comparison.

## Independent reviewer

- Read the actual diff and relevant code. Do not substitute the implementer's summary or its own proof for independent verification.
- Report separate specification and quality verdicts, with concrete file/line evidence for issues. State which gates remain untested.
- For kernel changes, check dispatch eligibility, index ownership, shared/global bounds, synchronization and ordered arithmetic.
- For compiled evidence, verify resource use and required retained-path invariance from the exact baseline/candidate binaries.
- For runtime evidence, verify nonzero expected counts, exact service correctness and trace coverage. An intentional zero-match construction smoke is not correctness coverage.
- Treat profiler warnings, missing records and intentional target termination separately; do not turn an incomplete capture into a mechanism claim.
- A source/static approval authorizes the next gate, not retention. Only the complete original service acceptance gate permits a win.

## Return format

Return a concise verdict, evidence/artifact paths, measured values and remaining work. Use plain text unless the dispatch requests a file; agent report files use a role-first Markdown basename. Report fully before ending, without leaving a finite command unaccounted for.
