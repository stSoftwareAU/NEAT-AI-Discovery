# PR summary — Issue #2244 (chunk 9d-2: GPU queue `scheduling.rs`, `executor.rs` and `mod.rs`)

## Summary

Audits `src/analysis/gpu/queue/scheduling.rs`, `executor.rs` and `mod.rs` at the chunk-9 baseline (`a7c3f651`) and records the result in the `queue-core` regions of `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`: the `Files swept` outcomes (plus the test-only `fake_evaluator.rs` and `wedge_tests.rs` they declare), one sub-section per file with checks 1–5, one `## Ledger` row, the queue-core `## Refuted / not findings` rows, and the `## Outcome` / `## Issues filed` entries. Documentation only — no source or test changes. Closes #2244.

This PR **files** one finding and fixes none: #2361 (CWE-755, low) stays open for its own fix issue.

- **Finding #2361.** A panic in `gpu_thread_loop` skips the exit signal the "Always signal exit" comment promises. The in-flight caller and later submitters get an `Err` at once, but requests already queued are stranded until the stall window or batch timeout, then report a wedge and trip the breaker. `Drop` discards the panic payload.
- **Check 4 (CWE-400): refuted.** The work queue is `bounded(get_work_queue_capacity())` (4, 8 or 16) and every other channel is `bounded(1)`. One caller holds the GPU thread for at most one request budget (≤ 300 s); a queued caller's own deadline ends its wait with a typed error. A full queue does not fail fast — that is #2339, already filed by #2243.
- **Check 5.** No production `unwrap`, `expect`, `.lock()` or `Mutex` in the three files; `fake_evaluator.rs` is test-only. The init-timeout path trips the breaker and returns `gpu_wedged_error`; init `Disconnected` returns an `Err` at once; a failed `shutdown()` send is backed by `Drop`'s abandoned-thread accounting. Thread/wedge lifecycle is cross-referenced to #2115.
- **`mod.rs` check 1.** `GpuFuture::collect` never turns a timed-out or abandoned request into an empty or zero `Ok`; `with_deadline` only stores the deadline.
- The last commit corrects three baseline citations: `from_caller_timeout` is `budget.rs:54`–`:56` (delegating to `from_caller_timeout_at` at `:61`–`:65`), the `mod.rs` HEAD offset is +3 for baseline L57–L59 and +6 from L60, and the depth refuted row now cites `memory.rs:655`–`:668` like the check-4 body.

## Evidence

Documentation-only audit, no UI.

- `cargo test --all-features --test issue_2088_sweep_ledger_contract --test issue_2288_chunk_09_ledger_scaffold --test issue_2243_chunk_09d_queue_core_sweep_test`: 17 passed. The #2243 test rejects any queue-core row still reading `pending — #2114`.
- `markdownlint-cli2 docs/audits/security-sweep-chunk-9-gpu-wgsl.md`: 0 issues.
- `./quality.sh < /dev/null` on the head commit: `✅ All quality checks passed!`
- #2361 carries the `finding-id` and `cwe` markers, the five house-format sections, and the labels `security`, `lang:rust`, `severity:low`, `confidence:medium`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The `queue-core` section has sub-sections for `scheduling.rs`, `executor.rs` and `mod.rs`, each with a verdict for checks 1–5, or `N/A — reason` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `#### src/analysis/gpu/queue/scheduling.rs`, `…/executor.rs`, `…/mod.rs` (Issue #2244) — reviewer: met
- **met** — Check 4 gives a fairness and depth-bound verdict (CWE-400: finding or refuted) with line numbers — evidence: `scheduling.rs` sub-section "Verdict (CWE-400): refuted" (`scheduling.rs:38`–`:40`, `memory.rs:655`–`:668`, `submission.rs:28`–`:34`) — reviewer: met
- **met** — Check 5 lists every `unwrap`/`expect`/lock site as production or test, and states the GPU-thread panic outcome for waiting callers, with lines — evidence: `scheduling.rs` check-5 site table and "GPU-thread panic path" (`scheduling.rs:62`, `:72`–`:73`, `:166`) — reviewer: met
- **met** — Refuted candidates are in the refuted table, each with its refuting line — evidence: `## Refuted / not findings` queue-core region, `(Issue #2244)` rows — reviewer: met
- **met** — Every surviving finding has its own house-format issue, linked from a `## Ledger` row — evidence: ledger row `open — #2361`, `## Issues filed` entry, #2361 body checked with `gh issue view 2361` — reviewer: met — reason: the reviewer confirmed the ledger link but could not see #2361's body; its markers, sections and labels were checked here
- **met** — `./quality.sh` passes — evidence: `./quality.sh < /dev/null` on the head commit printed `All quality checks passed` — reviewer: partial — reason: the reviewer could not run the gate and judged from the diff only; it was run here and passed

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942): the new audit prose cites `file:line` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` queue-core sub-sections, Ledger and Refuted rows — reason: stands. The issue requires line numbers re-verified at the baseline, the chunk-9 record pins baseline `a7c3f651` and every earlier slice uses the same convention, and most citations also name the symbol.
- **clean** — Australian English; edits confined to the queue-core regions plus the shared Outcome and Issues-filed lists, as in earlier slices; no `src/`, CI, dependency or `Cargo.toml` changes; no PR-summary misplacement. Optional note: `L##` and `:##` styles are mixed within some rows (cosmetic).

## Test Plan

- No new tests: the issue scopes this slice as documentation only. #2243's test pins the empty-vs-zero behaviour, and #2116 adds per-file verdict completeness.
- Ran the three ledger contract suites above (17 passed), markdownlint, and `./quality.sh`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
