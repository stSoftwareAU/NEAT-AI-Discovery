# PR summary — Issue #2244 (chunk 9d-2: GPU queue `scheduling.rs`, `executor.rs` and `mod.rs`)

## Summary

This PR audits `src/analysis/gpu/queue/scheduling.rs`, `src/analysis/gpu/queue/executor.rs` and `src/analysis/gpu/queue/mod.rs` at the chunk-9 baseline (`a7c3f651`). It records the results in the `queue-core` regions of `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`:

- the `Files swept` outcome for the three files, and for the test-only `fake_evaluator.rs` and `wedge_tests.rs` they declare;
- a sub-section per file with checks 1–5;
- one `## Ledger` row;
- twelve `## Refuted / not findings` rows;
- the `## Outcome` and `## Issues filed` entries.

The change is documentation only, with no source or test changes. Closes #2244.

Results:

- **One finding, filed as #2361** (`SEC-f0d19ede542c`, CWE-755, low).
  - A panic in `gpu_thread_loop` skips the exit signal that the "Always signal exit" comment promises.
  - The in-flight caller and later submitters get an `Err` at once.
  - Requests already queued in the bounded channel are stranded until the stall window or batch timeout. They then report a wedge and trip the breaker.
  - `Drop` discards the panic payload.
- **Check 4 (CWE-400): refuted.** The work queue is `bounded(get_work_queue_capacity())` (4, 8 or 16), and every other channel is `bounded(1)`. No caller can hold the GPU thread for longer than one request budget (at most 300 s). A queued caller's own deadline ends its wait with a typed error. A full queue does not fail fast, but that is #2339, already filed by #2243.
- **Check 5.** The three files have no production `unwrap`, `expect`, `.lock()` or `Mutex`. `fake_evaluator.rs` is test-only, as expected. The init-timeout path trips the breaker and returns `gpu_wedged_error`. The init `Disconnected` arm returns an `Err` at once. A failed `shutdown()` send is backed by `Drop`'s abandoned-thread accounting. Leaked-thread and wedge lifecycle are cross-referenced to #2115, not re-audited.
- **`mod.rs` check 1.** `GpuFuture::collect` never turns a timed-out or abandoned request into an empty or zero `Ok`, and `with_deadline` only stores the deadline.

## Evidence

This is a documentation-only audit change with no UI. The evidence is the record plus test and lint output:

- The ledger contract tests pass on the branch: `cargo test --all-features --test issue_2088_sweep_ledger_contract --test issue_2288_chunk_09_ledger_scaffold --test issue_2243_chunk_09d_queue_core_sweep_test` gives 9 + 4 + 4 passed. `issue_2243_chunk_09d_queue_core_sweep_test` already rejects any queue-core row still reading `pending — #2114`.
- `markdownlint-cli2 docs/audits/security-sweep-chunk-9-gpu-wgsl.md` reports 0 issues.
- #2361 is open in house format, with `<!-- finding-id: SEC-f0d19ede542c -->` and `<!-- cwe: CWE-755 -->`, the five body sections, and the labels `security`, `lang:rust`, `severity:low` and `confidence:medium`.
- The branch has passed the quality gate. The worker re-runs `./quality.sh` before the PR is raised.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The `queue-core` section has sub-sections for `scheduling.rs`, `executor.rs` and `mod.rs`, each with a verdict for checks 1–5, or `N/A — reason` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `#### src/analysis/gpu/queue/scheduling.rs`, `…/executor.rs` and `…/mod.rs` (Issue #2244) sub-sections — reviewer: met
- **met** — Check 4 gives a fairness and depth-bound verdict (CWE-400: finding or refuted) with line numbers — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `scheduling.rs` check 4 ("Verdict (CWE-400): refuted", `scheduling.rs:38`–`:40`, `memory.rs:655`–`:668`, `submission.rs:28`–`:34`) — reviewer: met — the reviewer noted one imprecision: `budget.rs:61`–`:65` is `from_caller_timeout_at`, not `from_caller_timeout`.
- **met** — Check 5 lists every `unwrap`/`expect`/lock site as production or test, and states the GPU-thread panic outcome for waiting callers, with lines — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `scheduling.rs` check 5 site table and "GPU-thread panic path" (`scheduling.rs:62`, `:72`–`:73`, `:163`, `:166`) — reviewer: met
- **met** — Refuted candidates are in the refuted table, each with its refuting line — evidence: `## Refuted / not findings` queue-core region, 12 new `(Issue #2244)` rows — reviewer: met
- **met** — Every surviving finding has its own house-format issue, linked from a `## Ledger` row — evidence: #2361, ledger row `SEC-f0d19ede542c`, `## Issues filed` entry — reviewer: met
- **met** — `./quality.sh` passes — evidence: the branch passed the quality gate, and the worker re-runs it before raising the PR; the ledger contract tests pass locally (see Evidence) — reviewer: partial — reason: the reviewer could not run `./quality.sh` and judged from the diff that the contract test is likely to pass.
- **unrequested** — The `Files swept` rows for `fake_evaluator.rs` and `wedge_tests.rs` moved from `pending — #2114` to "audited, test-only" — reviewer: unrequested — reason: the `fake_evaluator.rs` row answers the issue's "confirm `fake_evaluator.rs` test-only" ask. The `wedge_tests.rs` row goes slightly beyond item 4 but is accurate: both files are declared only under `#[cfg(test)]` in `mod.rs`.
- **unrequested** — The shared `## Outcome` paragraph and `## Issues filed` list were edited — reviewer: unrequested — reason: these edits are outside the regions the issue names. They follow the record's per-slice convention, which #2243 also followed, and the record's own "each slice lists the issues it files here" note.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942): the new audit prose cites `file:line` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (queue-core region, Ledger and Refuted rows) — reason: this stands. The issue explicitly requires baseline line numbers. The chunk-9 record pins baseline `a7c3f651`, and its Methodology requires baseline re-verification. Most line citations also name the symbol. Switching one slice alone would make the record inconsistent.
- **violation** — CONTRIBUTING.md "PR Summary File": the diff carried no `docs/archive/pr-summaries/pr-summary-2244.md` — evidence: `docs/archive/pr-summaries/pr-summary-2244.md` — reason: fixed here, because this file is that summary.
- **violation** — TDD and the chunk-9 per-slice convention of a contract test pinning each slice's verdicts — evidence: no `tests/issue_2244_*` file in the diff — reason: this stands. The issue states that "This slice is documentation only, so it adds no new runtime test". #2116 adds the per-file verdict-completeness test, and the existing #2243 test already rejects a pending queue-core row.
- **violation** — Citation accuracy against the baseline — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `scheduling.rs` check 4 and the starvation refuted row cite `GpuTimeBudget::from_caller_timeout` at `budget.rs:61`–`:65`, which is `from_caller_timeout_at` (`from_caller_timeout` is L54) — reason: this stands in this change and should be corrected in a follow-up. The line range is right and the logic is the same. This turn was limited to the PR summary.
- **violation** — Citation accuracy — evidence: the queue-core preamble says `mod.rs` lines "from L57 on sit 6 lower at HEAD". Baseline L57–L59 sit 3 lower, and only L60 onwards sits 6 lower — reason: this stands in this change and should be corrected in a follow-up. Every cited `mod.rs` line in the sub-sections is a baseline line, and the reviewer verified them.
- **violation** — Internal consistency — evidence: the depth-bound refuted row cites `memory.rs:655`–`:661`, while the check 4 body cites `:655`–`:668` — reason: this stands. `:655`–`:661` is `get_work_queue_capacity_for_tier`, where the values 4, 8 and 16 are defined.
- **clean** — Australian English; no Mermaid blocks; markdownlint clean; edits confined to the queue-core rows and regions plus the shared Outcome and Issues-filed lists, as in earlier slices; SEC id, CWE and severity consistent across ledger, section and Issues filed; no config surface, dead lever, `Cargo.toml` or CI changes; lifecycle concerns handed to #2115 and not re-audited; the reviewer confirmed every other spot-checked baseline citation.

## Test Plan

- No new tests. The issue scopes this slice as documentation only. #2243's test pins the empty-vs-zero behaviour, and #2116 adds per-file verdict completeness.
- `cargo test --all-features --test issue_2088_sweep_ledger_contract --test issue_2288_chunk_09_ledger_scaffold --test issue_2243_chunk_09d_queue_core_sweep_test`: 17 passed.
- `markdownlint-cli2 docs/audits/security-sweep-chunk-9-gpu-wgsl.md`: 0 issues.
- `./quality.sh`: re-run by the worker before the PR is raised.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
