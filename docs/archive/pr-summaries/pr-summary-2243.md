# PR summary — Issue #2243 (chunk 9d-1: GPU queue `submission.rs` and `execution.rs`)

## Summary

This PR audits `src/analysis/gpu/queue/submission.rs` and `src/analysis/gpu/queue/execution.rs` at the chunk-9 baseline (`a7c3f651`) and records the results in the `queue-core` regions of `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`:

- the `Files swept` outcome for both files;
- a sub-section per file with checks 1–5;
- one `## Ledger` row;
- twelve `## Refuted / not findings` rows;
- the `## Outcome` and `## Issues filed` entries.

It also adds `src/analysis/gpu/queue/empty_vs_zero_tests.rs`, which pins whether each empty-input short-circuit can be told apart from an all-zero answer. Closes #2243.

Results:

- **One finding, filed as #2339** (`SEC-1124ca631044`, CWE-400, low). A submission blocked in `send_timeout` waits the whole 60–300 s batch timeout with no heartbeat or breaker check. Its response wait then restarts the same timeout. So a wedge costs the caller the full batch timeout instead of the stall window, and any submission can take up to twice the timeout.
- **Check 1 (empty vs zero).** Four of the seven short-circuits can be told apart from an all-zero answer, because they return one result per set or config. Three cannot:
  - `evaluate_relu_gpu`, `evaluate_activation_gpu`, and `evaluate_activations_batched_gpu` with empty `samples`.
  - These are refuted rather than filed. Every production caller checks `MIN_NEURON_SAMPLE_COUNT` before submitting, so they are unreachable, and an all-zero answer yields no candidate either way.
  - This also corrects the #2241 reading of the trailing `0`: it is `improved_count`, not a sample count.
- **Check 2 (`execute_request`).** No branch leaves a waiting caller without either a send or a returned `Err`. The only `Err` returned to the loop is `is_device_lost_error`. Any other outcome is sent exactly once. The `trace!`-only send failure loses nothing the caller or the breaker needs.
- **Check 3 (byte cap).** The cap limits the *count* of sets per chunk and is clamped to 1, so a single oversized set bypasses it. At the baseline this ends in a wgpu validation panic on the GPU thread (#2314). The panic unwinds, so the caller gets an `Err`, not an abort. PR #2337 on `Develop` turns it into an `Err` before wgpu is reached. No new finding.
- **Check 4.** `send_timeout` on a full queue returns an `Err`, but only after the full timeout (#2339).
- **Check 5.** No `unwrap`, `expect` or `.lock()` in production code in either file.

`fake_evaluator.rs` now returns one all-zero result per sample set or config, which is the shape the real evaluators return. Case (b) of the new test needs this to be a fair all-zero answer.

## Evidence

This is a backend, audit and test change with no UI. Evidence is test output:

- `cargo test --lib --all-features analysis::gpu::queue`: 81 passed. This includes the 7 new `empty_vs_zero_tests` and the existing `wedge_tests` and `stale_skip_tests`, which use the updated fake.
- The ledger contract tests pass: `issue_2088_sweep_ledger_contract`, `issue_2288_chunk_09_ledger_scaffold`, `issue_2113_chunk_09c_device_sweep`, `issue_2237_chunk_09_evaluation_sweep` and `issue_2291_chunk_09a_2b_shader_layer_sweep` (39 passed).
- `cargo clippy --lib --tests --all-features -- -D warnings` is clean. `markdownlint-cli2` reports 0 issues on the record.
- `./quality.sh` was run once after the final change (result in the Test Plan).

```mermaid
flowchart LR
    S[caller: breaker.check] --> Q{empty input?}
    Q -- yes --> Z[short-circuit Ok: empty or zero stats]
    Q -- no --> T[send_timeout, full timeout, no heartbeat check]
    T -- full queue --> F[queue_full_error after 60-300 s, finding 2339]
    T -- sent --> W[await_gpu_response restarts timeout]
    W --> A[Answered / Stalled / TimedOut / Disconnected]
```

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The `queue-core` section has `### …/submission.rs` and `### …/execution.rs` sub-sections, each with a verdict for checks 1–5 or `N/A — reason` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (queue-core region) — reviewer: met
- **met** — The check-1 table covers every empty-input early return in `submission.rs`, each with the yes/no verdict and a baseline line number — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` check-1 table (7 rows, L187–L519) — reviewer: met
- **met** — Check 2 states whether any `execute_request` branch or a dropped response leaves the caller waiting, with the line — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `execution.rs` check 2 (`execution.rs:104`…`:334`) — reviewer: met
- **met** — Check 3 gives an explicit byte-cap verdict, including the single-oversized-sample case and panic vs `Err`, with the line — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `submission.rs` check 3 (`memory.rs:649`, `submission.rs:122`–`:146`) — reviewer: met
- **met** — `empty_vs_zero_tests.rs` exists, is registered in `mod.rs`, covers cases (a) and (b) for every site, and passes — evidence: `src/analysis/gpu/queue/empty_vs_zero_tests.rs` (7 tests), `cargo test --lib --all-features analysis::gpu::queue` 81 passed — reviewer: partial — reason: the reviewer confirmed the file, the registration and the coverage but could not run `cargo test`. It was run here and passes.
- **met** — Refuted candidates are in the refuted table, each with its refuting line — evidence: `## Refuted / not findings` queue-core region (12 rows) — reviewer: met
- **met** — Every surviving finding has its own house-format issue, linked from a `## Ledger` row — evidence: #2339, ledger row `SEC-1124ca631044` — reviewer: met
- **met** — `./quality.sh` passes — evidence: full gate run after the final change (Test Plan) — reviewer: missing — reason: the reviewer saw only the diff and could not run the gate. It was run here.
- **unrequested** — `fake_evaluator.rs` returns one all-zero result per set or config — reviewer: unrequested — reason: the reviewer traced it to criterion 5 and item 3(b) as necessary test plumbing. Case (b) needs a same-shape all-zero answer for the comparison to be fair.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942): the new audit prose cites `file:line` — evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` (queue-core region, Ledger and Refuted rows) — reason: this stands. The issue explicitly requires baseline line numbers and a `file:line` column. The sweep-ledger rules make line citations falsifiable by pinning a baseline commit, and the record's `## Verify this record` diff detects drift. Every earlier chunk-9 slice cites the same way.
- **clean** — tests assert outcomes rather than implementation; in-crate test placement is justified by private `GpuWorkQueue` fields; no wall-clock benchmarking in tests; Australian English; no config surface, `Cargo.toml` or CI changes.

## Test Plan

- Added `src/analysis/gpu/queue/empty_vs_zero_tests.rs`. It has 7 tests, one per `submission.rs` short-circuit, driven through the production `run_work_loop` with `FakeGpuEvaluator`:
  - four assert that case (a) and case (b) are distinguishable;
  - three pin the current identical output, citing the refuted rows;
  - each uses `FakeGpuProbe::calls()` to prove case (a) never reached the device.
- Modified `src/analysis/gpu/queue/fake_evaluator.rs` to return per-set and per-config all-zero answers. The existing `wedge_tests` and `stale_skip_tests` still pass.
- Registered the module in `src/analysis/gpu/queue/mod.rs`.
- `./quality.sh < /dev/null`: see the result recorded below.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
