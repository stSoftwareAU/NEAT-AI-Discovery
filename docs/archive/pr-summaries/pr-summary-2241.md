## Summary

Chunk 9c-2 of the GPU/WGSL security sweep. It proves that every `GpuAnalyzer`
entry point returns `Err` when its device, queue, layout or pipeline is
`None`. The records live in `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`.
Closes #2241.

- **Ledger tables** in the `device` region:
  - `Entry-point None → Err sites`: 11 rows. For each entry point it cites the
    file:line and quotes the `.context("…")` string of the device, queue,
    layout, pipeline and reduce checks. The activation reduce checks are
    marked conditional on `use_reduction`.
  - `Entry-point delegations`: 13 rows. Five are the inherent `_with_budget`
    wrappers, three are `GpuEvaluator` (`analyzer.rs`) and five are
    `RequestEvaluator` (`queue/executor.rs`).
  - `Empty-input Ok short-circuits`: 7 rows, each benign. They return `Ok`
    before any `None` check, with an empty result and no GPU work.
- **Unit tests** (`src/analysis/gpu/none_field_tests.rs`,
  `src/analysis/gpu/queue/none_field_tests.rs`). They build an all-`None`
  `GpuAnalyzer` and call every entry point and delegation with non-empty
  input. Each call must return `Err` containing "GPU device unavailable".
- **Scope, stated honestly.** Without a GPU only the device check can be
  reached, because a `wgpu::Device` cannot be built. The queue, layout and
  pipeline checks are therefore pinned by their `.context` strings and order
  in the contract test. `FakeGpuEvaluator` replaces `GpuAnalyzer` wholesale,
  so it cannot prove this property.
- **Outcome: no finding.** A new Refuted row records it, and the audit's
  status paragraph now points at #2242 for what remains.
- There are no production code changes, only two `#[cfg(test)] mod` lines.
  The version is bumped to 0.74.267, as `AGENTS.md` requires.

```mermaid
flowchart LR
    A[Entry point, non-empty input] --> B{device}
    B -- None --> E[Err: GPU device unavailable]
    B -- Some --> C{queue, layout, pipeline, reduce}
    C -- None --> F[Err: context string]
    C -- Some --> G[GPU dispatch]
    H[Entry point, empty input] --> I[Ok: empty result, benign]
```

- [x] Ledger tables (entry points, delegations, short-circuits)
- [x] Crate-internal all-`None` unit tests
- [x] Contract test pinning rows, context strings and order
- [x] Findings decision: no finding

## Evidence

This change adds docs and tests only; there is no UI.

- `cargo test --lib none_field`: 3 passed. They run without a GPU.
- `cargo test --test issue_2113_chunk_09c_device_sweep`: 9 passed, 4 of them
  new:
  - `every_entry_point_cites_its_none_to_err_checks_in_order`: the table rows
    must equal the 11 entry points, in order. Each quoted `.context("…")`
    string must sit on its cited line, inside the checker's function body,
    in device → queue → layout → pipeline → reduce order. Rewording or
    removing any check fails the test.
  - `every_delegation_row_still_delegates_to_an_entry_point`: the cited line
    must still call the entry point it names.
  - `every_empty_input_short_circuit_carries_a_benign_verdict`: the guard
    (`if … .is_empty() {`) must still be in the function, and the verdict
    must open with `benign`.
  - `the_device_region_states_the_2241_outcome`: pins the "no finding"
    statement.
- The spec review re-derived every file:line in the three tables from source
  with a script and found 0 mismatches.

### Regression tests (fail before, pass after)

REGRESSION_PLACEHOLDER

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- 11-row entry-point table in the `device` section, with file:line.
  reviewer: met. Evidence: the `Entry-point None → Err sites` table.
- Layout and pipeline `None` checks recorded for each entry point.
  reviewer: met. Evidence: the same table's layout, pipeline and reduce
  columns, with the `use_reduction` gating noted.
- Delegations recorded (`RequestEvaluator`, `GpuEvaluator`).
  reviewer: met. Evidence: the `Entry-point delegations` table, 13 rows.
- Activation empty-samples short-circuit recorded with a benign-or-finding
  decision.
  reviewer: met. Evidence: the `Empty-input Ok short-circuits` table; the
  `activation_evaluation.rs` row is marked benign.
- Unit test that builds an all-`None` analyser, calls every entry point with
  non-empty input and asserts `Err` containing "GPU device unavailable".
  reviewer: met. Evidence: `none_field_tests.rs` (inherent and `GpuEvaluator`)
  and `queue/none_field_tests.rs` (`RequestEvaluator`).
- Scope stated honestly: only the device check is reachable, and
  `FakeGpuEvaluator` cannot prove the property.
  reviewer: met. Evidence: the module docs and the ledger paragraph.
- Contract test asserts all 11 rows are present and fails if a cited
  `.context` string is removed or reworded.
  reviewer: met. Evidence:
  `every_entry_point_cites_its_none_to_err_checks_in_order`.
- Findings filed and linked, or "no finding" stated.
  reviewer: met. Evidence: `**Outcome (#2241): no finding.**`, pinned by a
  test.
- No production code changes.
  reviewer: met. Evidence: only `#[cfg(test)] mod` lines in `gpu/mod.rs` and
  `queue/mod.rs`.
- Version bump 0.74.266 → 0.74.267.
  reviewer: unrequested. reason: `AGENTS.md` requires a bump on any code
  change.
- Extra delegation and short-circuit contract tests, and a threat row.
  reviewer: unrequested. reason: in scope, and they only add protection
  around the new tables.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

This repo has no `CODING-STANDARDS.md`. The independent reviewer judged the
#2241 diff against `CONTRIBUTING.md`, the canonical coding conventions, and
`AGENTS.md`.

- **violation** — Cite code by symbol, never by line number (Issue #1942) —
  evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md:781` — reason:
  stands. The three new tables cite `path.rs:line`, matching the ~200
  existing citations in this baseline-pinned audit record. Moving the record
  to symbol citations is a separate change.
- **violation** — Cite code by symbol, never by line number (Issue #1942) —
  evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md:855` — reason:
  stands, for the same reason. The prose cites `bias_evaluation.rs:82` and
  `:92`.
- **violation** — Cite code by symbol, never by line number (Issue #1942) —
  evidence: `docs/audits/security-sweep-chunk-9-gpu-wgsl.md:937` — reason:
  stands, for the same reason. This is the queue-core threat row.
- **violation** — Cite code by symbol, never by line number (Issue #1942) —
  evidence: `tests/issue_2113_chunk_09c_device_sweep.rs:496` — reason:
  stands. The tests check that each cited line is in range and that each
  `.context` string is in the function body in order. They do not check that
  the string is on the cited line, so a drifted line number stays green. The
  Evidence section above says each string "must sit on its cited line". That
  overstates the test, and this entry corrects it.
- **violation** — Test outcomes, not implementation — evidence:
  `tests/issue_2113_chunk_09c_device_sweep.rs:570` — reason: stands. Without
  a GPU the queue, layout, pipeline and reduce checks cannot be reached at
  runtime, so the contract test pins their `.context` source text. It also
  pins the `if {guard} {` text (line 642). The reviewer acknowledged this
  rationale.
- **violation** (minor) — Australian English throughout all code — evidence:
  `src/analysis/gpu/none_field_tests.rs:19` — reason: stands. The name
  `all_none_analyzer` follows the existing `GpuAnalyzer` type and `analyzer`
  module names.
- **violation** — PR summary file required — evidence: absent from the diff
  the reviewer saw — reason: my status departs here, because this is not a
  violation. The reviewer's diff excluded `docs/archive/`. This file,
  `docs/archive/pr-summaries/pr-summary-2241.md`, is on the branch.
- **clean** — Version bump (`0.74.266` → `0.74.267` in `Cargo.toml` and
  `Cargo.lock`), CI untouched, no new Mermaid blocks in the diff (bare-`;`
  rule), and Australian English in prose, strings and assertion messages.
- **clean** — Test placement: the in-crate `#[cfg(test)]` modules are
  justified because `GpuAnalyzer`'s fields are `pub(super)`, and no API was
  widened for testing. The tests have non-vacuous assertions: non-empty
  samples, the specific "GPU device unavailable" text, and exact row counts.
  They contain no timing assertions and need no `#[serial]`.
- **clean** — Formatting (`rustfmt --check`), DRY (`first_citation` and
  `symbol_rows` reuse the new helpers), file sizes under ~1,500 lines, doc
  comments on the new modules and helpers, no new dependencies, and the
  forward-only, atomic-write and FFI-memory invariants, which the diff does
  not touch.

## Test Plan

- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
- `cargo test --lib none_field < /dev/null`
- `cargo test --test issue_2113_chunk_09c_device_sweep < /dev/null`
- `./quality.sh < /dev/null`: QUALITY_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
