# PR Summary — Issue #2346

## Summary

Closes #2346

`detect_correlated_error_patterns` used to allocate an `n_outputs × n_outputs`
correlation matrix with no ceiling (CWE-789), and nothing could cancel its pair
scan or the per-group `find_predictive_inputs` search once they had started.
This change adds plain constant ceilings (`MAX_CORRELATED_ERROR_OUTPUTS = 1_000`,
`MAX_CORRELATED_ERROR_INPUTS = 10_000`) and checks the analysis deadline in both
loops. The deadline check also covers global cancellation. Every skip is logged
and returned as a typed `CorrelatedErrorSkip`.

- [x] Output ceiling checked before the matrix is allocated
- [x] Deadline checked once per matrix row
- [x] Input ceiling plus a per-input deadline check in `find_predictive_inputs` (the scope extension in the trusted comment)
- [x] Wired into the real dispatch caller (`structural_specs.rs`)
- [x] Regression test, docs, version bump (0.74.274 → 0.74.275)

## Spec

### Intent and Rationale

- An untrusted `CreatureJson` with a very large number of error-carrying outputs could make the analysis allocate n² `f32`s. The allocation is now refused before it happens.
- The scan was O(n²·S) and could not be cancelled, so it held a dispatch worker past the analysis deadline.
- A trusted comment on the issue says the fix must cover both loops: the pair scan and `find_predictive_inputs`.

### Essential Design Decisions

- The ceilings are plain `pub const`s, not env vars, because of the dead-levers rule in AGENTS.md.
- A new `detect_correlated_error_patterns_with_deadline` returns `CorrelatedErrorScan { groups, skip }`. The old function stays as a thin wrapper that passes `&None`, so existing callers and tests are unchanged.
- The deadline is checked per matrix row and per candidate input, reusing `deadline_passed`. There is no hand-rolled clock check.
- If the deadline passes during the matrix build, the function returns no groups. If it passes during the group search, the groups already completed are kept.

### Undiscoverable Facts

- `detect_discovery_modules_parallel` (`src/analysis/discovery_dispatch.rs:668`) already skips a module whose deadline passed before the module started. The new checks only matter once the scan is running.
- `deadline_passed` returns true on global cancellation as well as on a wall-clock deadline.

## Evidence

```mermaid
flowchart TD
    A[structural_specs correlated-error closure] --> B[detect_correlated_error_patterns_with_deadline]
    B --> C{eligible outputs over MAX_CORRELATED_ERROR_OUTPUTS?}
    C -- yes --> S1[warn, skip OutputCeilingExceeded, no matrix]
    C -- no --> D[build matrix row by row]
    D --> E{deadline_passed per row?}
    E -- yes --> S2[skip DeadlinePassed, no groups]
    E -- no --> F{inputs over MAX_CORRELATED_ERROR_INPUTS?}
    F -- yes --> S3[warn, InputCeilingExceeded, groups without predictive inputs]
    F -- no --> G[find_predictive_inputs per group]
    G --> H{deadline_passed per input?}
    H -- yes --> S4[DeadlinePassed, keep completed groups]
    H -- no --> R[groups returned]
```

**Docs sweep** — grep: `detect_correlated_error_patterns`, `find_predictive_inputs`, `CorrelatedError`, "correlation matrix", "correlated error", `append_structural_specs` (README.md, `docs/` excluding `docs/archive/`, `*/README.md`); section: `docs/discoveries/correlated-error.md#-how-we-detect-it`, `docs/DISCOVERY_TYPES.md#correlated-error-pattern-detection`, `docs/ANALYSIS_DEEP_DIVE.md#-correlated-error-pattern-detection-issue-344`; updated: `docs/discoveries/correlated-error.md`, `docs/DISCOVERY_TYPES.md`, `docs/ANALYSIS_DEEP_DIVE.md`

- `docs/discoveries/correlated-error.md` and `docs/DISCOVERY_TYPES.md` each gain a new Bounds section.
- `docs/ANALYSIS_DEEP_DIVE.md:281` — the "Skip optimisation" note listed only the single-output skip. It now also names the output ceiling, the input ceiling and the deadline stop.

Every other line the grep terms hit at the head was read and is still true:

- docs/COST_FUNCTION_NOTES.md:124 — still true because it audits how the function reads `errors` (presence / first residual), which this change does not touch.
- docs/COST_FUNCTION_NOTES.md:125 — still true because it audits how the function reads `errors` (presence / first residual), which this change does not touch.
- docs/COST_FUNCTION_NOTES.md:331 — still true because it audits how the function reads `errors` (presence / first residual), which this change does not touch.
- docs/ANALYSIS_DEEP_DIVE.md:176 — still true because it describes redundant-path anti-correlated gradients, a different module this change does not touch.
- docs/ANALYSIS_DEEP_DIVE.md:268 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/ANALYSIS_DEEP_DIVE.md:276 — still true because the matrix is still built pairwise over all eligible outputs at or below the new ceiling; the ceiling is documented in the adjacent Bounds/Skip text.
- docs/ANALYSIS_DEEP_DIVE.md:287 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/DISCOVERY_TYPES.md:44 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/DISCOVERY_TYPES.md:193 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/DISCOVERY_TYPES.md:1085 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/DISCOVERY_TYPES.md:1089 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/DISCOVERY_TYPES.md:1096 — still true because the matrix is still built pairwise over all eligible outputs at or below the new ceiling; the ceiling is documented in the adjacent Bounds/Skip text.
- docs/DISCOVERY_TYPES.md:1172 — still true because it describes redundant-path anti-correlated gradients, a different module this change does not touch.
- docs/PRIOR_ART.md:114 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/PRIOR_ART.md:202 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/output-conflict.md:151 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/README.md:143 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/README.md:188 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:1 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:9 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:32 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:45 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:63 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:177 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- docs/discoveries/correlated-error.md:214 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- benches/cache_locality.rs:49 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- benches/sample_locality.rs:50 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- src/analysis/constants/sample_thresholds.rs:23 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:1 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:3 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:7 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:11 — still true because the matrix is still built pairwise over all eligible outputs at or below the new ceiling; the ceiling is documented in the adjacent Bounds/Skip text.
- src/analysis/detection/correlated_error.rs:21 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:90 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:107 — still true because the legacy wrapper keeps its signature and still returns sorted `CorrelatedErrorGroup`s.
- src/analysis/detection/correlated_error.rs:119 — still true because the legacy wrapper keeps its signature and still returns sorted `CorrelatedErrorGroup`s.
- src/analysis/detection/correlated_error.rs:224 — still true because the matrix is still built pairwise over all eligible outputs at or below the new ceiling; the ceiling is documented in the adjacent Bounds/Skip text.
- src/analysis/detection/correlated_error.rs:591 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/detection/correlated_error.rs:597 — still true because it names or describes the correlated-error module or its output, which this change does not alter.
- src/analysis/module_dispatch_specs/structural_specs.rs:30 — still true because the closure still pre-checks `output_count < 2` before loading records.
- src/analysis/detection/redundant_path.rs:13 — still true because it describes redundant-path anti-correlated gradients, a different module this change does not touch.
- src/analysis/detection/redundant_path.rs:157 — still true because it describes redundant-path anti-correlated gradients, a different module this change does not touch.
- src/analysis/detection/redundant_path.rs:241 — still true because it describes redundant-path anti-correlated gradients, a different module this change does not touch.
- tests/integration.rs:841 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/integration.rs:892 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/integration.rs:944 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/integration.rs:951 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/integration.rs:1169 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/integration.rs:1176 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/issue_2182_recommendation_core_non_finite_gain_ranking.rs:249 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/analysis/issue_340_discovery_categories.rs:20 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/analysis/issue_221_sample_locality.rs:106 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/analysis/issue_221_sample_locality.rs:439 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/analysis/target_map_optimization.rs:52 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_343_bottleneck_neuron_detection.rs:240 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:1 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:5 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:9 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:66 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:87 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:230 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_344_correlated_error_detection.rs:339 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/issue_940_unwrap_removal.rs:109 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/issue_940_unwrap_removal.rs:112 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/issue_940_unwrap_removal.rs:146 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/detection/issue_360_opposing_synapse_detection.rs:1175 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/gpu/gpu_activation_shaders.rs:230 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/infrastructure/issue_196_cache_locality_benchmark.rs:113 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/recommendation/issue_1249_categorical_error_sse_gating.rs:225 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.
- tests/regression/regression_v0_1_123.rs:89 — still true because it is a test/bench fixture comment describing generated data or test intent, unaffected by the ceilings and deadline checks.

## Acceptance Criteria

The issue has no explicit Acceptance Criteria heading. The spec-reviewer derived the criteria below from the requested fix and the required test.

Provenance: independent `spec-reviewer` subagent, given `/tmp/review-2346.diff` and the issue body with its comments.

- Output ceiling checked before the matrix allocation, and the skip is recorded — reviewer: met
- Deadline threaded into the pair loop — reviewer: met
- Fix wired into the real dispatch caller — reviewer: met
- `find_predictive_inputs` gets an input ceiling and a deadline check (trusted comment) — reviewer: met
- Regression test `rejects_output_count_above_ceiling_without_building_the_matrix` — reviewer: met
- Fails loud, not silent — reviewer: met
- Docs updated — reviewer: met
- Version bump (AGENTS.md rule) — reviewer: met

Reviewer found no scope creep. It judged the in-module unit test to be traceable to the scope extension.

## Standards Review

This repo has no `CODING-STANDARDS.md`, so the review was done against `CONTRIBUTING.md` and `AGENTS.md`.

Provenance: independent `standards-reviewer` subagent, given the same diff.

- **Missing PR summary file** (CONTRIBUTING.md "PR Summary File") — fixed. This file adds it.
- **`MAX_CORRELATED_ERROR_OUTPUTS` / `MAX_CORRELATED_ERROR_INPUTS` are `pub` but only an integration test uses them** ("Do not make APIs public just for testing") — declined, for two reasons:
  - Each ceiling is part of the module's reported contract. `CorrelatedErrorSkip::OutputCeilingExceeded { ceiling }` / `InputCeilingExceeded { ceiling }` returns the value to production callers.
  - The repo already publishes ceilings this way for tests to import, e.g. `analysis::scoring::weights::MAX_OUTGOING_WEIGHT` in `tests/scoring/issue_888_weight_constraints.rs`.
- **Optional, not adopted:**
  - Taking the deadline by value instead of `&Option<SystemTime>` — kept, to match `deadline_passed` and the existing call sites.
  - Doc wording repeated across three places — kept, because each surface stands alone.

Checked with no violation: version bump, Key Invariants, dependency discipline, CI untouched, Australian English, test style, no timing assertions.

## Test Plan

**Regression test:** `tests/issue_2346_correlated_error_matrix_guard_test.rs::rejects_output_count_above_ceiling_without_building_the_matrix`

- The test reproduces the original trigger: `MAX_CORRELATED_ERROR_OUTPUTS + 1` error-carrying outputs.
- It asserts that no groups are returned and that `skip == OutputCeilingExceeded`.
- It fails against the unfixed code and passes after the fix. The original trigger — an unbounded n_outputs² allocation and an uncancellable scan — is closed with no trivial bypass:
  - the ceiling counts eligible outputs before any allocation;
  - both loops check `deadline_passed`;
  - the only production caller uses the deadline-aware entry point.

**Other tests:** the same file holds five more tests, plus one in-module unit test, `find_predictive_inputs_returns_none_when_deadline_elapsed`.

**Branch outcomes:**

| Location | Outcome | Test that reaches it | Flip went red? |
|---|---|---|---|
| `src/analysis/detection/correlated_error.rs:194` | output ceiling exceeded | `rejects_output_count_above_ceiling_without_building_the_matrix` | Yes. Flipped to `usize::MAX - 1`. |
| `src/analysis/detection/correlated_error.rs:194` | under the ceiling | `small_correlated_output_set_is_not_skipped` | n/a |
| `src/analysis/detection/correlated_error.rs:230` | deadline passed per row | `elapsed_deadline_stops_matrix_build_with_no_input_neurons` | Yes. Flipped to `if false`. |
| `src/analysis/detection/correlated_error.rs:266` | input ceiling exceeded | `input_count_above_ceiling_skips_predictive_search` | Yes. Forced `false`. |
| `src/analysis/detection/correlated_error.rs:266` | under the input ceiling | `predictive_input_found_below_input_ceiling` | n/a |
| `src/analysis/detection/correlated_error.rs:517` | deadline passed per input, returns `None` | `analysis::detection::correlated_error::tests::find_predictive_inputs_returns_none_when_deadline_elapsed` | Yes. Flipped to `if false`. |
| `src/analysis/detection/correlated_error.rs:333` | caller maps `None` to `DeadlinePassed` and keeps completed groups | none | Accepted gap, see below. |

**Accepted gaps:**

- `correlated_error.rs:333` can only be reached if the deadline expires between the matrix build and the group search. A deadline that has already passed trips the per-row check first, so a deterministic test needs a clock seam. That seam is not added here.
- The `&deadline` wiring in `structural_specs.rs:43` has no test that goes red when it is reverted. The dispatch gate pre-checks the deadline before the closure starts, so this wiring only matters mid-scan, which needs the same clock seam.

**Removed assertions:** none. One earlier draft test was dropped before commit because it masked other flips. It never reached the base branch.

**Commands:**

- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test --test issue_2346_correlated_error_matrix_guard_test`: 6 passed.
- `cargo test --lib correlated_error`: 1 passed.
- `cargo test --test detection correlated`: 16 passed.
- `./quality.sh`: passed (exit 0, "All quality checks passed!") on the branch head, with the docs-sweep edits applied.

## Guards kept and call sites checked

- **Guards kept:** the dispatch-level deadline gate, `MIN_SAMPLES_FOR_CORRELATION`, and the `validate_creature` gate at the FFI entry point. All are untouched.
- **`structural_specs.rs:43`:** now calls the deadline-aware function. This is the accepted gap above.
- **Legacy `detect_correlated_error_patterns` callers:** these are tests only. They behave the same through the wrapper.

## Security self-check

- [x] Input validation: the output and input counts from an untrusted creature are bounded before allocation.
- [x] No secrets staged. No new shell, SQL or filesystem surface.
- [x] No new dependencies.
- [x] Errors are logged through `tracing::warn!` without leaking internal state to FFI callers.
