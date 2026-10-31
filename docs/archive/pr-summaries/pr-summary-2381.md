## Summary

`detect_output_squash_mismatches_with_cost_hint` no longer emits a candidate with a `NaN` or `+∞` `estimated_improvement`, and its final sort (plus two sibling sorts) now uses `total_cmp`. Closes #2381.

The CI follow-up also isolates runtime-directory test bases in `src/analysis/utils/platform.rs` to fix a Coverage test race.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- A `NaN` mean absolute error loses `mean_error < MIN_MEAN_ERROR_THRESHOLD`, so the neuron was kept and Strategy 1 pushed a `NaN` gain. 40 finite `3e38` errors overflow the `f32` sum to `+∞`, giving a `+∞` gain. Rejecting `!mean_error.is_finite()` at the gate closes both (CWE-697 / CWE-682).
- `partial_cmp(..).unwrap_or(Equal)` is not a total order under `NaN`; `total_cmp` is, as in every sibling detector.

### Essential Design Decisions

- The neuron is skipped (fail closed), not clamped: a non-finite mean error carries no usable signal.
- The issue comment asked for `total_cmp` in `squash_weight_rescale.rs` and `weight_magnitude_reset.rs` too. Their fallbacks were already dead (inputs filtered finite), so this is a behaviour-neutral conversion and has no test that can go red.
- With the gate in place no `NaN` can reach the `output_squash_mismatch.rs` sort, so the `total_cmp` there is defence in depth; the regression test guards the gate, not the comparator.

### Undiscoverable Facts

- None.

## Evidence

Regression test: `tests/issue_2381_output_squash_mismatch_non_finite_gain.rs::non_finite_error_emits_no_non_finite_candidate`.

- Base (`output_squash_mismatch.rs` restored from `HEAD`): all 3 tests fail; the emitted candidates carry `estimated_improvement: NaN, mean_error: NaN` and `estimated_improvement: inf, mean_error: inf`, and the ranking test returns `["out-lo", "out-nan", "out-hi"]`.
- Head: `cargo test --test issue_2381_output_squash_mismatch_non_finite_gain` passes 3 of 3.

```mermaid
flowchart LR
    R[Output records] --> M[mean absolute error]
    M --> G{finite and at least 0.05?}
    G -- no --> S[skip neuron]
    G -- yes --> C[Strategies 1-3]
    C --> T[total_cmp sort]
```

Original trigger closed: a `NaN` error or an overflowing error sum makes `mean_error` non-finite, which `!mean_error.is_finite()` now rejects before any strategy runs. When `mean_error` is finite every `|error|` is finite, and `mean_error * clipping_fraction * 0.5` cannot exceed `f32::MAX`, so no equivalent bypass reaches the gain. Strategies 2 and 3 already compared against `mean_error` and fail closed.

## Test Plan

- `tests/issue_2381_output_squash_mismatch_non_finite_gain.rs` (new): `non_finite_error_emits_no_non_finite_candidate` (positive control, one `NaN`, all `3e38`), `finite_candidates_rank_by_descending_improvement` (NaN neuron dropped, two finite neurons in descending order), `neg_infinity_and_infinity_errors_are_skipped`. All three go red on the base file and green at head.
- Passed at head: `cargo test --test issue_2381_output_squash_mismatch_non_finite_gain` (3), `issue_2300_chunk_08a_neuron_sweep` (10), `issue_2281_chunk_08a_shared_sweep` (4), `issue_2280_chunk_08a_ledger_scaffold` (6), and the other six `chunk_08a` ledger pinning tests (the ledger executor reported them passing); `cargo test --test detection squash` (21 passed).
- Not covered by a red test: the two sibling `total_cmp` conversions (dead fallbacks, behaviour-neutral).
- Docs updated: `docs/DISCOVERY_TYPES.md` (Output Squash Mismatch Detection, new criterion 4), `docs/audits/security-sweep-chunk-08a-detection-neuron.md` (#2381 row, bullet and list marked fixed), and `docs/audits/security-sweep-chunk-13-config-env.md` (platform file line count, inventory rows 11, 12 and 15, symlink-check row and disposition 3).

- CI follow-up (#2485): format `HARD_TANH` as inline code in the regression test's doc comment. `cargo clippy --test issue_2381_output_squash_mismatch_non_finite_gain -- -D warnings` reproduced `clippy::doc_markdown` before the correction and passed afterwards.
- CI follow-up (#2485): `xdg_runtime_dir_rejects_symlink` failed in Coverage when another test's temporary `TMPDIR` override was removed while it was in use. `src/analysis/utils/platform.rs` now passes an explicit base to the two tests that previously changed `TMPDIR`. Production still uses `std::env::temp_dir()` through the wrapper, so its runtime-directory behaviour is unchanged.
- CI follow-up (#2485): `test_apply_xdg_runtime_dir_sets_when_unset` sets `TMPDIR` to a persistent `neat-ai-discovery-tmpdir-wiring-<uid>` base, restores it afterwards, and asserts the exact `expected_runtime_dir(base)` path to pin the wrapper's `std::env::temp_dir()` wiring.
- Platform test plan: run the `xdg_runtime_dir` library tests with all features to cover the explicit-base rejection cases and the existing runtime-directory behaviour.
- CI-fix validation: all 3 regression tests passed with `--all-features`. The full gate passed its shell checks, dependency audit, debug build, formatting, all-target/all-feature Clippy and type checks.

<!-- vibe-quality-gate-skipped reason="timeout 900 ./quality.sh expired during library tests (exit 124); the full test suite, documentation build and release build remain incomplete" -->
