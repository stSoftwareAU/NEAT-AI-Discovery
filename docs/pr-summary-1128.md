## Summary

Enforce the coordinated-structural minimum expected-gain floor **after** all
post-merge discounts (module boost, ensemble penalty, diversity reranking,
synapse-analysis calibration), plugging the gap that let a 1.17e-7 candidate
reach the FFI response despite PR #1115's 1e-5 pre-merge filter. Closes #1128.

## Problem

The GRQ-sampler failure cache
`discovery/failures/6d8a5b6a/coordinated-structural/v2_coordinated-structural_29584442...json`
captured a coordinated-structural candidate with
`expectedCreatureScoreGain: 1.17e-7` that caused a post-apply `scoreDelta` of
-0.0019 (harming the network). PR #1115 introduced the 1e-5 floor but applied
it only before the post-merge stages, so downstream discounts could still
drive an above-floor candidate into the documented 1e-7/1e-8 noise range.

## Audit findings

Three gaps identified in the coordinated-structural emission pipeline:

1. `merge_coordinated_structural_replacements` (single-op branch) retained
   candidates on `> 0.0` with no lower bound.
2. `apply_module_boost_to_candidates` (0.5–2.0× clamp) and
   `apply_ensemble_scoring` (0.7× disagreement penalty) ran *after* the
   pre-merge 1e-5 filter with no follow-up sweep.
3. `apply_post_processing` filtered coordinated candidates on `> 0.0` *after*
   applying `COORDINATED_PREDICTION_CALIBRATION` (5e-5×), letting sub-noise
   gains survive on the synapse-analysis-internal path.

## Fix — dual-floor architecture

- **Pre-merge 1e-5 floor** (`COORDINATED_MIN_EXPECTED_GAIN`, unchanged from
  PR #1115) — strict gate before discounts are applied in the dispatch paths.
- **Post-discount noise floor** (`COORDINATED_POST_DISCOUNT_NOISE_FLOOR = 5e-7`,
  new) — final sweep applied in `analyze_all` after module boost, ensemble
  scoring, and diversity reranking. Catches the Issue #1127 noise range
  (1e-7 to 1e-8) while preserving legitimately-discounted borderline
  candidates (e.g., 9.95e-7 from the hidden-neuron collapse regression
  fixture) that cleared the pre-merge floor before downstream calibration.

Metadata (`candidates_returned`) is refreshed after the sweep so the JSON
output remains consistent with the returned arrays.

## Changes

- `src/analysis/constants/candidate_scoring.rs` — added
  `COORDINATED_POST_DISCOUNT_NOISE_FLOOR = 5e-7` with rationale doc comment.
- `src/analysis/candidate_aggregation.rs` — new
  `apply_coordinated_gain_floor` helper; no change to the merge-stage
  semantics (single-op still `> 0.0`, multi-op still `> 1e-5` after discount).
- `src/analysis/orchestration.rs` — final sweep call after module-boost and
  diversity reranking, with metadata refresh.
- `src/analysis/mod.rs` — re-export `apply_coordinated_gain_floor` for the
  integration test.
- `src/analysis/synapse/post_processing.rs` — documented why the `> 0.0`
  retain is preserved here (the orchestration sweep is the FFI-facing gate).
- `tests/coordinated_min_gain_floor.rs` — new regression test (5 cases) that
  fails against the pre-fix pipeline.
- `tests/neuron/issue_164_redundant_path_pruning.rs` — weakened two
  assertions: pure structural-simplification candidates with near-zero numeric
  improvement now legitimately filter below the noise floor; detection logic
  itself is still covered by `detect_redundant_paths` unit tests.

## Evidence

No UI changes. Backend-only filter and constant.

- `./quality.sh` passes locally (release build + full test suite).
- `cargo test --test coordinated_min_gain_floor` — 5/5 pass.
- `cargo test --test synapse` — 171/171 pass (including the previously
  failing `coordinated_structural_can_collapse_hidden_neuron_to_single_synapse`,
  `collapse_hidden_neuron_with_identity_squash`, and
  `single_op_with_small_gain_accepted`).
- `cargo test --test neuron` — 45/45 pass.

## Test plan

- [x] Regression test in `tests/coordinated_min_gain_floor.rs`:
  - `apply_floor_removes_below_and_retains_at_or_above` — helper behaviour
    at the floor, below, in the #1127 noise range, and well above.
  - `module_boost_discount_below_floor_is_filtered` — single-op 8e-7 →
    post-boost 4e-7 → filtered by final sweep.
  - `merge_filters_two_op_candidate_discounted_below_floor` — 2-op 1.5e-5 ×
    0.5 (2-op discount) = 7.5e-6 < multi-op floor → filtered at merge.
  - `merge_filters_single_op_candidate_below_floor` — 5e-7 single-op filtered
    by the pre-merge discovery-dispatch 1e-5 gate.
  - `floor_pass_filters_existing_candidates_in_syn` — 1.17e-7 internal
    structural-pattern candidate filtered by the final sweep.

## Pre-PR security self-check

- [x] No new external input surface; constant + filter only.
- [x] No secrets staged.
- [x] No new SQL/shell/filesystem/HTTP calls.
- [x] No new user-facing output; internal error handling unchanged.
- [x] No new dependencies.
