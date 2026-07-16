## Summary

Detects the *sample-vs-creature disconnect* failure mode — candidates whose
`improved_count / total_count` ratio is at or above 0.95 while the
aggregated `actual_error_reduction` is non-positive — and feeds the
offending `(change_type, target_squash, variant_key)` triple into the
calibration correction so subsequent predictions for that combination are
demoted faster than the standard EWMA would on a single failure. Closes
#1195.

The new layer halves the surviving correction per detected disconnect,
clamped to the existing
`[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]` (= `[0.001, 1.0]`)
calibration bounds so a stream of failures cannot collapse predictions to
zero. A structured `SampleCreatureDisconnect` event is emitted via the
observability surface added in #1194 whenever the detector fires.

## Evidence

This is a backend/calibration change — there is no UI to screenshot. The
behaviour is verified end-to-end by the new
`tests/issue_1195_sample_creature_disconnect.rs` integration suite (10
tests) and the unit-test layers inside the new modules.

```mermaid
flowchart TD
    A[Failure cache entry] --> B{improved_count/total >= 0.95?}
    B -->|no| Z[Standard EWMA path]
    B -->|yes| C{actual_error_reduction <= 0?}
    C -->|no| Z
    C -->|yes| D[Emit SampleCreatureDisconnect event]
    D --> E[Apply 0.5x penalty to (change_type, target_squash, variant_key)]
    E --> F[Clamp cumulative penalty to 0.001..1.0]
    F --> G[correction_for_triple returns demoted value]
```

Quality gate: `./quality.sh` passes cleanly (`cargo build`, `cargo fmt`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --lib --tests --all-features`, `cargo doc`,
`cargo build --release --lib`).

## Test Plan

- New integration suite `tests/issue_1195_sample_creature_disconnect.rs`:
  - `detector_fires_for_high_ratio_and_negative_reduction` — happy path.
  - `detector_silent_for_high_ratio_and_positive_reduction` — guards
    against false positives when per-sample gain aggregated.
  - `detector_silent_for_low_ratio_and_negative_reduction` — guards
    against firing on candidates already governed by the EWMA layer.
  - `event_emitted_when_detector_fires` /
    `event_not_emitted_when_detector_silent` — observability emission
    contract.
  - `replays_failure_cluster_and_demotes_calibration_entry` — replays
    the three-failure cluster pattern from #1189/#1195, asserts the
    detector fires for each entry, and verifies that the
    `(add-neurons, SINE, v2_add-neurons_*_e0_e-3_e0)` calibration
    triple is demoted via `disconnect_penalties_as_map()`.
  - `triple_lookup_strictly_below_pair_when_above_floor` — confirms
    the new triple-aware lookup demotes below the existing pair lookup
    when the EWMA correction is above the floor.
  - `cumulative_penalty_clamped_to_floor` — 20 disconnects in a row
    clamp to `MIN_CALIBRATION_CORRECTION` rather than collapsing to
    zero.
  - `unrelated_triple_unaffected_by_penalty` — disconnect against one
    triple does not bleed into adjacent (squash, variant) combinations.
  - `legacy_entries_without_counts_do_not_record_penalty` — backward
    compatibility for cache entries that omit per-sample counters.
- New unit tests inside
  `src/analysis/scoring/sample_creature_disconnect.rs`:
  - boundary cases (exactly at threshold, just under, zero total,
    corrupt counters, non-finite reductions).
- New unit tests inside
  `src/observability/sample_creature_disconnect.rs`:
  - event construction, emission helper, and camelCase JSON
    serialisation.
- New unit tests inside
  `src/analysis/scoring/calibration_correction.rs`:
  - `improved_and_total_counts_parsed_from_top_level_fields` and
    `_optional_for_legacy_entries` confirm the new `improvedCount` /
    `totalCount` fields round-trip through the failure-cache JSON
    schema.
- All existing calibration tests (#1131, #1162, #1163, #1192, #1194)
  continue to pass — the new penalty layer is additive and only fires
  when both per-sample counters are populated.
