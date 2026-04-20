## Summary

Calibrates the per-prediction factors using actual outcomes from a supplied failure cache. Closes #1131.

The compiled `NEURON_PREDICTION_CALIBRATION`, `SYNAPSE_PREDICTION_CALIBRATION`, and `COORDINATED_PREDICTION_CALIBRATION` constants are global averages. A single creature can drift from that average — for example, one whose recent failure cache shows ~1000× over-estimation for `add-neurons` should discount that class more than the global constant does. This change computes a per-change-type correction factor from a creature's failure cache and applies it multiplicatively to the base calibration constant before prediction scoring.

### Changes

- **New module** `src/analysis/scoring/calibration_correction.rs` defining:
  - `FailureCacheEntry { change_type, expected_error_reduction, actual_error_reduction }` (camelCase at the FFI boundary).
  - `CalibrationCorrection::from_failure_cache(&[FailureCacheEntry])` which groups entries by `change_type`, skips zero-predicted / non-finite ratios, computes an EWMA of `actual / expected` with `α = 0.3`, and clamps to `[MIN_CALIBRATION_CORRECTION = 0.001, NEUTRAL_CORRECTION = 1.0]`.
  - Stable change-type keys: `add-neurons`, `add-synapses`, `coordinated-structural`.
- **Input plumbing** — `failure_cache: Option<Vec<FailureCacheEntry>>` added to `AnalyzeParallelInput`, `AnalyzeAllInput`, `AnalyzeSynapsesInput`, and `AnalyzeNeuronsInput` with `#[serde(default)]` for backwards compatibility.
- **Synapse post-processing** (`src/analysis/synapse/post_processing.rs`) — builds the correction once from the input failure cache and applies it multiplicatively to `SYNAPSE_PREDICTION_CALIBRATION` (helpful + harmful) and `COORDINATED_PREDICTION_CALIBRATION` before invoking the logistic / flat calibration.
- **Neuron post-processing** (`src/analysis/neuron/post_processing.rs`) — same pattern, applied to `NEURON_PREDICTION_CALIBRATION`.
- **Metadata** — `calibration_corrections: HashMap<String, f32>` added to both `SynapseAnalysisMetadata` and `NeuronAnalysisMetadata`, and surfaced on the FFI JSON types (`SynapseAnalysisMetadataJson`, `NeuronAnalysisMetadataJson`, skipped when empty).

### Safety characteristics

- **Never inflates predictions** — upper clamp of 1.0 means the correction can only ever discount.
- **Never collapses to zero** — lower clamp of 0.001 preserves exploration for change-types that have been consistently over-estimating.
- **Deterministic** — same cache in, same corrections out; verified by an integration test asserting byte-equal correction maps across two runs.

## Test plan

- [x] Unit tests in `src/analysis/scoring/calibration_correction.rs` (9 tests): empty cache → neutral, all-negative → min, uniform cache returns mean ratio (20 entries of `0.000001/0.001` clamped to `0.001`), mixed cache → expected EWMA, zero-predicted skipped, non-finite ratios skipped, change-types independent, correction above 1.0 clamps to neutral, deterministic output.
- [x] Integration tests in `tests/analysis/issue_1131_calibration_correction.rs` (4 tests):
  - `twenty_over_estimations_clamp_to_minimum_correction` — acceptance criterion from the issue.
  - `analyze_all_exposes_calibration_corrections_in_metadata` — end-to-end through `analyze_all`, asserting synapse + neuron metadata carry the per-change-type corrections at the expected floor.
  - `calibration_corrections_are_deterministic` — repeated runs produce identical corrections, and they match a direct-compute against the same cache.
  - `no_failure_cache_leaves_corrections_empty` — absence of cache leaves the map empty (no behaviour change for existing callers).
- [x] `./quality.sh` passes cleanly (all existing tests green).
