## Summary

Add hold-out validation for synapse multi-weight search to combat overfitting. The 9-variant multi-weight search in synapse candidate evaluation picks the best-looking weight from 9 options, creating selection bias that contributes to 0% success rate in practice. This change splits samples into train/validate sets (70/30), selects the best weight using only training data, and reports improvement using only held-out validation data. Closes #893.

## Changes

- **`src/analysis/synapse/holdout_validation.rs`** (new): Deterministic train/validate splitting module using FNV-1a hash of (source_uuid, target_uuid) as split seed for reproducibility.
- **`src/analysis/synapse/target_analysis/evaluation.rs`**: Apply hold-out validation to the 9-weight search for new synapse candidates. Weight selection uses training samples; improvement and counts are reported from validation samples.
- **`src/analysis/synapse/activation_evaluation.rs`**: Apply hold-out validation to the 9-weight search for non-ReLU activation candidates with non-linear targets.
- **`src/analysis/constants.rs`**: Add `HOLDOUT_MIN_SAMPLE_COUNT` (20) and `HOLDOUT_VALIDATION_FRACTION` (0.3) constants.
- **`src/analysis/synapse/mod.rs`**: Register the new `holdout_validation` module.

## Design Decisions

- **Minimum sample threshold**: Below `HOLDOUT_MIN_SAMPLE_COUNT` (20), falls back to the current full-sample approach with existing pessimism discounting, since splitting would leave too few samples for reliable results.
- **Consistent total_count**: When hold-out validation is used, `total_count` is set to the validation sample count (not the full sample count) to keep `MIN_IMPROVED_RATIO` checks consistent.
- **Deterministic splitting**: Uses FNV-1a hash of (source_uuid, target_uuid) as the split seed, ensuring the same synapse candidate always gets the same split across evaluations.

## Evidence

- All 348 existing tests continue to pass (including GPU-dependent tests)
- `quality.sh` passes cleanly (clippy, fmt, tests, docs, release build)

## Test Plan

- Unit tests in `src/analysis/synapse/holdout_validation.rs`:
  - `test_split_returns_none_below_threshold` — verifies fallback below minimum sample count
  - `test_split_produces_correct_partition_sizes` — verifies 70/30 split
  - `test_split_is_deterministic` — verifies same UUIDs produce identical splits
  - `test_different_uuids_produce_different_splits` — verifies different UUIDs produce different splits
  - `test_no_sample_in_both_partitions` — verifies partitions are disjoint
  - `test_baseline_error_sq_computation` — verifies baseline error calculation
  - `test_collect_samples_roundtrip` — verifies sample collection helper
- Integration tests in `tests/synapse/issue_893_holdout_validation.rs`:
  - Constant validation: `HOLDOUT_MIN_SAMPLE_COUNT >= MIN_DISCOVERY_SAMPLE_COUNT`
  - Fraction in valid range: `HOLDOUT_VALIDATION_FRACTION` in (0.1, 0.5)
  - Training partition has at least `MIN_NEURON_SAMPLE_COUNT` samples
  - Validation partition has at least 3 samples
  - 70/30 split produces expected partition sizes
  - Threshold-edge split produces non-empty partitions
