## Summary

Audited and removed all 5 `#[allow(dead_code)]` markers across the codebase. Each suppressed field was written but never read, making them truly dead code. All fields and their associated `#[allow(dead_code)]` annotations have been removed. Closes #716.

### Changes by file

- **`src/analysis/recommendation/epistatic/pre_screening.rs`** — Removed `sample_index` and `primary_contribution` fields from `ResidualSample` (written but never read). Removed unnecessary `.enumerate()` call.
- **`src/analysis/cache/serialisation.rs`** — Removed `record_count` field from `CompressedCacheEntry` (set at construction but never accessed).
- **`src/analysis/early_termination.rs`** — Removed `alpha` and `beta` fields from `SequentialEvaluator` (only used during construction to compute `upper_bound` and `lower_bound`, then stored but never read again).
- **`src/analysis/streaming.rs`** — Removed `block_id` field from `CacheBlock` and its unused parameter from `CacheBlock::new()` (stored but never accessed after construction).

## Evidence

This is a backend-only change with no visual output. Evidence:
- `quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
- `grep -r '#\[allow(dead_code)\]' src/` returns no matches
- No new warnings introduced

## Test Plan

- All existing tests continue to pass unchanged
- No test modifications were needed since the removed fields were never read
