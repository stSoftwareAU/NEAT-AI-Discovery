## Summary

Implements quality-based module skipping during the discovery module merge phase and
activates previously unused heuristic early termination functions in the SPRT evaluator.
Closes #1074.

### Changes

1. **Quality-based module skipping (merge phase)**: After merging each module's detection
   results in `merge_discovery_module_results`, the system now checks whether enough
   high-quality candidates (>= 15 candidates with gain >= 0.01) have been accumulated.
   If so, remaining modules are skipped during merge, saving post-processing time.
   Two new configurable constants control the behaviour:
   - `QUALITY_SKIP_GAIN_THRESHOLD` (0.01) — minimum gain for a candidate to count as
     high quality
   - `QUALITY_SKIP_MIN_CANDIDATES` (15) — minimum number of high-quality candidates
     before skipping activates

2. **Heuristic early termination fast path**: The `is_strongly_beneficial()` (>70%
   positive at 30+ samples) and `is_strongly_harmful()` (<30% positive at 30+ samples)
   heuristics are now used as a fast path in `should_stop()`. This enables earlier
   Accept/Reject decisions for extreme improvement ratios before the full SPRT
   `min_samples` threshold is reached.

3. **Reduced `min_samples` default**: `EarlyTerminationConfig::default().min_samples`
   reduced from 100 to 50, allowing earlier SPRT decisions for moderate improvement
   ratios. The heuristic fast path further enables decisions at just 30 samples for
   extreme ratios.

4. **Benchmark**: Added `benches/quality_skip_dispatch.rs` to measure merge phase
   performance with varying module counts and candidate densities.

## Evidence

### Performance analysis

The quality-based module skipping operates during the **merge phase** (sequential, after
parallel detection). In production workloads with 47 discovery modules each producing
candidates, the merge phase iterates over all module results sequentially. When the
first few modules produce sufficient high-quality candidates, the remaining modules'
merge operations (filtering, sorting, truncation) are skipped entirely.

The heuristic early termination fast path reduces the minimum samples needed for a
decision from 100 (previous SPRT min_samples) to as few as 30 samples (heuristic
threshold) for candidates with extreme improvement ratios (>70% or <30%). This directly
reduces GPU evaluation time per candidate.

Benchmark suite (`cargo bench --bench quality_skip_dispatch`) validates that the merge
phase executes correctly with the quality skip logic across varying module counts
(10–47 modules, 5–10 candidates each).

## Test Plan

- Added 3 new unit tests in `discovery_dispatch_parallel_tests.rs`:
  - `quality_skip_skips_later_modules_when_enough_high_quality_candidates`
  - `quality_skip_does_not_skip_when_insufficient_high_quality_candidates`
  - `quality_skip_still_records_stats_for_skipped_modules`
- All 24 discovery dispatch tests pass (including 3 new)
- All 10 early termination unit tests pass
- All 12 early termination integration tests pass
- All 15 early termination proptest tests pass
- `./quality.sh` passes cleanly (lint, clippy, tests, docs, release build)
