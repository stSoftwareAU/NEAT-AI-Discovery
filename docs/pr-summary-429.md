# PR Summary: Issue #429 - Early Termination Improvements for Low-Value Candidates

## Summary

Extended the SPRT-based early termination system (Issue #219) to the candidate generation phase. This provides infrastructure for reducing analysis time by filtering low-value candidates before expensive GPU evaluation.

### New Features

1. **Hierarchical Candidate Filtering (`PreFilterConfig`, `PreFilterStats`)**
   - Quick pre-filter before detailed GPU analysis
   - Filters candidates based on:
     - Source activation variance (constant sources rejected)
     - Sample count requirements (insufficient data rejected)
     - Error-activation correlation strength (weak correlation rejected)
   - Configurable thresholds via `PreFilterConfig`

2. **Cross-Module Deduplication (`CandidateDeduplicator`, `CandidateSignature`)**
   - Skip candidates similar to already-generated ones
   - Uses bucketed signatures for similarity detection
   - Prevents redundant GPU evaluation of near-duplicate candidates

3. **Incremental Confidence Tracking (`IncrementalConfidenceTracker`)**
   - Early exit when sufficient high-confidence candidates found
   - Configurable target count and confidence threshold
   - Enables budget-aware prioritisation

### Implementation Details

The new functionality is implemented in `src/analysis/early_termination.rs`:
- `PreFilterStats::from_samples()` computes lightweight statistics for filtering
- `prefilter_candidates()` applies batch filtering before GPU submission
- `CandidateDeduplicator::is_duplicate()` prevents duplicate processing
- `IncrementalConfidenceTracker::record()` enables early termination when targets met

## Evidence

### Benchmark Results

Created `benches/early_termination.rs` to measure performance across different creature sizes:

| Creature Size | Input Count | Hidden | Synapses | Mean Analysis Time |
|--------------|-------------|--------|----------|-------------------|
| Small        | 50          | 10     | 100      | 147.8ms ± 9.1ms   |
| Medium       | 200         | 50     | 500      | 380.0ms ± 2.9ms   |
| Large        | 500         | 100    | 1500     | 1332ms ± 30ms     |

**Low-Value Filtering Test** (300 inputs, 90% low-value):
- Analysis with early termination: 452ms ± 3.9ms

**Note**: This PR provides the infrastructure for early termination in candidate generation.
The filtering primitives are now available for integration into the main analysis pipeline.
Full performance benefits will be realised when:
1. `prefilter_candidates()` is integrated into the synapse analysis loop
2. `CandidateDeduplicator` is used to skip redundant candidates across discovery modules
3. `IncrementalConfidenceTracker` is used to implement budget-aware termination

The conservative default thresholds (min_source_variance=0.001, min_correlation=0.05)
are designed to filter only clearly low-value candidates while maintaining discovery coverage.

## Test Plan

### Unit Tests Added
- `test_prefilter_stats_from_empty_samples` - Handles empty input correctly
- `test_prefilter_stats_from_constant_source` - Detects constant sources (zero variance)
- `test_prefilter_stats_from_correlated_source` - Computes correlation correctly
- `test_prefilter_rejects_constant_source` - Pre-filter rejects low-value candidates
- `test_prefilter_accepts_good_candidate` - Pre-filter accepts high-value candidates
- `test_prefilter_disabled_passes_all` - Disabled config allows all through
- `test_prefilter_candidates_batch` - Batch filtering works correctly
- `test_candidate_signature_similarity` - Signature similarity detection
- `test_deduplicator_tracks_duplicates` - Deduplicator prevents duplicates
- `test_deduplicator_clear` - Deduplicator reset works
- `test_incremental_confidence_stops_at_target` - Early exit when targets met
- `test_incremental_confidence_rate` - Confidence rate calculation correct
- `test_incremental_confidence_default` - Default configuration sensible

### Verification
- All 462 existing tests pass
- `./quality.sh` passes cleanly
- No decrease in test coverage for existing functionality

## Files Changed

- `src/analysis/early_termination.rs` - Extended with pre-filtering, deduplication, and incremental confidence
- `src/analysis/mod.rs` - Export new types
- `benches/early_termination.rs` - New benchmark for performance measurement
- `Cargo.toml` - Added benchmark entry
- `docs/pr-summary-429.md` - This file

## Related Issues

- Part of #412 (Plan discovery improvements)
- Extends #219 (SPRT early termination for GPU evaluation)
