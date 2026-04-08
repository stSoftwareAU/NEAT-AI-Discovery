## Summary

Add MCMC diagnostics tracking to the candidate selection pipeline: acceptance rates per candidate type (synapse, neuron, coordinated), proposal quality distribution (min, max, mean, median of improvement values), and source/target diversity metrics. All detail collection is zero-overhead when verbose mode is disabled — only atomic counters are incremented. Closes #1021.

## Changes

- **New file**: `src/analysis/diagnostics/mcmc_diagnostics.rs` — `McmcDiagnosticsTracker` with lock-free atomic counters for acceptance rates, mutex-guarded detail collection (improvement values, diversity sets) gated on verbose mode
- **Pipeline integration**: Tracker added to `TargetAnalysisContext`, created in orchestration, wired through to metadata output
- **Counter instrumentation**: `evaluation.rs` records `record_evaluated` for every candidate with computed improvement, `record_accepted` for candidates that pass all filters
- **JSON output**: `McmcDiagnosticsJson` added to `SynapseAnalysisMetadataJson` with per-type acceptance rates, proposal quality, and diversity metrics
- **Zero-overhead guarantee**: Improvement value collection and diversity set tracking only allocate when `NEAT_AI_DISCOVERY_VERBOSE=1`

## Evidence

All 171 integration tests pass. All quality gate checks pass (clippy, fmt, doc build, release build).

## Test Plan

- `test_acceptance_counters_empty` — verifies zero counters return 0.0 rate
- `test_acceptance_counters_tracking` — verifies proposed/accepted counting and rate calculation
- `test_tracker_per_type_counting` — verifies per-type (synapse, neuron, coordinated) breakdown
- `test_proposal_quality_statistics` — verifies min, max, mean, median for even-count values
- `test_proposal_quality_odd_count` — verifies median for odd-count values
- `test_diversity_metric` — verifies unique source/target counting in evaluated vs accepted
- `test_no_proposals_returns_zero_rate` — verifies empty tracker produces zero summary
- `test_concurrent_counter_updates` — verifies thread-safe counting with 8 parallel threads
