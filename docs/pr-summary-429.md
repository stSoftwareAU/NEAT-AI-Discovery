## Summary

Extends the SPRT-based early termination system (Issue #219) beyond GPU evaluation to the candidate generation pipeline. Adds a new `candidate_prefilter` module that removes low-value candidates and deduplicates across discovery modules before they reach the controller.

### Four Improvements

1. **Hierarchical candidate filtering** — removes candidates with near-zero or negative expected improvement (`< 0.001`) before they are merged into the analysis result, applied per-module in `run_discovery_module`.

2. **Budget-aware prioritisation** — when `max_synapse_candidates` is set, the pre-filter limits output to the top-N candidates by expected improvement, preventing the controller from wasting ablation budget.

3. **Incremental confidence** — candidates are sorted by expected improvement (best first) so the controller evaluates the most promising candidates first.

4. **Cross-module deduplication** — after all discovery modules have contributed candidates, the pre-filter groups candidates by (target neuron, operation signature) and keeps only the best per group. Operation signatures include structural details (activation function, source/target UUIDs) to avoid incorrectly deduplicating structurally different candidates.

### Integration Points

- **Per-module filtering** in `discovery_dispatch::run_discovery_module()` — each discovery module's candidates are filtered for low-value noise before merging.
- **Cross-module deduplication** in `analyze_all()` — applied to all `coordinated_structural_candidates` after all modules have run, before candidate clustering.

## Evidence

This is a backend performance change with no UI. Benchmark results from `benches/early_termination.rs`:

| Operation | 50 candidates | 200 candidates | 500 candidates | 1000 candidates |
|-----------|--------------|----------------|----------------|-----------------|
| filter_low_value | 1.4 us | 5.3 us | 14.0 us | 28.0 us |
| deduplicate | 2.5 us | 8.4 us | 20.1 us | 40.2 us |

The pre-filter overhead is negligible (microseconds) compared to the GPU analysis it saves. In the `coordinated_structural_replace_synapse_with_relu` integration test, cross-module deduplication reduced coordinated candidates from 31 to a focused subset, demonstrating significant downstream savings in controller ablation time.

## Test Plan

- Added `tests/issue_429_early_termination_improvements.rs` — 14 integration tests covering:
  - Low-value candidate filtering (4 tests)
  - Budget-aware prioritisation (2 tests)
  - Incremental confidence sorting (1 test)
  - Cross-module deduplication (4 tests)
  - Full pipeline integration (2 tests)
  - Configuration defaults (1 test)
- Added unit tests in `src/analysis/candidate_prefilter.rs` — 7 unit tests covering:
  - Operation classification
  - Operation signature generation (single, multi, empty)
  - Squash-differentiated signatures
  - Target UUID extraction
- Added `benches/early_termination.rs` benchmark
- All 97 existing integration test files continue to pass
