## Summary

Overlap candidate compression with discovery module detection phase in `analyze_all()` post-processing pipeline. Closes #1004.

Previously, candidate compression (identity + nonlinear) and discovery module detection (~48 modules) ran sequentially. Both phases read from the synapse result immutably during detection — compression reads `helpful_synapses`, and discovery module detection closures capture their data before execution. This change runs both detection phases concurrently via `rayon::join`, then merges results sequentially (compression first, then discovery modules) to preserve deterministic ordering.

### Changes

1. **`src/analysis/discovery_dispatch.rs`** — Split `run_discovery_modules_parallel` into two composable functions:
   - `detect_discovery_modules_parallel` — runs all detection closures concurrently, returns `DiscoveryModuleDetectionResults`
   - `merge_discovery_module_results` — merges detection results into synapse result sequentially
   - `run_discovery_modules_parallel` retained as a convenience wrapper calling both

2. **`src/analysis/module_dispatch_specs/mod.rs`** — Added `prepare_and_detect_discovery_modules` that builds specs, allocates budgets, and runs detection without merging. Removed unused `dispatch_and_merge_discovery_modules`.

3. **`src/analysis/orchestration.rs`** — Restructured post-processing to use `rayon::join` to overlap compression detection with discovery module detection, then merge sequentially.

### Performance

Wall-clock time for post-processing decreases because the lighter compression work (identity + nonlinear via nested `rayon::join`) overlaps with the heavier discovery module dispatch (~48 parallel detection closures). Benchmarks require GPU hardware and can be measured with `cargo bench --bench pipeline_utilisation`.

## Evidence
- 8 new integration tests verify:
  - Split detect + merge produces identical results to the combined function
  - Detection results preserve original module order
  - Empty inputs handled correctly
  - Overlapped compression + detection produces same results as sequential
  - Deterministic across 10 repeated runs (no race conditions)
  - Merge ordering: compression before discovery
  - `max_synapse_candidates` respected through split path
- All 17 existing `discovery_dispatch` unit tests pass unchanged
- All 158 analysis integration tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)

## Test Plan
- Added `tests/analysis/issue_1004_overlap_compression_discovery.rs` with 8 tests
- Existing tests verified: `discovery_dispatch_tests.rs`, `discovery_dispatch_parallel_tests.rs`, `issue_1003_parallel_candidate_compression.rs`
- Run: `cargo test --test analysis -- issue_1004`
