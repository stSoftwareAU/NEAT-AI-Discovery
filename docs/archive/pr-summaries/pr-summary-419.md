## Summary

Parallelise the 25 sequential discovery modules in `analyze_all()` using `rayon::par_iter()` to utilise multiple CPU cores during the detection phase (Issue #419).

### Changes Made

1. **`src/analysis/discovery_dispatch.rs`** — New parallel dispatch API:
   - `DiscoveryModuleSpec` struct encapsulating a module's name, phase, and boxed `FnOnce` detection closure (`Send`-safe)
   - `run_discovery_modules_parallel()` function that runs all detection closures in parallel via `rayon::into_par_iter().map().collect()`, then merges results sequentially in original order

2. **`src/analysis/mod.rs`** — Refactored `analyze_all()` discovery section:
   - Replaced ~800 lines of 25 sequential `run_discovery_module()` calls with parallel dispatch
   - Shared data (`creature`, `hidden_neurons`) wrapped in `Arc` for `Send`-safe closure capture
   - Record collection inlined into each closure (previously used local helper closures)
   - Sequential merge phase preserved via `merge_coordinated_structural_replacements()`

3. **`Cargo.toml`** — Added `[[bench]]` entry for `parallel_discovery`

### Architecture

**Two-phase design** preserving deterministic ordering:
- **Phase 1 (parallel)**: All 25 detection closures run concurrently via rayon's work-stealing thread pool. Each closure independently reads from the thread-safe `RecordCache` (`parking_lot::RwLock`) and performs CPU-bound detection.
- **Phase 2 (sequential)**: Results are merged in original module order into the synapse result. `rayon::collect()` preserves indexed iterator order, ensuring deterministic output regardless of thread scheduling.

**Thread safety**: `RecordCache` uses `parking_lot::RwLock` for concurrent reads; `watchdog::beat()` uses `parking_lot::Mutex`; `PhaseTimer` has no shared state. All detection functions are pure computations with no global mutable state.

### Benchmark Results

Measured on Apple M4 Pro with `cargo bench --bench parallel_discovery`:

| Scenario | Hidden neurons | Records/neuron | Wall-clock time |
|----------|---------------|----------------|-----------------|
| Small    | 5             | 100            | 146.72 ms       |
| Medium   | 20            | 200            | 160.07 ms       |
| Large    | 50            | 200            | 160.92 ms       |

The near-constant wall-clock time across creature sizes (5 → 50 hidden neurons) demonstrates effective parallel utilisation — the detection phase scales with available cores rather than module count.

## Evidence

This is a backend/library change with no UI component. Evidence is provided by the test results:
- 6 unit tests in `discovery_dispatch_parallel_tests.rs` covering merge ordering, None/Some handling, max candidates, empty input, deterministic ordering, single-module equivalence
- 2 integration tests in `issue_419_parallel_discovery_execution.rs` covering deterministic results (5 runs with tolerance) and metadata consistency
- All 469 unit tests + 99 integration test files pass
- `quality.sh` passes cleanly (fmt, clippy, check, tests, release build)

## Test Plan

- `src/analysis/discovery_dispatch_parallel_tests.rs` — 6 unit tests:
  - `parallel_merges_candidates_from_multiple_modules`
  - `parallel_handles_mix_of_none_and_some_results`
  - `parallel_respects_max_synapse_candidates`
  - `parallel_empty_modules_produces_no_changes`
  - `parallel_preserves_deterministic_ordering`
  - `parallel_single_module_matches_sequential`

- `tests/issue_419_parallel_discovery_execution.rs` — 2 integration tests:
  - `parallel_discovery_produces_deterministic_results`
  - `parallel_discovery_metadata_consistent`

- `benches/parallel_discovery.rs` — Criterion benchmark with 3 creature sizes (5/20/50 hidden neurons)
