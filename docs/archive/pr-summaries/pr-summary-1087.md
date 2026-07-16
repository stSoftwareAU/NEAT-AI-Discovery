## Summary

Harden Rayon parallel joins with explicit panic catching across the analysis
pipeline. Wraps key `rayon::join` and `par_iter` dispatch points with
`std::panic::catch_unwind()` so that a panic in one parallel branch does not
corrupt results from sibling branches. Panics are logged with module context
and converted to errors or empty results. Closes #1087.

### Changes

- **`src/analysis/discovery_dispatch.rs`** — Wrap each `par_iter` detection
  closure in `catch_unwind(AssertUnwindSafe(...))`. A panicking module produces
  `None` (empty result) and logs a warning with the module name and panic message.

- **`src/analysis/orchestration.rs`** — Wrap both branches of the
  `dispatch_analyses` `rayon::join` (synapse + neuron analysis) with
  `catch_unwind`, converting panics to `anyhow::Error` with context about which
  phase panicked. Also wrap the nested `rayon::join` for candidate compression
  and discovery module detection, returning empty results on panic.

- **`src/analysis/discovery_dispatch_parallel_tests.rs`** — Add three unit tests
  verifying that a simulated panic in a parallel detection module is caught,
  does not corrupt sibling results, and preserves module metadata.

## Evidence

This is a backend/library change with no UI. Evidence is provided by the new
unit tests:

- `parallel_detection_catches_panic_in_module_and_returns_error_result`
- `parallel_detection_panic_does_not_corrupt_sibling_module_results`
- `parallel_detection_panic_message_includes_module_context`

All tests pass via `cargo test --lib --all-features`. Full `./quality.sh` passes
cleanly.

## Test Plan

- Added 3 new tests in `src/analysis/discovery_dispatch_parallel_tests.rs`
  that simulate panics in parallel discovery module closures and verify:
  1. The panic is caught and the panicking module returns `None`
  2. Sibling modules' candidates are merged correctly despite the panic
  3. Module metadata (name) is preserved for observability
- All existing tests continue to pass (761 total library tests)
- No performance regression — `catch_unwind` has negligible overhead on the
  non-panic path
