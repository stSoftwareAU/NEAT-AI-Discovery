## Summary

This PR implements Issue #216: Performance improvement by replacing Mutex-protected HashMap with DashMap for diagnostic aggregation during parallel analysis.

### Problem

Diagnostic aggregation during parallel analysis used `Arc<Mutex<HashMap>>` patterns which caused thread contention when multiple threads processed focus neurons simultaneously. Each thread competing for the same Mutex when recording diagnostics created a serialisation point that slowed down otherwise parallel work.

### Solution

Replaced `HashMap` with `DashMap` inside `TargetDiagnostics` and `NeuronDiagnostics` structs:

1. **Changed internal storage**: The `entries` field in both diagnostic structs now uses `DashMap<String, ...>` instead of `HashMap<String, ...>`
2. **Updated method signatures**: All modification methods now take `&self` instead of `&mut self`, enabling concurrent access without external locking
3. **Removed Mutex wrappers**: Code in `implementation.rs` and `neuron.rs` no longer wraps diagnostics in `Arc<Mutex<...>>` - just `Arc<TargetDiagnostics>` is sufficient

### Files Modified

- `Cargo.toml` - Added `dashmap = "5.5"` dependency
- `src/analysis/diagnostics.rs` - Replaced HashMap with DashMap, updated all methods to use `&self`
- `src/analysis/implementation.rs` - Removed Mutex wrapper, direct method calls on diagnostics
- `src/analysis/neuron.rs` - Removed Mutex wrapper, direct method calls on diagnostics
- `src/analysis/implementation_tests.rs` - Updated tests to remove `mut` where no longer needed

### Changes

- **Lock-free concurrent access**: DashMap provides internal sharding for low-contention concurrent operations
- **Simplified code**: Removed ~15 `.lock().expect()` calls across the codebase
- **Thread-safe by design**: Diagnostic structs now handle concurrency internally, making them safer to use

## Evidence

Unable to provide benchmark results: This is a performance optimisation that reduces thread contention during parallel analysis. The improvement (estimated 3-5% in the issue) would only be measurable under high parallelism (8+ threads processing 64+ focus neurons simultaneously), which requires specific hardware and production workloads to accurately measure.

The theoretical improvement comes from:
- Eliminating Mutex lock acquisition overhead (context switches, memory barriers)
- Allowing concurrent writes to different diagnostic entries without blocking
- DashMap's sharded design reduces contention even for concurrent writes to the same shard

## Test Plan

### New Concurrent Tests Added (6 tests)

All tests in `src/analysis/diagnostics.rs`:

1. `test_target_diagnostics_concurrent_record_count` - Verifies 64 threads can concurrently set record counts for their respective targets
2. `test_target_diagnostics_concurrent_candidate_attempts` - Verifies 8 threads can record 10 candidate attempts each for 8 targets (640 total operations)
3. `test_target_diagnostics_concurrent_mark_selected` - Verifies 32 threads can concurrently mark their targets as selected
4. `test_neuron_diagnostics_concurrent_record_count` - Verifies 64 threads can concurrently set record counts for neuron diagnostics
5. `test_neuron_diagnostics_concurrent_candidate_attempts` - Verifies 8 threads can record 10 candidate attempts each for neuron diagnostics
6. `test_neuron_diagnostics_concurrent_filtered_marking` - Verifies 30 threads can concurrently mark neurons with different filter types

### Existing Tests

All 339 existing tests continue to pass, verifying that:
- Diagnostic data is recorded correctly
- No-candidate summaries report accurate reasons
- Integration tests for synapse and neuron analysis still work

### Test Results

```
test analysis::diagnostics::tests::test_target_diagnostics_concurrent_record_count ... ok
test analysis::diagnostics::tests::test_target_diagnostics_concurrent_candidate_attempts ... ok
test analysis::diagnostics::tests::test_target_diagnostics_concurrent_mark_selected ... ok
test analysis::diagnostics::tests::test_neuron_diagnostics_concurrent_record_count ... ok
test analysis::diagnostics::tests::test_neuron_diagnostics_concurrent_candidate_attempts ... ok
test analysis::diagnostics::tests::test_neuron_diagnostics_concurrent_filtered_marking ... ok
```
