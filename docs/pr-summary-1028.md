## Summary

Add a configurable `maxAnalysisMemoryMb` parameter to the analysis phase that
prevents unbounded memory growth on memory-constrained machines. When the Rust
heap allocation reaches 90% of the budget, the analysis returns partial results
rather than continuing until the OS kills the process. Closes #1028.

## Changes

- **`src/ffi_types/requests.rs`**: Added `max_analysis_memory_mb: Option<u64>` to
  `AnalyzeParallelInput` and `AnalyzeAllInput`
- **`src/ffi_internal/analysis.rs`**: Propagated the field through
  `build_analyze_all_input_from_parallel` and added `memory_budget_exceeded` to
  all `AnalyzeParallelOutput` constructions
- **`src/ffi_types/responses/analysis.rs`**: Added `memory_budget_exceeded: Option<bool>`
  to `AnalyzeParallelOutput` (omitted from JSON when `None` for backwards compatibility)
- **`src/analysis/shared/metadata.rs`**: Added `memory_budget_exceeded: bool` to
  `AnalyzeAllResult`
- **`src/analysis/utils/memory.rs`**: Added `check_memory_budget_exceeded()` (pure,
  testable) and `is_memory_budget_exceeded()` (reads global allocator) utility functions
  with a 90% threshold
- **`src/analysis/orchestration.rs`**: Added three memory budget check points:
  1. Before GPU analysis (after fingerprint filtering)
  2. After parquet cache loading
  3. After GPU analysis (gates post-processing)

## Backwards Compatibility

When `maxAnalysisMemoryMb` is omitted (the default), no memory limit is enforced
and behaviour is identical to prior versions. The `memoryBudgetExceeded` field is
omitted from the JSON response when not set.

## Evidence

All 11 new tests pass and the full quality gate (`./quality.sh`) passes cleanly.

## Test Plan

- `tests/ffi/issue_1028_memory_budget.rs`:
  - `analyze_parallel_input_accepts_memory_budget` — deserialises the new field
  - `analyze_parallel_input_defaults_to_none_when_omitted` — backwards compatibility
  - `analyze_all_input_accepts_memory_budget` — field on combined input
  - `check_memory_budget_returns_false_when_no_budget` — no budget = never exceeded
  - `check_memory_budget_returns_false_when_under_budget` — within budget
  - `check_memory_budget_returns_true_when_over_budget` — over budget
  - `check_memory_budget_returns_true_when_approaching_budget` — 90% threshold
  - `check_memory_budget_handles_zero_budget` — edge case
  - `analyze_parallel_output_serialises_memory_budget_exceeded` — JSON field present
  - `analyze_parallel_output_omits_memory_budget_exceeded_when_none` — omitted when None
  - `analyze_all_input_propagates_memory_budget_via_json` — propagation test
