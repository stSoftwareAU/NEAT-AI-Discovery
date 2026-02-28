## Summary

Replace all `"BUG:"` string literal prefixes in source code with structured `tracing` events at appropriate severity levels. Closes #718.

### Changes

1. **`src/analysis/utils/deadline.rs`**: Replaced `"BUG: analysis_deadline_ms..."` messages with clearer `tracing::warn!` messages that describe the condition without the `BUG:` prefix. The messages now explain the fallback behaviour directly.

2. **`src/analysis/synapse/target_analysis/mod.rs`**: Upgraded the `"BUG: Target has no eligible upstream neurons"` message from `tracing::warn!` to `tracing::error!` (this condition indicates a data integrity issue) and removed the `BUG:` prefix.

3. **Unit tests**: Added 7 boundary condition tests for `calculate_effective_timeout_ms` covering exact min/max boundaries, just-below/just-above boundaries, and zero input.

4. **Integration test**: Added `issue_718_bug_comment_error_handling.rs` with a test that verifies a target neuron with only constant upstream neurons gracefully returns empty results.

## Evidence

This is a backend-only change with no UI impact. Evidence is provided by the test suite:

- All 540+ existing tests continue to pass
- 7 new boundary condition unit tests in `deadline_tests.rs`
- 1 new integration test in `tests/issue_718_bug_comment_error_handling.rs`
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)

## Test Plan

- `effective_timeout_just_below_minimum_defaults_to_ten_minutes` — verifies 2999ms falls back to default
- `effective_timeout_at_exact_minimum_passes_through` — verifies 3000ms passes through
- `effective_timeout_just_above_minimum_passes_through` — verifies 3001ms passes through
- `effective_timeout_just_below_maximum_passes_through` — verifies 3599999ms passes through
- `effective_timeout_at_exact_maximum_passes_through` — verifies 3600000ms passes through
- `effective_timeout_just_above_maximum_defaults_to_ten_minutes` — verifies 3600001ms falls back to default
- `effective_timeout_zero_defaults_to_ten_minutes` — verifies 0ms falls back to default
- `target_with_only_constant_upstream_returns_empty_results` — verifies graceful handling when no eligible upstream neurons exist
