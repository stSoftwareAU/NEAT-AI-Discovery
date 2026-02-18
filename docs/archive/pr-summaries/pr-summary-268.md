## Summary

This PR extracts deadline handling and logging utility code from `implementation.rs` (~400 lines) to a dedicated module at `src/analysis/utils/deadline.rs`.

**Parent issue:** #185 (Complete refactoring of implementation.rs monolith)

### Functions Extracted

#### Deadline Utilities
- `calculate_effective_timeout_ms()` - Compute timeout from deadline (handles absolute timestamps and relative durations)
- `build_deadline()` - Create deadline from timeout ms
- `deadline_passed()` - Check if deadline expired
- `calculate_gpu_batch_timeout()` - Deadline-aware GPU timeout calculation

#### Logging Functions
- `log_analysis_start()` - Log analysis beginning with timeout and focus neuron details
- `log_analysis_timeout()` - Log timeout occurrences when analysis is interrupted

#### Randomisation Utilities
- `derive_seed()` - Deterministic seed derivation using FNV-1a hash
- `shuffle_slice()` - Seeded shuffling for reproducible randomisation
- `shuffle_within_top_k()` - Shuffle top K elements while preserving rest (for deadline-constrained runs)

#### Environment Variable Helpers
- `parse_input_index()` - Parse neuron index from "input-N" UUID strings
- `source_input_index_bias_from_env()` - Get input index bias override for source ordering
- `focus_unused_observations_from_env()` - Check focus on unused observations (PUBLIC API, re-exported)
- `order_eligible_sources()` - Order source neurons for analysis with optional bias

### Module Structure

- `src/analysis/utils/deadline.rs` - Main implementation
- `src/analysis/utils/deadline_tests.rs` - Comprehensive unit tests
- `src/analysis/utils/mod.rs` - Re-exports all public functions

### Success Criteria Met

- [x] All deadline/timeout functions in dedicated module
- [x] `focus_unused_observations_from_env` remains accessible via re-export (public API)
- [x] Randomisation functions properly grouped
- [x] All existing tests pass (`./quality.sh`)
- [x] No public API changes

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface. The refactoring is purely structural (moving code to new module) with no behaviour changes.

## Test Plan

### Unit Tests (in `deadline_tests.rs`)
- `shuffle_within_top_k_deterministic_with_seed` - Verifies deterministic shuffling
- `shuffle_within_top_k_only_shuffles_prefix` - Ensures suffix remains untouched
- `shuffle_within_top_k_preserves_multiset` - Confirms no elements lost
- `shuffle_within_top_k_no_op_when_top_k_is_zero` - Edge case handling
- `deadline_passed_detects_elapsed_wall_clock_deadline` - Deadline expiry detection
- `build_deadline_handles_absolute_timestamps_and_relative_durations` - Timestamp handling
- `build_deadline_validates_duration_bounds` - Validates 3s-1hr bounds
- `calculate_effective_timeout_ms_matches_build_deadline_logic` - Consistency check
- `parse_input_index_parses_valid_input_uuids` - UUID parsing
- `parse_input_index_returns_none_for_invalid_uuids` - Invalid input handling
- `derive_seed_is_deterministic` - Seed derivation consistency
- `derive_seed_varies_with_context` - Context affects seed
- `derive_seed_varies_with_salt` - Salt affects seed
- `derive_seed_varies_with_base_seed` - Base seed affects output
- `shuffle_slice_is_deterministic_with_seed` - Slice shuffling
- `shuffle_slice_preserves_elements` - No element loss
- `shuffle_slice_no_op_for_single_element` - Single element edge case
- `shuffle_slice_no_op_for_empty` - Empty slice edge case
- `calculate_gpu_batch_timeout_returns_max_when_no_deadline` - No deadline behaviour
- `calculate_gpu_batch_timeout_returns_min_when_deadline_passed` - Expired deadline
- `calculate_gpu_batch_timeout_uses_half_remaining_time` - Adaptive timeout
- `calculate_gpu_batch_timeout_clamps_to_min` - Lower bound enforcement
- `calculate_gpu_batch_timeout_clamps_to_max` - Upper bound enforcement
- `ordered_neuron_stores_uuid_and_index` - Struct construction
- `order_eligible_sources_shuffles_deterministically_with_seed` - Ordering consistency
- `order_eligible_sources_preserves_all_elements` - Element preservation
- `order_eligible_sources_no_op_for_single_element` - Single element edge case

### Integration Tests
All existing integration tests continue to pass, verifying no regression in deadline-constrained analysis behaviour:
- `analysis_timeout.rs` - Deadline timeout behaviour
- `analyze_all_deadline_prioritises_synapses.rs` - Deadline ordering
- `issue_182_focus_unused_observations.rs` - Environment variable handling
