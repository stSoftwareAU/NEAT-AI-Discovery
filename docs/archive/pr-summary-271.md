## Summary

Extract diagnostic tracking and rejection reason structures from `implementation.rs` (~900 lines) to a dedicated `diagnostics.rs` module as part of the ongoing refactoring effort (Issue #185).

### Changes Made

Created `src/analysis/diagnostics.rs` containing:

**Core Adapter**
- `RecordCacheProvider` - Adapter for focus impact calculation with recorded activations
- `compute_impact_scores_for_discounting()` - Impact score computation for candidate discounting

**Synapse Rejection Tracking**
- `RejectionReason` enum - Why a synapse candidate was rejected (NoSamples, ZeroImprovement, BelowThreshold)
- `RejectionDetail` struct - Detailed rejection information
- `ThresholdContext` struct - Threshold calculation context
- `TargetDiagnosticEntry` struct - Per-target diagnostic entry
- `TargetDiagnostics` struct - Collection of target diagnostics with logging

**Neuron Rejection Tracking**
- `NeuronRejectionDetail` struct - Why a neuron candidate was rejected
- `NeuronDiagnosticEntry` struct - Per-neuron diagnostic entry
- `NeuronDiagnostics` struct - Collection of neuron diagnostics with logging

**Target Data Structures**
- `TargetData` struct - Target neuron data for sample building
- `TargetMap` struct - Pre-built target map for efficient sample building

**Focus Target Filtering**
- `FocusTargetFilterResult` struct - Result of filtering focus targets
- `filter_focus_targets_for_neuron_analysis()` - Filter and classify focus targets

**Validation**
- `require_unique_focus()` - Validate focus targets are unique

### Lines Changed

- `implementation.rs`: -1119 lines (removed extracted code, added imports)
- `diagnostics.rs`: +1382 lines (new module with extracted code and tests)
- `mod.rs`: +2 lines (new module declaration and documentation update)

## Evidence

Unable to generate screenshot: This is a CLI library with no visual interface. The refactoring maintains functional equivalence with the original code.

## Test Plan

- All 283 unit tests pass (including new tests in `diagnostics.rs`)
- All 47 integration test files pass
- Verified via `./quality.sh` which runs:
  - Build (debug and release)
  - Linting (clippy)
  - Type checking
  - All tests

### New Tests Added

Tests in `src/analysis/diagnostics.rs`:
- `test_rejection_reason_display` - Verify RejectionReason Display impl
- `test_target_diagnostics_tracks_candidate` - Verify synapse diagnostic tracking
- `test_neuron_diagnostics_tracks_filtered` - Verify neuron diagnostic filtering
- `test_require_unique_focus_empty` - Verify empty focus validation
- `test_require_unique_focus_duplicates` - Verify duplicate detection
- `test_require_unique_focus_valid` - Verify valid focus list
- `test_filter_focus_targets_output_only` - Verify output-only filtering mode
- `test_filter_focus_targets_allow_hidden` - Verify hidden neuron inclusion
- `test_target_map_from_records` - Verify TargetMap construction
- `test_target_map_build_samples` - Verify sample building from TargetMap
