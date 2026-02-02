## Summary

This PR completes the extraction of target activation simulation functions from `implementation.rs` and `synapse.rs` to the dedicated `activation.rs` module as part of the ongoing refactoring of the implementation.rs monolith (Issue #185).

### Changes Made

1. **Moved to `src/analysis/activation.rs`:**
   - `TargetSimulationMode` enum - for saturation-aware candidate scoring
   - `get_target_activation_fn()` - gets the target simulation function for a squash name
   - `get_target_simulation_fn()` - checks if samples support target activation simulation
   - `get_target_simulation_mode()` - selects the best available target simulation mode
   - `can_use_hard_tanh()` - legacy backwards compatibility function
   - `has_sufficient_output_variance()` - checks if a new neuron would be saturated
   - `MIN_NEURON_OUTPUT_STD_DEV` constant - threshold for saturation detection

2. **Updated imports:**
   - `synapse.rs` now imports target simulation functions from `activation.rs`
   - `implementation.rs` now imports target simulation functions from `activation.rs`
   - Removed duplicate function definitions from both files

3. **Updated re-exports in `mod.rs`:**
   - All activation-related functions are now re-exported for backward compatibility
   - Public API remains unchanged

### Rationale

This refactoring follows the KISS and DRY principles:
- **Single Source of Truth**: Activation-related functions now live in one module
- **Self-contained Extraction**: Activation functions have no dependencies on other parts of the analysis code
- **Backward Compatibility**: All re-exports added to maintain the existing public API

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Added comprehensive unit tests for all moved functions:
  - `test_get_target_activation_fn` - verifies activation function lookup
  - `test_get_target_simulation_fn_with_complete_samples` - verifies simulation with complete data
  - `test_get_target_simulation_fn_with_incomplete_samples` - verifies fallback behaviour
  - `test_get_target_simulation_mode_none_squash` - verifies None handling
  - `test_get_target_simulation_mode_full` - verifies full simulation mode
  - `test_get_target_simulation_mode_approximate` - verifies approximation mode for HARD_TANH
  - `test_can_use_hard_tanh` - verifies legacy function behaviour
  - `test_has_sufficient_output_variance_with_variance` - verifies variance detection
  - `test_has_sufficient_output_variance_saturated` - verifies saturation detection
  - `test_has_sufficient_output_variance_constant_input` - verifies constant input handling
  - `test_has_sufficient_output_variance_insufficient_samples` - verifies edge cases

- All 354+ existing unit tests pass
- All integration tests pass
- `./quality.sh` passes cleanly (no warnings, no errors)
