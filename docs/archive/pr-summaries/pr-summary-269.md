## Summary

Extracted sample data structures and GPU data formats from `implementation.rs` (~465 lines) to a dedicated `src/analysis/samples.rs` module as part of the ongoing refactoring of the implementation.rs monolith (parent issue #185).

### Structures Extracted

**Core Sample Types:**
- `HelpfulSample` - Core evaluation unit passed between CPU and GPU
- `NeuronStats` - Neuronal statistics with methods for computing from records and samples

**GPU-Compatible Formats (with `#[repr(C)]`, `Pod`, `Zeroable` derives):**
- `GpuHelpfulSample` - GPU buffer format
- `HelpfulContribution` / `HelpfulUniforms` - Helpful synapse evaluation
- `HarmfulContribution` / `HarmfulUniforms` - Harmful synapse evaluation
- `ReluContribution` / `ReluUniforms` - ReLU activation evaluation
- `BiasResult` / `BiasUniforms` - Bias optimisation
- `ActivationOutput` / `ActivationUniforms` - Activation function evaluation

**Statistics Results:**
- `HelpfulStats` - Computed statistics for helpful synapse evaluation
- `HarmfulStats` - Computed statistics for harmful synapse evaluation
- `ReluOrientation` enum and `ReluStats` - ReLU evaluation results

**Supporting Functions:**
- `compute_source_variance_discount()` - Source reliability calculation
- `constant_source_effect_threshold_from_env()` - Environment-based threshold override
- `EPSILON` and `DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD` constants

### Module Organisation

- New `src/analysis/samples.rs` module with comprehensive documentation
- Re-exports through `src/analysis/mod.rs` maintain full backwards compatibility
- Updated module documentation to list samples.rs in the target structure

## Evidence

Unable to generate screenshot: This is a CLI library with no visual interface. The refactoring is purely internal code organisation.

## Test Plan

### New Tests Added (`src/analysis/samples.rs`)
- `test_helpful_sample_default` - Verifies default sample values
- `test_gpu_helpful_sample_from_helpful_sample` - Tests GPU format conversion
- `test_bias_result_zeroed` - Tests BiasResult::zeroed() constructor
- `test_relu_stats_new` - Tests ReluStats::new() constructor
- `test_harmful_stats_default` - Tests HarmfulStats default
- `test_helpful_stats_default` - Tests HelpfulStats default
- `test_compute_source_variance_discount_empty` - Empty samples edge case
- `test_compute_source_variance_discount_single` - Single sample edge case
- `test_compute_source_variance_discount_constant` - Constant source discounting
- `test_compute_source_variance_discount_high_variance` - High variance sources
- `test_neuron_stats_from_samples_empty` - Empty samples edge case
- `test_neuron_stats_from_samples_valid` - Valid sample statistics
- `test_neuron_stats_to_json` - JSON conversion
- `test_gpu_structs_are_pod` - Verifies all GPU structs implement Pod/Zeroable

### Existing Tests
All 273 existing unit tests and 165+ integration tests pass without modification, verifying backwards compatibility.
