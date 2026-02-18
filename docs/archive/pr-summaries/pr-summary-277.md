# PR Summary: Issue #277 - Extract GPU shader constants to src/analysis/gpu/shaders.rs

## Summary

Extracted GPU shader code and related constants from `analyzer.rs` and `queue.rs` to a dedicated `src/analysis/gpu/shaders.rs` module as part of the ongoing refactoring of the implementation.rs monolith (parent issue #185).

This extraction centralises GPU configuration and makes it easier to find and modify threshold values. The new module groups all shader-related constants together with comprehensive documentation about their purpose and valid ranges.

## Changes

### New File: `src/analysis/gpu/shaders.rs` (~250 lines)

**Shader Source References:**
- `HELPFUL_SHADER` - Reference to helpful.wgsl for helpful synapse analysis
- `HARMFUL_SHADER` - Reference to harmful.wgsl for harmful synapse analysis
- `RELU_SHADER` - Reference to relu.wgsl for ReLU activation analysis
- `ACTIVATION_SHADER` - Reference to activation.wgsl for general activation analysis
- `BIAS_SHADER` - Reference to bias.wgsl for bias optimisation

**Workgroup Configuration:**
- `WORKGROUP_SIZE` (256) - GPU workgroup size, must match @workgroup_size in shaders

**GPU Timing Constants:**
- `GPU_INIT_TIMEOUT_SECS` (30) - Device initialisation timeout
- `GPU_SHUTDOWN_TIMEOUT_SECS` (10) - Graceful shutdown timeout

**Analysis Threshold Constants:**
- `MIN_NEURON_SAMPLE_COUNT` (10) - Minimum samples for valid analysis

### Updated Files

**`src/analysis/gpu/mod.rs`:**
- Added `pub mod shaders;` declaration
- Added re-exports for all shader constants
- Updated module structure documentation

**`src/analysis/gpu/analyzer.rs`:**
- Removed local constant definitions (moved to shaders.rs)
- Added import from shaders module
- Retained `GPU_MAX_BATCH_ALLOC_BYTES` (batch-specific constant)

**`src/analysis/gpu/queue.rs`:**
- Removed local `GPU_SHUTDOWN_TIMEOUT_SECS` definition
- Added import from shaders module

## Evidence

Unable to generate screenshot: This is a CLI library with no visual interface. The changes are verified through:
- All existing tests pass (`./quality.sh`)
- New unit tests verify constant values and shader content

## Test Plan

### New Tests Added (`src/analysis/gpu/shaders.rs`)

1. `test_shader_sources_are_not_empty` - Verify all shaders have content
2. `test_shader_sources_contain_workgroup_size` - Verify shaders declare correct @workgroup_size(256)
3. `test_workgroup_size_is_valid` - Verify workgroup size is power of 2 between 64-1024
4. `test_gpu_timeouts_are_valid` - Compile-time assertions for timeout ranges
5. `test_min_neuron_sample_count_is_valid` - Verify sample count is reasonable (5-100)
6. `test_shader_wgsl_syntax_basics` - Verify shaders contain required struct/fn/@compute declarations

### New Tests Added (`src/analysis/gpu/mod.rs`)

7. `test_shader_constants_are_exported` - Verify all shader constants accessible via gpu module

### Existing Tests

All 330+ existing unit tests and integration tests continue to pass, verifying no regressions.

## Module Structure After Changes

```text
src/analysis/gpu/
├── mod.rs          <- Module router and re-exports
├── shaders.rs      <- GPU shader constants and references (NEW - Issue #277)
├── device.rs       <- GPU device management (Issue #272)
├── analyzer.rs     <- GpuAnalyzer struct and GpuEvaluator trait (Issue #273)
└── queue.rs        <- GpuWorkQueue struct and thread management (Issue #274)
```

## Benefits

1. **Centralised GPU configuration** - All shader-related constants in one place
2. **Easier threshold tuning** - Constants are documented with valid ranges
3. **Better documentation** - Relationships between constants and shaders are clear
4. **Extensibility** - Prepares for potential shader hot-reloading in future
