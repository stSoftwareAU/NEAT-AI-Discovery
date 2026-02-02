## Summary

This PR completes Issue #239 by creating a unified `system.rs` module that serves as a facade for all system-related utilities including memory detection, system requirements checking, and GPU performance tier detection.

### Background

The memory and system utility functions mentioned in Issue #239 were already extracted to more specialised modules during earlier refactoring phases:
- `utils/memory.rs` - Platform-specific memory detection (Issue #267)
- `gpu/device.rs` - GPU performance tier and unified memory detection (Issue #272)
- `gpu/analyzer.rs` - GPU batch size configuration (Issue #273)

### What This PR Adds

Rather than consolidating all code into a single file (which would violate the Single Responsibility Principle), this PR creates a **facade module** (`src/analysis/system.rs`) that:

1. **Re-exports** all system-related types and functions from their implementation modules
2. **Documents** the module organisation and available functionality
3. **Provides backward compatibility** for callers who expect a single access point for system utilities
4. **Adds comprehensive module-level documentation** explaining what's available

### Key Components Re-exported via `system.rs`

**Memory Detection:**
- `get_memory_info()` - Platform-specific memory detection (macOS vm_stat, Linux /proc/meminfo)
- `MemoryTier` enum (Low/Standard/High)
- `detect_memory_tier()` - Cached memory tier detection
- `categorise_memory_tier()` - Pure function for testing

**Parquet Memory Validation:**
- `check_memory_for_parquet()` - Validate memory before loading parquet files
- `validate_parquet_memory_requirements()` - Testable pure function

**System Requirements:**
- `check_system_memory_requirements()` - Validate minimum system requirements

**GPU Performance:**
- `GpuPerformanceTier` enum (High/Standard/Unknown)
- `detect_gpu_tier()` - Detect GPU tier from adapter info
- `detect_unified_memory()` - Detect unified memory architecture (Apple Silicon)

**GPU Batch Size:**
- `DEFAULT_GPU_BATCH_SIZE`, `HIGH_PERF_GPU_BATCH_SIZE`, `LOW_MEMORY_GPU_BATCH_SIZE` constants
- `cap_gpu_batch_size_by_bytes()` - Cap batch size based on memory constraints
- `get_work_queue_capacity()`, `get_work_queue_capacity_for_tier()` - Work queue capacity utilities

**Platform-Specific (Conditionally Compiled):**
- macOS: `parse_vm_stat_line`, `parse_vm_stat_page_size`
- Linux: `parse_meminfo_line`

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface.

All 323 unit tests pass, plus 89 integration tests covering:
- Memory tier classification
- GPU performance tier detection
- Batch size calculations
- Platform-specific parsing functions

## Test Plan

### New Tests Added in `src/analysis/system.rs`

- `test_memory_tier_re_exports_accessible` - Verifies memory tier types are accessible via the facade
- `test_gpu_performance_tier_re_exports_accessible` - Verifies GPU tier types are accessible
- `test_batch_size_constants_accessible` - Verifies batch size constants are reachable (compile-time assertions)
- `test_work_queue_capacity_accessible` - Verifies work queue capacity functions work correctly

### Existing Tests Verified

All existing tests in the following modules continue to pass:
- `src/analysis/utils/memory_tests.rs` - Memory tier and parquet validation tests
- `src/analysis/gpu/device.rs` - GPU detection tests
- `tests/unit/analysis_implementation.rs` - GPU tier detection and batch size tests

### Quality Verification

- `./quality.sh` passes all checks:
  - Build (debug and release)
  - Formatting (rustfmt)
  - Linting (clippy)
  - All unit tests (323 passed)
  - All integration tests (89 test files passed)
