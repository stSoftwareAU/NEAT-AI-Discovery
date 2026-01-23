## Summary

Fixed GPU detection failing on macOS due to an overly strict memory availability check.

The bug occurred when a Mac reported low "available" memory (e.g., 1.0GB on an 8GB system) due to macOS's aggressive file caching. The library's 1GB minimum threshold would fail even though macOS can quickly reclaim this cached memory when needed.

### Root Cause

1. **macOS memory reporting**: macOS aggressively caches files in RAM, making "available" memory appear artificially low
2. **Strict threshold**: The library used a 1GB minimum on all platforms
3. **Misleading error**: Memory check failure returned `is_error: false`, so NEAT-AI reported "no usable GPU detected" instead of the actual memory issue

### Changes

1. **Platform-specific memory thresholds**:
   - macOS: 0.5GB minimum (down from 1GB) - safe due to unified memory and quick cache reclamation
   - Linux: 1GB minimum (unchanged) - conservative for headless servers

2. **Error status on macOS**: Memory check failure now sets `is_error: true` on macOS so callers receive proper error messaging about the memory issue, not "no usable GPU"

3. **Improved precision**: Available memory now displayed with 2 decimal places (e.g., `0.95GB` instead of `1.0GB`) to avoid confusion when values are near thresholds

### Files Changed

- `src/analysis/utils/memory.rs` - Platform-specific `MINIMUM_AVAILABLE_MEMORY_GB` constant and improved documentation
- `src/analysis/gpu/analyzer.rs` - Platform-specific `is_error` flag in `check_minimum_system_requirements()`
- `src/analysis/utils/memory_tests.rs` - Platform-specific tests for memory thresholds

## Evidence

Unable to generate screenshot: This is a CLI/library tool with no visual interface.

The fix addresses the root cause identified in the issue logs:
```
[NEAT-AI-Discovery] Memory check failed: 1.0GB available / 8.0GB total. Discovery disabled.
```

With this fix:
- Macs with 0.5-1GB reported available memory will pass the memory check
- If memory check still fails, the error message will correctly identify it as a memory issue, not "no usable GPU"

## Test Plan

Added platform-specific tests to verify the memory threshold behaviour:

- `test_system_requirements_low_available_memory` - Updated to use 400MB (below all thresholds)
- `test_system_requirements_macos_lower_threshold` - Verifies 0.6GB passes on macOS
- `test_system_requirements_macos_edge_case` - Verifies edge cases at 0.51GB and 0.49GB
- `test_system_requirements_linux_stricter_threshold` - Verifies Linux still uses 1GB threshold

All existing tests continue to pass. Run `./quality.sh` to verify.
