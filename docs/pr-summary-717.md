## Summary

Create a central configuration module (`src/config.rs`) that documents all environment variables in one location, provides typed accessor functions with validation, and updates existing call sites to use the new accessors. Closes #717.

### What changed

- **New `src/config.rs`**: Central module documenting all 24+ runtime environment variables with their purpose, type, default value, and valid range. Provides typed accessor functions (e.g., `config::watchdog_stall_timeout() -> Option<Duration>`, `config::block_size() -> usize`) that parse, validate, and cache values.

- **Updated 13 source files** to delegate their environment variable reads to `config::*` accessors:
  - `analysis/utils/mod.rs` — `verbose_enabled()`, `gpu_timing_enabled()`
  - `observability.rs` — `timing_enabled()`, `gpu_metrics_enabled()`, `profile_mode()`
  - `watchdog.rs` — `WatchdogConfig::from_env()`
  - `analysis/streaming.rs` — `get_streaming_config_from_env()`, `is_streaming_enabled()`, `get_block_size()`
  - `analysis/gpu/analyzer.rs` — `get_batch_size_override()`
  - `analysis/gpu/queue/recovery.rs` — `get_gpu_retry_limit()`
  - `analysis/scoring/error_distribution.rs` — `outlier_analysis_enabled()`, `outlier_percentile_from_env()`
  - `analysis/neuron/preparation.rs` — output-only targets check
  - `analysis/utils/deadline.rs` — `source_input_index_bias_from_env()`, `focus_unused_observations_from_env()`
  - `analysis/shared.rs` — `ZeroCopyBufferConfig::from_env()`
  - `analysis/samples/thresholds.rs` — `constant_source_effect_threshold_from_env()`, `get_constant_source_threshold()`
  - `analysis/detection/input_sensitivity.rs` — dominance/gradient threshold
  - `analysis/detection/noise_signal.rs` — noise-signal threshold
  - `analysis/utils/platform.rs` — quiet GPU check
  - `debug.rs` — verbose check, sample program path

- **Backward compatibility preserved**: All existing public functions continue to exist as thin wrappers that delegate to `config::*`.

## Evidence

This is a backend/library change with no visual output. Evidence:
- All 31 new integration tests pass (`tests/issue_717_config_env_vars.rs`)
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
- All existing tests continue to pass

## Test Plan

- Added `tests/issue_717_config_env_vars.rs` with 31 integration tests covering:
  - Block size (default, custom, clamped low, clamped high)
  - Outlier configuration (percentile default/custom/invalid, analysis enabled/disabled)
  - Watchdog (stall timeout disabled/enabled, abort delay default)
  - Prefetch depth (default, custom)
  - Preload all / streaming enabled (default, enabled)
  - Constant source effect threshold (default, disabled, custom)
  - Source input index bias (default, enabled, negative rejection)
  - Zero-copy override (auto, force enable, force disable)
  - Detection thresholds (noise-signal default/custom)
  - Sample program (default, custom)
  - Focus unused observations (default, enabled)
- Added unit tests in `src/config.rs` for parsing logic, defaults, and dynamic threshold calculation
