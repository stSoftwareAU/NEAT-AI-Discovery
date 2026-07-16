## Summary

Updated all project documentation to reflect recent changes from issues #1001–#1029. Closes #1032.

### Documentation updates

- **README.md**: Added `discovery_memory_usage_bytes` to FFI API table, added `NEAT_AI_DISCOVERY_MH_TEMPERATURE` environment variable, updated discovery pipeline diagram to include temperature-scaled acceptance step, added MCMC audit document to additional documentation index
- **AGENTS.md**: Added `temperature.rs`, `adaptive_proposal.rs`, and `mcmc_diagnostics.rs` to source layout; updated test count (~286 files) and benchmark count (31 suites); added `NEAT_AI_DISCOVERY_MH_TEMPERATURE` to environment variables table
- **docs/FFI_API.md**: Added new section documenting `max_analysis_memory_mb`, `analysis_deadline_ms`, and `temperature` analysis parameters
- **CHANGELOG.md**: Added v0.72.34 entry covering memory budget enforcement, deadline enforcement, memory usage FFI, MCMC-inspired candidate selection, concurrent analysis, parallel compression, and new benchmarks

### Housekeeping

- Archived 15 PR summaries (pr-summary-998 through pr-summary-1029) to `docs/archive/pr-summaries/`

## Evidence

- All 6 new documentation accuracy tests pass, verifying that documented modules, FFI functions, and configuration accessors exist
- `./quality.sh` passes cleanly with all tests passing

## Test Plan

- Added `tests/infrastructure/issue_1032_documentation_accuracy.rs` with 6 tests:
  - `documented_temperature_module_exists` — verifies temperature scheduling module is accessible
  - `documented_adaptive_proposal_module_exists` — verifies adaptive proposal module is accessible
  - `documented_ffi_memory_usage_exists` — verifies `discovery_memory_usage_bytes` FFI function exists
  - `documented_mh_temperature_config_exists` — verifies MH temperature config accessor exists
  - `temperature_linear_cooling_produces_decreasing_values` — verifies linear cooling produces monotonically decreasing temperatures
  - `temperature_exponential_cooling_produces_decreasing_values` — verifies exponential cooling produces monotonically decreasing temperatures
