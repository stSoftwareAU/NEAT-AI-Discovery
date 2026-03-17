## Summary

Audit and add meaningful assertions to smoke-only tests across the codebase.
Closes #814.

For each assertion-free test, either:
- Added real assertions that verify observable output (metric values, JSON
  structure, elapsed time), or
- Added an explicit smoke test comment explaining why a no-panic check is the
  strongest available contract (for purely side-effecting functions with no
  return value or queryable state).

No dummy assertions (`assert!(true)`) were introduced. Australian English
used throughout.

### Files changed

| File | Tests updated | Approach |
|------|--------------|----------|
| `tests/issue_575_structured_logging.rs` | 4 | 3 upgraded with real assertions (`elapsed_ms`, `GpuMetrics` counters, `to_json` output); 1 documented as smoke test (global tracing subscriber init) |
| `tests/issue_713_deduplicate_gpu_env_setup.rs` | 3 | Documented as smoke tests — platform env-var setters with no return value |
| `src/observability.rs` | 2 | Replaced empty `profile_mode_default` with `profile_data_to_json_includes_timing_structure`; added `elapsed_ms` assertion to `phase_timer` test |
| `src/analysis/utils/platform.rs` | 2 | Documented as smoke tests — `Once`-guarded env-var setters |

### Other files audited (no changes needed)

- `src/analysis/discovery_dispatch_tests.rs` — all 10 tests already have assertions
- `src/analysis/synapse/tests.rs` — all 15 tests already have assertions
- `src/analysis/mod_tests.rs` — all 8 tests already have assertions
- `src/analysis/implementation_tests/` — all ~80 tests already have assertions

## Evidence
- `quality.sh` passes cleanly with all changes

## Test Plan
- Verified all modified tests pass via `quality.sh` (fmt, clippy, check, test, doc, release build)
- No tests removed — only upgraded or documented
