## Summary

Split `src/analysis/diagnostics.rs` (1,651 lines) into a focused `diagnostics/` directory module following the established pattern (synapse/, focus/, gpu/). Closes #524.

### New module structure

```
src/analysis/diagnostics/
├── mod.rs              # Public API, re-exports, RecordCacheProvider, impact scoring (532 lines)
├── rejection.rs        # Synapse rejection tracking and reporting (458 lines)
├── neuron_tracking.rs  # Neuron rejection tracking and reporting (417 lines)
├── target_data.rs      # Target data structures for sample building (152 lines)
└── focus_filter.rs     # Focus target filtering and validation (151 lines)
```

All sub-modules are under 600 lines. All existing diagnostic tests pass unchanged.

## Evidence

This is a pure refactoring (module split) with no UI or performance changes. Evidence:
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- All existing tests pass unchanged with `--test-threads=1`
- No API changes — all existing `crate::analysis::diagnostics::*` import paths continue to work

## Test Plan

- All existing inline unit tests preserved in `diagnostics/mod.rs` (13 tests)
- All existing `implementation_tests/diagnostics_tests.rs` tests pass unchanged (9 tests)
- All integration tests in `tests/` that reference diagnostics pass unchanged
- Concurrent diagnostic insertion tests (Issue #216) continue to verify thread safety
