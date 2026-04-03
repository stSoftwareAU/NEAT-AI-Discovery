## Summary

Split `src/analysis/synapse/scoring.rs` (877 lines) into a `scoring/` module directory with focused sub-modules, each handling a single scoring concern. Closes #982.

### New module structure

| File | Responsibility | Lines |
|------|---------------|-------|
| `scoring/mod.rs` | Module declarations, re-exports for backward compatibility | 43 |
| `scoring/boost_functions.rs` | Source-type, target-type, and activation-function-aware boosts | 84 |
| `scoring/improvement.rs` | Core improvement calculation algorithm and candidate dedup | 416 |
| `scoring/discounting.rs` | Pessimism discounting and prediction calibration | 137 |
| `scoring/test_helpers.rs` | Test-only wrapper functions (`#[cfg(test)]`) | 150 |
| `scoring/tests.rs` | Unit tests for improvement functions | 127 |

All public API remains accessible through the same `analysis::synapse::` paths — no import changes required in consuming code.

## Evidence
- All 158 lib+test suite tests pass
- All 248 scoring integration tests pass (including 10 new backward-compatibility tests)
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)

## Test Plan
- Added `tests/scoring/issue_982_scoring_module_split.rs` with 10 tests verifying:
  - All 3 boost functions accessible via `analysis::synapse::` path
  - All 4 discounting functions accessible via `analysis::synapse::` path
  - Discount ordering invariant preserved (synapse <= neuron <= generic)
- Existing inline unit tests preserved in `scoring/tests.rs`
- All existing integration tests pass without modification
