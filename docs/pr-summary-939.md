## Summary

Split `src/analysis/candidate_compression.rs` (1,136 lines) into a focused
`candidate_compression/` directory with four sub-modules. Closes #939.

### New module structure

| Module | Purpose | Lines |
|--------|---------|-------|
| `mod.rs` | Public API, `CompressibleGroup`, `generate_compression_uuid`, re-exports | 129 |
| `grouping.rs` | Candidate grouping and deduplication by target neuron | 140 |
| `identity.rs` | IDENTITY neuron compression logic (Issue #921) | 443 |
| `nonlinear.rs` | TANH/GELU non-linear compression logic (Issue #922) | 439 |
| `gain_estimation.rs` | Saturation-aware gain estimation helpers | 135 |

Business logic in each module is under 150 lines; the remaining lines are
comprehensive unit tests with test helper functions.

### Public API unchanged

All existing `use` paths continue to work via re-exports in `mod.rs`:
- `compress_identity_candidates`
- `compress_nonlinear_candidates`
- `detect_compressible_groups`
- `generate_compression_uuid`
- `CompressibleGroup`

## Evidence

- All 25 unit tests pass (unchanged, distributed across sub-modules)
- All 19 integration tests pass without modification
- `./quality.sh` passes with no warnings

## Test Plan

- Existing unit tests preserved and distributed to their respective modules:
  - `grouping::tests` — 5 tests for group detection
  - `identity::tests` — 7 tests for IDENTITY compression
  - `nonlinear::tests` — 7 tests for non-linear compression
  - `gain_estimation::tests` — 1 test for benefit ratio filtering
  - `mod::tests` — 4 tests for UUID generation
- Existing integration tests pass unmodified:
  - `tests/analysis/issue_921_candidate_compression.rs` (12 tests)
  - `tests/analysis/issue_922_nonlinear_candidate_compression.rs` (7 tests)
