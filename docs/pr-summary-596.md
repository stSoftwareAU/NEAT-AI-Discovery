## Summary

Split `ffi_types.rs` (~1,187 lines) into focused sub-modules under `src/ffi_types/`. Closes #596.

The original monolithic file has been replaced with 7 focused sub-modules:

| Module | Purpose | Lines |
|--------|---------|-------|
| `creature.rs` | Creature, neuron, synapse JSON representations | ~68 |
| `requests.rs` | FFI request structs (input from NEAT-AI) | ~168 |
| `responses.rs` | FFI response structs (output to NEAT-AI) | ~168 |
| `candidates.rs` | Candidate-related types (synapse, neuron, coordinated) | ~256 |
| `session.rs` | Streaming session types | ~98 |
| `diagnostics.rs` | Diagnostic, metadata, GPU info, timing types | ~272 |
| `conversions.rs` | Conversion helpers (internal analysis types → JSON) | ~147 |
| `mod.rs` | Public API, re-exports for backward compatibility | ~20 |

All public types are re-exported at the `ffi_types::` level via `mod.rs`, preserving full backward compatibility. No consumer code changes required.

## Evidence

This is a pure refactoring change with no visual or behavioural changes. Evidence:
- `./quality.sh` passes cleanly (fmt, clippy, check, all tests, release build)
- All existing tests pass without modification
- Public API is unchanged (wildcard re-exports preserve compatibility)
- FFI boundary contract is not altered

## Test Plan

- All existing integration and unit tests pass unchanged
- No new tests needed — this is a structural refactoring that preserves identical behaviour
- Verified via `cargo test --lib --tests --all-features -- --test-threads=1`
