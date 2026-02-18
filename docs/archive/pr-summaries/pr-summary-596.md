## Summary

Split `src/ffi_types.rs` (~1,187 lines) into focused sub-modules under `src/ffi_types/`. Closes #596.

The new structure groups related FFI types by concern:

| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | ~95 | Shared types (CreatureJson, NeuronJson, SynapseJson, NeuronData, TrainingRecord, NeuronStatsJson), re-exports |
| `requests.rs` | ~175 | All `*Input` request structs |
| `responses.rs` | ~395 | All `*Output` response structs, diagnostics, metadata, timing, conversion helpers |
| `candidates.rs` | ~245 | Candidate types (synapse, neuron, coordinated structural) |
| `session.rs` | ~100 | Streaming session types (start/append/finish/cancel) |

All sub-modules are re-exported from `mod.rs`, preserving full backward compatibility. No changes to the public API or FFI boundary contract.

## Evidence

This is a pure structural refactoring with no behavioural changes:
- `cargo build` — clean
- `cargo clippy` — no warnings
- `cargo test` — all tests pass (no test modifications needed)
- `./quality.sh` — passes cleanly including release build

## Test Plan

- All existing tests pass without modification — the re-exports preserve the public API
- No new tests needed (structural refactoring only, no logic changes)
