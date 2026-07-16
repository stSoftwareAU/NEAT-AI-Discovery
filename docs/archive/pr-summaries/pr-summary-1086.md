## Summary

Enrich error context across the analysis pipeline by adding `.context()` annotations to all major phase transitions, GPU queue creation, parquet cache loading, and per-neuron/synapse analysis paths. This produces readable error chains (e.g., "failed during synapse analysis phase: GPU batch evaluation timed out") that help the TypeScript host trace error origins through the multi-stage pipeline. Closes #1086.

No new dependencies added — uses existing `anyhow::Context`.

## Changes

- **`src/analysis/orchestration.rs`** — Added context to `run_optional_analysis` (phase name in errors), `dispatch_analyses` (synapse/neuron phase identification), GPU queue creation, parquet cache loading, and analysis dispatch error paths.
- **`src/analysis/neuron/mod.rs`** — Added context to input validation, parquet cache loading, GPU queue creation, analysis preparation, source record loading (includes neuron UUID), and GPU evaluation (includes neuron UUID).
- **`src/analysis/synapse/mod.rs`** — Added context to input validation, parquet cache loading, and GPU queue creation.
- **`src/analysis/synapse/orchestration.rs`** — Added context to input validation and per-target synapse analysis (includes target neuron UUID).
- **`src/analysis/gpu/queue/scheduling.rs`** — Added context to GPU analyser initialisation failure path.
- **`src/analysis/gpu/queue/mod.rs`** — Improved error message for disconnected GPU response channel.
- **`src/analysis/implementation_tests/synapse_analysis_tests.rs`** — Updated three existing tests to use `{err:#}` (alternate display) so they check the full error chain rather than just the outermost context message.

## Evidence

This is a backend/library change with no visual output. Correctness verified via:
- Two new unit tests for `run_optional_analysis` error context enrichment
- All 758 existing tests pass (3 tests updated to use `{err:#}` for full chain assertions)
- Full `./quality.sh` passes cleanly

## Test Plan

- Added `run_optional_analysis_error_includes_phase_context` — verifies phase name appears in error chain
- Added `run_optional_analysis_success_does_not_add_spurious_context` — verifies successful phases return clean results
- Updated `analyze_neurons_rejects_duplicate_focus_targets` to check full error chain
- Updated `analyze_synapses_rejects_duplicate_focus_targets` to check full error chain
- Updated `analyze_synapses_requires_focus_targets` to check full error chain
