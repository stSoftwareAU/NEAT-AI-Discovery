## Summary

Add two new discovery detection modules that improve candidate discovery for NEAT-AI networks. Closes #543.

### 1. Activation Mismatch Detection (`detection/activation_mismatch.rs`)

Detects neurons whose activation function is structurally mismatched for their observed data:

- **RELU Negative Bias**: When ≥70% of a RELU neuron's pre-activation values are negative, the activation clips most of the signal to zero. Recommends switching to ELU which preserves negative information.
- **Bounded Underutilisation**: When a bounded activation (TANH, LOGISTIC, etc.) only uses <15% of its theoretical output range, the neuron operates entirely in the linear region. Recommends IDENTITY as it would be equally effective without the computational overhead.

This complements the existing `activation_recommendation` module (Issue #417) which matches input distributions to activations proactively. The mismatch detector focuses on **information loss** — cases where the activation function actively wastes signal.

### 2. Observation Utilisation Detection (`detection/observation_utilisation.rs`)

Builds on the existing `observation_range` module (Issue #398) to generate actionable candidates for input neurons with low effective utilisation due to sentinel values:

- Reuses sentinel detection logic from `observation_range.rs`
- When an input neuron has sentinel values (e.g., -1.0 for "no data") occupying a significant portion of its range, recommends bias adjustments on downstream neurons to compensate for the sentinel-induced offset
- Helps the network focus on the effective data range rather than being influenced by meaningless sentinel clusters

Both modules follow the standard discovery dispatch pattern and run in parallel with all other detection modules via `run_discovery_modules_parallel()`.

## Evidence

This is a backend-only change (no UI). Evidence of correctness:

- All 15 new tests pass
- All 496 existing unit tests pass
- All 99 existing integration tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

### New integration tests
- `tests/issue_543_activation_mismatch.rs` (9 tests):
  - RELU with mostly negative pre-activation detected
  - RELU with balanced pre-activation NOT detected
  - TANH with narrow output range detected as underutilised
  - TANH with full range NOT detected
  - Insufficient samples returns empty
  - Neuron without records returns empty
  - Coordinated candidate conversion produces correct operations
  - IDENTITY never flagged as mismatched
  - Multiple mismatched neurons detected independently

- `tests/issue_543_observation_utilisation.rs` (6 tests):
  - Input with sentinel cluster detected as underutilised
  - Input without sentinels NOT detected
  - Too few samples returns empty
  - Hidden neurons ignored (only input neurons)
  - Coordinated candidate conversion produces SetBias operations
  - Multiple underutilised inputs detected independently

### New unit tests (inline)
- `activation_mismatch.rs`: 5 unit tests for helper functions
- `observation_utilisation.rs`: 2 unit tests for empty records and conversion
