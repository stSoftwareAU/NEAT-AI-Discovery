## Summary

Adds sentinel value gating discovery module (Issue #400) that detects observations
where sentinel values (e.g., -1 meaning "no data") degrade performance, and proposes
gated neuron structures using GPU-compatible scalar squash functions.

**Key design decisions:**

- **Error-correlation analysis**: Unlike the simpler bounded_range (#395) cluster
  detection, this module requires the sentinel cluster to have **lower error variance**
  than the useful range before proposing gating. This prevents gating sentinels that
  are actually informative.
- **STEP activation gate**: Uses `STEP` squash function (outputs 1 when input >= 0,
  0 otherwise) with a computed bias that places the threshold between the sentinel
  and useful range. This is scalar and GPU-compatible (no `IF` aggregate function).
- **Full downstream wiring**: The coordinated structural candidate includes synapses
  from the gate neuron to all downstream targets of the observation, so the gate
  can modulate the observation's influence on the network.
- **Reuses existing candidate types**: All operations use `coordinatedStructural`
  with `addNeuron` + `addSynapse` — no new candidate types required.
- **DRY dispatch**: Integrated via `run_discovery_module` pattern from Issue #375.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.
The module produces `coordinatedStructural` JSON candidates consumed by NEAT-AI.

## Test Plan

Added `tests/issue_400_sentinel_value_gating.rs` with 10 tests:

1. `test_detects_sentinel_gating_candidate_at_minus_one` — sentinel at -1 with noise
2. `test_coordinated_candidate_has_required_operations` — addNeuron + addSynapse present
3. `test_gate_neuron_uses_gpu_compatible_squash` — STEP is scalar/GPU-compatible
4. `test_gate_connects_to_downstream_targets` — gate wired to downstream targets
5. `test_no_detection_without_sentinel_cluster` — uniform distribution not flagged
6. `test_no_detection_with_insufficient_samples` — minimum sample threshold
7. `test_multiple_observations_produce_independent_candidates` — independent gating
8. `test_detects_sentinel_at_zero` — sentinel at 0.0
9. `test_output_neurons_excluded` — only input neurons analysed
10. `test_uses_error_correlation_for_detection` — high-variance sentinel not gated

All tests pass via `./quality.sh` (fmt, clippy, check, test, release build).
