## Summary

Add per-output error disaggregation detector for hidden neurons. This new detection module analyses the full `errors[0..n]` vector for each hidden neuron to identify cross-output conflicts — cases where a neuron helps some outputs but actively harms others. Closes #639.

### What changed

- **New detection module** `src/analysis/detection/output_conflict.rs`: Computes per-output mean error for each hidden neuron and flags those with sign conflicts (negative on some outputs, positive on others). Produces `CoordinatedStructuralCandidateJson` candidates recommending either weight attenuation (SetWeight) or compensating gating neurons (AddNeuron + AddSynapse).
- **Dispatch wiring** in `src/analysis/module_dispatch_specs/structural_specs.rs`: The module runs in parallel with other structural discovery modules, loading hidden neuron records from the shared cache.
- **Module registration** in `src/analysis/detection/mod.rs` and `src/analysis/mod.rs`: Re-exported at the analysis level for backward compatibility.

## Evidence

This is a backend detection module with no UI. Verified by 11 integration tests covering all acceptance criteria.

## Test Plan

Added `tests/issue_639_output_conflict_detection.rs` with 11 tests:

1. `test_conflicting_hidden_neuron_detected` — hidden neuron helping one output but harming another is flagged
2. `test_consistently_helpful_not_flagged` — uniformly helpful neuron is not flagged
3. `test_only_hidden_neurons_analysed` — input/output neurons excluded
4. `test_insufficient_samples_no_detection` — below minimum sample threshold
5. `test_single_output_no_detection` — single-output networks produce no detections
6. `test_produces_valid_coordinated_candidates` — valid structural candidates generated
7. `test_multiple_conflicting_neurons_all_flagged` — multiple conflicts all detected
8. `test_weak_conflict_not_flagged` — noise-level conflicts ignored
9. `test_results_sorted_by_severity` — worst conflicts ranked first
10. `test_empty_records_no_detections` — empty input handled gracefully
11. `test_three_output_partial_conflict` — partial conflict in 3-output network detected

All tests pass. `./quality.sh` passes cleanly.
