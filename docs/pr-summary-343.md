## Summary

Implements bottleneck neuron detection (Issue #343) to identify hidden neurons where many
input signals converge through a single neuron before reaching outputs. These bottleneck
neurons limit the network's ability to represent complex input combinations because one
neuron's activation range must encode all upstream information.

When a bottleneck is detected, the library recommends structural changes:
- **Add parallel neuron**: Create a new hidden neuron sharing a subset of inputs/outputs
  to widen the information channel
- **Add bypass synapse**: Add a direct connection from upstream to downstream, reducing
  dependency on the bottleneck

All candidates are emitted as `coordinatedStructuralCandidates` using existing operation
types (`addNeuron`, `addSynapse`), so no changes to NEAT-AI are required.

### Detection Criteria
- **Hidden neurons only** — output neurons are natural convergence points and excluded
- **Fan-in ≥ 3** — minimum incoming connections to be considered
- **Fan-in/fan-out ratio ≥ 2.0** — significantly more inputs than outputs
- **Minimum 20 samples** — for reliable detection
- **Bottleneck score** combines topology (60%) and error concentration (40%)

## Evidence

Unable to generate screenshot: This is a CLI library with no visual interface.

## Test Plan

Added 13 tests in `tests/issue_343_bottleneck_neuron_detection.rs`:

1. `test_detects_bottleneck_with_high_fan_in` — 5 inputs → 1 hidden → 2 outputs detected
2. `test_does_not_flag_wide_topology` — fan-in=1 neurons not flagged
3. `test_output_neurons_not_flagged` — output neurons excluded
4. `test_input_neurons_not_flagged` — input neurons excluded
5. `test_insufficient_samples_not_flagged` — <20 samples skipped
6. `test_candidates_produce_coordinated_operations` — correct JSON output
7. `test_higher_fan_in_ratio_scores_higher` — fan-in=8 ranks above fan-in=2
8. `test_error_contribution_increases_score` — high error → positive ratio
9. `test_pass_through_neuron_not_bottleneck` — fan-in=1 not flagged
10. `test_bypass_synapse_candidate` — bypass adds direct upstream→downstream synapse
11. `test_parallel_path_candidate` — parallel adds new neuron with shared wiring
12. `test_no_records_for_neuron` — graceful handling of missing records
13. `test_bottleneck_score_considers_errors` — error magnitude affects score

All existing tests continue to pass. `./quality.sh` passes cleanly.
