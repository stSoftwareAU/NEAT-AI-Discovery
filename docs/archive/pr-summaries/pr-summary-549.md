## Summary

Add topology diversification detection for structural jumps — detecting when the network topology is too simple for the observed error patterns and recommending `addNeuron` candidates at strategic insertion points. Closes #549.

Local minimums often result from insufficient network topology. Weight and bias adjustments operate in a fixed-dimension space. Adding neurons changes the dimensionality of the solution space, allowing the network to reach solutions unreachable by parameter tuning alone.

### What it does

- **Detects insufficient non-linear depth**: Identifies output neurons where all paths from inputs pass through zero hidden neurons (direct connections only) while error remains high
- **Filters out parametric issues**: Skips detection when hidden neurons on the path have high individual error variance (indicating the problem is parametric, not structural)
- **Generates `addNeuron` candidates**: Recommends inserting a TANH hidden neuron between the most informative input and the high-error output
- **Selects best insertion point**: Picks the input with highest activation variance as the source for the new neuron

### New files

- `src/analysis/detection/topology_diversification.rs` — Detection logic and candidate conversion
- `tests/issue_549_topology_diversification.rs` — 8 integration tests

### Modified files

- `src/analysis/detection/mod.rs` — Register new module
- `src/analysis/mod.rs` — Add re-export and dispatch registration in `analyze_all()`

## Evidence

This is a backend detection module with no UI. Verified by 8 passing integration tests covering all acceptance criteria.

## Test Plan

- `test_detects_insufficient_topology_direct_paths` — Verifies detection of under-connected direct input→output network with high error
- `test_no_detection_for_well_connected_network` — Verifies no false positives for networks with multiple hidden layers
- `test_no_detection_for_low_error_direct_path` — Verifies no detection when error is already low (topology is adequate)
- `test_insufficient_samples_returns_empty` — Verifies minimum sample count requirement
- `test_coordinated_candidates_contain_add_neuron` — Verifies candidates contain `addNeuron` and `addSynapse` operations with issue reference
- `test_multiple_outputs_mixed_results` — Verifies correct detection across multiple outputs with mixed structural needs
- `test_no_detection_when_individual_neurons_have_issues` — Verifies no false positive when hidden neurons have high error variance (parametric issue)
- `test_empty_creature_returns_empty` — Verifies empty input handling
