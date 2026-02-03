## Summary

Added hidden neuron operating-point analysis module (Issue #401) that detects neurons working outside their squash function's effective zone. Unlike the restricted range module (#399) which analyses post-activation output range, this module analyses the **pre-activation (`value`) distribution** against each squash function's active zone to measure dynamic range utilisation.

For example, a LOGISTIC neuron with a very negative bias always outputs ~0, wasting the sigmoid's S-curve. A TANH neuron with tiny incoming weights only uses the linear region near zero.

### Changes

- **`src/analysis/operating_point.rs`** (new): Detection module with:
  - `detect_operating_point_issues()` — analyses pre-activation distribution vs squash active zone
  - `operating_point_to_coordinated_candidates()` — generates `setBias`, `changeSquash`, and `setWeight` candidates
  - `OperatingPointIssue` struct with dynamic range utilisation metric
  - Reuses `target_simulation_fn()` from `src/activations.rs` for squash function evaluation
  - Reuses `get_bias_range()` concepts from `src/analysis/activation.rs` for active zone bounds
- **`src/analysis/mod.rs`**: Registered module and added `run_discovery_module` dispatch
- **`tests/issue_401_operating_point_analysis.rs`** (new): 11 integration tests

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

11 tests covering detection and candidate generation:
1. `test_detects_logistic_neuron_with_negative_operating_point` — LOGISTIC with very negative bias detected
2. `test_detects_tanh_neuron_using_only_linear_region` — TANH confined to linear region detected
3. `test_well_placed_tanh_not_flagged` — well-placed neuron not flagged
4. `test_output_neurons_excluded` — output neurons excluded
5. `test_input_neurons_excluded` — input neurons excluded
6. `test_insufficient_samples_skipped` — fewer than 20 samples skipped
7. `test_records_without_value_skipped` — records without pre-activation value skipped
8. `test_unbounded_activations_excluded` — IDENTITY/RELU excluded
9. `test_multiple_neurons_only_poorly_placed_detected` — only poorly-placed neurons flagged
10. `test_candidate_generation_includes_expected_operations` — setBias, changeSquash, setWeight generated
11. `test_configurable_utilisation_threshold` — custom threshold works
