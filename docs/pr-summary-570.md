## Summary

Add skip-connection discovery module that identifies beneficial residual connections
across layers in deep NEAT networks. The module analyses topological depth from inputs
and detects gradient attenuation in deep neurons (depth > 2), then suggests addSynapse
candidates connecting shallow sources (inputs or shallow hidden neurons) directly to
deep neurons with conservative near-zero weights. Closes #570.

## Changes

- **`src/analysis/detection/skip_connection.rs`** — New detection module implementing:
  - Forward BFS to compute topological depth of each neuron from inputs
  - Gradient attenuation detection (deep neurons with mean error < 50% of shallow mean)
  - Candidate generation prioritising largest depth gaps with conservative weights
- **`src/analysis/detection/mod.rs`** — Register `skip_connection` module
- **`src/analysis/mod.rs`** — Re-export `skip_connection` at analysis level
- **`src/analysis/module_dispatch_specs.rs`** — Add parallel dispatch spec for the new module
- **`tests/issue_570_skip_connection_discovery.rs`** — 13 integration tests

## Evidence

This is a backend detection module with no UI. All 13 tests pass and `quality.sh` passes cleanly:
- Deep network with gradient attenuation produces candidates
- Shallow networks produce no candidates
- Candidates prioritise largest depth gaps
- Conservative weights (≤ 0.15) on all candidates
- Existing skip connections are not duplicated
- Edge cases: no hidden neurons, insufficient samples, no records, uniform error

## Test Plan

- `test_deep_network_detects_skip_connection_candidates` — verifies detection in deep networks
- `test_shallow_network_no_candidates` — no false positives for shallow networks
- `test_candidates_prioritise_largest_depth_gaps` — sorting by depth gap
- `test_candidates_have_positive_improvement` — all candidates have positive gain
- `test_coordinated_candidates_use_conservative_weights` — weight ≤ 0.15
- `test_existing_skip_connection_not_duplicated` — no duplicate suggestions
- `test_no_hidden_neurons_no_candidates` — edge case: no hidden neurons
- `test_insufficient_samples_no_candidates` — edge case: too few samples
- `test_no_records_no_candidates` — edge case: empty records
- `test_multi_input_deep_network` — multi-input topology
- `test_source_is_shallow_neuron` — source always shallower than target
- `test_candidates_sorted_by_improvement` — sorted best-first
- `test_uniform_error_no_attenuation_no_candidates` — no attenuation = no candidates
