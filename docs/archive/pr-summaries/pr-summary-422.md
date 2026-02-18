## Summary

Implements topology-aware discovery (Issue #422) — a new discovery module that
analyses overall network structure to identify path length and connectivity
balance issues, complementing existing per-neuron activation-based discovery.

Two detection types are implemented:

1. **Long path detection**: Identifies hidden neurons whose shortest path to any
   output exceeds 3 hops. Suggests `addSynapse` skip connections to shorten the
   path and reduce gradient attenuation.

2. **Connectivity imbalance detection**: Identifies hidden neurons with
   significantly fewer connections than peers (fan-in ratio ≥ 3.0). Suggests
   `addSynapse` from unused inputs to balance information flow.

### Changes

- **New**: `src/analysis/topology.rs` — detection and candidate conversion
- **Modified**: `src/analysis/mod.rs` — module registration and dispatch wiring
- **New**: `tests/issue_422_topology_aware_discovery.rs` — 13 integration tests
- **Modified**: `docs/DISCOVERY_TYPES.md` — documentation for the new discovery type

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

13 tests added in `tests/issue_422_topology_aware_discovery.rs`:

- `test_detects_long_path_in_deep_chain` — 4-hop chain detected as long path
- `test_detects_connectivity_imbalance` — fan-in=1 vs fan-in=3 detected
- `test_balanced_topology_no_issues` — healthy topology produces no candidates
- `test_output_neurons_not_flagged` — output neurons excluded
- `test_input_neurons_not_flagged` — input neurons excluded
- `test_insufficient_samples_not_flagged` — <20 samples skipped
- `test_empty_network_no_candidates` — no hidden neurons → no candidates
- `test_candidates_have_positive_improvement` — all improvements > 0
- `test_conversion_to_coordinated_candidates` — valid coordinated operations
- `test_long_path_suggests_skip_connection` — addSynapse generated
- `test_candidates_sorted_by_improvement` — best first ordering
- `test_no_hidden_records_no_candidates` — graceful empty records handling
- `test_existing_skip_connection_not_duplicated` — no duplicate suggestions

All tests pass. `quality.sh` passes cleanly.
