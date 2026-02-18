## Summary

Add bounded range discovery module (Issue #395) to detect observation and hidden neurons where a significant cluster of activation values sits at a sentinel boundary (e.g., -1, 0, or +1), separate from the "useful" range. This addresses the problem where observations normalised to -1…1 use sentinel values like -1 to represent null/missing data, which negatively impacts the creature's score when simply multiplied by a weight.

The module recommends adding a gating neuron (via `addNeuron` + `addSynapse` coordinated operations) that can learn to suppress the sentinel region while passing through useful-range values.

### Key Design Decisions

- **Sentinel detection at -1, 0, and +1**: These are the most common null sentinels in normalised data
- **Minimum 20% boundary fraction**: A significant cluster must exist at the sentinel value
- **Gap requirement**: There must be a measurable gap (≥ 0.05) between sentinel and useful range
- **Input and hidden neurons only**: Output neurons are excluded
- **Follows DRY pattern**: Uses the existing `discovery_dispatch::run_discovery_module` pattern from Issue #375
- **GPU-compatible candidate**: Recommends `AddNeuron` (RELU gating) + `AddSynapse`, which are standard coordinated structural operations

### Files Changed

| File | Change |
|------|--------|
| `src/analysis/bounded_range.rs` | New module: detection logic and candidate conversion |
| `src/analysis/mod.rs` | Register module and add dispatch call |
| `tests/issue_395_bounded_range_detection.rs` | 10 integration tests covering detection and edge cases |

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. All behaviour is verified through automated tests.

## Test Plan

- `test_detects_observation_with_null_cluster_at_minus_one` — 40% at -1 sentinel detected
- `test_detects_observation_with_null_cluster_at_plus_one` — 35% at +1 sentinel detected
- `test_does_not_flag_uniform_distribution` — evenly spread values not flagged
- `test_detects_hidden_neuron_with_boundary_cluster` — hidden neuron with TANH saturation sentinel
- `test_insufficient_samples_not_detected` — below minimum sample threshold
- `test_all_same_values_not_detected` — constant values not flagged
- `test_detects_cluster_at_zero_sentinel` — 50% at 0.0 sentinel detected
- `test_coordinated_candidate_conversion` — valid coordinated structural operations produced
- `test_multiple_neurons_only_boundary_clustered_detected` — selective detection across multiple neurons
- `test_output_neurons_excluded` — output neurons correctly excluded
