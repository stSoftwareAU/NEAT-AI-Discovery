## Summary

Refactor 9 functions to use parameter structs instead of long argument lists,
removing all `#[allow(clippy::too_many_arguments)]` suppressions. Closes #715.

### Parameter structs introduced

| Struct | File | Functions served |
|--------|------|-----------------|
| `NeuronEvalContext` | `neuron/evaluation.rs` | `evaluate_neuron_candidates`, `evaluate_relu_split`, `evaluate_activation_specs` |
| `NeuronResultParams` | `neuron/post_processing.rs` | `build_neuron_results` |
| `ReplaceSynapseParams` | `synapse/filtering.rs` | `expected_gain_replace_synapse_with_hidden_neuron` |
| `SubsetEvalParams` | `synapse/gpu_evaluation.rs` | `evaluate_activation_for_subset` |
| `ActivationEvalParams` | `synapse/gpu_evaluation.rs` | `evaluate_activation_candidate` |
| `ThreeHopContext` | `recommendation/multi_hop.rs` | `find_three_hop_extensions` |

For `evaluate_all_activation_specs_batched` (6 params, under the clippy limit),
the suppression was simply removed since no struct was needed.

### No behavioural changes

This is a pure refactoring. All call sites construct the new structs with the
same values previously passed as positional arguments.

## Evidence

- `quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
- Zero `#[allow(clippy::too_many_arguments)]` annotations remain in the codebase
- No UI changes; backend-only refactoring

## Test Plan

- All existing tests pass without modification (except updating one test call site
  in `implementation_tests/optimal_weight_tests.rs` to use `ActivationEvalParams`)
- No new tests needed — this is a signature-level refactoring with no behavioural change
