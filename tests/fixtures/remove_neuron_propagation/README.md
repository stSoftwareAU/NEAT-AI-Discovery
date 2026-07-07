# remove-neuron propagation fixtures (Issue #1517)

Hermetic fixtures for `tests/remove_neuron_propagation.rs`. They are committed
here so the test is reproducible offline and does not reach out to other
repositories at runtime.

| File | Source | Purpose |
|------|--------|---------|
| `network.json` | `stSoftwareAU/GRQ-cluster` `network.json` (`main`) | Production creature topology. The target neuron `neuron-1802938338` (squash `Cube`) sits many layers from the single output, so its contribution is diluted/squashed through the intervening activations. |
| `v2_remove-neuron_neuron-1802938338.json` | `stSoftwareAU/GRQ-Discovery` `failures/247b83ab/remove-neuron/v2_remove-neuron_neuron-1802938338.json` (`Develop`) | Recorded remove-neuron failure. Carries the fabricated placeholder gain (`expectedCreatureScoreGain = +0.17882921`) and the empirically measured effect (`actualErrorReduction = -0.000194`). |

## Why these values matter

The placeholder gain is ~920× larger than — and opposite in sign to — the
measured actual effect. A propagation-aware estimate of the target neuron's
influence on the output is `~2.1e-5` (structural impact), within an order of
magnitude of the measured `~1.9e-4` and thousands of times below the
placeholder. See Issue #1516 for the root-cause analysis.

Do not edit these fixtures by hand. If either upstream source changes, refresh
the file and re-validate `placeholder_gain_is_wrong_at_depth`.
