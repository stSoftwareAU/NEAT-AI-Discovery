# change-squash propagation fixtures (Issue #1532)

Hermetic fixtures for `tests/change_squash_propagation.rs`. They are committed
here so the test is reproducible offline and does not reach out to other
repositories at runtime.

| File | Source | Purpose |
|------|--------|---------|
| `v2_change-squash_neuron-1481550544.json` | `stSoftwareAU/GRQ-Discovery` `failures/247b83ab/change-squash/…` (commit `2d07afca`) | Recorded change-squash failure. Carries the near-zero placeholder gain (`expectedCreatureScoreGain = 8.6e-10`) and the empirically measured effect (`actualErrorReduction = -0.000341`) for `neuron-1481550544` (`SELU → SQUARE`). This is the second cited failure on GRQ-Discovery commit `2596f073`. |

The **network topology** is shared with the remove-neuron fixture: the recorded
failure is on the same production creature (identical `originalScore`
`0.4236678629467732`), so the test loads the 1,666-neuron / 21,532-synapse
GRQ-cluster creature from `../remove_neuron_propagation/network.json` rather than
committing a second 3 MB copy (single source of truth).

## Why these values matter

The placeholder gain (`8.6e-10`) is ~400,000× *smaller* than — and opposite in
sign to — the measured actual effect (`-0.000341`). The change-squash effect is
driven by how much the neuron's emitted output changes when its activation
function is swapped (`SELU → SQUARE` amplifies the output by orders of
magnitude), so the pure small-perturbation structural influence (~`4.3e-7` for
this neuron) under-predicts the effect ~800×.

The propagation-aware estimator (`estimate_change_squash_gain`, Issue #1532)
combines the neuron's propagation-aware downstream influence
(`compute_impacts_public`, reused from the #1518 remove-neuron estimator) with
the local perturbation the swap induces (the reduction in the neuron's local
error, `currentError − improvedError`). Its signed estimate is a small negative
value within an order of magnitude of the measured `~3.4e-4` and hundreds of
thousands of times above the placeholder. See Issue #1516/#1518 for the
remove-neuron precedent and #1529 for the accuracy milestone.

Do not edit these fixtures by hand. If the upstream source changes, refresh the
file and re-validate `change_squash_placeholder_is_wrong_at_depth`.
