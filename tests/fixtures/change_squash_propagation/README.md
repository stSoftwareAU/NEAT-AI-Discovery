# change-squash propagation fixtures (Issue #1532, re-based by Issue #1722)

Hermetic fixtures for `tests/change_squash_propagation.rs`. Every file here is
**hand-authored and synthetic** — this public repository is fully self-contained,
so no fixture is captured from, or derived from, any other repository. The tests
load them from disk and never reach out over the network.

| File | Source | Purpose |
|------|--------|---------|
| `v2_change-squash_spine-1.json` | Synthetic (candidate-record shape) | A change-squash candidate (`SELU → SQUARE`) on `spine-1`. Carries the retired near-zero placeholder gain (`expectedCreatureScoreGain = +5e-10`), the neuron's local errors under the current/proposed squash (`currentError = 3.0`, `improvedError = 1.0`), and the closed-form propagated effect (`analyticErrorReduction = −4.8828125e-4`). |

The **network topology** is shared with the remove-neuron fixture: both
candidates sit on the same synthetic deep-chain creature, so the test loads
`../remove_neuron_propagation/network.json` rather than committing a second copy
(single source of truth).

## Why these values matter

The placeholder gain (`5e-10`) is ~1,000,000× *smaller* than — and opposite in
sign to — the analytic propagated effect (`−4.9e-4`). The change-squash effect is
driven by how much the neuron's emitted output changes when its activation
function is swapped, so a topology-blind, activation-blind near-zero placeholder
under-predicts it by orders of magnitude.

The propagation-aware estimator (`estimate_change_squash_gain`, Issue #1532)
combines the neuron's propagation-aware downstream influence
(`compute_impacts_public`, reused from the #1518 remove-neuron estimator) with
the local perturbation the swap induces (`currentError − improvedError`). On this
topology that product is derivable by hand, so the fixture is a genuine oracle
rather than a recording of whatever the implementation happened to emit:

- `spine-1` sits 12 halving hops from the output, so its influence is exactly
  `0.5¹² = 2.44140625e-4`.
- The swap reduces the neuron's local error by `3.0 − 1.0 = 2.0`.
- The honest gain is therefore `−(2.44140625e-4 × 2.0) = −4.8828125e-4`, exact in
  `f32`.

See Issue #1516/#1518 for the remove-neuron precedent and #1529 for the accuracy
milestone.

Do not edit these fixtures by hand to make a downstream test pass. If the
topology changes, recompute the analytic reference from the halving-hop count and
the local-error reduction, then re-validate
`change_squash_placeholder_is_wrong_at_depth`.
