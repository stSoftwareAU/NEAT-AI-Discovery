# Gradient-Based Synapse Adjustment

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/recommendation/gradient_discovery.rs`](../../src/analysis/recommendation/gradient_discovery.rs) | **Issue:** [#421](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/421)

---

## The Problem

Correlation-based discovery methods measure the *strength* of association
between a synapse's contribution and the target's error, but they do not
indicate *which direction* to adjust the weight. **Gradient-based discovery**
computes the local gradient (∂error/∂weight) for each synapse, providing
directional information about how to change the weight for maximum error
reduction.

```
  Correlation-based (existing):    Gradient-based (this module):
  ─────────────────────────────    ────────────────────────────────

  "Synapse S correlates with       "Synapse S should decrease
   error at r = 0.75"               weight by 0.03 to reduce error"

  Knows strength, not direction    Knows both strength AND direction
```

### Why It Matters

- **Directional guidance**: Gradients tell us not just *that* a weight
  matters, but *which way* to adjust it.
- **Higher success rate**: Gradient-based weight adjustments target 25–30%
  success rate, higher than correlation-based methods.
- **Conservative approach**: Uses a small learning rate (0.1) since NEAT-AI
  validates through ablation testing anyway.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each synapse targeting an output neuron:                  │
  │                                                                │
  │  1. Collect paired samples:                                    │
  │     source activation and target error (same observation)      │
  │  2. Compute local gradient:                                    │
  │     ∂error/∂weight ≈ mean(source_activation × target_error)    │
  │  3. Check gradient magnitude >= 0.01                           │
  │  4. Check gradient consistency:                                │
  │     |mean_gradient| / std_dev >= 0.3                           │
  │     (reliable direction, not noise-driven)                     │
  │  5. Compute new weight:                                        │
  │     new_weight = old_weight - 0.1 × gradient                   │
  │  6. Check effective weight delta > 1e-8                        │
  │  7. If all checks pass → gradient adjustment candidate         │
  └────────────────────────────────────────────────────────────────┘
```

### Gradient Computation

The local gradient approximates the partial derivative using the chain rule:

```
  ∂error/∂weight ≈ mean(source_activation × target_error)

  This measures: "If I increase the weight slightly, how much
  does the target error change, on average across all samples?"
```

---

## How We Fix It

```
  BEFORE                                 AFTER
  ┌─────┐                               ┌─────┐
  │ H3  │──(w=0.40)──→ O1               │ H3  │──(w=0.37)──→ O1
  └─────┘                               └─────┘
          gradient = 0.3                          w = 0.40 - 0.1×0.3
          → "increase weight                        = 0.37
             increases error"
          → decrease weight!
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Adjust weight** | `setWeight` | new_weight = old_weight − 0.1 × gradient |

The learning rate of 0.1 is conservative — NEAT-AI will validate the
adjustment through ablation testing, so aggressive changes are unnecessary.

Estimated improvement:
`|mean_gradient| × |weight_delta| × min(consistency, 3.0) × 0.01`.

---

## Example

```
  Synapse H5 → O2, current weight = 0.60

  Paired samples (source activation, target error):
    (0.8, 0.3), (0.5, 0.2), (0.9, 0.4), (0.3, 0.1), (0.7, 0.25)

  Gradient = mean(0.8×0.3 + 0.5×0.2 + 0.9×0.4 + 0.3×0.1 + 0.7×0.25)
           = mean(0.24 + 0.10 + 0.36 + 0.03 + 0.175)
           = 0.181

  |gradient| = 0.181 (>= 0.01 ✓)
  Std dev of gradients = 0.12
  Consistency = 0.181 / 0.12 = 1.51 (>= 0.3 ✓)

  New weight = 0.60 - 0.1 × 0.181 = 0.582
  Delta = 0.018

  Estimated improvement: 0.181 × 0.018 × min(1.51, 3.0) × 0.01 = 0.000049

  After fix: Weight adjusted in the error-reducing direction.
  NEAT-AI validates through ablation testing.
```

---

## References

- **Source module**: [`src/analysis/recommendation/gradient_discovery.rs`](../../src/analysis/recommendation/gradient_discovery.rs)
- **DISCOVERY_TYPES.md**: [Gradient-Based Synapse Adjustment](../DISCOVERY_TYPES.md#gradient-based-synapse-adjustment)
- **Related**: [Weight Magnitude Reset](weight-magnitude-reset.md) — tries
  dramatically different weights to escape plateaus (complementary approach)
- **Related**: [Opposing Synapse](opposing-synapse.md) — detects synapses
  with positive contribution–error correlation
- **Gradient descent** —
  [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent):
  The optimisation technique this module adapts for NEAT discovery.
