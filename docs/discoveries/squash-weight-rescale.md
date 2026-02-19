# Squash + Weight Rescale

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/squash_weight_rescale.rs`](../../src/analysis/detection/squash_weight_rescale.rs) | **Issue:** [#548](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/548)

---

## The Problem

When a neuron needs a different activation function, simply swapping the
squash can be destructive — the new function may interpret the same
pre-activation values completely differently, breaking downstream
expectations. A **coordinated squash change with weight rescaling** finds
the optimal rescale factor so the new activation best approximates the old
one's output, then bundles both changes atomically.

```
  Problem: bare squash swap breaks output

  TANH(-1.5) = -0.91                 IDENTITY(-1.5) = -1.50
       ↓ swap without rescale              ↓
  Downstream neurons expect ≈ -0.91    Now receive -1.50!
  → Predictions break                 → 65% output change
```

### Why It Hurts Without Rescaling

- **Output discontinuity**: A bare squash swap changes the neuron's output
  distribution instantly, invalidating all downstream weight calibrations.
- **Cascade disruption**: Every neuron downstream of the changed neuron
  receives unexpected input magnitudes.
- **Wasted mutation**: The squash change might be beneficial, but the
  disruption masks the improvement — NEAT-AI's ablation test fails.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each hidden neuron:                                       │
  │                                                                │
  │  1. Requires non-aggregate squash, at least one synapse        │
  │  2. Collect pre-activation (value) samples (minimum 20)        │
  │  3. Check mean |error| >= 0.03                                 │
  │  4. For each of 10 candidate activations:                      │
  │     (TANH, LOGISTIC, IDENTITY, SOFTSIGN, HARD_TANH,            │
  │      RELU, SELU, MISH, SWISH, ELU)                            │
  │  5. Grid search rescale factors from 0.25 to 4.0               │
  │     → Find factor f that minimises:                            │
  │       Σ |new_squash(f × value) - old_squash(value)|            │
  │  6. If best candidate shows positive improvement               │
  │     and rescale factor <= 5.0 → candidate emitted              │
  └────────────────────────────────────────────────────────────────┘
```

### Why Coordinated Is Better

The improvement estimate for a coordinated squash+rescale candidate is
boosted by 1.5× compared to a standalone squash change, because the weight
adjustment preserves output compatibility.

---

## How We Fix It

```
  BEFORE (TANH, incoming weights [0.5, 0.3])

  I1 ──(w=0.5)──→ H1 [TANH, bias=0.2] ──→ O1
  I2 ──(w=0.3)──→

  Grid search finds: IDENTITY with rescale factor 0.6
  best approximates TANH's output on observed data

  AFTER (IDENTITY, weights rescaled by 0.6)

  I1 ──(w=0.30)──→ H1 [IDENTITY, bias=0.2] ──→ O1
  I2 ──(w=0.18)──→

  Output stays approximately the same,
  but now IDENTITY provides unbounded range for learning.
```

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Squash + rescale** | `changeSquash` + `setWeight` (per synapse) | Atomic group: change activation and rescale all incoming weights |

The weight rescaling is only applied when the rescale factor deviates from
1.0 by at least 0.05 (to avoid trivial adjustments).

---

## Example

```
  Neuron H9: LOGISTIC, bias=1.2, mean |error|=0.08
  Incoming synapses: 3 (weights: 0.8, -0.4, 0.6)

  Grid search results:
    TANH with factor 0.45 → error reduction: 0.012
    IDENTITY with factor 0.35 → error reduction: 0.018  ← best
    RELU with factor 0.50 → error reduction: 0.009

  Candidate: Change to IDENTITY, rescale weights by 0.35
    New weights: 0.28, -0.14, 0.21
    Estimated improvement: 0.018 × 1.5 = 0.027 (coordinated boost)

  The new IDENTITY activation provides the same approximate output
  as LOGISTIC on the observed data, but with room to grow beyond
  the [0, 1] bound.
```

---

## References

- **Source module**: [`src/analysis/detection/squash_weight_rescale.rs`](../../src/analysis/detection/squash_weight_rescale.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Activation Recommendation](activation-recommendation.md) —
  proactive squash suggestions based on input distribution
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects neurons
  stuck at activation bounds
