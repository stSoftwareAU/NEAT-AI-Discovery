# Activation Recommendation

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/recommendation/activation_recommendation.rs`](../../src/analysis/recommendation/activation_recommendation.rs) | **Issue:** [#431](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/431)

---

## The Problem

Most activation function changes in NEAT are **reactive** — triggered after
a problem is detected (saturation, oscillation, etc.). But by the time the
problem manifests, the neuron may have already wasted many generations of
evolution operating inefficiently.

**Activation recommendation** takes a **proactive** approach: it analyses the
statistical distribution of each neuron's inputs and recommends the activation
function that best matches the data characteristics — before problems occur.

```
  Proactive vs Reactive:
  ──────────────────────

  Reactive (existing):
  Input data → [wrong squash] → problem develops → detect → fix
       ↑ generations wasted here

  Proactive (this module):
  Input data → analyse distribution → recommend best squash → apply
       ↑ catches mismatch immediately
```

### Why It Matters

- **Prevents future problems**: A well-matched activation function avoids
  saturation, oscillation, and restricted range issues before they start.
- **Better gradient flow**: Matching the activation to the input distribution
  ensures healthy gradients from the beginning.
- **Higher success rate**: Proactive recommendations target the root cause
  (activation/data mismatch) rather than symptoms.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each hidden neuron:                                       │
  │                                                                │
  │  Step 1: Classify input distribution                           │
  │  • Sparse:   > 50% of values near zero                        │
  │  • Bounded:  range < 2.0 and within [-1.1, +1.1]              │
  │  • Bimodal:  kurtosis < 2.5, high variance                    │
  │  • Gaussian: kurtosis 2–5, low skewness                       │
  │  • Uniform:  none of the above                                │
  │                                                                │
  │  Step 2: Score candidate activations by suitability            │
  │  Each distribution class has optimal activations:              │
  │  • Gaussian → TANH (0.9), SOFTPLUS (0.85)                     │
  │  • Sparse   → RELU (0.9), LEAKYRELU (0.85)                    │
  │  • Bounded  → LOGISTIC (0.85), HARD_TANH (0.80)               │
  │  • Uniform  → TANH (0.80), IDENTITY (0.75), GELU (0.75)       │
  │                                                                │
  │  Step 3: Apply gradient flow penalty                           │
  │  • TANH/LOGISTIC penalised 30% if inputs cause saturation     │
  │  • RELU penalised if many inputs are negative                  │
  │                                                                │
  │  Step 4: Check improvement threshold                           │
  │  • Best candidate score - current score >= 0.001               │
  │  • Recommended activation differs from current                 │
  │  • At least 20 samples available                               │
  └────────────────────────────────────────────────────────────────┘
```

---

## How We Fix It

```
  BEFORE (mismatched activation)         AFTER (distribution-matched)
  ┌─────┐    ┌──────────┐    ┌─────┐    ┌─────┐    ┌──────────┐    ┌─────┐
  │ I1  │───→│ H1       │───→│ O1  │    │ I1  │───→│ H1       │───→│ O1  │
  │     │    │ RELU     │    │     │    │     │    │ TANH     │    │     │
  │ I2  │───→│          │    │     │    │ I2  │───→│          │    │     │
  └─────┘    │ Gaussian │    └─────┘    └─────┘    │ matches  │    └─────┘
             │ inputs   │                          │ Gaussian │
             └──────────┘                          │ inputs!  │
                                                   └──────────┘
  RELU clips negative half              TANH handles full bell curve
  of Gaussian distribution              symmetrically
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Change squash** | `changeSquash` | Switch to distribution-matched activation |

Expected improvement: `improvement_delta × 0.02`, where `improvement_delta`
is the suitability score difference between recommended and current activation.

---

## Example

```
  Neuron H8: current activation = RELU
  Input distribution classified as: Gaussian
    Mean: 0.02, Std dev: 0.8, Kurtosis: 2.9, Skew: 0.1

  Suitability scores:
    TANH:      0.90  ← best for Gaussian
    SOFTPLUS:  0.85
    RELU:      0.60  (current — penalised for clipping negatives)
    LOGISTIC:  0.55

  Improvement: 0.90 - 0.60 = 0.30 (>> 0.001 threshold)

  Candidate: Change RELU → TANH
    Expected improvement: 0.30 × 0.02 = 0.006

  After fix: TANH handles the full bell-curve distribution
  symmetrically, preserving negative inputs that RELU was
  clipping to zero.
```

---

## References

- **Source module**: [`src/analysis/recommendation/activation_recommendation.rs`](../../src/analysis/recommendation/activation_recommendation.rs)
- **DISCOVERY_TYPES.md**: [Activation Function Recommendation](../DISCOVERY_TYPES.md#activation-function-recommendation)
- **Related**: [Saturated Neuron](saturated-neuron.md) — reactive fix for
  neurons already stuck at bounds
- **Related**: [Oscillating Neuron](oscillating-neuron.md) — reactive fix
  for sign-oscillating neurons
- **Related**: [Squash + Weight Rescale](squash-weight-rescale.md) —
  coordinated squash change with weight adjustment
