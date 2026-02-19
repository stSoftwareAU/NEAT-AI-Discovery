# Output Squash Mismatch

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/output_squash_mismatch.rs`](../../src/analysis/detection/output_squash_mismatch.rs) | **Issue:** [#545](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/545)

---

## The Problem

An **output squash mismatch** occurs when an output neuron's activation
function is fundamentally incompatible with the target data distribution.
No amount of weight or bias tuning can fix this — the activation function
itself prevents the neuron from reaching the correct output values.

Four distinct mismatch patterns exist:

```
  1. Clipping: Bounded squash clips activations at limits
  ─────────────────────────────────────────────────────
  Target: [0.0, 2.5]    LOGISTIC output: [0, 1]
  15%+ of samples hit the ceiling → systematic clipping error

  2. Range mismatch: All outputs are positive but targets need negative
  ────────────────────────────────────────────────────────────────────
  LOGISTIC: [0, 1] but targets include negative values
  The neuron literally cannot produce the needed outputs

  3. Unbounded mismatch: Outputs fly far outside expected range
  ─────────────────────────────────────────────────────────────
  IDENTITY outputs: [-50, +80] but targets are in [-1, +1]
  25%+ of samples are wildly out of range

  4. Pre-activation simulation: A different squash would reduce error
  ──────────────────────────────────────────────────────────────────
  Simulating TANH on the same pre-activation values reduces
  error by >= 15% compared to current activation
```

### Why It Hurts the Creature's Score

- **Impossible targets**: The activation function physically cannot produce
  the output values needed, guaranteeing minimum error above zero.
- **Systematic error**: Clipping or range mismatches produce consistent,
  directional errors that bias cannot fix.
- **Wasted convergence effort**: Weight tuning tries to compensate for an
  architectural problem, wasting optimisation budget.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each output neuron:                                       │
  │                                                                │
  │  Prerequisites:                                                │
  │  • Mean |error| >= 0.05                                        │
  │  • Sufficient samples available                                │
  │                                                                │
  │  Strategy 1 — Clipping:                                        │
  │  • >= 15% of activations at saturation bounds                  │
  │  • Bound error > 1.5× centre error (or hard-clipping squash)  │
  │                                                                │
  │  Strategy 2 — Range mismatch:                                  │
  │  • All-positive activations (min > -0.05)                      │
  │  • > 30% of samples have above-average error                  │
  │                                                                │
  │  Strategy 3 — Unbounded mismatch:                              │
  │  • Unbounded squash function                                   │
  │  • > 25% of activations outside ±1.05                          │
  │  • Out-of-range error > 1.2× overall mean error                │
  │                                                                │
  │  Strategy 4 — Pre-activation comparison:                       │
  │  • Simulate all candidate squashes on pre-activation values    │
  │  • If any reduces error by >= 15% → recommend it               │
  │                                                                │
  │  First matching strategy wins.                                 │
  └────────────────────────────────────────────────────────────────┘
```

---

## How We Fix It

```
  BEFORE (LOGISTIC, targets need [-1,+1])  AFTER (TANH, symmetric range)
  ┌─────┐    ┌──────────┐    ┌─────┐      ┌─────┐    ┌──────────┐    ┌─────┐
  │ H1  │───→│ O1       │    │ tgt │      │ H1  │───→│ O1       │    │ tgt │
  │     │    │ LOGISTIC │    │     │      │     │    │ TANH     │    │     │
  │     │    │ [0, +1]  │    │[-1,1]│     │     │    │ [-1,+1]  │    │[-1,1]│
  └─────┘    │ can't go │    └─────┘      └─────┘    │ full     │    └─────┘
             │ negative!│                            │ range!   │
             └──────────┘                            └──────────┘
```

| Strategy | Candidate | Operation | Typical Recommendation |
|----------|-----------|-----------|----------------------|
| **Clipping** | Change squash | `changeSquash` | Bounded → wider range (e.g., TANH) |
| **Range mismatch** | Change squash | `changeSquash` | Non-negative → symmetric (e.g., LOGISTIC → TANH) |
| **Unbounded mismatch** | Change squash | `changeSquash` | Unbounded → bounded (e.g., IDENTITY → TANH) |
| **Pre-activation comparison** | Change squash | `changeSquash` | Best-performing alternative from simulation |

---

## Example

```
  Output neuron O1: LOGISTIC, bias = 0.5
  Target data range: [-0.8, +1.2]

  Strategy 2 triggers:
    All activations are positive (LOGISTIC range: [0, 1])
    Activation min = 0.12 (> -0.05)
    42% of samples have above-average error
    The neuron cannot output negative values to match negative targets

  Candidate: Change to TANH
    TANH range: [-1, +1] — covers the negative target values
    Estimated improvement: 0.15

  After fix: O1 can now produce negative outputs,
  allowing it to match the full target distribution.
```

---

## References

- **Source module**: [`src/analysis/detection/output_squash_mismatch.rs`](../../src/analysis/detection/output_squash_mismatch.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Error Plateau](error-plateau.md) — detects output neurons
  stuck at uniformly high error
- **Related**: [Activation Recommendation](activation-recommendation.md) —
  proactive activation matching for hidden neurons
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects saturation
  in hidden neurons
