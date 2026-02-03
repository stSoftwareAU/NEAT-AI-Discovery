# Output Bias Drift Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/output_bias_drift.rs`](../../src/analysis/output_bias_drift.rs) | **Issue:** [#361](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/361)

---

## The Problem

**Output bias drift** occurs when an output neuron consistently predicts too
high or too low across the majority of training samples. The network has learned
the correct pattern shape but is offset by a constant amount.

```
  Prediction vs Target over training samples
  (each dot is one sample)

  Prediction
       ↑
   1.0 │        · ·                    · ·
       │      ·     ·               ·     ·
   0.8 │    ·         ·           ·         ·      ← predictions
       │  ·             ·       ·             ·
   0.6 │·               · · ·               · ·
       │
   0.4 │        · ·                    · ·
       │      ·     ·               ·     ·
   0.2 │    ·         ·           ·         ·      ← targets
       │  ·             ·       ·             ·
   0.0 │·               · · ·               · ·
       └──────────────────────────────────────────→ samples

  Predictions follow the correct pattern but are shifted UP by ~0.4
  → systematic positive bias
```

### Why It Hurts the Creature's Score

- Every sample has approximately the same error (the offset).
- The error does not cancel out — it consistently adds to the total score
  penalty.
- A simple bias adjustment would eliminate this entire class of error.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each OUTPUT neuron:                                   │
  │                                                            │
  │  1. Collect error samples (minimum 20)                     │
  │  2. Compute mean error                                     │
  │  3. Count fraction of positive errors vs negative errors   │
  │  4. Check:                                                 │
  │     • > 70% of errors share the same sign                  │
  │     • |mean error| >= 0.01                                 │
  │  5. If both → output has bias drift                        │
  └────────────────────────────────────────────────────────────┘
```

### Visualising the Detection

```
  Error distribution for output neuron O1:

  Count
    ↑
  8 │          ████
  6 │        ████████
  4 │      ████████████
  2 │    ██████████████████
  0 │──██████████████████████──→ error value
       -0.1  0  +0.1 +0.2 +0.3 +0.4 +0.5

  Mean error = +0.32
  85% of errors are positive
  → Clear positive bias drift
```

---

## How We Fix It

Adjust the output neuron's bias to centre the predictions:

```
  BEFORE                              AFTER
  ┌──────────┐                        ┌──────────┐
  │  Output   │                        │  Output   │
  │  bias=0.0 │                        │  bias=-0.32│
  │           │                        │           │
  │  mean     │                        │  mean     │
  │  error    │                        │  error    │
  │  = +0.32  │                        │  ≈ 0.0    │
  └──────────┘                        └──────────┘

  New bias = old_bias + (-mean_error)
           = 0.0 + (-0.32) = -0.32
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Set bias** | `setBias` | new_bias = current_bias - mean_error |

The estimated improvement is proportional to the magnitude of the drift and the
consistency (majority fraction).

---

## Example

```
  Output neuron O2 (classifying "cat" vs "not cat"):

  Samples: 500
  Mean error: +0.18 (predictions consistently too high)
  Positive errors: 412/500 = 82.4%
  Current bias: 0.05

  Recommended fix:
  setBias → 0.05 + (-0.18) = -0.13

  After fix:
  Predictions shift down by 0.18
  Mean error ≈ 0.0
  Score improves across 82% of samples
```

---

## References

- **Bias in neural networks** —
  [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neuron#Types_of_transfer_functions):
  How the bias parameter shifts the activation function's operating point.
- **Mean squared error** —
  [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error): The loss
  metric that output bias drift directly inflates.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
