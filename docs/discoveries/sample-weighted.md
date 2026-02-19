# Sample-Weighted Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/recommendation/sample_weighted.rs`](../../src/analysis/recommendation/sample_weighted.rs) | **Issue:** [#423](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/423)

---

## The Problem

Standard discovery treats all training samples equally when evaluating
neurons. But in many real-world problems, a small number of **hard samples**
dominate the creature's total error. Neurons that are mediocre on average
may be terrible on the hardest samples — and fixing those hard-sample
contributions would have the biggest impact on overall score.

**Sample-weighted discovery** re-weights the analysis so neurons contributing
disproportionately to high-error samples receive priority attention.

```
  Error distribution across samples:
  ──────────────────────────────────
  Sample:  1     2     3     4     5     6     7     8
  Error:  0.01  0.02  0.85  0.01  0.03  0.92  0.02  0.01

  Uniform weighting:    all samples contribute equally
  Importance weighting: samples 3 and 6 dominate (80% of total error)

  Neuron H5 has error:
    Easy samples:  mean error = 0.02
    Hard samples:  mean error = 0.45
    → Ratio: 22.5× worse on hard samples!
    → Fixing H5's response to hard samples would have outsized impact
```

### Why It Hurts the Creature's Score

- **Hidden contributors**: Neurons may look fine on average but fail badly
  on the samples that matter most.
- **Missed optimisation**: Equal weighting dilutes the signal from
  high-error samples, making it harder to identify the real problems.
- **Robustness gap**: The creature handles easy cases well but collapses
  on difficult inputs.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each neuron (input, hidden, or output):                   │
  │                                                                │
  │  1. Collect error samples (minimum 10)                         │
  │  2. Compute per-sample importance weights:                     │
  │     weight_i = |error_i| / sum(|all_errors|)                   │
  │  3. Compute weighted mean error:                               │
  │     sum(weight_i × |error_i|)                                  │
  │  4. If weighted mean error >= 0.25 → significant               │
  │  5. Stratify into easy (≤ median error) and hard (> median)    │
  │  6. Compute hard_to_easy_ratio                                 │
  │  7. If ratio is high → neuron struggles on hard samples        │
  └────────────────────────────────────────────────────────────────┘
```

---

## How We Fix It

The fix adjusts the neuron's bias to shift its operating point toward
better performance on high-error samples:

```
  BEFORE (biased toward easy samples)    AFTER (bias-adjusted)
  ┌─────┐    ┌──────────┐    ┌─────┐    ┌─────┐    ┌──────────┐    ┌─────┐
  │ H1  │───→│ Neuron   │───→│ O1  │    │ H1  │───→│ Neuron   │───→│ O1  │
  │     │    │ bias=0.5 │    │     │    │     │    │ bias=0.43│    │     │
  │     │    │ easy: OK │    │     │    │     │    │ better on│    │     │
  │     │    │ hard: BAD│    │     │    │     │    │ hard     │    │     │
  └─────┘    └──────────┘    └─────┘    └─────┘    │ samples  │    └─────┘
                                                   └──────────┘
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Bias adjustment** | `setBias` | bias += -(weighted_mean_error × 0.1) |

The adjustment is deliberately small (10% of weighted mean error) because
NEAT-AI validates through ablation — conservative changes are more likely
to pass validation.

Estimated improvement:
`min(weighted_mean_error × hard_to_easy_ratio × 0.01, 0.1)`.

---

## Example

```
  Neuron H12 across 200 samples:
    Weighted mean error: 0.38 (>= 0.25 threshold)
    Easy samples (100): mean error = 0.04
    Hard samples (100): mean error = 0.72
    Hard-to-easy ratio: 18.0

  Candidate:
    Current bias: 1.2
    Adjustment: -(0.38 × 0.1) = -0.038
    New bias: 1.162

  Estimated improvement: min(0.38 × 18.0 × 0.01, 0.1) = 0.068

  After fix: The bias shift nudges the neuron's operating
  point toward better handling of hard samples, where most
  of the creature's error is concentrated.
```

---

## References

- **Source module**: [`src/analysis/recommendation/sample_weighted.rs`](../../src/analysis/recommendation/sample_weighted.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Output Bias Drift](output-bias-drift.md) — corrects
  systematic bias in output predictions
- **Related**: [Error Plateau](error-plateau.md) — detects uniformly high
  error at outputs
