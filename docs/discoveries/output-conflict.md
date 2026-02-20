# Output Conflict Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/output_conflict.rs`](../../src/analysis/detection/output_conflict.rs) | **Issue:** [#639](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/639)

---

## The Problem

A **per-output error conflict** occurs when a hidden neuron has a positive
effect on some outputs but actively harms others. The net error may look
acceptable, but the neuron is creating a cross-output tug-of-war.

```
  Hidden neuron "h1":
    output-0 error contribution: -0.5  (helping — reducing error)
    output-1 error contribution: +0.3  (harming — increasing error)
                                 ────
    Net effect:                  -0.2  (looks helpful overall)

  But output-1 is being actively harmed!
```

### Why It Hurts the Creature's Score

- **Hidden harm**: The net error masks the damage to individual outputs.
  Summed-error metrics make the neuron appear beneficial when it is not.
- **Training interference**: Gradient updates that improve one output's
  contribution through this neuron may worsen another output.
- **Structural limitation**: A single hidden neuron cannot simultaneously
  optimise its contribution to outputs that require opposing effects.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each hidden neuron:                                       │
  │                                                                │
  │  1. Collect errors[0..n] across all observations               │
  │  2. Compute mean error per output index                        │
  │  3. Check for sign conflict:                                   │
  │     - At least one output has mean error < -threshold (helped) │
  │     - At least one output has mean error > +threshold (harmed) │
  │  4. Compute conflict severity = max_positive × |min_negative|  │
  │  5. Filter out weak conflicts (below significance threshold)   │
  └────────────────────────────────────────────────────────────────┘
```

### Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Minimum significant error | 0.01 | Ignore noise-level error contributions |
| Minimum samples | 10 | Ensure statistical reliability |
| Minimum outputs | 2 | Conflict requires multiple outputs |

---

## What We Recommend

### Strategy 1: Attenuate the harmful connection (SetWeight)

When a direct synapse exists from the conflicting neuron to the harmed
output, reduce its weight to 30% of the current value.

```
  Before:  h1 ──(w=0.8)──→ output-1 (harmed)
  After:   h1 ──(w=0.24)──→ output-1 (reduced harm)
```

### Strategy 2: Add a compensating gating neuron (AddNeuron + AddSynapse)

When the path to the harmed output is indirect, insert a compensating
neuron that counteracts the harmful contribution.

```
  Before:  h1 ──→ ... ──→ output-1 (harmed)
  After:   h1 ──(-w)──→ [gate] ──(1.0)──→ output-1
```

All recommendations are emitted as `CoordinatedStructuralCandidateJson`.

---

## Relationship to Other Modules

| Module | Scope | Difference |
|--------|-------|------------|
| **Correlated error** | Output-neuron error correlations | Analyses correlations between output neurons, not hidden neuron per-output contributions |
| **Output conflict** (this) | Hidden neuron per-output disaggregation | Analyses the `errors[0..n]` vector for hidden neurons to find cross-output sign conflicts |

---

## Example Scenario

A creature with 2 outputs and 1 hidden neuron:

```
  input-1 ──→ hidden-1 ──→ output-0 (speech recognition)
                        └──→ output-1 (noise classification)
```

Over 50 training samples, `hidden-1` consistently:
- Reduces error on output-0 by ~0.5 (mean error = -0.5)
- Increases error on output-1 by ~0.3 (mean error = +0.3)

The detector flags `hidden-1` with:
- `conflict_severity = 0.5 × 0.3 = 0.15`
- Recommends attenuating the synapse to output-1
