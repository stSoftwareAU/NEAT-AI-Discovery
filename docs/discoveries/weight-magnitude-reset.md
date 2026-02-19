# Weight Magnitude Reset

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/weight_magnitude_reset.rs`](../../src/analysis/detection/weight_magnitude_reset.rs) | **Issue:** [#550](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/550)

---

## The Problem

Some synapses feed target neurons that are **stuck in a plateau local
minimum** — the target has consistently high error that varies little across
samples. This flat error surface means gradient-based adjustments produce
no improvement; the synapse weight is trapped in a basin where small changes
have no effect.

The solution is to try **dramatically different weight values** — sign flips,
magnitude resets, and scaling — to escape the basin entirely.

```
  Error surface (conceptual)
  ──────────────────────────

  error
    │
  0.4├─────╮               ╭──────
    │     │   ← plateau   │
  0.3├─────┤    (stuck!)   ├──────
    │     │               │
  0.2├     │    gradient   │
    │     │    ≈ 0 here    │
  0.1├     │               │
    │     ╰───────────────╯  ← better minimum
  0.0├─────────────────────────────
    │  current    ?     better
    │  weight           weight
```

### Why It Hurts the Creature's Score

- **Stuck at high error**: The target neuron consistently produces wrong
  outputs but small weight tweaks cannot fix it.
- **Wasted optimisation**: Gradient-based methods keep trying small steps
  that go nowhere, consuming discovery budget.
- **Plateau trap**: The error coefficient of variation is low (< 0.4),
  confirming the error is not random noise but a genuine stuck state.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each synapse (source → target):                           │
  │                                                                │
  │  1. Both source and target have sufficient samples             │
  │  2. Source mean |activation| >= 0.01                           │
  │     (synapse is actively contributing)                         │
  │  3. Target mean |error| >= 0.1                                 │
  │     (significant stuck error)                                  │
  │  4. Target error coefficient of variation <= 0.4               │
  │     (plateau: errors tightly clustered, not noisy)             │
  │  5. At least one exploratory weight value is valid             │
  │  6. If all checks pass → stuck synapse, generate candidates    │
  └────────────────────────────────────────────────────────────────┘
```

---

## How We Fix It

Multiple exploratory weight candidates are generated for each stuck synapse,
each representing a different escape strategy:

```
  BEFORE                                 AFTER (multiple candidates)
  ┌─────┐                               ┌─────┐
  │ I1  │──(w=0.4)──→ H1 (stuck)        │ I1  │──(w=?)──→ H1
  └─────┘                               └─────┘

  Candidate weights tried:
  • Sign flip:       w = -0.4    (reverse direction)
  • Zero reset:      w = 0.0     (disconnect temporarily)
  • Double:          w = 0.8     (amplify)
  • Halve:           w = 0.2     (attenuate)
  • Tenth:           w = 0.04    (near-disconnect)
  • Fixed positive:  w = +1.0    (strong push)
  • Fixed negative:  w = -1.0    (strong reverse push)
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Exploratory weight** | `setWeight` | Each dramatically different value is a separate candidate |

Each candidate's estimated improvement considers the plateau tightness,
error magnitude, and sensitivity factor:
`improvement = mean_error × plateau_tightness × sensitivity × 0.15`.

---

## Example

```
  A creature has a synapse I3 → H7 with weight 0.35.

  Target H7 status:
    Mean |error| = 0.25 (stuck at high error)
    Error std_dev = 0.08
    Coefficient of variation = 0.32 (< 0.4 → plateau confirmed)

  Source I3 status:
    Mean |activation| = 0.6 (actively contributing)

  Candidates generated:
  1. w = -0.35  (sign flip)         improvement ≈ 0.025
  2. w = 0.0    (zero reset)        improvement ≈ 0.020
  3. w = 0.70   (double)            improvement ≈ 0.018
  4. w = 0.175  (halve)             improvement ≈ 0.015
  5. w = 0.035  (tenth)             improvement ≈ 0.012
  6. w = +1.0   (fixed positive)    improvement ≈ 0.022
  7. w = -1.0   (fixed negative)    improvement ≈ 0.022

  NEAT-AI validates each candidate through ablation testing
  to find which escape direction actually improves the score.
```

---

## References

- **Source module**: [`src/analysis/detection/weight_magnitude_reset.rs`](../../src/analysis/detection/weight_magnitude_reset.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Gradient-Based Synapse Adjustment](gradient-discovery.md) —
  small gradient-descent weight changes (complementary approach)
- **Related**: [Error Plateau](error-plateau.md) — detects output neurons
  stuck at high error (similar concept, different fix)
