# Bias Perturbation Regime Shift

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/bias_perturbation.rs`](../../src/analysis/detection/bias_perturbation.rs) | **Issue:** [#551](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/551)

---

## The Problem

A **bias perturbation regime shift** targets hidden neurons that are stuck
operating in a suboptimal region of their activation function — either deep
in a saturated tail or confined to a narrow linear sub-zone — when a large
bias shift could move them to a qualitatively different, more productive
regime.

Small gradient-based adjustments cannot escape these local minima because the
error surface is flat in the current region. A deliberate, large bias shift
is needed to "jump" to a better operating point.

```
  TANH activation function
  ────────────────────────

  output
   +1 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                                        ╱─────────
                                      ╱
                                    ╱     ← Active zone
                                  ╱         (useful range)
   0  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─
                             ╱
               ╭─── Neuron stuck here
               │     (saturated tail,
               │      <25% utilisation)
   -1 ─────────╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       -6  -4  -2   0   +2  +4  +6
                   input (value)
```

### Why It Hurts the Creature's Score

- **Gradient starvation**: In the saturated tail, the derivative is near zero,
  so normal weight updates cannot escape the region.
- **Wasted capacity**: The neuron occupies network resources but only produces
  a near-constant output.
- **Local minimum trap**: Small perturbations keep the neuron in the same
  flat region — only a large shift can reach the active zone.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each hidden neuron with a bounded activation:             │
  │                                                                │
  │  1. Determine the squash function's active zone                │
  │     (e.g., TANH: [-2, +2], LOGISTIC: [-4, +4])                │
  │  2. Measure dynamic range utilisation of the active zone       │
  │  3. Check ALL of:                                              │
  │     • Utilisation < 25% of the active zone                     │
  │     • Mean |error| >= 0.05                                     │
  │     • Required bias shift >= 0.1                               │
  │     • Sufficient samples available                             │
  │  4. If all checks pass → regime shift candidate                │
  └────────────────────────────────────────────────────────────────┘
```

### Active Zone Definitions

| Squash | Active Zone | Centre |
|--------|------------|--------|
| TANH / HARD_TANH / CLIPPED | [-2, +2] | 0 |
| LOGISTIC / SOFTSIGN | [-4, +4] | 0 |
| ARCTAN | [-3, +3] | 0 |
| RELU6 | [0, +6] | 3 |

---

## How We Fix It

The fix applies a large bias shift to relocate the neuron's operating point
from the saturated tail to the centre of the active zone:

```
  BEFORE                                 AFTER
  ┌─────┐    ┌────────────┐  ┌─────┐    ┌─────┐    ┌────────────┐  ┌─────┐
  │ I1  │───→│ H1         │─→│ O1  │    │ I1  │───→│ H1         │─→│ O1  │
  │     │    │ TANH       │  │     │    │     │    │ TANH       │  │     │
  │     │    │ bias=-5.0  │  │     │    │     │    │ bias=0.0   │  │     │
  │     │    │ output≈-1  │  │     │    │     │    │ output     │  │     │
  └─────┘    │ (stuck)    │  └─────┘    └─────┘    │ varies!    │  └─────┘
             └────────────┘                        └────────────┘
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Set bias** | `setBias` | Shift bias to centre of the active zone |

The estimated improvement is proportional to both the mean error and how
far from the active zone the neuron currently operates:
`improvement = mean_error × (1 - utilisation) × 0.2`.

---

## Example

```
  A creature has a TANH neuron H5 with bias = -4.8.
  Active zone for TANH is [-2, +2], centre = 0.

  Current state:
    Pre-activation values centred around -4.8
    → Deep in saturated tail, output ≈ -0.9999
    → Utilisation of active zone: 8%
    → Mean |error|: 0.12

  Fix: Set bias to 0.0 (centre of active zone)
  → Pre-activation values now centred around 0
  → Neuron operates in the responsive region
  → Output varies meaningfully with input
```

---

## References

- **Source module**: [`src/analysis/detection/bias_perturbation.rs`](../../src/analysis/detection/bias_perturbation.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Operating Point Analysis](operating-point.md) — analyses
  pre-activation distribution against the active zone
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects neurons
  already at activation bounds
