# Skip Connection Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/skip_connection.rs`](../../src/analysis/detection/skip_connection.rs) | **Issue:** [#570](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/570)

---

## The Problem

In deep networks, neurons far from the input layer suffer from **gradient
attenuation** — the error signal weakens as it passes back through multiple
layers, leaving deep neurons with little guidance for improvement. A **skip
connection** (also known as a residual connection) adds a direct shortcut
from a shallow neuron to a deep one, restoring the error signal.

```
  Input layer   Hidden layers (deep)      Output layer
  ┌─────┐      ┌────┐   ┌────┐   ┌────┐   ┌─────┐
  │ I1  │─────→│ H1 │──→│ H3 │──→│ H5 │──→│ O1  │
  │     │      │d=1 │   │d=2 │   │d=3 │   │     │
  └─────┘      └────┘   └────┘   └────┘   └─────┘
                                    ↑
                              Depth >= 3
                              Mean |error| only 30% of shallow neurons
                              → Gradient attenuation!
```

### Why It Hurts the Creature's Score

- **Weak learning signal**: Deep neurons receive attenuated error gradients,
  slowing their adaptation.
- **Wasted depth**: The creature has evolved a deep topology but cannot
  effectively use neurons far from the output.
- **Vanishing gradients**: A well-known problem in deep neural networks
  that skip connections directly address.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each hidden neuron:                                       │
  │                                                                │
  │  1. Compute topological depth (forward BFS from inputs)        │
  │  2. If depth >= 3 → potentially deep                           │
  │  3. Compare mean |error| to shallow neurons (depth <= 2)       │
  │  4. If deep neuron's error < 50% of shallow mean               │
  │     → gradient attenuation confirmed                           │
  │  5. Find a shallow source (input or depth <= 1) not already    │
  │     connected to the deep neuron                               │
  │  6. If source found → skip connection candidate                │
  └────────────────────────────────────────────────────────────────┘
```

### Source Selection

The best shallow source is chosen by maximising the depth gap (difference
in topological depth) while ensuring no existing connection exists.

---

## How We Fix It

```
  BEFORE (gradient attenuation)          AFTER (skip connection added)
  ┌─────┐   ┌────┐   ┌────┐   ┌────┐   ┌─────┐   ┌────┐   ┌────┐   ┌────┐
  │ I1  │──→│ H1 │──→│ H3 │──→│ H5 │   │ I1  │──→│ H1 │──→│ H3 │──→│ H5 │
  │     │   │    │   │    │   │    │   │     │   │    │   │    │   │    │
  └─────┘   └────┘   └────┘   └────┘   └─────┘   └────┘   └────┘   └────┘
                                │              │                      ↑
                                ▼              └──────────────────────┘
                               O1              Direct skip connection!
                                               (w = 0.01 × min(err, 1.0))
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add skip connection** | `addSynapse` | Connect shallow source directly to deep neuron |

The skip connection weight is conservative:
`weight = 0.01 × min(target_mean_error, 1.0)`.

Estimated improvement: `depth_gap × (1 - attenuation_ratio) × 0.005`.

---

## Example

```
  A creature has a chain: I1 → H1 → H2 → H3 → H4 → O1

  Shallow neurons (depth <= 2): mean |error| = 0.15
  Deep neuron H4 (depth = 4): mean |error| = 0.04

  Attenuation ratio: 0.04 / 0.15 = 0.27 (< 0.50 threshold)
  → Gradient attenuation confirmed

  Best shallow source: I1 (depth = 0, not connected to H4)
  Depth gap: 4

  Candidate: Add synapse I1 → H4
    Weight: 0.01 × min(0.04, 1.0) = 0.0004
    Estimated improvement: 4 × (1 - 0.27) × 0.005 = 0.015

  After fix: H4 receives direct signal from I1,
  bypassing the attenuating chain of hidden neurons.
```

---

## References

- **Source module**: [`src/analysis/detection/skip_connection.rs`](../../src/analysis/detection/skip_connection.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Topology Diversification](topology-diversification.md) —
  adds hidden neurons to flat (depth-0) paths
- **Related**: [Multi-Hop](multi-hop.md) — builds indirect connection paths
- **Residual connections (ResNets)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network):
  The skip connection concept from deep learning that inspired this module.
