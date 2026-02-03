# Redundant Path Pruning

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/redundant_path.rs`](../../src/analysis/redundant_path.rs) | **Issue:** [#164](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/164)

---

## The Problem

**Redundant paths** occur when two synapses feeding the same target neuron carry
effectively the same signal. The network wastes two connections to transmit
information that one could handle.

```
  Source A ──(w=+0.4)──→┌────────┐
                         │ Target │
  Source B ──(w=+0.3)──→│   T    │
                         └────────┘

  If A and B activations are highly correlated (r >= 0.85):
  Both synapses carry ~the same information
  → one is redundant
```

### Why It Hurts the Creature's Score

- **Structural complexity**: Two synapses where one would suffice increases
  the creature's complexity cost.
- **Fragile encoding**: If one path mutates slightly, the near-duplicate
  can cause unexpected changes.
- **Wasted capacity**: The creature could use those connections for genuinely
  different signals.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each target neuron T:                                 │
  │                                                            │
  │  1. Collect all existing source synapses with recorded     │
  │     activation samples (minimum 30 matched samples)        │
  │                                                            │
  │  2. Compute pairwise |Pearson correlation| of source       │
  │     activations                                            │
  │                                                            │
  │  3. For each pair where |correlation| >= 0.85:             │
  │     ┌──────────────────────────────────────────────────┐   │
  │     │  Weaker synapse (by |weight|) = prune candidate  │   │
  │     │  Stronger synapse = survivor                     │   │
  │     │                                                  │   │
  │     │  New survivor weight = keep_weight + prune_weight│   │
  │     │  (absorbs both contributions)                    │   │
  │     └──────────────────────────────────────────────────┘   │
  │                                                            │
  │  4. Estimate improvement by comparing MSE:                 │
  │     original two-path vs renormalised single-path          │
  │     + structural simplification bonus                      │
  └────────────────────────────────────────────────────────────┘
```

### Correlation Threshold

```
  |Pearson r|:

  0.0 ─── 0.5 ─── 0.85 ─── 1.0
   │        │        │        │
   │        │        │        └─ Identical signals
   │        │        └────────── REDUNDANT (threshold)
   │        └─────────────────── Different enough to keep
   └──────────────────────────── Unrelated signals

  Note: Uses ABSOLUTE correlation — both positively correlated
  (r ≈ +1.0) and anti-correlated (r ≈ -1.0) signals count as
  redundant (they carry the same information, just inverted).
```

---

## How We Fix It

Remove the weaker synapse and renormalise the survivor:

```
  BEFORE                              AFTER
  ┌───┐                               ┌───┐
  │ A │──(w=+0.4)──→┌───┐             │ A │──(w=+0.7)──→┌───┐
  └───┘              │ T │             └───┘    ↑        │ T │
  ┌───┐              │   │                     │        │   │
  │ B │──(w=+0.3)──→└───┘             ┌───┐   │        └───┘
  └───┘  ↑                            │ B │   combined
         weaker                       └───┘   weight
         (removed)                    (disconnected
                                       from T)

  Survivor weight = 0.4 + 0.3 = 0.7
  Signal to T is approximately preserved but simpler.
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove weaker synapse** | `removeSynapse` | The lower |weight| path |
| **Renormalise survivor** | `setWeight` | New weight = sum of both weights |

---

## Example

```
  Target hidden neuron H5 has two inputs:

  Synapse I2 → H5: weight = +0.45
  Synapse I7 → H5: weight = +0.22

  Activation correlation between I2 and I7: |r| = 0.91

  Since 0.91 >= 0.85 and I7 has the smaller |weight|:

  Fix:
  1. removeSynapse I7 → H5
  2. setWeight I2 → H5 to 0.45 + 0.22 = 0.67

  MSE comparison shows the single-path produces nearly identical
  output with one fewer synapse.
```

---

## References

- **Feature redundancy** —
  [Wikipedia](https://en.wikipedia.org/wiki/Feature_selection#Redundancy):
  The general problem of redundant features in machine learning.
- **Optimal Brain Damage (LeCun et al., 1989)** — Foundational work on
  identifying and removing unnecessary connections in neural networks.
- **Optimal Brain Surgeon (Hasselmo et al., 1992)** — Extends pruning to
  account for weight renormalisation after removal, similar to this approach.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
