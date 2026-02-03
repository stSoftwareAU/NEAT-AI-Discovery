# Add Synapse Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/synapse.rs`](../../src/analysis/synapse.rs)

---

## The Problem

A creature's network may be missing a **direct connection** between two neurons
that should communicate. The source neuron carries useful information that would
reduce the target's error, but no synapse exists to transmit it.

```
  Source neuron               Target neuron
  ┌──────────┐                ┌──────────┐
  │          │   No synapse   │          │
  │    S     │ · · · · · · ·→ │    T     │
  │          │   (missing!)   │          │
  └──────────┘                └──────────┘
       │                           │
  Has useful signal           Has error that
  for T                       S could reduce
```

### Why It Hurts the Creature's Score

- The target neuron cannot access information that would help it produce
  better predictions.
- Error persists because the useful signal from S never reaches T.
- In NEAT evolution, adding connections is random — discovery identifies
  **which specific connections** would help most.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each target neuron T:                                 │
  │                                                            │
  │  1. Enumerate all neurons S not directly connected to T    │
  │     (that appear earlier in evaluation order)              │
  │                                                            │
  │  2. Match samples: S's activation with T's error           │
  │     (by observation index)                                 │
  │                                                            │
  │  3. Compute optimal weight via least squares:              │
  │     w = Σ(error * activation) / Σ(activation²)             │
  │     (capped at ±0.1)                                       │
  │                                                            │
  │  4. Estimate improvement: how much would adding            │
  │     w * S_activation reduce T's error?                     │
  │                                                            │
  │  5. Filter: only keep candidates with positive             │
  │     expected improvement                                   │
  └────────────────────────────────────────────────────────────┘
```

### Epistatic Pair Detection

The analysis also detects **epistatic pairs** — two neurons that complement
each other and work better together than alone:

```
  Epistatic pair: S1 and S2 fire on different samples

  Sample │ S1 active │ S2 active │ Coverage
  ───────┼───────────┼───────────┼─────────
    s1   │    YES    │    no     │  S1
    s2   │    no     │    YES    │  S2
    s3   │    YES    │    no     │  S1
    s4   │    no     │    YES    │  S2
    s5   │    no     │    no     │  gap

  >= 70% non-overlapping activation patterns
  → S1 and S2 together cover more error cases
  → Proposed as a coordinated pair
```

---

## How We Fix It

Add the missing synapse with an optimally computed weight:

```
  BEFORE                              AFTER
  ┌───┐               ┌───┐          ┌───┐                ┌───┐
  │ S │               │ T │          │ S │──(w=+0.08)───→│ T │
  └───┘               └───┘          └───┘    new         └───┘
  (not connected)                    synapse

  Weight computed to minimise T's error given S's activation.
```

For epistatic pairs, both synapses are added together:

```
  BEFORE                              AFTER
  ┌────┐              ┌───┐          ┌────┐               ┌───┐
  │ S1 │              │ T │          │ S1 │──(w=+0.06)──→│ T │
  └────┘              └───┘          └────┘               │   │
  ┌────┐                             ┌────┐               │   │
  │ S2 │                             │ S2 │──(w=-0.04)──→│   │
  └────┘                             └────┘               └───┘

  Both added atomically as a coordinated structural candidate.
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add synapse** | `addSynapse` | Weight capped at ±0.1 |
| **Epistatic pair** | 2× `addSynapse` (coordinated) | Both added atomically |

---

## Example

```
  Output neuron O1 predicts house prices.
  Input I5 contains "number of bedrooms" but has no synapse to O1.

  Analysis:
  - Matched 500 samples of I5 activation with O1 error
  - Optimal weight: w = +0.07
  - Expected improvement: 12% reduction in O1's MSE

  Fix: addSynapse I5 → O1 (weight +0.07)

  Now O1 can factor in bedroom count directly.
```

---

## References

- **Least squares estimation** —
  [Wikipedia](https://en.wikipedia.org/wiki/Least_squares): The method used
  to compute optimal synapse weights that minimise target error.
- **Epistasis** —
  [Wikipedia](https://en.wikipedia.org/wiki/Epistasis): The biological
  concept of gene interactions, applied here to neural connections that work
  better in combination than individually.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends. NEAT adds
  connections through random mutation; this discovery identifies the most
  beneficial connections to add.
