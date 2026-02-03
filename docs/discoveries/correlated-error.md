# Correlated Error Pattern Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/correlated_error.rs`](../../src/analysis/correlated_error.rs) | **Issue:** [#344](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/344)

---

## The Problem

**Correlated error** occurs when multiple output neurons make errors in the same
direction on the same training samples. This pattern suggests a **missing shared
cause** — there is an input feature or combination that affects several outputs,
but the network has no shared hidden neuron to represent it.

```
  Training samples where errors align:

  Sample │ O1 error │ O2 error │ O3 error │ Same direction?
  ───────┼──────────┼──────────┼──────────┼────────────────
    s1   │  +0.3    │  +0.2    │  +0.4    │  YES (all +)
    s2   │  -0.1    │  -0.2    │  -0.15   │  YES (all -)
    s3   │  +0.25   │  +0.3    │  +0.28   │  YES (all +)
    s4   │  -0.2    │  -0.15   │  -0.22   │  YES (all -)
    ...

  O1, O2, and O3 err together → likely a shared missing feature
```

### Why It Hurts the Creature's Score

- Each output neuron independently tries to compensate for the missing
  feature, duplicating effort.
- Without a shared representation, the network cannot learn the common
  pattern efficiently.
- The correlated errors compound — fixing one output does not fix the others.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  1. Compute pairwise Pearson correlation of per-sample     │
  │     errors between ALL output neuron pairs                 │
  │                                                            │
  │  2. Cluster using complete-linkage:                        │
  │     group outputs where ALL pairwise correlations >= 0.7   │
  │                                                            │
  │      O1 ──── r=0.85 ──── O2                                │
  │       │                   │                                │
  │       └─── r=0.72 ───O3──┘ r=0.78                         │
  │       All pairs >= 0.7 → {O1, O2, O3} form a cluster      │
  │                                                            │
  │  3. For each cluster, find predictive inputs:              │
  │     correlation(input_activation, avg_group_error) >= 0.4  │
  │                                                            │
  │  4. Estimated improvement =                                │
  │     mean_correlation * mean_abs_error * group_size * 0.01  │
  └────────────────────────────────────────────────────────────┘
```

### Complete-Linkage Clustering

```
  Complete-linkage ensures tight clusters where EVERY pair is correlated:

  ┌──────────────────────────────────┐
  │  Single-linkage (NOT used):      │
  │  O1─(0.9)─O2─(0.8)─O3─(0.3)─O4 │
  │  → {O1,O2,O3,O4} all in one     │
  │    cluster (even though O1↔O4    │
  │    correlation might be low)     │
  │                                  │
  │  Complete-linkage (used):        │
  │  O1─(0.9)─O2─(0.8)─O3           │
  │  → {O1,O2,O3} only if ALL       │
  │    pairwise r >= 0.7             │
  └──────────────────────────────────┘
```

---

## How We Fix It

Add a **shared hidden neuron** that captures the common pattern and feeds all
correlated outputs:

```
  BEFORE                              AFTER
  ┌────┐         ┌────┐              ┌────┐         ┌────┐
  │ I1 │────────→│ O1 │              │ I1 │────────→│ O1 │
  └────┘         └────┘              └────┘    ↗    └────┘
  ┌────┐         ┌────┐              ┌────┐  ╱      ┌────┐
  │ I2 │────────→│ O2 │              │ I2 │→[H_new]→│ O2 │
  └────┘         └────┘              └────┘  ╲      └────┘
  ┌────┐         ┌────┐              ┌────┐    ↘    ┌────┐
  │ I3 │────────→│ O3 │              │ I3 │────────→│ O3 │
  └────┘         └────┘              └────┘         └────┘

  Correlated errors                   H_new captures the shared pattern
  across O1, O2, O3                   and feeds corrective signal to all
                                      three outputs simultaneously
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add shared neuron** | `addNeuron` (TANH) | Placed before the first output in the cluster |
| **Connect inputs** | `addSynapse` (weight 0.5) | From each predictive input to the new neuron |
| **Connect outputs** | `addSynapse` (weight 0.1) | From the new neuron to each correlated output |

---

## Example

```
  A creature predicting RGB colour values has 3 outputs: R, G, B.

  Analysis shows:
  - Pearson(R_error, G_error) = 0.82
  - Pearson(R_error, B_error) = 0.75
  - Pearson(G_error, B_error) = 0.79
  All >= 0.7 → {R, G, B} form a correlated error cluster.

  Predictive input I7 (brightness) correlates with average group
  error at r = 0.65.

  Fix: Add hidden neuron H_new
  - I7 → H_new (weight 0.5)
  - H_new → R output (weight 0.1)
  - H_new → G output (weight 0.1)
  - H_new → B output (weight 0.1)

  H_new now represents the shared brightness correction.
```

---

## References

- **Correlation clustering** —
  [Wikipedia](https://en.wikipedia.org/wiki/Correlation_clustering): The
  general framework for grouping items by pairwise similarity.
- **Multi-task learning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Multi-task_learning): The
  principle that shared representations improve learning when tasks are
  related (correlated errors indicate related output tasks).
- **Pearson correlation coefficient** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient):
  The statistical measure used to detect co-occurring error patterns.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
