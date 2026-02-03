# Multi-Hop Candidate Analysis

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/multi_hop.rs`](../../src/analysis/multi_hop.rs) | **Issue:** [#230](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/230)

---

## The Problem

Some useful signal paths require **two or three connections** to reach a target
neuron. Standard single-hop analysis only considers direct connections (source →
target). Multi-hop analysis discovers **indirect** improvements — neurons that
are correlated with the target's error but are too far away in the network
topology to connect directly.

```
  Single-hop sees:                Multi-hop sees:
  ┌───┐       ┌───┐              ┌───┐       ┌───┐       ┌───┐
  │ A │──?──→│ T │              │ A │──?──→│ B │──?──→│ T │
  └───┘       └───┘              └───┘       └───┘       └───┘

  Only checks if A should        Also checks if A can reach T
  connect directly to T          through an intermediate B
```

### Why It Hurts the Creature's Score

- Valuable signal paths that require intermediaries are **invisible** to
  single-hop analysis.
- The creature misses structural improvements that require adding a relay
  neuron.
- Complex functions often need multi-layer computation that single-hop
  cannot discover.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each target neuron T with error data:                 │
  │                                                            │
  │  STEP 1: Find correlated neurons (not directly connected)  │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │ For each neuron A not connected to T:               │   │
  │  │   Pearson(A_activation, T_error) = r                │   │
  │  │   If |r| >= 0.3 → A is a candidate source           │   │
  │  └─────────────────────────────────────────────────────┘   │
  │                                                            │
  │  STEP 2: Two-hop candidates (A → T bypass)                 │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │ Add direct synapse from A to T                      │   │
  │  │ Weight = ±0.1 (sign from correlation direction)     │   │
  │  └─────────────────────────────────────────────────────┘   │
  │                                                            │
  │  STEP 3: Three-hop candidates (S → relay → T)              │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │ Find source S whose activation correlates with A:   │   │
  │  │   Pearson(S_activation, A_activation) = r2          │   │
  │  │ Combined score = geometric_mean(|r|, |r2|)          │   │
  │  │ If combined >= 0.3 → add relay neuron               │   │
  │  └─────────────────────────────────────────────────────┘   │
  └────────────────────────────────────────────────────────────┘
```

### Pruning to Avoid Combinatorial Explosion

```
  Limits:
  ┌──────────────────────────────────────┐
  │  Max intermediates per target: 10    │
  │  Max total candidates:         50    │
  │  Max hops:                     3     │
  │  Min samples for correlation:  20    │
  └──────────────────────────────────────┘

  Without these limits, a creature with 200 neurons would
  generate 200 * 199 = 39,800 potential pairs to check.
```

---

## How We Fix It

### Two-Hop: Add Bypass Synapse

When neuron A correlates with target T but is not directly connected:

```
  BEFORE                              AFTER
  ┌───┐                ┌───┐          ┌───┐                ┌───┐
  │ A │                │ T │          │ A │────(w=0.1)────→│ T │
  └───┘                └───┘          └───┘    new         └───┘

  A's activation correlates with T's error (|r| >= 0.3)
  → add direct synapse to let the signal through
```

### Three-Hop: Add Relay Neuron

When the signal needs to be transformed before reaching T:

```
  BEFORE                              AFTER
  ┌───┐                ┌───┐          ┌───┐         ┌─────┐         ┌───┐
  │ S │                │ T │          │ S │─(w=0.5)→│relay│─(w=0.1)→│ T │
  └───┘                └───┘          └───┘         │TANH │         └───┘
                                                    │b=0  │
                                                    └─────┘
  S correlates with intermediate A, A correlates with T's error
  → add a new TANH relay neuron to bridge the gap
```

| Hops | Candidate | Operations |
|------|-----------|------------|
| 2 | **Bypass synapse** | `addSynapse` (weight ±0.1) |
| 3 | **Relay neuron** | `addNeuron` (TANH, bias 0) + 2× `addSynapse` |

---

## Example

```
  Target output O1 has high error.

  Analysis finds:
  - Hidden neuron H8 (not connected to O1) has
    Pearson(H8_activation, O1_error) = -0.45

  Two-hop candidate: Add synapse H8 → O1 (weight -0.1)

  Further analysis finds:
  - Input I3 has Pearson(I3_activation, H8_activation) = 0.52
  - Combined score = sqrt(0.45 * 0.52) = 0.48

  Three-hop candidate: Add relay neuron R
  - I3 → R (weight 0.5)
  - R → O1 (weight -0.1)

  Both candidates provide O1 with access to signal it previously
  could not reach, potentially reducing its error.
```

---

## References

- **Multi-hop reasoning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Multi-hop_question_answering):
  The general concept of reasoning through intermediate steps, applied here
  to neural network topology.
- **Network depth and expressiveness** — Deeper networks can represent more
  complex functions; multi-hop analysis discovers where additional depth is
  needed.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends. NEAT naturally
  adds complexity over generations; multi-hop accelerates the discovery of
  useful depth.
