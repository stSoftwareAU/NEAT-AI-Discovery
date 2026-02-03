# Bottleneck Neuron Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/bottleneck.rs`](../../src/analysis/bottleneck.rs) | **Issue:** [#343](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/343)

---

## The Problem

A **bottleneck neuron** is a single hidden neuron through which many input
signals are forced to pass before reaching the output. One neuron's activation
range cannot encode all upstream information — detail is lost.

```
  Many inputs                One neuron              Outputs
  ┌────┐                   ┌──────────┐
  │ I1 │─────────────────→ │          │
  ├────┤                   │          │
  │ I2 │─────────────────→ │    H1    │ ──────→ O1
  ├────┤                   │  (TANH)  │
  │ I3 │─────────────────→ │          │ ──────→ O2
  ├────┤                   │          │
  │ I4 │─────────────────→ │          │
  ├────┤                   └──────────┘
  │ I5 │─────────────────→      ↑
  └────┘                   Information
                           bottleneck!
                           5 signals compressed
                           into 1 value
```

### Why It Hurts the Creature's Score

- **Information loss**: Five independent input signals are compressed into a
  single scalar. The downstream neurons cannot distinguish which input caused
  the output.
- **Error concentration**: The bottleneck neuron accumulates
  disproportionately large error because it is the only path for error to
  flow back to multiple inputs.
- **Limited expressiveness**: The network cannot learn functions that require
  independent use of the compressed inputs.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  1. Build fan-in / fan-out maps from creature topology     │
  │  2. Filter hidden neurons where:                           │
  │     • fan-in >= 3                                          │
  │     • fan-in / fan-out >= 2.0 (compression ratio)          │
  │  3. Compute error contribution ratio:                      │
  │     neuron_abs_error / total_abs_error                     │
  │  4. Calculate bottleneck score:                            │
  │     60% topology score + 40% error concentration           │
  │     (topology uses log-dampened compression ratio)          │
  │  5. Rank by bottleneck score                               │
  └────────────────────────────────────────────────────────────┘
```

### Scoring Breakdown

```
  Bottleneck Score = 0.6 * topology_score + 0.4 * error_ratio

  topology_score = log2(fan_in / fan_out) / log2(max_ratio)
                   ↑ dampened to avoid over-weighting extreme fan-in
```

---

## How We Fix It

Two complementary strategies relieve the bottleneck:

### Strategy 1: Parallel Neuron

Add a second neuron alongside the bottleneck to share the load:

```
  BEFORE                              AFTER
                                      ┌──────────┐
  I1 ──→                    I1 ──────→│   H1     │──→ O1
  I2 ──→  ┌────┐            I2 ──────→│(original)│──→ O2
  I3 ──→  │ H1 │ ──→ O1     I3 ──────→└──────────┘
  I4 ──→  │    │ ──→ O2
  I5 ──→  └────┘            I4 ──────→┌──────────┐
                             I5 ──────→│   H2     │──→ O1
                                      │  (new)   │──→ O2
                                      └──────────┘
```

The new neuron takes the top half of upstream inputs (by weight magnitude) at
50% scaled weights and connects to all downstream outputs.

### Strategy 2: Bypass Synapse

Connect the strongest upstream neuron directly to downstream targets, skipping
the bottleneck entirely:

```
  BEFORE                              AFTER
                                      I1 ───────────────────→ O1  (bypass)
  I1 ──→ ┌────┐                                │
  I2 ──→ │ H1 │ ──→ O1      I1 ──→ ┌────┐     │
  I3 ──→ │    │ ──→ O2      I2 ──→ │ H1 │ ──→ O1
         └────┘              I3 ──→ │    │ ──→ O2
                                    └────┘
```

Bypass weight = upstream_weight * downstream_weight * 0.5

| Candidate | Operation | When |
|-----------|-----------|------|
| **Parallel neuron** | `addNeuron` + `addSynapse` (multiple) | Always proposed |
| **Bypass synapse** | `addSynapse` (direct) | When fan-in > 3 |

---

## Example

```
  A creature classifying images has 100 pixel inputs feeding through
  a single hidden neuron to 3 output classes:

  100 inputs → [H1] → 3 outputs

  H1 can only output one value per sample.
  It cannot represent all 100 input dimensions.

  Fix: Add H2 that handles half the inputs
  100 inputs → [H1] (50 inputs) → 3 outputs
            → [H2] (50 inputs) → 3 outputs

  Now each neuron handles a subset → more expressive network
```

---

## References

- **Information bottleneck theory** —
  [Wikipedia](https://en.wikipedia.org/wiki/Information_bottleneck_method):
  The information-theoretic framework for understanding compression in neural
  networks.
- **Tishby & Zaslavsky (2015)** — *Deep learning and the information
  bottleneck principle*: Formalises how intermediate layers compress
  information and why bottlenecks limit learning.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
