# Symmetry Breaking Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/symmetry_breaking.rs`](../../src/analysis/detection/symmetry_breaking.rs) | **Issue:** [#569](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/569)

---

## The Problem

**Symmetry** in a neural network occurs when two hidden neurons have converged
to near-identical weight configurations — the same activation function, similar
biases, and highly similar incoming weight vectors. These neurons compute
essentially the same function, wasting representational capacity.

Unlike [co-adaptation](co-adaptation.md) (which detects correlated
*activations*), symmetry breaking detects structural similarity in *weights*
— the neurons are configured identically regardless of whether correlations
show up in the current training data.

```
  Input layer        Hidden layer         Output layer
  ┌─────┐
  │ I1  │──(w=0.5)──→┌────────┐
  │     │──(w=0.3)──→│ H1     │────────→┌─────┐
  └─────┘            │ TANH   │         │ O1  │
  ┌─────┐            │bias=0.2│         └─────┘
  │ I2  │──(w=0.5)──→└────────┘
  └─────┘
  ┌─────┐            ┌────────┐
  │ I1  │──(w=0.48)─→│ H2     │────────→┌─────┐
  │     │──(w=0.31)─→│ TANH   │         │ O1  │
  └─────┘            │bias=0.1│         └─────┘
  ┌─────┐            └────────┘
  │ I2  │──(w=0.49)─→
  └─────┘       ↑
           Cosine similarity ≥ 0.95
           Same function, wasted neuron!
```

### Why It Hurts the Creature's Score

- **Redundant computation**: Two neurons computing the same transformation.
- **Blocked learning**: Gradient updates affect both neurons similarly,
  preventing them from diverging through normal training.
- **Complexity cost**: NEAT's cost of growth penalises the extra neuron
  and its synapses without any benefit.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each pair of hidden neurons (H_a, H_b):                   │
  │                                                                │
  │  1. Check same activation function                             │
  │  2. Check |bias_a - bias_b| <= 0.5                             │
  │  3. Build weight vectors from shared source neurons            │
  │  4. Compute cosine similarity of weight vectors                │
  │  5. If cosine similarity >= 0.95 → symmetric pair detected     │
  └────────────────────────────────────────────────────────────────┘
```

Requires at least 2 hidden neurons in the network.

---

## How We Fix It

Rather than removing one neuron (which would be destructive), the fix
**perturbs** one neuron's weights and bias to break the symmetry, giving
it a chance to specialise on a different function:

```
  BEFORE (symmetric)                     AFTER (perturbed)
  ┌────────┐                             ┌────────┐
  │ H2     │                             │ H2     │
  │ TANH   │                             │ TANH   │
  │bias=0.1│                             │bias=0.4│  (+0.3)
  │w=[.48, .31, .49]│                    │w=[.34, .22, .34]│  (×0.7)
  └────────┘                             └────────┘
  Cosine sim = 0.98                      Now different from H1!
```

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Perturb neuron B** | `setBias` + `setWeight` (per synapse) | Shift bias by +0.3, scale all incoming weights by 0.7 |

The perturbation is deliberately conservative — enough to break the symmetry
but not so large as to destroy the neuron's learned contribution.

---

## Example

```
  A creature has 20 hidden neurons. Discovery finds:

  Neuron H4:  TANH, bias=0.15, weights from I1,I2,I3 = [0.6, -0.2, 0.8]
  Neuron H12: TANH, bias=0.10, weights from I1,I2,I3 = [0.58, -0.19, 0.81]

  |bias difference| = 0.05 (< 0.5 threshold)
  Cosine similarity = 0.998 (>= 0.95 threshold)

  Candidate: Perturb H12
    New bias: 0.10 + 0.30 = 0.40
    New weights: [0.41, -0.13, 0.57] (×0.7)

  After fix: H12 now computes a different function,
  freeing up capacity to learn new patterns.
```

---

## References

- **Source module**: [`src/analysis/detection/symmetry_breaking.rs`](../../src/analysis/detection/symmetry_breaking.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Co-Adaptation](co-adaptation.md) — detects correlated
  activations (behavioural similarity)
- **Related**: [Redundant Path](redundant-path.md) — detects duplicate
  signal paths feeding the same target
