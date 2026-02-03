# Add Neuron Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/neuron.rs`](../../src/analysis/neuron.rs)

---

## The Problem

A creature's network may lack **intermediate computation** between its inputs
and outputs. Some functions cannot be learned with direct connections alone —
they need a hidden neuron to transform the signal first.

```
  Without intermediate neuron:       With intermediate neuron:

  Input ──(w)──→ Output              Input ──(w_in)──→ [Hidden] ──(w_out)──→ Output
                                                        RELU
  Can only learn:                    Can learn:
  output = w * input                 output = w_out * RELU(w_in * input + bias)

  Linear only!                       Non-linear functions!
```

### Why It Hurts the Creature's Score

- The network cannot represent **non-linear relationships** between certain
  inputs and outputs.
- Error persists because no amount of weight adjustment on existing
  connections can produce the needed transformation.
- The creature has hit the limits of its current topology.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each target neuron T (primarily outputs):             │
  │                                                            │
  │  1. Enumerate all upstream neurons S not directly          │
  │     connected to T                                         │
  │                                                            │
  │  2. Match samples: S's activation with T's error           │
  │     (by observation index, minimum 10 samples)             │
  │                                                            │
  │  3. For each candidate activation function:                │
  │     ┌──────────────────────────────────────────────────┐   │
  │     │  RELU, TANH, LOGISTIC, IDENTITY, ABSOLUTE, etc.  │   │
  │     │                                                  │   │
  │     │  a. Compute: activated = squash(w_in * S + bias)  │   │
  │     │  b. Optimal w_out via least squares:              │   │
  │     │     w_out = Σ(error * activated) / Σ(activated²) │   │
  │     │  c. Optimal bias via grid search (GPU-accelerated)│   │
  │     │  d. Expected improvement = reduction in MSE       │   │
  │     └──────────────────────────────────────────────────┘   │
  │                                                            │
  │  4. Select the activation function with best improvement   │
  │  5. Apply source variance discount + impact discount       │
  └────────────────────────────────────────────────────────────┘
```

### GPU-Accelerated Evaluation

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    GPU Compute Shaders                       │
  │                                                             │
  │  ┌─────────┐    ┌─────────────┐    ┌──────────────────┐    │
  │  │ Source   │    │ Activation  │    │ Weight + Bias     │    │
  │  │ samples  │───→│ functions   │───→│ grid search       │    │
  │  │ (batch)  │    │ (parallel)  │    │ (parallel)        │    │
  │  └─────────┘    └─────────────┘    └──────────────────┘    │
  │                                             │               │
  │                                    ┌────────▼──────────┐    │
  │                                    │ Best improvement   │    │
  │                                    │ per candidate      │    │
  │                                    └───────────────────┘    │
  └─────────────────────────────────────────────────────────────┘

  Evaluating thousands of source × activation × bias combinations
  in parallel on the GPU makes this analysis feasible within the
  discovery deadline.
```

---

## How We Fix It

Insert a new hidden neuron between a source and target:

```
  BEFORE                              AFTER
  ┌───┐               ┌───┐          ┌───┐         ┌─────┐         ┌───┐
  │ S │               │ T │          │ S │─(w_in)─→│H_new│─(w_out)→│ T │
  └───┘               └───┘          └───┘         │RELU │         └───┘
  (not connected)                                  │b=0.1│
                                                   └─────┘

  The new neuron transforms S's signal through RELU
  before feeding it to T, enabling non-linear mapping.
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add neuron** | `addNeuron` | Includes incoming weight, outgoing weight (capped at ±0.1), bias, and activation function |

### Parameter Constraints

```
  ┌──────────────────────────────────────────┐
  │  |incoming weight|  <= 20                │
  │  |outgoing weight|  <= 0.1               │
  │  |bias|             <= 10                │
  │  weight ratio: in/out >= 50 (when in>1)  │
  │                                          │
  │  IDENTITY with |bias| < 0.01 filtered    │
  │  (redundant with direct synapse)         │
  └──────────────────────────────────────────┘
```

---

## Example

```
  Output O1 has persistent error that no weight adjustment can fix.

  Analysis finds: Input I3's activation correlates with O1's error
  pattern, but the relationship is non-linear.

  Best candidate:
  - Source: I3
  - Activation: RELU
  - Incoming weight: 1.2
  - Bias: -0.3
  - Outgoing weight: 0.08

  New neuron H_new:
  output = 0.08 * RELU(1.2 * I3 - 0.3)

  This creates a threshold detector:
  - When I3 < 0.25: output = 0 (RELU cuts off)
  - When I3 > 0.25: output scales linearly
  → captures the non-linear boundary O1 needs

  Production success rate: 5.9% (556 successes from 9,500 candidates)
  This is the highest-volume discovery type.
```

---

## References

- **Universal approximation theorem** —
  [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem):
  Proves that networks with at least one hidden layer can approximate any
  continuous function. Adding neurons increases approximation capacity.
- **Least squares estimation** —
  [Wikipedia](https://en.wikipedia.org/wiki/Least_squares): The method used
  to compute optimal outgoing weights that minimise error.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends. NEAT adds
  neurons through mutation; this discovery accelerates the process by
  identifying where neurons are most needed.
