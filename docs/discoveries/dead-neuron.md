# Dead Neuron Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/dead_neuron.rs`](../../src/analysis/dead_neuron.rs) | **Issue:** [#341](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/341)

---

## The Problem

A **dead neuron** is a hidden neuron that always outputs zero (or near-zero)
regardless of the input. It consumes computation and adds complexity to the
network topology without contributing any signal.

```
  Input layer        Hidden layer         Output layer
  ┌─────┐           ┌──────────┐         ┌─────┐
  │ I1  │──────────→│  H1      │────────→│ O1  │
  │     │           │  output≈0│         │     │
  │ I2  │──────────→│  always  │         └─────┘
  └─────┘           └──────────┘
                         ↑
                    Dead neuron!
                    Adds cost, no benefit
```

### Why It Hurts the Creature's Score

- **Wasted computation**: Every evaluation computes the neuron's activation
  for nothing.
- **Structural bloat**: Extra synapses (both incoming and outgoing) increase
  the creature's complexity cost without improving its score.
- **Evolutionary drag**: NEAT's complexity penalty (cost of growth) means dead
  neurons actively hurt the creature's fitness score.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────┐
  │  For each hidden neuron:                                   │
  │                                                            │
  │  1. Collect activation samples (minimum 20)                │
  │  2. Compute mean absolute activation                       │
  │  3. Compute standard deviation of activations              │
  │  4. Check ALL of:                                          │
  │     • mean |activation| < 0.000001                         │
  │     • std deviation     < 0.000001                         │
  │     • fewer than 1% of samples exceed 0.01                 │
  │  5. If all checks pass → neuron is dead                    │
  └────────────────────────────────────────────────────────────┘
```

### Confidence Scoring

```
  Removal confidence = weighted combination of:

  ┌──────────────────────────────────────────┐
  │  40%  How close mean is to zero          │
  │  40%  How close variance is to zero      │
  │  20%  Sample count (plateaus at 1000)    │
  │                                          │
  │  Final score scaled to [0.5, 1.0]        │
  └──────────────────────────────────────────┘
```

More samples and lower activation both increase confidence that the neuron is
truly dead and not just rarely active.

---

## How We Fix It

The fix is straightforward — **remove the dead neuron** and all its synapses:

```
  BEFORE                              AFTER
  ┌─────┐    ┌────┐    ┌─────┐       ┌─────┐              ┌─────┐
  │ I1  │───→│ H1 │───→│ O1  │       │ I1  │              │ O1  │
  │     │    │DEAD│    │     │       │     │              │     │
  │ I2  │───→│    │    └─────┘       │ I2  │              └─────┘
  └─────┘    └────┘                  └─────┘
                                          ↑                   ↑
                                     Still here          Still here
                                     (no synapses        (other paths
                                      to dead H1)        still work)
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove neuron** | `removeNeuron` | Emitted as a coordinated structural candidate |

The removal also eliminates all incoming and outgoing synapses, simplifying the
network topology.

---

## Example

```
  A creature has evolved 50 hidden neurons over many generations.
  3 of them have mean |activation| < 0.0000001:

  Neuron H12: mean=0.0000000, std=0.0000000, active_fraction=0.0%
  Neuron H34: mean=0.0000000, std=0.0000000, active_fraction=0.2%
  Neuron H47: mean=0.0000000, std=0.0000000, active_fraction=0.0%

  Each dead neuron has ~5 synapses → 15 wasted connections.

  Fix: Remove H12, H34, H47
  Result: 47 neurons, 15 fewer synapses
  → Lower complexity cost, same functional output
```

---

## References

- **Dying RELU problem** —
  [Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Dying_ReLU_problem):
  A well-known variant where RELU neurons become permanently inactive.
- **Network pruning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)):
  The general technique of removing unnecessary neurons and connections to
  improve efficiency.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
