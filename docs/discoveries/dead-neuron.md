# 💀 Dead Neuron Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/dead_neuron.rs`](../../src/analysis/detection/dead_neuron.rs) | **Issue:** [#341](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/341)

---

## 🔍 The Problem

A **dead neuron** is a hidden neuron that always outputs zero (or near-zero)
regardless of the input. It consumes computation and adds complexity to the
network topology without contributing any signal.

```mermaid
graph LR
    I1["🔵 I1"] -->|"w=0.5"| H1["💀 H1<br/>output ≈ 0<br/><i>always</i>"]
    I2["🔵 I2"] -->|"w=0.3"| H1
    H1 -->|"w=0.4"| O1["🎯 O1"]
    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style I2 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#e74c3c,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
```

> 💀 **Dead neuron!** Adds cost, no benefit — the neuron fires nothing useful.

### ⚠️ Why It Hurts the Creature's Score

- **Wasted computation**: Every evaluation computes the neuron's activation
  for nothing.
- **Structural bloat**: Extra synapses (both incoming and outgoing) increase
  the creature's complexity cost without improving its score.
- **Evolutionary drag**: NEAT's complexity penalty (cost of growth) means dead
  neurons actively hurt the creature's fitness score.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B["📊 Collect activation samples<br/><i>minimum 20</i>"]
    B --> C["📐 Compute mean |activation|"]
    C --> D["📐 Compute std deviation"]
    D --> E{"🧪 ALL checks pass?"}
    E -->|"mean |act| < 0.000001<br/>std dev < 0.000001<br/>< 1% samples > 0.01"| F["💀 Neuron is dead"]
    E -->|"Any check fails"| G["✅ Neuron is alive"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📊 Confidence Scoring

```mermaid
flowchart LR
    subgraph Weights["⚖️ Confidence Components"]
        W1["40% — How close<br/>mean is to zero"]
        W2["40% — How close<br/>variance is to zero"]
        W3["20% — Sample count<br/><i>plateaus at 1000</i>"]
    end
    W1 --> R["🎯 Final score<br/>scaled to [0.5, 1.0]"]
    W2 --> R
    W3 --> R
    style W1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style W2 fill:#e3f2fd,stroke:#1565c0,color:#000
    style W3 fill:#e3f2fd,stroke:#1565c0,color:#000
    style R fill:#e8f5e9,stroke:#2e7d32,color:#000
```

More samples and lower activation both increase confidence that the neuron is
truly dead and not just rarely active.

---

## 🛠️ How We Fix It

The fix is straightforward — **remove the dead neuron** and all its synapses:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BI1["🔵 I1"] -->|"w=0.5"| BH1["💀 H1<br/>DEAD"]
        BI2["🔵 I2"] -->|"w=0.3"| BH1
        BH1 -->|"w=0.4"| BO1["🎯 O1"]
    end
    subgraph After["✅ After"]
        AI1["🔵 I1"]
        AI2["🔵 I2"]
        AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BI2 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AI2 fill:#4a9eff,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove neuron** | `removeNeuron` | Emitted as a coordinated structural candidate |

The removal also eliminates all incoming and outgoing synapses, simplifying the
network topology.

---

## 📝 Example

> A creature has evolved 50 hidden neurons over many generations.
> 3 of them have mean |activation| < 0.0000001:
>
> | Neuron | Mean | Std Dev | Active Fraction |
> |--------|------|---------|-----------------|
> | H12 | 0.0000000 | 0.0000000 | 0.0% |
> | H34 | 0.0000000 | 0.0000000 | 0.2% |
> | H47 | 0.0000000 | 0.0000000 | 0.0% |
>
> Each dead neuron has ~5 synapses → **15 wasted connections**.
>
> **Fix:** Remove H12, H34, H47
> **Result:** 47 neurons, 15 fewer synapses → lower complexity cost, same functional output ✅

---

## 📚 References

- **Dying ReLU problem** —
  [Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Dying_ReLU_problem):
  A well-known variant where ReLU neurons become permanently inactive.
- **Network pruning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)):
  The general technique of removing unnecessary neurons and connections to
  improve efficiency.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
