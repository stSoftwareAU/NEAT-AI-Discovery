# 📉 Remove Low-Impact Neurons

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/neuron/`](../../src/analysis/neuron/)

---

## 🔍 The Problem

Some neurons contribute **less benefit** to the network than the **cost of
their complexity**. In NEAT, every neuron and synapse incurs a "cost of growth"
penalty on the creature's fitness. If a neuron's impact on accuracy is smaller
than this penalty, the creature would score better without it.

```mermaid
graph LR
    subgraph Legend["📊 Impact vs Cost"]
        H1["🧠 H1<br/>HIGH impact ✅"]
        H2["🧠 H2<br/>Medium impact"]
        H3["🧠 H3<br/>LOW impact ❌<br/><i>below cost of growth</i>"]
    end
    style H1 fill:#2ecc71,stroke:#333,color:#fff
    style H2 fill:#f39c12,stroke:#333,color:#fff
    style H3 fill:#e74c3c,stroke:#333,color:#fff
```

> 📉 **H3 costs more to maintain than it contributes.** Removing H3 improves the
> creature's overall fitness.

### ⚠️ Why It Hurts the Creature's Score

- NEAT's fitness function penalises structural complexity.
- A low-impact neuron adds penalty without enough accuracy benefit to
  compensate.
- Removing it reduces the complexity cost while barely affecting predictions.
- The net effect: **higher fitness score**.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B["📊 Compute<br/>activation_weighted_impact"]
    B --> C{"⚖️ impact < costOfGrowth?<br/><i>(default: 1e-7)</i>"}
    C -->|No| Z["✅ Neuron earns its keep"]
    C -->|Yes| D["📐 Factor in synapse count<br/><i>more synapses = bigger savings</i>"]
    D --> E["📉 Candidate for removal"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📊 Impact Scoring

The `activation_weighted_impact` considers:

| Factor | Description |
|--------|-------------|
| 🔥 **Activation magnitude** | How often and how strongly the neuron fires across samples |
| ⚡ **Connection strength** | Weight magnitude of outgoing synapses |
| 📏 **Distance to output** | Closer to output = higher impact |

> A neuron deep in the network with tiny outgoing weights has very low impact —
> prime removal target. 🎯

---

## 🛠️ How We Fix It

Remove the low-impact neuron and all its connections:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BI["🔵 I"] -->|"w=0.1"| BH3["📉 H3<br/>low impact"]
        BI -->|"w=0.1"| BH3
        BH3 -->|"w=0.05"| BO["🎯 O"]
    end
    subgraph After["✅ After"]
        AI["🔵 I"] -->|"via other paths"| AO["🎯 O"]
    end
    style BI fill:#4a9eff,stroke:#333,color:#fff
    style BH3 fill:#e74c3c,stroke:#333,color:#fff
    style BO fill:#2ecc71,stroke:#333,color:#fff
    style AI fill:#4a9eff,stroke:#333,color:#fff
    style AO fill:#2ecc71,stroke:#333,color:#fff
```

> 2 synapses removed, 1 neuron removed → lower complexity cost ✅

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove neuron** | `removeNeuron` | Emitted as a removal candidate |

---

## 📝 Example

> Creature with 50 hidden neurons, costOfGrowth = 1e-7
>
> **Neuron H28:**
> - activation_weighted_impact = 3.2e-8 (below 1e-7 threshold)
> - Fan-in: 4 synapses
> - Fan-out: 2 synapses
> - Total synapses removed: 6
>
> | Metric | Value |
> |--------|-------|
> | Accuracy loss | ~0.000000032 (negligible) |
> | Complexity saved | 1 neuron + 6 synapses |
> | Net fitness improvement | positive ✅ |
>
> **Production success rate: 17.6%** (65 successes from 369 candidates)
> This is the highest success-rate discovery type. 🏆

---

## 📚 References

- **Network pruning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)):
  The general technique of removing low-contribution components from neural
  networks.
- **NEAT complexity penalty** — In NEAT, fitness is adjusted by a complexity
  metric (cost of growth) that penalises larger networks. This ensures
  evolution favours simpler solutions when accuracy is similar.
- **Occam's razor** —
  [Wikipedia](https://en.wikipedia.org/wiki/Occam%27s_razor): The principle
  that simpler explanations (smaller networks) are preferable when they
  perform equally well.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
