# 🚧 Bottleneck Neuron Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/bottleneck.rs`](../../src/analysis/detection/bottleneck.rs) | **Issue:** [#343](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/343)

---

## 🔍 The Problem

A **bottleneck neuron** is a single hidden neuron through which many input
signals are forced to pass before reaching the output. One neuron's activation
range cannot encode all upstream information — detail is lost.

```mermaid
graph LR
    I1["🔵 I1"]:::input --> H1["🔴 H1\n(TANH)"]:::problem
    I2["🔵 I2"]:::input --> H1
    I3["🔵 I3"]:::input --> H1
    I4["🔵 I4"]:::input --> H1
    I5["🔵 I5"]:::input --> H1
    H1 --> O1["🟢 O1"]:::output
    H1 --> O2["🟢 O2"]:::output

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
```

> 🚧 **Information bottleneck!** Five signals are compressed into a single
> neuron value. The downstream neurons cannot recover the original inputs.

### ⚠️ Why It Hurts the Creature's Score

- **Information loss**: Five independent input signals are compressed into a
  single scalar. The downstream neurons cannot distinguish which input caused
  the output.
- **Error concentration**: The bottleneck neuron accumulates
  disproportionately large error because it is the only path for error to
  flow back to multiple inputs.
- **Limited expressiveness**: The network cannot learn functions that require
  independent use of the compressed inputs.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    S1["📐 Build fan-in / fan-out maps\nfrom creature topology"]:::process
    S2{"🧮 fan-in ≥ 3?\nfan-in / fan-out ≥ 2.0?"}:::decision
    S3["📊 Compute error contribution ratio\nneuron_abs_error / total_abs_error"]:::process
    S4["🧮 Calculate bottleneck score\n60% topology + 40% error concentration\n(topology uses log-dampened compression ratio)"]:::process
    S5["📋 Rank by bottleneck score"]:::process

    S1 --> S2
    S2 -->|Yes| S3
    S2 -->|No| Skip["⏭️ Skip neuron"]:::output
    S3 --> S4
    S4 --> S5

    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,color:#000
    classDef output fill:#2ecc71,stroke:#333,color:#fff
```

### 📊 Scoring Breakdown

```mermaid
graph LR
    TS["📐 topology_score\nlog₂(fan_in / fan_out)\n÷ log₂(max_ratio)"]:::process
    ER["📈 error_ratio\nneuron_abs_error\n÷ total_abs_error"]:::process
    BS["🏆 Bottleneck Score\n0.6 × topology\n+ 0.4 × error"]:::decision

    TS -->|"× 0.6"| BS
    ER -->|"× 0.4"| BS

    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,color:#000
```

> 💡 **Note:** The topology score is log-dampened to avoid over-weighting
> extreme fan-in values.

---

## 🛠️ How We Fix It

Two complementary strategies relieve the bottleneck:

### 🔀 Strategy 1: Parallel Neuron

Add a second neuron alongside the bottleneck to share the load:

```mermaid
graph LR
    subgraph before ["❌ Before"]
        direction LR
        bI1["🔵 I1"]:::input --> bH1["🔴 H1"]:::problem
        bI2["🔵 I2"]:::input --> bH1
        bI3["🔵 I3"]:::input --> bH1
        bI4["🔵 I4"]:::input --> bH1
        bI5["🔵 I5"]:::input --> bH1
        bH1 --> bO1["🟢 O1"]:::output
        bH1 --> bO2["🟢 O2"]:::output
    end

    subgraph after ["✅ After"]
        direction LR
        aI1["🔵 I1"]:::input --> aH1["🟢 H1\n(original)"]:::fixed
        aI2["🔵 I2"]:::input --> aH1
        aI3["🔵 I3"]:::input --> aH1
        aH1 --> aO1["🟢 O1"]:::output
        aH1 --> aO2["🟢 O2"]:::output
        aI4["🔵 I4"]:::input --> aH2["🟢 H2\n(new)"]:::fixed
        aI5["🔵 I5"]:::input --> aH2
        aH2 --> aO1
        aH2 --> aO2
    end

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
    classDef fixed fill:#2ecc71,stroke:#333,color:#fff
```

The new neuron takes the top half of upstream inputs (by weight magnitude) at
50% scaled weights and connects to all downstream outputs.

### 🔗 Strategy 2: Bypass Synapse

Connect the strongest upstream neuron directly to downstream targets, skipping
the bottleneck entirely:

```mermaid
graph LR
    subgraph before2 ["❌ Before"]
        direction LR
        bI1b["🔵 I1"]:::input --> bH1b["🔴 H1"]:::problem
        bI2b["🔵 I2"]:::input --> bH1b
        bI3b["🔵 I3"]:::input --> bH1b
        bH1b --> bO1b["🟢 O1"]:::output
        bH1b --> bO2b["🟢 O2"]:::output
    end

    subgraph after2 ["✅ After"]
        direction LR
        aI1b["🔵 I1"]:::input --> aH1b["🟢 H1"]:::fixed
        aI2b["🔵 I2"]:::input --> aH1b
        aI3b["🔵 I3"]:::input --> aH1b
        aH1b --> aO1b["🟢 O1"]:::output
        aH1b --> aO2b["🟢 O2"]:::output
        aI1b -->|"bypass"| aO1b
    end

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
    classDef fixed fill:#2ecc71,stroke:#333,color:#fff
```

> 💡 **Bypass weight** = `upstream_weight × downstream_weight × 0.5`

| Candidate | Operation | When |
|-----------|-----------|------|
| **Parallel neuron** | `addNeuron` + `addSynapse` (multiple) | Always proposed |
| **Bypass synapse** | `addSynapse` (direct) | When fan-in > 3 |

---

## 📝 Example

> A creature classifying images has 100 pixel inputs feeding through a single
> hidden neuron to 3 output classes.

```mermaid
graph LR
    subgraph before3 ["❌ Before — single bottleneck"]
        direction LR
        inputs["🔵 100 inputs"]:::input --> H1e["🔴 H1"]:::problem
        H1e --> outputs["🟢 3 outputs"]:::output
    end

    subgraph after3 ["✅ After — load shared"]
        direction LR
        inputs2["🔵 100 inputs"]:::input --> H1f["🟢 H1\n(50 inputs)"]:::fixed
        inputs2 --> H2f["🟢 H2\n(50 inputs)"]:::fixed
        H1f --> outputs2["🟢 3 outputs"]:::output
        H2f --> outputs2
    end

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
    classDef fixed fill:#2ecc71,stroke:#333,color:#fff
```

> **H1** can only output one value per sample — it cannot represent all 100
> input dimensions. By adding **H2** to handle half the inputs, each neuron
> manages a subset, yielding a more expressive network.

---

## 📚 References

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
