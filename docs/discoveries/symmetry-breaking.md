# 🪞 Symmetry Breaking Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/symmetry_breaking.rs`](../../src/analysis/detection/symmetry_breaking.rs) | **Issue:** [#569](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/569)

---

## 🔍 The Problem

**Symmetry** in a neural network occurs when two hidden neurons have converged
to near-identical weight configurations — the same activation function, similar
biases, and highly similar incoming weight vectors. These neurons compute
essentially the same function, wasting representational capacity.

Unlike [co-adaptation](co-adaptation.md) (which detects correlated
*activations*), symmetry breaking detects structural similarity in *weights*
— the neurons are configured identically regardless of whether correlations
show up in the current training data.

```mermaid
graph TD
    I1["🔵 I1"] -->|"w=0.50"| H1["🪞 H1<br/>TANH<br/>bias=0.2"]
    I1 -->|"w=0.48"| H2["🪞 H2<br/>TANH<br/>bias=0.1"]
    I2["🔵 I2"] -->|"w=0.30"| H1
    I2 -->|"w=0.31"| H2
    H1 --> O1["🎯 O1"]
    H2 --> O1
    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style I2 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#e74c3c,stroke:#333,color:#fff
    style H2 fill:#e74c3c,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
```

> 🪞 **Mirror neurons!** Cosine similarity ≥ 0.95 — H1 and H2 compute essentially the same function, wasting a neuron slot.

### ⚠️ Why It Hurts the Creature's Score

- **Redundant computation**: Two neurons computing the same transformation.
- **Blocked learning**: Gradient updates affect both neurons similarly,
  preventing them from diverging through normal training.
- **Complexity cost**: NEAT's cost of growth penalises the extra neuron
  and its synapses without any benefit.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each pair of hidden neurons<br/>(H_a, H_b)"] --> B{"🧪 Same activation function?"}
    B -->|"No"| Z["✅ Not symmetric"]
    B -->|"Yes"| C{"|bias_a − bias_b| <= 0.5?"}
    C -->|"No"| Z
    C -->|"Yes"| D["📐 Build weight vectors<br/>from shared source neurons"]
    D --> E["📐 Compute cosine similarity"]
    E --> F{"🪞 Cosine similarity >= 0.95?"}
    F -->|"Yes"| G["🪞 Symmetric pair detected"]
    F -->|"No"| Z
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

Requires at least 2 hidden neurons in the network.

---

## 🛠️ How We Fix It

Rather than removing one neuron (which would be destructive), the fix
**perturbs** one neuron's weights and bias to break the symmetry, giving
it a chance to specialise on a different function:

```mermaid
graph LR
    subgraph Before["❌ Before — symmetric"]
        BH2["🪞 H2<br/>TANH<br/>bias=0.1<br/>w=[.48, .31, .49]<br/><i>cosine sim = 0.98</i>"]
    end
    subgraph After["✅ After — perturbed"]
        AH2["✨ H2<br/>TANH<br/>bias=0.4 (+0.3)<br/>w=[.34, .22, .34] (×0.7)<br/><i>now different from H1!</i>"]
    end
    Before -->|"setBias + setWeight"| After
    style BH2 fill:#e74c3c,stroke:#333,color:#fff
    style AH2 fill:#2ecc71,stroke:#333,color:#fff
```

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Perturb neuron B** | `setBias` + `setWeight` (per synapse) | Shift bias by +0.3, scale all incoming weights by 0.7 |

The perturbation is deliberately conservative — enough to break the symmetry
but not so large as to destroy the neuron's learned contribution.

---

## 📝 Example

> A creature has 20 hidden neurons. Discovery finds:
>
> | Neuron | Activation | Bias | Weights from I1, I2, I3 |
> |--------|-----------|------|------------------------|
> | H4 | TANH | 0.15 | [0.6, −0.2, 0.8] |
> | H12 | TANH | 0.10 | [0.58, −0.19, 0.81] |
>
> |bias difference| = 0.05 (< 0.5 threshold)
> Cosine similarity = 0.998 (>= 0.95 threshold)
>
> **Candidate:** Perturb H12
> New bias: 0.10 + 0.30 = **0.40**
> New weights: [0.41, −0.13, 0.57] (×0.7)
>
> **After fix:** H12 now computes a different function,
> freeing up capacity to learn new patterns ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/symmetry_breaking.rs`](../../src/analysis/detection/symmetry_breaking.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Co-Adaptation](co-adaptation.md) — detects correlated
  activations (behavioural similarity)
- **Related**: [Redundant Path](redundant-path.md) — detects duplicate
  signal paths feeding the same target
