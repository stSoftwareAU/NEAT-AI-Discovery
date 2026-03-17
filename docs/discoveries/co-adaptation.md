# 🪞 Co-Adaptation Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/co_adaptation.rs`](../../src/analysis/detection/co_adaptation.rs) | **Issue:** [#571](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/571)

---

## 🔍 The Problem

**Co-adaptation** occurs when two hidden neurons evolve to encode the same
information — their activations become highly correlated (or anti-correlated)
across training samples. This wastes network capacity because two neurons are
doing one neuron's job.

```mermaid
graph LR
    I1["🔵 I1"] --> H1["🧠 H1<br/>act = [.2, .8, .1, .9]"]
    I2["🔵 I2"] --> H1
    I1 --> H2["🪞 H2<br/>act = [.2, .8, .1, .9]"]
    I2 --> H2
    H1 --> O1["🎯 O1"]
    H2 --> O1
    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style I2 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#9b59b6,stroke:#333,color:#fff
    style H2 fill:#e74c3c,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
```

> 🪞 **Correlation ≥ 0.9 — same information!** H2 is a copy of H1.
> Wasted capacity and unnecessary complexity cost.

### ⚠️ Why It Hurts the Creature's Score

- **Redundant computation**: Two neurons consuming resources to produce
  effectively the same signal.
- **Wasted capacity**: The network could represent an additional independent
  feature if one neuron were freed from copying the other.
- **Complexity cost**: NEAT penalises structural complexity — co-adapted
  neurons add cost without adding capability.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each pair of<br/>hidden neurons (H_a, H_b)"] --> B["📊 Collect activation samples<br/><i>minimum 20 each</i>"]
    B --> C["🔗 Match samples by<br/>observation index"]
    C --> D["📈 Compute Pearson correlation<br/>of activations"]
    D --> E{"|correlation| ≥ 0.9?"}
    E -->|Yes| F["🪞 Co-adapted pair detected!"]
    E -->|No| G["✅ Neurons are independent"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
```

> **Note:** Anti-correlation (≤ −0.9) also counts — the neurons are encoding
> the same information with flipped sign.

Requires at least 2 hidden neurons in the network.

---

## 🛠️ How We Fix It

Two candidate strategies are proposed for each co-adapted pair:

### ✂️ Strategy 1: Remove the Redundant Neuron

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BI1["🔵 I1"] --> BH1["🧠 H1"]
        BI1 --> BH2["🪞 H2"]
        BH1 --> BO1["🎯 O1"]
        BH2 --> BO1
    end
    subgraph After["✅ After"]
        AI1["🔵 I1"] --> AH1["🧠 H1"]
        AH1 --> AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#9b59b6,stroke:#333,color:#fff
    style BH2 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AH1 fill:#9b59b6,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

> H2 removed (was redundant) ✅

### 🔧 Strategy 2: Perturb Weights to Break Co-Adaptation

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BH2b["🪞 H2<br/>weights: [0.5, 0.3]<br/><i>copying H1</i>"]
    end
    subgraph After["✅ After"]
        AH2b["🧠 H2<br/>weights: [0.30, 0.18]<br/><i>×0.6 perturbation</i>"]
    end
    style BH2b fill:#e74c3c,stroke:#333,color:#fff
    style AH2b fill:#2ecc71,stroke:#333,color:#fff
```

> Now learns independently! 🔓

| Strategy | Candidate | Operation | Detail |
|----------|-----------|-----------|--------|
| **Remove redundant** | Remove neuron | `removeNeuron` | Remove the lower-activation neuron |
| **Break co-adaptation** | Perturb weights | `setWeight` | Scale incoming weights by 0.6 |

---

## 📝 Example

> A creature has 15 hidden neurons. Discovery finds:
>
> | Neuron | Mean Activation |
> |--------|----------------|
> | H3 | 0.45 |
> | H11 | 0.42 |
>
> Pearson correlation = **0.96** — these neurons respond almost identically to all inputs.
>
> **Candidates:**
> 1. Remove H11 (lower mean activation) → saves computation, reduces complexity cost
> 2. Scale H11's incoming weights by 0.6 → breaks the correlation, allowing H11 to specialise
>
> Either fix frees capacity for the network to represent a new independent feature. ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/co_adaptation.rs`](../../src/analysis/detection/co_adaptation.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Symmetry Breaking](symmetry-breaking.md) — detects neurons
  with near-identical weight configurations (structural similarity)
- **Related**: [Redundant Path](redundant-path.md) — detects correlated
  activation patterns feeding the same target
