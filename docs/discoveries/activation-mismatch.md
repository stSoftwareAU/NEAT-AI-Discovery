# ⚡ Activation Mismatch Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/activation_mismatch.rs`](../../src/analysis/detection/activation_mismatch.rs) | **Issue:** [#543](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/543)

---

## 🔍 The Problem

An **activation mismatch** occurs when a neuron's activation function is structurally
incompatible with the data flowing through it, causing it to waste information
silently. Two common variants exist:

1. **RELU with negative bias**: A RELU neuron whose pre-activation values are
   predominantly negative gets clipped to zero — the neuron is "alive" but
   discarding most of its input signal.
2. **Bounded underutilisation**: A bounded activation (TANH, LOGISTIC, etc.)
   where the neuron only exercises a tiny fraction of the available output range,
   squashing useful variation into a narrow band.

```mermaid
graph TD
    subgraph V1["⚡ Variant 1: RELU Negative Bias"]
        I1["🔵 Input<br/>−3.2, −1.8, −2.5, −0.9…"] --> H1["⚡ H1<br/>RELU<br/>clips all to 0<br/><i>70%+ signal lost!</i>"]
    end
    subgraph V2["📏 Variant 2: Bounded Underutilisation"]
        I2["🔵 Input"] --> H2["📏 H2<br/>TANH range: [−1, +1]<br/>observed: [+0.12, +0.18]<br/><i>only 3% of range!</i>"]
    end
    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#e74c3c,stroke:#333,color:#fff
    style I2 fill:#4a9eff,stroke:#333,color:#fff
    style H2 fill:#e74c3c,stroke:#333,color:#fff
```

### ⚠️ Why It Hurts the Creature's Score

- **Signal destruction (RELU)**: When most pre-activation values are negative,
  RELU discards the majority of the input signal, turning a functional neuron
  into a near-dead one.
- **Wasted capacity (bounded)**: Using only a sliver of a bounded function's
  range means the neuron cannot distinguish between inputs that differ only
  slightly — fine-grained information is lost.
- **Downstream starvation**: Neurons receiving the mismatched neuron's output
  see a near-constant signal, limiting their ability to learn useful patterns.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B{"🧪 Activation type?"}
    B -->|"RELU / RELU6"| C["📊 Collect pre-activation samples<br/><i>minimum 20</i>"]
    C --> D{"📏 Negative fraction >= 70%?"}
    D -->|"Yes"| E["⚡ RELU negative bias mismatch"]
    D -->|"No"| Z["✅ No mismatch"]
    B -->|"Bounded<br/>(TANH, LOGISTIC, etc.)"| F["📐 Compute observed range<br/>÷ theoretical range"]
    F --> G{"📏 Utilisation < 15%?"}
    G -->|"Yes"| H["📏 Bounded underutilisation"]
    G -->|"No"| Z
    B -->|"Unbounded<br/>(IDENTITY, ELU, etc.)"| Z2["🛡️ Skipped"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#fce4ec,stroke:#c62828,color:#000
    style F fill:#e3f2fd,stroke:#1565c0,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 🛡️ Skipped Activations

Activations that inherently handle wide input ranges are excluded from this
check: IDENTITY, ELU, SELU, LEAKYRELU, GELU, MISH, SOFTPLUS, BENTIDENTITY.

---

## 🛠️ How We Fix It

```mermaid
graph LR
    subgraph Before1["❌ RELU + negative bias"]
        BI1["🔵 I1"] --> BH1["⚡ H1<br/>RELU<br/>70% → 0"]
        BH1 --> BO1["🎯 O1"]
    end
    subgraph After1["✅ ELU preserves negatives"]
        AI1["🔵 I1"] --> AH1["✨ H1<br/>ELU<br/>signal flows!"]
        AH1 --> AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

```mermaid
graph LR
    subgraph Before2["❌ Bounded, 3% range"]
        BI2["🔵 I1"] --> BH2["📏 H1<br/>TANH<br/>[.12, .18]"]
        BH2 --> BO2["🎯 O1"]
    end
    subgraph After2["✅ IDENTITY, full range"]
        AI2["🔵 I1"] --> AH2["✨ H1<br/>IDENTITY<br/>full range"]
        AH2 --> AO2["🎯 O1"]
    end
    style BI2 fill:#4a9eff,stroke:#333,color:#fff
    style BH2 fill:#e74c3c,stroke:#333,color:#fff
    style BO2 fill:#2ecc71,stroke:#333,color:#fff
    style AI2 fill:#4a9eff,stroke:#333,color:#fff
    style AH2 fill:#2ecc71,stroke:#333,color:#fff
    style AO2 fill:#2ecc71,stroke:#333,color:#fff
```

| Variant | Candidate | Operation | Detail |
|---------|-----------|-----------|--------|
| **RELU negative bias** | Change squash | `changeSquash` | RELU → ELU (preserves negative values) |
| **Bounded underutilisation** | Change squash | `changeSquash` | Bounded → IDENTITY (removes range constraint) |

---

## 📝 Example

> A creature has 30 hidden neurons. Discovery finds:
>
> **Neuron H8:** RELU, 82% of pre-activation values are negative
> → Only 18% of input signal passes through
> **Fix:** Change to ELU to preserve negative information
>
> **Neuron H21:** TANH, observed activation range [+0.05, +0.11]
> → Using only 3% of [−1, +1] range
> **Fix:** Change to IDENTITY to use full range
>
> **Result:** Both neurons now pass meaningful signal variation
> to downstream neurons, enabling better learning ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/activation_mismatch.rs`](../../src/analysis/detection/activation_mismatch.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Dying RELU problem** —
  [Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Dying_ReLU_problem):
  A well-known variant where RELU neurons become permanently inactive.
