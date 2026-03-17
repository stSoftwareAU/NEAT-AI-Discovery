# 🚀 Unbounded Capping Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/unbounded_capping.rs`](../../src/analysis/detection/unbounded_capping.rs) | **Issue:** [#441](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/441)

---

## 🔍 The Problem

Neurons with **unbounded activation functions** (RELU, IDENTITY, LEAKYRELU,
etc.) can produce arbitrarily large outputs. When these neurons consistently
"spike" — producing high activations across many training samples — they
inject disproportionate signal magnitudes into the network, overwhelming
downstream neurons and introducing noise.

```mermaid
graph LR
    I1["🔵 I1"] --> H1["🚀 H1<br/>RELU<br/>activations:<br/>12, 45, 8, 67, 23, 91…<br/><i>spiking above 6.0<br/>in 60% of samples</i>"]
    H1 -->|"huge values!"| O1["🎯 O1<br/><i>overwhelmed!</i>"]
    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#e74c3c,stroke:#333,color:#fff
    style O1 fill:#f39c12,stroke:#333,color:#fff
```

> 🚀 **Spiking out of control!** Unbounded activations push massive values into downstream neurons, drowning out other signals.

### ⚠️ Why It Hurts the Creature's Score

- **Downstream saturation**: Large values push receiving neurons into
  saturation, reducing their discrimination ability.
- **Noise amplification**: Extreme activations amplify noise, making
  predictions brittle and variable.
- **Weight scaling issues**: Other synapses' contributions are dwarfed
  by the spiking neuron's oversized signal.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B{"🧪 Unbounded activation?<br/>(RELU, IDENTITY, LEAKYRELU,<br/>SOFTPLUS, ELU, SELU, SWISH,<br/>MISH, GELU, EXPONENTIAL,<br/>SQUARE, CUBE)"}
    B -->|"No"| Z["🛡️ Not applicable"]
    B -->|"Yes"| C["📊 Collect activation samples<br/><i>minimum 20</i>"]
    C --> D{"📏 Max exceeds threshold?<br/>(RELU family: > 6.0<br/>IDENTITY: > 1.0)"}
    D -->|"No"| G["✅ Normal range"]
    D -->|"Yes"| E{"📊 >= 30% above threshold?"}
    E -->|"Yes"| F["🚀 Unbounded capping candidate"]
    E -->|"No"| G
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

The 30% threshold ensures we only flag consistent spiking, not occasional
outliers.

---

## 🛠️ How We Fix It

The fix replaces the unbounded activation with a bounded version that caps
extreme values while preserving the function's character in the normal range:

```mermaid
graph LR
    subgraph Before["❌ Before — RELU, unbounded"]
        BI1["🔵 I1"] --> BH1["🚀 H1<br/>RELU<br/>→ 91!"]
        BH1 --> BO1["🎯 O1"]
    end
    subgraph After["✅ After — RELU6, capped at 6"]
        AI1["🔵 I1"] --> AH1["✨ H1<br/>RELU6<br/>→ 6 max"]
        AH1 --> AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

| Current Squash | Recommended | Rationale |
|---------------|-------------|-----------|
| RELU, LEAKYRELU, ELU, SELU, GELU, SWISH, MISH, SOFTPLUS | RELU6 | Caps at 6.0, preserves zero-threshold behaviour |
| IDENTITY (positive mean) | RELU6 | Caps positive spikes |
| IDENTITY (negative mean) | HARD_TANH | Caps both directions at ±1 |
| EXPONENTIAL | SOFTPLUS | Smooth, bounded-growth alternative |
| SQUARE, CUBE | RELU6 | Prevents polynomial explosion |

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Cap activation** | `changeSquash` | Switch to bounded version |

---

## 📝 Example

> A creature has 40 hidden neurons. Discovery finds:
>
> **Neuron H22:** RELU
>
> | Metric | Value |
> |--------|-------|
> | Max activation | 91.3 |
> | Fraction above 6.0 | 62% |
>
> → Consistently spiking, overwhelming downstream neurons
>
> **Fix:** Change RELU → RELU6
> → Activations capped at 6.0, downstream neurons receive manageable signal magnitudes,
> network stability improves ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/unbounded_capping.rs`](../../src/analysis/detection/unbounded_capping.rs)
- **DISCOVERY_TYPES.md**: [Unbounded Capping Detection](../DISCOVERY_TYPES.md#unbounded-capping-detection)
- **Related**: [Activation Mismatch](activation-mismatch.md) — detects
  structural mismatch between activation and data
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects neurons
  stuck at activation bounds (the opposite problem)
