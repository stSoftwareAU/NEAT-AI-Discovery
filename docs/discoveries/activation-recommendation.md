# 🧬 Activation Recommendation

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/recommendation/activation_recommendation.rs`](../../src/analysis/recommendation/activation_recommendation.rs) | **Issue:** [#431](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/431)

---

## 🔍 The Problem

Most activation function changes in NEAT are **reactive** — triggered after
a problem is detected (saturation, oscillation, etc.). But by the time the
problem manifests, the neuron may have already wasted many generations of
evolution operating inefficiently.

**Activation recommendation** takes a **proactive** approach: it analyses the
statistical distribution of each neuron's inputs and recommends the activation
function that best matches the data characteristics — before problems occur.

```mermaid
graph LR
    subgraph Reactive["❌ Reactive (existing)"]
        R1["📥 Input data"] --> R2["🔧 Wrong squash"]
        R2 --> R3["⚠️ Problem develops"]
        R3 --> R4["🔍 Detect"]
        R4 --> R5["🛠️ Fix"]
    end
    subgraph Proactive["✅ Proactive (this module)"]
        P1["📥 Input data"] --> P2["📊 Analyse distribution"]
        P2 --> P3["🧬 Recommend best squash"]
        P3 --> P4["✨ Apply"]
    end
    style R1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style R2 fill:#e74c3c,stroke:#333,color:#fff
    style R3 fill:#f39c12,stroke:#333,color:#fff
    style R4 fill:#fff3e0,stroke:#f57c00,color:#000
    style R5 fill:#e8f5e9,stroke:#2e7d32,color:#000
    style P1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style P2 fill:#e3f2fd,stroke:#1565c0,color:#000
    style P3 fill:#e8f5e9,stroke:#2e7d32,color:#000
    style P4 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

> 🧬 **Proactive catches the mismatch immediately** — no generations wasted waiting for symptoms to appear.

### ✨ Why It Matters

- **Prevents future problems**: A well-matched activation function avoids
  saturation, oscillation, and restricted range issues before they start.
- **Better gradient flow**: Matching the activation to the input distribution
  ensures healthy gradients from the beginning.
- **Higher success rate**: Proactive recommendations target the root cause
  (activation/data mismatch) rather than symptoms.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B["📊 Step 1: Classify input distribution"]
    B --> C{"📈 Distribution type?"}
    C -->|"Sparse: > 50% near zero"| D1["🧬 Best: RELU (0.9), LEAKYRELU (0.85)"]
    C -->|"Bounded: range < 2.0"| D2["🧬 Best: LOGISTIC (0.85), HARD_TANH (0.80)"]
    C -->|"Bimodal: kurtosis < 2.5"| D3["🧬 Needs: specialised handling"]
    C -->|"Gaussian: kurtosis 2–5"| D4["🧬 Best: TANH (0.9), SOFTPLUS (0.85)"]
    C -->|"Uniform"| D5["🧬 Best: TANH (0.80), IDENTITY (0.75)"]
    D1 --> E["⚖️ Step 2: Apply gradient flow penalty"]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    E --> F{"📏 Score improvement >= 0.001?<br/>Different from current?<br/>At least 20 samples?"}
    F -->|"Yes"| G["🧬 Recommendation emitted"]
    F -->|"No"| H["✅ Current activation is adequate"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style D2 fill:#e3f2fd,stroke:#1565c0,color:#000
    style D3 fill:#e3f2fd,stroke:#1565c0,color:#000
    style D4 fill:#e3f2fd,stroke:#1565c0,color:#000
    style D5 fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
    style H fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### ⚖️ Gradient Flow Penalties

- TANH/LOGISTIC penalised 30% if inputs cause saturation
- RELU penalised if many inputs are negative

---

## 🛠️ How We Fix It

```mermaid
graph LR
    subgraph Before["❌ Before — mismatched"]
        BI1["🔵 I1"] --> BH1["⚡ H1<br/>RELU<br/>Gaussian inputs<br/><i>clips negative half!</i>"]
        BI2["🔵 I2"] --> BH1
        BH1 --> BO1["🎯 O1"]
    end
    subgraph After["✅ After — distribution-matched"]
        AI1["🔵 I1"] --> AH1["✨ H1<br/>TANH<br/>handles full<br/>bell curve"]
        AI2["🔵 I2"] --> AH1
        AH1 --> AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BI2 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AI2 fill:#4a9eff,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Change squash** | `changeSquash` | Switch to distribution-matched activation |

Expected improvement: `improvement_delta × 0.02`, where `improvement_delta`
is the suitability score difference between recommended and current activation.

---

## 📝 Example

> **Neuron H8:** current activation = RELU
> Input distribution classified as: **Gaussian**
>
> | Metric | Value |
> |--------|-------|
> | Mean | 0.02 |
> | Std dev | 0.8 |
> | Kurtosis | 2.9 |
> | Skew | 0.1 |
>
> **Suitability scores:**
>
> | Activation | Score | Notes |
> |-----------|-------|-------|
> | **TANH** | **0.90** | ← best for Gaussian |
> | SOFTPLUS | 0.85 | |
> | RELU | 0.60 | current — penalised for clipping negatives |
> | LOGISTIC | 0.55 | |
>
> Improvement: 0.90 − 0.60 = 0.30 (>> 0.001 threshold)
>
> **Candidate:** Change RELU → TANH
> Expected improvement: 0.30 × 0.02 = 0.006
>
> After fix: TANH handles the full bell-curve distribution
> symmetrically, preserving negative inputs that RELU was clipping to zero ✅

---

## 📚 References

- **Source module**: [`src/analysis/recommendation/activation_recommendation.rs`](../../src/analysis/recommendation/activation_recommendation.rs)
- **DISCOVERY_TYPES.md**: [Activation Function Recommendation](../DISCOVERY_TYPES.md#activation-function-recommendation)
- **Related**: [Saturated Neuron](saturated-neuron.md) — reactive fix for
  neurons already stuck at bounds
- **Related**: [Oscillating Neuron](oscillating-neuron.md) — reactive fix
  for sign-oscillating neurons
- **Related**: [Squash + Weight Rescale](squash-weight-rescale.md) —
  coordinated squash change with weight adjustment
