# ⚖️ Squash + Weight Rescale

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/squash_weight_rescale.rs`](../../src/analysis/detection/squash_weight_rescale.rs) | **Issue:** [#548](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/548)

---

## 🔍 The Problem

When a neuron needs a different activation function, simply swapping the
squash can be destructive — the new function may interpret the same
pre-activation values completely differently, breaking downstream
expectations. A **coordinated squash change with weight rescaling** finds
the optimal rescale factor so the new activation best approximates the old
one's output, then bundles both changes atomically.

```mermaid
graph LR
    subgraph Bare["❌ Bare Swap — Breaks Output"]
        B1["TANH(−1.5) = −0.91"] -->|"swap without rescale"| B2["IDENTITY(−1.5) = −1.50<br/><i>65% output change!</i>"]
    end
    style B1 fill:#e74c3c,stroke:#333,color:#fff
    style B2 fill:#e74c3c,stroke:#333,color:#fff
```

> ⚖️ **Bare squash swap = breakage!** Downstream neurons expect ≈ −0.91 but now receive −1.50. Coordinated rescaling keeps the output stable.

### ⚠️ Why It Hurts Without Rescaling

- **Output discontinuity**: A bare squash swap changes the neuron's output
  distribution instantly, invalidating all downstream weight calibrations.
- **Cascade disruption**: Every neuron downstream of the changed neuron
  receives unexpected input magnitudes.
- **Wasted mutation**: The squash change might be beneficial, but the
  disruption masks the improvement — NEAT-AI's ablation test fails.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron"] --> B{"🧪 Non-aggregate squash?<br/>At least one synapse?"}
    B -->|"No"| Z["🛡️ Skip"]
    B -->|"Yes"| C["📊 Collect pre-activation samples<br/><i>minimum 20</i>"]
    C --> D{"📏 Mean |error| >= 0.03?"}
    D -->|"No"| Z
    D -->|"Yes"| E["🔄 For each of 10 candidate activations<br/>(TANH, LOGISTIC, IDENTITY, SOFTSIGN,<br/>HARD_TANH, RELU, SELU, MISH, SWISH, ELU)"]
    E --> F["📐 Grid search rescale factors<br/>0.25 to 4.0<br/>minimise Σ|new_squash(f×value) − old_squash(value)|"]
    F --> G{"✅ Positive improvement?<br/>Rescale factor <= 5.0?"}
    G -->|"Yes"| H["⚖️ Candidate emitted"]
    G -->|"No"| Z
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#e3f2fd,stroke:#1565c0,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 🚀 Why Coordinated Is Better

The improvement estimate for a coordinated squash+rescale candidate is
boosted by **1.5×** compared to a standalone squash change, because the weight
adjustment preserves output compatibility.

---

## 🛠️ How We Fix It

```mermaid
graph TD
    subgraph Before["❌ Before — TANH, weights [0.5, 0.3]"]
        I1B["🔵 I1"] -->|"w=0.5"| H1B["⚖️ H1<br/>TANH<br/>bias=0.2"]
        I2B["🔵 I2"] -->|"w=0.3"| H1B
        H1B --> O1B["🎯 O1"]
    end
    subgraph After["✅ After — IDENTITY, weights × 0.6"]
        I1A["🔵 I1"] -->|"w=0.30"| H1A["✨ H1<br/>IDENTITY<br/>bias=0.2"]
        I2A["🔵 I2"] -->|"w=0.18"| H1A
        H1A --> O1A["🎯 O1"]
    end
    style I1B fill:#4a9eff,stroke:#333,color:#fff
    style I2B fill:#4a9eff,stroke:#333,color:#fff
    style H1B fill:#e74c3c,stroke:#333,color:#fff
    style O1B fill:#2ecc71,stroke:#333,color:#fff
    style I1A fill:#4a9eff,stroke:#333,color:#fff
    style I2A fill:#4a9eff,stroke:#333,color:#fff
    style H1A fill:#2ecc71,stroke:#333,color:#fff
    style O1A fill:#2ecc71,stroke:#333,color:#fff
```

> Output stays approximately the same, but now IDENTITY provides unbounded range for learning.

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Squash + rescale** | `changeSquash` + `setWeight` (per synapse) | Atomic group: change activation and rescale all incoming weights |

The weight rescaling is only applied when the rescale factor deviates from
1.0 by at least 0.05 (to avoid trivial adjustments).

---

## 📝 Example

> **Neuron H9:** LOGISTIC, bias=1.2, mean |error|=0.08
> Incoming synapses: 3 (weights: 0.8, −0.4, 0.6)
>
> **Grid search results:**
>
> | Candidate | Rescale Factor | Error Reduction |
> |-----------|---------------|----------------|
> | TANH | 0.45 | 0.012 |
> | **IDENTITY** | **0.35** | **0.018** ← best |
> | RELU | 0.50 | 0.009 |
>
> **Candidate:** Change to IDENTITY, rescale weights by 0.35
> New weights: 0.28, −0.14, 0.21
> Estimated improvement: 0.018 × 1.5 = **0.027** (coordinated boost)
>
> The new IDENTITY activation provides the same approximate output
> as LOGISTIC on the observed data, but with room to grow beyond
> the [0, 1] bound ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/squash_weight_rescale.rs`](../../src/analysis/detection/squash_weight_rescale.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Activation Recommendation](activation-recommendation.md) —
  proactive squash suggestions based on input distribution
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects neurons
  stuck at activation bounds
