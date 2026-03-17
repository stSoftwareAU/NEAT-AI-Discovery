# 📏 Output Range Compression

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/output_range_compression.rs`](../../src/analysis/detection/output_range_compression.rs) | **Issue:** [#645](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/645)

---

## 🔍 The Problem

An **output range compression** occurs when an output neuron uses a bounded
activation function (e.g., TANH with range [-1, 1]) but its actual activations
cluster in a narrow sub-range (e.g., [0.3, 0.7]). The neuron is using the
correct *type* of activation but operating at reduced dynamic resolution.

> 💡 **Key insight:** The neuron cannot make fine-grained distinctions in the
> output because weight adjustments produce tiny activation changes relative to
> the full range.

```mermaid
graph LR
    subgraph "TANH Output Range: -1.0 to +1.0"
        direction LR
        A["🔵 -1.0"] ~~~ B["Unused\nRegion"]
        B ~~~ C["Actual\nActivations\n0.3 – 0.7"]
        C ~~~ D["Unused\nRegion"]
        D ~~~ E["🔵 +1.0"]
    end

    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#f5f5f5,stroke:#ccc,color:#999
    style C fill:#e74c3c,stroke:#333,color:#fff
    style D fill:#f5f5f5,stroke:#ccc,color:#999
    style E fill:#4a9eff,stroke:#333,color:#fff
```

### ⚠️ Why It Hurts the Creature's Score

- **Reduced resolution**: Small weight changes produce negligible activation
  differences within the compressed band, making optimisation sluggish.
- **Wasted capacity**: The activation function's full range could encode much
  more information than the narrow operating band allows.
- **Imprecise predictions**: The network cannot distinguish between target
  values that differ by less than the compressed range's granularity.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    Start(["🔄 For each output neuron"]) --> CheckBounded{{"Bounded squash\nfunction?\n(TANH, LOGISTIC, etc.)"}}

    CheckBounded -->|No| ExcludeUnbounded["⛔ Skip:\nUnbounded activation\n(IDENTITY, RELU, etc.)"]
    CheckBounded -->|Yes| CheckSamples{{"Sufficient\nsamples?\n(≥ 20)"}}

    CheckSamples -->|No| ExcludeInsufficient["⛔ Skip:\nInsufficient samples"]
    CheckSamples -->|Yes| CheckDead{{"Observed range\n> 0.01?"}}

    CheckDead -->|No| ExcludeDead["⛔ Skip:\nDead neuron"]
    CheckDead -->|Yes| CheckSaturated{{"Not near\nactivation bounds?"}}

    CheckSaturated -->|No| ExcludeSaturated["⛔ Skip:\nSaturated neuron"]
    CheckSaturated -->|Yes| CalcUtil["📐 Calculate:\nrange_utilisation =\nobserved_range ÷\ntheoretical_range"]

    CalcUtil --> CheckThreshold{{"range_utilisation\n< 40%?"}}

    CheckThreshold -->|No| Pass["✅ No compression\ndetected"]
    CheckThreshold -->|Yes| Flag["🚩 Flag as\nOutput Range\nCompression"]

    style Start fill:#4a9eff,stroke:#333,color:#fff
    style CheckBounded fill:#fff3e0,stroke:#f57c00,color:#000
    style CheckSamples fill:#fff3e0,stroke:#f57c00,color:#000
    style CheckDead fill:#fff3e0,stroke:#f57c00,color:#000
    style CheckSaturated fill:#fff3e0,stroke:#f57c00,color:#000
    style CheckThreshold fill:#fff3e0,stroke:#f57c00,color:#000
    style CalcUtil fill:#e3f2fd,stroke:#1565c0,color:#000
    style Flag fill:#e74c3c,stroke:#333,color:#fff
    style Pass fill:#2ecc71,stroke:#333,color:#fff
    style ExcludeUnbounded fill:#f5f5f5,stroke:#ccc,color:#999
    style ExcludeInsufficient fill:#f5f5f5,stroke:#ccc,color:#999
    style ExcludeDead fill:#f5f5f5,stroke:#ccc,color:#999
    style ExcludeSaturated fill:#f5f5f5,stroke:#ccc,color:#999
```

> 🔎 **Exclusions:** Hidden neurons are handled by the
> [restricted range](restricted-range.md) module instead.

---

## 🛠️ How We Fix It

```mermaid
graph LR
    subgraph Before ["❌ Before — Compressed"]
        H1B["🔵 H1"] --> O1B["🔴 O1\nTANH\n0.3 – 0.7\n20% utilised"]
        O1B -.-> TgtB["🎯 Target"]
    end

    subgraph After ["✅ After — Rescaled"]
        H1A["🔵 H1"] --> O1A["🟢 O1\nLOGISTIC\n0.1 – 0.9\n80% utilised"]
        O1A -.-> TgtA["🎯 Target"]
    end

    style H1B fill:#4a9eff,stroke:#333,color:#fff
    style O1B fill:#e74c3c,stroke:#333,color:#fff
    style TgtB fill:#f5f5f5,stroke:#333,color:#333
    style H1A fill:#4a9eff,stroke:#333,color:#fff
    style O1A fill:#2ecc71,stroke:#333,color:#fff
    style TgtA fill:#f5f5f5,stroke:#333,color:#333
```

| Strategy | Candidate | Operations | Description |
|----------|-----------|------------|-------------|
| **Change squash** | Better-fitting activation | `changeSquash` | Switch to activation whose range matches target distribution |
| **Rescale pathway** | Coordinated adjustment | `setBias` + `setWeight` | Scale incoming weights to expand operating range and recentre bias |

> 🛠️ **Repair strategy:** The detector prefers `changeSquash` when the target
> distribution clearly suits a different activation function, and falls back to
> coordinated weight/bias rescaling otherwise.

---

## 📝 Example

> **Output neuron O1:** TANH (range [-1, +1]), bias = 0.5
> Observed activations: [0.3, 0.7] across 50 samples
> Range utilisation: 0.4 / 2.0 = **20%**
>
> **Candidate 1 — changeSquash → LOGISTIC**
> LOGISTIC range [0, 1] better matches the positive-only distribution.
> Expected improvement: 0.004
>
> **Candidate 2 — setBias + setWeight (coordinated)**
> Scale incoming weights by 4.0× to expand operating range.
> Adjust bias from 0.5 to 0.0 to recentre in activation range.
> Expected improvement: 0.0028

---

## 🔗 Relationship to Other Modules

| Module | What It Detects | Difference |
|--------|----------------|------------|
| **Restricted range** (Issue #399) | Hidden neurons with compressed range | This module targets *output* neurons |
| **Output squash mismatch** (Issue #546) | Wrong activation function type | That module detects fundamentally wrong functions; this detects correct type but compressed usage |
| **Squash weight rescale** (Issue #548) | Coordinated squash+weight changes | That module is for hidden neurons; this produces similar candidates for outputs |

---

## 📚 References

- **Source module**: [`src/analysis/detection/output_range_compression.rs`](../../src/analysis/detection/output_range_compression.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Restricted Range](restricted-range.md) — hidden neuron version
- **Related**: [Output Squash Mismatch](output-squash-mismatch.md) — wrong function type
- **Related**: [Squash Weight Rescale](squash-weight-rescale.md) — coordinated rescaling for hidden neurons
