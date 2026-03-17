# 📈 Activation-Error Monotonicity Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/monotonicity.rs`](../../src/analysis/detection/monotonicity.rs) | **Issue:** [#643](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/643)

---

## 🔍 The Problem

**Non-monotonic activation-error relationships** occur when a hidden neuron's
activation is inconsistently related to output error — higher activation sometimes
correlates with lower error and sometimes with higher error. This contradictory
behaviour suggests the neuron is trying to encode two or more features and should
be split or restructured.

> [!WARNING]
> 🚨 A non-monotonic neuron sends **contradictory signals** to downstream neurons — it cannot consistently reduce error by adjusting its activation in any single direction.

```mermaid
---
title: "U-Shaped Activation vs Error (Non-Monotonic Neuron)"
---
xychart-beta
    x-axis "Activation" [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y-axis "Error" 0.0 --> 1.0
    line [0.8, 0.7, 0.4, 0.2, 0.12, 0.1, 0.12, 0.3, 0.5, 0.7, 0.85]
```

> [!NOTE]
> 📊 The **U-shaped curve** shows error is high at both low and high activation — the neuron encodes contradictory information and should be split.

### ⚠️ Why It Hurts the Creature's Score

- The neuron cannot consistently reduce error by increasing or decreasing activation
- It encodes multiple features that interfere with each other
- Downstream neurons receive contradictory signals from the same source
- The creature's learning process cannot find a stable weight configuration

### 🔀 Difference from `noise_signal.rs`

The noise-to-signal detector measures **variance ratio** — whether error variance
dominates activation variance. A neuron can have high noise-to-signal ratio but
still be monotonic (e.g., noisy but consistently trending). Conversely, a neuron
can have low noise but be non-monotonic (e.g., a clean U-shaped curve).

> [!TIP]
> 🎯 The monotonicity detector specifically measures **directional consistency** using Spearman's rank correlation, whereas the noise-to-signal detector measures **variance ratio**.

### 🔀 Difference from `gradient_discovery.rs`

Gradient discovery analyses existing synapse gradients to suggest weight adjustments.
It operates on synapse-level data. The monotonicity detector analyses the
neuron-level activation-error relationship independent of specific synapses.

---

## 🔬 Detection Method

```mermaid
flowchart TD
    A["📥 Collect activation-error pairs\nfor each hidden neuron"]
    B["📊 Compute Spearman's rank\ncorrelation (rho)"]
    C{"🤔 |rho| < 0.3?"}
    D["✅ Monotonic\nNo action needed"]
    E["⚠️ Non-monotonic\nFlag neuron"]
    F["📐 Estimate improvement\nbased on non-monotonicity degree"]

    A --> B --> C
    C -- "No" --> D
    C -- "Yes" --> E --> F

    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D fill:#2ecc71,stroke:#333,color:#fff
    style E fill:#e74c3c,stroke:#333,color:#fff
    style F fill:#e3f2fd,stroke:#1565c0,color:#000
```

1. For each hidden neuron, collect all (activation, error) pairs from recorded samples
2. Compute **Spearman's rank correlation** (rho) between activation and absolute error
3. If |rho| < 0.3, the relationship is considered non-monotonic
4. Estimate improvement based on the degree of non-monotonicity and sample count

### 📊 Spearman's Rank Correlation

Spearman's rho measures the monotonicity of the relationship between two variables:

| rho value | Interpretation |
|-----------|----------------|
| +1.0 | Perfectly monotonically increasing |
| -1.0 | Perfectly monotonically decreasing |
| ~0.0 | No monotonic relationship |
| < 0.3 | Weak/non-monotonic (flagged) |

---

## 💡 Recommended Actions

```mermaid
flowchart TD
    A["⚠️ Non-monotonic neuron\ndetected"]
    B{"🔗 Has incoming AND\noutgoing synapses?"}
    C["🟣 addNeuron\n(coordinated split)"]
    D["🔄 changeSquash\n(try different activation)"]

    A --> B
    B -- "Yes" --> C
    B -- "No" --> D

    style A fill:#e74c3c,stroke:#333,color:#fff
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#9b59b6,stroke:#333,color:#fff
    style D fill:#9b59b6,stroke:#333,color:#fff
```

| Candidate Type | When Produced | Rationale |
|---------------|---------------|-----------|
| `addNeuron` (coordinated) | Neuron has incoming and outgoing synapses | Split the neuron into two, each handling one direction of the activation-error mapping |
| `changeSquash` | Neuron lacks full connectivity | Try a different activation function that may better fit the data |

---

## 📝 Example Scenario

> **Network:** `input-1` → `hidden-1` → `output-1`
>
> **hidden-1 activation-error relationship:**
>
> | Activation | Error | Level |
> |-----------|-------|-------|
> | 0.1 | 0.7 | 🔴 High |
> | 0.3 | 0.2 | 🟢 Low |
> | 0.5 | 0.1 | 🟢 Low |
> | 0.7 | 0.3 | 🟡 Medium |
> | 0.9 | 0.8 | 🔴 High |
>
> **Spearman's rho ≈ 0.1** (non-monotonic)
>
> ➡️ **Recommendation:** Add a parallel neuron (`split-hidden-1`) to handle the high-activation region separately.

```mermaid
flowchart LR
    subgraph Before
        I1["🔵 input-1"]
        H1["🔴 hidden-1\n(non-monotonic)"]
        O1["🟢 output-1"]
        I1 --> H1 --> O1
    end

    subgraph After["After (recommended)"]
        I2["🔵 input-1"]
        H2["🟢 hidden-1\n(low activations)"]
        H3["🟣 split-hidden-1\n(high activations)"]
        O2["🟢 output-1"]
        I2 --> H2 --> O2
        I2 --> H3 --> O2
    end

    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#e74c3c,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
    style I2 fill:#4a9eff,stroke:#333,color:#fff
    style H2 fill:#2ecc71,stroke:#333,color:#fff
    style H3 fill:#9b59b6,stroke:#333,color:#fff
    style O2 fill:#2ecc71,stroke:#333,color:#fff
```

> [!TIP]
> 🧬 Splitting a non-monotonic neuron allows each resulting neuron to specialise in one region of the activation space, producing consistent error reduction.

---

## ⚙️ Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MONOTONICITY_THRESHOLD` | 0.3 | Minimum \|rho\| to consider a relationship monotonic |
| `MIN_SAMPLES_FOR_DETECTION` | 20 | Minimum samples for reliable rank correlation |

> [!IMPORTANT]
> 📏 At least **20 samples** are required for reliable Spearman's rank correlation. Below this threshold, the detector will not flag neurons to avoid false positives.
