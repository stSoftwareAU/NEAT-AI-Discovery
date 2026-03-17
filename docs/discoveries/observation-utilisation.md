# 📊 Observation Utilisation

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/observation_utilisation.rs`](../../src/analysis/detection/observation_utilisation.rs) | **Issue:** [#543](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/543)

---

## 🔍 The Problem

Input neurons often receive data where a large fraction of samples are
**sentinel values** (e.g., −1 for "missing data", 0 for "not applicable").
These sentinels reduce the effective utilisation of the input range, and
downstream neurons cannot distinguish between "the value is zero because
the data is missing" and "the value is zero because the measurement is zero".

This module builds on the observation range characterisation (Issue #398)
to recommend bias adjustments on downstream neurons that compensate for the
sentinel-induced offset.

```mermaid
graph LR
    subgraph InputData["📊 Input Neuron I3"]
        I3["🔵 I3<br/>50% sentinel (−1)<br/>Useful range: [0.3, 0.7]<br/>Utilisation: 40%"]
    end
    I3 -->|"w = 0.8"| H1["🧠 H1"]
    style I3 fill:#e74c3c,stroke:#333,color:#fff
    style H1 fill:#9b59b6,stroke:#333,color:#fff
```

> 📊 When I3 = −1 (sentinel): contribution = **−0.8**
> When I3 = 0.5 (useful): contribution = **+0.4**
> Average contribution is **biased by the sentinel!**

### ⚠️ Why It Hurts the Creature's Score

- **Biased downstream activation**: Sentinel values shift the mean input
  to downstream neurons, biasing their operating point.
- **Underused input range**: Less than 80% of the input's range carries
  useful information.
- **Confounded learning**: Downstream weights learn a compromise between
  handling sentinel values and real data, doing neither well.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each input neuron"] --> B["📊 Run observation range<br/>characterisation<br/><i>detect sentinel clusters at −1, 0, +1</i>"]
    B --> C["📐 Compute utilisation ratio<br/><i>fraction of range that is useful</i>"]
    C --> D{"utilisation < 80%?"}
    D -->|No| Z["✅ Well-utilised input"]
    D -->|Yes| E["🔍 For each downstream synapse"]
    E --> F{"weight ≥ 1e-8?"}
    F -->|No| Z2["⏭️ Skip — negligible weight"]
    F -->|Yes| G["📐 Bias compensation =<br/>−(effective_centre × weight)"]
    G --> H{"compensation ≥ 1e-6?"}
    H -->|No| Z2
    H -->|Yes| I["📊 Candidate emitted!"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#e3f2fd,stroke:#1565c0,color:#000
    style H fill:#fff3e0,stroke:#f57c00,color:#000
    style I fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

---

## 🛠️ How We Fix It

The fix adjusts the bias of each downstream target neuron to compensate for
the sentinel-induced offset:

```mermaid
graph LR
    subgraph Before["❌ Before — sentinel biases downstream"]
        BI3["🔵 I3<br/>50% sentinel"] -->|"w = 0.8"| BH1["🧠 H1<br/>bias = 0.0"]
    end
    subgraph After["✅ After — bias-compensated"]
        AI3["🔵 I3<br/>50% sentinel"] -->|"w = 0.8"| AH1["🧠 H1<br/>bias = −0.4"]
    end
    style BI3 fill:#e74c3c,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style AI3 fill:#4a9eff,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
```

> Bias compensates for sentinel offset! ✅

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Bias compensation** | `setBias` | Per downstream target: bias += −(effective_centre × weight) |

Estimated improvement: `(1.0 − utilisation_ratio) × 0.005`.

---

## 📝 Example

> Input neuron I5:
> - Utilisation ratio: **55%** (< 80%)
> - Sentinel: 0 detected (35% of samples)
> - Effective centre of useful range: 0.6
>
> Downstream synapses from I5:
>
> | Synapse | Weight | Current Bias | New Bias | Calculation |
> |---------|--------|-------------|----------|-------------|
> | I5 → H2 | 0.5 | 0.1 | **−0.2** | 0.1 + (−(0.6 × 0.5)) = 0.1 − 0.3 |
> | I5 → H7 | −0.3 | 0.0 | **0.18** | 0.0 + (−(0.6 × −0.3)) = 0.0 + 0.18 |
>
> **After fix:** Downstream neurons' operating points are
> compensated for the sentinel-induced offset. 🎯

---

## 📚 References

- **Source module**: [`src/analysis/detection/observation_utilisation.rs`](../../src/analysis/detection/observation_utilisation.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Bounded Range](bounded-range.md) — adds gating neurons to
  suppress sentinel values
- **Related**: [Sentinel Value Gating](sentinel-gating.md) — input-neuron
  sentinel detection with error-correlation checks
