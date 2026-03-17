# 🚪 Sentinel Value Gating

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/sentinel_gating.rs`](../../src/analysis/detection/sentinel_gating.rs) | **Issue:** [#400](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/400)

---

## 🔍 The Problem

**Sentinel values** are special marker values in input data (typically −1, 0,
or +1) that indicate "missing", "not applicable", or a boundary condition.
When an input neuron's sentinel cluster has lower error variance than the
useful data range, it confirms the sentinel carries no meaningful signal — the
network should learn to **gate it out** rather than treating it as real data.

```mermaid
graph TD
    subgraph InputData["📊 Input Neuron I2"]
        Sentinel["🚫 Sentinel cluster: −1<br/><i>50% of samples</i><br/>Error variance: 0.002 (low)"]
        Useful["✅ Useful range: [0.3, 0.7]<br/>Error variance: 0.045 (higher)"]
        Gap["📏 Gap: 1.3"]
    end
    style Sentinel fill:#e74c3c,stroke:#333,color:#fff
    style Useful fill:#2ecc71,stroke:#333,color:#fff
    style Gap fill:#fff3e0,stroke:#f57c00,color:#000
```

> 🚪 Sentinel is confirmed noise (low error variance); useful range has real
> information (higher error variance).

### ⚠️ Why It Hurts the Creature's Score

- **Signal pollution**: Sentinel values are treated as real data points,
  confusing the neuron's contribution to downstream computations.
- **Averaged weights**: The network learns weights that compromise between
  handling sentinels and handling real values, doing neither optimally.
- **False correlations**: Sentinel values can create spurious statistical
  patterns that mislead other discovery modules.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each input neuron"] --> B["📊 Collect activation samples<br/><i>minimum 20</i>"]
    B --> C["🔎 For each sentinel<br/>candidate (−1, 0, +1)"]
    C --> D{"fraction ≥ 15%?"}
    D -->|No| Z["✅ No sentinel"]
    D -->|Yes| E["📏 Measure gap between<br/>sentinel and useful range"]
    E --> F{"gap sufficient?"}
    F -->|No| Z
    F -->|Yes| G["📊 Error-correlation check"]
    G --> H["📐 Compute error variance<br/>for sentinel samples"]
    H --> I["📐 Compute error variance<br/>for useful-range samples"]
    I --> J{"sentinel error var<br/>< useful error var?"}
    J -->|No| Z2["❌ Sentinel carries signal"]
    J -->|Yes| K["🚪 Sentinel gating<br/>candidate confirmed!"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#e3f2fd,stroke:#1565c0,color:#000
    style H fill:#e3f2fd,stroke:#1565c0,color:#000
    style I fill:#e3f2fd,stroke:#1565c0,color:#000
    style J fill:#fff3e0,stroke:#f57c00,color:#000
    style K fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 🔑 Key Difference from Bounded Range

[Bounded range detection](bounded-range.md) works on both input and hidden
neurons but does not verify error correlation. Sentinel gating is restricted
to input neurons and adds the error-variance check, providing higher
confidence that the sentinel truly carries no signal.

---

## 🛠️ How We Fix It

The fix adds a **STEP gating neuron** that produces binary output — 0 for
sentinel values, 1 for useful values — and wires it to all downstream
targets of the input:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BI2["🔵 I2<br/>[−1, 0.5]<br/><i>mixed</i>"] --> BH1["🧠 H1"]
        BI2 --> BH3["🧠 H3"]
    end
    subgraph After["✅ After"]
        AI2["🔵 I2<br/>[−1, 0.5]"] --> AGate["🚪 Gate<br/>STEP"]
        AI2 --> AH1["🧠 H1"]
        AI2 --> AH3["🧠 H3"]
        AGate -->|"gated"| AH1
        AGate -->|"gated"| AH3
    end
    style BI2 fill:#e74c3c,stroke:#333,color:#fff
    style BH1 fill:#9b59b6,stroke:#333,color:#fff
    style BH3 fill:#9b59b6,stroke:#333,color:#fff
    style AI2 fill:#4a9eff,stroke:#333,color:#fff
    style AGate fill:#f39c12,stroke:#333,color:#fff
    style AH1 fill:#9b59b6,stroke:#333,color:#fff
    style AH3 fill:#9b59b6,stroke:#333,color:#fff
```

> Gate outputs **0** for sentinel, **1** for useful data. 🚪

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Add gating neuron** | `addNeuron` + `addSynapse` (input → gate) + `addSynapse` (gate → each target) | STEP gate that separates sentinel from useful data |

The gate neuron's synapses to downstream targets preserve the original
synapse weights.

---

## 📝 Example

> Input neuron I7 has 1000 samples:
> - 280 samples (28%) at value = 0.0 (sentinel for "off")
> - 720 samples in range [0.2, 0.9] (useful data)
>
> | Cluster | Error Variance | Interpretation |
> |---------|---------------|----------------|
> | Sentinel (0.0) | 0.003 | Low — no signal ❌ |
> | Useful [0.2, 0.9] | 0.042 | Higher — real signal ✅ |
>
> → Sentinel error variance < useful → confirmed no-signal sentinel
>
> I7 feeds: H2 (w=0.6), H5 (w=−0.4)
>
> **Fix:** Add STEP gate neuron
> - Gate receives I7
> - Gate outputs: 0 when I7 ≈ 0 (sentinel), 1 when I7 in [0.2, 0.9]
> - Gate → H2 (w=0.6), Gate → H5 (w=−0.4)
>
> **Result:** Downstream neurons can now distinguish "input is off"
> from "input has a specific value". 🎯

---

## 📚 References

- **Source module**: [`src/analysis/detection/sentinel_gating.rs`](../../src/analysis/detection/sentinel_gating.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Bounded Range](bounded-range.md) — sentinel detection for
  input and hidden neurons (without error-correlation check)
- **Related**: [Observation Utilisation](observation-utilisation.md) — bias
  compensation for sentinel-dominated inputs
