# 🚧 Bounded Range Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/bounded_range.rs`](../../src/analysis/detection/bounded_range.rs) | **Issue:** [#395](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/395)

---

## 🔍 The Problem

Some input and hidden neurons receive data where a significant fraction of
samples sit at a **sentinel boundary value** (typically −1, 0, or +1 meaning
"missing", "off", or "maximum"). These sentinel values carry no useful signal
but are mixed in with genuine data, confusing downstream neurons that cannot
distinguish "missing" from "actually zero".

```mermaid
graph LR
    subgraph Data["📊 Input Neuron Activations"]
        S["🚫 Sentinel<br/>−1.0<br/><i>40% of samples</i>"]
        G["📏 Gap = 1.3"]
        U["✅ Useful range<br/>[0.3, 0.7]"]
    end
    S ~~~ G
    G ~~~ U
    style S fill:#e74c3c,stroke:#333,color:#fff
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style U fill:#2ecc71,stroke:#333,color:#fff
```

> 🚧 Sentinel values (−1) are mixed with useful data ([0.3, 0.7]) —
> downstream neurons can't tell the difference!

### ⚠️ Why It Hurts the Creature's Score

- **Signal contamination**: The sentinel value is treated as a real data
  point, biasing learned weights.
- **Wasted capacity**: The neuron cannot learn separate responses for
  "missing data" vs "data with value near the sentinel".
- **Downstream confusion**: Neurons receiving this signal cannot distinguish
  meaningful values from sentinel markers.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each input and<br/>hidden neuron"] --> B["📊 Collect activation samples<br/><i>minimum 20</i>"]
    B --> C["🔎 For each sentinel<br/>candidate (−1, 0, +1)"]
    C --> D["📐 Count fraction of samples<br/>within tolerance of sentinel"]
    D --> E{"fraction ≥ 20%?"}
    E -->|No| Z["✅ No sentinel found"]
    E -->|Yes| F["📏 Measure gap between<br/>sentinel and nearest<br/>useful value"]
    F --> G{"gap exceeds minimum?"}
    G -->|No| Z
    G -->|Yes| H["🚧 Sentinel confirmed!<br/>Bounded range candidate"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#e3f2fd,stroke:#1565c0,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📊 Confidence Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| **Fraction factor** | 40% | Higher sentinel fraction → higher confidence |
| **Gap factor** | 40% | Wider gap → higher confidence |
| **Sample factor** | 20% | More samples → higher confidence |

> `Confidence = 0.5 + (fraction × 0.4 + gap × 0.4 + samples × 0.2) × 0.5`

---

## 🛠️ How We Fix It

The fix adds a **gating neuron** that learns to suppress the sentinel values
while passing useful data through:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BIn["🔵 Input<br/>[−1, 0.5]<br/><i>sentinel mixed</i>"] --> BH1["🧠 H1"]
    end
    subgraph After["✅ After"]
        AIn["🔵 Input<br/>[−1, 0.5]"] --> AGate["🚪 Gate<br/>ReLU<br/>bias = −centre"]
        AGate -->|"suppresses sentinel!"| AH1["🧠 H1"]
    end
    style BIn fill:#e74c3c,stroke:#333,color:#fff
    style BH1 fill:#9b59b6,stroke:#333,color:#fff
    style AIn fill:#4a9eff,stroke:#333,color:#fff
    style AGate fill:#f39c12,stroke:#333,color:#fff
    style AH1 fill:#9b59b6,stroke:#333,color:#fff
```

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Add gating neuron** | `addNeuron` + `addSynapse` | ReLU gate with bias set to suppress the sentinel value |

The gate neuron's bias is set to `−useful_centre`, causing sentinel values
to produce zero output while useful values pass through.

---

## 📝 Example

> Input neuron I5 has 500 samples:
> - 180 samples (36%) at value = −1.0 (sentinel for "missing")
> - 320 samples in range [0.2, 0.8] (useful data)
> - Gap between sentinel and useful range: 1.2
>
> **Fix:** Add ReLU gate neuron
> - Bias = −(0.2 + 0.8)/2 = **−0.5**
>
> | Input | Calculation | Output |
> |-------|-------------|--------|
> | −1.0 (sentinel) | ReLU(−1.0 − 0.5) = ReLU(−1.5) | **0** (suppressed) ✅ |
> | 0.5 (useful) | ReLU(0.5 − 0.5) = ReLU(0.0) | 0.0 (borderline) |
> | 0.8 (useful) | ReLU(0.8 − 0.5) = ReLU(0.3) | **0.3** (passes through) ✅ |
>
> **Result:** Downstream neurons receive 0 for missing data
> and proportional values for real data. 🎯

---

## 📚 References

- **Source module**: [`src/analysis/detection/bounded_range.rs`](../../src/analysis/detection/bounded_range.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Sentinel Value Gating](sentinel-gating.md) — similar concept
  for input neurons with additional error-correlation checks
- **Related**: [Observation Utilisation](observation-utilisation.md) — flags
  underutilised input ranges dominated by sentinels
