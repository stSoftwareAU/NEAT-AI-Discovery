# 🎚️ Input Sensitivity Analysis

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/input_sensitivity.rs`](../../src/analysis/detection/input_sensitivity.rs) | **Issue:** [#435](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/435)

---

## 🔍 The Problem

Input sensitivity analysis detects two related brittleness patterns where
small input changes cause disproportionately large output swings:

### ⚡ 1. Dominant Input Detection

A single input neuron has excessive **leverage** over the network output —
its weight, correlation with error, and variance combine to give it outsized
influence. If this input receives noisy or missing data, the entire
prediction collapses.

```mermaid
graph LR
    I1["🔵 I1"]:::input -- "w = 0.1" --> O1["🟢 O1"]:::output
    I2["🔵 I2"]:::input -- "w = 0.2" --> O1
    I3["🔴 I3"]:::problem -- "w = 2.8" --> O1

    note["⚠️ I3 dominates:<br/>sensitivity = 4.5<br/>threshold: 2.0<br/>One bad I3 value<br/>→ prediction collapse"]:::warn

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef warn fill:#fff3e0,stroke:#f57c00,color:#000
```

### 📉 2. Threshold Effect Detection

An input feeds a hidden neuron through a region of extreme gradient in the
activation function (e.g., near the steep part of TANH or LOGISTIC). Tiny
input changes cause sudden, large output changes — a cliff effect.

```mermaid
graph TD
    subgraph gentle["Gentle Region"]
        G["input: 0.98 → 1.02<br/>output: 0.75 → 0.77<br/>∆ small"]:::process
    end

    subgraph steep["⚠️ Steep Region — Cliff Effect"]
        S["input: −0.02 → 0.02<br/>output: −0.02 → 0.02<br/>∆ large gradient × weight = amplified"]:::problem
    end

    G --> S

    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
```

### ⚠️ Why It Hurts the Creature's Score

> [!WARNING]
> These sensitivity patterns make a creature's predictions unreliable and
> fragile, leading to poor generalisation.

- **Brittle predictions**: Dominant inputs make the model fragile — noise
  or missing values in one input can swing the entire output.
- **Cliff effects**: Threshold regions amplify small input perturbations into
  large output changes, reducing prediction reliability.
- **Overfitting risk**: High sensitivity to one input often means the model
  has memorised training-set-specific patterns.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔵 For each input path"]:::input --> B{"Dominant Input<br/>Detection"}:::decision

    B --> C["1. Compute leverage ratio:<br/>|weight| × |corr(input, error)|<br/>× √(input_var / error_var)"]:::process
    C --> D["2. Compute sensitivity =<br/>leverage_ratio × |weight|"]:::process
    D --> E{"sensitivity > 2.0?"}:::decision
    E -- "Yes" --> F["🔴 Dominant input detected"]:::problem

    A --> G{"Threshold Effect<br/>Detection"}:::decision
    G --> H["1. Compute effective gradient:<br/>max_finite_difference × |weight|"]:::process
    H --> I{"gradient > 10.0?"}:::decision
    I -- "Yes" --> J["🔴 Threshold effect detected"]:::problem

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,color:#000
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
```

### ⚙️ Environment Variables

> [!NOTE]
> These thresholds can be tuned via environment variables to adjust
> detection sensitivity for your specific use case.

- `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD`: Sensitivity threshold for dominant
  input detection (default: 2.0).
- `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD`: Effective gradient threshold for
  threshold effect detection (default: 10.0).

---

## 🛠️ How We Fix It

```mermaid
flowchart LR
    subgraph before["🔴 Before — Dominant Input"]
        I3a["🔵 I3"]:::input -- "w = 2.8<br/>sensitivity = 4.5" --> O1a["🟢 O1"]:::output
    end

    subgraph after["🟢 After — Weight Reduced"]
        I3b["🔵 I3"]:::input -- "w = 1.4<br/>reduced to<br/>threshold × 0.8" --> O1b["🟢 O1"]:::fixed
    end

    before --> after

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef output fill:#e74c3c,stroke:#333,color:#fff
    classDef fixed fill:#2ecc71,stroke:#333,color:#fff
```

```mermaid
flowchart TD
    T["🔴 Threshold Effect Detected"]:::problem --> Opt1 & Opt2 & Opt3

    Opt1["Option 1: Add dampening neuron<br/>I1 → IDENTITY gate → H1<br/>dampens signal"]:::process
    Opt2["Option 2: Shift bias<br/>H1 bias shifted ±0.5<br/>moves away from cliff"]:::process
    Opt3["Option 3: Reduce weight<br/>I1 →(w × 0.3)→ H1<br/>reduces amplification"]:::process

    Opt1 --> R["🟢 Sensitivity Reduced"]:::fixed
    Opt2 --> R
    Opt3 --> R

    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef fixed fill:#2ecc71,stroke:#333,color:#fff
```

> [!TIP]
> The repair system selects the most appropriate fix based on the pattern
> detected and the network topology.

| Pattern | Candidate | Operation | Detail |
|---------|-----------|-----------|--------|
| **Dominant input** | Reduce weight | `setWeight` | Scale to dominance_threshold × 0.8 |
| **Threshold effect** | Add dampening | `addNeuron` | IDENTITY neuron to attenuate signal |
| **Threshold effect** | Shift bias | `setBias` | Move operating point away from cliff |
| **Threshold effect** | Reduce weight | `setWeight` | Scale weight by gradient × 0.3 |

---

## 📝 Example

> **Dominant Input:**
> Input I7 → Output O1, weight = 3.2
> Correlation(I7, O1\_error) = 0.85
> Leverage ratio = 3.2 × 0.85 × 1.8 = 4.9
> Sensitivity = 4.9 × 3.2 = 15.7 (>> 2.0)
> Fix: Set weight to 2.0 × 0.8 = 1.6
>
> **Threshold Effect:**
> Input I2 → Hidden H3 (TANH), weight = 1.5
> Max finite difference in activation = 0.98
> Effective gradient = 0.98 × 1.5 = 14.7 (> 10.0)
> Fix options:
> 1. Add IDENTITY dampening neuron between I2 and H3
> 2. Shift H3 bias by ±0.5 to move away from steep region
> 3. Reduce I2→H3 weight to 14.7 × 0.3 = 4.4

---

## 📚 References

- **Source module**: [`src/analysis/detection/input_sensitivity.rs`](../../src/analysis/detection/input_sensitivity.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Part of**: "Brilliant but Brittle" initiative (Issue #432)
- **Related**: [Noise-to-Signal Ratio](noise-signal.md) — detects noisy
  neurons and synapses
- **Related**: [Weight Coherence](weight-coherence.md) — detects incoherent
  weight configurations
