# 📡 Noise-to-Signal Ratio Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/noise_signal.rs`](../../src/analysis/detection/noise_signal.rs) | **Issue:** [#434](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/434)

---

## 🔍 The Problem

Part of the "Brilliant but Brittle" initiative (Issue #432), this module
identifies neurons and synapses with high **noise-to-signal ratios** that
make predictions fragile. A noisy neuron produces unpredictable error
contributions despite having low activation variance — it adds randomness
without information. A noisy synapse amplifies upstream variance without
contributing to error reduction.

### 📡 Noisy Neuron

| Sample | Activation (signal) | Error (noise) |
|--------|-------------------|---------------|
| 1 | 0.50 | 0.3 |
| 2 | 0.51 | −0.8 |
| 3 | 0.49 | 0.5 |
| 4 | 0.50 | −0.2 |
| 5 | 0.52 | 0.9 |

> Activation variance: **0.0001** | Error variance: **0.35**
> Noise-to-signal ratio: 0.35 / 0.0001 = **3500** (>> 2.0 threshold)
> → Neuron contributes randomness, not information! 📡

### 🔊 Noisy Synapse

> Source neuron: high variance activation, poor error correlation
> Weight: **0.8** (large — amplifies the noise)
> Noise contribution >> 2× signal contribution
> → Synapse is a noise amplifier! 🔊

### ⚠️ Why It Hurts the Creature's Score

- **Brittle predictions**: High noise-to-signal neurons cause unpredictable
  output swings on noisy or missing inputs.
- **Noise amplification**: Large-weight synapses connected to noisy sources
  propagate and amplify randomness through the network.
- **Wasted resources**: Noisy components consume computation without
  improving predictions.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    subgraph NeuronDetection["📡 Noisy Neuron Detection"]
        NA["🔍 For each hidden neuron"] --> NB["📊 Collect activation + error samples<br/><i>minimum 20</i>"]
        NB --> NC["📐 Compute activation variance<br/><i>(signal)</i>"]
        NC --> NERR["📐 Compute error variance<br/><i>(noise)</i>"]
        NERR --> NE{"activation variance < 1e-8?"}
        NE -->|Yes| NF["⏭️ Skip — constant neuron"]
        NE -->|No| NG["📊 ratio = error_var / activation_var"]
        NG --> NH{"ratio > 2.0?"}
        NH -->|Yes| NI["📡 Noisy neuron detected!"]
        NH -->|No| NJ["✅ Neuron is fine"]
    end
    style NA fill:#e3f2fd,stroke:#1565c0,color:#000
    style NB fill:#e3f2fd,stroke:#1565c0,color:#000
    style NC fill:#e3f2fd,stroke:#1565c0,color:#000
    style NERR fill:#e3f2fd,stroke:#1565c0,color:#000
    style NE fill:#fff3e0,stroke:#f57c00,color:#000
    style NF fill:#e8f5e9,stroke:#2e7d32,color:#000
    style NG fill:#e3f2fd,stroke:#1565c0,color:#000
    style NH fill:#fff3e0,stroke:#f57c00,color:#000
    style NI fill:#fce4ec,stroke:#c62828,color:#000
    style NJ fill:#e8f5e9,stroke:#2e7d32,color:#000
```

```mermaid
flowchart TD
    subgraph SynapseDetection["🔊 Noisy Synapse Detection"]
        SA["🔍 For each synapse"] --> SB{"weight ≥ 0.1?"}
        SB -->|No| SC["⏭️ Skip — weight too small"]
        SB -->|Yes| SD["📐 Noise = |weight| × √(source_variance)"]
        SD --> SE["📐 Signal = max(−cov(act, error) × |weight|, 0)"]
        SE --> SF{"noise > 2× signal?"}
        SF -->|Yes| SG["🔊 Noisy synapse detected!"]
        SF -->|No| SH["✅ Synapse is fine"]
    end
    style SA fill:#e3f2fd,stroke:#1565c0,color:#000
    style SB fill:#fff3e0,stroke:#f57c00,color:#000
    style SC fill:#e8f5e9,stroke:#2e7d32,color:#000
    style SD fill:#e3f2fd,stroke:#1565c0,color:#000
    style SE fill:#e3f2fd,stroke:#1565c0,color:#000
    style SF fill:#fff3e0,stroke:#f57c00,color:#000
    style SG fill:#fce4ec,stroke:#c62828,color:#000
    style SH fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### ⚙️ Environment Variable

- `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD`: Configure the neuron
  noise-to-signal threshold (default: 2.0).

---

## 🛠️ How We Fix It

### 📡 Noisy Neuron Fix

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BI1["🔵 I1"] --> BH1["📡 H1<br/>NOISY<br/>N/S = 50"]
        BH1 --> BO1["🎯 O1"]
    end
    subgraph After["✅ After"]
        AI1["🔵 I1"]
        AO1["🎯 O1"]
    end
    style BI1 fill:#4a9eff,stroke:#333,color:#fff
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style BO1 fill:#2ecc71,stroke:#333,color:#fff
    style AI1 fill:#4a9eff,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

> H1 removed (noise source) ✅

### 🔊 Noisy Synapse Fix

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BH2["🧠 H2"] -->|"w = 0.8 🔊"| BH5["🧠 H5<br/><i>noise amplified</i>"]
    end
    subgraph After["✅ After"]
        AH2["🧠 H2"] -->|"w = 0.4 🔇"| AH5["🧠 H5<br/><i>dampened</i>"]
    end
    style BH2 fill:#9b59b6,stroke:#333,color:#fff
    style BH5 fill:#e74c3c,stroke:#333,color:#fff
    style AH2 fill:#9b59b6,stroke:#333,color:#fff
    style AH5 fill:#2ecc71,stroke:#333,color:#fff
```

> Weight halved to dampen noise (or removed entirely if no signal). ✅

| Target | Candidate | Operation | Detail |
|--------|-----------|-----------|--------|
| **Noisy neuron** | Remove neuron | `removeNeuron` | Eliminate the noise source |
| **Noisy synapse (no signal)** | Remove synapse | `removeSynapse` | When signal contribution ≤ 0.01 |
| **Noisy synapse (some signal)** | Reduce weight | `setWeight` | Halve the weight to dampen noise |

---

## 📝 Example

> **Noisy Neuron:**
> Neuron H6: activation variance = 0.0003, error variance = 0.12
> Noise-to-signal ratio: **400** (>> 2.0)
> **Fix:** Remove H6
>
> **Noisy Synapse:**
> Synapse H3 → H9, weight = 0.6
> Source H3 activation variance = 0.8
> Noise contribution: 0.6 × √0.8 = **0.537**
> Signal contribution: **0.05**
> Noise/signal: 10.7× (>> 2×)
> Signal > 0.01, so some useful information
> **Fix:** Set weight to 0.3 (halved to dampen noise while preserving signal) ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/noise_signal.rs`](../../src/analysis/detection/noise_signal.rs)
- **DISCOVERY_TYPES.md**: [Noise-to-Signal Ratio Detection](../DISCOVERY_TYPES.md#noise-to-signal-ratio-detection)
- **Part of**: "Brilliant but Brittle" initiative (Issue #432)
- **Related**: [Input Sensitivity](input-sensitivity.md) — detects dominant
  inputs and threshold effects
- **Related**: [Weight Coherence](weight-coherence.md) — detects incoherent
  weight configurations
