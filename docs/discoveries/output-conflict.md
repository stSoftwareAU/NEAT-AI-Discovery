# ⚔️ Output Conflict Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/output_conflict.rs`](../../src/analysis/detection/output_conflict.rs) | **Issue:** [#639](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/639)

---

## 🔍 The Problem

A **per-output error conflict** occurs when a hidden neuron has a positive
effect on some outputs but actively harms others. The net error may look
acceptable, but the neuron is creating a cross-output tug-of-war.

```mermaid
graph LR
    H1["🧠 Hidden neuron h1"]
    O0["✅ output-0<br/>error: -0.5<br/>(helping)"]
    O1["❌ output-1<br/>error: +0.3<br/>(harming)"]
    NET["📊 Net effect: -0.2<br/>(looks helpful overall)"]

    H1 -->|"-0.5"| O0
    H1 -->|"+0.3"| O1
    O0 --> NET
    O1 --> NET

    style H1 fill:#4a9eff,stroke:#333,color:#fff
    style O0 fill:#2ecc71,stroke:#333,color:#fff
    style O1 fill:#e74c3c,stroke:#333,color:#fff
    style NET fill:#fff3e0,stroke:#f57c00,color:#000
```

> [!WARNING]
> Although the net effect appears helpful (-0.2), output-1 is being actively harmed. Summed-error metrics hide this cross-output tug-of-war.

### ⚠️ Why It Hurts the Creature's Score

- **Hidden harm**: The net error masks the damage to individual outputs.
  Summed-error metrics make the neuron appear beneficial when it is not.
- **Training interference**: Gradient updates that improve one output's
  contribution through this neuron may worsen another output.
- **Structural limitation**: A single hidden neuron cannot simultaneously
  optimise its contribution to outputs that require opposing effects.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔄 For each hidden neuron"]
    B["📥 Collect errors 0..n<br/>across all observations"]
    C["📊 Compute mean error<br/>per output index"]
    D{"🔎 Sign conflict?<br/>Any mean &lt; -threshold<br/>AND any mean &gt; +threshold"}
    E["📐 Compute conflict severity<br/>= max_positive × |min_negative|"]
    F{"🎚️ Above significance<br/>threshold?"}
    G["🚨 Flag as output conflict"]
    H["✅ No conflict — skip"]
    I["⏭️ Too weak — skip"]

    A --> B
    B --> C
    C --> D
    D -->|Yes| E
    D -->|No| H
    E --> F
    F -->|Yes| G
    F -->|No| I

    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#e74c3c,stroke:#333,color:#fff
    style H fill:#2ecc71,stroke:#333,color:#fff
    style I fill:#2ecc71,stroke:#333,color:#fff
```

> [!NOTE]
> The detector analyses the `errors[0..n]` vector for each hidden neuron, looking for outputs where the mean error contribution has opposing signs.

### 📏 Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Minimum significant error | 0.01 | Ignore noise-level error contributions |
| Minimum samples | 10 | Ensure statistical reliability |
| Minimum outputs | 2 | Conflict requires multiple outputs |

---

## 💡 What We Recommend

### 🔧 Strategy 1: Attenuate the Harmful Connection (SetWeight)

When a direct synapse exists from the conflicting neuron to the harmed
output, reduce its weight to 30% of the current value.

```mermaid
graph LR
    subgraph Before
        B_H1["🧠 h1"] -->|"w = 0.8"| B_O1["❌ output-1<br/>(harmed)"]
    end

    subgraph After
        A_H1["🧠 h1"] -->|"w = 0.24"| A_O1["✅ output-1<br/>(reduced harm)"]
    end

    style B_H1 fill:#4a9eff,stroke:#333,color:#fff
    style B_O1 fill:#e74c3c,stroke:#333,color:#fff
    style A_H1 fill:#4a9eff,stroke:#333,color:#fff
    style A_O1 fill:#2ecc71,stroke:#333,color:#fff
```

> [!TIP]
> Attenuating the weight to 30% preserves the neuron's beneficial contributions to other outputs while reducing the harm to the conflicting output.

### 🧩 Strategy 2: Add a Compensating Gating Neuron (AddNeuron + AddSynapse)

When the path to the harmed output is indirect, insert a compensating
neuron that counteracts the harmful contribution.

```mermaid
graph LR
    subgraph Before
        B_H1["🧠 h1"] --> B_X["..."] --> B_O1["❌ output-1<br/>(harmed)"]
    end

    subgraph After
        A_H1["🧠 h1"] -->|"-w"| GATE["🔀 gate"]
        GATE -->|"1.0"| A_O1["✅ output-1<br/>(compensated)"]
    end

    style B_H1 fill:#4a9eff,stroke:#333,color:#fff
    style B_X fill:#e3f2fd,stroke:#1565c0,color:#000
    style B_O1 fill:#e74c3c,stroke:#333,color:#fff
    style A_H1 fill:#4a9eff,stroke:#333,color:#fff
    style GATE fill:#fff3e0,stroke:#f57c00,color:#000
    style A_O1 fill:#2ecc71,stroke:#333,color:#fff
```

> [!NOTE]
> All recommendations are emitted as `CoordinatedStructuralCandidateJson`.

---

## 🔗 Relationship to Other Modules

| Module | Scope | Difference |
|--------|-------|------------|
| **Correlated error** | Output-neuron error correlations | Analyses correlations between output neurons, not hidden neuron per-output contributions |
| **Output conflict** (this) | Hidden neuron per-output disaggregation | Analyses the `errors[0..n]` vector for hidden neurons to find cross-output sign conflicts |

---

## 📝 Example Scenario

A creature with 2 outputs and 1 hidden neuron:

```mermaid
graph LR
    I1["📥 input-1"]
    H1["🧠 hidden-1"]
    O0["✅ output-0<br/>(speech recognition)<br/>mean error: -0.5"]
    O1["❌ output-1<br/>(noise classification)<br/>mean error: +0.3"]

    I1 --> H1
    H1 --> O0
    H1 --> O1

    style I1 fill:#4a9eff,stroke:#333,color:#fff
    style H1 fill:#4a9eff,stroke:#333,color:#fff
    style O0 fill:#2ecc71,stroke:#333,color:#fff
    style O1 fill:#e74c3c,stroke:#333,color:#fff
```

Over 50 training samples, `hidden-1` consistently:
- Reduces error on output-0 by ~0.5 (mean error = -0.5)
- Increases error on output-1 by ~0.3 (mean error = +0.3)

> [!IMPORTANT]
> The detector flags `hidden-1` with a `conflict_severity = 0.5 × 0.3 = 0.15` and recommends attenuating the synapse to output-1.
