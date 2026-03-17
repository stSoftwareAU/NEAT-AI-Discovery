# 🏔️ Error Plateau Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/error_plateau.rs`](../../src/analysis/detection/error_plateau.rs) | **Issue:** [#545](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/545)

---

## 🔍 The Problem

An **error plateau** occurs when an output neuron is stuck in a local minimum
characterised by persistently high, statistically uniform error. The error
surface is flat — gradient-based learning cannot find a direction to improve
because all nearby weight configurations produce similar error.

```mermaid
graph LR
    subgraph Plateau["🏔️ Error Plateau"]
        O1["🏔️ O1<br/>errors: 0.28, 0.31, 0.27, 0.30…<br/>mean = 0.294 (high)<br/>CV = 0.058 (tight)<br/><i>consistently wrong!</i>"]
    end
    style O1 fill:#e74c3c,stroke:#333,color:#fff
```

> 🏔️ **Stuck on a plateau!** The error is not random noise — it's consistently wrong, and the flat error surface provides no gradient signal for improvement.

### ⚠️ Why It Hurts the Creature's Score

- **Stuck at high error**: The output neuron reliably produces incorrect
  predictions, and weight adjustments cannot fix it.
- **Flat gradients**: The error surface provides no useful gradient signal
  for improvement.
- **Wrong activation function**: Often caused by a fundamental mismatch
  between the activation function and the target data distribution.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each output neuron"] --> B["📊 Collect error samples<br/><i>minimum 20</i>"]
    B --> C["📐 Compute mean |error|"]
    C --> D{"📏 Mean |error| >= 0.05?"}
    D -->|"No"| Z["✅ Error is low enough"]
    D -->|"Yes"| E["📐 Compute CV = std_dev / mean"]
    E --> F{"📏 CV <= 0.3?"}
    F -->|"No"| Z2["✅ Errors are varied — not a plateau"]
    F -->|"Yes"| G{"🧬 Different squash available?"}
    G -->|"Yes"| H["🏔️ Plateau detected"]
    G -->|"No"| Z3["🛡️ No alternative available"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z3 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📐 Plateau Tightness

The lower the coefficient of variation, the tighter the plateau. This
metric feeds into both confidence and improvement estimates:
`plateau_tightness = 1.0 − CV/0.3`.

---

## 🛠️ How We Fix It

The fix is a coordinated **activation function change + bias recentring**
to break out of the flat region of the error surface:

```mermaid
graph LR
    subgraph Before["❌ Before — stuck on plateau"]
        BH1["🏔️ O1<br/>HARD_TANH<br/>err ≈ 0.30<br/><i>stuck</i>"]
    end
    subgraph After["✅ After — new activation"]
        AH1["✨ O1<br/>TANH<br/>err ↓<br/><i>moving!</i>"]
    end
    Before -->|"changeSquash + setBias"| After
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
```

### 🧬 Activation Recommendations

| Current Squash | Recommended | Rationale |
|---------------|-------------|-----------|
| HARD_TANH, CLIPPED | TANH | Smooth version for better gradients |
| TANH | SOFTSIGN or IDENTITY | Different gradient profile |
| LOGISTIC | TANH or SOFTSIGN | Wider range, symmetric |
| RELU | TANH | Adds negative range |
| IDENTITY | TANH | Adds non-linearity |

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Squash change + bias recentre** | `changeSquash` + `setBias` | Atomic pair: new activation + bias adjustment |

The bias adjustment is: `new_bias = current_bias − mean_signed_error`,
recentring the output. Only included if the adjustment exceeds 0.02.

---

## 📝 Example

> **Output neuron O2:** LOGISTIC, bias = 0.8
>
> | Metric | Value |
> |--------|-------|
> | Error samples | [0.22, 0.25, 0.21, 0.24, 0.23, 0.22, 0.25, 0.24] |
> | Mean |error| | 0.233 (> 0.05) |
> | CV | 0.065 (< 0.3 → plateau confirmed) |
> | Mean signed error | −0.15 |
>
> **Candidate:**
> Change squash: LOGISTIC → TANH
> Set bias: 0.8 − (−0.15) = 0.95
> Plateau tightness: 1.0 − 0.065/0.3 = 0.78
> Estimated improvement: 0.233 × 0.78 × 0.3 = **0.055**
>
> The TANH activation provides a different gradient landscape,
> and the bias recentring shifts the output towards the targets ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/error_plateau.rs`](../../src/analysis/detection/error_plateau.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Output Squash Mismatch](output-squash-mismatch.md) — detects
  activation/target-data incompatibility at outputs
- **Related**: [Weight Magnitude Reset](weight-magnitude-reset.md) — escapes
  plateau by trying dramatically different weight values
- **Related**: [Output Bias Drift](output-bias-drift.md) — corrects
  systematic bias in output predictions
