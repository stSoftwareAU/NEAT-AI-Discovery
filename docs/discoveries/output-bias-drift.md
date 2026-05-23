# 📉 Output Bias Drift Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/output_bias_drift.rs`](../../src/analysis/output_bias_drift.rs) | **Issue:** [#361](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/361)

---

## 🔍 The Problem

**Output bias drift** occurs when an output neuron consistently predicts too
high or too low across the majority of training samples. The network has learned
the correct pattern shape but is offset by a constant amount.

```mermaid
graph LR
    subgraph Drift["📉 Systematic Bias"]
        O1["🎯 O1<br/>bias = 0.0<br/>mean error = +0.32"]
    end
    style O1 fill:#e74c3c,stroke:#333,color:#fff
```

> 📉 **Predictions follow the correct pattern but are shifted UP by ~0.4** — systematic positive bias. Every sample carries the same offset error.

### ⚠️ Why It Hurts the Creature's Score

- Every sample has approximately the same error (the offset).
- The error does not cancel out — it consistently adds to the total score
  penalty.
- A simple bias adjustment would eliminate this entire class of error.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each OUTPUT neuron"] --> B["📊 Collect error samples<br/><i>minimum 20</i>"]
    B --> C["📐 Compute mean error"]
    C --> D["📊 Count positive vs negative errors"]
    D --> E{"🧪 ALL checks pass?"}
    E -->|"> 70% errors share same sign<br/>AND |mean error| >= 0.01"| F["📉 Bias drift detected"]
    E -->|"No"| G["✅ No drift"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📊 Visualising the Detection

> **Error distribution for output neuron O1:**
>
> Mean error = **+0.32**, 85% of errors are positive → **clear positive bias drift**.

---

## 🛠️ How We Fix It

Adjust the output neuron's bias to centre the predictions:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BO1["📉 O1<br/>bias = 0.0<br/>mean error = +0.32"]
    end
    subgraph After["✅ After"]
        AO1["✨ O1<br/>bias = −0.32<br/>mean error ≈ 0.0"]
    end
    Before -->|"setBias: 0.0 + (−0.32) = −0.32"| After
    style BO1 fill:#e74c3c,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Set bias** | `setBias` | new_bias = current_bias − mean_error |

The estimated improvement is proportional to the magnitude of the drift and the
consistency (majority fraction).

---

## 📝 Example

> Output neuron O2 (classifying "cat" vs "not cat"):
>
> | Metric | Value |
> |--------|-------|
> | Samples | 500 |
> | Mean error | +0.18 (predictions consistently too high) |
> | Positive errors | 412/500 = 82.4% |
> | Current bias | 0.05 |
>
> **Recommended fix:** `setBias` → 0.05 + (−0.18) = **−0.13**
>
> After fix: predictions shift down by 0.18, mean error ≈ 0.0,
> score improves across 82% of samples ✅

---

## 📚 References

- **Bias in neural networks** —
  [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neuron#Types_of_transfer_functions):
  How the bias parameter shifts the activation function's operating point.
- **Mean squared error** —
  [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error): A common
  loss metric. Output bias drift inflates the network's loss under any cost
  function; the inflation is exact for `MSE` and a ranking signal for the
  other built-in costs (see
  [`docs/COST_FUNCTION_NOTES.md`](../COST_FUNCTION_NOTES.md) §4).
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
