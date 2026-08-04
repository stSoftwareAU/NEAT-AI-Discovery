# ⚖️ Sample-Weighted Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/recommendation/sample_weighted.rs`](../../src/analysis/recommendation/sample_weighted.rs) | **Issue:** [#423](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/423)

---

## 🔍 The Problem

Standard discovery treats all training samples equally when evaluating
neurons. But in many real-world problems, a small number of **hard samples**
dominate the creature's total error. Neurons that are mediocre on average
may be terrible on the hardest samples — and fixing those hard-sample
contributions would have the biggest impact on overall score.

**Sample-weighted discovery** re-weights the analysis so neurons contributing
disproportionately to high-error samples receive priority attention.

> [!IMPORTANT]
> Neurons that appear fine under uniform weighting can be the **primary drivers** of poor performance on the hardest samples. Sample-weighted discovery surfaces these hidden contributors.

```mermaid
graph TD
    subgraph "📊 Error Distribution Across Samples"
        S1["Sample 1<br/>Error: 0.01"]
        S2["Sample 2<br/>Error: 0.02"]
        S3["Sample 3<br/>Error: 0.85"]
        S4["Sample 4<br/>Error: 0.01"]
        S5["Sample 5<br/>Error: 0.03"]
        S6["Sample 6<br/>Error: 0.92"]
        S7["Sample 7<br/>Error: 0.02"]
        S8["Sample 8<br/>Error: 0.01"]
    end

    subgraph "⚖️ Weighting Comparison"
        UW["Uniform Weighting<br/>All samples contribute equally"]
        IW["Importance Weighting<br/>Samples 3 &amp; 6 dominate<br/>80% of total error"]
    end

    subgraph "🔎 Neuron H5 Analysis"
        EASY["Easy Samples<br/>Mean error = 0.02"]
        HARD["Hard Samples<br/>Mean error = 0.45"]
        RATIO["Ratio: 22.5x worse<br/>on hard samples!"]
        IMPACT["Fixing H5 on hard samples<br/>would have outsized impact"]
    end

    S1 & S2 & S4 & S5 & S7 & S8 --> UW
    S3 & S6 --> IW

    EASY --> RATIO
    HARD --> RATIO
    RATIO --> IMPACT

    style S1 fill:#2ecc71,stroke:#333,color:#fff
    style S2 fill:#2ecc71,stroke:#333,color:#fff
    style S3 fill:#e74c3c,stroke:#333,color:#fff
    style S4 fill:#2ecc71,stroke:#333,color:#fff
    style S5 fill:#2ecc71,stroke:#333,color:#fff
    style S6 fill:#e74c3c,stroke:#333,color:#fff
    style S7 fill:#2ecc71,stroke:#333,color:#fff
    style S8 fill:#2ecc71,stroke:#333,color:#fff
    style UW fill:#e3f2fd,stroke:#1565c0,color:#000
    style IW fill:#fff3e0,stroke:#f57c00,color:#000
    style EASY fill:#2ecc71,stroke:#333,color:#fff
    style HARD fill:#e74c3c,stroke:#333,color:#fff
    style RATIO fill:#fff3e0,stroke:#f57c00,color:#000
    style IMPACT fill:#4a9eff,stroke:#333,color:#fff
```

### ⚠️ Why It Hurts the Creature's Score

- **Hidden contributors**: Neurons may look fine on average but fail badly
  on the samples that matter most.
- **Missed optimisation**: Equal weighting dilutes the signal from
  high-error samples, making it harder to identify the real problems.
- **Robustness gap**: The creature handles easy cases well but collapses
  on difficult inputs.

---

## 🔬 How We Detect It

> [!NOTE]
> A minimum of **10 error samples** is required before sample-weighted analysis is performed. The weighted mean error threshold is **0.25**.

```mermaid
flowchart TD
    START["🔬 For each neuron<br/>input, hidden, or output"]
    STEP1["1️⃣ Collect error samples<br/>minimum 10"]
    STEP2["2️⃣ Compute per-sample<br/>importance weights<br/>weight_i = |error_i| / sum |all_errors|"]
    STEP3["3️⃣ Compute weighted mean error<br/>sum weight_i x |error_i|"]
    CHECK1{"4️⃣ Weighted mean<br/>error >= 0.25?"}
    SKIP["Skip neuron"]
    STEP5["5️⃣ Stratify into easy<br/>≤ median error<br/>and hard > median"]
    STEP6["6️⃣ Compute<br/>hard_to_easy_ratio"]
    CHECK2{"7️⃣ Ratio is<br/>high?"}
    RESULT["🎯 Neuron struggles<br/>on hard samples"]
    NORMAL["Neuron performs<br/>consistently"]

    START --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> CHECK1
    CHECK1 -->|No| SKIP
    CHECK1 -->|Yes| STEP5
    STEP5 --> STEP6
    STEP6 --> CHECK2
    CHECK2 -->|Yes| RESULT
    CHECK2 -->|No| NORMAL

    style START fill:#4a9eff,stroke:#333,color:#fff
    style STEP1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style STEP2 fill:#e3f2fd,stroke:#1565c0,color:#000
    style STEP3 fill:#e3f2fd,stroke:#1565c0,color:#000
    style CHECK1 fill:#fff3e0,stroke:#f57c00,color:#000
    style SKIP fill:#e0e0e0,stroke:#333,color:#000
    style STEP5 fill:#e3f2fd,stroke:#1565c0,color:#000
    style STEP6 fill:#e3f2fd,stroke:#1565c0,color:#000
    style CHECK2 fill:#fff3e0,stroke:#f57c00,color:#000
    style RESULT fill:#e74c3c,stroke:#333,color:#fff
    style NORMAL fill:#2ecc71,stroke:#333,color:#fff
```

---

## 🛠️ How We Fix It

The fix adjusts the neuron's bias to shift its operating point toward
better performance on high-error samples:

```mermaid
graph LR
    subgraph "❌ Before — biased toward easy samples"
        H1_B["H1"]
        N_B["Neuron<br/>bias = 0.5<br/>easy: OK<br/>hard: BAD"]
        O1_B["O1"]
        H1_B --> N_B --> O1_B
    end

    subgraph "✅ After — bias-adjusted"
        H1_A["H1"]
        N_A["Neuron<br/>bias = 0.43<br/>better on<br/>hard samples"]
        O1_A["O1"]
        H1_A --> N_A --> O1_A
    end

    style H1_B fill:#4a9eff,stroke:#333,color:#fff
    style N_B fill:#e74c3c,stroke:#333,color:#fff
    style O1_B fill:#4a9eff,stroke:#333,color:#fff
    style H1_A fill:#4a9eff,stroke:#333,color:#fff
    style N_A fill:#2ecc71,stroke:#333,color:#fff
    style O1_A fill:#4a9eff,stroke:#333,color:#fff
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Bias adjustment** | `setBias` | bias += -(weighted_mean_error × 0.1) |

> [!TIP]
> The adjustment is deliberately small (10% of weighted mean error) because NEAT-AI validates through ablation — conservative changes are more likely to pass validation.

Estimated improvement
(`sample_weighted.rs::detect_high_error_neurons`):

```text
min(weighted_mean_error × min(hard_to_easy_ratio, 10) × 0.01, 0.1)
```

> [!IMPORTANT]
> The ratio is clamped at **10** *before* scaling, so a neuron with a ratio of
> 18 contributes no more than one with a ratio of 10. Only the outer `0.1` cap
> was documented previously.

---

## 📝 Example

> **Neuron H12** across 200 samples:
>
> | Metric | Value |
> |--------|-------|
> | Weighted mean error | 0.38 (>= 0.25 threshold) |
> | Easy samples (100) | mean error = 0.04 |
> | Hard samples (100) | mean error = 0.72 |
> | Hard-to-easy ratio | 18.0 |
>
> **Candidate fix:**
>
> | Parameter | Value |
> |-----------|-------|
> | Current bias | 1.2 |
> | Adjustment | -(0.38 × 0.1) = -0.038 |
> | New bias | 1.162 |
> | Estimated improvement | min(0.38 × min(18.0, 10) × 0.01, 0.1) = 0.038 |
>
> **After fix:** The bias shift nudges the neuron's operating point toward
> better handling of hard samples, where most of the creature's error is
> concentrated.

---

## 📚 References

- **Source module**: [`src/analysis/recommendation/sample_weighted.rs`](../../src/analysis/recommendation/sample_weighted.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Output Bias Drift](output-bias-drift.md) — corrects
  systematic bias in output predictions
- **Related**: [Error Plateau](error-plateau.md) — detects uniformly high
  error at outputs
