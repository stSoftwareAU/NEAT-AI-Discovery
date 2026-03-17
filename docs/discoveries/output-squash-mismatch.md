# 🔧 Output Squash Mismatch

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/output_squash_mismatch.rs`](../../src/analysis/detection/output_squash_mismatch.rs) | **Issue:** [#545](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/545)

---

## 🔍 The Problem

An **output squash mismatch** occurs when an output neuron's activation
function is fundamentally incompatible with the target data distribution.
No amount of weight or bias tuning can fix this — the activation function
itself prevents the neuron from reaching the correct output values.

Four distinct mismatch patterns exist:

```mermaid
graph TD
    subgraph P1["✂️ 1. Clipping"]
        C1["Target: [0.0, 2.5]<br/>LOGISTIC: [0, 1]<br/><i>15%+ hit the ceiling</i>"]
    end
    subgraph P2["🚫 2. Range Mismatch"]
        C2["LOGISTIC: [0, 1]<br/>targets include negatives<br/><i>impossible to produce!</i>"]
    end
    subgraph P3["💥 3. Unbounded Mismatch"]
        C3["IDENTITY: [−50, +80]<br/>targets: [−1, +1]<br/><i>25%+ wildly out of range</i>"]
    end
    subgraph P4["🔬 4. Pre-activation Simulation"]
        C4["A different squash<br/>reduces error by >= 15%"]
    end
    style C1 fill:#fce4ec,stroke:#c62828,color:#000
    style C2 fill:#fce4ec,stroke:#c62828,color:#000
    style C3 fill:#fce4ec,stroke:#c62828,color:#000
    style C4 fill:#fff3e0,stroke:#f57c00,color:#000
```

### ⚠️ Why It Hurts the Creature's Score

- **Impossible targets**: The activation function physically cannot produce
  the output values needed, guaranteeing minimum error above zero.
- **Systematic error**: Clipping or range mismatches produce consistent,
  directional errors that bias cannot fix.
- **Wasted convergence effort**: Weight tuning tries to compensate for an
  architectural problem, wasting optimisation budget.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each output neuron"] --> B{"📏 Mean |error| >= 0.05?<br/>Sufficient samples?"}
    B -->|"No"| Z["✅ Not applicable"]
    B -->|"Yes"| C["🧪 Try each strategy in order"]
    C --> S1{"✂️ Strategy 1 — Clipping?<br/>>= 15% at saturation bounds<br/>bound error > 1.5× centre error"}
    S1 -->|"Yes"| R["🔧 Mismatch detected"]
    S1 -->|"No"| S2{"🚫 Strategy 2 — Range?<br/>All-positive activations<br/>> 30% above-average error"}
    S2 -->|"Yes"| R
    S2 -->|"No"| S3{"💥 Strategy 3 — Unbounded?<br/>Unbounded squash<br/>> 25% outside ±1.05<br/>out-of-range error > 1.2× mean"}
    S3 -->|"Yes"| R
    S3 -->|"No"| S4{"🔬 Strategy 4 — Simulation?<br/>Any candidate squash<br/>reduces error >= 15%"}
    S4 -->|"Yes"| R
    S4 -->|"No"| Z
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style S1 fill:#fff3e0,stroke:#f57c00,color:#000
    style S2 fill:#fff3e0,stroke:#f57c00,color:#000
    style S3 fill:#fff3e0,stroke:#f57c00,color:#000
    style S4 fill:#fff3e0,stroke:#f57c00,color:#000
    style R fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

> First matching strategy wins.

---

## 🛠️ How We Fix It

```mermaid
graph LR
    subgraph Before["❌ Before — LOGISTIC, targets need [−1,+1]"]
        BH1["🔧 O1<br/>LOGISTIC<br/>[0, +1]<br/><i>can't go negative!</i>"]
    end
    subgraph After["✅ After — TANH, symmetric range"]
        AH1["✨ O1<br/>TANH<br/>[−1, +1]<br/><i>full range!</i>"]
    end
    Before -->|"changeSquash"| After
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
```

| Strategy | Candidate | Operation | Typical Recommendation |
|----------|-----------|-----------|----------------------|
| **Clipping** | Change squash | `changeSquash` | Bounded → wider range (e.g., TANH) |
| **Range mismatch** | Change squash | `changeSquash` | Non-negative → symmetric (e.g., LOGISTIC → TANH) |
| **Unbounded mismatch** | Change squash | `changeSquash` | Unbounded → bounded (e.g., IDENTITY → TANH) |
| **Pre-activation comparison** | Change squash | `changeSquash` | Best-performing alternative from simulation |

---

## 📝 Example

> **Output neuron O1:** LOGISTIC, bias = 0.5
> Target data range: [−0.8, +1.2]
>
> **Strategy 2 triggers:**
>
> | Check | Result |
> |-------|--------|
> | Activation min | 0.12 (> −0.05 → all positive) |
> | Samples with above-average error | 42% (> 30%) |
> | Problem | Neuron cannot output negative values to match negative targets |
>
> **Candidate:** Change to TANH
> TANH range: [−1, +1] — covers the negative target values
> Estimated improvement: **0.15**
>
> After fix: O1 can now produce negative outputs,
> allowing it to match the full target distribution ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/output_squash_mismatch.rs`](../../src/analysis/detection/output_squash_mismatch.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Error Plateau](error-plateau.md) — detects output neurons
  stuck at uniformly high error
- **Related**: [Activation Recommendation](activation-recommendation.md) —
  proactive activation matching for hidden neurons
- **Related**: [Saturated Neuron](saturated-neuron.md) — detects saturation
  in hidden neurons
