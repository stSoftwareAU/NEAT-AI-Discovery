# 🔗 Add Synapse Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/synapse/`](../../src/analysis/synapse/)

---

## 🔍 The Problem

A creature's network may be missing a **direct connection** between two neurons
that should communicate. The source neuron carries useful information that would
reduce the target's error, but no synapse exists to transmit it.

```mermaid
graph LR
    S["🧠 Source Neuron (S)<br/>Has useful signal for T"]
    T["🧠 Target Neuron (T)<br/>Has error that S could reduce"]

    S -. "❌ No synapse<br/>(missing!)" .-> T

    style S fill:#4a9eff,stroke:#333,color:#fff
    style T fill:#e74c3c,stroke:#333,color:#fff
```

> 💡 **Key Insight:** The source neuron carries a signal that would help the
> target produce better predictions, but there is no connection to transmit it.

### ⚠️ Why It Hurts the Creature's Score

- The target neuron cannot access information that would help it produce
  better predictions.
- Error persists because the useful signal from S never reaches T.
- In NEAT evolution, adding connections is random — discovery identifies
  **which specific connections** would help most.

> ⚠️ **Without discovery**, NEAT must rely on random mutation to stumble upon
> beneficial connections. Discovery targets the most impactful ones directly.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["📋 For each target neuron T"]
    B["🔎 Enumerate all neurons S<br/>not directly connected to T<br/>(earlier in evaluation order)"]
    C["📊 Match samples:<br/>S's activation ↔ T's error<br/>(by observation index)"]
    D["📐 Compute optimal weight via least squares:<br/>w = Σ(error × activation) / Σ(activation²)<br/>(clamped to ±MAX_OUTGOING_WEIGHT)"]
    E["📈 Estimate improvement:<br/>How much would adding<br/>w × S_activation reduce T's error?"]
    F{"✅ Positive<br/>improvement?"}
    G["🎯 Keep as candidate"]
    H["🚫 Discard"]

    A --> B --> C --> D --> E --> F
    F -- "Yes" --> G
    F -- "No" --> H

    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#e3f2fd,stroke:#1565c0,color:#000
    style F fill:#f39c12,stroke:#333,color:#fff
    style G fill:#2ecc71,stroke:#333,color:#fff
    style H fill:#e74c3c,stroke:#333,color:#fff
```

### 🧬 Epistatic Pair Detection

The analysis also detects **epistatic pairs** — two neurons that complement
each other and work better together than alone:

```mermaid
graph TD
    subgraph coverage ["📊 Activation Coverage"]
        direction LR
        S1["🧠 S1"]
        S2["🧠 S2"]
    end

    subgraph samples ["🧪 Sample Responses"]
        s1["s1: S1 ✅ S2 ❌ → S1 covers"]
        s2["s2: S1 ❌ S2 ✅ → S2 covers"]
        s3["s3: S1 ✅ S2 ❌ → S1 covers"]
        s4["s4: S1 ❌ S2 ✅ → S2 covers"]
        s5["s5: S1 ❌ S2 ❌ → gap"]
    end

    result["✅ ≥ 70% non-overlapping activation<br/>→ Proposed as a coordinated pair"]

    S1 --> samples
    S2 --> samples
    samples --> result

    style S1 fill:#4a9eff,stroke:#333,color:#fff
    style S2 fill:#4a9eff,stroke:#333,color:#fff
    style result fill:#2ecc71,stroke:#333,color:#fff
    style s1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style s2 fill:#e3f2fd,stroke:#1565c0,color:#000
    style s3 fill:#e3f2fd,stroke:#1565c0,color:#000
    style s4 fill:#e3f2fd,stroke:#1565c0,color:#000
    style s5 fill:#fde3e3,stroke:#c01515,color:#000
    style coverage fill:#f0f4ff,stroke:#4a9eff,color:#000
    style samples fill:#f9f9f9,stroke:#999,color:#000
```

> 🧬 **Epistatic pairs** fire on different samples, so together they cover more
> error cases than either neuron alone — they are proposed as a coordinated pair.

---

## 🛠️ How We Fix It

Add the missing synapse with an optimally computed weight:

```mermaid
graph LR
    subgraph before ["❌ Before"]
        S_b["🧠 S"]
        T_b["🧠 T"]
    end

    subgraph after ["✅ After"]
        S_a["🧠 S"]
        T_a["🧠 T"]
        S_a -- "new synapse<br/>w = +0.008" --> T_a
    end

    style S_b fill:#4a9eff,stroke:#333,color:#fff
    style T_b fill:#e74c3c,stroke:#333,color:#fff
    style S_a fill:#4a9eff,stroke:#333,color:#fff
    style T_a fill:#2ecc71,stroke:#333,color:#fff
    style before fill:#fde3e3,stroke:#e74c3c,color:#000
    style after fill:#e3fde3,stroke:#2ecc71,color:#000
```

> 🎯 **Weight** is computed to minimise T's error given S's activation.

For epistatic pairs, both synapses are added together:

```mermaid
graph LR
    subgraph before ["❌ Before"]
        S1_b["🧠 S1"]
        S2_b["🧠 S2"]
        T_b["🧠 T"]
    end

    subgraph after ["✅ After"]
        S1_a["🧠 S1"]
        S2_a["🧠 S2"]
        T_a["🧠 T"]
        S1_a -- "w = +0.006" --> T_a
        S2_a -- "w = −0.004" --> T_a
    end

    style S1_b fill:#4a9eff,stroke:#333,color:#fff
    style S2_b fill:#4a9eff,stroke:#333,color:#fff
    style T_b fill:#e74c3c,stroke:#333,color:#fff
    style S1_a fill:#4a9eff,stroke:#333,color:#fff
    style S2_a fill:#4a9eff,stroke:#333,color:#fff
    style T_a fill:#2ecc71,stroke:#333,color:#fff
    style before fill:#fde3e3,stroke:#e74c3c,color:#000
    style after fill:#e3fde3,stroke:#2ecc71,color:#000
```

> 🔧 Both synapses are added **atomically** as a coordinated structural candidate.

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add synapse** | `addSynapse` | Weight clamped to ±`MAX_OUTGOING_WEIGHT` (0.01, `src/analysis/scoring/weights/mod.rs`) |
| **Epistatic pair** | 2× `addSynapse` (coordinated) | Both added atomically |

> [!NOTE]
> Issue #888 tightened the clamp by an order of magnitude (0.1 → 0.01) against
> the production discovery cache: successful synapses land at 0.001–0.005, while
> the 0.01–0.1 band almost always fails. Every weight illustrated on this page
> therefore sits in the 1e-3 band.

---

## 📝 Example

> **Scenario:** Output neuron O1 predicts house prices.
> Input I5 contains "number of bedrooms" but has no synapse to O1.
>
> **Analysis:**
> - Matched 500 samples of I5 activation with O1 error
> - Optimal weight: w = +0.007
> - Expected improvement: 12% reduction in O1's sum-of-squared error (exact for `MSE`; a ranking signal for other costs)
>
> **Fix:** `addSynapse I5 → O1 (weight +0.007)`
>
> ✅ Now O1 can factor in bedroom count directly.

---

## 📚 References

- **Least squares estimation** —
  [Wikipedia](https://en.wikipedia.org/wiki/Least_squares): The method used
  to compute optimal synapse weights that minimise target error.
- **Epistasis** —
  [Wikipedia](https://en.wikipedia.org/wiki/Epistasis): The biological
  concept of gene interactions, applied here to neural connections that work
  better in combination than individually.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends. NEAT adds
  connections through random mutation; this discovery identifies the most
  beneficial connections to add.
