# ⚔️ Opposing Synapse Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/opposing_synapse.rs`](../../src/analysis/opposing_synapse.rs) | **Issue:** [#360](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/360)

---

## 🔍 The Problem

An **opposing synapse** is a connection whose contribution actively pushes the
output neuron further in the wrong direction. When the output is already too
high, the synapse makes it higher; when too low, it makes it lower.

```mermaid
graph LR
    A["🧠 A"] -->|"contribution = w × a"| O1["🎯 O1"]
    style A fill:#9b59b6,stroke:#333,color:#fff
    style O1 fill:#e74c3c,stroke:#333,color:#fff
```

> ⚔️ **Opposing synapse!** When O1 error is positive, contribution is also positive —
> pushing O1 even further in the wrong direction. Error persists or grows.

### ⚠️ Why It Hurts the Creature's Score

- The synapse is **actively harmful** — it increases error rather than
  reducing it.
- Unlike a dormant synapse (which does nothing), an opposing synapse makes
  things **worse**.
- Removing or flipping it provides an immediate improvement.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 Only synapses targeting<br/>OUTPUT neurons"] --> B["📊 Collect paired samples:<br/><i>(source activation, target error)</i><br/>matched by observation index"]
    B --> C["📐 Compute contribution<br/>= weight × source_activation"]
    C --> D["📈 Compute Pearson correlation<br/>between contribution and error"]
    D --> E{"🧪 correlation ≥ 0.3<br/>AND mean |contribution| ≥ 0.01?"}
    E -->|Yes| F["⚔️ Opposing synapse!<br/>harm = correlation × mean |contribution|"]
    E -->|No| G["✅ Synapse is fine"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📈 Interpreting the Correlation

| Correlation | Meaning | Effect |
|-------------|---------|--------|
| **≈ +1.0** | When error is positive, contribution is positive | ⚔️ **HARMFUL** — pushes output further from target |
| **≈ 0.0** | No consistent relationship | 😐 Synapse is neutral |
| **≈ −1.0** | When error is positive, contribution is negative | ✅ **HELPFUL** — pushes output toward target |

---

## 🛠️ How We Fix It

The fix depends on how strongly opposing the synapse is:

### ✂️ Strong Opposition (correlation > 0.5): Remove

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BA["🧠 A"] -->|"w = +0.3 ⚔️"| BO1["🎯 O1"]
        BB["🧠 B"] -->|"w = −0.2 ✅"| BO1
    end
    subgraph After["✅ After"]
        AA["🧠 A"]
        AB["🧠 B"] -->|"w = −0.2 ✅"| AO1["🎯 O1"]
    end
    style BA fill:#9b59b6,stroke:#333,color:#fff
    style BB fill:#9b59b6,stroke:#333,color:#fff
    style BO1 fill:#e74c3c,stroke:#333,color:#fff
    style AA fill:#9b59b6,stroke:#333,color:#fff
    style AB fill:#9b59b6,stroke:#333,color:#fff
    style AO1 fill:#2ecc71,stroke:#333,color:#fff
```

> Remove the harmful A→O1 synapse entirely.

### 🔄 Moderate Opposition (correlation 0.3–0.5): Flip

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BA2["🧠 A"] -->|"w = +0.3 ⚔️"| BO2["🎯 O1"]
    end
    subgraph After["✅ After"]
        AA2["🧠 A"] -->|"w = −0.3 ✅"| AO2["🎯 O1"]
    end
    style BA2 fill:#9b59b6,stroke:#333,color:#fff
    style BO2 fill:#e74c3c,stroke:#333,color:#fff
    style AA2 fill:#9b59b6,stroke:#333,color:#fff
    style AO2 fill:#2ecc71,stroke:#333,color:#fff
```

> Negate the weight: the synapse now pushes in the correct direction. 🔄

| Correlation | Candidate | Operation |
|-------------|-----------|-----------|
| > 0.5 | **Remove synapse** | `removeSynapse` |
| 0.3–0.5 | **Flip weight** | `setWeight` (negated, at 70% confidence) |

---

## 📝 Example

> Output neuron O1 consistently predicts too high (positive error).
>
> Synapse H3→O1 has weight **+0.4**.
> When O1 error is positive, H3's activation is also positive.
> contribution = 0.4 × activation ≈ **+0.2** on average.
>
> Pearson correlation between contribution and error: **r = 0.72**
>
> Since r > 0.5 → **remove synapse H3→O1**
> Expected improvement: 0.72 × 0.2 = **0.144** ✅

---

## 📚 References

- **Pearson correlation coefficient** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient):
  The statistical measure used to quantify the linear relationship between
  synapse contribution and output error.
- **Hebbian theory** —
  [Wikipedia](https://en.wikipedia.org/wiki/Hebbian_theory): The principle
  that connections should strengthen when they reduce error (the opposing
  synapse is the inverse — it strengthens error).
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
