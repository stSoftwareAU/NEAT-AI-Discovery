# 🔀 Redundant Path Pruning

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/redundant_path.rs`](../../src/analysis/detection/redundant_path.rs) | **Issue:** [#164](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/164)

---

## 🔍 The Problem

**Redundant paths** occur when two synapses feeding the same target neuron carry
effectively the same signal. The network wastes two connections to transmit
information that one could handle.

```mermaid
graph LR
    A["🧠 Source A"] -->|"w = +0.4"| T["🧠 Target T"]
    B["🧠 Source B"] -->|"w = +0.3"| T
    style A fill:#9b59b6,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style T fill:#e74c3c,stroke:#333,color:#fff
```

> 🔀 If A and B activations are highly correlated (r ≥ 0.85):
> both synapses carry ~the same information → **one is redundant**.

### ⚠️ Why It Hurts the Creature's Score

- **Structural complexity**: Two synapses where one would suffice increases
  the creature's complexity cost.
- **Fragile encoding**: If one path mutates slightly, the near-duplicate
  can cause unexpected changes.
- **Wasted capacity**: The creature could use those connections for genuinely
  different signals.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each target neuron T"] --> B["📊 Collect source synapses<br/>with activation samples<br/><i>minimum 30 matched samples</i>"]
    B --> C["📈 Compute pairwise<br/>|Pearson correlation|<br/>of source activations"]
    C --> D{"🧪 |correlation| ≥ 0.85?"}
    D -->|No| Z["✅ Sources are independent"]
    D -->|Yes| E["🔀 Redundant pair found!"]
    E --> F["⚖️ Weaker synapse (by |weight|)<br/>= prune candidate"]
    F --> G["📐 New survivor weight<br/>= keep_weight + prune_weight"]
    G --> H["📊 Estimate improvement:<br/>SSE original vs renormalised<br/>(exact for MSE; ranking signal<br/>for other costs)<br/>+ structural bonus"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#fce4ec,stroke:#c62828,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style G fill:#e3f2fd,stroke:#1565c0,color:#000
    style H fill:#e3f2fd,stroke:#1565c0,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### 📏 Correlation Threshold

| |Pearson r| Range | Interpretation |
|-------------------|----------------|
| 0.0 – 0.5 | 🟢 Unrelated signals |
| 0.5 – 0.85 | 🟡 Different enough to keep |
| **0.85 – 1.0** | 🔴 **REDUNDANT** (threshold) |

> **Note:** Uses **absolute** correlation — both positively correlated
> (r ≈ +1.0) and anti-correlated (r ≈ −1.0) signals count as
> redundant (they carry the same information, just inverted).

---

## 🛠️ How We Fix It

Remove the weaker synapse and renormalise the survivor:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BA["🧠 A"] -->|"w = +0.4"| BT["🧠 T"]
        BB["🧠 B"] -->|"w = +0.3 🔀"| BT
    end
    subgraph After["✅ After"]
        AA["🧠 A"] -->|"w = +0.7 ⚖️"| AT["🧠 T"]
        AB["🧠 B"]
    end
    style BA fill:#9b59b6,stroke:#333,color:#fff
    style BB fill:#9b59b6,stroke:#333,color:#fff
    style BT fill:#e74c3c,stroke:#333,color:#fff
    style AA fill:#9b59b6,stroke:#333,color:#fff
    style AB fill:#9b59b6,stroke:#333,color:#fff
    style AT fill:#2ecc71,stroke:#333,color:#fff
```

> Survivor weight = 0.4 + 0.3 = **0.7** — signal to T is approximately preserved but simpler. ✅

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove weaker synapse** | `removeSynapse` | The lower |weight| path |
| **Renormalise survivor** | `setWeight` | New weight = sum of both weights |

---

## 📝 Example

> Target hidden neuron H5 has two inputs:
>
> | Synapse | Weight |
> |---------|--------|
> | I2 → H5 | +0.45 |
> | I7 → H5 | +0.22 |
>
> Activation correlation between I2 and I7: **|r| = 0.91**
>
> Since 0.91 ≥ 0.85 and I7 has the smaller |weight|:
>
> **Fix:**
> 1. `removeSynapse` I7 → H5
> 2. `setWeight` I2 → H5 to 0.45 + 0.22 = **0.67**
>
> Sum-of-squared-error comparison shows the single-path produces nearly identical
> output with one fewer synapse. ✅ (SSE reduction is exact for `MSE`; for
> other costs it serves as a ranking signal — see
> [`docs/COST_FUNCTION_NOTES.md`](../COST_FUNCTION_NOTES.md) §4.)

---

## 📚 References

- **Feature redundancy** —
  [Wikipedia](https://en.wikipedia.org/wiki/Feature_selection#Redundancy):
  The general problem of redundant features in machine learning.
- **Optimal Brain Damage (LeCun et al., 1989)** — Foundational work on
  identifying and removing unnecessary connections in neural networks.
- **Optimal Brain Surgeon (Hasselmo et al., 1992)** — Extends pruning to
  account for weight renormalisation after removal, similar to this approach.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
