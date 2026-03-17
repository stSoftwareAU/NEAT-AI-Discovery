# 😴 Dormant Synapse Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/dormant_synapse.rs`](../../src/analysis/dormant_synapse.rs) | **Issue:** [#359](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/359)

---

## 🔍 The Problem

A **dormant synapse** is a connection between two neurons whose weight has
decayed to near-zero. It contributes negligible signal to its target but still
adds to the creature's structural complexity.

```mermaid
graph LR
    A["🧠 A"] -->|"w ≈ 0.00 😴"| B["🧠 B"]
    style A fill:#9b59b6,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
```

> 😴 **Dormant synapse!** Signal = activation × 0.00 ≈ 0 — adds cost, contributes nothing.

### ⚠️ Why It Hurts the Creature's Score

- **Structural bloat**: Each synapse adds to the creature's complexity cost
  (cost of growth penalty in NEAT).
- **Wasted evaluation**: The connection is computed during forward pass but
  contributes nothing.
- **Evolutionary noise**: A near-zero weight can mutate back to a small
  value, creating misleading signals.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each synapse"] --> B{"📏 |weight| < 0.0001?"}
    B -->|No| Z["✅ Synapse is active"]
    B -->|Yes| C{"🔗 Target has fan-in > 1?"}
    C -->|No| Z2["🛡️ Skip — only input!"]
    C -->|Yes| D["📊 Compute mean |contribution|<br/><i>mean(|weight × source|)</i>"]
    D --> E{"📏 mean |contribution| < 0.0001?"}
    E -->|No| Z
    E -->|Yes| F["😴 Synapse is dormant"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

The fan-in check is a safety guard — we never remove a target neuron's only
remaining input, as that would effectively disconnect it.

```mermaid
graph LR
    subgraph Safe["✅ Safe to remove"]
        SA["🧠 A"] -->|"w ≈ 0 😴"| SC["🧠 C"]
        SB["🧠 B"] -->|"w = 0.5"| SC
    end
    subgraph Unsafe["❌ Not safe"]
        UA["🧠 A"] -->|"w ≈ 0 😴"| UC["🧠 C<br/><i>only input!</i>"]
    end
    style SA fill:#9b59b6,stroke:#333,color:#fff
    style SB fill:#9b59b6,stroke:#333,color:#fff
    style SC fill:#2ecc71,stroke:#333,color:#fff
    style UA fill:#9b59b6,stroke:#333,color:#fff
    style UC fill:#e74c3c,stroke:#333,color:#fff
```

> C has **fan-in = 2** → safe to remove A→C. C has **fan-in = 1** → do **NOT** remove.

---

## 🛠️ How We Fix It

Simply **remove the dormant synapse**:

```mermaid
graph LR
    subgraph Before["❌ Before"]
        BA["🧠 A"] -->|"w ≈ 0 😴"| BC["🧠 C"]
        BB["🧠 B"] -->|"w = 0.5"| BC
        BC --> BOut["🎯 output"]
    end
    subgraph After["✅ After"]
        AA["🧠 A"]
        AB["🧠 B"] -->|"w = 0.5"| AC["🧠 C"]
        AC --> AOut["🎯 output"]
    end
    style BA fill:#9b59b6,stroke:#333,color:#fff
    style BB fill:#9b59b6,stroke:#333,color:#fff
    style BC fill:#9b59b6,stroke:#333,color:#fff
    style BOut fill:#2ecc71,stroke:#333,color:#fff
    style AA fill:#9b59b6,stroke:#333,color:#fff
    style AB fill:#9b59b6,stroke:#333,color:#fff
    style AC fill:#9b59b6,stroke:#333,color:#fff
    style AOut fill:#2ecc71,stroke:#333,color:#fff
```

> Signal through C is essentially unchanged (lost only w ≈ 0 contribution from A)
> but the creature is simpler. ✅

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Remove synapse** | `removeSynapse` | Emitted as a coordinated structural candidate |

---

## 📝 Example

> A creature has 200 synapses. Analysis finds 12 with |weight| < 0.0001:
>
> | Synapse | Weight | Contribution |
> |---------|--------|-------------|
> | I3 → H5 | 0.00002 | 0.000008 |
> | H2 → H7 | 0.00001 | 0.000003 |
> | H8 → O1 | 0.00009 | 0.000041 |
> | … (9 more) | | |
>
> All targets have fan-in > 1.
>
> **Fix:** Remove all 12 dormant synapses
> **Result:** 188 synapses, lower complexity cost, same functional output ✅

---

## 📚 References

- **Network pruning** —
  [Wikipedia](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)):
  The general technique of removing low-magnitude connections, closely related
  to magnitude-based pruning methods.
- **LeCun, Denker & Solla (1989)** — *Optimal Brain Damage*: The foundational
  paper on removing low-saliency weights from neural networks.
- **NEAT (NeuroEvolution of Augmenting Topologies)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies):
  The evolutionary algorithm framework this library extends.
