# ⏭️ Skip Connection Discovery

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/skip_connection.rs`](../../src/analysis/detection/skip_connection.rs) | **Issue:** [#570](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/570)

---

## 🔍 The Problem

In deep networks, neurons far from the input layer suffer from **gradient
attenuation** — the error signal weakens as it passes back through multiple
layers, leaving deep neurons with little guidance for improvement. A **skip
connection** (also known as a residual connection) adds a direct shortcut
from a shallow neuron to a deep one, restoring the error signal.

```mermaid
graph LR
    I1["🟦 I1<br/>Input"]:::input --> H1["H1<br/>depth=1"]:::hidden
    H1 --> H3["H3<br/>depth=2"]:::hidden
    H3 --> H5["⚠️ H5<br/>depth=3"]:::problem
    H5 --> O1["O1<br/>Output"]:::output

    style H5 fill:#e74c3c,stroke:#333,color:#fff

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef hidden fill:#f39c12,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
```

> [!WARNING]
> 📉 **Gradient Attenuation Detected!** Neuron H5 at depth >= 3 has a mean |error| of only 30% compared to shallow neurons. The error signal has weakened significantly through the chain.

### ⚠️ Why It Hurts the Creature's Score

- **Weak learning signal**: Deep neurons receive attenuated error gradients,
  slowing their adaptation.
- **Wasted depth**: The creature has evolved a deep topology but cannot
  effectively utilise neurons far from the output.
- **Vanishing gradients**: A well-known problem in deep neural networks
  that skip connections directly address.

> [!NOTE]
> 🧠 Skip connections are inspired by **residual connections (ResNets)** from deep learning, where shortcut paths allow gradients to flow directly to earlier layers.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔄 For each hidden neuron"]:::process --> B{"📏 Compute topological<br/>depth via forward BFS"}:::process
    B --> C{"depth >= 3?"}:::decision
    C -- No --> Skip["✅ Skip — neuron<br/>is not deep"]:::output
    C -- Yes --> D["📊 Compare mean |error|<br/>to shallow neurons<br/>depth <= 2"]:::process
    D --> E{"Deep neuron error<br/>< 50% of<br/>shallow mean?"}:::decision
    E -- No --> NoAtten["✅ No attenuation<br/>detected"]:::output
    E -- Yes --> F["🔍 Find shallow source<br/>input or depth <= 1<br/>not already connected"]:::process
    F --> G{"Source<br/>found?"}:::decision
    G -- No --> NoSource["⏸️ No candidate<br/>available"]:::problem
    G -- Yes --> H["🎯 Skip connection<br/>candidate identified!"]:::output

    classDef process fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,color:#000
    classDef output fill:#2ecc71,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
```

> [!TIP]
> 🎯 **Detection Summary:** A neuron qualifies for a skip connection when it sits at depth >= 3 **and** its mean |error| is less than 50% of the shallow neuron average — confirming that gradient attenuation is occurring.

### 🔎 Source Selection

The best shallow source is chosen by maximising the depth gap (difference
in topological depth) while ensuring no existing connection exists.

---

## 🛠️ How We Fix It

```mermaid
graph LR
    subgraph Before["❌ Before — Gradient Attenuation"]
        direction LR
        B_I1["🟦 I1"]:::input --> B_H1["H1"]:::hidden
        B_H1 --> B_H3["H3"]:::hidden
        B_H3 --> B_H5["⚠️ H5"]:::problem
        B_H5 --> B_O1["O1"]:::output
    end

    subgraph After["✅ After — Skip Connection Added"]
        direction LR
        A_I1["🟦 I1"]:::input --> A_H1["H1"]:::hidden
        A_H1 --> A_H3["H3"]:::hidden
        A_H3 --> A_H5["H5"]:::output
        A_H5 --> A_O1["O1"]:::output
        A_I1 -. "⏭️ skip connection<br/>w = 0.01 × min(err, 1.0)" .-> A_H5
    end

    classDef input fill:#4a9eff,stroke:#333,color:#fff
    classDef hidden fill:#f39c12,stroke:#333,color:#fff
    classDef problem fill:#e74c3c,stroke:#333,color:#fff
    classDef output fill:#2ecc71,stroke:#333,color:#fff
```

| Candidate | Operation | Detail |
|-----------|-----------|--------|
| **Add skip connection** | `addSynapse` | Connect shallow source directly to deep neuron |

> [!IMPORTANT]
> ⚖️ The skip connection weight is deliberately conservative:
> `weight = 0.01 × min(target_mean_error, 1.0)` — this avoids destabilising the existing network while still restoring gradient flow.

**Estimated improvement:** `depth_gap × (1 - attenuation_ratio) × 0.005`

---

## 📝 Example

> **Scenario:** A creature has a chain: `I1 → H1 → H2 → H3 → H4 → O1`
>
> | Measurement | Value |
> |---|---|
> | Shallow neurons (depth <= 2) mean \|error\| | **0.15** |
> | Deep neuron H4 (depth = 4) mean \|error\| | **0.04** |
> | Attenuation ratio | 0.04 / 0.15 = **0.27** (< 0.50 threshold) |
>
> ✅ **Gradient attenuation confirmed**
>
> | Parameter | Value |
> |---|---|
> | Best shallow source | **I1** (depth = 0, not connected to H4) |
> | Depth gap | **4** |
>
> 🔧 **Candidate:** Add synapse `I1 → H4`
> - Weight: `0.01 × min(0.04, 1.0)` = **0.0004**
> - Estimated improvement: `4 × (1 - 0.27) × 0.005` = **0.015**
>
> 🎯 **Result:** H4 receives a direct signal from I1, bypassing the attenuating chain of hidden neurons.

---

## 📚 References

- **Source module**: [`src/analysis/detection/skip_connection.rs`](../../src/analysis/detection/skip_connection.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Topology Diversification](topology-diversification.md) —
  adds hidden neurons to flat (depth-0) paths
- **Related**: [Multi-Hop](multi-hop.md) — builds indirect connection paths
- **Residual connections (ResNets)** —
  [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network):
  The skip connection concept from deep learning that inspired this module.
