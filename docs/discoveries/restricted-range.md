# 📏 Restricted Range Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/restricted_range.rs`](../../src/analysis/detection/restricted_range.rs) | **Issue:** [#399](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/399)

---

## 🔍 The Problem

A **restricted range** neuron is one that is active (not dead, not saturated)
but confined to a narrow band within its bounded activation function's
theoretical output range. The neuron is technically working, but using only
a small fraction of its capacity.

```mermaid
graph LR
    subgraph Range["📏 Restricted Range"]
        H1["📏 H1<br/>TANH range: [−1, +1]<br/>observed: [+0.2, +0.35]<br/><i>only 7.5% of range used!</i>"]
    end
    style H1 fill:#e74c3c,stroke:#333,color:#fff
```

> 📏 **Tiny window!** The neuron only uses 7.5% of the available [−1, +1] range — downstream neurons see a near-constant signal.

### ⚠️ Why It Hurts the Creature's Score

- **Poor resolution**: Downstream neurons see a near-constant signal with
  tiny variations, making it hard to distinguish between different inputs.
- **Wasted non-linearity**: The bounded activation function adds computational
  cost without providing meaningful non-linear transformation.
- **Capacity underuse**: The neuron could represent a much wider range of
  values but is constrained by its current weight/bias configuration.

---

## 🔬 How We Detect It

```mermaid
flowchart TD
    A["🔍 For each hidden neuron<br/>with bounded activation"] --> B["📐 Compute theoretical range<br/>(TANH: [−1, +1], LOGISTIC: [0, +1])"]
    B --> C["📊 Collect post-activation samples<br/><i>minimum 20</i>"]
    C --> D["📐 Compute utilisation:<br/>(observed_max − observed_min)<br/>÷ theoretical_range"]
    D --> E{"📏 Utilisation < 20%?"}
    E -->|"No"| Z["✅ Adequate range"]
    E -->|"Yes"| F{"💀 Not dead?<br/>range >= 0.01"}
    F -->|"No"| Z2["💀 Dead neuron"]
    F -->|"Yes"| G{"🫠 Not saturated?<br/>not touching bounds"}
    G -->|"No"| Z3["🫠 Saturated"]
    G -->|"Yes"| H["📏 Restricted range detected"]
    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e3f2fd,stroke:#1565c0,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#fce4ec,stroke:#c62828,color:#000
    style Z fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Z2 fill:#fce4ec,stroke:#c62828,color:#000
    style Z3 fill:#fce4ec,stroke:#c62828,color:#000
```

### 🎯 Key Distinction

This detector catches neurons in the "middle ground" between dead and
saturated — they produce output that varies, but not enough to be useful.

---

## 🛠️ How We Fix It

Up to three candidates are proposed per neuron:

```mermaid
graph LR
    subgraph Before["❌ Before — restricted range"]
        BH1["📏 H1<br/>TANH<br/>[.2, .35]"]
    end
    subgraph After["✅ After — full range"]
        AH1["✨ H1<br/>IDENTITY<br/>[wider]"]
    end
    Before -->|"changeSquash"| After
    style BH1 fill:#e74c3c,stroke:#333,color:#fff
    style AH1 fill:#2ecc71,stroke:#333,color:#fff
```

| Priority | Candidate | Operation | Detail |
|----------|-----------|-----------|--------|
| 1 | **Change squash** | `changeSquash` | Switch to IDENTITY (removes bounding) |
| 2 | **Adjust bias** | `setBias` | Shift to centre activation in theoretical range |
| 3 | **Rescale weights** | `setWeight` | Scale incoming weights by 0.80/utilisation to expand range |

---

## 📝 Example

> A creature has 25 hidden neurons. Discovery finds:
>
> **Neuron H14:** TANH, observed activations in [+0.20, +0.35]
>
> | Metric | Value |
> |--------|-------|
> | Theoretical range | [−1, +1] (width 2.0) |
> | Observed range | 0.15 (width) |
> | Utilisation | 7.5% |
> | Status | Not dead (range > 0.01), not saturated (away from ±1) |
>
> **Candidates:**
> 1. Change to IDENTITY → removes bounding, allows full range
> 2. Adjust bias → centres the operating point
> 3. Scale incoming weights by 0.80/0.075 ≈ 10.7× → expands pre-activation range
>
> **Result:** Neuron output spans a wider range, providing richer signal to downstream neurons ✅

---

## 📚 References

- **Source module**: [`src/analysis/detection/restricted_range.rs`](../../src/analysis/detection/restricted_range.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Operating Point](operating-point.md) — pre-activation
  analysis against the active zone
- **Related**: [Dead Neuron](dead-neuron.md) — neurons with zero output
- **Related**: [Saturated Neuron](saturated-neuron.md) — neurons stuck at
  activation bounds
