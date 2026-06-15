# 🧠 Neuron Impact Calculation

This document explains how neuron impact is calculated in NEAT-AI-Discovery. Impact
is a crucial metric that determines:

1. 🎯 **Focus neuron ranking**: Higher impact neurons are prioritised for discovery
2. 🗑️ **Removal candidate detection**: Low impact neurons can be safely removed
3. 📉 **Hidden neuron prediction discounting**: Impact affects confidence in predictions

## Table of Contents

- [What is Impact?](#what-is-impact)
- [Basic Impact Calculation](#basic-impact-calculation)
- [Visual Examples](#visual-examples)
- [Special Squash Function Handling](#special-squash-function-handling)
  - [STEP and BIPOLAR (Threshold Functions)](#step-and-bipolar-threshold-functions)
  - [MINIMUM and MAXIMUM (Selection Functions)](#minimum-and-maximum-selection-functions)
- [Activation-Weighted Impact](#activation-weighted-impact)
- [Implementation Status](#implementation-status-v01132)
- [Future Improvements](#future-improvements)

---

## ⚡ What is Impact?

**Impact** measures how much a neuron's activation affects the creature's final output
(and hence its score). The formula captures:

$$\text{impact} = \text{how much does changing this neuron's activation change the output?}$$

For **output neurons**, impact = 1.0 (they directly determine the score).

For **hidden neurons**, impact depends on:
1. The weights of synapses connecting them to outputs (directly or indirectly)
2. The squash functions of neurons they connect through
3. Their activation patterns

---

## 📊 Basic Impact Calculation

The current implementation uses a **normalised path weight** approach:

$$\text{impact}(n) = \sum_{\text{synapses}} \frac{|w|}{T} \times \text{impact}(\text{child})$$

Where:
- $w$ = the synapse weight from this neuron to its child
- $T$ = `total_inbound` = sum of |weights| of all synapses going INTO the child
- $\text{impact}(\text{child})$ = recursively computed impact of the child neuron 🔄
- Output neurons have `impact = 1.0`

### Example: Simple Chain

```mermaid
graph LR
    A["🔵 input-0<br/><i>(not computed)</i>"] -->|"w=0.5"| B["🧠 hidden-1<br/>impact=1.0<br/><i>(1.0/1.0 × 1.0)</i>"]
    B -->|"w=1.0"| C["🎯 output-0<br/>impact=1.0"]
    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#2ecc71,stroke:#333,color:#fff
```

- **output-0** 🎯: impact = 1.0 (it's an output)
- **hidden-1** 🧠: impact = |1.0| / 1.0 × 1.0 = 1.0 (only synapse to output, 100% contribution)
- **input-0**: Not computed (input neurons are observation sources, not selectable for discovery)
  - *If we did calculate it*: |0.5| / 0.5 × 1.0 = 1.0 (only synapse to hidden-1, 100% contribution)

### Example: Branching Network

```mermaid
graph LR
    A["🔵 input-0"] -->|"w=0.3"| B["🧠 hidden-1<br/>impact=0.3"]
    A -->|"w=0.7"| C["🎯 output-0<br/>impact=1.0<br/>total_inbound=1.0"]
    B -->|"w=0.3"| C
    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#2ecc71,stroke:#333,color:#fff
```

hidden-1 impact = |0.3| / 1.0 × 1.0 = 0.3

When both hidden-1 and input-0 connect to output-0:

```mermaid
graph LR
    A["🔵 input-0"] -->|"w=1.0"| B["🧠 hidden-1"]
    A -->|"w=0.7"| C["🎯 output-0<br/>total_inbound=1.0"]
    B -->|"w=0.3"| C
    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#2ecc71,stroke:#333,color:#fff
```

hidden-1 impact = |0.3| / |0.3+0.7| × 1.0 = 0.3

---

## 📈 Visual Examples

### Example 1: Multiple Outputs (Cumulative Impact) ➕

When a neuron connects to multiple outputs, its impact is the **SUM** of all paths:

```mermaid
graph LR
    H["🧠 hub<br/>impact=2.0"] -->|"w=1.0"| O0["🎯 output-0<br/>impact=1.0"]
    H -->|"w=1.0"| O1["🎯 output-1<br/>impact=1.0"]
    style H fill:#e67e22,stroke:#333,color:#fff
    style O0 fill:#2ecc71,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
```

hub impact = (1.0/1.0 × 1.0) + (1.0/1.0 × 1.0) = 2.0

⚡⚡ This reflects that removing 'hub' affects TWO outputs!

### Example 2: Deep Network with Dilution 📉

Impact dilutes as you go deeper into the network:

```mermaid
graph LR
    A["🔵 input-0"] -->|"w=1.0"| B["🧠 layer-1<br/>impact=1.0"]
    B -->|"w=1.0"| C["🧠 layer-2<br/>impact=1.0"]
    C -->|"w=1.0"| D["🎯 output-0<br/>impact=1.0"]
    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#9b59b6,stroke:#333,color:#fff
    style D fill:#2ecc71,stroke:#333,color:#fff
```

But if layer-2 has many inputs (99 other synapses):

```mermaid
graph LR
    A["🔵 input-0"] -->|"w=1.0"| B["🧠 layer-1<br/>impact=0.01<br/><i>(0.01/1.0 × 1.0)</i>"]
    B -->|"w=0.01"| C["🧠 layer-2<br/>impact=1.0<br/>total=1.0"]
    C -->|"w=1.0"| D["🎯 output-0<br/>impact=1.0"]
    E["⋯ 99 other<br/>synapses"] -.->|"w=..."| C
    style A fill:#4a9eff,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#9b59b6,stroke:#333,color:#fff
    style D fill:#2ecc71,stroke:#333,color:#fff
    style E fill:#95a5a6,stroke:#333,color:#fff
```

---

## ⚙️ Special Squash Function Handling

The impact calculation is **squash-aware** to handle special activation functions correctly.

### 🎚️ STEP and BIPOLAR (Threshold Functions)

**STEP** and **BIPOLAR** are threshold (binary) functions:

```mermaid
graph TD
    subgraph STEP["🎚️ STEP(x)"]
        direction TB
        S1["Output = 0 when x &lt; 0"]
        S2["Output = 1 when x ≥ 0"]
        S1 --- S2
    end
    subgraph BIPOLAR["🎚️ BIPOLAR(x)"]
        direction TB
        B1["Output = −1 when x &lt; 0"]
        B2["Output = +1 when x ≥ 0"]
        B1 --- B2
    end
    style STEP fill:#fff3cd,stroke:#f0ad4e,color:#333
    style BIPOLAR fill:#fff3cd,stroke:#f0ad4e,color:#333
    style S1 fill:#e74c3c,stroke:#333,color:#fff
    style S2 fill:#2ecc71,stroke:#333,color:#fff
    style B1 fill:#e74c3c,stroke:#333,color:#fff
    style B2 fill:#2ecc71,stroke:#333,color:#fff
```

> **Key insight**: These are binary threshold functions — the output jumps discontinuously at threshold = 0.

#### ⚠️ The Problem

With threshold functions, **tiny signals can have HUGE effects**:

```mermaid
graph LR
    A["🧠 hidden-a<br/>activation=0.000_001"] -->|"w=0.000_001"| B["🧠 hidden-b"]
    B -->|"w=0.000_002"| C["🎚️ STEP/BIPOLAR<br/>output-0"]
    style A fill:#9b59b6,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#e74c3c,stroke:#333,color:#fff
```

> **OLD impact calculation**: contribution = 0.000_001 × 0.000_002 ≈ 0 (negligible!)
>
> **BUT ACTUAL EFFECT**: If hidden-b's input sum is at −0.000_001 (just below threshold 0):
> - **Before**: output = 0 (or −1 for BIPOLAR)
> - **After adding synapse**: input ≈ 0 — could flip to output = 1!
> - **Impact on score**: Δoutput = 1.0 (or 2.0 for BIPOLAR −1 → 1)
> - **Calculated impact**: ≈ 0

🚨 **ERROR: Off by ~1,000,000x**

#### ✅ Implemented Solution (v0.1.132+)

For neurons feeding into STEP/BIPOLAR targets, **don't normalise by total_inbound**.
Each synapse is treated as potentially decisive:

$$\text{contribution} = \text{child\_impact}$$

**Rationale:**
- Any synapse crossing the threshold causes full output change 🔀
- Without activation data, we can't know which synapses are "close" to threshold
- Conservative: won't underestimate impact (may overestimate)
- Better for removal candidate detection (won't incorrectly flag as "safe to remove") ✅

#### 🔮 Future Enhancement

With activation data, we could compute the actual threshold-crossing probability:

$$\text{threshold\_impact} = P(\text{crossing}) \times \Delta\text{output}$$

Where:
- $P(\text{crossing})$ = fraction of samples where synapse pushes target across threshold 🎲
- $\Delta\text{output}$ = 1.0 for STEP, 2.0 for BIPOLAR

### 🏆 MINIMUM and MAXIMUM (Selection Functions)

**MINIMUM** and **MAXIMUM** are selection functions - they don't sum inputs, they
select one:

```mermaid
graph LR
    subgraph MIN["🏆 MINIMUM"]
        direction LR
        MS1["synapse-1"] --> MOUT["min(all)"]
        MS2["synapse-2"] --> MOUT
        MS3["synapse-3"] --> MOUT
    end
    subgraph MAX["🏆 MAXIMUM"]
        direction LR
        XS1["synapse-1"] --> XOUT["max(all)"]
        XS2["synapse-2"] --> XOUT
        XS3["synapse-3"] --> XOUT
    end
    style MOUT fill:#3498db,stroke:#333,color:#fff
    style XOUT fill:#e67e22,stroke:#333,color:#fff
    style MIN fill:#eaf2f8,stroke:#3498db,color:#333
    style MAX fill:#fdf2e9,stroke:#e67e22,color:#333
```

#### ⚠️ The Problem

The old impact calculation assumed **summing** of inputs and normalised by
total weight:

Old formula: `impact = |weight| / Σ|weights| × child_impact`

For MINIMUM/MAXIMUM, only ONE synapse "wins" at any time.
The others contribute NOTHING to the output!

```mermaid
graph LR
    A["🧠 hidden-a"] -->|"w=0.1"| M["🏆 MINIMUM<br/>output"]
    B["🧠 hidden-b"] -->|"w=0.5"| M
    C["🧠 hidden-c"] -->|"w=10.0"| M
    style A fill:#2ecc71,stroke:#333,color:#fff
    style B fill:#9b59b6,stroke:#333,color:#fff
    style C fill:#e74c3c,stroke:#333,color:#fff
    style M fill:#3498db,stroke:#333,color:#fff
```

> **OLD calculation (WRONG for MINIMUM):**
> - hidden-a impact = 0.1 / (0.1+0.5+10.0) × 1.0 = 0.0094 (~1%)
> - hidden-b impact = 0.5 / 10.6 × 1.0 = 0.047 (~5%)
> - hidden-c impact = 10.0 / 10.6 × 1.0 = 0.94 (~94%)
>
> **ACTUAL behaviour (MINIMUM):**
> If hidden-a×0.1 = 0.01, hidden-b×0.5 = 0.25, hidden-c×10.0 = 5.0:
> - MINIMUM selects 0.01 (hidden-a's contribution)
> - Output = 0.01 — **hidden-a has 100% impact!**
> - hidden-b, hidden-c have **0% impact!**

🚨 **ERROR: Completely inverted!**

#### ✅ Implemented Solution (v0.1.132+)

For MINIMUM/MAXIMUM targets, use **equal probability** for each synapse:

$$\text{contribution} = \frac{\text{child\_impact}}{N}$$

Where:
- $N$ = number of incoming synapses to the target
- Each synapse has $\frac{1}{N}$ probability of "winning" the selection 🎲

**Rationale:**
- Without activation data, we can't know which synapse wins
- Equal probability is conservative: won't incorrectly flag neurons as "low impact"
- MUCH better than old sum-based approach which was completely wrong!

❌ **Old behaviour (BROKEN)**: sum-based normalisation gave large weights ~90% impact
in MINIMUM, but small weights are more likely to win!

✅ **New behaviour (v0.1.143+)**: When activation records are available, we compute
the **actual selection probability** for each synapse:

$$\text{selection\_impact} = P(\text{winning}) \times \text{child\_impact}$$

Where:
- $P(\text{winning})$ = fraction of samples where this synapse has min/max value 🎲
- For MINIMUM: the synapse with smallest weighted contribution wins
- For MAXIMUM: the synapse with largest weighted contribution wins

**Example**: If hidden-small wins MINIMUM 90% of the time, it gets 90% impact.

When activation records are **not** available (e.g., using `compute_impacts_public`),
we fall back to the conservative 1/N equal probability approach.

---

## ⚖️ Activation-Weighted Impact

The final impact metric used for removal candidates is **activation-weighted impact**:

$$\text{activation\_weighted\_impact} = \text{structural\_impact} \times \text{mean\_absolute\_activation}$$

This captures that a neuron with:
- High structural impact but zero activation → no contribution → can remove 🗑️
- Low structural impact but huge activation → still contributes → don't remove ⛔

### Example

```mermaid
graph LR
    D["💤 dormant<br/>activation ≈ 0"] -->|"w=0.001"| O1["🎯 output-0"]
    style D fill:#95a5a6,stroke:#333,color:#fff
    style O1 fill:#2ecc71,stroke:#333,color:#fff
```

> **structural_impact** = 0.001 | **mean_activation** = 0.0001
> **activation_weighted_impact** = 0.001 × 0.0001 = **1e-7**

✅ **REMOVAL CANDIDATE** (💤 dormant neuron)

```mermaid
graph LR
    A["⚡ active<br/>activation ≈ 1e5"] -->|"w=0.001"| O2["🎯 output-0"]
    style A fill:#e67e22,stroke:#333,color:#fff
    style O2 fill:#2ecc71,stroke:#333,color:#fff
```

> **structural_impact** = 0.001 | **mean_activation** = 1e5
> **activation_weighted_impact** = 0.001 × 1e5 = **100**

❌ **NOT A REMOVAL CANDIDATE** (⚡ active neuron)

### Removal Threshold (`costOfGrowth`)

Neurons with `activation_weighted_impact < costOfGrowth` are flagged as removal
candidates. The `costOfGrowth` parameter is configurable (default: `1e-7`) and
matches NEAT-AI's `Score.ts` complexity penalty per neuron.

| `costOfGrowth` Value | Purpose |
|----------------------|---------|
| `1e-7` | Default — standard complexity penalty per neuron |
| `1e-9` or lower | Encourages creature expansion for evolution on new neurons |
| Higher values | More aggressive pruning (use with caution) |

Removal candidates are sorted by `activation_weighted_impact` ascending (lowest
first = safest to remove). Each candidate also includes `removalSavings`
calculated from NEAT-AI's complexity formula:

```
savings = costOfGrowth × (1 + (incomingSynapses + outgoingSynapses) / 10)
```

> **Note**: Non-finite activation values (NaN, Infinity) are filtered out when
> computing `mean_absolute_activation` to prevent corruption of the removal
> candidate ranking.

---

## 📋 Implementation Status (v0.2.1+)

The impact calculation is now **squash-aware** and uses **activation-based statistics**
when available. Different squash functions use different impact formulas:

> **Issue #130 fix (v0.2.1)**: The Linear squash formula now correctly uses normalised
> weights (`|w|/T × child_impact`) as documented. Previously, it incorrectly used
> absolute weights (`|w| × child_impact`), causing hidden neurons to get impact >= 1.0.

> **Issue #1300 fix (mirrors `NEAT-AI-Explore#266`)**: Two further correctness gaps
> have been addressed:
>
> 1. **Squash-bounded contribution.** The sum of inbound contributions to a neuron
>    cannot exceed the downstream squash's *emit magnitude* (`M`). For bounded
>    squashes (TANH, LOGISTIC, HARD_TANH, STEP, BIPOLAR, RELU6, ...), the sum is
>    capped at `M` even when the neuron feeds multiple outputs (so `child_impact > 1`).
>    Threshold squashes (STEP/BIPOLAR) previously returned the full `child_impact`
>    for *every* inbound synapse — overstating influence by a factor of `N` for
>    `N` inbound synapses. They now normalise by total inbound weight and apply
>    the same emit-magnitude cap as Linear bounded squashes.
> 2. **Downstream consumer gates.** A new `ConsumerContract` API lets callers
>    declare external `min(output, constant)` / `max(output, constant)` gates so
>    impact attribution is scaled by the gate's pass-through probability. Outputs
>    not in the contract default to a fully-open gate (no change in behaviour).
>    A helper `derive_regime_threshold_from_records` picks a percentile from the
>    recorded activation distribution when no external constant is known.

### 🛡️ Squash emit magnitude (Issue #1300)

| Squash | Output range | `squash_emit_magnitude` |
|--------|--------------|-------------------------|
| `TANH`, `LOGISTIC`, `HARD_TANH`, `SOFTSIGN`, `BIPOLAR_SIGMOID`, `ISRU` | bounded `±1` | `Some(1.0)` |
| `STEP`, `BIPOLAR`, `GAUSSIAN` | `{0, 1}` / `{-1, 1}` / `(0, 1]` | `Some(1.0)` |
| `ARCTAN` | `(-π/2, π/2)` | `Some(π/2)` |
| `RELU6` | `[0, 6]` | `Some(6.0)` |
| `IDENTITY`, `RELU`, `LEAKYRELU`, `ELU`, `SELU`, `CUBE`, `SQUARE`, ... | unbounded | `None` |
| `MINIMUM`, `MAXIMUM`, `IF`, `HYPOT`, `MEAN` | aggregate (selection stats handle these) | `None` |

The bounding formula for a per-synapse contribution `c` into a child neuron with
emit magnitude `M` and currently-accumulated `child_impact`:

$$
c_{\text{bounded}} = \begin{cases}
c & \text{if } M = \infty \text{ or } \text{child\_impact} \le M \\
c \cdot \dfrac{M}{\text{child\_impact}} & \text{otherwise}
\end{cases}
$$

This preserves the relative shares of inbound synapses while ensuring the
total influence respects the squash's saturation ceiling.

### 🔁 Consumer contracts (Issue #1300)

```rust
use neat_ai_discovery::focus::{
    ConsumerContract, OutputGate, compute_impacts_with_contract,
};

let contract = ConsumerContract::new()
    .with_gate("volume-output", OutputGate::MinAgainstConstant(0.25));
let impacts = compute_impacts_with_contract(&creature, Some(&records), Some(&contract))?;
```

When the gate is `MinAgainstConstant(t)`, the network output only drives the
downstream consumer in the regime where `output < t`. The impact attribution is
scaled by the fraction of recorded observations satisfying that condition.
`MaxAgainstConstant(t)` is the symmetric case (gate fires when `output > t`).
`Identity` is the no-op gate (equivalent to omitting the output from the
contract). The above flow is depicted below.

```mermaid
flowchart LR
    A[Recorded output<br/>activations] --> B{Gate?}
    B -- "Identity" --> C[Factor = 1.0]
    B -- "min(out, t)" --> D[Factor = P(out < t)]
    B -- "max(out, t)" --> E[Factor = P(out > t)]
    C --> F[Scale output<br/>initial impact]
    D --> F
    E --> F
    F --> G[Backward propagate<br/>through network]
```

| Squash Function | Impact Model | Accuracy | Notes |
|-----------------|--------------|----------|-------|
| **IDENTITY** | Linear (normalised) | ✅ Accurate | Mathematically exact |
| **TANH/LOGISTIC** | Linear (normalised) | ⚠️ Approx | Saturation not modelled |
| **HARD_TANH** | Linear (normalised) | ⚠️ Approx | Clamping not modelled |
| **STEP** | Threshold (full impact) | ✅ Conservative | Any synapse can flip output 🎚️ |
| **BIPOLAR** | Threshold (full impact) | ✅ Conservative | Any synapse can flip output 🎚️ |
| **MINIMUM** | Activation-based | ✅ Accurate | Actual win probability from samples 🏆 |
| **MAXIMUM** | Activation-based | ✅ Accurate | Actual win probability from samples 🏆 |
| **IF** | Synapse-type-aware | ✅ Accurate | Condition/positive/negative branches |
| **ReLU** | Linear (normalised) | ⚠️ Approx | Zero region not modelled |

### 🏷️ Squash Categories

The implementation categorises squash functions into three types:

```rust
enum SquashCategory {
    Linear,     // IDENTITY, TANH, LOGISTIC, HARD_TANH, ReLU, etc.
    Threshold,  // STEP, BIPOLAR
    Selection,  // MINIMUM, MAXIMUM, IF
}
```

### 📊 Impact Formulas by Category

**Linear squashes** (default):

$$\text{contribution} = \frac{|w|}{T} \times \text{child\_impact}$$

**Threshold squashes** (STEP/BIPOLAR) 🎚️:

$$\text{contribution} = \text{child\_impact}$$

No normalisation - any synapse can flip output!

**Selection squashes** (MINIMUM/MAXIMUM/IF) 🏆:

When activation records are available:

$$\text{contribution} = P(\text{winning}) \times \text{child\_impact}$$

Where $P(\text{winning})$ is computed from actual activation data.

Without activation records (fallback):

$$\text{contribution} = \frac{\text{child\_impact}}{N}$$

Equal probability for N incoming synapses.

**IF neurons** use synapse type information:
- `"condition"` synapses: $P = 1.0$ (always active)
- `"positive"` synapses: $P =$ fraction where condition sum > 0
- `"negative"` synapses: $P =$ fraction where condition sum ≤ 0

---

## 🔮 Future Improvements

### ✅ Completed

- [x] Document the problem (this document)
- [x] Track squash functions in impact calculation
- [x] Add squash-aware impact functions
- [x] STEP/BIPOLAR support (conservative full-impact approach) 🎚️
- [x] MINIMUM/MAXIMUM support (equal-probability approach) 🏆
- [x] **MINIMUM/MAXIMUM: Compute actual selection probability from samples** 🎲 (v0.1.143)
- [x] **IF: Synapse-type-aware impact (condition/positive/negative)** (v0.1.143)
- [x] Integration with removal candidate detection
- [x] Add tests for new edge cases

### 📋 Future Enhancements

These could further improve accuracy:

- [ ] STEP/BIPOLAR: Compute actual threshold-crossing probability from samples 🎲
- [ ] TANH/LOGISTIC: Model saturation using activation values
- [ ] ReLU: Model zero region using activation values

---

## 💻 Related Code

| File | Function | Purpose |
|------|----------|---------|
| `src/focus.rs` | `compute_impacts_internal()` | Core impact calculation 🔄 |
| `src/focus.rs` | `compute_impact_recursive()` | Recursive path traversal |
| `src/focus.rs` | `rank_focus_neurons()` | Uses impact for ranking 📊 |
| `src/analysis.rs` | `compute_impacts_public()` | Public API for analysis |
| `tests/focus.rs` | Various | Impact calculation tests ✅ |

---

## 📐 Appendix: Mathematical Formulas

### Current Formula (Normalised Path Weight)

$$
\text{impact}(n) = \begin{cases}
1.0 & \text{if } n \text{ is output} \\
\sum_i \frac{|w_i|}{T_i} \times \text{impact}(\text{child}_i) & \text{otherwise}
\end{cases}
$$

Where:
- $w_i$ = weight of synapse from $n$ to $\text{child}_i$
- $T_i$ = $\sum|w|$ for all synapses INTO $\text{child}_i$

### Squash-Aware Formula

$$
\text{impact}(n) = \begin{cases}
1.0 & \text{if } n \text{ is output} \\
\sum_i f_{\text{squash}}(n, \text{child}_i) & \text{otherwise}
\end{cases}
$$

Where $f_{\text{squash}}$ depends on child's squash function:

**Linear** (IDENTITY, etc.):

$$f = \frac{|w|}{T} \times \text{impact}(\text{child})$$

**Threshold** (STEP, BIPOLAR) 🎚️:

$$f = \text{impact}(\text{child})$$

**Selection** (MINIMUM, MAXIMUM) 🏆:

$$f = \frac{\text{impact}(\text{child})}{N}$$
