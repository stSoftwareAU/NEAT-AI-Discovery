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

```
    ┌─────────┐    w=0.5     ┌─────────┐    w=1.0     ┌──────────┐
    │ input-0 │─────────────▶│ hidden-1│─────────────▶│ output-0 │
    └─────────┘              └─────────┘              └──────────┘
         │                        │                        │
   (not computed)            impact=1.0              impact=1.0
                            (1.0/1.0 × 1.0)
```

- **output-0** 🎯: impact = 1.0 (it's an output)
- **hidden-1** 🧠: impact = |1.0| / 1.0 × 1.0 = 1.0 (only synapse to output, 100% contribution)
- **input-0**: Not computed (input neurons are observation sources, not selectable for discovery)
  - *If we did calculate it*: |0.5| / 0.5 × 1.0 = 1.0 (only synapse to hidden-1, 100% contribution)

### Example: Branching Network

```
                          ┌─────────┐
                    w=0.3 │ hidden-1│─────────┐
                 ┌───────▶└─────────┘         │w=0.3
    ┌─────────┐  │                            │
    │ input-0 │──┤                            ▼
    └─────────┘  │                       ┌──────────┐
                 │              w=0.7    │ output-0 │ impact=1.0
                 └─────────────────────▶│          │ total_inbound=1.0
                                        └──────────┘

hidden-1 impact = |0.3| / 1.0 × 1.0 = 0.3

When both hidden-1 and input-0 connect to output-0:

    ┌─────────┐
    │ input-0 │──┬───────────────────────────────────────┐
    └─────────┘  │                                       │w=0.7
                 │w=1.0   ┌─────────┐      w=0.3         ▼
                 └───────▶│ hidden-1│───────────────▶┌──────────┐
                          └─────────┘                │ output-0 │
                                                     └──────────┘
                                                total_inbound = 1.0

hidden-1 impact = |0.3| / |0.3+0.7| × 1.0 = 0.3
```

---

## 📈 Visual Examples

### Example 1: Multiple Outputs (Cumulative Impact) ➕

When a neuron connects to multiple outputs, its impact is the **SUM** of all paths:

```
                               ┌──────────┐
                         w=1.0 │ output-0 │ impact=1.0
                    ┌─────────▶└──────────┘
    ┌─────────┐     │
    │   hub   │─────┤
    └─────────┘     │
                    │    w=1.0 ┌──────────┐
                    └─────────▶│ output-1 │ impact=1.0
                               └──────────┘

hub impact = (1.0/1.0 × 1.0) + (1.0/1.0 × 1.0) = 2.0
```

⚡⚡ This reflects that removing 'hub' affects TWO outputs!

### Example 2: Deep Network with Dilution 📉

Impact dilutes as you go deeper into the network:

```
    ┌─────────┐  w=1.0  ┌─────────┐  w=1.0  ┌─────────┐  w=1.0  ┌──────────┐
    │ input-0 │────────▶│ layer-1 │────────▶│ layer-2 │────────▶│ output-0 │
    └─────────┘         └─────────┘         └─────────┘         └──────────┘
                             │                  │                    │
                        impact=1.0          impact=1.0           impact=1.0

    But if layer-2 has many inputs:

    ┌─────────┐  w=1.0  ┌─────────┐  w=0.01  ┌─────────┐  w=1.0  ┌──────────┐
    │ input-0 │────────▶│ layer-1 │─────────▶│ layer-2 │────────▶│ output-0 │
    └─────────┘         └─────────┘          └─────────┘         └──────────┘
                             │     (99 other │     │                  │
                             │     synapses) │     │                  │
                        impact=0.01      total=1.0 │             impact=1.0
                    (0.01/1.0 × 1.0)         impact=1.0
```

---

## ⚙️ Special Squash Function Handling

The impact calculation is **squash-aware** to handle special activation functions correctly.

### 🎚️ STEP and BIPOLAR (Threshold Functions)

**STEP** and **BIPOLAR** are threshold (binary) functions:

```
STEP(x):                          BIPOLAR(x):
    output                            output
       │                                 │
     1 ├────────────                   1 ├────────────
       │            │                    │            │
       │            │                    │            │
       ├────────────┴─────▶ x     -1 ├───┘            │
     0 │                             └────────────────┴─────▶ x
       │                                           threshold=0
```

#### ⚠️ The Problem

With threshold functions, **tiny signals can have HUGE effects**:

```
    ┌──────────┐  w=0.000_001  ┌──────────┐  w=0.000_002  ┌─────────────┐
    │ hidden-a │──────────────▶│ hidden-b │──────────────▶│ STEP/BIPOLAR│
    └──────────┘               └──────────┘               │  output-0   │
    activation=0.000_001                                  └─────────────┘

    OLD impact calculation:
    contribution = 0.000_001 × 0.000_002 ≈ 0 (negligible!)

    BUT ACTUAL EFFECT:
    If hidden-b's input sum is at -0.000_001 (just below threshold 0):
    - Before: output = 0 (or -1 for BIPOLAR)
    - After adding synapse: input = -0.000_001 + 0.000_001×0.000_002 ≈ 0
    - After: Could flip to output = 1!

    Impact on score: Δoutput = 1.0 (or 2.0 for BIPOLAR -1→1)
    Calculated impact: ≈ 0
```

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

```
MINIMUM:                              MAXIMUM:
┌────────────────┐                   ┌────────────────┐
│   synapse-1 ───┤                   │   synapse-1 ───┤
│   synapse-2 ───┤───▶ min(all)      │   synapse-2 ───┤───▶ max(all)
│   synapse-3 ───┤                   │   synapse-3 ───┤
└────────────────┘                   └────────────────┘
```

#### ⚠️ The Problem

The old impact calculation assumed **summing** of inputs and normalised by
total weight:

```
    Old formula: impact = |weight| / Σ|weights| × child_impact

    For MINIMUM/MAXIMUM, only ONE synapse "wins" at any time.
    The others contribute NOTHING to the output!

    Example:
    ┌──────────┐  w=0.1
    │ hidden-a │────────────┐
    └──────────┘            │
                            │
    ┌──────────┐  w=0.5     ▼
    │ hidden-b │───────▶┌─────────────┐
    └──────────┘        │   MINIMUM   │
                        │   output    │
    ┌──────────┐  w=10.0│             │
    │ hidden-c │───────▶└─────────────┘
    └──────────┘

    OLD calculation (WRONG for MINIMUM):
    hidden-a impact = 0.1 / (0.1+0.5+10.0) × 1.0 = 0.0094 (~1%)
    hidden-b impact = 0.5 / 10.6 × 1.0 = 0.047 (~5%)
    hidden-c impact = 10.0 / 10.6 × 1.0 = 0.94 (~94%)

    ACTUAL behaviour (MINIMUM):
    If hidden-a×0.1 = 0.01, hidden-b×0.5 = 0.25, hidden-c×10.0 = 5.0:
    - MINIMUM selects 0.01 (hidden-a's contribution)
    - Output = 0.01
    - hidden-a has 100% impact on output!
    - hidden-b, hidden-c have 0% impact!
```

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

```
    ┌─────────────┐  w=0.001  ┌──────────┐
    │   dormant   │──────────▶│ output-0 │
    │ activation≈0│           └──────────┘
    └─────────────┘
    structural_impact = 0.001
    mean_activation = 0.0001
    activation_weighted_impact = 0.001 × 0.0001 = 1e-7
```
✅ **REMOVAL CANDIDATE** (💤 dormant neuron)

```
    ┌────────────────┐  w=0.001  ┌──────────┐
    │     active     │──────────▶│ output-0 │
    │ activation≈1e5 │           └──────────┘
    └────────────────┘
    structural_impact = 0.001
    mean_activation = 1e5
    activation_weighted_impact = 0.001 × 1e5 = 100
```
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
