# Discovery Types

This document is the **single source of truth** for all discovery types used by
NEAT-AI-Discovery. It covers detection criteria, recommended actions, candidate
output format, and production success/failure rates.

> **Last updated**: 1 Feb 2026

## Table of Contents

- [Overview](#overview)
- [Discovery Type Summary](#discovery-type-summary)
- [Detailed Descriptions](#detailed-descriptions)
  - [Saturated Neuron Detection](#saturated-neuron-detection)
  - [Bottleneck Neuron Detection](#bottleneck-neuron-detection)
  - [Dead Neuron Detection](#dead-neuron-detection)
  - [Dormant Synapse Detection](#dormant-synapse-detection)
  - [Opposing Synapse Detection](#opposing-synapse-detection)
  - [Output Bias Drift Detection](#output-bias-drift-detection)
  - [Oscillating Neuron Detection](#oscillating-neuron-detection)
  - [Correlated Error Pattern Detection](#correlated-error-pattern-detection)
  - [Multi-Hop Candidate Analysis](#multi-hop-candidate-analysis)
  - [Redundant Path Pruning](#redundant-path-pruning)
  - [Add Neurons](#add-neurons)
  - [Add Synapses](#add-synapses)
  - [Remove Low-Impact Neurons](#remove-low-impact-neurons)
  - [Remove Harmful Synapse](#remove-harmful-synapse)
  - [Remove Neuron (High Error)](#remove-neuron-high-error)
  - [Combo Successful](#combo-successful)
- [Coordinated Structural Candidates](#coordinated-structural-candidates)
- [Production Success Rates](#production-success-rates)
- [Analysis and Recommendations](#analysis-and-recommendations)
- [Related Documentation](#related-documentation)

---

## Overview

Discovery types represent different mutation strategies that NEAT-AI-Discovery
suggests to improve a creature's score. The Rust library analyses recorded neuron
activations and errors to propose candidates, which are then validated by NEAT-AI
through ablation testing.

The workflow is:

```
NEAT-AI-Discovery (Rust)          NEAT-AI (TypeScript)
─────────────────────────         ────────────────────
  Analyse recordings      ──▶     Receive candidates
  Propose candidates              Apply mutation to clone
  Predict improvement             Re-score against full training set
                                  Record success/failure
```

---

## Discovery Type Summary

| Discovery Type | Source Module | Issue | Candidate Operations | Status |
|----------------|--------------|-------|---------------------|--------|
| [Saturated Neuron](#saturated-neuron-detection) | `saturation.rs` | #342 | `changeSquash`, `setBias` | 🟢 Active |
| [Bottleneck Neuron](#bottleneck-neuron-detection) | `bottleneck.rs` | #343 | `addNeuron`, `addSynapse` | 🟢 Active |
| [Dead Neuron](#dead-neuron-detection) | `dead_neuron.rs` | #341 | `removeNeuron` | 🟢 Active |
| [Dormant Synapse](#dormant-synapse-detection) | `dormant_synapse.rs` | #359 | `removeSynapse` | 🟢 Active |
| [Opposing Synapse](#opposing-synapse-detection) | `opposing_synapse.rs` | #360 | `removeSynapse`, `setWeight` | 🟢 Active |
| [Output Bias Drift](#output-bias-drift-detection) | `output_bias_drift.rs` | #361 | `setBias` | 🟢 Active |
| [Oscillating Neuron](#oscillating-neuron-detection) | `oscillating_neuron.rs` | #358 | `changeSquash`, `setBias` | 🟢 Active |
| [Correlated Error](#correlated-error-pattern-detection) | `correlated_error.rs` | #344 | `addNeuron`, `addSynapse` | 🟢 Active |
| [Multi-Hop](#multi-hop-candidate-analysis) | `multi_hop.rs` | #230 | `addNeuron`, `addSynapse` | 🟢 Active |
| [Redundant Path](#redundant-path-pruning) | `redundant_path.rs` | #164 | `removeSynapse`, `setWeight` | 🟢 Active |
| [Add Neurons](#add-neurons) | `neuron.rs` | — | `addNeuron` | 🟢 Active |
| [Add Synapses](#add-synapses) | `synapse.rs` | — | `addSynapse` | ⚠️ Low volume |
| [Remove Low-Impact](#remove-low-impact-neurons) | `neuron.rs` | — | `removeNeuron` | 🟢 Active |
| [Remove Harmful Synapse](#remove-harmful-synapse) | `synapse.rs` | — | `removeSynapse` | 🟠 Not tested |
| [Remove Neuron (Error)](#remove-neuron-high-error) | `neuron.rs` | — | `removeNeuron` | 🔴 Not working |
| [Combo Successful](#combo-successful) | — | — | Multiple | 🔴 Not working |

### Status Legend

| Status | Meaning |
|--------|---------|
| 🟢 Active | Working and producing results |
| 🟠 Not tested | Rust produces candidates but NEAT-AI does not test them yet |
| ⚠️ Low volume | Working but rarely suggested |
| 🔴 Not working | Being tested but 0% success rate |

---

## Detailed Descriptions

### Saturated Neuron Detection

**Source**: `src/analysis/saturation.rs` (Issue #342)

**Purpose**: Identifies neurons that are permanently saturated (stuck at
activation ceiling or floor) and recommends activation function changes or bias
adjustments. Saturated neurons pass no gradient information and block learning
in their region of the network.

**Detection criteria**:

1. **Activation near bounds**: For TANH, mean activation > 0.95 or < −0.95
   across samples.
2. **Low relative variance**: Input varies but output does not (activation
   function is squashing all variation).
3. **Uses a bounded activation**: Only bounded activations (TANH, LOGISTIC,
   HARD_TANH, etc.) can saturate. Unbounded activations (RELU, IDENTITY) are
   excluded, except RELU dead-zone detection (all activations at zero).

**Recommended actions**:

1. **Change activation function**: Switch from TANH to IDENTITY to restore
   signal flow.
2. **Adjust bias**: Shift bias to move the neuron's operating point away from
   saturation.

**Output**: Emitted as `coordinatedStructuralCandidates` with `changeSquash`
and/or `setBias` operations.

---

### Bottleneck Neuron Detection

**Source**: `src/analysis/bottleneck.rs` (Issue #343)

**Purpose**: Identifies hidden neurons that form information bottlenecks —
single points where many input signals converge through one neuron before
reaching outputs. A bottleneck limits the network's ability to represent
complex input combinations because one neuron's activation range must encode
all upstream information.

**Detection criteria**:

1. **High fan-in**: Many incoming connections (≥ `MIN_FAN_IN_FOR_BOTTLENECK`).
2. **High fan-in / fan-out ratio**: Significantly more inputs than outputs.
3. **Error concentration**: The neuron carries a disproportionate share of
   output error.
4. **Not an output neuron**: Output neurons are natural convergence points
   and are excluded.

**Recommended actions**:

1. **Add parallel neuron**: Create a new hidden neuron sharing a subset of
   inputs/outputs to increase capacity at the bottleneck.
2. **Add bypass synapse**: Add a direct connection from an upstream neuron
   to a downstream neuron, reducing dependency on the bottleneck.

**Output**: Emitted as `coordinatedStructuralCandidates` with `addNeuron`
and/or `addSynapse` operations.

---

### Dead Neuron Detection

**Source**: `src/analysis/dead_neuron.rs` (Issue #341)

**Purpose**: Identifies neurons that have become effectively dead (always
outputting zero or near-zero activation) and recommends their removal. Dead
neurons waste computation without contributing to the network's output.

**Detection criteria**:

1. **Near-zero activation**: Mean absolute activation < 1e-6 across all
   samples.
2. **Zero variance**: Standard deviation of activation ≈ 0 (always outputs
   the same value).
3. **Fewer than 1% active samples**: Less than 1% of samples show activation
   above 0.01.
4. **Minimum 20 samples**: Required for statistical reliability.
5. **Hidden neurons only**: Output and input neurons are excluded.

**Removal confidence** combines activation proximity to zero (40%), variance
proximity to zero (40%), and sample count (20%, plateaus at 1000 samples).

**Recommended actions**:

1. **Remove the neuron**: Dead neurons consume GPU resources during both
   training and inference without contributing useful information.

**Output**: Emitted as `coordinatedStructuralCandidates` with `removeNeuron`
operations.

---

### Dormant Synapse Detection

**Source**: `src/analysis/dormant_synapse.rs` (Issue #359)

**Purpose**: Identifies synapses with near-zero weights that contribute
negligible signal to their target neuron. Dormant synapses waste computation
during both forward pass and discovery analysis without providing meaningful
information flow.

**Detection criteria**:

1. **Near-zero weight**: The absolute weight is below a threshold (e.g.,
   1e-4).
2. **Low contribution**: The product of source activation and synapse weight
   is negligible relative to other inputs to the target neuron.
3. **Not the sole connection**: The target neuron has other incoming synapses
   (removing the only input would be destructive).

**Recommended actions**:

1. **Remove the synapse**: Reduces network complexity and may allow the
   controller to allocate evaluation budget to more promising candidates.

**Output**: Emitted as `coordinatedStructuralCandidates` with `removeSynapse`
operations.

---

### Opposing Synapse Detection

**Source**: `src/analysis/opposing_synapse.rs` (Issue #360)

**Purpose**: Identifies synapses whose contribution consistently works against
error reduction. When a synapse's activation–error correlation is strongly
positive (meaning the synapse increases the output when the error is already
positive, or decreases it when the error is already negative), the synapse is
actively hindering performance.

**Detection criteria**:

1. **Positive contribution–error correlation**: The Pearson correlation between
   `weight × source_activation` and the target neuron's error is strongly
   positive (≥ threshold).
2. **Meaningful contribution**: The synapse has a non-negligible mean absolute
   contribution (distinguishing from dormant synapses).
3. **Sufficient samples**: Enough recorded samples for statistical reliability.

**Recommended actions**:

1. **Remove the synapse**: If the opposition is strong, removing the synapse
   eliminates the harmful contribution.
2. **Flip the weight sign**: If the synapse carries useful magnitude but the
   wrong direction, negating the weight may help.

**Output**: Emitted as `coordinatedStructuralCandidates` with `removeSynapse`
or `setWeight` operations.

---

### Output Bias Drift Detection

**Source**: `src/analysis/output_bias_drift.rs` (Issue #361)

**Purpose**: Identifies output neurons with a consistent error sign bias —
neurons whose errors are predominantly positive (predicting too low) or
predominantly negative (predicting too high) across training samples. This
systematic bias indicates the neuron's bias parameter needs adjustment.

**Detection criteria**:

1. **Consistent error sign**: More than a threshold fraction of errors share
   the same sign (e.g., > 70% positive or > 70% negative).
2. **Meaningful mean error**: The absolute mean error is above a minimum
   threshold (not just noise).
3. **Sufficient samples**: Enough recorded samples for statistical reliability.
4. **Output neurons only**: Hidden and input neurons are excluded (their errors
   are indirect).

**Recommended actions**:

1. **Set bias**: Adjust the output neuron's bias by the negative of the mean
   error to centre the predictions.

**Output**: Emitted as `coordinatedStructuralCandidates` with a `setBias`
operation.

---

### Oscillating Neuron Detection

**Source**: `src/analysis/oscillating_neuron.rs` (Issue #358)

**Purpose**: Identifies hidden neurons whose activations oscillate between
positive and negative values across training samples, indicating the neuron
is fighting between two contradictory functions. Oscillating neurons may
benefit from an activation function change or bias adjustment to stabilise
their output.

**Detection criteria**:

1. **Sign changes**: The activation crosses zero frequently (more than a
   minimum fraction of samples show sign changes).
2. **Balanced signs**: Both positive and negative activations appear in
   substantial proportions (neither dominates overwhelmingly).
3. **Meaningful magnitude**: The mean absolute activation is above a minimum
   threshold (distinguishing from dead neurons).
4. **Hidden neurons only**: Input and output neurons are excluded.

**Recommended actions**:

1. **Change activation function**: Switch to ABSOLUTE or RELU to stabilise
   sign.
2. **Adjust bias**: Shift the operating point to favour the dominant sign.

**Output**: Emitted as `coordinatedStructuralCandidates` with `changeSquash`
and/or `setBias` operations.

---

### Correlated Error Pattern Detection

**Source**: `src/analysis/correlated_error.rs` (Issue #344)

**Purpose**: Identifies groups of output neurons that exhibit correlated error
patterns across samples, suggesting they share a common missing cause.
Recommends structural changes that address the shared cause rather than
treating each output independently.

**Detection method**:

1. **Compute error correlation matrix**: For each pair of output neurons,
   compute the Pearson correlation of their per-sample errors.
2. **Cluster correlated outputs**: Group outputs with correlation > threshold
   (0.7).
3. **Identify shared error samples**: Find samples where all neurons in a
   cluster err in the same direction.
4. **Find predictive inputs**: Identify which input neuron activations predict
   the shared error pattern.

**Recommended actions**:

1. **Add shared hidden neuron**: A new neuron connecting predictive inputs to
   all outputs in the correlated group, addressing the shared missing cause.

**Output**: Emitted as `coordinatedStructuralCandidates` with `addNeuron` and
`addSynapse` operations.

---

### Multi-Hop Candidate Analysis

**Source**: `src/analysis/multi_hop.rs` (Issue #230)

**Purpose**: Current discovery only considers single-hop improvements (adding
one synapse or neuron). For deep networks, multi-hop improvements (adding a
path of 2–3 connections) may be more effective. This module analyses
intermediate neurons to find deeper structural improvements.

**Detection method**:

1. **Identify target neurons with errors**: Focus on output and hidden neurons
   that have recorded errors.
2. **Find correlated intermediates**: For each target, find neurons whose
   activation correlates with the target's error but are not directly
   connected.
3. **Build multi-hop paths**: Chain source → intermediate(s) → target paths
   where each hop has a correlation-based improvement estimate.
4. **Prune aggressively**: Limit depth to 3 hops max, filter by correlation
   threshold, and skip already-connected pairs.

**Recommended actions**:

1. **Add bypass synapse**: Connect the best source directly to the target
   (2-hop path).
2. **Add relay neuron**: Insert a new hidden neuron along the path to relay
   information (3-hop path).

**Output**: Emitted as `coordinatedStructuralCandidates` with `addNeuron`
and/or `addSynapse` operations.

---

### Redundant Path Pruning

**Source**: `src/analysis/redundant_path.rs` (Issue #164)

**Discovery type**: `COORDINATED_PRUNE_AND_REWEIGHT`

**Purpose**: Detect that two existing paths feeding the same output compute
the same thing, prune the redundant path, and renormalise the survivor's
weight.

**Detection criteria**:

1. **Highly correlated activations**: Pearson correlation ≥ 0.85 between the
   two sources' activation patterns.
2. **Anti-correlated error gradients**: Both paths push the error in the same
   direction (strengthens redundancy case).
3. **Shared downstream synapses**: Both sources feed into the same target
   neuron.

**How it works**:

1. For each target neuron, collect all existing synapse sources with recorded
   activation samples.
2. For each pair of existing sources, compute the Pearson activation
   correlation.
3. If correlation ≥ 0.85, the weaker synapse (by absolute weight) is a prune
   candidate.
4. The survivor's weight is renormalised to `keep_weight + prune_weight` (sum
   of both weights).

**Example scenario**:
```
input-0 ──(w=0.5)──→ output-0   (activation pattern: linear ramp)
input-1 ──(w=0.3)──→ output-0   (activation pattern: identical linear ramp)

Detected: correlation = 0.99 → redundant
Result:   removeSynapse(input-1 → output-0)
          setWeight(input-0 → output-0, weight=0.8)
```

**Output**: Emitted as `coordinatedStructuralCandidates` with `removeSynapse`
and `setWeight` operations.

---

### Add Neurons

**Source**: `src/analysis/neuron.rs`

**Purpose**: Add a new hidden neuron by inserting it between a source and
target neuron.

**How it works**:

1. Rust analyses which (source → target) pairs would benefit from an
   intermediate computation.
2. Suggests a new hidden neuron with:
   - Incoming synapse from source neuron
   - Outgoing synapse to target neuron
   - Squash function (activation function) from a candidate set
   - Bias value to shift the activation

**Example success**:
```json
{
  "changeType": "add-neurons",
  "scoreDelta": 9.92e-7,
  "rustRequest": {
    "neuronCandidate": {
      "incomingWeight": 0.35,
      "outgoingWeight": 0.01,
      "bias": 10,
      "squash": "ABSOLUTE",
      "expectedCreatureScoreGain": 0.000010214326
    }
  }
}
```

**Output**: Emitted as `neuronCandidates` in the analysis result with
`addNeuron` operations.

---

### Add Synapses

**Source**: `src/analysis/synapse.rs`

**Purpose**: Add a new synapse connection between existing neurons.

**How it works**:

1. Rust analyses which neuron pairs would benefit from a direct connection.
2. Suggests a weight for the new synapse based on correlation analysis.
3. Target neurons get signals from the source that can reduce their error.

**Output**: Emitted as `helpfulSynapses` in the analysis result with
`addSynapse` operations.

---

### Remove Low-Impact Neurons

**Source**: `src/analysis/neuron.rs`

**Purpose**: Remove neurons that contribute less than the cost of their
complexity.

**How it works**:

1. Rust computes each neuron's `activation_weighted_impact`.
2. Neurons with impact < `costOfGrowth` (default: 1e-7) are candidates.
3. Removing these neurons reduces complexity without meaningful accuracy loss.

**Example success**:
```json
{
  "changeType": "remove-low-impact",
  "scoreDelta": 1.3e-7,
  "rustRequest": {
    "removalCandidate": {
      "impact": 1.1132797e-10,
      "reason": "Impact 2.14e-11 < costOfGrowth (1.00e-7), 2 synapses, saves 1.20e-7"
    }
  }
}
```

**Output**: Emitted as `removalCandidates` in the analysis result with
`removeNeuron` operations.

---

### Remove Harmful Synapse

**Source**: `src/analysis/synapse.rs`

**Purpose**: Remove existing synapses that are actively increasing creature
error.

**How it works**:

1. Rust analyses the correlation between synapse contributions and output
   error.
2. Identifies synapses where removing the connection would reduce error.
3. Returns `harmful_synapses` in the analysis result.

**Current status**: 🟠 Rust produces candidates but no samples are recorded in
production. Investigation needed into whether candidates are filtered out by
the NEAT-AI score gain check.

**Output**: Emitted as `harmfulSynapses` in the analysis result with
`removeSynapse` operations.

---

### Remove Neuron (High Error)

**Source**: `src/analysis/neuron.rs`

**Purpose**: Remove neurons with extremely high error magnitude (harmful
neurons).

**How it works**:

1. Rust identifies neurons with abnormally high error (e.g., 5.8e+17).
2. These neurons are presumed to be destabilising the network.
3. Removing them is predicted to improve the overall score.

**Current status**: 🔴 Not working — 0 successes from 2 attempts. The massive
discrepancy between predicted improvement and actual result suggests the error
magnitude calculation may not translate to actual score improvement.

**Output**: Emitted as `removalCandidates` in the analysis result with
`removeNeuron` operations.

---

### Combo Successful

**Purpose**: Apply multiple individually-successful changes together for
compounding gains.

**How it works**:

1. Multiple candidates that passed individual ablation tests are combined.
2. The combined mutation is applied and re-scored.
3. If synergistic, the combo should improve score more than individual changes.

**Current status**: 🔴 Not working — 0 successes from 8 attempts. Individual
successes may be interfering with each other when combined.

---

## Coordinated Structural Candidates

Most discovery types emit their candidates as `coordinatedStructuralCandidates`,
which group dependent edits into a single atomic candidate. NEAT-AI evaluates
the entire group in one ablation test.

**Operation vocabulary** (as of 1 Feb 2026):

| Operation | Parameters | Description |
|-----------|-----------|-------------|
| `removeSynapse` | `fromNeuronUuid`, `toNeuronUuid` | Remove a connection |
| `addSynapse` | `fromNeuronUuid`, `toNeuronUuid`, `weight` | Add a connection |
| `addNeuron` | `neuronUuid`, `neuronType`, `squash`, `bias`, `insertBeforeNeuronUuid?` | Add a hidden neuron |
| `removeNeuron` | `neuronUuid` | Remove a neuron |
| `changeSquash` | `neuronUuid`, `squash` | Change activation function |
| `setBias` | `neuronUuid`, `bias` | Adjust bias |
| `setWeight` | `fromNeuronUuid`, `toNeuronUuid`, `weight` | Adjust weight |

All 7 operation types are implemented in NEAT-AI's
`ApplyCoordinatedStructuralCandidate.ts` (verified Issue #337).

**Example** (redundant path pruning):
```json
{
  "coordinatedStructuralCandidates": [
    {
      "expectedCreatureScoreGain": 0.001,
      "comment": "Redundant path pruning: activation correlation 0.99",
      "operations": [
        { "type": "removeSynapse", "fromNeuronUuid": "input-1", "toNeuronUuid": "output-0" },
        { "type": "setWeight", "fromNeuronUuid": "input-0", "toNeuronUuid": "output-0", "weight": 0.8 }
      ]
    }
  ]
}
```

**Forward-only note**: For forward-only creatures,
`addNeuron.insertBeforeNeuronUuid` is used to place the neuron in the
`neurons[]` array before the target neuron so subsequent
`addSynapse(newNeuron → target)` respects the forward-only ordering constraint.

---

## Production Success Rates

| Discovery Type | Successes | Failures | Success Rate | Status |
|----------------|-----------|----------|--------------|--------|
| **add-neurons** | 556 | 8,944 | 5.9% | 🟢 Active |
| **add-synapses** | 1 | 9 | 10.0% | ⚠️ Low volume |
| **coordinated-structural** | — | — | — | 🟢 Active |
| **change-squash** | 2 | 9 | 18.2% | ⚠️ Low volume |
| **remove-low-impact** | 65 | 304 | 17.6% | 🟢 Active |
| **remove-harmful-synapse** | — | — | — | 🟠 Not tested |
| **remove-neuron** | 0 | 2 | 0.0% | 🔴 Not working |
| **dead-neuron-removal** | — | — | — | 🟢 Active |
| **redundant-path-pruning** | — | — | — | 🟢 Active |
| **combo-successful** | 0 | 8 | 0.0% | 🔴 Not working |

**Total**: 624 successes / 9,276 failures (6.3% overall success rate)

---

## Analysis and Recommendations

### What is Working

1. **add-neurons** (5.9% success rate, 556 successes) — Our primary source of
   successful discoveries. The gentle nudge variants with tight outgoing weights
   perform well.

2. **remove-low-impact** (17.6% success rate, 65 successes) — Reliable way to
   reduce complexity. Impact-weighted predictions are reasonably accurate.

3. **change-squash** (18.2% success rate when suggested) — High success rate
   but very rarely suggested. Investigation needed: why are more squash changes
   not being proposed?

### What Needs Investigation

1. **add-synapses** — Predictions are inverting (expected +0.00016, actual
   −0.00015). The correlation analysis may be flawed or missing saturation
   effects.

2. **change-squash** — Low suggestion rate. Only 11 total samples across all
   experiments.

### What is Not Working

1. **remove-neuron** — Massive prediction errors. Error magnitude (5.8e+17)
   does not translate to score impact.

2. **combo-successful** — Interference between changes. Individual successes
   do not combine well.

### Recommended Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| Done | Verify coordinated-structural implementation (Issue #337) | NEAT-AI implements all 7 operation types |
| High | Investigate why harmful_synapses are not recorded | Mapping exists but no samples in discovery folder |
| High | Investigate add-synapses prediction inversion | 10 samples show consistent wrong-direction predictions |
| High | Disable or fix remove-neuron | 0% success rate, wasting validation cycles |
| Medium | Review combo-successful strategy | 0% success rate, may be attempting incompatible combinations |
| Medium | Investigate change-squash suggestion rate | 18.2% success rate but only 11 samples |
| Low | Optimise add-neurons variants | Already working, but room for improvement |

---

## Related Documentation

- [Impact Calculation](IMPACT_CALCULATION.md) — How neuron impact is computed
- [Analysis Deep Dive](ANALYSIS_DEEP_DIVE.md) — Detailed analysis workflow and
  detection algorithms
- [README](../README.md) — Main project documentation
