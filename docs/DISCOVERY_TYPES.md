# Discovery Types

This document is the **single source of truth** for all discovery types used by
NEAT-AI-Discovery. It covers detection criteria, recommended actions, candidate
output format, and production success/failure rates.

> **Last updated**: 6 Feb 2026

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
  - [Unbounded Capping Detection](#unbounded-capping-detection)
  - [Noise-to-Signal Ratio Detection](#noise-to-signal-ratio-detection)
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
| [Unbounded Capping](#unbounded-capping-detection) | `unbounded_capping.rs` | #441 | `changeSquash` | 🟢 Active |
| [Noise-to-Signal](#noise-to-signal-ratio-detection) | `noise_signal.rs` | #434 | `removeNeuron`, `removeSynapse`, `setWeight` | 🟢 Active |
| [Activation Recommendation](#activation-function-recommendation) | `activation_recommendation.rs` | #431 | `changeSquash` | 🟢 Active |
| [Correlated Error](#correlated-error-pattern-detection) | `correlated_error.rs` | #344 | `addNeuron`, `addSynapse` | 🟢 Active |
| [Multi-Hop](#multi-hop-candidate-analysis) | `multi_hop.rs` | #230 | `addNeuron`, `addSynapse` | 🟢 Active |
| [Redundant Path](#redundant-path-pruning) | `redundant_path.rs` | #164 | `removeSynapse`, `setWeight` | 🟢 Active |
| [Add Neurons](#add-neurons) | `neuron.rs` | — | `addNeuron` | 🟢 Active |
| [Add Synapses](#add-synapses) | `synapse.rs` | #413 | `addSynapse` | 🟡 Fixed |
| [Remove Low-Impact](#remove-low-impact-neurons) | `neuron.rs` | — | `removeNeuron` | 🟢 Active |
| [Remove Harmful Synapse](#remove-harmful-synapse) | `implementation.rs` | #416 | `removeSynapse` | 🟢 Active |
| [Remove Neuron (Error)](#remove-neuron-high-error) | `focus.rs` | #414 | `removeNeuron` | ⛔ Disabled |
| [Combo Successful](#combo-successful) | `epistatic.rs` | #415 | Multiple | 🟡 Fixed |

### Status Legend

| Status | Meaning |
|--------|---------|
| 🟢 Active | Working and producing results |
| 🟡 Fixed | Issue addressed, awaiting production validation |
| 🟠 Not tested | Rust produces candidates but NEAT-AI does not test them yet |
| ⚠️ Low volume | Working but rarely suggested |
| 🔴 Not working | Being tested but 0% success rate |
| ⛔ Disabled | Permanently disabled due to fundamental flaw |

---

## Detailed Descriptions

### Saturated Neuron Detection

**Source**: `src/analysis/saturation.rs` (Issue #342)

**Purpose**: Identifies neurons that are permanently saturated (stuck at
activation ceiling or floor) and recommends activation function changes or bias
adjustments. Saturated neurons pass no gradient information and block learning
in their region of the network.

**Detection criteria**:

1. **Activation near bounds**: For TANH, mean activation > 0.85 or < −0.85
   across samples (Issue #417: lowered from 0.95 to catch near-saturated
   neurons earlier). For LOGISTIC, mean activation > 0.90 or < 0.10
   (lowered from 0.95/0.05). For HARD_TANH, mean activation > 0.95
   (lowered from 0.99).
2. **Low relative variance**: Input varies but output does not (activation
   function is squashing all variation). Maximum std dev: 0.08 (Issue #417:
   raised from 0.05 to accommodate near-saturated neurons).
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

1. **Sign changes**: The activation crosses zero frequently (more than 15%
   of consecutive sample pairs show sign changes; Issue #417: lowered from
   30% to catch mildly oscillating neurons).
2. **Balanced signs**: Both positive and negative activations appear in
   at least 10% of samples (Issue #417: lowered from 20% to catch more
   oscillating neurons).
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

### Unbounded Capping Detection

**Source**: `src/analysis/unbounded_capping.rs` (Issue #441)

**Purpose**: Identifies neurons with unbounded activation functions (RELU,
IDENTITY, LEAKYRELU, etc.) that are producing high activations ("spiking")
and recommends capping them with a bounded version (e.g., RELU → RELU6).
High activations from unbounded functions can introduce noise into the
network.

**Detection criteria**:

1. **Uses an unbounded activation**: RELU, IDENTITY, LEAKYRELU, SOFTPLUS,
   ELU, SELU, SWISH, MISH, GELU, EXPONENTIAL, SQUARE, CUBE.
2. **High activations**: Maximum activation exceeds the capping threshold
   (e.g., > 6.0 for RELU → RELU6).
3. **Consistent spiking**: At least 30% of samples exceed the threshold
   (not just occasional spikes).
4. **Hidden neurons only**: Output neurons are excluded.

**Recommended actions**:

1. **Change RELU → RELU6**: Cap activations at 6.0 to reduce noise.
2. **Change LEAKYRELU → RELU6**: Cap positive activations (loses negative
   leak, but caps the positive side).
3. **Change IDENTITY → HARD_TANH or RELU6**: For high positive activations,
   use RELU6; for mixed activations, use HARD_TANH.

**Output**: Emitted as `coordinatedStructuralCandidates` with `changeSquash`
operations.

---

### Activation Function Recommendation

**Source**: `src/analysis/activation_recommendation.rs` (Issue #431)

**Purpose**: Provides **proactive** activation function recommendations based
on input distribution analysis. Unlike reactive `changeSquash` candidates
triggered by saturation or oscillation detection, this module analyses input
patterns BEFORE problems occur and suggests activations that match the data
characteristics.

**Input distribution matching**:

1. **Gaussian inputs → TANH or SOFTPLUS**: Bell-curve distributed inputs work
   well with smooth, symmetric activations that map the full range.
2. **Sparse inputs → RELU variants**: Inputs with many zeros benefit from
   activations that preserve the sparsity pattern.
3. **Bounded inputs → LOGISTIC or HARD_TANH**: Inputs constrained to a tight
   range (e.g., [0,1]) match bounded activations.
4. **Uniform inputs**: Evenly distributed inputs are flexible; smooth
   activations like TANH, IDENTITY, or GELU work well.

**Output range analysis**:

1. **Binary outputs**: Recommend LOGISTIC, STEP, or BIPOLAR.
2. **Unit interval [0,1]**: Recommend LOGISTIC.
3. **Symmetric unit [-1,1]**: Recommend TANH or HARD_TANH.
4. **Unbounded**: Recommend IDENTITY or RELU.

**Gradient flow analysis**:

The module also considers gradient flow risk:
- Penalises TANH/LOGISTIC if inputs would cause saturation.
- Penalises RELU if many inputs are negative (information loss).

**Detection criteria**:

1. **Minimum 20 samples**: Required for reliable distribution analysis.
2. **Classification possible**: Input distribution must be classifiable.
3. **Significant improvement**: Recommended activation must score higher than
   current activation by at least 0.001.
4. **Different activation**: Does not recommend the same activation currently
   in use.

**Recommended actions**:

1. **Change activation function**: Switch to an activation that better matches
   the observed input distribution pattern.

**Output**: Emitted as `coordinatedStructuralCandidates` with `changeSquash`
operations.

**Expected improvement**: 20% reduction in saturation/oscillation issues
through proactive matching of activation to data characteristics.

---

### Noise-to-Signal Ratio Detection

**Source**: `src/analysis/noise_signal.rs` (Issue #434)

**Purpose**: Part of the "Brilliant but Brittle" initiative (Issue #432). This
module identifies neurons and synapses with high noise-to-signal ratios that
contribute to brittle predictions when bad or missing observations wildly
affect outputs.

**Detection criteria for noisy neurons**:

1. **Low activation variance**: The neuron's activation varies little across
   samples, indicating it doesn't respond to meaningful input patterns.
2. **High error variance**: The neuron's error varies significantly, indicating
   unpredictable contribution to network output.
3. **Poor correlation**: Activation changes don't correlate with error reduction.
4. **Noise-to-signal ratio > threshold**: The ratio of error variance to
   activation variance exceeds the configurable threshold (default: 2.0).
5. **Hidden neurons only**: Input and output neurons are excluded.
6. **Minimum 20 samples**: Required for statistical reliability.

**Detection criteria for noisy synapses**:

1. **Large weight**: Amplifies variance from upstream neurons (≥ 0.1).
2. **Noisy source**: Source neuron has high variance but poor error correlation
   with the target.
3. **Low signal contribution**: The synapse contributes more noise (variance)
   than signal (error reduction).
4. **Noise exceeds signal by 2×**: Noise contribution is at least twice the
   signal contribution.

**Environment variable**:

- `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD`: Configure the noise-to-signal
  ratio threshold for neuron detection (default: 2.0).

**Recommended actions**:

1. **Remove noisy neurons**: Hidden neurons with poor signal-to-noise are
   candidates for removal via `removeNeuron`.
2. **Remove noisy synapses**: Synapses that amplify noise without signal benefit
   can be removed via `removeSynapse`.
3. **Reduce synapse weight**: For synapses with some signal contribution,
   `setWeight` reduces the weight by 50% to dampen noise while preserving
   some signal.

**Output**: Emitted as `coordinatedStructuralCandidates` with `removeNeuron`,
`removeSynapse`, or `setWeight` operations.

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

**Source**: `src/analysis/synapse.rs` (Issue #413)

**Purpose**: Add a new synapse connection between existing neurons.

**How it works**:

1. Rust analyses which neuron pairs would benefit from a direct connection.
2. Suggests a weight for the new synapse based on correlation analysis.
3. Target neurons get signals from the source that can reduce their error.

**Issue #413 Fix**: Predictions were inverting for targets with saturating
activation functions (HARD_TANH, TANH, LOGISTIC, etc.). The synapse weight was
computed using a linear least-squares model but evaluated against a
saturation-aware model. When the target neuron operates near saturation, the
linear-model weight overshoots into the saturated region, causing inverted
predictions (expected +0.00016, actual −0.00015).

**Root cause**: Linear vs saturation model mismatch. The add-neuron path already
handled this by searching over multiple weight candidates, but the add-synapse
path used only the single linear-model weight.

**Fix**: For saturating target activations, the add-synapse path now searches
over 9 weight candidates (scaled versions of the linear-model weight), matching
the approach used by add-neuron candidates. The candidate with the best
predicted improvement is selected.

**Current status**: 🟡 Fixed (Issue #413) — awaiting production validation.
Target: 15–20% success rate (up from 10%).

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

**Source**: `src/analysis/implementation.rs`

**Purpose**: Remove existing synapses that are actively increasing creature
error.

**How it works**:

1. For each existing synapse targeting a focus neuron, Rust evaluates the
   synapse using GPU-accelerated batch processing.
2. The GPU shader (`harmful.wgsl`) counts samples where the synapse contribution
   (activation × weight) has the **same sign** as the error (harmful) versus
   **opposite sign** (helpful).
3. A synapse is considered harmful when removing it would improve the score:
   `expected_improvement = (harmful_count - helpful_count) / total_count > 0`
4. Only synapses with positive expected improvement are returned in
   `harmful_synapses`.

**Detection criteria**:

1. **Same-sign contribution**: The synapse's contribution (source_activation ×
   weight) has the same sign as the target neuron's error on a significant
   fraction of samples.
2. **Positive expected improvement**: The proportion of harmful samples exceeds
   the proportion of helpful samples.
3. **Minimum samples**: At least one sample must exist for evaluation.

**Issue #416 Fix**: Previously, all existing synapses were returned in
`harmful_synapses` regardless of whether they were actually harmful. This
resulted in candidates with negative `expected_creature_score_gain` being
included, which NEAT-AI correctly filtered out on its side. The fix adds a
threshold check (`neuron_error_improvement > 0.0`) to only include truly
harmful synapses.

**Current status**: 🟢 Active (Issue #416 fixed)

**Output**: Emitted as `harmfulSynapses` in the analysis result with
`removeSynapse` operations.

---

### Remove Neuron (High Error)

**Source**: `src/focus.rs` (previously active, now disabled)

**Purpose**: Remove neurons with extremely high error magnitude (harmful
neurons).

**How it worked**:

1. Rust identified neurons with raw_error ≥ 10× max_output_error.
2. These neurons were presumed to be destabilising the network.
3. Removing them was predicted to improve the overall score.

**Current status**: 🔴 **DISABLED** (Issue #414)

Production data showed a 0% success rate (0 successes from 2 attempts). The
fundamental assumption was flawed:

**Root cause: High error ≠ harmful neuron**

A neuron with high recorded error is often:
1. **Handling difficult samples**: It's the only computation path for hard cases
2. **Receiving bad inputs**: The error is a symptom, not a cause
3. **Fighting incorrect biases**: It's compensating for problems elsewhere

Removing such neurons typically makes performance **worse** because:
- Difficult samples lose their only computation path
- The network loses the only neuron attempting to handle a specific pattern

Error magnitude measures how **wrong** the neuron's output is, not how
**harmful** the neuron is to the network's overall score. This is why predicted
improvements (based on error magnitude) did not match actual outcomes.

**Resolution**: This discovery type has been disabled. The legitimate
"remove-low-impact" discovery (based on activation_weighted_impact < costOfGrowth)
remains active with a 17.6% success rate.

**Output**: No longer emitted (disabled).

---

### Combo Successful

**Purpose**: Apply multiple individually-successful changes together for
compounding gains.

**How it works**:

1. Multiple candidates that passed individual ablation tests are combined.
2. The combined mutation is applied and re-scored.
3. If synergistic, the combo should improve score more than individual changes.

**Issue #415 Fix**: Interference detection was added to filter out incompatible
candidate pairs before they're proposed as coordinated candidates. The following
interference patterns are now detected and filtered:

1. **Conflicting Weights**: Two candidates targeting the same synapse with
   opposite sign weights (cancelling each other out).

2. **Saturation Risk**: Combined contributions that would push a target neuron
   into activation saturation, making the combined effect sub-additive.

3. **Redundant Contribution**: Two candidates with highly correlated activation
   patterns (≥90% correlation). These are redundant — adding both is no better
   than adding one with adjusted weight.

**Interference Detection Algorithm**:
- For each pair of candidate sources, compute Pearson correlation of activations
- If correlation ≥ 0.9, mark as redundant and filter out
- For saturating activation functions, check if combined contribution exceeds
  the saturation threshold (1.5×)
- Filter conflicting weight candidates (same source, opposite signs)

**Current status**: 🟡 Fixed — Interference filtering now prevents incompatible
combinations. The combo-successful discovery should only propose pairs that have
a reasonable chance of success (complementary activation patterns, no redundancy,
no saturation risk).

**Note**: The fix is in the Rust library's epistatic detection module. NEAT-AI
(TypeScript) may still need to be updated to take advantage of the improved
candidate filtering.

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
| **add-synapses** | 1 | 9 | 10.0% | 🟡 Fixed (#413) |
| **coordinated-structural** | — | — | — | 🟢 Active |
| **change-squash** | 2 | 9 | 18.2% | 🟡 Fixed (#417) |
| **remove-low-impact** | 65 | 304 | 17.6% | 🟢 Active |
| **remove-harmful-synapse** | — | — | — | 🟢 Active (#416) |
| **remove-neuron (high error)** | 0 | 2 | 0.0% | ⛔ Disabled (#414) |
| **dead-neuron-removal** | — | — | — | 🟢 Active |
| **redundant-path-pruning** | — | — | — | 🟢 Active |
| **combo-successful** | 0 | 8 | 0.0% | 🟡 Fixed (#415) |

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
   but previously very rarely suggested. Issue #417 addressed this by lowering
   detection thresholds and integrating proactive activation recommendations.

### What Needs Investigation

1. *(None currently — previous items addressed by Issues #413–#417.)*

### What is Not Working

1. **remove-neuron (high error)** — ⛔ **DISABLED (Issue #414)**. High error
   neurons are often handling difficult samples, not causing harm. Error
   magnitude does not translate to score impact.

### What Was Fixed

1. **combo-successful** — 🟡 **FIXED (Issue #415)**. Interference detection was
   added to filter out incompatible candidate pairs before proposing them as
   coordinated candidates. The fix detects three interference patterns:
   - Conflicting weights (opposite sign weights on same synapse)
   - Saturation risk (combined contributions exceeding activation bounds)
   - Redundant contribution (≥90% activation correlation)

2. **add-synapses** — 🟡 **FIXED (Issue #413)**. Predictions were inverting for
   targets with saturating activations. The weight was computed using a linear
   model but evaluated against a saturation-aware model, causing overshoot into
   the saturated region. The fix searches over multiple weight candidates for
   saturating targets, matching the approach already used by add-neuron
   candidates.

3. **remove-harmful-synapse** — 🟢 **FIXED (Issue #416)**. The harmful synapse
   detection was including ALL existing synapses without filtering, resulting
   in candidates with negative `expected_creature_score_gain`. NEAT-AI correctly
   filtered these out, but no candidates with positive expected gain were being
   generated because truly harmful synapses are relatively rare.

   **Root cause**: Missing threshold check in `src/analysis/implementation.rs`.
   The code was creating candidates for every synapse without checking if
   `neuron_error_improvement > 0.0`.

   **Fix**: Added a threshold check to only include candidates where removing
   the synapse would actually improve the score (positive expected gain).

4. **change-squash** — 🟡 **FIXED (Issue #417)**. The change-squash discovery
   type had an 18.2% success rate (highest among active types) but very low
   volume — only 11 total samples across all experiments.

   **Root cause**: Detection thresholds were too conservative. Saturation
   threshold (TANH) was 0.95, oscillation sign-change threshold was 0.30,
   and the proactive activation recommendation engine was not integrated.

   **Fix**: Three changes to increase candidate volume:
   - Lowered saturation thresholds (TANH: 0.95→0.85, LOGISTIC: 0.95/0.05→0.90/0.10,
     HARD_TANH: 0.99→0.95) to catch neurons approaching saturation
   - Lowered oscillation thresholds (sign change fraction: 0.30→0.15,
     minority sign fraction: 0.20→0.10) to catch milder oscillation
   - Integrated proactive activation recommendation engine (Issue #431) into
     the analysis pipeline to recommend activation changes based on input
     distribution analysis before problems occur

### Recommended Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| Done | Verify coordinated-structural implementation (Issue #337) | NEAT-AI implements all 7 operation types |
| Done | Disable remove-neuron (high error) (Issue #414) | 0% success rate; fundamental assumption flawed |
| Done | Add interference detection (Issue #415) | Filter incompatible pairs before combo-successful |
| Done | Fix harmful synapse threshold filtering (Issue #416) | Candidates with non-positive expected gain were included |
| Done | Fix add-synapses prediction inversion (Issue #413) | Saturation-aware weight search for bounded targets |
| Done | Increase change-squash suggestion rate (Issue #417) | Lowered thresholds, integrated proactive recommendations |
| Low | Optimise add-neurons variants | Already working, but room for improvement |

---

## Related Documentation

- [Impact Calculation](IMPACT_CALCULATION.md) — How neuron impact is computed
- [Analysis Deep Dive](ANALYSIS_DEEP_DIVE.md) — Detailed analysis workflow and
  detection algorithms
- [README](../README.md) — Main project documentation
