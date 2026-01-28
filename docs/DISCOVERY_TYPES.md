# 🧬 Discovery Types B

This document itemises all the discovery types used by NEAT-AI-Discovery and tracks their
success/failure rates in production.

> **Last updated**: 3 Jan 2026

## Table of Contents

- [Overview](#overview)
- [Discovery Type Summary](#discovery-type-summary)
- [Detailed Descriptions](#detailed-descriptions)
  - [add-neurons](#add-neurons)
  - [add-synapses](#add-synapses)
  - [coordinated-structural](#coordinated-structural)
  - [change-squash](#change-squash)
  - [remove-low-impact](#remove-low-impact)
  - [remove-harmful-synapse](#remove-harmful-synapse)
  - [remove-neuron](#remove-neuron)
  - [combo-successful](#combo-successful)
- [Analysis & Recommendations](#analysis--recommendations)

---

## Overview

Discovery types represent different mutation strategies that NEAT-AI-Discovery suggests
to improve a creature's score. The Rust library analyses recorded neuron activations and
errors to propose candidates, which are then validated by NEAT-AI through ablation testing.

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

| Discovery Type | Description | ✅ Success | ❌ Failure | Success Rate | Status |
|----------------|-------------|------------|------------|--------------|--------|
| **add-neurons** | Add a new hidden neuron between existing neurons | 556 | 8,944 | 5.9% | 🟢 Active |
| **add-synapses** | Add a new synapse connection | 1 | 9 | 10.0% | ⚠️ Low volume |
| **coordinated-structural** | Apply a *group* of dependent edits as a single candidate | — | — | — | 🟢 Active |
| **change-squash** | Change a neuron's activation function | 2 | 9 | 18.2% | ⚠️ Low volume |
| **remove-low-impact** | Remove neurons with activation_weighted_impact < costOfGrowth | 65 | 304 | 17.6% | 🟢 Active |
| **remove-harmful-synapse** | Remove synapses that increase error | — | — | — | 🟠 Not tested |
| **remove-neuron** | Remove harmful neurons (high error magnitude) | 0 | 2 | 0.0% | 🔴 Not working |
| **combo-successful** | Apply multiple successful changes together | 0 | 8 | 0.0% | 🔴 Not working |

**Total**: 624 successes / 9,276 failures (6.3% overall success rate)

### Status Legend

| Status | Meaning |
|--------|---------|
| 🟢 Active | Working and producing results |
| 🟠 Not tested | Rust produces candidates but NEAT-AI doesn't test them yet |
| 🔵 Not implemented | TypeScript expects this type but Rust doesn't produce it yet |
| ⚠️ Low volume | Working but rarely suggested |
| 🔴 Not working | Being tested but 0% success rate |

---

## Detailed Descriptions

### add-neurons

💡 **Purpose**: Add a new hidden neuron by inserting it between a source and target neuron.

**How it works**:
1. Rust analyses which (source → target) pairs would benefit from an intermediate computation
2. Suggests a new hidden neuron with:
   - Incoming synapse from source neuron
   - Outgoing synapse to target neuron
   - Squash function (activation function) from a candidate set
   - Bias value to shift the activation

**Example success**:
```json
{
  "changeType": "add-neurons",
  "description": "💡 Added neuron e54fe0ef -> ABSOLUTE -> 7035992a",
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

**Current status**: ✅ **This is our most successful discovery type** with 556 successes.

---

### add-synapses

🔗 **Purpose**: Add a new synapse connection between existing neurons.

**How it works**:
1. Rust analyses which neuron pairs would benefit from a direct connection
2. Suggests a weight for the new synapse based on correlation analysis
3. Target neurons get signals from the source that can reduce their error

**Example (failed)**:
```json
{
  "changeType": "add-synapses",
  "description": "🔗 Added helpful synapse input-1521 -> output-0",
  "scoreDelta": -0.000148,
  "expectedErrorReduction": 0.00016195967,
  "actualErrorReduction": -0.00014864216953747178
}
```

**Current status**: ⚠️ **Very low volume** – only 10 total samples. The one success shows
it can work, but predictions are inverting (predicting improvement but making it worse).

---

### coordinated-structural

🧩 **Purpose**: Apply a *group* of dependent edits as a single candidate.

Some beneficial structural changes are **epistatic**: no single add/remove/adjust operation improves fitness in isolation. Improvement occurs only when a set of structural edits are applied together (for example, removing a noisy input while increasing the trusted input weight).

This discovery type exists to escape neutral plateaus and handle interference cases where:
- A removal unlocks the benefit of an addition/adjustment
- A weight adjustment only helps once a competing path is removed
- Redundant paths mask each other’s error signal

**How it works (high level)**:
1. Rust proposes *atomic* edits (for example: remove harmful synapse, add helpful synapse, adjust existing synapse weight)
2. Rust *groups* compatible edits into a single candidate (a “grouped candidate”)
3. NEAT-AI evaluates the entire group in one ablation test (apply all ops to a clone, then rescore on the full training set)

**Example scenario (thermometer)**:
- Remove synapse from noisy mercury input
- Increase (or adjust) weight for digital thermometer input

**Candidate shape** (Rust JSON output from `analyze_parallel`):

```json
{
  "coordinatedStructuralCandidates": [
    {
      "expectedCreatureScoreGain": 0.0000123,
      "comment": "Coordinated: remove harmful synapse, adjust competing synapse weight",
      "operations": [
        { "type": "removeSynapse", "fromNeuronUuid": "input-10", "toNeuronUuid": "output-0" },
        { "type": "removeSynapse", "fromNeuronUuid": "input-11", "toNeuronUuid": "output-0" },
        { "type": "addSynapse", "fromNeuronUuid": "input-11", "toNeuronUuid": "output-0", "weight": 0.08 }
      ]
    }
  ]
}
```

**Operation vocabulary** (29-Jan-2026):
- `removeSynapse(fromNeuronUuid,toNeuronUuid)`
- `addSynapse(fromNeuronUuid,toNeuronUuid,weight)`
- `addNeuron(neuronUuid,neuronType,squash,bias,insertBeforeNeuronUuid?)`
- `removeNeuron(neuronUuid)`
- `changeSquash(neuronUuid,squash)`
- `setBias(neuronUuid,bias)`
- `setWeight(fromNeuronUuid,toNeuronUuid,weight)` — Issue #180: direct weight adjustment replacing the previous `removeSynapse` + `addSynapse` pattern

**Forward-only note**: for forward-only creatures, `addNeuron.insertBeforeNeuronUuid` is used to place the neuron in the `neurons[]` array before the target neuron so subsequent `addSynapse(newNeuron -> target)` respects the forward-only ordering constraint.

**Example scenario (replace synapse with hidden neuron)**:

```json
{
  "coordinatedStructuralCandidates": [
    {
      "expectedCreatureScoreGain": 0.00042,
      "comment": "Coordinated replacement: remove synapse and insert ReLU hidden neuron",
      "operations": [
        { "type": "removeSynapse", "fromNeuronUuid": "input-0", "toNeuronUuid": "output-0" },
        { "type": "addNeuron", "neuronUuid": "coordinated-hidden-deadbeef", "neuronType": "hidden", "squash": "ReLU", "bias": 0, "insertBeforeNeuronUuid": "output-0" },
        { "type": "addSynapse", "fromNeuronUuid": "input-0", "toNeuronUuid": "coordinated-hidden-deadbeef", "weight": 1.0 },
        { "type": "addSynapse", "fromNeuronUuid": "coordinated-hidden-deadbeef", "toNeuronUuid": "output-0", "weight": 0.9 }
      ]
    }
  ]
}
```

**Example scenario (collapse a 1-in/1-out hidden neuron)**:

```json
{
  "coordinatedStructuralCandidates": [
    {
      "expectedCreatureScoreGain": 0.00031,
      "comment": "Coordinated collapse: remove 1-in/1-out hidden neuron and add bypass synapse",
      "operations": [
        { "type": "removeSynapse", "fromNeuronUuid": "input-0", "toNeuronUuid": "hidden-0" },
        { "type": "removeSynapse", "fromNeuronUuid": "hidden-0", "toNeuronUuid": "output-0" },
        { "type": "removeNeuron", "neuronUuid": "hidden-0" },
        { "type": "addSynapse", "fromNeuronUuid": "input-0", "toNeuronUuid": "output-0", "weight": 1.0 }
      ]
    }
  ]
}
```

**Current status**: 🟢 **Active and tested** – Rust emits ordered groups; NEAT-AI applies the full ordered operation list atomically and re-scores on the full training set. All 7 operation types are implemented in NEAT-AI's `ApplyCoordinatedStructuralCandidate.ts` (verified Issue #337).

**Synergistic discovery** (Issue #189): Cross-neuron interactions (e.g., XOR-like patterns) are detected via residual analysis and emitted as coordinated candidates containing paired `addSynapse` operations. No new operation type is needed — NEAT-AI handles these through the existing coordinated-structural path.

---

### change-squash

🎨 **Purpose**: Change a neuron's activation function (squash) to one that better fits its role.

**How it works**:
1. Rust identifies neurons whose current squash function may not be optimal
2. Suggests alternative activation functions that could reduce error
3. Compares error reduction across different squash options

**Example success**:
```json
{
  "changeType": "change-squash",
  "description": "🎨 Changed activation function for 752308f5 (ELU -> ABSOLUTE)",
  "scoreDelta": 0.000010989105,
  "rustRequest": {
    "squashCandidate": {
      "previousSquash": "ELU",
      "squash": "ABSOLUTE",
      "improvedError": 862.75,
      "currentError": 921.71
    }
  }
}
```

**Current status**: ⚠️ **Very low volume** – only 11 total samples but 18.2% success rate
when suggested. Needs investigation into why it's rarely suggested.

---

### remove-low-impact

🪶 **Purpose**: Remove neurons that contribute less than the cost of their complexity.

**How it works**:
1. Rust computes each neuron's `activation_weighted_impact`
2. Neurons with impact < `costOfGrowth` (default: 1e-7) are candidates
3. Removing these neurons reduces complexity without meaningful accuracy loss

**Example success**:
```json
{
  "changeType": "remove-low-impact",
  "description": "🪶 Removed neuron 12141d6c (impact: 1.11e-10)",
  "scoreDelta": 1.3e-7,
  "rustRequest": {
    "removalCandidate": {
      "impact": 1.1132797e-10,
      "reason": "Impact 2.14e-11 < costOfGrowth (1.00e-7), 2 synapses, saves 1.20e-7"
    }
  }
}
```

**Current status**: ✅ **Working well** with 17.6% success rate. The savings from reducing
complexity often outweigh the minimal contribution these neurons provide.

---

### remove-harmful-synapse

🗑️ **Purpose**: Remove existing synapses that are actively increasing creature error.

**How it works**:
1. Rust analyses the correlation between synapse contributions and output error
2. Identifies synapses where removing the connection would reduce error
3. Returns `harmful_synapses` in the analysis result

**Rust output** (from `AnalyzeSynapsesResult`):
```rust
pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,  // ← This field
    // ...
}
```

**TypeScript interface** (from `DiscoverResult.ts`):
```typescript
interface DiscoverResult {
  removeHarmfulSynapse: CandidateSynapse | undefined;  // ← Expects single synapse
  // ...
}
```

**Current status**: 🟠 **Rust produces candidates, but no samples in discovery folder** –
The Rust library returns `harmful_synapses` in `AnalyzeParallelOutput`. NEAT-AI does
map `harmfulSynapses[0]` to `removeHarmfulSynapse` (see `DiscoverStructure.ts` line 1957-1959),
but applies a filter requiring `expectedCreatureScoreGain < 0` (line 1917-1919).

**Investigation needed**: 
- Are harmful synapses being filtered out by the score gain check?
- Is the ablation test for harmful synapses actually running?
- No samples appear in the discovery folder – are results not being recorded?

---

### remove-neuron

💀 **Purpose**: Remove neurons with extremely high error magnitude (harmful neurons).

**How it works**:
1. Rust identifies neurons with abnormally high error (e.g., 5.8e+17)
2. These neurons are presumed to be destabilising the network
3. Removing them is predicted to improve the overall score

**Example (failed)**:
```json
{
  "changeType": "remove-neuron",
  "description": "💀 Removed harmful neuron a002b568 (error: 5.80e+17)",
  "scoreDelta": -0.000030316980,
  "expectedErrorReduction": 0.4105461317300899,
  "actualErrorReduction": -0.000030536980
}
```

**Current status**: 🔴 **Not working** – 0 successes from 2 attempts. The massive discrepancy
between predicted improvement (0.41) and actual result (-0.00003) suggests the error
magnitude calculation may not translate to actual score improvement.

---

### combo-successful

🧬 **Purpose**: Apply multiple individually-successful changes together for compounding gains.

**How it works**:
1. Multiple candidates that passed individual ablation tests are combined
2. The combined mutation is applied and re-scored
3. If synergistic, the combo should improve score more than individual changes

**Example (failed)**:
```json
{
  "changeType": "combo-successful",
  "description": "🧬 Added 2 neurons",
  "scoreDelta": -2.07e-7,
  "actualCreatureChange": {
    "addedNeurons": [
      { "squash": "SOFTSIGN", "bias": 10 },
      { "squash": "ArcTan", "bias": 1 }
    ]
  }
}
```

**Current status**: 🔴 **Not working** – 0 successes from 8 attempts. Individual successes
may be interfering with each other when combined, or the combo testing methodology needs
review.

---

## Analysis & Recommendations

### What's Working ✅

1. **add-neurons** (5.9% success rate, 556 successes)
   - Our primary source of successful discoveries
   - The gentle nudge variants with tight outgoing weights perform well

2. **remove-low-impact** (17.6% success rate, 65 successes)
   - Reliable way to reduce complexity
   - Impact-weighted predictions are reasonably accurate

3. **change-squash** (18.2% success rate when suggested)
   - High success rate but very rarely suggested
   - Investigation needed: why aren't more squash changes being proposed?

### What Needs Investigation ⚠️

1. **add-synapses** – Predictions are inverting
   - Expected improvement: +0.00016
   - Actual result: -0.00015
   - Suggests the correlation analysis may be flawed or missing saturation effects

2. **change-squash** – Low suggestion rate
   - Only 11 total samples across all experiments
   - May need to broaden the conditions under which squash changes are suggested

### What's Broken 🔴

1. **remove-neuron** – Massive prediction errors
   - Error magnitude (5.8e+17) doesn't translate to score impact
   - Need to revisit the relationship between neuron error and creature score

2. **combo-successful** – Interference between changes
   - Individual successes don't combine well
   - May need stricter independence criteria or sequential application

### Not Implemented / Not Tested

1. **coordinated-structural** 🟢 – **Verified** (Issue #337): NEAT-AI implements all 7 operation types
   (`removeSynapse`, `addSynapse`, `addNeuron`, `removeNeuron`, `changeSquash`, `setBias`, `setWeight`)
   in `ApplyCoordinatedStructuralCandidate.ts`. Synergistic candidates (Issue #189) use existing
   `addSynapse` operations and require no additional NEAT-AI changes.

2. **remove-harmful-synapse** 🟠 – Rust produces, no samples recorded
   - Rust returns `harmful_synapses[]` array
   - TypeScript maps `harmfulSynapses[0]` → `removeHarmfulSynapse` ✓
   - Filter requires `expectedCreatureScoreGain < 0` – may be filtering all candidates
   - No samples in discovery folder – investigate why

### Recommended Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| ✅ Done | Verify coordinated-structural implementation (Issue #337) | NEAT-AI implements all 7 operation types; no gaps found |
| 🔴 High | Investigate why harmful_synapses aren't recorded | Mapping exists but no samples in discovery folder |
| 🔴 High | Investigate add-synapses prediction inversion | 10 samples show consistent wrong-direction predictions |
| 🔴 High | Disable or fix remove-neuron | 0% success rate, wasting validation cycles |
| 🟡 Medium | Review combo-successful strategy | 0% success rate, may be attempting incompatible combinations |
| 🟡 Medium | Investigate change-squash suggestion rate | 18.2% success rate but only 11 samples |
| 🟢 Low | Optimise add-neurons variants | Already working, but room for improvement |

---

## Related Documentation

- [Impact Calculation](IMPACT_CALCULATION.md) - How neuron impact is computed
- [README](../README.md) - Main project documentation

