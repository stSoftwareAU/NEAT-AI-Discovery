# Analysis Deep Dive

This document contains detailed analysis workflow information, discovery detection
algorithms, and implementation notes extracted from the main README. For a high-level
overview, see [README.md](../README.md).

---

## Analysis Workflow Details

- Call `analyze_parallel` with your chosen focus targets. Passing a single focus
  neuron where practical keeps diagnostics easy to map back to the Deno request
  and mirrors how NEAT-AI orchestrates discovery.
- The Rust side now refuses to run if `focus_neurons` is empty or contains
  duplicates. Controllers **must** validate and de-duplicate targets before
  calling into FFI so any upstream issues are surfaced promptly.
- For each focus target the Rust side enumerates **all** upstream neurons (every
  observation/input slot and every hidden neuron whose index precedes the
  target) that do **not** already have a synapse. This quickly grows into
  thousands of potential new synapses for realistic creatures (e.g. 1,486
  observations × 450+ hidden neurons).
- **GPU batching for improved utilisation (v0.1.118)**: Both helpful and harmful
  synapse analysis now batch multiple GPU operations into single command buffer
  submissions (batch size 512 default, 1024 on M4/high-perf GPUs). This reduces
  CPU-GPU round trips and keeps the GPU busy with larger workloads. Sample
  building is done on CPU in parallel to avoid GPU sync overhead per source.
- **TargetMap pre-building (v0.1.150)**: When analysing a focus neuron, the target
  HashMap (mapping obs_index to target data) is now built **once** and reused for
  all ~1000+ source neurons. Previously this HashMap was rebuilt for each source,
  causing significant CPU overhead. With 64 focus neurons, this eliminated ~64,000
  redundant HashMap constructions.
- **Centralised GPU work queue (v0.1.151)**: Instead of each parallel focus neuron
  thread creating its own GPU device (expensive ~100ms overhead per device), a
  single `GpuWorkQueue` is created and shared via Arc. The queue owns a dedicated
  GPU thread that processes all operations, eliminating device creation overhead.
  The `GpuEvaluator` trait allows helper functions to work with either direct
  `GpuAnalyzer` access or the shared queue. This improves GPU utilisation when
  processing many focus neurons in parallel.
- **Batch buffer mapping optimisation (v0.1.151)**: GPU buffer mapping now calls
  `map_async` on ALL staging buffers first, then performs a single `device.poll(Wait)`
  to wait for all buffers simultaneously. This reduces GPU-CPU round trips compared
  to the previous sequential mapping approach.
- **Sample locality grouping (Issue #221)**: When analysing multiple source neurons for
  the same target, sources with ≥80% obs_index overlap are grouped together. Instead of
  building samples separately for each source, samples are built once per group in a
  single pass through the target data. For typical creatures where input neurons share
  the same observation indices, this reduces sample building overhead by up to 100x
  (e.g., 100 sources with identical obs_indices → 1 group instead of 100 separate builds).
- The GPU kernels (helpful/harmful statistics) produce sufficient aggregates to
  derive the suggested weight and the expected error reduction. Results are sorted
  by expected improvement before being returned, so callers can simply read the
  first entry or pass `max_candidates=1` to receive the best.
- When no candidate "makes the grade" set `NEAT_AI_DISCOVERY_VERBOSE=1` before launching
  your Deno worker. The library will emit a single line per focus neuron that
  summarises why the top candidate was rejected and how many potential synapses
  were evaluated.
- The `analyze_parallel` JSON response also exposes a `diagnostics` array
  describing each focus neuron that finished without a candidate. These entries
  summarise the reason (no samples, below threshold, etc.) plus supporting
  counts so controllers can relay the explanation even when verbose logging is
  disabled.
- Optional production experiment (29-Dec-2025): If you are seeing a large volume of failed
  add-neuron candidates targeting hidden neurons, you can force **output-only** focus targets
  for add-neuron analysis by setting `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY=1`.
  This is intentionally opt-in for backwards compatibility; when enabled, hidden focus targets
  will be reported as `HiddenNeuronFiltered` in diagnostics.
- When an analysis deadline is supplied, discovery honours it **vertically**:
  focus neurons are processed in priority order and each neuron is analysed
  completely (including upstream candidates) where possible before moving to
  the next. If the timeout is reached mid-run you will still receive completed
  results for earlier focus neurons, and later targets may be skipped or only
  partially analysed.

---

## Coordinated Structural Discovery (Issue #165)

Some beneficial structural changes are **epistatic**: no single add/remove operation improves score in isolation, but a *group* of edits does. This often shows up on **neutral plateaus** where different parameterisations produce near-identical outputs, and the signal is in second-order effects (error variance, correlation, redundancy) rather than direct score gradients.

To support this, the Rust analysis can return **grouped candidates** via `coordinatedStructuralCandidates` (see [Issue #165](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/165)). Each entry is a single candidate that must be applied atomically (as a unit) during the controller-side ablation test.

- **Operations**: A group contains an `operations` array of atomic edits:
  - `removeSynapse` / `addSynapse` (with a `weight`)
  - `setWeight` (adjust an existing synapse's weight - Issue #180)
  - `addNeuron` (with deterministic `neuronUuid` so candidates are replayable)
  - `removeNeuron`
  - `changeSquash`
  - `setBias`
- **Weight changes**: Existing synapse weight adjustments are represented as a single `setWeight` operation (Issue #180), directly expressing the intent to modify the weight.
- **Candidate budgets**: `maxSynapseCandidates` is a **global cap** across `helpfulSynapses + harmfulSynapses + coordinatedStructuralCandidates`. If you set `maxSynapseCandidates: 0`, coordinated structural candidates will also be truncated to zero.

### Epistatic Neuron Pair Pre-Detection (Issue #202)

During synapse analysis, the library proactively detects **epistatic neuron pairs** - cases where two source neurons targeting the same output would provide better improvement when added together than either would alone. This addresses the "neutral plateau" problem where individual operations appear to have little benefit.

**Detection strategy**:
- **Complementary pattern detection**: Identifies source neurons with non-overlapping "firing" patterns (activation ≥ 0.5). When neuron A fires on one subset of samples and neuron B fires on a different subset, adding both synapses together covers more samples than either alone.
- **Combined improvement estimation**: For pairs with high complementarity (≥70% non-overlap), the library estimates the combined improvement as roughly the sum of individual improvements.

**When epistatic pairs are detected**:
- Both individual improvements are positive, AND
- Complementarity is ≥70% (firing patterns have low overlap), AND
- Combined improvement exceeds the best individual improvement

**Output**: Epistatic pair candidates appear as entries in `coordinatedStructuralCandidates` with two `addSynapse` operations and a comment indicating the epistatic relationship.

### Example: "noisy vs trusted" inputs (thermometer pattern)

If two inputs feed the same target with the same starting weight, but one input is much noisier (higher activation variance), a coordinated candidate may:

- remove the noisy synapse
- remove the trusted synapse
- add the trusted synapse back with a higher weight

This preserves (or improves) behaviour while reducing variance and redundancy, and avoids the "single edit looks bad" trap during ablation.

### Redundant Path Pruning with Renormalisation (Issue #164)

When two existing subnetworks (paths) feeding the same output compute effectively the same
thing, one can be pruned and the other's weight scaled to compensate. This reduces network
complexity without degrading fitness.

**Detection signals**:
- **Highly correlated activations** – Pearson correlation ≥ 0.85 between the two sources
- **Anti-correlated error gradients** – Both paths push error in the same direction
- **Shared downstream synapses** – Both sources feed the same target neuron

**How it works**:
1. For each target neuron, collect activation samples from all existing incoming synapses
2. Compute pairwise activation correlation between sources
3. If correlation ≥ 0.85, the weaker synapse (by absolute weight) is a prune candidate
4. The survivor's weight is renormalised to `keep_weight + prune_weight`

**Output**: Redundant path candidates appear in `coordinatedStructuralCandidates` with a
`removeSynapse` operation (for the pruned path) and a `setWeight` operation (for the
renormalised survivor). No new operation types are needed.

### Saturated Neuron Detection (Issue #342)

Neurons using bounded activation functions (e.g., TANH, LOGISTIC) can become saturated when
their input is consistently very large or very small. A TANH neuron with input always > 5
outputs ≈ 1.0 regardless of input variation, effectively becoming a constant. This blocks
useful signal propagation and wastes gradient capacity.

**Detection criteria**:
- **Activation near bounds**: For TANH, mean activation > 0.95 or < -0.95 across samples
- **Low relative variance**: Activation standard deviation < 0.05 (output doesn't vary)
- **Bounded activation**: Only bounded functions (TANH, LOGISTIC, HARD_TANH, etc.) can saturate
- **RELU dead-zone**: RELU neurons with all-zero output are also detected

**Supported activation functions**:

| Squash | Saturation Type | Threshold |
|--------|----------------|-----------|
| TANH | Ceiling/floor | \|mean\| > 0.95 |
| LOGISTIC | Ceiling/floor | mean > 0.95 or < 0.05 |
| HARD_TANH | Clamped | \|mean\| > 0.99 |
| RELU | Dead zone | mean ≈ 0, std ≈ 0 |
| SOFTSIGN, ISRU, ARCTAN | Ceiling/floor | \|mean\| > 0.95 |
| RELU6 | Ceiling/dead | mean > 5.9 or ≈ 0 |

**Recommended actions**:
1. **Change activation function**: Switch to IDENTITY to restore signal flow
2. **Adjust bias**: Shift bias to move the neuron's operating point away from saturation

**Output**: Saturation candidates appear in `coordinatedStructuralCandidates` with
`changeSquash` and/or `setBias` operations.

### Bottleneck Neuron Detection (Issue #343)

In evolved NEAT networks, structural mutations can create bottleneck neurons where many
input signals converge through a single hidden neuron before reaching outputs. This limits
the network's ability to represent complex input combinations because one neuron's activation
range must encode all upstream information.

**Detection criteria**:
- **High fan-in**: At least 3 incoming connections
- **High fan-in / fan-out ratio**: At least 2× more inputs than outputs
- **Error concentration**: Disproportionate share of output error traces through this neuron
- **Hidden neurons only**: Output neurons are natural convergence points and are excluded

**Bottleneck score** combines two components:
- **Topology score** (60%): Based on the fan-in / fan-out compression ratio
- **Error score** (40%): Based on the fraction of total error flowing through the neuron

**Recommended actions**:
1. **Add parallel neuron**: Create a new hidden neuron sharing a subset of inputs/outputs
   to increase capacity at the bottleneck
2. **Add bypass synapse**: Add a direct connection from an upstream neuron to a downstream
   neuron, reducing dependency on the bottleneck

**Output**: Bottleneck candidates appear in `coordinatedStructuralCandidates` with
`addNeuron` and/or `addSynapse` operations.

### Dead Neuron Detection (Issue #341)

As NEAT networks evolve, some neurons may become dead through weight changes that push
their inputs to always land in the zero region of their activation function (e.g., RELU
neurons that never receive positive input). Dead neurons consume GPU resources during both
training and inference without contributing useful information to the network's output.

**Detection criteria**:
- **Near-zero activation**: Mean absolute activation < 1e-6 across all samples
- **Zero variance**: Activation standard deviation ≈ 0 (always outputs the same value)
- **No meaningful activity**: Fewer than 1% of samples show activation above 0.01
- **Hidden neurons only**: Output and input neurons are excluded

**Removal confidence** combines:
- **Activation factor** (40%): How close mean absolute activation is to zero
- **Variance factor** (40%): How close standard deviation is to zero
- **Sample size factor** (20%): More samples increase confidence (plateaus at 1000)

**Output**: Dead neuron candidates appear in `coordinatedStructuralCandidates` with
`removeNeuron` operations.

### Correlated Error Pattern Detection (Issue #344)

When multiple output neurons consistently err in the same direction on the same samples,
it suggests a missing input feature or hidden representation that would benefit all of them.
Rather than treating each output independently and potentially creating redundant candidates,
this analysis identifies shared causes and recommends a single structural change.

**Detection method**:
1. **Error correlation matrix**: Pearson correlation of per-sample errors between all output pairs
2. **Complete-linkage clustering**: Groups outputs with pairwise correlation > 0.7
3. **Shared error samples**: Counts samples where all neurons in a group err in the same direction
4. **Predictive input identification**: Finds input neuron activations that predict the shared error

**Skip optimisation**: This analysis is skipped when there is only one output neuron, since
there is nothing to correlate.

**Output**: Correlated error groups appear in `coordinatedStructuralCandidates` with
`addNeuron` and `addSynapse` operations.

### Multi-Hop Candidate Analysis (Issue #230)

Current discovery considers single-hop improvements (adding one synapse or neuron). For deep
networks, multi-hop improvements (adding a path of 2-3 connections) may be more effective.
This analysis finds neurons whose activations correlate with a target's error but are not
directly connected, then recommends bypass synapses or relay neurons.

**Detection method**:
1. **Find correlated intermediates**: For each target neuron with errors, find neurons whose
   activation correlates (Pearson |r| ≥ 0.3) with the target's error but are not directly
   connected.
2. **Build two-hop paths**: intermediate → target bypass candidates.
3. **Extend to three-hop**: source → intermediate → target relay candidates, where the
   source's activation correlates with the intermediate's activation.
4. **Aggressive pruning**: Max 10 intermediates per target, max 50 total candidates,
   max 3 hops depth.

**Recommended actions**:
- **Add bypass synapse** (two-hop): Connect the correlated neuron directly to the target.
- **Add relay neuron** (three-hop): Insert a new hidden neuron along the path to relay
  information from source through to target.

**Output**: Multi-hop candidates appear in `coordinatedStructuralCandidates` with `addNeuron`
and/or `addSynapse` operations.

### Oscillating Neuron Detection (Issue #358)

Identifies hidden neurons whose activations frequently change sign across training samples.
An oscillating neuron is fighting between two contradictory functions — it activates
positively for some samples and negatively for others, with frequent sign changes. This
wastes representational capacity and can be stabilised by changing the activation function.

**Detection criteria**:
1. **Sign change fraction ≥ 0.3**: At least 30% of consecutive sample pairs show a sign change.
2. **Balanced signs**: Both positive and negative activations appear in substantial proportions
   (minority sign ≥ 20%).
3. **Meaningful magnitude**: Mean absolute activation ≥ 0.01 (distinguishes from dead neurons).
4. **Hidden neurons only**: Input and output neurons are excluded.

**Recommended actions**:
- **Change activation function**: Switch to ABSOLUTE (for symmetric activations like TANH) or
  RELU (for other functions) to stabilise the output sign.
- **Adjust bias**: Shift the operating point to favour the dominant sign direction.

**Output**: Oscillating neuron candidates appear in `coordinatedStructuralCandidates` with
`changeSquash` and optionally `setBias` operations.

### Dormant Synapse Detection (Issue #359)

Identifies synapses with near-zero weights that contribute negligible signal to their
target neuron. Dormant synapses waste computation during both forward pass and discovery
analysis without providing meaningful information flow.

**Detection criteria**:
1. **Near-zero weight**: Absolute weight < 1e-4.
2. **Low contribution**: Mean absolute contribution (|weight × source_activation|) < 1e-4.
3. **Not the sole connection**: The target neuron has other incoming synapses (removing the
   only input would be destructive).
4. **Sufficient samples**: At least 20 samples for statistical reliability.

**Output**: Dormant synapse candidates appear in `coordinatedStructuralCandidates` with
`removeSynapse` operations.

### Opposing Synapse Detection (Issue #360)

Identifies synapses whose contribution consistently works against error reduction. When a
synapse's contribution (weight × source_activation) correlates positively with the target
neuron's error, the synapse is actively hindering performance by pushing the output in the
wrong direction.

**Detection criteria**:
1. **Positive contribution–error correlation**: Pearson correlation ≥ 0.3 between the
   synapse's contribution and target error.
2. **Meaningful contribution**: Mean absolute contribution ≥ 0.01 (distinguishes from
   dormant synapses).
3. **Output targets only**: Only synapses targeting output neurons are analysed (where
   error is directly measured).
4. **Sufficient samples**: At least 20 matched sample pairs.

**Recommended actions**:
- **Remove the synapse** (correlation > 0.5): Eliminate the harmful connection entirely.
- **Flip the weight sign** (correlation 0.3–0.5): Negate the weight to reverse the
  harmful contribution direction.

**Output**: Opposing synapse candidates appear in `coordinatedStructuralCandidates` with
`removeSynapse` or `setWeight` operations.

### Output Bias Drift Detection (Issue #361)

Identifies output neurons with a consistent error sign bias — neurons whose errors are
predominantly positive (predicting too low) or predominantly negative (predicting too
high) across training samples. This systematic bias indicates the neuron's bias parameter
needs adjustment.

**Detection criteria**:
1. **Consistent error sign**: More than 70% of errors share the same sign.
2. **Meaningful mean error**: Absolute mean error ≥ 0.01 (not just noise).
3. **Output neurons only**: Hidden and input neurons are excluded.
4. **Sufficient samples**: At least 20 samples for statistical reliability.

**Output**: Bias drift candidates appear in `coordinatedStructuralCandidates` with
`setBias` operations.

### Candidate Clustering for Redundancy Reduction (Issue #224)

When discovery returns many similar candidates (e.g., multiple synapses from the same source
region targeting the same neuron), the controller would otherwise evaluate each independently,
wasting CPU on redundant ablation tests. Candidate clustering groups these into clusters so
the controller can test a representative first and skip the rest if it fails.

**Clustering criteria**:
1. **Same target neuron** (`toNeuronUuid`) — candidates must target the same neuron
2. **Same source type** (input vs hidden) — different neuron types have different signal
   characteristics and should not be mixed
3. **Similar improvement prediction** — candidates with very different expected improvements
   (>5× ratio) are split into separate sub-clusters

**JSON output**: Clusters appear in `candidateClusters` (optional field, omitted when empty):

```json
{
  "candidateClusters": [
    {
      "representativeFromUuid": "input-42",
      "representativeToUuid": "hidden-5",
      "representativeImprovement": 0.101,
      "memberCount": 5,
      "memberFromUuids": ["input-42", "input-43", "input-44", "input-45", "input-46"],
      "internalCorrelation": 0.92
    }
  ]
}
```

**TypeScript usage**:

```typescript
// Test representative first, skip cluster if it fails:
for (const cluster of candidateClusters) {
    const result = testCandidate(cluster.representativeFromUuid, cluster.representativeToUuid);
    if (!result.improved) {
        // Representative failed — skip remaining members (high correlation)
        console.log(`Skipping ${cluster.memberCount - 1} similar candidates`);
    }
}
```

**Backward compatible**: The `candidateClusters` field is optional and only present when
clusters are detected. Existing consumers that do not read this field continue to work
unchanged.

---

## Detection Module Reference

The library contains 38+ detection and recommendation modules, grouped by concern.
Each module follows the same pipeline: load records → detect pattern → convert to
coordinated candidates.

### Activation & Neuron State Modules

These modules detect issues with how neurons process activations.

#### Bimodal Neuron Detection (Issue #640)

Detects hidden neurons whose pre-activation distribution is bimodal or multimodal.
Such neurons effectively serve two distinct input regimes, limiting representational
capacity. The module analyses the histogram of pre-activation values, identifies
distinct modes, and recommends splitting the neuron.

**Algorithm**:
1. Compute a histogram of pre-activation values for each hidden neuron.
2. Identify local maxima (modes) in the histogram.
3. If two or more modes are sufficiently separated relative to their widths,
   flag the neuron as bimodal.
4. Propose an `addNeuron` candidate to split the neuron's two regimes.

#### Restricted Range Detection (Issue #399)

Detects hidden neurons confined to a narrow sub-range of their activation function's
output domain. A TANH neuron only producing values in [0.2, 0.4] wastes most of its
representational capacity. The module compares the observed output range to the
theoretical range of the activation function.

**Algorithm**:
1. Compute min/max activation for each hidden neuron.
2. Compute the theoretical output range for the neuron's squash function.
3. If the observed range is less than a configured fraction of the theoretical
   range, flag as restricted.
4. Propose `changeSquash`, `setBias`, or `setWeight` candidates to expand the
   effective operating range.

#### Operating Point Analysis (Issue #401)

Analyses hidden neuron pre-activation distributions against the squash function's
active zone. Detects neurons whose operating point is shifted away from the dynamic
region, resulting in underutilisation of gradient capacity.

**Algorithm**:
1. Compute the mean and standard deviation of pre-activation values.
2. Determine the squash function's "active zone" (region of maximum gradient).
3. If the mean pre-activation is significantly outside the active zone, flag
   the neuron.
4. Propose `setBias` or `setWeight` candidates to shift the operating point.

#### Activation Mismatch Detection (Issue #543)

Detects neurons with poorly matched activation functions. Examples include RELU
neurons with negative bias (gating out useful signal) and bounded activations
receiving inputs that never reach the active region.

**Algorithm**:
1. For each hidden neuron, analyse the relationship between the squash function
   and the observed pre-activation distribution.
2. Flag mismatches: e.g., RELU with negative bias where most inputs are negative,
   bounded activations where inputs cluster far from the active zone.
3. Propose `changeSquash` or `setBias` to correct the mismatch.

#### Monotonicity Detection (Issue #643)

Detects hidden neurons with non-monotonic activation–error relationships. When
activations increase but errors both improve and worsen depending on the region,
the neuron is serving conflicting purposes.

**Algorithm**:
1. Sort samples by activation value and partition into bins.
2. Compute mean error for each bin.
3. Check whether the error curve is monotonically increasing or decreasing.
4. If the curve reverses direction, flag as non-monotonic.
5. Propose `addNeuron` (split) or `changeSquash` (reshape) candidates.

#### Error Plateau Detection (Issue #545)

Detects output neurons stuck in error stagnation — high mean error combined with
low error variance — indicating a local minimum plateau.

**Algorithm**:
1. Compute mean and variance of error for each output neuron.
2. Flag neurons where mean absolute error exceeds a threshold AND error variance
   is below a separate threshold (consistent, significant error).
3. Propose `changeSquash` or `setBias` to escape the local minimum by altering
   the error surface.

#### Output Range Compression Detection (Issue #645)

Detects output neurons operating in a compressed sub-range of their activation
function's domain. An output using TANH but only producing values in [0.1, 0.3]
wastes most of its output precision.

**Algorithm**:
1. Compute the observed output range for each output neuron.
2. Compare against the activation function's full theoretical range.
3. If the compression ratio exceeds a threshold, flag the neuron.
4. Propose `changeSquash` to switch to a function matching the actual range.

#### Output Squash Mismatch Detection (Issue #545)

Detects when output neurons use activation functions mismatched to their target
data range. For example, using HARD_TANH (range [-1, 1]) when targets lie in
[0, 1] (LOGISTIC would be more appropriate).

**Algorithm**:
1. Analyse the target data distribution for each output neuron.
2. Determine the ideal output range from the target distribution.
3. Compare against the current squash function's output range.
4. If mismatched, propose `changeSquash` to a function whose range matches.

#### Bias Perturbation Detection (Issue #551)

Identifies neurons in suboptimal activation regimes and recommends large bias
shifts to escape local minima. Unlike gradient-based bias adjustments, this
proposes regime-shifting perturbations.

**Algorithm**:
1. Analyse the neuron's operating regime relative to its activation function.
2. Identify if the neuron is stuck in a low-gradient region (e.g., deep
   saturation) where normal training cannot escape.
3. Compute candidate bias values that would shift the neuron to a different
   regime (e.g., from saturated to linear region).
4. Propose `setBias` with the computed perturbation value.

#### Squash + Weight Rescale Detection (Issue #548)

Coordinates activation function changes with compensating weight adjustments.
When changing a neuron's squash function, downstream weights must be rescaled to
maintain equivalent signal magnitude.

**Algorithm**:
1. Identify neurons where a squash change would be beneficial.
2. Compute the scale factor between the old and new activation function's
   output ranges.
3. Generate a coordinated candidate with `changeSquash` and `setWeight`
   operations applied atomically to preserve the operating point.

### Weight & Synapse Modules

These modules detect issues with synapse weights and connections.

#### Weight Coherence Validation (Issue #437)

Part of the "Brilliant but Brittle" initiative. Contains three sub-detectors:

**Incoherent weight ratios**: Flags neurons where the largest incoming weight
magnitude is orders of magnitude larger than the smallest, meaning some inputs
are effectively ignored.

**Near-constant paths**: Flags neuron paths where the product of incoming and
outgoing weights is so small that the neuron contributes a near-constant signal
regardless of input.

**Symmetric cancellation**: Flags pairs of synapses with nearly equal magnitude
but opposite signs feeding the same target, cancelling each other's contribution.

#### Weight Magnitude Reset Detection (Issue #550)

Identifies synapses stuck in local weight minima. Generates exploratory
`setWeight` candidates with large magnitude changes (double, halve, or negate)
to jump to a different region of the loss surface.

**Algorithm**:
1. For each synapse, compute the local gradient and check if the weight has
   been stable despite ongoing error.
2. If the gradient is small but error remains, the weight may be in a local
   minimum.
3. Generate multiple candidate weights at different magnitudes to explore
   the loss surface.

#### Weight Polarity Flip Detection (Issue #644)

Detects synapses where the gradient sign is consistently opposite to the weight
sign. Rather than waiting for many small gradient descent steps to cross zero,
recommends direct sign inversion.

**Algorithm**:
1. Compute the mean gradient for each synapse across samples.
2. Compare the gradient sign to the current weight sign.
3. If they consistently disagree (e.g., positive weight but negative gradient),
   propose flipping the weight sign via `setWeight`.

#### Fan-in Polarity Conflict Detection (Issue #641)

Identifies hidden neurons with incoming synapses having conflicting polarities.
A mix of strong positive and strong negative incoming weights indicates the neuron
is trying to combine contradictory signals.

**Algorithm**:
1. For each hidden neuron, partition incoming synapses into positive and
   negative weight groups.
2. Compute the aggregate magnitude of each group.
3. If both groups have significant magnitude and partially cancel, flag the
   conflict.
4. Propose splitting the conflicting paths via `addNeuron` and `addSynapse`.

### Structural & Topology Modules

These modules detect structural and topological issues.

#### Topology Diversification Detection (Issue #549)

Detects when the network topology is too simple for the problem complexity.
Assesses global network capacity (unlike bottleneck detection which finds local
convergence points).

**Algorithm**:
1. Compute the network's topological complexity (number of hidden neurons,
   layers, connectivity density).
2. Estimate the problem complexity from the error distribution.
3. If the network is too simple relative to the remaining error, propose
   adding neurons to increase representational capacity.

#### Skip Connection Detection (Issue #570)

Analyses topological depth and gradient attenuation to identify deep hidden
neurons that would benefit from residual-style skip connections.

**Algorithm**:
1. Compute shortest path length from each hidden neuron to the nearest output
   via reverse BFS.
2. Estimate gradient attenuation along the path (product of activation
   derivatives and weight magnitudes).
3. For deep neurons with significant error and no existing shortcut, propose
   an `addSynapse` skip connection directly to an output.

#### Symmetry Breaking Detection (Issue #569)

Identifies pairs of hidden neurons with near-identical weight configurations
and activation functions. Symmetric neurons waste capacity by computing the
same function.

**Algorithm**:
1. For each pair of hidden neurons, compare incoming and outgoing weight
   vectors.
2. Compute weight similarity (cosine similarity or Euclidean distance).
3. If similarity exceeds a threshold and both use the same squash function,
   flag as symmetric.
4. Propose breaking symmetry via `setBias`, `setWeight`, or `changeSquash`
   on one of the pair.

#### Co-Adaptation Detection (Issue #571)

Identifies pairs of hidden neurons with highly correlated activations,
indicating functional redundancy even when weight configurations differ.

**Algorithm**:
1. Compute Pearson correlation between activation patterns for each pair
   of hidden neurons across training samples.
2. If correlation ≥ 0.9, flag the pair as co-adapted.
3. Propose removing the weaker neuron and rescaling the survivor's weights
   via `removeNeuron` and `setWeight`.

#### Output Conflict Detection (Issue #639)

Identifies hidden neurons with conflicting per-output error contributions.
The neuron helps some outputs while simultaneously hurting others.

**Algorithm**:
1. For each hidden neuron, disaggregate its error contribution per output
   neuron.
2. Compute the sign of the error contribution for each output.
3. If the signs conflict (positive for some outputs, negative for others),
   flag the neuron.
4. Propose `addSynapse` or `addNeuron` to specialise the neuron's role.

#### Hard Sample Cluster Detection (Issue #642)

Identifies observation groups consistently high-error across all outputs.
Finds which input features discriminate hard from easy observations.

**Algorithm**:
1. Compute per-observation error across all output neurons.
2. Identify observations with above-average error across most outputs.
3. Cluster hard observations by input feature similarity.
4. For each cluster, identify discriminative input features (inputs whose
   values differ significantly between hard and easy observations).
5. Propose `addNeuron` and `addSynapse` targeting the discriminative features.

### Range & Input Analysis Modules

These modules analyse input ranges and gating.

#### Bounded Range Detection (Issue #395)

Detects input and hidden neurons with sentinel value clusters at activation
boundaries. Many datasets use special values (-1, 0, etc.) for missing data,
which distort the neuron's effective operating range.

**Algorithm**:
1. Compute a histogram of activation values for each input/hidden neuron.
2. Identify boundary clusters (values clustered at min or max).
3. If a boundary cluster contains more than a threshold fraction of samples,
   flag as bounded range with sentinels.
4. Propose an `addNeuron` gating structure to suppress sentinel values.

#### Sentinel Gating Detection (Issue #400)

Detects input neurons where sentinel values actively degrade performance.
Proposes gated neuron structures using STEP activation to mask sentinel regions.

**Algorithm**:
1. Identify sentinel value clusters in input neuron activations.
2. Compare error rates for sentinel vs non-sentinel samples.
3. If sentinel samples have significantly higher error, propose a STEP-gated
   neuron that outputs 0 for sentinel values and passes through data values.

#### Observation Utilisation Detection (Issue #543)

Detects underutilised input neurons with low effective range. Builds on
observation range analysis to identify inputs not contributing meaningful
information.

**Algorithm**:
1. Compute the effective range of each input (excluding sentinel clusters).
2. Compare effective range to total observed range.
3. If the utilisation ratio is below a threshold, propose gating neurons
   to normalise or amplify the useful signal.

#### Input Sensitivity Detection (Issue #435)

Part of the "Brilliant but Brittle" initiative. Analyses prediction
sensitivity to input changes via two sub-detectors:

**Dominant input detection**: Identifies inputs with excessive leverage —
a single input's contribution dominates the output, creating brittleness.

**Threshold effect detection**: Identifies operating points near activation
function thresholds where small input changes cause disproportionate output
swings.

### Recommendation & Scoring Modules

#### Sample-Weighted Discovery (Issue #423)

Prioritises high-error samples during discovery by weighting each sample
proportionally to its absolute error magnitude. Ensures difficult samples
receive proportional attention rather than being averaged away.

---

## Discrete Activation Function Handling

The standard discovery algorithm uses a **linear error model** to predict improvement:

```
expected_improvement ≈ (2×w×Σ(error×activation) - w²×Σ(activation²)) / Σ(error²)
```

This formula assumes the relationship between a neuron's input and error is
**continuous and differentiable**. For neurons with **discrete or saturating
activation functions**, this model fails because:

1. Small input changes either do **nothing** (if threshold not crossed)
2. Or cause a **binary flip** (massive discrete output change)
3. Or are in a flat/saturated region where the gradient is zero

### Threshold-crossing model for STEP/BIPOLAR

**STEP** and **BIPOLAR** neurons now use a specialised **threshold-crossing model**
instead of the standard linear error model:

| Activation | Output | Threshold Model |
|------------|--------|-----------------|
| **STEP** | 0 or 1 | Counts samples where adding a connection would flip the output in the helpful direction |
| **BIPOLAR** | -1 or 1 | Same approach, accounting for the -1/1 output range |

The threshold-crossing model:
- Examines each sample's target value (pre-activation input sum)
- Predicts which samples would cross the 0-threshold if we add a new connection
- Counts "helpful flips" (error-reducing) vs "harmful flips" (error-increasing)
- Returns candidates where net helpful flips exceed the improvement threshold

### HARD_TANH saturation-aware model

For **HARD_TANH** target neurons, the library uses a **saturation-aware model**
instead of the linear approximation. This is critical for accurate predictions
because HARD_TANH clamps outputs to [-1, 1]:

| Scenario | Linear Model | HARD_TANH Model | Difference |
|----------|--------------|-----------------|------------|
| **Near saturation** (value=0.9, error=0.1) | **-125%** (overshoots!) | **+100%** (saturates at 1.0) | 225% |
| **Already saturated** (value=1.5, error=-0.2) | **+94%** (thinks it helps) | **0%** (still saturated) | 94% |

The saturation-aware model:
- Uses the target neuron's pre-activation value (input sum before clamping)
- Computes `new_output = clamp(value + contribution, -1, 1)`
- Calculates error reduction against the actual clamped output

### GPU-accelerated target activation simulation

The library performs GPU-accelerated sample matching to build candidate evaluation
datasets. The GPU matching shader passes through **both** `target_value` (pre-activation
input sum) and `target_activation` (post-squash output) for each matched sample. This
enables accurate activation function simulation for the following target neuron types:

| Activation | Simulation | Why It Matters |
|------------|------------|----------------|
| **HARD_TANH** | Saturation-aware | Avoids overprediction near ±1 clamp boundaries |
| **TANH** | Saturation-aware | Gradual saturation at extremes |
| **LOGISTIC** | Saturation-aware | Asymptotic bounds at 0 and 1 |
| **ReLU** | Threshold-aware | Zero output for negative inputs |
| **LeakyReLU** | Threshold-aware | Different slopes for positive/negative |
| **BIPOLAR** | Discrete | Binary -1/+1 output |
| **CLIPPED** | Saturation-aware | Hard clamp at ±1 |

For these activations, the library computes the actual new error after applying
the candidate contribution through the target's activation function, rather than
using the linear approximation.

**Linear fallback**: If `target_value` or `target_activation` data is missing
(e.g., older Parquet files), the library falls back to the linear model.

### All other activations

All other activation functions (including IDENTITY, INVERSE, IF, MAXIMUM,
MINIMUM, ReLU6, Softplus, GELU, SELU, ELU, etc.) use the **standard linear
error model**. No activations are skipped.

The discovery process treats source neurons as **black boxes** - we don't care
how they computed their activations, only what the values are. For any target
neuron, we look at:

1. **Observed errors** on the target (how wrong is the output?)
2. **Observed activations** from potential source neurons
3. **Correlation** between them (when source is high, is error positive?)

### Split-error ReLU evaluation (complementary pairs)

When target errors are split roughly 50/50 between positive (output should be higher)
and negative (output should be lower), no single ReLU can improve all samples.
Discovery evaluates **complementary ReLU pairs**:

| Evaluation | Weight Computed From | Net Improvement Computed From |
|------------|---------------------|------------------------------|
| **Positive-error ReLU** | Samples with error > 0 | **ALL samples** |
| **Negative-error ReLU** | Samples with error < 0 | **ALL samples** |

**CRITICAL**: The optimal weight is computed from the target subset (to find the right
direction), but the **net improvement is computed across ALL samples**. This is essential
because a ReLU that helps positive-error samples may harm negative-error samples.

The candidate map uses a key that includes:
`(source_uuid, target_uuid, squash, sign(incoming_weight), sign(outgoing_weight))`

This ensures complementary pairs are kept as separate entries.

### Bias-aware neuron improvement calculation

When evaluating neuron candidates (add-neurons), the **bias parameter** is critical
for accurate improvement predictions. The bias shifts the activation threshold:

| Bias | Effect | Samples Affected |
|------|--------|-----------------|
| bias > 0 | Shifts threshold left | More samples activate the neuron |
| bias = 0 | Default threshold | Only positive pre-activation values activate |
| bias < 0 | Shifts threshold right | Fewer samples activate the neuron |

The improvement calculation includes the proposed bias when evaluating neuron
candidates.

### Sensible parameter ranges (add-neurons)

As a production guard rail, add-neuron candidates are only returned when
their parameters are within sensible bounds:

- **incomingWeight**: |w| ≤ 20
- **bias**: |b| ≤ 10
- **outgoingWeight**: |w| ≤ 0.1 (already clamped by the optimiser)

### IDENTITY neuron filtering

**IDENTITY neurons with bias ≈ 0 are redundant** because they're mathematically
equivalent to a direct synapse. Discovery filters out these candidates:

1. **Bias filtering**: IDENTITY candidates with `|bias| < 0.01` are rejected
2. **Use synapse analysis**: Direct connections should use `add-synapses`, not `add-neurons`

### Add-neuron target neuron filtering

**Output and hidden neurons are valid targets** for add-neuron analysis. Input
and constant neurons are filtered out from the focus list:

| Neuron Type | Filtered? | Reason | Diagnostic Code |
|-------------|-----------|--------|-----------------|
| **output** | No | Direct impact on creature score | (not filtered) |
| **hidden** | No | Analysed with impact-based discounting (v0.1.123) | (not filtered) |
| **input** | Yes | Observation sources, not computation nodes | `input_neuron_filtered` |
| **constant** | Yes | Don't receive inputs - always output fixed value | `constant_neuron_filtered` |

---

## Error Distribution Analysis (Issue #192)

The library computes comprehensive error distribution statistics for target neurons, enabling
targeted discovery for specific error patterns like outliers, bimodal distributions, and
error clusters.

**Distribution statistics included in metadata:**

| Statistic | Description |
|-----------|-------------|
| `mean` | Average error across samples |
| `stdDev` | Standard deviation of errors |
| `variance` | Variance of errors |
| `skewness` | Asymmetry indicator (positive = right-tailed outliers) |
| `kurtosis` | Tail heaviness (> 3 = heavy tails, outliers likely) |
| `percentiles` | [p10, p25, p50, p75, p90] values |
| `min`, `max` | Error range |
| `iqr` | Interquartile range (p75 - p25) |
| `sampleCount` | Number of samples analysed |

**How to interpret:**

- **High skewness** (> 0.5): Outlier samples with high error exist
- **High kurtosis** (> 4): Distribution has heavy tails (more extreme values)
- **Large IQR** relative to mean: High variability in errors

**Configuration:**

```bash
# Enable outlier-focused analysis (off by default)
export NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS=1

# Set the percentile threshold for outlier identification (default: 90)
export NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE=90
```

---

## Tiered Loading Strategy (Issue #215)

The library automatically selects the optimal loading strategy based on file size
and available system memory.

**Loading Strategies:**

| Strategy | When Selected | Behaviour |
|----------|---------------|-----------|
| **PreloadAll** | Estimated expanded < available_memory ÷ 4 | Loads entire file upfront (fastest access) |
| **LruCache** | Expanded fits in memory but exceeds 1/4 | Per-neuron caching with LRU eviction |
| **Streaming** | Expanded exceeds available memory | Block-based loading (lowest memory) |

**How it works:**

1. The library estimates expanded memory = file_size × 3 (decompression ratio)
2. Compares against available system memory
3. Automatically selects the best strategy

**LRU Cache Benefits:**

- **Bounded memory**: Uses half of available memory as cache capacity
- **Per-neuron caching**: More efficient than block-based for focus neuron analysis
- **Smart eviction**: Least-recently-used neurons are evicted when capacity exceeded
- **Thread-safe**: Supports concurrent access during parallel analysis

**Example file size thresholds (8GB system):**

| File Size | Expanded Size | Strategy |
|-----------|---------------|----------|
| 100 MB    | 300 MB        | PreloadAll (< 2GB = 8GB ÷ 4) |
| 500 MB    | 1.5 GB        | LruCache (< 8GB but > 2GB) |
| 3 GB      | 9 GB          | Streaming (> 8GB available) |

**API Usage:**

```rust
// Automatic strategy selection (recommended)
let cache = RecordCache::new_tiered("data.parquet")?;

// Or use the TieredRecordCache directly
let cache = TieredRecordCache::new("data.parquet")?;

// Force LRU mode with specific capacity
let cache = TieredRecordCache::new_with_memory_limit("data.parquet", 4 * 1024 * 1024 * 1024)?;
```
