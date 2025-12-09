# NEAT-AI-Discovery

A high-performance Rust companion library for
[`stSoftwareAU/NEAT-AI`](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend structural upgrades (new synapses or neurons) that reduce
error. Controllers call into the library via Deno FFI to power
`Creature.discoveryDir()` workflows.

## Why use this library?

- **Production-ready discovery** – Handles millions of observations without the
  memory blow-outs that limit the TypeScript implementation.
- **Single-file artefacts** – Writes per-run Parquet files so results are easy to
  transfer, archive, or inspect with standard tooling.
- **Drop-in for NEAT-AI** – Exposes the `libneat_ai_discovery` symbol set expected
  by the TypeScript bindings in `NEAT-AI`.

## GPU Requirement

**This library requires a GPU.** There is no CPU fallback. If no compatible GPU is
available, discovery is simply skipped – NEAT-AI continues training without the
discovery phase. This is by design:

- **Simplicity**: One code path means fewer bugs. No subtle differences between
  CPU and GPU implementations.
- **Performance**: GPU-accelerated analysis is the whole point. A CPU fallback
  would be too slow to be useful.
- **Optional feature**: Discovery is an optimisation, not a requirement. NEAT-AI
  works fine without it.

## Quick start

1. Install prerequisites (`rustup`, `cargo`, build tools, and `jq`). The
   `scripts/runlib.sh` helper will guide you if anything is missing.
2. Build and install the library using `runlib.sh`:

   ```bash
   ./scripts/runlib.sh
   ```

   This script:
   - Installs Rust and Cargo if missing (no sudo required)
   - Builds the library in release mode
   - Installs it to `~/.cargo/lib/` with version tracking
   - Signs it on macOS for FFI compatibility

   **From NEAT-AI directory**, you can call this script directly:

   ```bash
   ../NEAT-AI-Discovery/scripts/runlib.sh
   ```

3. Confirm the artefact exists at `~/.cargo/lib/libneat_ai_discovery.*`.
4. Run the quality gate before committing:

   ```bash
   ./quality.sh
   ```

## Deployment Checklist

Before committing code changes, ensure you complete the following steps:

1. **Run quality checks in both repositories:**
   ```bash
   # In NEAT-AI-Discovery
   ./quality.sh
   
   # In NEAT-AI
   cd ../NEAT-AI
   ./quality.sh
   ```

2. **Increment version numbers:**
   - **NEAT-AI-Discovery**: Update `Cargo.toml` version field (e.g., `0.1.41` → `0.1.42`)
   - **NEAT-AI**: Update `deno.json` version field (e.g., `0.204.1` → `0.204.2`)

3. **Verify all tests pass** in both repositories before committing.

These steps ensure code quality, proper versioning, and that all tests pass before deployment.

## Using the library with NEAT-AI

1. Place the compiled artefact where Deno can load it:
   - Copy `libneat_ai_discovery.*` into `~/.cargo/lib`, **or**
   - Export `NEAT_AI_DISCOVERY_LIB_PATH=/absolute/path/to/libneat_ai_discovery.*`.
2. Grant FFI permissions when running discovery jobs:
   ```bash
   deno run --allow-env --allow-ffi --allow-read your-script.ts
   ```
3. From your controller, guard calls with
   `isRustDiscoveryEnabled()` so the job fails fast if the module cannot be
   loaded.
4. Follow the end-to-end discovery orchestration documented in the
   [`DiscoveryDir` guide](https://github.com/stSoftwareAU/NEAT-AI/blob/main/docs/DiscoveryDir.md).
   The guide covers safe-write practices, worker loops, and how to persist the
   improved creatures that this library exports.

## Analysis workflow expectations

### Discovery → Evolution Pipeline

The discovery process works as follows:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RUST (this library)                                                        │
│  ─────────────────────                                                      │
│  1. Find ALL candidates with positive expected improvement                  │
│  2. Apply impact discounting (creature-level predictions)                   │
│  3. Sort by expected improvement (best first)                               │
│  4. Return candidates (optionally limited by max_candidates)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TYPESCRIPT (NEAT-AI)                                                       │
│  ────────────────────                                                       │
│  1. Receive candidates from Rust (e.g., 100 candidates)                     │
│  2. Select top N based on available CPUs (e.g., 10-20)                      │
│  3. Re-score each candidate IN PARALLEL (apply mutation, measure score)     │
│  4. Keep candidates that ACTUALLY improve the creature's score              │
│  5. Return improved creatures to population                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVOLUTION                                                                  │
│  ─────────                                                                  │
│  • Improved creatures compete in the population                             │
│  • Natural selection breeds out unsuccessful mutations                      │
│  • No manual filtering needed - evolution handles it                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Rust finds structural improvements that reduce error. TypeScript
validates by measuring actual score. Evolution does the rest. No arbitrary thresholds
or manual filtering - just physics and natural selection.

### Detailed workflow

- Call `analyze_synapses` once per focused neuron where practical. Passing a
  single `focus_neurons` entry keeps diagnostics easy to map back to the Deno
  request and mirrors how NEAT-AI orchestrates discovery.
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
  submissions (batch size 512). This reduces CPU-GPU round trips and keeps the
  GPU busy with larger workloads. Sample building is done on CPU in parallel to
  avoid GPU sync overhead per source.
- The GPU kernels (helpful/harmful statistics) produce sufficient aggregates to
  derive the suggested weight and the expected error reduction. Results are sorted
  by expected improvement before being returned, so callers can simply read the
  first entry or pass `max_candidates=1` to receive the best.
- When no candidate “makes the grade” (e.g. there were no overlapping samples,
  the GPU observed zero consistent improvements, or every candidate fell under
  the requested threshold) set `NEAT_AI_DISCOVERY_VERBOSE=1` before launching
  your Deno worker. The library will emit a single line per focus neuron that
  summarises why the top candidate was rejected and how many potential synapses
  were evaluated.
- The `analyze_synapses` and `analyze_neurons` JSON responses also expose a
  `diagnostics` array describing each focus neuron that finished without a
  candidate. These entries summarise the reason (no samples, below threshold,
  etc.) plus supporting counts so controllers can relay the explanation even
  when verbose logging is disabled.
- When an analysis deadline is supplied, discovery honours it **vertically**:
  focus neurons are processed in priority order and each neuron is analysed
  completely (including upstream candidates) where possible before moving to
  the next. If the timeout is reached mid-run you will still receive completed
  results for earlier focus neurons, and later targets may be skipped or only
  partially analysed.

### Discrete activation function handling

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

#### Threshold-crossing model for STEP/BIPOLAR

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

This allows discovery to find meaningful improvements for STEP/BIPOLAR neurons
by proposing connections that flip the output to the correct state on more samples.

#### HARD_TANH saturation-aware model

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

This ensures predictions match actual results when the candidate is applied,
which is essential for production systems where HARD_TANH is commonly used.

#### GPU-accelerated target activation simulation

The library performs GPU-accelerated sample matching to build candidate evaluation
datasets. As of v0.1.114, the GPU matching shader passes through **both**
`target_value` (pre-activation input sum) and `target_activation` (post-squash
output) for each matched sample. This enables accurate activation function
simulation for the following target neuron types:

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
using the linear approximation. This is verified by unit tests:
`sample_matching_preserves_target_value_and_activation` and
`sample_matching_enables_target_activation_simulation`.

**Linear fallback**: If `target_value` or `target_activation` data is missing
(e.g., older Parquet files from before this feature), the library falls back to
the linear model. The linear model works reasonably well when errors are small
relative to the activation function's linear region.

#### Bias-aware weight calculation (v0.1.115)

For **add-neuron** candidates, the optimal outgoing weight must be computed using
the new neuron's **actual activation pattern** (which includes bias). Previously,
the weight was computed without bias, then a separate bias optimisation was
performed. This caused predictions to fail when bias significantly shifted the
activation threshold.

**Example failure scenario (now fixed):**
- New TANH neuron with `bias=1`
- Without bias: `TANH(x)` fires when x > 0 (~50% of samples)
- With bias: `TANH(x+1)` fires when x > -1 (almost always!)
- The optimal weight for these two patterns is completely different

**The fix**: After finding the optimal bias, the library now **recomputes** the
optimal outgoing weight using the actual activation pattern (with bias). This
ensures predictions match reality.

This is verified by unit tests: `add_neuron_weight_must_include_bias_in_calculation`
and integration test: `test_add_neuron_with_hard_tanh_target_uses_bias_aware_weight`.

#### Tighter outgoing weight clamp (v0.1.138)

**CRITICAL IMPROVEMENT**: Analysis of 2030 failed add-neuron candidates vs ~22 successful
discoveries revealed that outgoing weights were being computed far too large.

**Successful discoveries (survived evolution):**
- |outgoing_weight|: 0.00002 to 0.03 (all < 0.05)
- incoming/outgoing ratio: 71x to 104,000x
- Example: incoming=100, outgoing=-0.00096 (ratio 104,000x)

**Failed discoveries:**
- 36% had |outgoing_weight| > 0.05 (up to 50!)
- Many had ratio < 10x (even 1:1)
- Previous clamp: [-10.0, 10.0] was far too loose

**The fix**: Three-part improvement to weight calculation:

1. **Tighter outgoing weight clamp**: Changed from `[-10.0, 10.0]` to `[-0.1, 0.1]`.
   New neurons should contribute a SMALL correction, not dominate the network.

2. **Weight ratio validation**: For add-neuron candidates where `incoming_weight > 1.0`,
   we now validate that `incoming/outgoing >= 50`. Candidates with nearly equal incoming
   and outgoing weights are rejected as unreliable predictions.

3. **Shared weight function**: Created `calculate_optimal_outgoing_weight()` to ensure
   consistent weight calculation across add-synapse and add-neuron analysis (DRY).

**Also fixed**: Split-error ReLU evaluation was computing bias from the error subset only,
which could produce large positive biases that made the ReLU fire for ALL samples (defeating
the purpose of split-error). Now uses bias=0 for split-error ReLU candidates.

**Expected impact**: ~36% of failed candidates (with |outgoing_weight| > 0.05) will now
produce tighter, more accurate predictions. The remaining candidates may still fail due
to other factors (sample overfitting, bias-weight interaction, activation saturation)
which can be addressed in follow-up improvements.

#### Expanded activation functions and discrete weight fix (v0.1.139)

**MAJOR FEATURE**: Added 8 new activation functions based on analysis of successful
discoveries. Many successful neurons evolved TO activations we weren't trying!

| New Activation | Evidence |
|----------------|----------|
| **LeakyReLU** | 4 successful discoveries evolved ReLU → LeakyReLU! |
| **Mish** | 2 successful discoveries evolved TO Mish (from ELU, Softplus) |
| **Swish** | 1 successful discovery evolved ReLU → Swish |
| **HARD_TANH** | 1 successful discovery evolved CLIPPED → HARD_TANH |
| **SOFTSIGN** | Successful discovery neuron with SOFTSIGN |
| **BENT_IDENTITY** | 1 successful discovery evolved LeakyReLU → BENT_IDENTITY |
| **ArcTan** | Similar to SOFTSIGN, bounded output |
| **ReLU6** | Capped ReLU, useful for bounded outputs |

Total activations now: **19** (was 11).

**Philosophy change**: The goal is finding MORE successful candidates, not filtering
out failures. Failed candidates are excluded after evaluation anyway. "Kiss more frogs
to find more princes."

**BUG FIX**: Discrete evaluation generating huge outgoing weights

The `evaluate_discrete_candidate` function (for STEP/BIPOLAR targets with IDENTITY neurons)
was generating outgoing weights up to ±50, far exceeding `MAX_OUTGOING_WEIGHT` (0.1).

| Before | After |
|--------|-------|
| OUTGOING_SCALES: [0.1..50.0] | OUTGOING_SCALES: [0.01..0.1] |

**Production evidence**: 455 out of 793 large-weight failed candidates were IDENTITY neurons
from this code path. None produced real improvements.

#### Prediction tracing and validation (v0.1.140)

**INVESTIGATION**: With ~100k samples, predictions should be accurate. Production data shows
predictions are inverted (~84% in wrong direction). Added tools to investigate.

**Finding from synthetic tests**: The prediction formula is **mathematically correct**!
All 6 synthetic tests pass with predictions matching manual simulation to within 0.01%.
This means the issue is in **sample collection or interpretation**, not the formula.

| Test Scenario | Predicted | Manual | Match? |
|---------------|-----------|--------|--------|
| Linear region | 75.00% | 75.00% | ✓ |
| Near saturation | 100.00% | 100.00% | ✓ |
| Negative error | 75.00% | 75.00% | ✓ |
| Mixed errors | 7.10% | 7.10% | ✓ |
| TypeScript simulation | 66.38% | 66.38% | ✓ |

**New feature**: Prediction tracing for debugging. Set environment variable:
```bash
export NEAT_AI_DISCOVERY_TRACE_PREDICTION=1
```

This logs sample-level details showing:
- Input parameters (weights, bias, sample count)
- First 5 samples with detailed calculation breakdown
- Contribution statistics (average, positive/negative counts)
- Final improvement calculation

**Next steps**: The investigation suggests recording more data in TypeScript to understand
why production samples produce inverted predictions despite correct formula.

#### VALUE domain error interpretation (v0.1.117)

**CRITICAL BUG FIX**: The NEAT-AI TypeScript library stores errors in the **VALUE
domain** (pre-activation), not the ACTIVATION domain (post-squash). This affects
how the Rust library interprets and uses error data for improvement predictions.

**TypeScript error calculation (NEAT-AI `Neuron.record()`):**
```typescript
const targetValue = unSquash(desiredActivation);  // Convert desired output to pre-activation
const error = targetValue - currentValue;         // VALUE domain error
```

**Previous (incorrect) Rust interpretation:**
```rust
// WRONG: Treated error as activation domain
let expected = target_activation + avg_error;  // Mixing ACTIVATION + VALUE domains!
```

**Corrected Rust interpretation (v0.1.117):**
```rust
// CORRECT: Error is in VALUE domain, so compute expected via squash
let desired_value = target_value + avg_error;
let expected = squash(desired_value);  // Convert to ACTIVATION domain
```

**Why this matters for saturation:**

| Scenario | Current Value | Error (VALUE) | Old Formula | Correct Formula |
|----------|---------------|---------------|-------------|-----------------|
| Near saturation | 0.8 | 0.3 | `expected = 0.8 + 0.3 = 1.1` | `expected = clamp(1.1) = 1.0` |
| In saturation | 1.5 | -1.0 | `expected = 1.0 + (-1.0) = 0.0` | `expected = clamp(0.5) = 0.5` |

The old formula produced incorrect `expected` values when the target neuron was
near or in saturation, causing predictions to be wildly inaccurate.

**Fixed locations:**
- `compute_net_improvement_new` (HARD_TANH model)
- `compute_activation_improvement_and_count` (all 4 activation paths)
- `compute_synapse_improvement_with_target_squash`
- `count_improved_samples_with_target_squash`

This fix ensures predictions match actual results when candidates are applied,
resolving the "add-neuron candidates always fail" production issue.

#### ACTIVATION domain consistency (v0.1.120)

**CRITICAL BUG FIX**: When using target activation function simulation (to handle
saturation in HARD_TANH, TANH, etc.), the improvement calculation was comparing
errors from **different domains**:

- **Baseline error**: VALUE domain (`avg_error²`)
- **New error**: ACTIVATION domain (`(expected - new_output)²`)

Near saturation, VALUE domain errors are much larger than ACTIVATION domain errors
(because the activation function compresses them). This caused **massive
overprediction** of improvements.

**Example of the bug:**

| Value | Computation | Result |
|-------|-------------|--------|
| target_value | Pre-activation input | 0.9 |
| avg_error | VALUE domain error | 0.3 |
| desired_value | target_value + avg_error | 1.2 |
| expected | HARD_TANH(1.2) | 1.0 (saturated) |
| target_activation | Current output | 0.9 |
| contribution | Weight × new_neuron_output | 0.05 |
| new_input | target_value + contribution | 0.95 |
| new_output | HARD_TANH(0.95) | 0.95 |

**Buggy calculation (mixed domains):**
- Baseline error² = 0.3² = 0.09 (VALUE domain)
- New error² = (1.0 - 0.95)² = 0.0025 (ACTIVATION domain)
- Improvement = (0.09 - 0.0025) / 0.09 = **97%** ❌

**Correct calculation (consistent ACTIVATION domain):**
- Baseline error² = (1.0 - 0.9)² = 0.01 (ACTIVATION domain)
- New error² = (1.0 - 0.95)² = 0.0025 (ACTIVATION domain)
- Improvement = (0.01 - 0.0025) / 0.01 = **75%** ✓

The fix computes **both baseline and new error** in the same domain (ACTIVATION
when simulating, VALUE for linear approximation). This is verified by
`improvement_calculation_uses_consistent_domains`.

**Fixed functions:**
- `compute_synapse_improvement_and_count`
- `compute_relu_improvement_and_count`
- `compute_activation_improvement_and_count`

#### Split-error evaluation for all activations (v0.1.135)

**BUG FIX**: When target neuron errors are split ~50/50 between positive and negative,
the standard linear model would predict small positive improvements that were actually
negative in practice. This caused systematic prediction failures for add-neuron candidates.

**Root cause**: Computing optimal weight from ALL samples averages out when errors
are balanced. The model predicts +0.08% but actual result is -0.08% because helping
one group hurts the other equally.

**Fix**: Extended ReLU's split-error handling to ALL activations:
1. Split samples by error sign (positive vs negative)
2. For EACH subset, compute optimal weight from that subset
3. Evaluate NET improvement across ALL samples
4. Only return candidates where net improvement > 0

**Result**: `expected_improvement_percentage` is now the TRUE net improvement across
all samples, not just a subset prediction. Candidates that would hurt one group more
than they help the other are filtered out automatically.

**Test added**: `tests/split_error_all_activations.rs` verifies the fix.

#### Split-error fallback candidate fix (v0.1.136)

**BUG FIX #1**: The split-error evaluation introduced in v0.1.135 had a threshold bug that
broke the fallback mechanism. Candidates with small positive improvements (below threshold)
were silently dropped instead of being returned as fallbacks.

**Root cause**: `evaluate_activation_for_subset` initialised `best_net_improvement` to
`threshold`, meaning candidates with `0 < improvement <= threshold` failed the comparison
check and were never returned.

**Fix**: Changed `best_net_improvement` initialisation from `threshold` to `0.0`.

**BUG FIX #2**: When split-error evaluation was attempted (both positive and negative error
subsets had enough samples) but found NO candidates with positive net improvement, the code
incorrectly fell back to all-samples evaluation. This produced unreliable small-improvement
predictions (~0.05%) that consistently failed in production.

**Root cause**: When errors are truly split ~50/50 AND source activations don't correlate
with error sign, there's NO good weight. Any weight helps one group but hurts the other
equally. Split-error correctly rejects these candidates. But all-samples would then compute
a weak weight (due to error cancellation) and return small positive predictions that were
within the model's error margin - essentially noise.

**Fix**: Track whether split-error evaluation was properly attempted. If both subsets had
enough samples but NEITHER produced candidates, return None instead of falling through
to all-samples. The all-samples fallback is now ONLY used when errors aren't clearly split
(e.g., all positive, all negative, or one subset too small).

**Test added**: `tests/split_error_fallback_candidates.rs` verifies the fix.

#### Simplified candidate filtering (v0.1.134)

**SIMPLIFICATION**: Removed all arbitrary percentage thresholds. The creature's score
is the **only** measure that matters.

**Rust's job**:
1. Find ALL candidates with positive expected error reduction
2. Apply impact discounting (convert to creature-level predictions)
3. Sort by expected improvement (best first)
4. Return candidates to TypeScript

**TypeScript's job**:
1. Select top N candidates based on available CPUs
2. Apply each candidate mutation and measure ACTUAL score change
3. Keep candidates that improve the score
4. Return improved creatures to the population

**Evolution's job**:
- Successful mutations compete in the population
- Unsuccessful mutations get bred out naturally
- No manual filtering needed

**Why no thresholds?** Previous versions had arbitrary thresholds (2%, 0.1%) that
filtered candidates before TypeScript could evaluate them. This was wrong:
- The cost of growth is ~1e-7 per neuron, ~1e-8 per synapse
- Any measurable improvement easily exceeds this cost
- The old 2% threshold was **10,000x too aggressive**
- Candidates that looked "too small" in Rust could still improve the actual score

**Current behaviour**: Return everything positive. Let TypeScript measure. Let evolution decide.

#### Hidden neuron impact discounting (v0.1.123)

**FEATURE**: Hidden neurons are valid targets for add-neuron and add-synapse analysis.
Predictions are discounted by the neuron's impact score to give creature-level
expected improvements.

**How it works**:
- **Output neurons**: Impact = 1.0 (direct contribution to score). No discount applied.
- **Hidden neurons**: Impact = path weight product to outputs. Predictions are
  discounted by impact factor.

For a hidden neuron with impact 0.5:
- Raw predicted improvement: 10%
- Discounted improvement: 10% × 0.5 = 5%

This discounting ensures hidden neuron predictions reflect their actual contribution
to the creature's score based on their position in the network topology.

#### All other activations

All other activation functions (including IDENTITY, INVERSE, IF, MAXIMUM,
MINIMUM, ReLU6, Softplus, GELU, SELU, ELU, etc.) use the **standard linear
error model**. No activations are skipped.

Some of these (IDENTITY, INVERSE) are mathematically linear, so the linear model
is exact. For others (Softplus, GELU, SELU, ELU), the linear model is a
reasonable approximation when the target neuron isn't near saturation. The model
may over- or under-predict improvement, but typically finds useful candidates.

The discovery process treats source neurons as **black boxes** - we don't care
how they computed their activations, only what the values are. For any target
neuron, we look at:

1. **Observed errors** on the target (how wrong is the output?)
2. **Observed activations** from potential source neurons
3. **Correlation** between them (when source is high, is error positive?)

#### Split-error ReLU evaluation (complementary pairs)

When target errors are split roughly 50/50 between positive (output should be higher)
and negative (output should be lower), no single ReLU can improve all samples.
Discovery evaluates **complementary ReLU pairs**:

| Evaluation | Weight Computed From | Net Improvement Computed From |
|------------|---------------------|------------------------------|
| **Positive-error ReLU** | Samples with error > 0 | **ALL samples** |
| **Negative-error ReLU** | Samples with error < 0 | **ALL samples** |

**CRITICAL**: The optimal weight is computed from the target subset (to find the right
direction), but the **net improvement is computed across ALL samples**. This is essential
because a ReLU that helps positive-error samples may harm negative-error samples:

- When source neurons fire on both positive and negative error samples, adding a ReLU
  will push the output in one direction for ALL samples
- The improvement on the target subset may be cancelled (or exceeded) by harm to the
  other subset
- The true expected improvement is `(baseline_sq - new_error_sq) / baseline_sq` computed
  over the entire dataset

Candidates are only returned if the **net improvement across ALL samples** exceeds the
threshold. This ensures predictions match actual results when the candidate is applied.

The candidate map uses a key that includes:
`(source_uuid, target_uuid, squash, sign(incoming_weight), sign(outgoing_weight))`

This ensures complementary pairs are kept as separate entries:
- Different ReLU orientations (`incoming_weight` ±1) don't collide
- Split-error pairs (same `incoming_weight`, opposite `outgoing_weight`) don't collide

This correlation analysis works regardless of the target's activation function.
The linear model is an approximation for ALL non-linear functions - it may be
more or less accurate depending on the function, but it finds useful patterns.

#### Bias-aware neuron improvement calculation

When evaluating neuron candidates (add-neurons), the **bias parameter** is critical
for accurate improvement predictions. The bias shifts the activation threshold:

| Bias | Effect | Samples Affected |
|------|--------|-----------------|
| bias > 0 | Shifts threshold left | More samples activate the neuron |
| bias = 0 | Default threshold | Only positive pre-activation values activate |
| bias < 0 | Shifts threshold right | Fewer samples activate the neuron |

For example, with a ReLU neuron:
- Without bias: `ReLU(1.0 × activation)` only fires when activation > 0
- With bias=0.5: `ReLU(1.0 × activation + 0.5)` fires when activation > -0.5

The improvement calculation now includes the proposed bias when evaluating neuron
candidates. This ensures the predicted improvement matches the actual improvement
when the neuron is applied with its computed bias value.

#### IDENTITY neuron filtering

**IDENTITY neurons with bias ≈ 0 are redundant** because they're mathematically
equivalent to a direct synapse:

```
IDENTITY(input × incoming_weight + 0) × outgoing_weight = input × incoming × outgoing
```

This is just a synapse with `weight = incoming_weight × outgoing_weight`. Discovery
now filters out these candidates:

1. **Minimum improvement threshold**: IDENTITY requires at least 5% improvement
2. **Bias filtering**: IDENTITY candidates with `|bias| < 0.01` are rejected
3. **Use synapse analysis**: Direct connections should use `add-synapses`, not `add-neurons`

#### Add-neuron target neuron filtering

**Output and hidden neurons are valid targets** for add-neuron analysis. Input
and constant neurons are filtered out from the focus list:

| Neuron Type | Filtered? | Reason | Diagnostic Code |
|-------------|-----------|--------|-----------------|
| **output** | No | Direct impact on creature score | (not filtered) |
| **hidden** | No | Analysed with impact-based discounting (v0.1.123) | (not filtered) |
| **input** | Yes | Observation sources, not computation nodes | `input_neuron_filtered` |
| **constant** | Yes | Don't receive inputs - always output fixed value | `constant_neuron_filtered` |

This filtering occurs before analysis begins. The diagnostics response includes
the appropriate reason code for each filtered neuron, so callers know why a
focus neuron received no candidates.

**Post-analysis filtering** (v0.1.125): Hidden neurons that have candidates
found but ALL candidates are filtered by impact discounting (below 2% after
discount) receive the diagnostic code `impact_discounted_below_threshold`. This
ensures focus neurons never silently disappear from the response.

#### Impact calculation fix (v0.1.126)

**CRITICAL BUG FIX**: The neuron impact calculation was severely underestimating
impact by normalising weights. This caused ~75% of "low-impact" removal
candidates to actually INCREASE error when removed.

**The bug**: Impact was computed as `weight / total_inbound × child_impact` which
gave the "fraction of downstream's input from this neuron" instead of the actual
contribution to output.

**Example of the bug**:
- Neuron A → Target (weight 0.001), Other → Target (weight 100)
- Old (normalised): impact = 0.001 / 100.001 × 1.0 ≈ **1e-5**
- New (absolute): impact = 0.001 × 1.0 = **0.001**

The normalised formula underestimated by **100x** in this case! For deep networks
with many competing inputs at each layer, the underestimation compounds to
**1000x or more**.

**Production evidence**: Neurons with calculated impact 1e-10 to 1e-17 caused
score deltas of 1e-5 to 1e-2 when removed - off by 5-15 orders of magnitude.

**The fix**: Impact now uses absolute weight products along paths to outputs:
```
impact = weight × downstream_impact
```

This matches the actual contribution: `activation × weight × downstream_impact`.

#### Dynamic removal threshold based on synapse counts (v0.1.127)

**BUG FIX**: After the v0.1.126 impact calculation fix, ZERO removal candidates
were being found. The fix implements a dynamic threshold based on NEAT-AI's
actual Score.ts complexity formula.

**The issue**: The removal candidate threshold was a static value that didn't
account for the complexity savings from removing the neuron's synapses.

**NEAT-AI's Score.ts formula** (the authoritative source):
```typescript
const complexityPenalty = hiddenNeuronCount * growthCost +
    creature.synapses.length * growthCost / 10 +
    penalty * growthCost / 100;
```

**So removing a neuron with N incoming and M outgoing synapses saves:**
```
savings = growthCost × (1 + (N + M) / 10)
```

**The fix**: ALL neurons with `activation_weighted_impact < costOfGrowth` (1e-7) are
returned as removal candidates, sorted by impact ascending.

```
activation_weighted_impact = structural_impact × mean_absolute_activation
```

Where:
- `structural_impact` = NORMALISED impact through the network
- `mean_absolute_activation` = sum(|finite activation|) / finite_record_count

**Note**: Non-finite activation values (NaN, Infinity) are filtered out when computing
`mean_absolute_activation` to prevent corruption of the removal candidate ranking.

**Normalised impact calculation**:

For each synapse from neuron A to target B with weight w:
```
A's contribution to B = |w| / total_inbound_to_B × B's_impact
```

This is **recursive** - if B has 100 inputs, A only contributes 1/100th of B's signal.

#### Squash-aware impact calculation (v0.1.132)

**ENHANCEMENT**: The impact calculation is now **squash-aware**. Different squash
functions use different impact formulas to avoid underestimating impact.

| Squash Category | Functions | Impact Formula | Rationale |
|-----------------|-----------|----------------|-----------|
| **Linear** | IDENTITY, TANH, LOGISTIC, etc. | `\|w\| / total_inbound × child` | Sum of weighted inputs |
| **Threshold** | STEP, BIPOLAR | `child_impact` (full, not normalised) | Any synapse can flip output |
| **Selection** | MINIMUM, MAXIMUM, IF | `child_impact / N` (equal probability) | Only one synapse "wins" |

**Why this matters**:

- **STEP/BIPOLAR**: A tiny weight (1e-8) feeding into a STEP neuron could flip the
  output from 0→1 if the neuron is near its threshold. The old sum-based formula
  would calculate impact ≈ 0, but the actual effect could be 1.0!

- **MINIMUM/MAXIMUM**: The old formula gave large weights high impact in MINIMUM
  (~90%), but small weights are actually more likely to win! The new formula gives
  each synapse equal probability (1/N).

For detailed explanation with diagrams, see [Impact Calculation](docs/IMPACT_CALCULATION.md).

**Example**: Output has 107 incoming synapses with total |weight| = 343.
A neuron with weight 3.0 to output contributes: `3.0 / 343 ≈ 0.9%` of output.

This correctly captures that removing a neuron with many competing inputs
has a small effect on the downstream signal.

**Removal candidate JSON response** now includes:
- `incomingSynapses` / `outgoingSynapses`: synapse counts used in calculation
- `removalSavings`: the raw savings value from NEAT-AI formula
- Candidates sorted by activation_weighted_impact ascending (lowest first = safest to remove)

The `calculate_removal_savings(incoming, outgoing, growth_cost)` function is
available for use in other analyses and is tested against the NEAT-AI formula.

If verbose logging is enabled (`NEAT_AI_DISCOVERY_VERBOSE=1`), you'll see
messages like:

```
[NEAT-AI-Discovery][verbose] Using threshold-crossing model for 2 STEP/BIPOLAR neurons: [...]
```

#### Synapse candidate impact discounting (v0.1.133)

**IMPORTANT**: The `expectedImprovementPercentage` field in synapse candidates is now
**creature-level**, not neuron-level. This makes Rust the **single source of truth**
for expected improvement calculations.

| Candidate Type | Impact Discounted? | TypeScript Action |
|----------------|-------------------|-------------------|
| **Neurons** | ✅ Yes (v0.1.123) | Use value directly |
| **Synapses** | ✅ Yes (v0.1.133) | Use value directly |
| **Removal** | ✅ Yes (built-in) | Use value directly |

**How it works**:
- Synapse candidates targeting **output neurons** have impact = 1.0 (no discount)
- Synapse candidates targeting **hidden neurons** are discounted by the target's impact
  score (0.0 to 1.0 based on weighted paths to outputs)

**Example**: A synapse candidate improving a hidden neuron by 70% that has impact 0.1:
- **Old (neuron-level)**: `expectedImprovementPercentage = 0.70` (70%)
- **New (creature-level)**: `expectedImprovementPercentage = 0.07` (7%)

**TypeScript should NOT re-calculate impact**. The returned `expectedImprovementPercentage`
is the actual expected improvement on the creature's score. Simply use:
```typescript
const creatureLevelImprovement = candidate.expectedImprovementPercentage;
// Don't multiply by getNeuronShare() or any other impact factor!
```

## Verifying the installation

Use the NEAT-AI helper script after copying the library:

```bash
cd /path/to/NEAT-AI
./scripts/check_discovery.ts
```

If the script reports that discovery is enabled, you are ready to schedule
`Creature.discoveryDir()` jobs against your sampled datasets. Otherwise revisit
`NEAT_AI_DISCOVERY_LIB_PATH` and the permissions passed to `deno run`.

### Checking for a usable GPU from NEAT-AI

Discovery **requires a GPU** – there is no CPU fallback. On machines without a
suitable GPU, controllers must disable discovery entirely. This is intentional:
the library has one code path (GPU) to avoid bugs from divergent implementations.

The library exposes a lightweight FFI entry point to allow NEAT-AI to decide
whether discovery should be enabled:

- **Symbol**: `check_gpu_available`
- **Input**: no arguments (the function takes no parameters)
- **Output**: JSON string:

  ```json
  {
    "success": true,
    "gpuAvailable": true,
    "reason": null
  }
  ```

  When GPU is unavailable, the response includes a diagnostic reason:

  ```json
  {
    "success": true,
    "gpuAvailable": false,
    "reason": "No GPU adapter found. Discovery disabled on this machine..."
  }
  ```

- When `"gpuAvailable"` is `false`, controllers should treat discovery as
  disabled on that worker, in the same way discovery is disabled when the
  Rust FFI module cannot be loaded (for example, when `--allow-ffi` is
  missing).
- When `"gpuAvailable"` is `true`, controllers may safely schedule discovery
  jobs. If a later GPU initialisation error occurs, the Rust side will return
  a structured error and mark the JSON `success` flag as `false`.

#### Platform-specific GPU behaviour

- **macOS**: GPU (Metal) should always be available. If `gpuAvailable` is
  `false`, this is treated as an error (`success: false`) indicating a system
  configuration issue that should be investigated.
- **Linux**: GPU may not be available on headless servers without GPU hardware
  or without proper permissions to access `/dev/dri` devices. If `gpuAvailable`
  is `false`, this is **not** an error (`success: true`) - discovery is simply
  disabled on that machine. This is normal for older headless Linux servers.
  
  **Note:** On Linux, the library only probes the Vulkan backend (not OpenGL/EGL)
  to avoid panics from EGL initialisation errors on systems without proper GPU
  drivers. This is intentional - old hardware without Vulkan support will simply
  have discovery disabled rather than causing crashes.

## Troubleshooting

- **Library not found**: Double-check the artefact path, file extension (e.g.
  `.dylib` on macOS, `.so` on Linux), and `NEAT_AI_DISCOVERY_LIB_PATH`.
- **FFI permission errors**: Ensure discovery workers launch with
  `--allow-ffi --allow-env --allow-read --allow-write` and only point to trusted
  library locations.
- **Empty Parquet output**: Confirm the caller supplies the sampled discovery
  dataset and that each record bundles observations, activations, and errors for
  the same training index.
- **XDG_RUNTIME_DIR warnings on Linux**: The library automatically sets
  `XDG_RUNTIME_DIR` to a temporary directory if it's not already set. This is
  required by wgpu (WebGPU) on Linux systems using Wayland. The warnings are
  harmless and the library handles this automatically. On macOS, this variable
  is not needed.
- **EGL/DRI permission denied warnings on Linux**: If you see warnings like
  `libEGL warning: failed to open /dev/dri/renderD128: Permission denied` or
  similar for `/dev/dri/card0`, the user running the process needs access to
  the GPU device nodes. These warnings typically appear when wgpu probes for
  available GPU backends.
  
  **Solutions (choose one):**
  1. **Add user to the render/video groups** (recommended for dedicated GPU
     access):
     ```bash
     sudo usermod -a -G render $USER
     sudo usermod -a -G video $USER
     # Log out and back in for group changes to take effect
     ```
  2. **Set device permissions** (temporary fix):
     ```bash
     sudo chmod 666 /dev/dri/renderD128 /dev/dri/card0
     ```
  3. **Suppress warnings** (if wgpu finds an alternative backend and discovery
     still works): Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to suppress Mesa/libEGL
     debug output. This sets `EGL_LOG_LEVEL=fatal` and `MESA_DEBUG=silent`
     internally before GPU initialisation.
  
  **Diagnosing GPU access:**
  ```bash
  # Check which groups own the DRI devices
  ls -la /dev/dri/
  # Check your current groups
  groups
  # Test GPU availability directly
  vulkaninfo --summary 2>/dev/null || echo "Vulkan not available"
  ```
  
  If the warnings appear but discovery still proceeds successfully (you see
  "Training ... with N binary file" after the warnings), wgpu has found an
  alternative GPU backend and the warnings can be safely ignored.
- **Out of memory errors (exit code 137)**: Exit code 137 indicates the process
  was killed by the Linux OOM (Out of Memory) killer (128 + SIGKILL). This
  commonly occurs when `--max-old-space-size` exceeds available system RAM.
  
  **For heterogeneous environments** (old Linux servers to new Mac M4 Pro):
  
  ```bash
  # Detect available memory and set V8 heap appropriately
  # Linux: use 50-75% of available RAM
  AVAILABLE_MB=$(free -m | awk '/^Mem:/{print int($7 * 0.6)}')
  # macOS: use 50-75% of available RAM  
  AVAILABLE_MB=$(vm_stat | awk '/Pages free/{free=$3} /Pages inactive/{inactive=$3} END{print int((free+inactive)*4096/1024/1024*0.6)}')
  
  # Set a sensible default if detection fails (2GB works on most machines)
  HEAP_SIZE=${AVAILABLE_MB:-2048}
  
  deno run --v8-flags=--max-old-space-size=${HEAP_SIZE} ...
  ```
  
  **Common scenarios:**
  - **Large machines** (32GB+ RAM): Use `--max-old-space-size=8192` or higher
  - **Medium machines** (8-16GB RAM): Use `--max-old-space-size=4096`
  - **Small/old machines** (4GB or less): Use `--max-old-space-size=2048`
  
  **Note:** The Rust library itself is memory-efficient and streams data from
  Parquet files. The TypeScript/Deno controller typically consumes more memory.
  Setting `--max-old-space-size` too high on memory-constrained machines causes
  V8 to allocate beyond available RAM, triggering the OOM killer.
- **Analysis timeout**: The analysis phase has a default 10-minute timeout when
  `analysis_deadline_ms` is not provided. If a timeout is explicitly provided
  but is less than 3 seconds or greater than 1 hour, it will be clamped to the
  10-minute default with a warning message.

## Additional documentation

- [Impact Calculation](docs/IMPACT_CALCULATION.md) - Detailed explanation of how
  neuron impact is calculated, including special handling for threshold (STEP/BIPOLAR)
  and selection (MINIMUM/MAXIMUM) squash functions.

## Existing reference material

The sections below capture the original project brief, scale targets, and
engineering standards. They remain authoritative for contributors and are linked
here for convenience:

- [Project goal](#goal)
- [Problem statement](#problem-statement)
- [Performance requirements](#performance-requirements)
- [Features](#features)
- [Development guidelines](#development)
- [File format](#file-format)
- [JSON interface](#json-interface)
- [Code quality expectations](#code-quality)
- [Cross-platform support](#cross-platform-support)
- [Distributed build & versioning](#distributed-build--versioning)

---

## Goal

The goal is to record neuron activations and errors during the discovery
training phase, then scan this recorded data to identify beneficial new
synapses/neurons that would reduce error. **The current DenoJS implementation has
severe performance and memory issues that make discovery unviable for larger
models.** This Rust library must solve these performance/memory problems while
maintaining the same functional behavior.

## Problem Statement

The current DenoJS implementation requires extreme filtering of the training
data (millions of records) to make discovery work in reasonable time. The
DenoJS has severe performance and memory issues that make discovery unviable
for larger models. This library aims to solve these problems while maintaining
the same functional behavior.

**Target Scale:**
- Training records: Millions (not hard-coded, but that's the scale)
- Observations per record: 1,486 (float32 values - this is the input size)
- Neurons: 447
- Synapses: 16,012

## Performance Requirements

- Must handle millions of training records efficiently without memory issues
- Must process significantly more data than DenoJS can handle (DenoJS requires extreme filtering to work)
- Must be significantly faster than TypeScript implementation
- Must use minimal memory (avoid loading all data into memory at once)
- Files are temporary (deleted after discovery phase)
- Only needs compatibility within the Rust discovery phase
- Goal: Process full dataset (or much more) compared to filtered subset in DenoJS

## Features

- Record neuron activations and errors during discovery training phase
- Single Parquet file format (eliminates many-small-files problem)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging
- Cross-platform support (macOS, Ubuntu, AWS Linux)

## Development

### Development Guidelines

**IMPORTANT: All development must follow these mandatory practices:**

1. **Test-Driven Development (TDD)**: Always write tests first before implementing features
   - Write a failing test for the new feature
   - Implement the feature to make the test pass
   - Refactor if needed while keeping tests green
   - All new tests should pass after implementation
   - **Always read this README before making any changes**
   - See [Testing Philosophy](#testing-philosophy) below for test organisation guidelines

2. **Code Quality Enforcement**: **MUST run quality checks after EVERY code change**
   - **CRITICAL**: Execute `./quality.sh` after making ANY code modifications
   - This script runs formatting, linting, type checking, and all tests
   - Fix all linting issues automatically before committing
   - Ensure code formatting and quality standards are maintained
   - **Never commit code without running `./quality.sh` first**

### Prerequisites

**User-installable (automatically handled by `runlib.sh`):**
- Rust (latest stable version) - automatically installed by `runlib.sh` if missing
- Cargo - automatically installed by `runlib.sh` if missing

**System packages (must be installed by administrator):**
- **jq** - must be installed system-wide (required for build scripts)
- **Build tools (gcc/cc)** - required on Linux systems:
  - **Ubuntu/Debian**: `sudo apt-get install -y build-essential`
  - **RHEL/CentOS/Amazon Linux**: `sudo yum groupinstall -y "Development Tools" && sudo yum install -y gcc`
  - **Fedora**: `sudo dnf groupinstall -y "Development Tools" && sudo dnf install -y gcc`
- **macOS**: Xcode Command Line Tools (typically already installed, or can be installed via `xcode-select --install` without sudo)

### Building

```bash
cargo build
```

Build library for release:

```bash
cargo build --release --lib
```

### Building with runlib.sh

The library can be built and installed using the `scripts/runlib.sh` script:

```bash
./scripts/runlib.sh
```

This will build the library and install it to `~/.cargo/lib/` with version tracking.

**Note:** The script automatically installs Rust and Cargo if missing (no sudo required). However, system packages must be installed by an administrator:
- **jq** must be installed system-wide
- **Build tools (gcc/cc)** must be installed on Linux systems (see Prerequisites above)
- If build tools are missing, the script will display clear error messages with installation instructions for the administrator

### Testing

```bash
# Run all tests (unit + integration)
cargo test

# Run unit tests only (in src/)
cargo test --lib

# Run integration tests only (in tests/)
cargo test --test '*'

# Run specific test file
cargo test --test integration
cargo test --test regression_v0_1_123
cargo test --test weights  # All weight-related tests

# Run tests matching a pattern
cargo test test_hidden_neuron
```

**Note**: GPU-dependent tests include `skip_without_gpu!()` and will be skipped
automatically on machines without a GPU. Run `./quality.sh` locally with a GPU
for full test coverage.

### Testing Philosophy

**The quality of tests is what makes a good system.** This project follows these
testing principles:

#### 1. Test OUTCOMES, not implementation

Tests should verify **what** the system does, not **how** it does it. The same
test should pass regardless of whether we use GPU, CPU, or TPU internally.

```rust
// GOOD: Tests the outcome
#[test]
fn test_low_impact_neurons_are_detected() {
    let creature = create_test_creature();
    let impacts = compute_impacts(&creature);
    
    // Verify we detected the expected low-impact neurons
    assert!(impacts["far-from-output"] < 0.1);
    assert!(impacts["close-to-output"] > 0.9);
}

// BAD: Tests implementation details
#[test]
fn test_gpu_kernel_computes_impacts() {
    // Don't test HOW we compute, test WHAT we compute
}
```

#### 2. Separate test files organised by feature

Small, focused test files make it **obvious when tests change**:
- Adding a new test file = good (new coverage)
- Modifying existing tests = raises questions (why?)
- Removing tests = requires justification

| Directory | Purpose | Example |
|-----------|---------|---------|
| `tests/` | Integration tests for public API | `tests/integration.rs` |
| `tests/regression_*.rs` | Prevent re-introducing fixed bugs | `tests/regression_v0_1_123.rs` |
| `tests/<feature>.rs` | Tests grouped by feature/concern | `tests/weights.rs`, `tests/impacts.rs` |
| `src/*.rs` (`#[cfg(test)]`) | Unit tests for private functions | Only when necessary |

**Prefer separate test files** in `tests/` over inline unit tests:
- Easier to see what changed in code review
- Clear separation of concerns
- Don't need to make APIs public just for testing

#### 3. Don't make APIs public just for testing

If a function is internal, keep it internal. Use integration tests to verify
behaviour through the public API. Only use inline unit tests (`#[cfg(test)]`
in `src/`) when you genuinely need to test private implementation details.

#### 4. Group related tests by concern

When investigating an issue (e.g., "something's wrong with weight calculations"),
you should be able to find all relevant tests in one place:

```
tests/
├── common/mod.rs           # Shared test utilities
├── integration.rs          # General integration tests
├── regression_v0_1_123.rs  # Regression tests for v0.1.123 fixes
├── weights.rs              # All weight calculation tests (future)
├── impacts.rs              # All impact score tests (future)
└── activations.rs          # All activation function tests (future)
```

#### 5. Test changes are significant

In code review:
- **New test file**: Generally good - more coverage
- **Modified test**: Why? Did requirements change? Was it wrong?
- **Removed/skipped test**: Red flag - must be justified

Tests are the specification. Changing them changes what the system promises to do.

### Continuous Integration

GitHub Actions runs quality checks on every pull request to `Develop`:

```yaml
# .github/workflows/ci.yml jobs:
- auto-format          # Applies rustfmt and commits fixes
- version-increment    # Auto-bumps patch version when src/ changes  
- quality              # fmt check, clippy, cargo check, tests, build
- shell-checks         # Validates bash script syntax
- spell-check          # Runs codespell on codebase
- validation           # Checks required files and Cargo.toml
- security             # Runs security audit workflow
```

**Test coverage**: The quality job runs `cargo test --all-targets --all-features`
which includes:
- Unit tests in `src/` (lib target)
- Integration tests in `tests/` directory
- All feature-gated tests

**GPU tests are skipped in CI** (no GPU available). The CI ensures:
- Code compiles and passes linting
- Non-GPU unit tests pass
- Public API contract is maintained (integration tests)

For full GPU test coverage, run `./quality.sh` locally before pushing.

**⚠️ CRITICAL: Do NOT modify `.github/workflows/ci.yml` without explicit approval.**
This workflow is essential for PR checks. If accidentally modified, restore from Develop:
```bash
git checkout Develop -- .github/workflows/ci.yml
```

## File Format

### Single Parquet File

Instead of many small CSV files (one per neuron), we use a single Parquet file:

- File location: `.discovery/{creature_uuid}_{random}/discovery_data.parquet`
- Schema:
  - `obs_index: u32` - Observation index (training record index) for ordering
  - `neuron_uuid: string` - Neuron identifier
  - `value: f32` - Neuron value (optional, can be null)
  - `activation: f32` - Neuron activation
  - `errors: list<f32>` - Array of error values

**Benefits:**
- Single file handle (eliminates small-file problems)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging
- Good performance for single file (overhead acceptable)
- Widely supported format

### Debugging Parquet Files

Parquet files can be viewed with standard tools:

**Python:**
```python
import pandas as pd
df = pd.read_parquet('discovery_data.parquet')
print(df.head())
```

**DuckDB:**
```sql
SELECT * FROM 'discovery_data.parquet' LIMIT 10;
```

**Command-line:**
- `parquet-tools` (Java-based)
- `parquet-cli` (Rust-based)

Many data tools support Parquet natively (Tableau, Apache Spark, etc.)

## CRITICAL REQUIREMENT: Atomic Record Writes

**For each discovery record, all data (observations, activations, errors) MUST come from the same training record.** This is essential because:
- The analysis phase matches records by index (record 0 from neuron A corresponds to record 0 from neuron B)
- When evaluating synapse candidates, records must align - all data in record i must be from the same training record
- If observations, activations, and errors don't line up from the same training record, analysis will be incorrect

**Implementation Requirements:**
- **Atomic writes**: For each training record, activate creature, collect ALL neuron data (activations, errors), then write ALL neuron rows together
- **Parallelisation allowed**: Since training dataset is already randomised, we CAN process different training records in parallel
- **Per-record atomicity**: Each parallel task must process one complete training record (activate → collect all neurons → write all neurons atomically)
- **Cross-neuron alignment**: Records with the same `obs_index` across different neurons correspond to the same training record
- **No mixing**: Never mix data from different training records within a single discovery record write
- **Matching by obs_index**: TypeScript matches records across neurons by `obs_index` (not by array position), so record order from Rust doesn't matter

## JSON Interface

### Input Format

```json
{
  "creature": {
    "neurons": [
      {
        "uuid": "hidden-1",
        "type": "hidden",
        "squash": "TANH",
        "bias": 0.0
      }
    ],
    "synapses": [
      {
        "from_uuid": "input-0",
        "to_uuid": "hidden-1",
        "weight": 0.5
      }
    ],
    "input": 20,
    "output": 2
  },
  "training_data": [
    {"input": [0.1, 0.2, ...], "output": [0.5, 0.3]},
    ...
  ],
  "temp_dir": ".discovery/abc123_456789",
  "binary_file_path": "/path/to/binary.bin",  // optional
  "record_indices": [0, 5, 10, ...],  // optional
  "timeout_seconds": 300  // optional
}
```

### Output Format

Success:
```json
{
  "success": true,
  "temp_dir": ".discovery/abc123_456789",
  "file": "discovery_data.parquet"
}
```

Error:
```json
{
  "success": false,
  "error": "Error message here"
}
```

## Code Quality

```bash
# Format code
cargo fmt

# Lint code
cargo clippy

# Check code
cargo check

# Run quality checks
./quality.sh
```

## Cross-Platform Support

The library must work on:
- **macOS** (primary target)
- Ubuntu
- AWS Linux (x86_64 and ARM64)

All dependencies build automatically on remote, unattended machines.

## Distributed Build & Versioning

- Versions are managed in `Cargo.toml` and are automatically incremented by CI on pull requests when files in `src/` change.
- Local and remote runs use a distributed build pattern via `scripts/runlib.sh`:
  - The library is installed to `~/.cargo/lib/` and tracked with a version marker at `~/.cargo/lib/.neat_ai_discovery.version`.
  - On run, if the installed version differs from `Cargo.toml`, the library is rebuilt and reinstalled; otherwise it runs silently without rebuilding.
- Do not manually edit version numbers; CI handles patch bumps when source changes are detected.

## License

This project is licensed under the terms specified in the LICENSE file.
