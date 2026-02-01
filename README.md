# NEAT-AI-Discovery

A high-performance Rust companion library for
[`stSoftwareAU/NEAT-AI`](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend **mutation candidates** (add/remove/modify) that are likely
to improve the creature's score.

**Important**: This library does **not** directly “fix” a creature. It returns
candidates derived from the recorded samples. The NEAT-AI controller performs an
**ablation test** style validation step by cloning the creature, applying a
candidate (for example, disabling or removing a neuron), then re-scoring against
the **full training set**. Only candidates that measurably improve the score are
admitted back into the population, where normal NEAT evolution takes over. This
guided approach helps avoid the slow random-mutation search as creatures grow
larger.

Controllers call into the library via Deno FFI to power `Creature.discoveryDir()`
workflows.

## Project Mission

**The sole goal of this library is to discover changes that improve the creature's
score — as fast as possible.**

Everything in this repository serves that objective:

1. **Improve the creature's score.** Only return candidates which are expected to improve the creature's score.
   Every candidate proposed by the analysis pipeline
   must have a positive expected improvement; anything else wastes the controller's
   evaluation budget.

2. **Discover improvements as fast as possible.** We leverage GPU compute shaders
   and SIMD where available so that large creatures (thousands of neurons / synapses)
   can be analysed in seconds rather than minutes. Speed matters because faster
   discovery means more generations per hour and therefore faster evolution.

3. **Minimise changes to the calling programme (NEAT-AI).** New discovery features
   should reuse existing candidate types (`addNeuron`, `removeSynapse`,
   `setBias`, `setWeight`, `coordinatedStructural`, etc.) whenever possible. If
   a new candidate type is truly required it must be documented in this README and
   a corresponding handler added to NEAT-AI.

> **In short:** discover score-improving mutations, use GPU/SIMD to do it quickly,
> and reuse the candidate types that NEAT-AI already understands.

## TL;DR

- **This library finds candidates, it does not “auto-fix” creatures**: NEAT-AI validates candidates by rescoring on the full training set.
- **GPU required**: discovery is skipped when no compatible GPU is available (see `check_gpu_available()`).
- **Build/install**: `./scripts/runlib.sh` (installs to `~/.cargo/lib/` with version tracking).
- **Preferred recording API**: streaming (`start_discovery_session` → `append_discovery_records` → `finish_discovery_session`) to avoid JS/V8 string limits.
- **Free FFI results**: every FFI call returning a `char*` must be freed with `free_discovery_result()`.

## Current FFI API (Deno FFI entry points)

The library exposes a Deno FFI-friendly symbol set. The authoritative list of exported
symbols lives in `src/lib.rs` as `#[no_mangle] pub extern "C"` functions.

The most commonly used entry points are:

- **GPU probe**: `check_gpu_available()` (returns JSON)
- **Version probe**: `get_library_version()` (returns JSON)
- **Recording**:
  - Streaming: `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, `cancel_discovery_session`
  - Single-call: `record_discovery` (avoid for large runs; prefer streaming to prevent JS/V8 string limits)
- **Analysis**: `rank_focus_neurons`, `analyze_parallel`
- **Utilities**: `merge_discovery_parquet`, `read_discovery_records_ffi`, `export_visualisation_snapshot`
- **Memory management**: `free_discovery_result`

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

### Minimum System Requirements

Discovery is automatically disabled on machines that don't meet minimum requirements:

| Requirement | Minimum | Reason |
|-------------|---------|--------|
| **Total RAM** | 4 GB | GPU operations require memory for staging buffers |
| **Available RAM** | 1 GB | Prevents hangs from memory pressure/swap thrashing |
| **GPU** | Metal (macOS) or Vulkan (Linux) | Required for compute shaders |

**Note on macOS memory reporting**: Apple Silicon Macs aggressively cache files in
memory, so "available" RAM may appear low (e.g., 1-2GB on an 8GB machine). This is
normal – macOS can quickly reclaim cached memory when needed. The 1GB minimum is
sufficient for modern Macs with unified memory architecture.

When requirements aren't met, `check_gpu_available()` returns `gpuAvailable: false`
with a descriptive reason. NEAT-AI's evolution process continues normally - only
the discovery optimisation is skipped.

**Very old machines** (pre-2015 Macs, old Linux servers without GPU) will have
discovery disabled gracefully. This prevents hangs while allowing evolution to run.

### Parquet file memory check

Before loading a parquet file, the library checks if there's enough available memory.
Parquet files are compressed, so they typically expand to 2-4× their file size when
loaded into memory. The library uses conservative checks to prevent memory pressure
that causes the system to become unresponsive:

1. **3× file size** for in-memory representation
2. **1GB headroom** after loading (for GPU buffers, etc.)
3. **50% RAM limit** - won't use more than half of total RAM for parquet data

If insufficient memory is detected, you'll see an error like:

```
Parquet file too large for system memory.
• Parquet file: 1231 MB (/path/to/records.parquet)
• Estimated memory needed: 3693 MB (46% of 8 GB total RAM)
• Maximum safe usage: 4096 MB (50% of RAM = 4.0 GB)
• Maximum parquet file size for this machine: 1365 MB

Suggestions:
• Reduce sample rate (e.g., discoverySampleRate: 0.02)
• Reduce recording time (discoveryRecordTimeOutMinutes)
• Use a machine with more RAM (16GB+ recommended for large creatures)
```

**Maximum parquet file sizes by RAM:**
| Total RAM | Max Parquet File | Explanation |
|-----------|------------------|-------------|
| 8 GB      | ~1.3 GB          | 50% limit = 4GB, ÷3 decompression = 1.3GB |
| 16 GB     | ~2.6 GB          | 50% limit = 8GB, ÷3 decompression = 2.6GB |
| 32 GB     | ~5.3 GB          | 50% limit = 16GB, ÷3 decompression = 5.3GB |

To reduce parquet file size:
- Lower `discoverySampleRate` (e.g., from 0.05 to 0.02)
- Reduce `discoveryRecordTimeOutMinutes`
- Use fewer training data files

### Streaming Parquet Loading (Issue #193)

For very large datasets, the library supports streaming parquet loading with block-based
caching and prefetch. Instead of loading the entire file into memory, records are loaded
on-demand in blocks with LRU eviction.

**Benefits:**
- Lower peak memory usage (only loaded blocks in memory)
- Faster time-to-first-result (analysis begins as soon as first block loads)
- Predictable memory usage (configurable block limit)
- Better I/O parallelism (loading and analysis overlap via prefetch)

**Configuration:**

```bash
# Maximum blocks to keep in memory (default: adaptive based on RAM)
# Low memory: 10 blocks, Standard: 50 blocks, High: 100 blocks
export NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS=100

# Prefetch depth - how many blocks ahead to load (default: 2)
export NEAT_AI_DISCOVERY_PREFETCH_DEPTH=2

# Disable streaming and use full preload (like before)
export NEAT_AI_DISCOVERY_PRELOAD_ALL=1

# Block size in records (default: 10000, minimum: 10)
export NEAT_AI_DISCOVERY_BLOCK_SIZE=10000
```

**When to use streaming:**
- Parquet files > 500MB
- Memory-constrained environments (< 8GB RAM)
- When you want faster time-to-first-result

**When to use full preload:**
- Small to medium parquet files (< 500MB)
- When you have plenty of RAM
- When neurons are accessed in random order repeatedly

### Tiered Loading Strategy (Issue #215)

The library now automatically selects the optimal loading strategy based on file size
and available system memory. This extends the streaming functionality with a more
intelligent neuron-level LRU cache for medium-sized files.

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

**Benchmark Results (100 neurons, 1000 records each):**

| Strategy | Sequential Access | Notes |
|----------|-------------------|-------|
| PreloadAll | 4.4 µs | Fastest, all data in memory |
| LruCache (large) | 5.9 µs | Near-preload performance |
| LruCache (evicting) | 155 ms | Eviction overhead when capacity exceeded |
| Tiered (auto) | 4.3 µs | Auto-selects best strategy |

### Error Distribution Analysis (Issue #192)

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
# When enabled, candidates include information about how they affect outlier samples
export NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS=1

# Set the percentile threshold for outlier identification (default: 90)
# Samples above this percentile are considered outliers
export NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE=90
```

**Benefits:**

1. **Targeted improvement**: Focus on fixing the worst samples first
2. **Discovery efficiency**: Smaller outlier sets can be analysed faster
3. **Better predictions**: Homogeneous error patterns are easier to model
4. **Debugging insight**: Understand WHY errors occur through distribution analysis

### Parallel focus selection

Focus neuron selection is now parallelised using rayon for better CPU utilisation:
- Neuron ranking (computing errors/impacts for each neuron)
- Selection statistics for MIN/MAX/IF neurons
- Removal candidate identification

This can significantly speed up focus selection on multi-core systems, especially for
large creatures with hundreds of neurons.

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
4. **Always run the quality gate before committing** (CI treats warnings as errors, so don’t skip this):

   ```bash
   ./quality.sh
   ```

## Deployment Checklist

Before committing code changes, ensure you complete the following steps.

**Important**: Always run `./quality.sh` locally before committing/pushing. This repo runs Clippy with `-D warnings`, so even “minor” warnings (dead code, Clippy lints, etc.) will fail CI.

1. **Run quality checks in both repositories:**
   ```bash
   # In NEAT-AI-Discovery
   ./quality.sh
   
   # In NEAT-AI
   cd ../NEAT-AI
   ./quality.sh
   ```

2. **Verify all tests pass** in both repositories before committing.

**Note on versions**: Do not manually bump versions. This repo uses CI to increment
`Cargo.toml` patch versions when `src/` changes are detected (see
[Distributed Build & Versioning](#distributed-build--versioning)). If you need to
confirm what a worker has loaded, call `get_library_version()`.

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
│  4. If `analysisDeadlineMs` is set, randomise within the top-K and preserve  │
│     that diversified ordering through truncation to avoid category starvation│
│     across repeated runs                                                    │
│  5. Return candidates (optionally limited by max_candidates)                │
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

#### What we mean by “ablation test”

In this project, “ablation test” refers to the controller-side validation step
where a candidate mutation is applied to a cloned creature (commonly removing or
disabling a neuron/synapse), then the modified creature is re-scored against the
**full training set**. If (and only if) the score improves, that modified
creature is accepted back into the population.

This keeps discovery honest: the Rust analysis uses recorded samples to propose
candidates, but the only metric that matters is the real, full-dataset score
measured by NEAT-AI.

<details>
<summary>Deep dive: analysis workflow details and historical notes</summary>

### Detailed workflow

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
- When no candidate “makes the grade” (e.g. there were no overlapping samples,
  the GPU observed zero consistent improvements, or every candidate fell under
  the requested threshold) set `NEAT_AI_DISCOVERY_VERBOSE=1` before launching
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

### Coordinated Structural Discovery (Issue #165)

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

#### Epistatic Neuron Pair Pre-Detection (Issue #202)

During synapse analysis, the library proactively detects **epistatic neuron pairs** - cases where two source neurons targeting the same output would provide better improvement when added together than either would alone. This addresses the "neutral plateau" problem where individual operations appear to have little benefit.

**Detection strategy**:
- **Complementary pattern detection**: Identifies source neurons with non-overlapping "firing" patterns (activation ≥ 0.5). When neuron A fires on one subset of samples and neuron B fires on a different subset, adding both synapses together covers more samples than either alone.
- **Combined improvement estimation**: For pairs with high complementarity (≥70% non-overlap), the library estimates the combined improvement as roughly the sum of individual improvements.

**When epistatic pairs are detected**:
- Both individual improvements are positive, AND
- Complementarity is ≥70% (firing patterns have low overlap), AND
- Combined improvement exceeds the best individual improvement

**Example**: If input-0 correlates with positive error on the first half of samples and input-1 correlates with positive error on the second half, neither alone improves overall score significantly, but adding both synapses together addresses all samples.

**Output**: Epistatic pair candidates appear as entries in `coordinatedStructuralCandidates` with two `addSynapse` operations and a comment indicating the epistatic relationship.

#### Example: "noisy vs trusted" inputs (thermometer pattern)

If two inputs feed the same target with the same starting weight, but one input is much noisier (higher activation variance), a coordinated candidate may:

- remove the noisy synapse
- remove the trusted synapse
- add the trusted synapse back with a higher weight

This preserves (or improves) behaviour while reducing variance and redundancy, and avoids the "single edit looks bad" trap during ablation.

#### Redundant Path Pruning with Renormalisation (Issue #164)

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

#### Saturated Neuron Detection (Issue #342)

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
`changeSquash` and/or `setBias` operations. No new candidate types are needed — this
reuses the existing coordinated structural change mechanism.

#### Bottleneck Neuron Detection (Issue #343)

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
`addNeuron` and/or `addSynapse` operations. No new candidate types are needed — this
reuses the existing coordinated structural change mechanism.

#### Dead Neuron Detection (Issue #341)

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

**Recommended action**:
- **Remove neuron**: Emit a `RemoveNeuron` operation to eliminate wasted computation

**Output**: Dead neuron candidates appear in `coordinatedStructuralCandidates` with
`removeNeuron` operations. No new candidate types are needed — this reuses the existing
coordinated structural change mechanism.

#### Correlated Error Pattern Detection (Issue #344)

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

**Recommended action**:
- **Add shared hidden neuron**: A single new hidden neuron that connects predictive inputs
  to all outputs in the correlated group, addressing the shared missing cause as an atomic
  coordinated structural change

**Output**: Correlated error groups appear in `coordinatedStructuralCandidates` with
`addNeuron` and `addSynapse` operations. No new candidate types are needed — this reuses
the existing coordinated structural change mechanism.

#### Multi-Hop Candidate Analysis (Issue #230)

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
and/or `addSynapse` operations. No new candidate types are needed — this reuses the existing
coordinated structural change mechanism.

#### Oscillating Neuron Detection (Issue #358)

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
`changeSquash` and optionally `setBias` operations. This reuses the existing coordinated
structural change mechanism.

#### Dormant Synapse Detection (Issue #359)

Identifies synapses with near-zero weights that contribute negligible signal to their
target neuron. Dormant synapses waste computation during both forward pass and discovery
analysis without providing meaningful information flow.

**Detection criteria**:
1. **Near-zero weight**: Absolute weight < 1e-4.
2. **Low contribution**: Mean absolute contribution (|weight × source_activation|) < 1e-4.
3. **Not the sole connection**: The target neuron has other incoming synapses (removing the
   only input would be destructive).
4. **Sufficient samples**: At least 20 samples for statistical reliability.

**Recommended actions**:
- **Remove the synapse**: Emit a `removeSynapse` operation to reduce network complexity.

**Output**: Dormant synapse candidates appear in `coordinatedStructuralCandidates` with
`removeSynapse` operations. This reuses the existing coordinated structural change mechanism.

#### Opposing Synapse Detection (Issue #360)

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
`removeSynapse` or `setWeight` operations. This reuses the existing coordinated structural
change mechanism.

#### Output Bias Drift Detection (Issue #361)

Identifies output neurons with a consistent error sign bias — neurons whose errors are
predominantly positive (predicting too low) or predominantly negative (predicting too
high) across training samples. This systematic bias indicates the neuron's bias parameter
needs adjustment.

**Detection criteria**:
1. **Consistent error sign**: More than 70% of errors share the same sign.
2. **Meaningful mean error**: Absolute mean error ≥ 0.01 (not just noise).
3. **Output neurons only**: Hidden and input neurons are excluded.
4. **Sufficient samples**: At least 20 samples for statistical reliability.

**Recommended actions**:
- **Set bias**: Adjust the output neuron's bias by the negative of the mean error to
  centre the predictions.

**Output**: Bias drift candidates appear in `coordinatedStructuralCandidates` with
`setBias` operations. This reuses the existing coordinated structural change mechanism.

#### Candidate Clustering for Redundancy Reduction (Issue #224)

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

> **Version history**: Detailed version-by-version change notes for the analysis
> pipeline have been moved to [CHANGELOG.md](CHANGELOG.md).

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

#### Sensible parameter ranges (add-neurons)

In practice, we cache failures and do not re-try them. That makes it important to
avoid proposing candidates with absurd parameters that are very unlikely to survive
full rescoring.

As a production guard rail (Dec 2025), add-neuron candidates are only returned when
their parameters are within sensible bounds:

- **incomingWeight**: \(|w| \le 20\)
- **bias**: \(|b| \le 10\)
- **outgoingWeight**: \(|w| \le 0.1\) (already clamped by the optimiser)

This filtering is applied after the "Extreme → Conservative/Gentle Nudge" pairing so
unsafe originals can be dropped while still allowing safe variants to be evaluated.

#### IDENTITY neuron filtering

**IDENTITY neurons with bias ≈ 0 are redundant** because they're mathematically
equivalent to a direct synapse:

```
IDENTITY(input × incoming_weight + 0) × outgoing_weight = input × incoming × outgoing
```

This is just a synapse with `weight = incoming_weight × outgoing_weight`. Discovery
now filters out these candidates:

1. **Bias filtering**: IDENTITY candidates with `|bias| < 0.01` are rejected
2. **Use synapse analysis**: Direct connections should use `add-synapses`, not `add-neurons`

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

</details>

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
  
  **Design goal (coverage over time)**: Production runs are expected to be
  **deadline-constrained** and repeated (for example, a worker loop that calls
  discovery many times per day). The system is designed so that **all discovery
  work is covered over time**, even when a single run times out:
  - **All focus neurons** will be covered over time because focus ordering is
    randomised when a deadline is configured.
  - **All eligible source neurons** (inputs + hidden + constants, respecting
    forward-only constraints) will be covered over time because source ordering
    is also randomised under deadlines.
  - **Synapse vs neuron analysis**: When a deadline is configured and both analyses
    are enabled, the library **randomises which analysis runs first** each invocation.
    This means one run may return only synapse candidates and the next may return
    only neuron candidates. Over repeated runs, both get opportunities to run first.
  - **All discovery candidate types that Rust emits** (e.g., add-synapse and
    add-neuron) are intended to get a fair share of work over time under repeated
    runs. This library now avoids returning the exact same top candidates every
    run under deadlines (to reduce starvation when controllers cache failures).
  
  **Important**: With a hard timeout, a single invocation will often return
  **partial results** by design. Coverage is achieved via repeated invocations,
  not by making one run arbitrarily long (which would delay returning improved
  creatures back into the population).
  
  **Timeout logging** (v0.1.163): The library now logs when analysis starts and
  when a timeout is reached:
  
  ```
  [NEAT-AI-Discovery] Starting neuron analysis: 5 focus neurons, timeout: 10.0 minutes
  [NEAT-AI-Discovery] neuron analysis reached timeout. Completed 3/5 focus neurons. Returning partial results.
  ```
  
  **Focus neuron randomisation**: When a timeout is configured, focus neurons are
  processed in randomised order. This ensures that repeated runs with timeouts
  will eventually cover all neurons, rather than always processing (and timing
  out on) the same neurons. Enable verbose logging (`NEAT_AI_DISCOVERY_VERBOSE=1`)
  to see the randomised order:
  
  ```
  [NEAT-AI-Discovery][verbose] Randomised focus order: ["output-3", "output-1", "output-4"]... (+2 more)
  ```

  **Source neuron randomisation (forward-only)**: Eligible source neurons are evaluated in a
  randomised order (inputs, hidden, constants), while still enforcing **forward-only**
  candidates (a source must be upstream of the target in the creature ordering).
  This matters when you use timeouts: repeated runs will scan different sources over time.

  **Optional bias to newer inputs (29-Dec-2025)**: If your input list grows over time and
  you want discovery to prefer newer inputs (higher `input-N` indices) earlier in the run,
  set `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` to a finite number > 0:

  - `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS=1`: mild bias
  - `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS=3`: stronger bias toward the end

  Hidden/constant sources keep weight 1.0; only `input-N` sources are biased.

  **Focus on unused observations (10-Jan-2026, Issue #182)**: If you've added new observations
  (input neurons) to your training data and want discovery to focus on these "unused" inputs
  before evaluating existing connections, set `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1`.

  When enabled, input neurons that have NO existing outgoing synapses are prioritised
  (moved to the front of the evaluation queue). This is particularly useful when:

  - You've added a few hundred new observations to the training data
  - You want to quickly evaluate whether these new inputs improve the network
  - You're using deadline-constrained runs and want new inputs evaluated first

  Values that enable the feature: `1`, `true`, `yes` (case-insensitive)
  Values that disable the feature: unset, empty, `0`, `false`, `no`

  Example:
  ```bash
  export NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1
  ```

  **Optional folding of constant sources into bias (7-Jan-2026)**: When a source neuron's
  activation range is ~0, an add-synapse from that source behaves like a constant offset
  on the downstream neuron (equivalent to a bias change). To avoid paying complexity cost
  for a new edge, the library may emit a coordinated-structural candidate containing a
  single `setBias` operation instead of an `addSynapse`.

  **Dynamic threshold (Issue #199, 28-Jan-2026)**: The threshold is now dynamically calculated
  based on the creature's source variance profile. This captures more coordinated candidates
  in creatures where "constant" is relative to the overall variance distribution.

  ```
  dynamic_threshold = 1e-7 × max(1.0, source_std_dev_avg / 0.05)
  ```

  This means:
  - For creatures with mostly low-variance sources: threshold stays at 1e-7
  - For creatures with high-variance sources: threshold scales up proportionally

  Control this behaviour with `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD`:

  - unset / empty: enabled with **dynamic threshold** based on source variance profile
  - `0`: disable folding (always emit `addSynapse` when otherwise valid)
  - `> 0`: enable folding with a **fixed custom threshold** (overrides dynamic calculation)

  The heuristic compares the predicted contribution range:

  - `effectRange ≈ |weight| × (maxActivation - minActivation)`
  - fold when `effectRange <= threshold`
- **Low GPU utilisation**: If you're seeing low GPU utilisation (e.g., 20%) during
  analysis, see the [GPU Performance Tuning](#gpu-performance-tuning) section below.
- **Deadlock or stuck process**: If the process appears stuck (0% CPU/GPU), see the
  [Debugging Deadlocks](#debugging-deadlocks) section below.
- **GPU timeout errors**: The library includes automatic timeout protection for GPU
  operations. If the GPU becomes unresponsive, you'll see an error like:
  
  ```
  GPU helpful batch evaluation timed out after 150s. The GPU may be unresponsive.
  Consider reducing batch size or restarting.
  ```
  
  **Adaptive timeout architecture** (v0.1.166):
  - **Minimum GPU batch timeout**: 60 seconds - catches unresponsive GPU quickly
  - **Maximum GPU batch timeout**: 5 minutes - prevents infinite waits
  - **Deadline-aware**: When analysis has a deadline, uses up to half remaining time
  - **Non-blocking work submission**: `send_timeout()` prevents deadlock if GPU hangs
  - **Shutdown timeout**: 12 seconds max (2s send + 10s exit wait) - prevents hung cleanup
  
  The adaptive timeout scales with the analysis deadline. For large datasets (>1GB
  Parquet files), GPU batch evaluations may legitimately take longer than 60 seconds.
  Previously, this caused false "GPU unresponsive" errors. Now the timeout adapts:
  - With 15 minute deadline: batch timeout = min(7.5min, 5min) = 5 minutes
  - With 10 minute deadline: batch timeout = 5 minutes
  - With 3 minute deadline: batch timeout = 1.5 minutes
  - Without deadline: batch timeout = 5 minutes (maximum)
  
  **Critical deadlock fix (v0.1.166)**: Previous versions could hang forever if the GPU
  thread became unresponsive. The work queue channel had limited capacity (4-16 items),
  and if the GPU hung, sending threads would block forever waiting to submit work.
  Now all work submissions use `send_timeout()` which returns an error instead of
  blocking indefinitely:
  ```
  GPU work queue full - send timed out after 300s. The GPU thread may be hung.
  Consider restarting the process.
  ```
  
  This ensures the process always returns (possibly with errors) instead of hanging
  for hours.
  
  **Causes and solutions:**
  - **GPU driver hang**: Restart the process. If persistent, restart the machine.
  - **GPU memory exhaustion**: Reduce `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` (try 256 or 128).
  - **System memory pressure**: Close other applications or reduce workload.
  - **Hardware issue**: Check system logs (`dmesg` on Linux, Console.app on macOS).
  
  **Unattended recovery**: The timeout ensures the process returns an error rather
  than hanging forever, allowing orchestration systems to retry or skip the operation.

## GPU Performance Tuning

The library auto-detects GPU capabilities **and available system memory** to optimise
settings. On startup, it logs the detected configuration:

```
[NEAT-AI-Discovery] Memory: 12.3GB available / 24.0GB total | Tier: standard
[NEAT-AI-Discovery] GPU: Apple M4 (integrated metal) | Tier: high-performance | Batch size: 512
```

### Automatic Adaptation

The library adapts to your machine's capabilities:

| System Memory | Memory Tier | Work Queue | Batch Size Adjustment |
|---------------|-------------|------------|----------------------|
| < 8GB available | Low | 4 | Reduced to 256 |
| 8-16GB available | Standard | 8 | Capped at 512 (v0.1.158) |
| > 16GB available | High | 16 | GPU tier default |

| GPU Type | GPU Tier | Default Batch Size |
|----------|----------|-------------------|
| M4, M4 Pro, M4 Max, M4 Ultra | High | 1024 (512 if standard memory, 256 if low) |
| M3 Pro, M3 Max, M2 Pro, M2 Max | High | 1024 (512 if standard memory, 256 if low) |
| M1, M2, M3 (base) | Standard | 512 (256 if low memory) |
| Discrete GPUs (NVIDIA, AMD) | High | 1024 (512 if standard memory, 256 if low) |
| Integrated GPUs (Intel, etc.) | Standard | 512 (256 if low memory) |

**Why memory matters**: GPU operations require staging buffers in system RAM.
On memory-constrained systems, smaller batches and fewer in-flight requests
prevent swap thrashing which can cause GPU driver hangs.

### Manual Tuning

Override the batch size with an environment variable:

```bash
# For M4 Mac or high-end GPUs - larger batches for better utilisation
export NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=1024

# For older machines or memory-constrained systems - smaller batches
export NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=256

# Experimental: Very large batches for M4 Max/Ultra with lots of GPU memory
export NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=2048
```

Valid range: 64 to 4096. Values outside this range are ignored.

### Understanding GPU Utilisation

Low GPU utilisation during analysis is typically caused by:

1. **CPU-bound sample building**: The library builds sample data on CPU before
   sending to GPU. This is intentional - it reduces GPU memory pressure and
   allows parallel processing. If your analysis is CPU-bound, you'll see bursts
   of GPU activity followed by idle periods.

2. **Small workloads**: If your creature has few neurons or samples, the GPU
   completes work faster than the CPU can prepare new batches.

3. **I/O bottlenecks**: Reading from Parquet files or slow storage can cause
   the GPU to wait for data.

### Tuning for M4 Mac

M4 Macs have significantly more GPU cores than earlier Apple Silicon. The library
automatically detects M4 and uses larger batch sizes (1024 vs 512). For M4 Max
or Ultra, you may benefit from even larger batches:

```bash
# M4 Max/Ultra with 128GB+ RAM
export NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=2048
```

### Compatibility with Older Machines

All tuning options are backwards-compatible. Older machines will:
- Use smaller default batch sizes (512)
- Automatically fall back to safe values if specified batch size is too large
- Continue to work without any environment variables set

### Verbose GPU Diagnostics

Enable verbose logging to see detailed GPU information:

```bash
export NEAT_AI_DISCOVERY_VERBOSE=1
```

This logs:
- GPU adapter name and type
- Detected performance tier
- Selected batch size
- Tuning hints

### GPU Kernel Profiling (Issue #195)

For performance diagnostics, the library can collect timing data for GPU operations.
This helps identify:
- Which shaders are slowest
- CPU vs GPU time breakdown
- Buffer transfer overhead

Enable GPU timing collection:

```bash
export NEAT_AI_DISCOVERY_GPU_TIMING=1
```

When enabled, the `synapseMetadata` (and `neuronMetadata`) in the JSON response
includes a `timing` object:

```json
{
  "synapseMetadata": {
    "candidatesFound": 150,
    "timing": {
      "totalAnalysisMs": 5678.5,
      "gpu": {
        "shaderExecutionMs": 2345.2,
        "bufferTransferMs": 1234.1,
        "shaderTimings": {
          "helpful": { "calls": 150, "totalMs": 234.5, "avgMs": 1.56 },
          "harmful": { "calls": 150, "totalMs": 189.2, "avgMs": 1.26 }
        }
      },
      "cpu": {
        "sampleBuildingMs": 1500.0,
        "resultProcessingMs": 599.2
      }
    }
  }
}
```

**Notes**:
- Timing collection adds approximately 5% overhead when enabled
- Disabled by default for production use
- Timing data is only present when the env var is set before analysis starts
- The env var check is cached on first use, so it must be set before the first analysis

## Debugging Deadlocks

The library includes built-in debugging tools for diagnosing stuck processes and
deadlocks, similar to Java's `kill -3` thread dump.

### Automatic Deadlock Detection

The library automatically detects deadlocks every 10 seconds using `parking_lot`'s
deadlock detection feature. When a deadlock is detected, the process panics with
full backtrace information for all involved threads.

```
================================================================================
DEADLOCK DETECTED - 2 deadlock(s) found
================================================================================

--- Deadlock #1 (2 threads involved) ---

Thread ID: ThreadId(5)
Backtrace:
   0: parking_lot_core::parking_lot::park
   1: neat_ai_discovery::analysis::GpuWorkQueue::evaluate_helpful_batch
   ...

Thread ID: ThreadId(3)
Backtrace:
   ...

================================================================================
PANICKING due to deadlock. See above for thread backtraces.
================================================================================
```

### Thread Dump on Signal (kill -USR1)

Send `SIGUSR1` to dump thread information without terminating the process:

```bash
# Find the process ID
ps aux | grep deno

# Send SIGUSR1 - prints full thread dump without exiting
kill -USR1 <pid>
```

**On macOS**, this automatically runs the `sample` command and prints:
- Deadlock detection results (mutex contention)
- **Full thread backtraces** for ALL threads (filtered to show relevant frames)
- Threads stuck in `neat_ai_discovery`, `wgpu`, `Metal`, `crossbeam`, `rayon`
- Threads waiting in `recv`, `poll`, `wait`, `park`, `sleep`, `pthread_cond`

**On Linux**, this prints:
- Deadlock detection results
- Signal handler thread backtrace
- Instructions for using `gdb` to get full thread dumps

**Example output** (filtered for brevity):
```
================================================================================
THREAD DUMP - 2025-day359 01:20:39 UTC (kill -USR1 received)
Process ID: 8229
================================================================================

--- No mutex deadlocks detected ---

--- Running 'sample' for full thread analysis (1 second) ---

Call graph:

    803 Thread_1343597: deadlock-detector
    +   803 std::thread::sleep  (in libneat_ai_discovery.dylib)

    803 Thread_1343620
    +   803 neat_ai_discovery::analysis::GpuWorkQueue::evaluate_helpful_batch
    +     803 crossbeam_channel::channel::Receiver::recv
    +       803 std::thread::park

--- End of call graph (37 threads) ---
```

**Important (unattended workers)**: `sample` is best-effort and has an internal timeout
so it cannot hang your process forever while trying to print diagnostics.

### Hang watchdog (unattended machines)

Deadlocks are only one kind of hang. If the process becomes stuck (eg GPU driver wedge,
wgpu call that never returns, or other kernel-level stall), you want the worker to crash
so your orchestration can capture logs and restart.

The library includes an optional stall watchdog that:
- triggers a SIGUSR1 thread dump, then
- aborts the process (so logs/crash reports are captured).

Enable it with:

```bash
# Abort if discovery makes no progress for 30 minutes
export NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS=1800

# Optional: delay between SIGUSR1 dump and abort (default: 2 seconds)
export NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS=2
```

**Note**: We use `SIGUSR1` (user-defined signal) which has no default action,
making it safe for diagnostics. Java uses `SIGQUIT` (kill -3).

### Manual Thread Inspection with LLDB

For full thread dumps (Rust doesn't have built-in thread enumeration like Java):

```bash
# On macOS - attach debugger and print all thread backtraces
lldb -p <pid> -o 'thread backtrace all' -o 'quit'

# Or use the sample tool
sudo sample <pid> 1 -file /tmp/sample.txt
cat /tmp/sample.txt
```

### On Linux with GDB

```bash
# Attach to running process
gdb -p <pid> -ex 'thread apply all bt' -ex 'quit'
```

## Additional documentation

- [CodeWiki](https://codewiki.google/github.com/stsoftwareau/neat-ai-discovery) -
  AI-powered documentation and code exploration for this repository.
- [Discovery Types](docs/DISCOVERY_TYPES.md) - Itemised list of all discovery types
  with success/failure counts, descriptions, and recommendations for improvement.
- [Impact Calculation](docs/IMPACT_CALCULATION.md) - Detailed explanation of how
  neuron impact is calculated, including special handling for threshold (STEP/BIPOLAR)
  and selection (MINIMUM/MAXIMUM) squash functions.
- [Changelog](CHANGELOG.md) - Version-by-version history of bug fixes, features,
  and formula corrections.

## Existing reference material

<details>
<summary>Open the original project brief, scale targets, and background rationale (kept for contributors)</summary>

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

> See also the top-level [Project Mission](#project-mission) for the guiding
> principles that apply to every contribution.

The goal is to record neuron activations and errors during the discovery
training phase, then scan this recorded data to identify **high-quality
mutation candidates** (add / remove / modify) that are expected to improve the
creature's score — and to do so as fast as possible using GPU/SIMD acceleration.

This is a guided alternative to NEAT's purely random structural mutations. As
creatures grow large, randomly stumbling into beneficial mutations can take a
very long time. By using the recorded samples to propose promising candidates,
we reduce wasted exploration while keeping the core evolutionary loop unchanged:
NEAT-AI still performs a full re-score on the training set and only keeps
mutations that actually improve.

**The current DenoJS implementation has severe performance and memory issues
that make discovery unviable for larger models.** This Rust library must solve
these performance/memory problems while maintaining the same functional
behaviour.

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

</details>

## Development

> **For AI agents**: Detailed coding conventions, testing philosophy, and
> development guidelines live in [AGENTS.md](AGENTS.md). The sections below
> cover user-facing build and test instructions.

### Development Guidelines

**Always run `./quality.sh` before committing.** This script runs formatting,
linting, type checking, and all tests. CI treats warnings as errors.

See [AGENTS.md](AGENTS.md) for the full set of mandatory development practices
including TDD workflow, code organisation rules, testing philosophy, and
dependency licence requirements.

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

Unit tests verify correctness; benchmarks measure performance. See
[AGENTS.md — Unit Tests vs Benchmarks](AGENTS.md#unit-tests-vs-benchmarks) for
the full distinction.

#### Running Tests

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

#### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench cache_locality
cargo bench --bench batched_activation
```

**Note**: GPU-dependent tests include `skip_without_gpu!()` and will be skipped
automatically on machines without a GPU. Run `./quality.sh` locally with a GPU
for full test coverage.

### Testing Philosophy

See [AGENTS.md — Testing Philosophy](AGENTS.md#4-testing-philosophy) for the
full testing guidelines including TDD workflow, test organisation, and the
distinction between unit tests and benchmarks.

### Continuous Integration

GitHub Actions runs quality checks on every pull request to `Develop`. See
[AGENTS.md — Quality Gate](AGENTS.md#5-quality-gate) for the full list of CI
jobs.

**GPU tests are skipped in CI** (no GPU available). For full GPU test coverage,
run `./quality.sh` locally before pushing.

**Do NOT modify `.github/workflows/ci.yml` without explicit approval.** If
accidentally modified, restore from Develop:
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

## CRITICAL REQUIREMENT: Forward-only activation order (no feedback)

Discovery assumes **forward-only** networks (no recurrent feedback). This is critical for both recording and for applying discovery candidates:

- **Evaluation order matters**: A neuron may only read activations from **earlier** neurons in the creature's evaluation order. In NEAT-AI this means the `creature.neurons[]` list must be in a valid *topological* order for feed-forward execution.
- **Synapse direction constraint**: For feed-forward creatures, synapses must point from an **earlier** neuron to a **later** neuron. If a new neuron is inserted, it must appear **before** any neuron that consumes it.
- **Discovered neurons must be inserted, not appended**: When applying an add-neuron candidate, the new neuron must be inserted at the correct index so it is activated before its outgoing synapse is used by the target neuron.
- **No “remembering” across samples**: If feedback/recurrent connections were enabled, a later neuron could effectively read an activation from a *previous* training sample (stateful recurrence). Discovery explicitly does **not** support that mode - every recorded activation/error is for a single training sample with no cross-sample state.

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

### Streaming Recording API (v0.2.8+)

The streaming API solves the JavaScript "Invalid string length" error that occurs when
trying to serialise large datasets (6+ minutes of recording) into a single JSON string.
Instead of one monolithic `record_discovery` call, data is streamed incrementally.

**Why this matters**: JavaScript/V8 has a maximum string length (~2^28 chars). When
TypeScript accumulated 6+ minutes of discovery data and tried to JSON.stringify it all
at once for the FFI call, it hit this limit. The streaming API keeps each FFI call small.

#### Usage Pattern

```text
TypeScript                              Rust (this library)
────────────────────────────────────────────────────────────────────────
1. start_discovery_session()       →    Creates session + Parquet file
   ↓ returns sessionId

2. Loop while collecting data:
   - Collect records (estimate size)
   - When batch reaches ~50MB or ~10,000 records:
     append_discovery_records()    →    Writes batch to Parquet

3. finish_discovery_session()      →    Finalises Parquet file
   ↓ returns { tempDir, file, totalRecords }
```

#### FFI Functions

**`start_discovery_session`** - Start a new recording session

Input:
```json
{
  "creature": { "neurons": [...], "synapses": [...], "input": 20, "output": 2 },
  "tempDir": ".discovery/abc123_456789"
}
```

Output:
```json
{
  "success": true,
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**`append_discovery_records`** - Append records to an existing session

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "observations": [
    {
      "obsIndex": 0,
      "neuronData": [
        { "neuronUuid": "hidden-1", "activation": 0.5, "value": 0.4, "errors": [0.1] }
      ],
      "inputs": [0.1, 0.2, 0.3]
    }
  ]
}
```

Output:
```json
{
  "success": true,
  "recordsWritten": 42
}
```

**`finish_discovery_session`** - Finalise and close the Parquet file

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

Output:
```json
{
  "success": true,
  "tempDir": ".discovery/abc123_456789",
  "file": "discovery_data.parquet",
  "totalRecords": 12345
}
```

**`cancel_discovery_session`** - Cancel a session (cleanup without finalising)

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

Output:
```json
{
  "success": true
}
```

#### Size Estimation for TypeScript

To decide when to flush, estimate the JSON size before serialising:

```typescript
// Rough estimate: ~200 bytes per neuron record + input array
const estimatedBytes = observations.length * (
  200 * creature.neurons.length + 
  4 * creature.input
);

// Flush when approaching 50MB (well under JS string limits)
const FLUSH_THRESHOLD = 50 * 1024 * 1024;
if (estimatedBytes > FLUSH_THRESHOLD) {
  await appendDiscoveryRecords(sessionId, observations);
  observations = []; // Reset batch
}
```

#### Benefits

- **No string length limits**: Each batch is small enough to serialise
- **Unlimited sample sizes**: Can record for hours without memory issues
- **Fail-safe**: If process crashes, already-written data is preserved in the Parquet file
- **Reduced memory pressure**: TypeScript can discard batches after flushing

## Code Quality

```bash

# Run quality checks ( format, check & test)
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

This project is licensed under the terms of the Apache License 2.0. For the full
license text, please see [LICENSE](./LICENSE)

