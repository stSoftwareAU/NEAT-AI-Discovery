# NEAT-AI-Discovery

A high-performance Rust companion library for
[`stSoftwareAU/NEAT-AI`](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend **mutation candidates** (add/remove/modify) that are likely
to improve the creature's score.

> **creature**: an individual candidate neural network (a genome) in the evolving
> [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
> population. NEAT-AI evolves a population of creatures; this library helps decide
> which mutations to a creature are worth trying.

**Important**: This library does **not** directly "fix" a creature. It returns
candidates derived from the recorded samples. The NEAT-AI controller performs an
**ablation test** style validation step by cloning the creature, applying a
candidate (for example, disabling or removing a neuron), then re-scoring against
the **full training set**. Only candidates that measurably improve the score are
admitted back into the population, where normal NEAT evolution takes over. This
guided approach helps avoid the slow random-mutation search as creatures grow
larger.

Controllers call into the library via Deno FFI to power `Creature.discoveryDir()`
workflows.

## 🎯 Project Mission

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
   a new candidate type is truly required it must be documented and a corresponding
   handler added to NEAT-AI.

> **In short:** discover score-improving mutations, use GPU/SIMD to do it quickly,
> and reuse the candidate types that NEAT-AI already understands.

## ⚡ TL;DR

- **This library finds candidates, it does not "auto-fix" creatures**: NEAT-AI validates candidates by rescoring on the full training set.
- **GPU required**: discovery is skipped when no compatible GPU is available (see `check_gpu_available()`).
- **Build/install**: `./scripts/runlib.sh` (installs to `~/.cargo/lib/` with version tracking).
- **Preferred recording API**: streaming (`start_discovery_session` → `append_discovery_records` → `finish_discovery_session`) to avoid JS/V8 string limits.
- **Free FFI results**: every FFI call returning a `char*` must be freed with `free_discovery_result()`.

## 🚀 Quick Start

1. Install prerequisites (`rustup`, `cargo`, build tools, and `jq`). The
   `scripts/runlib.sh` helper will guide you if anything is missing.
2. Build and install the library:

   ```bash
   ./scripts/runlib.sh
   ```

   This script:
   - Installs Rust and Cargo if missing (no sudo required)
   - Builds the library in release mode
   - Installs it to `~/.cargo/lib/` with version tracking
   - Signs it on macOS for FFI compatibility

   **From the NEAT-AI directory**, run it in a subshell that changes into this
   crate's root first — the script reads `Cargo.toml` from the current working
   directory and aborts if it is not there:

   ```bash
   (cd ../NEAT-AI-Discovery && ./scripts/runlib.sh)
   ```

3. Confirm the artefact exists at `~/.cargo/lib/libneat_ai_discovery.*`.
4. **Always run the quality gate before committing** (CI treats warnings as errors):

   ```bash
   ./quality.sh
   ```

## 🔌 FFI API Summary

The library exposes a Deno FFI-friendly symbol set. The authoritative list lives under
`src/ffi/` (`mod.rs`, `analysis.rs`, `gpu.rs`, `recording.rs`, `utilities.rs`) as
`#[unsafe(no_mangle)] pub extern "C"` functions.

| Category | Entry Points |
|----------|-------------|
| **GPU probe** | `check_gpu_available()` |
| **Version probe** | `get_library_version()` |
| **Recording (streaming)** | `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, `cancel_discovery_session` |
| **Recording (single-call)** | `record_discovery` (avoid for large runs) |
| **Analysis** | `rank_focus_neurons`, `analyze_parallel` |
| **Cancellation** | `cancel_analysis`, `cancel_analysis_memory_pressure` (CRITICAL memory pressure), `reset_cancellation`, `is_analysis_active` |
| **Utilities** | `merge_discovery_parquet`, `read_discovery_records_ffi`, `export_visualisation_snapshot` |
| **Calibration** | `get_calibration_summary` |
| **Lifecycle / cleanup** | `cleanup_discovery_lib` (call before process exit), `cleanup_discovery_dir`, `clean_orphaned_discovery_dirs` |
| **Memory usage** | `discovery_memory_usage_bytes` |
| **Memory management** | `free_discovery_result` |

For the full JSON interface specification and streaming API details, see
[docs/FFI_API.md](docs/FFI_API.md).

## 🖥️ GPU Requirement

**This library requires a GPU.** There is no CPU fallback. If no compatible GPU is
available, discovery is simply skipped — NEAT-AI continues training without the
discovery phase. This is by design:

- **Simplicity**: One code path means fewer bugs.
- **Performance**: GPU-accelerated analysis is the whole point.
- **Optional feature**: Discovery is an optimisation, not a requirement. NEAT-AI
  works fine without it.

### Minimum System Requirements

| Requirement | Minimum | Reason |
|-------------|---------|--------|
| **Total RAM** | 4 GB | GPU operations require memory for staging buffers |
| **Available RAM** | 0.5 GB (macOS) / 1 GB (Linux) | Prevents hangs from memory pressure/swap thrashing |
| **GPU** | Metal (macOS) or Vulkan (Linux) | Required for compute shaders |

When requirements aren't met, `check_gpu_available()` returns `gpuAvailable: false`
with a descriptive `reason` plus a structured capability verdict — `errorKind`
(`gpu_permanent`, `gpu_transient`, or `memory_exhausted`) and `retryable` — so the
caller can branch on a permanent skip versus a transient retry **before** starting
a pass (Issue #1419). NEAT-AI's evolution process continues normally — only the
discovery optimisation is skipped. There is no CPU fallback: callers must not
advertise one, since a GPU-less host would otherwise run every pass to a
guaranteed `0 candidates` result indistinguishable from genuine search exhaustion.

For GPU performance tuning, troubleshooting, and debugging, see
[docs/GPU_GUIDE.md](docs/GPU_GUIDE.md).

## 🎚️ Supported Cost Functions

NEAT-AI-Discovery is **cost-agnostic by construction** — the analysis pipeline
consumes the per-neuron residuals captured in `DiscoverRecord.errors` and never
references NEAT-AI's configured cost function by name. Every NEAT-AI built-in
cost (`MSE`, `MAE`, `MAPE`, `MSLE`, `HINGE`, `CROSS_ENTROPY`,
`CATEGORICAL_ERROR`) is supported. The authoritative per-cost table — residual
semantics, support caveats, and the per-consumer audit — lives in
[docs/COST_FUNCTION_NOTES.md](docs/COST_FUNCTION_NOTES.md).

> [!NOTE]
> 📐 Where discovery reports an "expected improvement" it is **sum-of-squared
> residual (SSE) reduction** computed from the per-neuron `errors` slot. SSE
> equals the network's loss reduction exactly only when NEAT-AI's cost is
> `MSE`; under the other six costs it is a useful ranking signal but not the
> configured loss reduction. See [docs/COST_FUNCTION_NOTES.md](docs/COST_FUNCTION_NOTES.md)
> for the per-consumer audit, the cost-agnostic invariants, and the checklist
> for adding a new cost to NEAT-AI.

## 🔬 Discovery → Evolution Pipeline

The discovery process works as follows:

```mermaid
flowchart TD
    subgraph RUST["🦀 Rust — this library"]
        R1["1. Find ALL candidates with positive expected improvement"]
        R2["2. Apply impact discounting — creature-level predictions"]
        R3["3. Temperature-scaled acceptance (optional MH probabilistic)"]
        R4["4. Sort by expected improvement — best first"]
        R5["5. Randomise within top-K under deadlines to avoid starvation"]
        R6["6. Return candidates — optionally limited by max_candidates"]
        R1 --> R2 --> R3 --> R4 --> R5 --> R6
    end

    subgraph TS["📘 TypeScript — NEAT-AI"]
        T1["1. Receive candidates from Rust — e.g. 100 candidates"]
        T2["2. Select top N based on available CPUs — e.g. 10–20"]
        T3["3. Re-score each candidate IN PARALLEL — apply mutation, measure"]
        T4["4. Keep candidates that ACTUALLY improve the creature's score"]
        T5["5. Return improved creatures to population"]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph EVO["🧬 Evolution"]
        E1["Improved creatures compete in the population"]
        E2["Natural selection breeds out unsuccessful mutations"]
        E3["No manual filtering needed — evolution handles it"]
        E1 --> E2 --> E3
    end

    RUST --> TS --> EVO

    style RUST fill:#e8f4f8,stroke:#2196F3,stroke-width:2px,color:#000
    style TS fill:#fff3e0,stroke:#FF9800,stroke-width:2px,color:#000
    style EVO fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px,color:#000
```

**Key principle**: 🦀 Rust finds structural improvements that reduce error. 📘 TypeScript
validates by measuring actual score. 🧬 Evolution does the rest.

For detailed analysis workflow, coordinated structural discovery, discrete
activation function handling, and detection algorithms, see
[docs/ANALYSIS_DEEP_DIVE.md](docs/ANALYSIS_DEEP_DIVE.md).

## 🗺️ Discovery Scenarios

For a **visual, beginner-friendly overview** of every discovery scenario — with
diagrams, worked examples, and links to research papers — see the
[Discovery Scenarios Guide](docs/discoveries/README.md).

## 🔍 Discovery Types

The library analyses recorded neuron activations and errors to propose mutation
candidates. Detection modules are grouped by concern — the authoritative
per-type reference (source module, tracking issue, candidate operations, and
status) lives in the
[Discovery Type Summary](docs/DISCOVERY_TYPES.md#discovery-type-summary):

- **⚡ Activation & Neuron State** — how neurons process activations: saturation,
  dead/oscillating/bimodal neurons, restricted or compressed output ranges,
  activation mismatches, and low-impact removal.
- **⚖️ Weight & Synapse** — synapse weights and connections: dormant, opposing,
  incoherent, or polarity-flipped weights, noise-to-signal pruning, and
  gradient-directed adjustments.
- **🏗️ Structural & Topology** — network structure: bottlenecks, correlated
  error, redundant/skip paths, symmetry breaking, co-adaptation, merged
  redundant neurons, epistatic pairs, and cross-detection synthesis.
- **📊 Range & Input Analysis** — input ranges and gating: bounded/sentinel
  ranges, observation utilisation, and input sensitivity.
- **🏆 Scoring & Recommendation** — proactive candidate scoring: output bias
  drift, sample-weighted discovery, add/remove neuron and synapse, and
  batch-successful grouping.

For each type's detection criteria, recommended actions, output format, and
production success rates, see [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md).

For impact calculation details, see
[docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md).

### ♻️ Remove-Neuron Weight-Redistribution Compensation

A hygiene-forced `removeNeuron` is usually **regressive** because NEAT-AI's
mean-preserving bias compensation cancels only the *mean* of the removed
neuron's downstream contribution. The `remove_neuron_compensation` module
(Issue #1559) recovers the surviving **per-sample variance** signal by folding
it into a correlated survivor's downstream weight — a perfectly correlated
survivor makes the removal fully compensable and non-regressive.

The live dispatch path wires this in (Issue #1689): every emitted sole-op
`RemoveNeuron` candidate that removes a variance-carrying neuron now carries a
`removeNeuronCompensation` block (optimal `Δw`, the covariance statistic, and the
`fullyCompensable` flag) so the applier redistributes weight instead of folding
the mean.

Routing between the two remedies is by **measured** constancy (Issue #1779): a
neuron whose recorded activations are constant within the fold's
evaluate-before-accept gate takes the #1623 bias fold — its removal candidate
carries a `constantNeuronBiasFold` block of per-target `biasDelta = w × c` values
and is promoted past the gain floor (#1622), because folding those deltas makes
the removal behaviour-preserving. Everything else takes redistribution. The
earlier gate asked for the *declared* `"constant"` neuron class, which no
`removeNeuron` producer emits, so the fold never fired.

```mermaid
flowchart TD
    A[Sole-op removeNeuron candidate] --> B[Per-sample activations<br/>DiscoverRecords]
    B --> C{"Fold gate: max residual<br/>|w·(a_i − c)| ≤ 1e-6?"}
    C -->|accepted| D[constantNeuronBiasFold<br/>biasDelta = w × c per target]
    D --> E[Promoted past the gain floor #1622]
    C -->|rejected / no records| F{Correlated shared-target<br/>survivor?}
    F -->|yes| G[removeNeuronCompensation<br/>optimal Δw + covariance]
    F -->|no| H[No remedy emitted —<br/>applier flags the removal]
```

The compact-covariance sufficient statistic, the `Δw`/residual maths, and the
evaluation flow are documented in
[docs/IMPACT_CALCULATION.md § Remove-Neuron Weight-Redistribution Compensation](docs/IMPACT_CALCULATION.md#remove-neuron-weight-redistribution-compensation-issue-1559).

## 🎯 Focus Selection

Discovery cannot evaluate *every* neuron within a run's budget, so each run
**focuses** on a small set (about half a dozen, ~6) of selectable neurons. That
focus list started as a *random* pick, moved to an **impact-weighted ranking**
(`rank_focus_neurons*`, `src/focus/`) that scores neurons by their estimated
effect on the output error, and now runs under a **wall-clock budget**
(`NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS`, default 120 s). On budget overrun
the crate returns a retryable `Timeout` and the caller falls back to an instant,
**error-guided** local ranking (not a literal random pick).

The full design — rationale, history, performance guard, fallback semantics, and
env knobs — is documented end-to-end in
[docs/FOCUS_SELECTION.md](docs/FOCUS_SELECTION.md).

## ⚙️ Configuration

### 🔧 Environment Variables

Every `NEAT_AI_DISCOVERY_*` tunable — with its default and description — is
maintained in one authoritative place: **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

That single reference replaces the table that used to live here (and a duplicate
in `AGENTS.md`) so the two can never drift apart again (Issue #1611).

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **Library not found** | Check artefact path, file extension (`.dylib` / `.so`), and `NEAT_AI_DISCOVERY_LIB_PATH` |
| **FFI permission errors** | Launch with `--allow-ffi --allow-env --allow-read --allow-write` |
| **Empty Parquet output** | Confirm caller supplies sampled discovery dataset with aligned observations, activations, and errors |
| **GPU not available** | Check system meets minimum requirements; on Linux check `/dev/dri` permissions |
| **`Memory check failed — discovery disabled` on a capable host** | On an ~8GB host the discovery runtime can hold most of the RAM, leaving free memory below the default floor (0.5GB macOS / 1.0GB Linux) every pass. Lower the floor with `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB=0.1` (or `0` to disable the gate). If the host is genuinely too small, exclude it at the scheduler rather than aborting every pass (Issue #1420). |
| **Out of memory (exit 137)** | Reduce `--max-old-space-size`; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |
| **Analysis timeout** | Expected under deadlines; coverage improves over repeated runs |
| **Synapse/neuron starvation** | Grep logs for `DEADLINE-BREAKDOWN` to see the per-cycle deadline-consumption breakdown, and `STARVED` for the curtailed-phase warning with skipped/total counts; the `starved` flag on `synapseMetadata`/`neuronMetadata` exposes the same signal programmatically |
| **GPU timeout errors** | Reduce `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE`; restart if GPU driver hung |
| **Low GPU utilisation** | Often CPU-bound sample building; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |
| **Deadlock or stuck** | Send `kill -USR1 <pid>` for thread dump; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |

For detailed troubleshooting steps, see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md).

## 🔗 Using the Library with NEAT-AI

1. Place the compiled artefact where Deno can load it:
   - Copy `libneat_ai_discovery.*` into `~/.cargo/lib`, **or**
   - Export `NEAT_AI_DISCOVERY_LIB_PATH=/absolute/path/to/libneat_ai_discovery.*`.
2. Grant FFI permissions when running discovery jobs:
   ```bash
   deno run --allow-env --allow-ffi --allow-read --allow-write your-script.ts
   ```
3. From your controller, guard calls with `isRustDiscoveryEnabled()` so the job
   fails fast if the module cannot be loaded.
4. Follow the end-to-end discovery orchestration documented in the
   [`DiscoveryDir` guide](https://github.com/stSoftwareAU/NEAT-AI/blob/main/docs/DiscoveryDir.md).

### Verifying the Installation

Use the NEAT-AI helper script after copying the library:

```bash
cd /path/to/NEAT-AI
./scripts/check_discovery.ts
```

If the script reports that discovery is enabled, you are ready to schedule
`Creature.discoveryDir()` jobs against your sampled datasets.

## 💻 Development

For prerequisites, building, testing, code style, and the full development
workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

> **For AI agents**: coding conventions and the testing philosophy live in
> [CONTRIBUTING.md](CONTRIBUTING.md); the agent-only invariants and rules live in
> [AGENTS.md](AGENTS.md).

**Quick reference:**

```bash
# Build and install
./scripts/runlib.sh

# Quality gate (run before every commit)
./quality.sh

# Run all tests
cargo test --lib --tests --all-features -- --test-threads=2

# Run benchmarks
cargo bench --bench <bench_name>

# Check documentation builds without warnings
./scripts/doc-check.sh

# Run fuzz tests (requires nightly toolchain and cargo-fuzz)
./scripts/fuzz-ci.sh            # 30s per target (default)
./scripts/fuzz-ci.sh 60         # 60s per target
```

### 📊 Studying the Candidates Cache

The production discovery cache holds one JSON record per candidate the
controller evaluated, filed under `success|failures/<model-hash>/<strategy>/`.
The `study_candidates_cache` example turns that corpus into a Markdown report
covering **volume** (records per day, model hash and strategy) and **gain size**
(which record fields correlate with a bigger `scoreDelta`):

```bash
cargo run --example study_candidates_cache -- /path/to/discovery-cache-checkout
```

The live tree only holds the current model hash — periodic "Clean up OLD
discovery caches" commits delete earlier ones — so by default the corpus is
widened with records recovered from git history. Pass `--no-history` to study
the working tree alone.

See [docs/analysis/candidates-cache-study-1920.md](docs/analysis/candidates-cache-study-1920.md)
for the first run's findings, and
[docs/analysis/neuron-ranking-score-1924.md](docs/analysis/neuron-ranking-score-1924.md)
for the add-neuron ranking change those findings produced.

### 🔀 Fuzz Testing

The `fuzz/` directory contains [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets that exercise the FFI JSON boundary with arbitrary inputs. This helps
catch panics from malformed, truncated, or adversarial JSON before they reach
production.

**Prerequisites:**

```bash
# Install cargo-fuzz (one-time setup)
cargo install cargo-fuzz

# Ensure the nightly toolchain is available
rustup toolchain install nightly
```

**Available targets:**

| Target | Description |
|--------|-------------|
| `fuzz_ffi_deserialisation` | Fuzzes `serde_json::from_str` for all FFI input types |
| `fuzz_ffi_entry_points` | Fuzzes the `*_internal` business-logic functions |

**Running:**

```bash
# Run all fuzz targets via the CI helper script (30s each by default)
./scripts/fuzz-ci.sh

# Run with a custom duration (60s per target)
./scripts/fuzz-ci.sh 60

# Run a specific target directly (--locked honours the committed fuzz/Cargo.lock)
cargo +nightly fuzz run --locked fuzz_ffi_deserialisation -- -max_total_time=60

# Run with a maximum input length of 4096 bytes
cargo +nightly fuzz run --locked fuzz_ffi_entry_points -- -max_total_time=60 -max_len=4096

# List all available fuzz targets
cargo +nightly fuzz list
```

Crash-reproducing inputs (if any) are saved to `fuzz/artifacts/`. The fuzzer
corpus is stored in `fuzz/corpus/` and grows over successive runs.

## 💡 Why Use This Library?

- **Production-ready discovery** — Handles millions of observations without the
  memory blow-outs that limit the TypeScript implementation.
- **Single-file artefacts** — Writes per-run Parquet files so results are easy to
  transfer, archive, or inspect with standard tooling.
- **Drop-in for NEAT-AI** — Exposes the `libneat_ai_discovery` symbol set expected
  by the TypeScript bindings in `NEAT-AI`.

<details>
<summary>Original project brief and scale targets (kept for contributors)</summary>

### Goal

Record neuron activations and errors during the discovery training phase, then scan
this recorded data to identify **high-quality mutation candidates** that are expected
to improve the creature's score — using GPU/SIMD acceleration.

### Problem Statement

The current DenoJS implementation requires extreme filtering of the training
data (millions of records) to make discovery work in reasonable time. This Rust
library solves the performance/memory problems while maintaining the same
functional behaviour.

**Target Scale:**
- Training records: Millions
- Observations per record: 1,486 (float32 values)
- Neurons: 447
- Synapses: 16,012

### Performance Requirements

- Must handle millions of training records efficiently without memory issues
- Must process significantly more data than DenoJS can handle
- Must be significantly faster than the TypeScript implementation
- Must use minimal memory (avoid loading all data into memory at once)

### Features

- Record neuron activations and errors during discovery training phase
- Single Parquet file format (eliminates many-small-files problem)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging
- Cross-platform support (macOS, Ubuntu, AWS Linux)

</details>

## 🌏 Cross-Platform Support

The library must work on:
- **macOS** (primary target)
- Ubuntu
- AWS Linux (x86_64 and ARM64)

All dependencies build automatically on remote, unattended machines.

## 📦 Distributed Build & Versioning

- Versions are managed in `Cargo.toml`. The `version-increment` CI job
  auto-bumps the patch version on **every pull request** (unless the PR branch
  already carries a bump), not only when `src/` changes — see
  `.github/workflows/ci.yml`.
- Local and remote runs use a distributed build pattern via `scripts/runlib.sh`:
  the library is installed to `~/.cargo/lib/` and tracked with a version marker at
  `~/.cargo/lib/.neat_ai_discovery.version`.
- In the normal PR workflow you do not need to bump the version yourself — CI
  does it. If you commit **directly** (outside the PR workflow, where CI does not
  run), you **must** manually increment the patch version so cached builds pick
  up your change. This is the single authoritative version-bump policy; other
  docs link here.

## 🔗 Related Repositories

The NEAT-AI project is split across seven public repositories. Each focuses on one concern and composes with the others as shown below. This repository is **NEAT-AI-Discovery**.

| Repository | Role |
|------------|------|
| [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) | Primary Deno/TypeScript neural-network engine (evolution, training, WASM activation). |
| [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) | Shared native Rust library (`neat-core`) with numerics, topology helpers, and the chunked `.bin` training stream. |
| [NEAT-AI-Discovery](https://github.com/stSoftwareAU/NEAT-AI-Discovery) | Rust discovery module invoked by NEAT-AI via Deno FFI to propose structural/bias/weight/squash mutation candidates. |
| [NEAT-AI-Snapshot](https://github.com/stSoftwareAU/NEAT-AI-Snapshot) | Creature/genome snapshot format and fixtures produced by NEAT-AI and consumed by downstream tools. |
| [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) | Production forward-only scoring application built on `neat-core` via a path dependency. |
| [NEAT-AI-Explore](https://github.com/stSoftwareAU/NEAT-AI-Explore) | Visualiser for creatures that reads NEAT-AI-Snapshot data. |
| [NEAT-AI-Examples](https://github.com/stSoftwareAU/NEAT-AI-Examples) | Worked examples and tutorials that depend on NEAT-AI. |

### Dependency graph

```mermaid
graph TD
    Core[NEAT-AI-core<br/>Rust shared lib]
    Main[NEAT-AI<br/>Deno/TypeScript engine]
    Discovery[NEAT-AI-Discovery<br/>Rust, via Deno FFI]
    Snapshot[NEAT-AI-Snapshot<br/>creature data]
    Scorer[NEAT-AI-scorer<br/>Rust scorer app]
    Explore[NEAT-AI-Explore<br/>visualiser]
    Examples[NEAT-AI-Examples<br/>tutorials]

    Main -->|Deno FFI| Discovery
    Main -->|produces| Snapshot
    Scorer -->|path dependency| Core
    Explore -->|reads| Snapshot
    Examples -->|depends on| Main
```

## 📚 Additional Documentation

| Document | Description |
|----------|-------------|
| [docs/discoveries/](docs/discoveries/README.md) | Visual discovery scenario guides with diagrams and examples |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development guidelines for contributors |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version history of changes |
| [AGENTS.md](AGENTS.md) | Agent-only invariants and rules (conventions live in CONTRIBUTING.md) |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Authoritative reference for every `NEAT_AI_DISCOVERY_*` environment variable |
| [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) | All discovery types with success/failure rates |
| [docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md) | Neuron impact calculation details |
| [docs/FOCUS_SELECTION.md](docs/FOCUS_SELECTION.md) | Focus-selection design end-to-end: why ~6 neurons, random → impact-weighted ranking, the wall-clock budget guard, and the error-guided fallback |
| [docs/ANALYSIS_DEEP_DIVE.md](docs/ANALYSIS_DEEP_DIVE.md) | Detailed analysis workflow and detection algorithms |
| [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) | GPU performance tuning, troubleshooting, and debugging |
| [docs/FFI_API.md](docs/FFI_API.md) | Full FFI API reference and JSON interface |
| [docs/COST_FUNCTION_NOTES.md](docs/COST_FUNCTION_NOTES.md) | Cost-agnostic invariants, per-cost behaviour catalogue, and per-consumer audit |
| [docs/STREAMING_GUIDE.md](docs/STREAMING_GUIDE.md) | Step-by-step streaming recording API guide with TypeScript examples |
| [docs/CACHE_TUNING.md](docs/CACHE_TUNING.md) | Cache tier tuning, diagnostics, and example configurations |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Benchmark regression tracking and comparison workflow |
| [docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md](docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md) | MCMC applicability audit for candidate selection pipeline |
| [docs/DROUGHT_PLAYBOOK.md](docs/DROUGHT_PLAYBOOK.md) | Operator playbook for diagnosing "no successful candidates" droughts (suppression layers, regimes, env vars) |
| [docs/analysis/candidates-cache-study-1920.md](docs/analysis/candidates-cache-study-1920.md) | Findings from the production candidates-cache study: candidate volume and gain-size predictors |
| [docs/analysis/neuron-ranking-score-1924.md](docs/analysis/neuron-ranking-score-1924.md) | The reliability-weighted add-neuron rank score, and the before/after correlation that accepts it |
| [CodeWiki](https://codewiki.google/github.com/stsoftwareau/neat-ai-discovery) | AI-powered documentation and code exploration |

## 📄 Licence

This project is licensed under the terms of the Apache License 2.0. For the full
license text, please see [LICENSE](./LICENSE).
