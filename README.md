# NEAT-AI-Discovery

A high-performance Rust companion library for
[`stSoftwareAU/NEAT-AI`](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend **mutation candidates** (add/remove/modify) that are likely
to improve the creature's score.

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
   a new candidate type is truly required it must be documented and a corresponding
   handler added to NEAT-AI.

> **In short:** discover score-improving mutations, use GPU/SIMD to do it quickly,
> and reuse the candidate types that NEAT-AI already understands.

## TL;DR

- **This library finds candidates, it does not "auto-fix" creatures**: NEAT-AI validates candidates by rescoring on the full training set.
- **GPU required**: discovery is skipped when no compatible GPU is available (see `check_gpu_available()`).
- **Build/install**: `./scripts/runlib.sh` (installs to `~/.cargo/lib/` with version tracking).
- **Preferred recording API**: streaming (`start_discovery_session` → `append_discovery_records` → `finish_discovery_session`) to avoid JS/V8 string limits.
- **Free FFI results**: every FFI call returning a `char*` must be freed with `free_discovery_result()`.

## Quick Start

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

   **From NEAT-AI directory**, you can call this script directly:

   ```bash
   ../NEAT-AI-Discovery/scripts/runlib.sh
   ```

3. Confirm the artefact exists at `~/.cargo/lib/libneat_ai_discovery.*`.
4. **Always run the quality gate before committing** (CI treats warnings as errors):

   ```bash
   ./quality.sh
   ```

## FFI API Summary

The library exposes a Deno FFI-friendly symbol set. The authoritative list lives in
`src/lib.rs` as `#[no_mangle] pub extern "C"` functions.

| Category | Entry Points |
|----------|-------------|
| **GPU probe** | `check_gpu_available()` |
| **Version probe** | `get_library_version()` |
| **Recording (streaming)** | `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, `cancel_discovery_session` |
| **Recording (single-call)** | `record_discovery` (avoid for large runs) |
| **Analysis** | `rank_focus_neurons`, `analyze_parallel` |
| **Utilities** | `merge_discovery_parquet`, `read_discovery_records_ffi`, `export_visualisation_snapshot` |
| **Memory management** | `free_discovery_result` |

For the full JSON interface specification and streaming API details, see
[docs/FFI_API.md](docs/FFI_API.md).

## GPU Requirement

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
| **Available RAM** | 1 GB | Prevents hangs from memory pressure/swap thrashing |
| **GPU** | Metal (macOS) or Vulkan (Linux) | Required for compute shaders |

When requirements aren't met, `check_gpu_available()` returns `gpuAvailable: false`
with a descriptive reason. NEAT-AI's evolution process continues normally — only
the discovery optimisation is skipped.

For GPU performance tuning, troubleshooting, and debugging, see
[docs/GPU_GUIDE.md](docs/GPU_GUIDE.md).

## Discovery → Evolution Pipeline

The discovery process works as follows:

```
┌───────────────────────────────────────────────────────────────────────┐
│  RUST (this library)                                                  │
│  1. Find ALL candidates with positive expected improvement            │
│  2. Apply impact discounting (creature-level predictions)             │
│  3. Sort by expected improvement (best first)                         │
│  4. Randomise within top-K under deadlines to avoid starvation        │
│  5. Return candidates (optionally limited by max_candidates)          │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  TYPESCRIPT (NEAT-AI)                                                 │
│  1. Receive candidates from Rust (e.g., 100 candidates)               │
│  2. Select top N based on available CPUs (e.g., 10-20)                │
│  3. Re-score each candidate IN PARALLEL (apply mutation, measure)     │
│  4. Keep candidates that ACTUALLY improve the creature's score        │
│  5. Return improved creatures to population                           │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  EVOLUTION                                                            │
│  • Improved creatures compete in the population                       │
│  • Natural selection breeds out unsuccessful mutations                │
│  • No manual filtering needed — evolution handles it                  │
└───────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Rust finds structural improvements that reduce error. TypeScript
validates by measuring actual score. Evolution does the rest.

For detailed analysis workflow, coordinated structural discovery, discrete
activation function handling, and detection algorithms, see
[docs/ANALYSIS_DEEP_DIVE.md](docs/ANALYSIS_DEEP_DIVE.md).

## Discovery Scenarios

For a **visual, beginner-friendly overview** of every discovery scenario — with
diagrams, worked examples, and links to research papers — see the
[Discovery Scenarios Guide](docs/discoveries/README.md).

## Discovery Types

The library analyses recorded neuron activations and errors to propose mutation
candidates. Each discovery type targets a specific network pathology:

| Discovery Type | What It Detects | Candidate Operations |
|----------------|----------------|---------------------|
| [Saturated Neuron](docs/DISCOVERY_TYPES.md#saturated-neuron-detection) | Neurons stuck at activation bounds | `changeSquash`, `setBias` |
| [Bottleneck Neuron](docs/DISCOVERY_TYPES.md#bottleneck-neuron-detection) | Information bottlenecks (high fan-in) | `addNeuron`, `addSynapse` |
| [Dead Neuron](docs/DISCOVERY_TYPES.md#dead-neuron-detection) | Neurons with near-zero activation | `removeNeuron` |
| [Dormant Synapse](docs/DISCOVERY_TYPES.md#dormant-synapse-detection) | Synapses with near-zero weight | `removeSynapse` |
| [Opposing Synapse](docs/DISCOVERY_TYPES.md#opposing-synapse-detection) | Synapses increasing error | `removeSynapse`, `setWeight` |
| [Output Bias Drift](docs/DISCOVERY_TYPES.md#output-bias-drift-detection) | Output neurons with systematic bias | `setBias` |
| [Oscillating Neuron](docs/DISCOVERY_TYPES.md#oscillating-neuron-detection) | Neurons oscillating between ± values | `changeSquash`, `setBias` |
| [Correlated Error](docs/DISCOVERY_TYPES.md#correlated-error-pattern-detection) | Outputs with shared error patterns | `addNeuron`, `addSynapse` |
| [Multi-Hop](docs/DISCOVERY_TYPES.md#multi-hop-candidate-analysis) | Deeper structural improvements | `addNeuron`, `addSynapse` |
| [Redundant Path](docs/DISCOVERY_TYPES.md#redundant-path-pruning) | Duplicate paths to same target | `removeSynapse`, `setWeight` |
| [Add Neurons](docs/DISCOVERY_TYPES.md#add-neurons) | Beneficial intermediate neurons | `addNeuron` |
| [Add Synapses](docs/DISCOVERY_TYPES.md#add-synapses) | Beneficial direct connections | `addSynapse` |
| [Remove Low-Impact](docs/DISCOVERY_TYPES.md#remove-low-impact-neurons) | Neurons below cost of growth | `removeNeuron` |

For detection criteria, recommended actions, output format, and production
success rates, see [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md).

For impact calculation details, see
[docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_LIB_PATH` | `~/.cargo/lib/` | Path to the compiled library |
| `NEAT_AI_DISCOVERY_VERBOSE` | off | Enable verbose logging |
| `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` | auto | GPU batch size (64–4096) |
| `NEAT_AI_DISCOVERY_GPU_TIMING` | off | Enable GPU kernel profiling |
| `NEAT_AI_DISCOVERY_QUIET_GPU` | off | Suppress Mesa/libEGL debug output |
| `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` | adaptive | Streaming parquet block cache limit |
| `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` | 2 | Streaming prefetch depth |
| `NEAT_AI_DISCOVERY_PRELOAD_ALL` | off | Force full parquet preload |
| `NEAT_AI_DISCOVERY_BLOCK_SIZE` | 10000 | Streaming block size in records |
| `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` | off | Enable outlier-focused analysis |
| `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` | 90 | Outlier identification threshold |
| `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY` | off | Force output-only focus targets |
| `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` | off | Bias toward newer input indices |
| `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS` | off | Prioritise unused input neurons |
| `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` | dynamic | Constant-source folding threshold |
| `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` | off | Stall watchdog timeout |
| `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` | 2 | Delay between dump and abort |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Library not found** | Check artefact path, file extension (`.dylib` / `.so`), and `NEAT_AI_DISCOVERY_LIB_PATH` |
| **FFI permission errors** | Launch with `--allow-ffi --allow-env --allow-read --allow-write` |
| **Empty Parquet output** | Confirm caller supplies sampled discovery dataset with aligned observations, activations, and errors |
| **GPU not available** | Check system meets minimum requirements; on Linux check `/dev/dri` permissions |
| **Out of memory (exit 137)** | Reduce `--max-old-space-size`; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |
| **Analysis timeout** | Expected under deadlines; coverage improves over repeated runs |
| **GPU timeout errors** | Reduce `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE`; restart if GPU driver hung |
| **Low GPU utilisation** | Often CPU-bound sample building; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |
| **Deadlock or stuck** | Send `kill -USR1 <pid>` for thread dump; see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |

For detailed troubleshooting steps, see [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md).

## Using the Library with NEAT-AI

1. Place the compiled artefact where Deno can load it:
   - Copy `libneat_ai_discovery.*` into `~/.cargo/lib`, **or**
   - Export `NEAT_AI_DISCOVERY_LIB_PATH=/absolute/path/to/libneat_ai_discovery.*`.
2. Grant FFI permissions when running discovery jobs:
   ```bash
   deno run --allow-env --allow-ffi --allow-read your-script.ts
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

## Development

For prerequisites, building, testing, code style, and the full development
workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

> **For AI agents**: Machine-readable coding conventions and invariants live in
> [AGENTS.md](AGENTS.md).

**Quick reference:**

```bash
# Build and install
./scripts/runlib.sh

# Quality gate (run before every commit)
./quality.sh

# Run all tests
cargo test --lib --tests --all-features -- --test-threads=1

# Run benchmarks
cargo bench --bench <bench_name>

# Run fuzz tests (requires nightly toolchain and cargo-fuzz)
cargo +nightly fuzz run fuzz_ffi_deserialisation -- -max_total_time=60
cargo +nightly fuzz run fuzz_ffi_entry_points -- -max_total_time=60
```

### Fuzz Testing

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
# Run a specific target for 60 seconds
cargo +nightly fuzz run fuzz_ffi_deserialisation -- -max_total_time=60

# Run with a maximum input length of 4096 bytes
cargo +nightly fuzz run fuzz_ffi_entry_points -- -max_total_time=60 -max_len=4096

# List all available fuzz targets
cargo +nightly fuzz list
```

Crash-reproducing inputs (if any) are saved to `fuzz/artifacts/`. The fuzzer
corpus is stored in `fuzz/corpus/` and grows over successive runs.

## Why Use This Library?

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

## Cross-Platform Support

The library must work on:
- **macOS** (primary target)
- Ubuntu
- AWS Linux (x86_64 and ARM64)

All dependencies build automatically on remote, unattended machines.

## Distributed Build & Versioning

- Versions are managed in `Cargo.toml` and automatically incremented by CI when
  `src/` changes are detected.
- Local and remote runs use a distributed build pattern via `scripts/runlib.sh`:
  the library is installed to `~/.cargo/lib/` and tracked with a version marker at
  `~/.cargo/lib/.neat_ai_discovery.version`.
- Do not manually edit version numbers; CI handles patch bumps.

## Additional Documentation

| Document | Description |
|----------|-------------|
| [docs/discoveries/](docs/discoveries/README.md) | Visual discovery scenario guides with diagrams and examples |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development guidelines for contributors |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version history of changes |
| [AGENTS.md](AGENTS.md) | Coding guidelines for AI agents |
| [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) | All discovery types with success/failure rates |
| [docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md) | Neuron impact calculation details |
| [docs/ANALYSIS_DEEP_DIVE.md](docs/ANALYSIS_DEEP_DIVE.md) | Detailed analysis workflow and detection algorithms |
| [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) | GPU performance tuning, troubleshooting, and debugging |
| [docs/FFI_API.md](docs/FFI_API.md) | Full FFI API reference and JSON interface |
| [CodeWiki](https://codewiki.google/github.com/stsoftwareau/neat-ai-discovery) | AI-powered documentation and code exploration |

## License

This project is licensed under the terms of the Apache License 2.0. For the full
license text, please see [LICENSE](./LICENSE).
