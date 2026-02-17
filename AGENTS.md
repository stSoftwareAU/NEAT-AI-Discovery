# AGENTS.md — Coding Guidelines for AI Agents

This file is the single source of truth for AI coding agents working in this
repository. For user-facing documentation, see [README.md](README.md).

---

## 1. Project Overview

**NEAT-AI-Discovery** is a high-performance Rust companion library for
[stSoftwareAU/NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend **mutation candidates** (add/remove/modify) that are likely
to improve the creature's score.

### Sole Mission

> **Discover changes that improve the creature's score — as fast as possible.**

1. **Improve the creature's score.** Only return candidates with a positive
   expected improvement.
2. **Discover improvements as fast as possible.** Leverage GPU compute shaders
   and SIMD so large creatures can be analysed in seconds.
3. **Minimise changes to NEAT-AI.** Reuse existing candidate types whenever
   possible (see [Candidate Types](#11-candidate-types) below).

This library does **not** directly "fix" a creature. NEAT-AI validates each
candidate by cloning the creature, applying the mutation, and re-scoring against
the full training set. Only candidates that measurably improve the score are
admitted back into the population.

---

## 2. Architecture

### Source Layout

```
src/
├── lib.rs                    # Module declarations, re-exports, version init
├── ffi.rs                    # FFI entry points (no_mangle extern "C")
├── ffi_types.rs              # JSON request/response structs for FFI boundary
├── ffi_internal.rs           # Internal business-logic functions for FFI
├── types.rs                  # Core type definitions
├── activations.rs            # Activation function calculations
├── record.rs                 # Discovery data recording
├── streaming.rs              # Streaming session management
├── parquet_format.rs         # Parquet I/O
├── export.rs                 # Data export utilities
├── discovery_history.rs      # Historical tracking
├── debug.rs                  # Debugging utilities
├── observability.rs          # Observability / logging
├── intern.rs                 # Neuron UUID interning
├── watchdog.rs               # Signal handling (SIGUSR1)
├── focus/                    # Focus neuron selection (Issue #491)
│   ├── mod.rs                # Public API, re-exports, module declarations
│   ├── layers.rs             # Network layer computation via BFS
│   ├── allocation.rs         # Budget allocation strategies
│   ├── gradient.rs           # Gradient flow analysis
│   ├── impact.rs             # Impact calculation (squash-aware)
│   ├── ranking/              # Neuron ranking and selection (Issue #564)
│   │   ├── mod.rs            # Public API, orchestration, rank_focus_neurons
│   │   ├── record_providers.rs # Record provider trait, eager/lazy implementations
│   │   ├── score_calculation.rs # Neuron score computation, frequency, variance
│   │   └── removal_candidates.rs # Removal candidates, SynapseCounts, constant removal
│   └── tests.rs              # Unit tests for internal components
│
├── analysis/                 # Core analysis engine
│   ├── mod.rs                # Module organisation, re-exports
│   ├── orchestration.rs      # Top-level analyse_all dispatch (Issue #562)
│   ├── candidate_aggregation.rs # Candidate merging and post-processing (Issue #562)
│   ├── module_dispatch_specs/   # Discovery module spec builders (Issue #562, #595)
│   │   ├── mod.rs              # Public API, build_discovery_module_specs, orchestration
│   │   ├── neuron_specs.rs     # Neuron-focused dispatch specs
│   │   ├── synapse_specs.rs    # Synapse-focused dispatch specs
│   │   ├── structural_specs.rs # Structural discovery dispatch specs
│   │   └── scoring_specs.rs    # Scoring and recommendation dispatch specs
│   ├── constants.rs          # Central discovery thresholds (Issue #424)
│   ├── shared.rs             # Common types, results, diagnostics
│   ├── synapse/              # Synapse analysis (Issue #482)
│   │   ├── mod.rs            # Public API, entry points, orchestration
│   │   ├── target_analysis.rs # Per-target analysis loop
│   │   ├── scoring.rs        # Improvement calculation, boosting
│   │   ├── gpu_evaluation.rs # GPU batch orchestration
│   │   ├── candidate_generation.rs # Sample building, locality grouping
│   │   ├── filtering.rs      # Candidate filtering, deduplication
│   │   ├── structural_patterns.rs # Coordinated structural discovery
│   │   └── post_processing.rs # Impact discounting, sorting, metadata
│   ├── neuron.rs             # Neuron analysis
│   ├── activation.rs         # Activation function analysis
│   ├── samples.rs            # Sample data structures
│   ├── diagnostics/          # Diagnostic tracking (Issue #524)
│   │   ├── mod.rs            # Public API, re-exports, impact scoring adapter
│   │   ├── rejection.rs      # Synapse rejection tracking and reporting
│   │   ├── neuron_tracking.rs # Neuron rejection tracking and reporting
│   │   ├── target_data.rs    # Target data structures for sample building
│   │   └── focus_filter.rs   # Focus target filtering and validation
│   ├── detection/            # Pattern detection modules (Issue #528)
│   │   ├── mod.rs            # Module declarations
│   │   ├── saturation.rs     # Saturated neuron detection
│   │   ├── bottleneck.rs     # Bottleneck neuron detection
│   │   ├── dead_neuron.rs    # Dead neuron detection
│   │   ├── dormant_synapse.rs # Dormant synapse detection
│   │   ├── opposing_synapse.rs # Opposing synapse detection
│   │   ├── oscillating_neuron.rs # Oscillating neuron detection
│   │   ├── correlated_error.rs # Correlated error patterns
│   │   ├── redundant_path.rs # Redundant path detection
│   │   ├── bounded_range.rs  # Bounded range detection
│   │   ├── observation_range.rs # Observation effective range
│   │   ├── sentinel_gating.rs # Sentinel value gating
│   │   ├── restricted_range.rs # Restricted activation range
│   │   ├── operating_point.rs # Hidden neuron operating point
│   │   ├── unbounded_capping.rs # Unbounded activation capping
│   │   ├── noise_signal.rs   # Noise-to-signal ratio detection
│   │   ├── input_sensitivity.rs # Input sensitivity analysis
│   │   ├── topology.rs       # Topology-aware structure analysis
│   │   └── weight_coherence.rs # Weight coherence validation
│   ├── recommendation/       # Candidate recommendation modules (Issue #528)
│   │   ├── mod.rs            # Module declarations
│   │   ├── activation_recommendation.rs # Activation function recommendation
│   │   ├── output_bias_drift.rs # Output bias drift detection
│   │   ├── epistatic/         # Epistatic interaction analysis (Issue #563)
│   │   │   ├── mod.rs            # Public API, types, re-exports
│   │   │   ├── candidate_generation.rs # Pair generation, complementarity, conversion
│   │   │   ├── pre_screening.rs  # Residual analysis, synergistic detection
│   │   │   ├── deduplication.rs  # Dominant-neuron deduplication (Issue #509)
│   │   │   └── scoring.rs       # Interference detection, filtering (Issue #415)
│   │   ├── multi_hop.rs      # Multi-hop candidate analysis
│   │   ├── gradient_discovery.rs # Gradient-based synapse adjustment
│   │   └── sample_weighted.rs # Sample-weighted discovery
│   ├── scoring/              # Scoring and confidence modules (Issue #528)
│   │   ├── mod.rs            # Module declarations
│   │   ├── confidence.rs     # Confidence metrics
│   │   ├── weights.rs        # Weight analysis and calculation
│   │   ├── error_distribution.rs # Error distribution stats
│   │   └── cross_validation.rs # Cross-validation scoring
│   ├── cache/                # Record caching (Issue #565)
│   │   ├── mod.rs            # Public API, RecordCache, re-exports
│   │   ├── loading_strategy.rs # LoadingStrategy, select_loading_strategy
│   │   ├── lru_cache.rs      # LruRecordCache, LruCacheStats, eviction
│   │   ├── compressed_cache.rs # CompressedLruRecordCache (LZ4)
│   │   ├── tiered_cache.rs   # TieredRecordCache, auto strategy selection
│   │   └── serialisation.rs  # Binary serialisation, CompressedCacheEntry
│   ├── streaming.rs          # Streaming parquet loading
│   ├── discovery_dispatch.rs # Generic discovery module dispatch (Issue #375)
│   ├── candidate_clustering.rs # Redundancy reduction
│   ├── early_termination.rs  # SPRT-based early stopping
│   ├── implementation_tests/ # Synapse analysis pipeline tests
│   │
│   ├── gpu/                  # GPU infrastructure (Issue #520)
│   │   ├── mod.rs
│   │   ├── device.rs         # GPU device management
│   │   ├── analyzer.rs       # Core GpuAnalyzer struct, initialisation, shared logic
│   │   ├── helpful_evaluation.rs  # Helpful synapse GPU evaluation
│   │   ├── harmful_evaluation.rs  # Harmful synapse GPU evaluation
│   │   ├── relu_evaluation.rs     # ReLU activation GPU evaluation
│   │   ├── activation_evaluation.rs # Activation function GPU evaluation
│   │   ├── bias_evaluation.rs     # Bias GPU evaluation
│   │   ├── queue.rs          # GPU work queue
│   │   └── shaders.rs        # Shader management
│   │
│   └── utils/                # Utilities
│       ├── mod.rs
│       ├── memory.rs         # Memory detection
│       ├── deadline.rs       # Deadline handling
│       └── platform.rs       # Platform setup
│
└── shaders/                  # WGSL compute shaders
    ├── activation.wgsl
    ├── helpful.wgsl / helpful_reduce.wgsl
    ├── harmful.wgsl / harmful_reduce.wgsl
    ├── bias.wgsl
    ├── matching.wgsl
    └── relu.wgsl
```

### Other Key Directories

| Directory | Purpose |
|-----------|---------|
| `tests/` | Integration tests (~97 files) |
| `benches/` | Criterion benchmarks (7 suites) |
| `examples/` | Standalone examples (parquet inspection, snapshot generation) |
| `scripts/` | Build and install helpers (`runlib.sh`) |
| `docs/` | Supplementary documentation and PR summaries |

### Library Type

The crate produces both `cdylib` (for FFI via Deno) and `rlib` (for Rust
integration). See `Cargo.toml` for the full dependency list.

---

## 3. Coding Conventions

### Language

**Use Australian English** throughout all code, comments, and documentation:
- colour, behaviour, organisation, favour, metre, centre, analyse, minimise,
  optimise, serialise, initialise, utilise, recognise, emphasise, prioritise,
  licence (noun), defence, modelling, travelling, programme (except "program"
  in computing contexts)

### Principles

- **KISS** — Favour simplicity; avoid unnecessary complexity.
- **DRY** — Avoid duplication; maintain a single source of truth.
- **Boy Scout Rule** — Leave the code cleaner than you found it.
- **Prefer smaller files** — Favour many smaller, focused source files over
  large monolithic ones (Single Responsibility Principle). Target individual
  source files under ~1,500 lines; consider splitting at ~2,000 lines.
- **Separate concerns** — GPU infrastructure, business logic, and types belong
  in separate files.

### Rust Best Practices

- Follow standard Rust idioms (`Result`, `Option`, pattern matching).
- Use `anyhow` for error handling in application code.
- Prefer Rust standard library features over external crates when possible.
- All dependencies must be Apache-2.0 compatible (see
  [README.md — Dependency License Requirements](README.md#development-guidelines)
  for the full list of allowed licences).

### Avoid Over-engineering

- Only make changes that are directly requested or clearly necessary.
- Do not add features, refactor code, or make "improvements" beyond what was
  asked.
- Do not add error handling for scenarios that cannot happen.
- Do not create helpers or abstractions for one-time operations.
- Do not add docstrings, comments, or type annotations to code you did not
  change.

---

## 4. Testing Philosophy

**The quality of tests is what makes a good system.**

### Test-Driven Development (TDD)

1. Write a failing test for the new feature.
2. Implement the feature to make the test pass.
3. Refactor if needed while keeping tests green.

### Test Outcomes, Not Implementation ("What" vs "How")

Tests verify **what** the system does, not **how** it does it. The same test
should pass regardless of whether we use GPU, CPU, or TPU internally.

**"What" tests (GOOD)** — test observable outcomes:
- Call a function with test data, assert on the result.
- Verify the system produces correct candidates, detects patterns, returns
  expected JSON structure, etc.
- Will still pass if we switch quick sort to bubble sort, HashMap to BTreeMap,
  or GPU to CPU.

**"How" tests (BAD)** — test implementation details:
- Assert that a specific internal function is called.
- Check that a particular data structure is used internally.
- Verify iteration order, internal cache state, or call counts for
  non-observable behaviour.
- Break on any refactor even when behaviour is unchanged.

**Benchmarks disguised as tests (BAD)** — measure performance in unit tests:
- Loop N times and assert on elapsed time.
- Compare durations between two code paths.
- Use `Instant::now()` / `.elapsed()` to validate speed.
- These always produce unreliable results because tests run in parallel with
  other system activity.

```rust
// GOOD: Tests the outcome ("what")
#[test]
fn test_low_impact_neurons_are_detected() {
    let creature = create_test_creature();
    let impacts = compute_impacts(&creature);
    assert!(impacts["far-from-output"] < 0.1);
    assert!(impacts["close-to-output"] > 0.9);
}

// BAD: Tests implementation details ("how")
#[test]
fn test_gpu_kernel_computes_impacts() {
    // Don't test HOW we compute, test WHAT we compute
}

// BAD: Benchmark disguised as a test
#[test]
fn test_cache_is_fast() {
    let start = Instant::now();
    for _ in 0..10000 { cache.get("key"); }
    assert!(start.elapsed().as_millis() < 50); // Unreliable!
}
```

### Unit Tests vs Benchmarks

| Concern | Unit Tests | Benchmarks |
|---------|-----------|------------|
| **Purpose** | Verify correctness | Measure performance |
| **Location** | `tests/` (integration) or `src/` with `#[cfg(test)]` | `benches/` |
| **Run with** | `cargo test` | `cargo bench --bench <name>` |
| **Asserts on** | Results, structure, correctness | Timing, throughput |
| **Must not** | Use `Instant`/`elapsed` for pass/fail | Verify correctness |

**Why this matters**: Unit tests run in parallel with other tests and system
activity, making timing measurements unreliable. If you switch quick sort to
bubble sort, unit tests should still pass — but a benchmark would correctly
show the regression. Put timing assertions in `benches/`, not `tests/`.

**Never reduce iteration counts to make "performance tests" faster in unit
tests.** If you need to confirm performance, create proper benchmarks.

### Test Organisation

- **Prefer `tests/` directory** (integration tests) over inline unit tests.
- Only place tests under `src/` when the behaviour cannot be exercised cleanly
  via the public API.
- If unit tests in `src/` grow large, extract them into a dedicated `tests.rs`
  module file.
- Do not make APIs public just for testing.
- Group related tests by concern (e.g., all weight tests in one file).

### Test Change Significance

- **New test file** — generally good (more coverage).
- **Modified test** — requires justification (did requirements change?).
- **Removed/skipped test** — red flag; must be justified.

Tests run sequentially (`--test-threads=1`) due to shared global state
(deadline overrides, GPU failure guards, environment variables).

GPU-dependent tests include `skip_without_gpu!()` and are skipped automatically
on machines without a GPU.

---

## 5. Quality Gate

**Always run `./quality.sh` before committing.** CI treats warnings as errors,
so do not skip this step.

`./quality.sh` performs these checks in order:

1. Bash syntax check (all `.sh` files)
2. `cargo build` (debug, quick feedback)
3. `cargo fmt --all` (auto-formatting)
4. `cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args`
5. `cargo check --all-targets --all-features`
6. `cargo test --lib --tests --all-features -- --test-threads=1`
7. `cargo build --release --lib`

If any step fails, fix the issue and re-run. Do **not** commit code that fails
`./quality.sh`.

### CI Pipeline

GitHub Actions runs on every pull request to `Develop`:

- `auto-format` — applies `rustfmt` and commits fixes
- `version-increment` — auto-bumps patch version when `src/` changes
- `quality` — fmt check, Clippy, cargo check, tests, build
- `shell-checks` — validates bash script syntax
- `spell-check` — runs codespell on the codebase
- `validation` — checks required files and `Cargo.toml`
- `security` — runs security audit workflow

**Do NOT modify `.github/workflows/ci.yml` without explicit approval.**

---

## 6. Build and Install

```bash
# Build and install the library
./scripts/runlib.sh
```

This script:
- Installs Rust and Cargo if missing (no sudo required)
- Builds the library in release mode
- Installs it to `~/.cargo/lib/` with version tracking
- Signs it on macOS for FFI compatibility

### Version Management

**Do not manually bump versions.** CI increments `Cargo.toml` patch versions
when `src/` changes are detected. To confirm what a worker has loaded, call
`get_library_version()`.

---

## 7. FFI Contract

The library exposes a Deno FFI-friendly symbol set. The authoritative list of
exported symbols lives in `src/lib.rs` as `#[no_mangle] pub extern "C"`
functions.

### Key Symbols

| Category | Symbols |
|----------|---------|
| **GPU probe** | `check_gpu_available()` |
| **Version probe** | `get_library_version()` |
| **Recording (streaming)** | `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, `cancel_discovery_session` |
| **Recording (single-call)** | `record_discovery` (avoid for large runs) |
| **Analysis** | `rank_focus_neurons`, `analyze_parallel` |
| **Utilities** | `merge_discovery_parquet`, `read_discovery_records_ffi`, `export_visualisation_snapshot` |
| **Memory management** | `free_discovery_result` |

### Memory Management

Every FFI call returning a `char*` **must** be freed with
`free_discovery_result()`. Failure to do so will leak memory.

### JSON Interface

All FFI functions accept and return JSON strings. See
[README.md — JSON Interface](README.md#json-interface) for the full input/output
format specification.

---

## 8. GPU Requirement

**This library requires a GPU.** There is no CPU fallback.

- **macOS**: Metal (should always be available)
- **Linux**: Vulkan only (no OpenGL/EGL probe to avoid panics)

### Minimum System Requirements

| Requirement | Minimum | Reason |
|-------------|---------|--------|
| **Total RAM** | 4 GB | GPU operations require staging buffers |
| **Available RAM** | 1 GB | Prevents hangs from memory pressure |
| **GPU** | Metal (macOS) or Vulkan (Linux) | Required for compute shaders |

If no compatible GPU is available, discovery is simply skipped — NEAT-AI
continues training without the discovery phase. See
[README.md — GPU Requirement](README.md#gpu-requirement) for full details
including memory checks and streaming parquet loading.

---

## 9. Key Invariants

These invariants **must not** be violated by any code change:

### Forward-only Activation Order

Discovery assumes **forward-only** networks (no recurrent feedback):
- A neuron may only read activations from **earlier** neurons in the creature's
  evaluation order.
- Synapses must point from an earlier neuron to a later neuron.
- New neurons must be inserted at the correct index (not appended).
- No cross-sample state — each recorded activation/error is for a single
  training sample.

### Atomic Record Writes

For each discovery record, all data (observations, activations, errors) **must**
come from the same training record:
- The analysis phase matches records by `obs_index`.
- Parallelisation across training records is allowed.
- Per-record atomicity is required (activate -> collect all neurons -> write
  atomically).
- Never mix data from different training records within a single write.

### VALUE Domain Errors

All FFI functions return structured JSON with a `success` field. When
`success` is `false`, the `error` field contains a descriptive message.
Controllers must check this field before processing results.

---

## 10. Environment Variables

Key environment variables that control library behaviour:

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Control log level via `tracing` (e.g. `neat_ai_discovery=info`) |
| `NEAT_AI_DISCOVERY_VERBOSE` | Enable verbose logging (`1` to enable) |
| `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` | Override GPU batch size (64–4096) |
| `NEAT_AI_DISCOVERY_GPU_TIMING` | Enable GPU kernel profiling (`1` to enable) |
| `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` | Abort if no progress for N seconds |
| `NEAT_AI_DISCOVERY_QUIET_GPU` | Suppress Mesa/libEGL debug output |
| `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` | Max blocks in streaming cache |
| `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` | Streaming prefetch depth |
| `NEAT_AI_DISCOVERY_PRELOAD_ALL` | Disable streaming, use full preload |
| `NEAT_AI_DISCOVERY_BLOCK_SIZE` | Block size in records (default: 10000) |
| `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` | Enable outlier-focused analysis |
| `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS` | Prioritise unused input neurons |
| `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` | Bias toward newer inputs |
| `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY` | Output-only focus targets |
| `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` | Constant source folding threshold |

See [README.md — Troubleshooting](README.md#troubleshooting) and
[README.md — GPU Performance Tuning](README.md#gpu-performance-tuning) for
full details.

---

## 11. Candidate Types

Reuse existing candidate types whenever possible. If a new type is truly
required, it must be documented in README.md and a corresponding handler
added to NEAT-AI.

| Type | Operation | Description |
|------|-----------|-------------|
| `addNeuron` | Add a hidden neuron | With deterministic `neuronUuid` |
| `removeSynapse` | Remove a connection | Dormant, opposing, or redundant |
| `addSynapse` | Add a connection | Helpful synapse with computed weight |
| `removeNeuron` | Remove a neuron | Dead or low-impact neurons |
| `setBias` | Adjust bias | Output bias drift, saturation fix |
| `setWeight` | Adjust weight | Renormalisation, opposing flip |
| `changeSquash` | Change activation | Oscillation or saturation fix |
| `coordinatedStructural` | Atomic group of operations | Epistatic changes that must be applied together |

See [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) for the full reference
with success/failure rates and detailed descriptions.

---

## 12. Quick Reference

```bash
# Build and install
./scripts/runlib.sh

# Quality gate (run before EVERY commit)
./quality.sh

# Run all tests
cargo test --lib --tests --all-features -- --test-threads=1

# Run specific test
cargo test --test <test_name>

# Run benchmarks
cargo bench --bench <bench_name>

# Check formatting
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args
```

---

## 13. Further Reading

- [README.md](README.md) — User-facing documentation, FFI API, troubleshooting
- [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) — Discovery type reference
- [docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md) — Impact scoring algorithm
- [CHANGELOG.md](CHANGELOG.md) — Version-by-version history
