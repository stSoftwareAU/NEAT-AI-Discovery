# AGENTS.md — Coding Guidelines for AI Agents

This file is the single source of truth for AI coding agents working in this
repository. For user-facing documentation, see [README.md](README.md).

---

## 1. Project Overview

For the full project description, mission statement, and user-facing
documentation, see [README.md](README.md). The sections below cover
agent-specific conventions and invariants.

---

## 2. Architecture

### Source Layout

```
src/
├── lib.rs                    # Module declarations, re-exports, version init
├── ffi/                      # FFI entry points (no_mangle extern "C") (Issue #601)
│   ├── mod.rs                # Public API, re-exports, free_discovery_result
│   ├── gpu.rs                # GPU probe entry points (check_gpu_available)
│   ├── recording.rs          # Recording entry points (streaming and single-call)
│   ├── analysis.rs           # Analysis entry points (rank_focus_neurons, analyze_parallel)
│   └── utilities.rs          # Utility entry points (merge, read, export, version, memory usage)
├── ffi_types/                # JSON request/response structs for FFI boundary (Issue #596)
│   ├── mod.rs                # Public API, re-exports, shared types (Creature, Neuron, Synapse)
│   ├── requests.rs           # FFI request structs (input from NEAT-AI)
│   ├── responses/             # FFI response structs (output to NEAT-AI) (Issue #874)
│   │   ├── mod.rs            # Re-exports, record/version/rank output types, conversion helpers
│   │   ├── analysis.rs       # AnalyzeParallelOutput, metadata JSON, diagnostic types
│   │   ├── gpu.rs            # CheckGpuOutput, GPU timing/adapter JSON types
│   │   └── export.rs         # Export, merge, read, calibration output types
│   ├── candidates.rs         # Candidate-related types (synapse, neuron, coordinated)
│   └── session.rs            # Streaming session types
├── ffi_internal/             # Internal business-logic functions for FFI (Issue #665)
│   ├── mod.rs                # Public API, re-exports, unit tests
│   ├── recording.rs          # Recording business logic (record_discovery_internal)
│   ├── analysis.rs           # Analysis business logic (analyze_parallel, rank_focus, calibration)
│   ├── gpu.rs                # GPU probe and version business logic
│   └── utilities.rs          # Utility business logic (merge, read, export)
├── config/                   # Central environment variable configuration (Issue #717, #981)
│   ├── mod.rs                # Public API, re-exports, env var documentation
│   ├── user_facing.rs        # User-facing configuration accessors
│   ├── detection.rs          # Detection-specific configuration helpers
│   ├── observability.rs      # Observability configuration (logging, tracing)
│   └── helpers.rs            # Shared parsing and validation helpers
├── types.rs                  # Core type definitions
├── activations.rs            # Activation function calculations
├── record/                   # Discovery data recording (Issue #604, #942)
│   ├── mod.rs                # Public API, re-exports, orchestration
│   ├── validation.rs         # Input validation and observation index resolution
│   ├── processing.rs         # Record building from training data
│   └── tests.rs              # Unit tests for record module
├── streaming.rs              # Streaming session management
├── parquet_format/            # Parquet I/O (Issue #600)
│   ├── mod.rs                # Public API, re-exports
│   ├── schema.rs             # Schema definitions and validation
│   ├── writer.rs             # Parquet writing and serialisation
│   └── reader.rs             # Parquet reading and deserialisation
├── export/                   # Visualisation snapshot export (Issue #980)
│   ├── mod.rs                # Public API, re-exports
│   ├── types.rs              # Snapshot structs, export options, stats types
│   ├── stats.rs              # Aggregation functions, JSON-safe float conversion
│   ├── snapshot.rs           # Main export_visualisation_snapshot pipeline
│   └── timestamp.rs          # chrono_lite_now, leap year calculation
├── discovery_history.rs      # Historical tracking
├── debug.rs                  # Debugging utilities
├── observability/             # Observability / logging (Issue #874)
│   ├── mod.rs                # Tracing init, env var parsing, re-exports, tests
│   ├── phase_timer.rs        # PhaseTimer and ScopedPhaseTimer RAII guards
│   ├── gpu_metrics.rs        # GpuMetrics thread-safe tracking and global instance
│   └── profile.rs            # ProfileData for JSON output
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
│   ├── constants/             # Central discovery thresholds (Issue #424, #938)
│   │   ├── mod.rs                # Re-exports all constants (backward compatibility)
│   │   ├── sample_thresholds.rs  # Sample count thresholds, hold-out validation
│   │   ├── sentinel_detection.rs # Sentinel values and clustering thresholds
│   │   ├── source_variance.rs    # Source variance filtering thresholds
│   │   ├── candidate_scoring.rs  # Scoring boosts, pessimism, calibration, comparisons
│   │   ├── compression.rs        # Candidate compression thresholds
│   │   ├── detection_thresholds.rs # Detection filtering (removal, weight constraints)
│   │   └── temperature.rs        # Temperature scheduling for exploration-exploitation (Issue #1020)
│   ├── shared/               # Common types, results, diagnostics (Issue #874)
│   │   ├── mod.rs            # Re-exports for backward compatibility
│   │   ├── timing.rs         # TimingCollector, TimingScope, ShaderTiming, timing breakdowns
│   │   ├── metadata.rs       # SynapseAnalysisMetadata, NeuronAnalysisMetadata, result types
│   │   └── gpu_info.rs       # GpuAdapterInfo, GpuDeviceType, ZeroCopyBufferConfig
│   ├── synapse/              # Synapse analysis (Issue #482)
│   │   ├── mod.rs            # Public API, entry points, orchestration
│   │   ├── orchestration.rs  # Top-level synapse analysis orchestration
│   │   ├── preparation.rs    # Focus target filtering, source loading
│   │   ├── target_analysis/  # Per-target analysis loop (Issue #599)
│   │   │   ├── mod.rs            # Public API, types, main analysis loop
│   │   │   ├── evaluation.rs     # GPU work submission, result collection, candidate processing
│   │   │   ├── candidate_selection.rs # Epistatic, synergistic, redundant path detection
│   │   │   └── statistics.rs     # Source filtering, record loading, sample building
│   │   ├── scoring/           # Scoring pipeline (Issue #982)
│   │   │   ├── mod.rs            # Re-exports for backward compatibility
│   │   │   ├── boost_functions.rs # Source/target/activation type boosts
│   │   │   ├── improvement.rs    # Core improvement calculation, candidate dedup
│   │   │   ├── discounting.rs    # Pessimism discounting, prediction calibration
│   │   │   ├── test_helpers.rs   # Shared test helpers for scoring tests
│   │   │   └── tests.rs         # Unit tests for scoring pipeline
│   │   ├── gpu_evaluation.rs # GPU batch orchestration
│   │   ├── activation_evaluation.rs # Activation function evaluation for synapses
│   │   ├── activation_subset_evaluation.rs # Subset-based activation evaluation
│   │   ├── relu_evaluation.rs # ReLU-specific synapse evaluation
│   │   ├── holdout_validation.rs # Hold-out validation for multi-weight search
│   │   ├── candidate_generation.rs # Sample building, locality grouping
│   │   ├── filtering.rs      # Candidate filtering, deduplication
│   │   ├── structural_patterns.rs # Coordinated structural discovery
│   │   ├── post_processing.rs # Impact discounting, sorting, metadata
│   │   ├── metadata.rs       # Synapse analysis metadata types
│   │   ├── results.rs        # Result types and assembly
│   │   ├── adaptive_proposal.rs # Adaptive Gaussian proposal distribution (Issue #1019)
│   │   └── tests.rs          # Unit tests for synapse analysis
│   ├── neuron/               # Neuron analysis (Issue #598)
│   │   ├── mod.rs            # Public API, orchestration, parallel loop
│   │   ├── preparation.rs    # Focus target filtering, neuron type maps, source loading
│   │   ├── evaluation.rs     # GPU-based candidate evaluation (ReLU, activation specs)
│   │   └── post_processing.rs # Impact discounting, sorting, filtering, result assembly
│   ├── activation/           # Activation function analysis (Issue #607)
│   │   ├── mod.rs            # Public API, re-exports, tests
│   │   ├── functions.rs      # CPU activation function implementations
│   │   ├── specs.rs          # Candidate specs, GPU ID mapping, bias helpers
│   │   └── simulation.rs     # Target simulation, predicates, variance checking
│   ├── samples/              # Sample data structures (Issue #597)
│   │   ├── mod.rs            # Public API, re-exports, core sample types
│   │   ├── gpu_types.rs      # GPU-compatible data formats (#[repr(C)], bytemuck)
│   │   ├── statistics.rs     # Statistics types (NeuronStats, HelpfulStats, etc.)
│   │   └── thresholds.rs     # Threshold computation and source variance analysis
│   ├── diagnostics/          # Diagnostic tracking (Issue #524)
│   │   ├── mod.rs            # Public API, re-exports, impact scoring adapter
│   │   ├── rejection.rs      # Synapse rejection tracking and reporting
│   │   ├── neuron_tracking.rs # Neuron rejection tracking and reporting
│   │   ├── target_data.rs    # Target data structures for sample building
│   │   ├── focus_filter.rs   # Focus target filtering and validation
│   │   └── mcmc_diagnostics.rs # MCMC-style acceptance rate tracking (Issue #1021)
│   ├── detection/            # Pattern detection modules (Issue #528)
│   │   ├── mod.rs            # Module declarations
│   │   ├── activation_mismatch.rs # Activation function mismatch detection
│   │   ├── activation_properties.rs # Shared activation classification helpers
│   │   ├── bias_perturbation.rs # Bias perturbation regime shift detection
│   │   ├── bimodal_neuron.rs # Bimodal pre-activation distribution detection
│   │   ├── bottleneck.rs     # Bottleneck neuron detection
│   │   ├── bounded_range.rs  # Bounded range detection
│   │   ├── co_adaptation.rs  # Redundant neuron pair co-adaptation detection
│   │   ├── compound_degradation.rs # Compound bias+weight degradation detection (Issue #929)
│   │   ├── correlated_error.rs # Correlated error patterns
│   │   ├── cross_detection_synthesis.rs # Cross-detection candidate synthesis (Issue #963)
│   │   ├── dead_neuron.rs    # Dead neuron detection
│   │   ├── dormant_synapse.rs # Dormant synapse detection
│   │   ├── error_plateau.rs  # Output error stagnation plateau detection
│   │   ├── fanin_polarity_conflict.rs # Fan-in weight polarity conflict detection
│   │   ├── hard_sample_cluster.rs # High-error observation cluster detection
│   │   ├── helpers.rs        # Shared detection helper utilities
│   │   ├── high_error_squash_exploration.rs # Proactive activation exploration for high-error neurons
│   │   ├── input_sensitivity.rs # Input sensitivity analysis
│   │   ├── low_impact_neuron.rs # Low-impact (near-zero) neuron removal detection
│   │   ├── monotonicity.rs   # Activation-error monotonicity detection
│   │   ├── noise_signal.rs   # Noise-to-signal ratio detection
│   │   ├── observation_range.rs # Observation effective range
│   │   ├── observation_utilisation.rs # Underutilised input observation detection
│   │   ├── operating_point.rs # Hidden neuron operating point
│   │   ├── opposing_synapse.rs # Opposing synapse detection
│   │   ├── oscillating_neuron.rs # Oscillating neuron detection
│   │   ├── output_conflict.rs # Per-output error disaggregation detection
│   │   ├── output_range_compression.rs # Output activation range compression
│   │   ├── output_squash_mismatch.rs # Output activation mismatch detection
│   │   ├── redundant_path.rs # Redundant path detection
│   │   ├── restricted_range.rs # Restricted activation range
│   │   ├── saturation.rs     # Saturated neuron detection
│   │   ├── sentinel_gating.rs # Sentinel value gating
│   │   ├── skip_connection.rs # Skip connection (residual) discovery
│   │   ├── squash_weight_rescale.rs # Coordinated squash change with weight rescale
│   │   ├── stats.rs          # Statistical analysis helpers
│   │   ├── symmetry_breaking.rs # Converged duplicate neuron detection
│   │   ├── topology.rs       # Topology-aware structure analysis
│   │   ├── topology_cache.rs # Shared pre-computed topology cache
│   │   ├── topology_diversification.rs # Topology diversification for structural jumps
│   │   ├── unbounded_capping.rs # Unbounded activation capping
│   │   ├── weight_coherence.rs # Weight coherence validation
│   │   ├── weight_magnitude_reset.rs # Stuck synapse weight magnitude reset
│   │   └── weight_polarity_flip.rs # Gradient–weight sign disagreement detection
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
│   │   ├── batch_successful/  # Batch-successful candidate grouping (Issue #965)
│   │   │   ├── mod.rs            # Public API, orchestration
│   │   │   ├── detection.rs      # Batch candidate detection logic
│   │   │   └── grouping.rs      # Non-conflicting candidate grouping
│   │   ├── fan_in.rs          # Fan-in candidate generation (Issue #908)
│   │   ├── multi_hop.rs      # Multi-hop candidate analysis
│   │   ├── gradient_discovery.rs # Gradient-based synapse adjustment
│   │   └── sample_weighted.rs # Sample-weighted discovery
│   ├── scoring/              # Scoring and confidence modules (Issue #528)
│   │   ├── mod.rs            # Module declarations
│   │   ├── confidence.rs     # Confidence metrics
│   │   ├── weights/          # Weight analysis and calculation (Issue #609)
│   │   │   ├── mod.rs            # Public API, re-exports, constants
│   │   │   ├── calculation.rs    # Core weight calculation (least squares, bias search)
│   │   │   ├── normalisation.rs  # Range-aware weight computation (sentinel filtering)
│   │   │   └── adjustment.rs     # Dynamic weight adjustments (delta clamping)
│   │   ├── error_distribution.rs # Error distribution stats
│   │   └── cross_validation.rs # Cross-validation scoring
│   ├── cache/                # Record caching (Issue #565)
│   │   ├── mod.rs            # Public API, RecordCache, re-exports
│   │   ├── loading_strategy.rs # LoadingStrategy, select_loading_strategy
│   │   ├── lru_cache.rs      # LruRecordCache, LruCacheStats, eviction
│   │   ├── compressed_cache.rs # CompressedLruRecordCache (LZ4)
│   │   ├── tiered_cache.rs   # TieredRecordCache, auto strategy selection
│   │   └── serialisation.rs  # Binary serialisation, CompressedCacheEntry
│   ├── candidate_compression/ # Candidate compression (Issue #939)
│   │   ├── mod.rs            # Public API, re-exports
│   │   ├── identity.rs       # IDENTITY candidate compression
│   │   ├── nonlinear.rs      # Non-linear squash function compression (TANH, GELU)
│   │   ├── grouping.rs       # Compatible candidate grouping
│   │   └── gain_estimation.rs # Gain estimation for compressed candidates
│   ├── streaming.rs          # Streaming parquet loading
│   ├── discovery_dispatch.rs # Generic discovery module dispatch (Issue #375)
│   ├── candidate_clustering.rs # Redundancy reduction
│   ├── candidate_diversity.rs # Diversity-aware candidate reranking (Issue #610)
│   ├── candidate_cache.rs    # Candidate outcome cache for success/failure tracking
│   ├── ensemble_scoring.rs   # Cross-module ensemble scoring (Issue #572)
│   ├── module_weights.rs     # Per-module success rate tracking for adaptive weighting
│   ├── scale_outcomes.rs     # Per-scale success rate tracking for weight variants (Issue #964)
│   ├── neuron_fingerprint.rs # Neuron structural fingerprinting for incremental analysis
│   ├── system.rs             # System utilities facade (memory, GPU tier detection)
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
│   │   ├── pipeline_builder.rs    # Shared compute pipeline builder (Issue #978)
│   │   ├── queue/            # GPU work queue (Issue #608)
│   │   │   ├── mod.rs            # Public API, re-exports, queue types
│   │   │   ├── submission.rs     # Work item submission and batching
│   │   │   ├── execution.rs      # GPU execution and result collection
│   │   │   └── scheduling.rs     # Work scheduling and prioritisation
│   │   └── shaders.rs        # Shader management
│   │
│   └── utils/                # Utilities
│       ├── mod.rs
│       ├── memory.rs         # Memory detection
│       ├── deadline.rs       # Deadline handling
│       └── platform.rs       # Platform setup
│
└── shaders/                  # WGSL compute shaders
    ├── activation.wgsl / activation_reduce.wgsl
    ├── helpful.wgsl / helpful_reduce.wgsl
    ├── harmful.wgsl / harmful_reduce.wgsl
    ├── bias.wgsl
    ├── matching.wgsl
    └── relu.wgsl / relu_reduce.wgsl
```

### Other Key Directories

| Directory | Purpose |
|-----------|---------|
| `tests/` | Integration tests (~286 files) |
| `benches/` | Criterion benchmarks (31 suites) |
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

Tests that mutate shared global state (environment variables, deadline
overrides, watchdog) are marked with `#[serial]` from the `serial_test` crate.
All other tests may run in parallel (`--test-threads=2`).

GPU-dependent tests include `skip_without_gpu!()` and are skipped automatically
on machines without a GPU.

---

## 5. Quality Gate

**Always run `./quality.sh` before committing.** CI treats warnings as errors,
so do not skip this step.

`./quality.sh` performs these checks in order:

1. Bash syntax check (all `.sh` files)
2. `cargo upgrade --incompatible` + `cargo update` (dependency upgrade, Issue #959)
3. `cargo deny check` (licence and dependency audit)
4. `cargo build` (debug, quick feedback)
5. `cargo fmt --all` (auto-formatting)
6. `cargo clippy --all-targets --all-features -- -D warnings`
7. `cargo check --all-targets --all-features`
8. `cargo test --lib --tests --all-features -- --test-threads=2`
9. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` (documentation build)
10. `cargo build --release --lib`

If any step fails, fix the issue and re-run. Do **not** commit code that fails
`./quality.sh`.

### CI Pipeline

GitHub Actions runs on every pull request to `Develop`:

- `auto-format` — applies `rustfmt` and commits fixes
- `version-increment` — auto-bumps patch version when changes exist (uses
  `ACTIONS_PUSH` PAT so the push re-triggers workflows, matching NEAT-AI)
- `quality` — fmt check, Clippy, cargo check, doc build, tests, build
- `spell-check` — runs codespell on the codebase
- `validation` — checks required files and `Cargo.toml`
- `security` — runs security audit workflow
- `shellcheck` (separate workflow `.github/workflows/shellcheck.yml`) — lints bash scripts via ShellCheck

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

> **CRITICAL: The version in `Cargo.toml` must always be incremented on any
> code change.** Remote and unattended machines cache the compiled library by
> version number. If the version is not incremented, those machines will
> continue using the old compiled library and never pick up the new changes.

**How versions are incremented:**

- **CI auto-increment (primary)**: The `version-increment` CI job auto-bumps the
  patch version on every PR when changes exist compared to the base branch. It
  uses a PAT (`secrets.ACTIONS_PUSH`) so that the push re-triggers workflows,
  matching the approach used in the NEAT-AI repository (Issue #955).
- **Manual increment**: If you are making changes outside of the normal PR
  workflow, or if CI does not run (e.g., direct commits), you **must** manually
  increment the patch version in `Cargo.toml` (e.g., `0.43.8` → `0.43.9`).
- **Verification**: To confirm what version a worker has loaded, call
  `get_library_version()`.

---

## 7. FFI Contract

For the full FFI API reference, exported symbols, JSON interface specification,
and streaming recording API, see [docs/FFI_API.md](docs/FFI_API.md).

### Key Invariant — Memory Management

Every FFI call returning a `char*` **must** be freed with
`free_discovery_result()`. Failure to do so will leak memory.

---

## 8. GPU Requirement

**This library requires a GPU.** There is no CPU fallback. See
[README.md — GPU Requirement](README.md#gpu-requirement) for system requirements
and [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) for performance tuning and
troubleshooting.

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
| `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` | Streaming session TTL for orphan cleanup (default 3600) |
| `NEAT_AI_DISCOVERY_MH_TEMPERATURE` | Metropolis-Hastings probabilistic acceptance temperature |
| `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` | Overall wall-clock cap for discovery time in minutes (default 20, range 1–120) |
| `NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET` | Max add-neuron candidates per target within a single batch (default 3, range 1–32) (Issue #1140) |

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
cargo test --lib --tests --all-features -- --test-threads=2

# Run specific test
cargo test --test <test_name>

# Run benchmarks
cargo bench --bench <bench_name>

# Check formatting
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

---

## 13. Further Reading

See the [Additional Documentation](README.md#additional-documentation) table in
README.md for a comprehensive index of all project documentation.
