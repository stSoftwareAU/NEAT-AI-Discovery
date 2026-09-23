# Security sweep — chunk `8b`: Analysis engine — synapse, scoring, recommendation, shared

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `8b` — matches the `id` in the index.
- **Human name:** Analysis engine — `src/analysis/synapse`, `src/analysis/scoring`,
  `src/analysis/recommendation`, `src/analysis/shared`.
- **Sweep date:** `2026-09-23`
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2`
  — `git diff b85a551..HEAD -- src/analysis/synapse src/analysis/scoring
  src/analysis/recommendation src/analysis/shared` is empty at the time this
  scaffold was written, so every line reference below also describes the current
  tree.
- **Exposure:** `internal` — none of these 58 files is an FFI entry point. They
  are reached only through the `src/ffi` boundary (chunk 2), so every input they
  see has already crossed one validation layer. Untrusted values still arrive
  here: the caller-supplied `creature` topology, the Parquet record stream, and
  the operator's environment variables.
- **Swept by:** Issue #2093 (chunk 8b of the #2083 overflow tracker), split
  across seven audit sub-issues.
- **Tracker issue:** `#2083`
- **Parent issue:** `#2093`
- **Scaffold issue:** `#2103` — created this record and swept
  `src/analysis/shared/`.

### Sweep status — IN PROGRESS

This record is filled **incrementally**: one audit sub-issue per `###` section
below, each editing only its own section so concurrent PRs do not conflict. A
row reading `pending` has **not** been swept, and only `src/analysis/shared/`
is swept so far.

`lib-sweep-coverage.json` cannot express that: the ledger contract
(`tests/issue_2088_sweep_ledger_contract.rs`) rejects a `record` that names no
`last_swept` date, so the index entry carries the date this scaffold was cut
against. **Read that date as "the baseline this record is pinned to", not as
"every row is finished."** The per-file table below is the authority on which
files have actually been read; the chunk is complete only when no row reads
`pending`, and the finalisation sub-issue rejects the record until then.

### Why this record cites symbols, not line numbers

Issue #2103 specified a `file:line` column for the two finding tables. This
record uses `file.rs::symbol` instead, per **CONTRIBUTING.md § Cite Code by
Symbol, Never by Line Number** (Issue #1942), which binds every doc in the
repository: a line number rots at the next refactor of a file this chunk has
not even swept yet, while a symbol survives it and a test can check the symbol
still exists. The column's purpose — naming the exact site — is unchanged.

## Defect classes probed

The classes below are what the chunk 8b sub-issues look for. A file marked
`clean` was read for **these** classes; it is not a claim of general
correctness.

- **Capacity-from-input** — `Vec::with_capacity` / `reserve` / `vec![_; n]`
  where `n` derives from a caller-supplied count (creature topology, candidate
  count, sample count) without a bound. This is the class #2078 and the chunk
  8a sweep both landed on: an unbounded `n` aborts the process on allocation
  failure, which a `catch_unwind` at the FFI boundary cannot recover.
- **Float comparison** — `==` / `!=` / ordering against a value that can be
  `NaN` or `-0.0`, and `partial_cmp(...).unwrap()`, where the operand
  originates in caller data or in a division whose divisor can be zero.
- **Integer overflow in size/score arithmetic** — `+`, `*` and `as` casts on
  counts and indices derived from input; debug builds panic, release builds
  wrap into a wrong bound.
- **Shared-state races** — every reader and writer of mutable state shared
  across the rayon parallel sections: lock scope, atomic ordering, unbounded
  growth of shared collections keyed by input-derived strings, and torn reads
  between a lock-protected field and a separately-loaded atomic.
- **Panics on hostile environment values** — `unwrap` / `expect` / slicing in
  `from_env`-style parsing, where an operator (or anything that can set the
  process environment) supplies the value.

## Files swept

58 files, 20,997 lines. Line counts as at the baseline commit. Every row starts
`pending`; the sub-issue owning the section replaces it with an outcome and a
one-line reason.

### synapse pipeline

3,391 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/mod.rs` | 204 | pending |
| `src/analysis/synapse/orchestration.rs` | 374 | pending |
| `src/analysis/synapse/preparation.rs` | 326 | pending |
| `src/analysis/synapse/candidate_generation.rs` | 216 | pending |
| `src/analysis/synapse/cpu_pre_reject.rs` | 125 | pending |
| `src/analysis/synapse/gpu_evaluation.rs` | 250 | pending |
| `src/analysis/synapse/activation_evaluation.rs` | 543 | pending |
| `src/analysis/synapse/activation_subset_evaluation.rs` | 210 | pending |
| `src/analysis/synapse/relu_evaluation.rs` | 174 | pending |
| `src/analysis/synapse/adaptive_proposal.rs` | 511 | pending |
| `src/analysis/synapse/add_synapse_gating.rs` | 458 | pending |

### synapse post-processing

2,864 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/post_processing.rs` | 945 | pending |
| `src/analysis/synapse/filtering.rs` | 266 | pending |
| `src/analysis/synapse/holdout_validation.rs` | 244 | pending |
| `src/analysis/synapse/metadata.rs` | 99 | pending |
| `src/analysis/synapse/results.rs` | 115 | pending |
| `src/analysis/synapse/structural_patterns.rs` | 700 | pending |
| `src/analysis/synapse/tests.rs` | 495 | pending |

### synapse scoring + target_analysis

3,649 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/scoring/mod.rs` | 45 | pending |
| `src/analysis/synapse/scoring/boost_functions.rs` | 84 | pending |
| `src/analysis/synapse/scoring/discounting.rs` | 340 | pending |
| `src/analysis/synapse/scoring/improvement.rs` | 717 | pending |
| `src/analysis/synapse/scoring/test_helpers.rs` | 150 | pending |
| `src/analysis/synapse/scoring/tests.rs` | 487 | pending |
| `src/analysis/synapse/target_analysis/mod.rs` | 478 | pending |
| `src/analysis/synapse/target_analysis/candidate_selection.rs` | 153 | pending |
| `src/analysis/synapse/target_analysis/evaluation.rs` | 865 | pending |
| `src/analysis/synapse/target_analysis/statistics.rs` | 330 | pending |

### scoring

4,426 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/scoring/mod.rs` | 12 | pending |
| `src/analysis/scoring/calibration_correction.rs` | 1490 | pending |
| `src/analysis/scoring/confidence.rs` | 592 | pending |
| `src/analysis/scoring/cross_validation.rs` | 441 | pending |
| `src/analysis/scoring/error_distribution.rs` | 607 | pending |
| `src/analysis/scoring/sample_creature_disconnect.rs` | 201 | pending |
| `src/analysis/scoring/weights/mod.rs` | 531 | pending |
| `src/analysis/scoring/weights/adjustment.rs` | 71 | pending |
| `src/analysis/scoring/weights/calculation.rs` | 398 | pending |
| `src/analysis/scoring/weights/normalisation.rs` | 83 | pending |

### recommendation core

3,463 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/recommendation/mod.rs` | 14 | pending |
| `src/analysis/recommendation/activation_recommendation.rs` | 927 | pending |
| `src/analysis/recommendation/fan_in.rs` | 608 | pending |
| `src/analysis/recommendation/gradient_discovery.rs` | 330 | pending |
| `src/analysis/recommendation/multi_hop.rs` | 497 | pending |
| `src/analysis/recommendation/output_bias_drift.rs` | 389 | pending |
| `src/analysis/recommendation/output_competition.rs` | 325 | pending |
| `src/analysis/recommendation/sample_weighted.rs` | 373 | pending |

### recommendation batch_successful + epistatic

2,309 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/recommendation/batch_successful/mod.rs` | 67 | pending |
| `src/analysis/recommendation/batch_successful/detection.rs` | 240 | pending |
| `src/analysis/recommendation/batch_successful/grouping.rs` | 162 | pending |
| `src/analysis/recommendation/epistatic/mod.rs` | 155 | pending |
| `src/analysis/recommendation/epistatic/candidate_generation.rs` | 616 | pending |
| `src/analysis/recommendation/epistatic/deduplication.rs` | 100 | pending |
| `src/analysis/recommendation/epistatic/pre_screening.rs` | 464 | pending |
| `src/analysis/recommendation/epistatic/scoring.rs` | 505 | pending |

### shared

895 lines. Swept by Issue #2103 for the shared-state race class (plus the
capacity, float-comparison, overflow and hostile-environment classes above).

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/shared/mod.rs` | 16 | clean — module declarations and glob re-exports only; no executable code |
| `src/analysis/shared/gpu_info.rs` | 124 | clean — `from_env` cannot panic on any env value; `buffer_count` is a hardcoded `3`, never input-derived |
| `src/analysis/shared/metadata.rs` | 454 | clean — plain `derive`d result structs; no interior mutability, no arithmetic, no comparisons |
| `src/analysis/shared/timing.rs` | 301 | clean — all shared mutation is `parking_lot::Mutex` or atomic RMW; `finalize` runs after the rayon join; the map's key set is four compile-time literals |

## Capacity-from-input sites

Every `with_capacity` / `reserve` / `vec![_; n]` whose `n` derives from caller
input, with the bound that stops it. A `verdict` of `bounded` names what bounds
it; `unbounded` means a finding was filed.

| Site (`file.rs::symbol`) | Expression | Bound source | Verdict |
| --- | --- | --- | --- |
<!-- section: synapse pipeline -->
<!-- section: synapse post-processing -->
<!-- section: synapse scoring + target_analysis -->
<!-- section: scoring -->
<!-- section: recommendation core -->
<!-- section: recommendation batch_successful + epistatic -->
<!-- section: shared -->
| — | none | `src/analysis/shared/` allocates no collection sized from input | n/a |

## Float comparison sites

Every comparison against a float that can be `NaN` or `-0.0`, with where the
value comes from and what happens when it is `NaN`.

| Site (`file.rs::symbol`) | Comparator | Value origin | NaN handling | Verdict |
| --- | --- | --- | --- | --- |
<!-- section: synapse pipeline -->
<!-- section: synapse post-processing -->
<!-- section: synapse scoring + target_analysis -->
<!-- section: scoring -->
<!-- section: recommendation core -->
<!-- section: recommendation batch_successful + epistatic -->
<!-- section: shared -->
| — | none | `src/analysis/shared/` compares no floats | n/a | n/a |

## Outcome

### shared (Issue #2103)

**Negative result — no finding filed.** All four `src/analysis/shared/` files
are clean for every defect class probed. What was actually traced:

**`timing.rs` — the shared-state race class.** `timing.rs::TimingCollector` is
the only mutable shared state in the whole of `shared/`. Every reader and
writer was traced:

- **Writers.** `timing.rs::TimingCollector::record_shader` mutates the
  `parking_lot::Mutex<HashMap<String, (u32, u64)>>` under
  `lock_contention.rs::traced_lock_default`;
  `timing.rs::TimingCollector::record_buffer_transfer`,
  `::record_sample_building` and `::record_result_processing` are
  `AtomicU64::fetch_add` with `Ordering::Relaxed`. `fetch_add` is an atomic
  read-modify-write, so `Relaxed` cannot lose an update — these counters carry
  no ordering dependency on any other datum, which is the only thing a stronger
  ordering would buy. The sole entry point to all four is the `Drop` impl of
  `timing.rs::TimingScope`.
- **Readers.** `timing.rs::TimingCollector::finalize` is the only reader, and
  it takes the same lock before iterating the map. It has exactly **two** call
  sites: `synapse/results.rs::finalise_synapse_results` and
  `neuron/post_processing.rs::build_neuron_results`.
- **Dispatch — synapse.** The collector is constructed as an `Arc` in
  `synapse/orchestration.rs::analyze_synapses_with_cache_impl`, cloned into
  the per-target `TargetAnalysisContext`, and shared across that function's
  `par_iter()` over the focus order. The same function calls
  `finalise_synapse_results` **after** the parallel iterator has joined.
- **Dispatch — neuron.** `neuron/mod.rs::analyze_neurons_with_cache_and_gpu_queue`
  constructs its own `Arc<TimingCollector>` and shares it across two
  `par_iter()` sections, then calls
  `neuron/post_processing.rs::build_neuron_results` after both have joined.
- Rayon's join establishes the happens-before edge on **both** paths, so the
  `Relaxed` loads in `finalize` observe every worker's `fetch_add`. There is no
  window in which a writer and the reader overlap.
- **The two dispatch files #2103 named carry no `TimingCollector` reference at
  all.** `src/analysis/orchestration.rs` and
  `src/analysis/discovery_dispatch.rs` each have zero occurrences at the
  baseline commit; the real dispatch sites are the two named above. Recorded
  here so a later sweep does not re-trace the wrong files.
- **Unbounded map growth was the one plausible attack.** `shader_timings` is
  keyed by `shader_name.to_string()`, so an input-derived name would grow the
  map without bound under the lock. Every call site of
  `timing.rs::TimingScope::shader` passes a compile-time literal — `"relu"` and
  `"activation"` in `neuron/evaluation.rs::evaluate_relu_split` and
  `::evaluate_activation_specs`, `"helpful"` and `"harmful"` in
  `synapse/target_analysis/evaluation.rs::submit_helpful_gpu_work` and
  `::process_harmful_batch_from_prepared` — so the key set is four entries and
  no caller can extend it.
- **Overflow.** The `+=` on the per-shader `(calls, total_ns)` tuple in
  `timing.rs::TimingCollector::record_shader`, and the `total_shader_ns`
  accumulator in `::finalize`, are unchecked. The `u32` call counter is the
  tightest: it needs 2^32 shader dispatches in a single analysis run, and
  dispatch count is bounded by focus neurons × candidates, itself bounded by
  the creature the FFI layer already validated. The whole collector is inert
  unless the operator sets `NEAT_AI_DISCOVERY_GPU_TIMING=1`. Noted, not filed:
  no input reaches the bound, and a telemetry counter wrapping in release costs
  a wrong diagnostic number, not memory safety.

**`gpu_info.rs` — hostile environment values.**
`gpu_info.rs::ZeroCopyBufferConfig::from_env` delegates to
`config/user_facing.rs::zero_copy_override`, which is
`config/helpers.rs::parse_optional_bool_env`. That function is total:
`std::env::var(...)` returns `Err` for a non-UTF-8 value and `.ok()` maps it to
`None`, and any unrecognised string falls through the `match` to `None`. No
`unwrap`, no slicing, no numeric parse. `ZeroCopyBufferConfig::buffer_count` is
the literal `3` and is never read from the environment, so no hostile value
reaches an allocation size.

**`metadata.rs` — interior mutability.** Confirmed absent. All **11** public
types are `#[derive]`d plain-data structs and enums; the file contains no
`Mutex`, `RwLock`, `RefCell`, `Cell`, `UnsafeCell`, `Atomic*`, `static mut`,
`OnceLock` or `Lazy`, and no arithmetic or comparison of any kind. Its `f32`
fields — `calibration_corrections` and `rolling_success_rate` on both
`SynapseAnalysisMetadata` and `NeuronAnalysisMetadata`, and the
`expected_improvement` / `threshold` / `suggested_weight` / `outgoing_weight`
options on `SynapseNoCandidateDetail` and `NeuronNoCandidateDetail` — are
carried, never compared here. The only two types deriving `PartialEq` are the
fieldless `SynapseNoCandidateReason` and `NeuronNoCandidateReason` enums, so no
float comparison is generated either.

**`mod.rs`.** Three `pub mod` declarations and three glob re-exports. No
executable code.

**Deliberately out of scope for this sub-issue:** the other 54 rows above, which
belong to the six remaining chunk 8b audit sub-issues.

## Issues filed

- `negative-result` — the `shared/` sweep found nothing worth filing. The
  remaining sections list their own findings as they are swept.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- `#2078` — the house issue format every chunk 8b finding follows.
- `#1933`, `#1930`, `#1929` — GPU queue liveness and breaker work adjacent to
  `src/analysis/synapse/gpu_evaluation.rs`; not a sweep of it.

## Verify this record

```bash
git diff b85a551ed2521ed327469b20eb88aeda828357d2..HEAD -- \
  src/analysis/synapse src/analysis/scoring \
  src/analysis/recommendation src/analysis/shared
```

An empty diff means this record still describes the current code.
