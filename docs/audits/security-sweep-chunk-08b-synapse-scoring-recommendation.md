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
  src/analysis/recommendation src/analysis/shared` was empty when this scaffold
  was written. Since then the only change under those paths is the pair of
  `#[cfg(test)]` regression tests Issue #2104 added to
  `src/analysis/synapse/holdout_validation.rs`, which add no production code, and
  the Issue #2161 deadline/ceiling fix in
  `src/analysis/synapse/candidate_generation.rs` with its `#[cfg(test)]`
  regression file `src/analysis/synapse/issue_2161_locality_cancellation_test.rs`
  — both swept in the row below — and the Issue #2184 ranking fix in
  `src/analysis/recommendation/activation_recommendation.rs` (PR #2187, commit
  `3d1b24f`), which the `recommendation core` row records as swept-and-fixed.
  Two further changes landed while the `recommendation batch_successful +
  epistatic` section was being swept, and both were read against the tree this
  record now describes: the Issue #2185 dispatch wiring in
  `src/analysis/recommendation/output_competition.rs` (PR #2188, commit
  `0971470`), which makes that module's `+inf` accumulator reachable and so
  **reopens** the `recommendation core` row's "not reachable" verdict for it,
  and a one-line change in
  `src/analysis/synapse/target_analysis/statistics.rs` from the same PR.
  So every outcome below still describes the current tree, with that one
  verdict flagged for the finalisation sub-issue.
  **Line counts stay as at the baseline commit** — that is what a
  later reader diffs against.
- **Exposure:** `internal` — none of these 59 files is an FFI entry point. They
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
row reading `pending` has **not** been swept; `src/analysis/shared/` and the
`synapse pipeline` section are swept so far.

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

59 files, 21,157 lines. Line counts as at the baseline commit, except the one
file added after it (`issue_2161_locality_cancellation_test.rs`, counted as it
stands today). Every row starts
`pending`; the sub-issue owning the section replaces it with an outcome and a
one-line reason.

### synapse pipeline

3,801 lines. Swept by Issue #2104. The 15 rows are the files that sub-issue
owns; the scaffold split the synapse root differently from the sub-issues that
edit it, so `filtering`, `holdout_validation`, `metadata`, `results` and
`tests` moved here from `synapse post-processing`, and `adaptive_proposal` and
`add_synapse_gating` moved the other way, leaving each section equal to exactly
one sub-issue's file list.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/mod.rs` | 204 | clean — module declarations, re-exports and two entry wrappers; the only logic is the wedged-GPU and short-deadline early returns, which allocate nothing and compare nothing |
| `src/analysis/synapse/orchestration.rs` | 374 | clean — the per-target `par_iter` is gated by `deadline_passed` (so also by `cancellation::is_cancelled`); every shared field is an `Arc` of atomics or a mutex, and all of them are read after the rayon join |
| `src/analysis/synapse/preparation.rs` | 326 | clean — `NeuronIndex::with_capacity` sums three bounded terms (`creature.input` is capped by `MAX_CREATURE_INPUT_NEURONS`, the other two count live vectors), and the one division is guarded by `std_dev_count > 0` |
| `src/analysis/synapse/candidate_generation.rs` | 216 | **finding #2161** — `group_sources_by_locality` runs an O(n²) pairwise scan with no deadline or cancellation check; its allocations and the overlap division are bounded |
| `src/analysis/synapse/cpu_pre_reject.rs` | 125 | clean — the empty slice short-circuits, the two least-squares sums accumulate in `f64` skipping non-finite samples, and the verdict is delegated to `calculate_optimal_outgoing_weight`, which rejects non-finite input |
| `src/analysis/synapse/gpu_evaluation.rs` | 250 | clean — `vec![None; plan.specs.len()]` is sized from a compile-time `ACTIVATION_SPECS` subset, and `gpu_results[idx]` is length-matched by the evaluator contract that both production implementations uphold |
| `src/analysis/synapse/activation_evaluation.rs` | 543 | clean — the `f32::MIN` and `NEG_INFINITY` sentinels are only ever displaced by a `>` comparison, which a NaN loses, and `finalise_improvement` makes every improvement finite before it is compared |
| `src/analysis/synapse/activation_subset_evaluation.rs` | 210 | clean — same finite-improvement invariant; every accept gate is a `>` comparison, so no non-finite gain can be stored as the best candidate |
| `src/analysis/synapse/relu_evaluation.rs` | 174 | clean — the `total_baseline_error_sq <= EPSILON` guard fails open on a NaN baseline, but every downstream accept is `> best_improvement`, which a NaN loses, so no candidate is emitted |
| `src/analysis/synapse/filtering.rs` | 266 | clean — the descending `total_cmp` would rank a positive NaN first, but both call sites feed it gains that are finite by construction or already non-finite-filtered (Issue #1367) |
| `src/analysis/synapse/holdout_validation.rs` | 244 | clean — `samples.len() - validate_count` cannot wrap: the function returns `None` below 20 samples and `round(0.3n) < n` for every `n >= 20`, so the empty-sample case #1906 posited is unreachable |
| `src/analysis/synapse/results.rs` | 115 | clean — assembly only; it moves merged vectors into the result and reads the metadata atomics after the rayon join has completed |
| `src/analysis/synapse/metadata.rs` | 99 | clean — `Relaxed` atomics written inside the parallel section and read only after the join; `fetch_min` / `fetch_max` are read-modify-write, so no worker's update can be lost |
| `src/analysis/synapse/tests.rs` | 495 | clean — `#[cfg(test)]` only, so no untrusted-input reachability; every fixture is built from compile-time literals and small loop indices |
| `src/analysis/synapse/issue_2161_locality_cancellation_test.rs` | 160 | clean — `#[cfg(test)]` only (declared behind `#[cfg(test)] #[path = …]` in `candidate_generation.rs`), so no untrusted-input reachability; sources are built from loop indices with the one cast guarded by `u32::try_from`, and the timing assertion compares two readings of the same work rather than a wall-clock constant |

### synapse post-processing

2,614 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/post_processing.rs` | 945 | findings filed — #2167 (the three descending `total_cmp` sorts rank a non-finite gain first, and no `retain` filter ahead of them drops `+inf`) and #2168 (`apply_impact_to_helpful` byte-slices a UUID at index 12 inside a verbose log, panicking on a multi-byte char boundary) |
| `src/analysis/synapse/structural_patterns.rs` | 700 | finding filed — #2169: `detect_noisy_vs_trusted` runs a quadratic pairwise scan and `detect_collapsible_hidden_neurons` a linear neuron pass whose per-neuron body walks the records, and neither consults `deadline_passed` or the cancellation flag; the four capacity sites are all bounded by live collection lengths |
| `src/analysis/synapse/adaptive_proposal.rs` | 511 | clean — the only `with_capacity` is sized by the compile-time `ADAPTIVE_PROPOSAL_CANDIDATE_COUNT`; the Box-Muller `ln` is floored away from zero, and `record_batch`'s counters are incremented once per real candidate batch |
| `src/analysis/synapse/add_synapse_gating.rs` | 458 | finding filed — #2170: the FFI-supplied `ModuleOutcomeTracker` reaches `should_skip_add_synapse_by_outcome` unvalidated, so a deserialised `successes > attempts` underflows `ModuleStats::success_rate` and flips the gate; the density gate's divisor is guarded against zero |

### synapse scoring + target_analysis

3,649 lines.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/synapse/scoring/mod.rs` | 45 | clean — module declarations and glob re-exports only; no executable code |
| `src/analysis/synapse/scoring/boost_functions.rs` | 84 | clean — three public functions with `clamp` / numeric scaling only; no untrusted-input reachability |
| `src/analysis/synapse/scoring/discounting.rs` | 340 | clean — `apply_synapse_pessimism_discount` guards `total_count == 0` before dividing |
| `src/analysis/synapse/scoring/improvement.rs` | 717 | clean — `compute_synapse_improvement_and_count` top-level guard closes NaN class; three dispatch targets use `select_finite` |
| `src/analysis/synapse/scoring/test_helpers.rs` | 150 | clean — test-only, no untrusted-input reachability; defensive pattern with final `if x.is_finite()` |
| `src/analysis/synapse/scoring/tests.rs` | 487 | clean — test-only, no untrusted-input reachability; existing passing test proves NaN-safety |
| `src/analysis/synapse/target_analysis/mod.rs` | 478 | clean — module declarations, struct definitions, entry point only; no arithmetic or comparison logic |
| `src/analysis/synapse/target_analysis/candidate_selection.rs` | 153 | clean — two public functions; divisor guards present; no untrusted-input reachability |
| `src/analysis/synapse/target_analysis/evaluation.rs` | 865 | clean — `process_harmful_batch_from_prepared` divides with guarded denominator; FNV-1a hashing on `.as_bytes()` safe |
| `src/analysis/synapse/target_analysis/statistics.rs` | 330 | clean — all capacity sites bounded by live data; subtraction guarded; five uncovered loops all bounded |

### scoring

4,426 lines. Swept by Issue #2107.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/scoring/mod.rs` | 12 | clean — six `pub mod` declarations and a doc comment; no executable code, so nothing to allocate, compare or divide |
| `src/analysis/scoring/calibration_correction.rs` | 1490 | clean — the untrusted `failureCache` ratio is skipped unless `expected_error_reduction != 0.0` **and** the quotient is finite, and every EWMA is a convex combination clamped to `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]` |
| `src/analysis/scoring/confidence.rs` | 592 | clean — the `df` narrowing needs 2³² samples to misbehave, the interval margin passes a `clamp` with a finite ceiling, and `model_r_squared` is `None` at all seven production call sites |
| `src/analysis/scoring/cross_validation.rs` | 441 | clean — `fold_count` is the compile-time `5` of `CrossValidationConfig::default()`, the only production constructor; no FFI field reaches it |
| `src/analysis/scoring/error_distribution.rs` | 607 | clean — both `from_errors` call sites pre-filter `is_finite`, so neither the `total_cmp` sort nor the histogram bin cast can see a NaN, and the bin index is saturating-cast then `.min(NUM_BINS - 1)` |
| `src/analysis/scoring/sample_creature_disconnect.rs` | 201 | clean — `detect_disconnect` rejects `total_count == 0`, `improved_count > total_count` and a non-finite actual **before** it divides; the model guard for the whole chunk |
| `src/analysis/scoring/weights/mod.rs` | 531 | clean — constants and re-exports only; the two ceilings and two ratio floors are compile-time `f32` literals no caller can move |
| `src/analysis/scoring/weights/adjustment.rs` | 71 | clean — both helpers would return `Some(non-finite)` if fed one, but every operand is finite by an upstream gate: synapse weights are rejected at deserialisation (Issue #2132) and the proposed delta comes from `compute_outgoing_weight` |
| `src/analysis/scoring/weights/calculation.rs` | 398 | clean — `compute_outgoing_weight` rejects a non-finite raw weight before its clamp, and the bias grid search only ever accepts on a `>` comparison a NaN loses |
| `src/analysis/scoring/weights/normalisation.rs` | 83 | clean — the accumulator skips samples whose activation or error is non-finite and delegates the verdict to `calculate_optimal_outgoing_weight` |

### recommendation core

3,463 lines. Swept by Issue #2108.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/recommendation/mod.rs` | 14 | clean — nine `pub mod` declarations and a doc comment; no executable code, so nothing to allocate, compare or divide |
| `src/analysis/recommendation/activation_recommendation.rs` | 927 | clean — every suitability score starts as a compile-time literal, and the one input-derived multiplier (`apply_gradient_flow_penalty`'s `(1.0 - negative_fraction * 0.5).max(0.3)`) is bounded to `[0.3, 1.0]` by a clamp on both ends, so the ranked value cannot leave the constant space however hostile the records; the two ranking defects found here were out of class, filed as #2184 and **since fixed** (PR #2187, commit `3d1b24f`, merged into this branch) |
| `src/analysis/recommendation/fan_in.rs` | 608 | findings filed — #2181 (the two `partial_cmp(…).unwrap_or(Equal)` sorts are not total orders and the `corr.abs() < THRESHOLD` filter fails open, so a NaN correlation reachable from finite records empties the `MAX_INPUTS_PER_TARGET` window), #2182 (`compute_least_squares_improvement`'s `f32` `original_sse` and `compute_two_input_regression`'s `improvement as f32` narrowing both yield `+inf` from finite records, and all four `evaluate_fan_in_pair` gates are fail-open for it, so a crafted pair ranks first on an honest correlation) and #2183 (the target × input scan consults no deadline) |
| `src/analysis/recommendation/gradient_discovery.rs` | 330 | findings filed — #2182 (`mean_gradient.abs() * effective_delta.abs()` overflows to `+inf` from finite operands and the descending `total_cmp` ranks it first; the `!mean_gradient.is_finite()` guard closes only the NaN half) and #2183 |
| `src/analysis/recommendation/multi_hop.rs` | 497 | findings filed — #2182 (two paths to rank 1: `compute_mean_abs_error` sums the target's **whole** error map in `f32`, so `+inf` is reachable on observation indices the gating correlation never sees; and `find_three_hop_extensions` spells its correlation filter `<`, the fail-open direction, so a NaN `combined_corr` reaches the same descending `total_cmp` and sorts **above** `+inf`) and #2183. Only the two-hop filter at `detect_multi_hop_candidates` is fail-closed — the two filters in this file point in opposite directions |
| `src/analysis/recommendation/output_bias_drift.rs` | 389 | finding filed — #2182: `sum_error / n` overflows to `+inf`, the `mean_error.abs() < MIN` noise gate is fail-open, the descending `total_cmp` ranks it first and the emitted `SetBias` carries a non-finite bias |
| `src/analysis/recommendation/output_competition.rs` | 325 | clean — unreachable: no production caller, so no untrusted input reaches its `sum_min` overflow or its O(outputs²) pair loop; the dead module itself is filed as #2185 |
| `src/analysis/recommendation/sample_weighted.rs` | 373 | clean — every per-record error is laundered through `is_finite` to `0.0`, an overflowed weight total renormalises to zero rather than to an infinity, and `.min(10.0)` / `.min(0.1)` cap the ranked improvement |

### recommendation batch_successful + epistatic

2,309 lines. Swept by Issue #2109.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/recommendation/batch_successful/mod.rs` | 67 | clean — module declarations, re-exports and two plain `derive`d structs; no executable code, so nothing to allocate, compare or divide |
| `src/analysis/recommendation/batch_successful/detection.rs` | 240 | clean — the `!improvement.is_finite()` rejection sits **before** the descending `total_cmp`, both least-squares sums accumulate in `f64`, and the weight comes from `calculate_optimal_outgoing_weight`, which rejects a non-finite result itself |
| `src/analysis/recommendation/batch_successful/grouping.rs` | 162 | clean — no division anywhere despite the issue body's expectation; the greedy batching is bounded by the compile-time `MAX_BATCHES` (20) × `MAX_BATCH_SIZE` (4) over an already-truncated 50-candidate list, and the only float test is a `> 0.0` a NaN loses |
| `src/analysis/recommendation/epistatic/mod.rs` | 155 | clean — module declarations, re-exports, four `derive`d result structs and one `enum`; no executable code |
| `src/analysis/recommendation/epistatic/candidate_generation.rs` | 616 | **finding filed — #2190.** The O(n²) pair scan has no ceiling, no candidate cap and no deadline check, and the deadline is already in scope one frame up. Float handling is clean: `improvement.is_finite()` gates the only value that escapes, and `target_impact` is the compile-time `1.0` / `0.5` |
| `src/analysis/recommendation/epistatic/deduplication.rs` | 100 | **out-of-class finding filed — #2191.** All four sorts are `total_cmp`, so the order is total and a NaN cannot suppress anything (see **NaN suppression** below); the defect is that tied keys come out in `HashMap`-seed order, which the stable sort preserves |
| `src/analysis/recommendation/epistatic/pre_screening.rs` | 464 | clean — the `individual_improvement >= 0.0` pre-screen is fail-closed for a NaN dominance key, every accumulator is `f64`, all four divisors are guarded above the division, and the emitted gain passes a `> 0.0` test in `f64` before it is narrowed |
| `src/analysis/recommendation/epistatic/scoring.rs` | 505 | clean on **unreachability**, not soundness — `detect_interfering_pairs` and `check_saturation_risk` have no production caller (#2192) and the `f32` `total_combined` accumulator in the latter has the #2182 overflow shape; the two `filter_interfering_*` functions production does use compare only an `f64` Pearson |

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
| `preparation.rs::build_creature_lookups` | `NeuronIndex::with_capacity(neurons.len() + input + synapses.len() / 10)` | `MAX_CREATURE_INPUT_NEURONS` (1,000,000) caps `input` at every FFI entry point; the other two terms count live `Vec`s, so the sum cannot overflow `usize` | bounded |
| `candidate_generation.rs::build_ordered_neurons` | `Vec::with_capacity(creature.input + creature.neurons.len())` | same cap on `input`; `neurons.len()` counts a live `Vec`, so the sum is at most 1,000,000 + a deserialised vector's length | bounded |
| `candidate_generation.rs::group_sources_by_locality` | `vec![false; sources.len()]` | `sources` is a live slice of per-target sources already materialised by `filter_and_load_sources` | bounded |
| `gpu_evaluation.rs::evaluate_all_activation_specs_batched` | `vec![None; plan.specs.len()]` | `plan.specs` is a subset of the compile-time `ACTIVATION_SPECS` array; no caller-supplied count reaches it | bounded |
| `filtering.rs::truncate_combined_synapse_candidate_sets` | `Vec::with_capacity(total)` | `total` is the sum of three live vector lengths, and the branch is only reached when `total > limit` | bounded |
| `holdout_validation.rs::split_samples_holdout` | `Vec::with_capacity(samples.len() - validate_count)` | the early return rejects fewer than `HOLDOUT_MIN_SAMPLE_COUNT` (20) samples and `validate_count = round(0.3n) < n` for every `n >= 20`, so the subtraction cannot wrap | bounded |
| `holdout_validation.rs::split_samples_holdout` | `Vec::with_capacity(validate_count)` | `validate_count = round(0.3 × samples.len())`, at most 30% of a live slice's length | bounded |
| `issue_2161_locality_cancellation_test.rs::build_sources` | `Vec::with_capacity(count)` ×2 | fixture builder in a file declared behind `#[cfg(test)] #[path = …]` in `candidate_generation.rs`, so it is not compiled into the shipped `cdylib` — `count` is a compile-time literal passed by the tests, and no untrusted path reaches it | n/a — test-only |
<!-- section: synapse post-processing -->
| `structural_patterns.rs::detect_noisy_vs_trusted` | `HashMap::with_capacity(records.len())` | `records` is a live slice of already-materialised observation records, so the hint is the length of a collection the loader has itself allocated | bounded |
| `structural_patterns.rs::detect_noisy_vs_trusted` | `Vec::with_capacity(target_map.map.len())` | `target_map.map` is a live `HashMap` built earlier in the same pass; its length is at most the record count | bounded |
| `structural_patterns.rs::detect_collapsible_hidden_neurons` | `HashMap::with_capacity(records.len())` | same live record slice; the reservation cannot exceed what the records already occupy | bounded |
| `structural_patterns.rs::detect_collapsible_hidden_neurons` | `Vec::with_capacity(target_map_b.map.len())` | same live `HashMap` length, not a caller-supplied count | bounded |
| `adaptive_proposal.rs::generate_gaussian_candidates` | `Vec::with_capacity(count)` | `count` is bound from the compile-time constant `ADAPTIVE_PROPOSAL_CANDIDATE_COUNT` (12) and no caller can override it — the issue body's question about the origin of `count` resolves to a constant, not to input | bounded |
| `structural_patterns.rs::build_collapse_input` | `Vec::with_capacity(n_samples as usize)` ×3 | `#[cfg(test)]` fixture builder — `n_samples` is a loop bound from compile-time literals, and the symbol is not compiled into the shipped `cdylib`, so no untrusted path reaches it | n/a — test-only |
| `post_processing.rs`, `add_synapse_gating.rs` | none | neither file allocates a collection with a size hint | n/a |
<!-- section: synapse scoring + target_analysis -->
| `statistics.rs::filter_and_load_sources` | materialised `Vec<SourceMetadata>` on the per-target source list | `sources` is the result of SQL-like filtering of a live candidate set, bounded by the input creature's topology and the FFI validation layer | bounded |
| `statistics.rs::prepare_harmful_samples` | materialised `Vec<(SourceUuid, TargetUuid)>` for pairwise work | `bad_pairs` iterates over a constructed vector of pre-computed neuron pairs, bounded by `max(neurons.len()²)` | bounded |
| `evaluation.rs::process_harmful_batch_from_prepared` | `batch_stats.len()` and `work.samples.len()` iteration counts | both bounded by the live length of input slices passed in from the caller, which are themselves materialised within the same analysis pass | bounded |
| `tests.rs::test_magnitude_ratio_noise_level_improvements_collapse_neuron_gain` | test fixture sizes | compile-time test values; no caller-supplied bounds | bounded |
<!-- section: scoring -->
| `cross_validation.rs::compute_cross_validation_score` | `Vec::with_capacity(config.fold_count)` | `fold_count` is **not** caller-controllable: the only production constructor is `CrossValidationConfig::default()` in `neuron/evaluation.rs::apply_cross_validation_penalty`, which sets the compile-time `5`, and no FFI request field deserialises into the struct. The secondary bound is weaker than it looks and must not be relied on: `fold_count < 2` does return early, but `samples.len() / fold_count < min_samples_per_fold` is never true against a `min_samples_per_fold` of `0`, so a hypothetical in-crate caller that set both `fold_count: usize::MAX` and `min_samples_per_fold: 0` would reach the reservation. The bound is the absent constructor, not the precondition | bounded |
| `error_distribution.rs::detect_modes_histogram` | `vec![Vec::new(); NUM_BINS]` | `NUM_BINS` is a `const usize = 20` local to the function; no input reaches the size. Note this allocation is currently unreachable in production for a second reason — its only caller, `detect_error_modes`, is itself dead (see the outcome below) | bounded |
| the other eight `scoring` files | none | none of them allocates a collection with a size hint; the input-keyed `HashMap` growth in `calibration_correction.rs` has no size hint and is analysed in the outcome below | n/a |
<!-- section: recommendation core -->
| `output_bias_drift.rs::output_bias_drift_to_coordinated_candidates` | `Vec::with_capacity(candidates.len())` | `candidates` is the live vector the detector just built, at most one entry per output neuron of the deserialised creature | bounded |
| `gradient_discovery.rs::gradient_candidates_to_coordinated` | `Vec::with_capacity(candidates.len())` | same shape — at most one entry per synapse that survived the magnitude and consistency gates | bounded |
| `output_competition.rs::output_competition_to_coordinated_candidates` | `Vec::with_capacity(candidates.len())` | same shape, and unreachable in production in any case: the module has no caller (#2185) | bounded |
| `sample_weighted.rs::compute_sample_weights` | `vec![uniform; abs_errors.len()]` | `abs_errors` is one `f32` per record of a live slice the loader has already materialised, so the reservation cannot exceed what the records occupy; the `records.is_empty()` early return makes the `1.0 / len` divisor non-zero | bounded |
| `sample_weighted.rs::stratify_samples` | `abs_errors.clone()` median scratch | same live record slice, cloned once per neuron for the `select_nth_unstable_by` median (Issue #943); `median_idx = len / 2 < len` for every non-empty slice, so the select cannot panic | bounded |
| the other four `recommendation core` files | none | `mod.rs`, `activation_recommendation.rs`, `fan_in.rs` and `multi_hop.rs` allocate no collection with a size hint; their `HashMap`s grow entry-by-entry from live record slices, with no hint, no multiplier and no per-entry fan-out | n/a |
<!-- section: recommendation batch_successful + epistatic -->
| all eight files | none | neither subtree contains a `with_capacity`, a `reserve` or a `vec![value; count]` in its production half. Every collection here grows entry-by-entry — the `HashMap`s in the batch-successful detector are keyed on already-materialised record slices, and the dominant-neuron groups on a UUID that came from the creature. The one **unbounded** growth is a plain `Vec::push` with no size hint: the epistatic pair vector grows one candidate per surviving pair of an O(n²) scan, which is recorded as part of #2190 rather than as a capacity-from-input site | n/a — no sized allocation |
<!-- section: shared -->
| — | none | `src/analysis/shared/` allocates no collection sized from input | n/a |

## Float comparison sites

Every comparison against a float that can be `NaN` or `-0.0`, with where the
value comes from and what happens when it is `NaN`.

| Site (`file.rs::symbol`) | Comparator | Value origin | NaN handling | Verdict |
| --- | --- | --- | --- | --- |
<!-- section: synapse pipeline -->
| `filtering.rs::truncate_combined_synapse_candidate_sets` | `sort_by` on descending `total_cmp` | `expected_creature_score_gain` of the helpful, harmful and coordinated buckets | a positive NaN sorts **above** `+inf`, so it would be kept first and survive truncation | bounded — every gain it sorts is finite by construction: each producer routes through `improvement.rs::finalise_improvement`, which applies `select_finite`. The two call sites differ in what else protects them: `candidate_aggregation.rs::merge_coordinated_structural_replacements` runs `reject_non_finite_gains` over the coordinated bucket first (Issue #1367), while `post_processing.rs::apply_post_processing` has no such filter and rests on the construction invariant alone — the discount multipliers it applies before the sort are #2105's and #2106's rows |
| `filtering.rs::expected_gain_replace_synapse_with_hidden_neuron` | `total_baseline_error_sq <= EPSILON` | sum of squared per-sample errors after the removed synapse's contribution is folded back in | a NaN total fails the `<=`, so the guard passes it through | bounded — the improvement helper it then calls returns a `select_finite` value, and `merge_coordinated_structural_replacements` drops a non-finite gain before any sort |
| `activation_evaluation.rs::evaluate_activation_candidate` | `gain > fallback_score` against the `f32::MIN` sentinel | subset candidate's `expected_creature_score_gain` | a NaN loses `>`, so it never displaces the sentinel and never becomes the fallback candidate | bounded |
| `activation_evaluation.rs::evaluate_activation_candidate` | `improvement > best_train_improvement` against the `f32::NEG_INFINITY` sentinel | hold-out training improvement per weight variant | a NaN loses `>`; when every variant is NaN the base weight and a zero bias are reported unchanged | bounded — the improvement helpers cannot return NaN |
| `activation_evaluation.rs::evaluate_activation_candidate` | `absolute_improvement < 0.001` | `neuron_error_improvement × baseline_sq` | a NaN fails the `<`, so this filter passes it through | bounded — the two accept gates after it are `>` comparisons a NaN loses |
| `activation_subset_evaluation.rs::evaluate_activation_for_subset` | `net_improvement <= 0.0`, then `> best_net_improvement` from a `0.0` sentinel | net improvement over all samples | a NaN fails the `<=` and loses the `>`, so it is never stored as best | bounded |
| `relu_evaluation.rs::evaluate_relu_candidates_split` | `total_baseline_error_sq <= EPSILON`, then `net_improvement > best_improvement` from the `threshold` sentinel | per-sample squared errors and the ReLU net improvement | a NaN baseline fails the early `<=` guard, but the accept gate is a `>` a NaN loses | bounded |
| `gpu_evaluation.rs::evaluate_all_activation_specs_batched` | `net_improvement <= threshold`, then `absolute_improvement < 0.001`, then `current_best.is_none() \|\| net_improvement > …` | batched GPU sufficient statistics fed through `compute_activation_improvement_and_count` | a NaN fails both the `<=` and the `<`, and the `is_none()` short-circuit would then store it without ever comparing it — the only accept path in these files that does not end in a `>` a NaN loses | bounded — `finalise_improvement` applies `select_finite`, so `net_improvement` is finite before any of the three |
| `candidate_generation.rs::compute_obs_index_overlap` | `overlap >= MIN_LOCALITY_OVERLAP` | intersection size over the smaller observation-index set | the division cannot produce a NaN — both sets are checked non-empty first, so the divisor is at least 1 | bounded |
<!-- section: synapse post-processing -->
| `post_processing.rs::apply_post_processing` | `sort_by` on descending `total_cmp`, helpful bucket | `expected_creature_score_gain` after the helpful discount multipliers are applied | IEEE-754 totalOrder puts a positive NaN **above** `+inf` and every finite value below both, so descending order heads the list with a non-finite gain | **unbounded — #2167.** The `retain(gain >= floor)` filter ahead of it drops a NaN but keeps `+inf`, and this call site never runs `reject_non_finite_gains` |
| `post_processing.rs::apply_post_processing` | `sort_by` on descending `total_cmp`, harmful bucket | same field on the harmful bucket | same totalOrder placement | **unbounded — #2167.** The harmful bucket runs the same `apply_min_expected_gain_floor_for_synapses` pass as the helpful one, so `+inf` survives here too |
| `post_processing.rs::apply_post_processing` | `sort_by` on descending `total_cmp`, coordinated bucket | same field on the coordinated structural bucket | same totalOrder placement | **unbounded — #2167.** The `retain(gain > 0.0)` filter (Issues #1110, #1128) drops a NaN and keeps `+inf` |
| `post_processing.rs::scale_by_error_fraction` | `total_error_sq <= EPSILON` | summed squared per-neuron errors from `compute_neuron_error_sq_map` | a NaN or `+inf` total fails the `<=`, so the guard passes it to the division; `inf / inf` is NaN and `clamp` propagates NaN unchanged | the `+inf` reachability of this sum is the escalation path recorded in #2167 |
| `post_processing.rs::compute_neuron_error_sq_map` | `e.is_finite()`, then `error_sq > EPSILON` on the summed square | per-sample neuron error | a NaN error is filtered out before the map is built, and a NaN sum would lose the `>` and simply not be inserted | bounded for NaN — but the retained `e * e` overflows to `+inf` above `f32::MAX.sqrt()` ≈ 1.84e19, which neither the filter nor the `>` catches: `+inf > EPSILON` is true, so the infinity is inserted |
| `post_processing.rs::apply_min_expected_gain_floor_for_synapses` | `retain(gain >= floor)` | `expected_creature_score_gain` on the helpful and harmful buckets, before both sorts | a NaN loses the `>=` and is dropped; `+inf` wins it and is kept, which is the asymmetry #2167 rests on | **unbounded — #2167.** This is the only filter either bucket gets |
| `structural_patterns.rs::detect_noisy_vs_trusted` | `(a.weight - b.weight).abs() > WEIGHT_EPS`, `(a.mean - b.mean).abs() > MEAN_EPS`, `ratio < MIN_VAR_RATIO` — the first two skip on `>`, the third on `<` | pairwise synapse-weight and activation-mean differences, and the variance ratio, over untrusted records | every one is a skip-on-comparison filter, so a NaN fails the comparison whichever way it points and **falls through** rather than being skipped | bounded — `var.max(0.0)` launders a NaN variance to `0.0`, `trusted.var.max(EPSILON)` keeps the ratio's divisor positive, and `improvement.rs::finalise_improvement` routes every improvement through `select_finite`, so no non-finite gain is emitted |
| `structural_patterns.rs::detect_noisy_vs_trusted::activation_mean_and_variance` | `n <= 0.0` before `sum / n` | count of finite activations in the record set | the counter is incremented only inside `r.activation.is_finite()`, so it is a whole number and never NaN; the guard returns `None` when nothing finite was seen | bounded — the divisor is at least `1.0` whenever the division runs |
| `structural_patterns.rs::detect_noisy_vs_trusted` | `improvement <= 0.0`, then `best_gain >= improvement` against the running best | `compute_synapse_improvement_and_count` | a NaN would fail the `<=` and pass, then lose the `>=` and displace the running best, but the helper's `select_finite` makes both unreachable | bounded by construction |
| `structural_patterns.rs::detect_collapsible_hidden_neurons` | `baseline_sq <= EPSILON`, then `weight.abs() < bypass_floor`, then its own `improvement <= 0.0` | summed squared baseline error, the optimal outgoing weight, and the collapse improvement | a NaN fails all three comparisons and passes through each | bounded — `calculation.rs::compute_outgoing_weight` returns `None` on a non-finite raw weight, and the improvement arrives through the same `select_finite` path, so the values compared are finite whenever they exist |
| `add_synapse_gating.rs::should_skip_add_synapse_by_outcome` | `rate < success_threshold` | `ModuleStats::success_rate` over the FFI-supplied `ModuleOutcomeTracker` | a NaN rate fails the `<`, so the gate **fails open** and every add-synapse candidate is kept | **unbounded — #2170.** The rate is not NaN-free: a deserialised `soft_failures` of `NaN` or `±inf` reaches the divisor unvalidated |
| `add_synapse_gating.rs::should_skip_add_synapse_by_density` | `density > density_threshold` | `synapses.len() / total_neurons` on the untrusted creature | the divisor is guarded by an `== 0` early return, so the quotient is finite and the comparison is well-defined | bounded |
| `adaptive_proposal.rs` | none | no float comparator or sort in the file | n/a | n/a |
<!-- section: synapse scoring + target_analysis -->
| `evaluation.rs::process_harmful_batch_from_prepared` | `neuron_error_improvement <= 0.0` | `(harmful_count as f32 - helpful_count as f32) / total_count as f32`, both operands are u32-cast-to-f32, divisor is guarded `> 0` before the division | a NaN would fail the `<=`, so this guard passes it through; reachable only if arithmetic itself produced NaN, which cannot happen with finite operands and valid arithmetic | bounded — direct arithmetic on finite u32-derived numerands cannot produce NaN |
| `scoring/improvement.rs::compute_synapse_improvement_and_count` | `select_finite` applied before final `clamp` on all improvement return paths | the improvement is computed from helpful/harmful counts and squared errors; three dispatch targets guard by match statement | the top-level guard returns all-finite tuple `(0.0, 0, 0, samples.len(), 0.0)` before any dispatch, closing the NaN production path entirely | bounded — three guards + one defensive return |
<!-- section: scoring -->
| `error_distribution.rs::compute_percentiles` | `sort_by(f32::total_cmp)`, then interpolation between the two bracketing indices | the per-sample `avg_error` column of the caller's Parquet records, via `ErrorDistribution::from_errors` | IEEE-754 totalOrder would place a negative NaN **first** and a positive NaN **last**, so `p10` and `p90` would report a NaN and `iqr` would follow it; the input cannot contain one | bounded — both production callers (`synapse/post_processing.rs::build_metadata` and `neuron/post_processing.rs::build_neuron_results`) collect the error column through `.filter(\|e\| e.is_finite())`, and `from_samples` applies the same filter. `from_errors` is `pub`, so the guarantee lives at the callers, not in this function |
| `error_distribution.rs::detect_modes_histogram` | `range < 1e-6` early return, then `((error - min) / bin_width).floor() as usize` | same finite-filtered error column | a NaN `range` would fail the `<` and fall through, and a NaN bin index saturates to `0` under Rust's float→int cast, not to UB; `+inf / +inf` would likewise yield NaN → `0` | bounded — the errors are finite-filtered before the fold, the `< 1e-6` return makes `bin_width` strictly positive, and `.min(NUM_BINS - 1)` caps the index whatever the cast produced, so `bins[bin_idx]` cannot panic |
| `error_distribution.rs::detect_modes_histogram` | `modes.sort_by_key(Reverse(sample_count))` | bin populations | integer key, no float compared | n/a — not a float comparison |
| `error_distribution.rs::from_errors` | `std_dev > 1e-10` before the skewness and kurtosis divisions | `f64` moments of the finite-filtered error column | a NaN `std_dev` fails the `>` and takes the `else` arm, emitting the documented `0.0` / `3.0` degenerate defaults rather than dividing | bounded — the guard is fail-closed for NaN, which is the safe direction here |
| `error_distribution.rs::ErrorDistribution::count_outliers` and `::filter_outliers` | `s.avg_error.abs() > threshold.abs()` | unfiltered sample errors (both helpers take the raw slice) | a NaN error loses the `>` and is simply not counted or collected as an outlier | bounded — and no production caller exists: the outlier helpers and their two env levers are dead, see the outcome below |
| `error_distribution.rs::ErrorDistribution::is_likely_bimodal` | `sample_count < 20`, then `(skewness² + 1) / kurtosis > 5/9`, and `mean_median_gap > std_dev * 0.5` | the distribution's own moments | a zero `kurtosis` divides to `±inf`, which answers the `>` definitely rather than trapping; a NaN coefficient would lose both comparisons and report "not bimodal" — the conservative answer | bounded — the moments are finite by the `from_errors` filter, and the helper has no production caller |
| `error_distribution.rs::ErrorDistribution::has_significant_outliers` | `min < p25 - 1.5·iqr` and `max > p75 + 1.5·iqr` | the percentile array and the `iqr` derived from it | a NaN fence would lose both comparisons and report "no outliers" | bounded — same finite-percentile guarantee, and no production caller |
| `calibration_correction.rs::from_failure_cache` | `expected_error_reduction == 0.0`, then `ratio.is_finite()` | the caller-supplied `failureCache` JSON, both fields `f32` | `-0.0 == 0.0` is true, so a negative zero divisor is skipped by the same test; a NaN or `±inf` quotient — including one produced by `0.0 / 0.0` if the first test were ever removed — is dropped by the explicit `is_finite` test rather than entering the EWMA | bounded — the two tests together are a whitelist, not a blacklist: only a finite ratio is pushed |
| `calibration_correction.rs::ewma` | `(1.0 - alpha) * acc + alpha * x` accumulated in supplied order | the finite ratios above | unreachable — a convex combination of finite values cannot leave `[min, max]` of those values, so `acc` can neither overflow to `±inf` nor become NaN | bounded by construction, then `clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION)` |
| `sample_creature_disconnect.rs::detect_disconnect` | `improved_count as f32 / total_count as f32 >= SAMPLE_DISCONNECT_RATIO_THRESHOLD`, and `actual_error_reduction <= 0.0` | `improvedCount` / `totalCount` / `actualErrorReduction` of the untrusted failure cache | the divisor is rejected at `total_count == 0` and the numerator at `improved_count > total_count`, both **before** the division, and a non-finite `actual_error_reduction` returns `false` up front | bounded — the reference guard for this chunk: divisor, numerator and finitude all checked ahead of the arithmetic |
| `weights/adjustment.rs::clamp_weight_update_delta` | `(old_weight + proposed_delta_weight).clamp(±MAX_OUTGOING_WEIGHT)`, then `delta_weight.abs() <= EPSILON` | an existing synapse's weight from the caller's creature JSON, and the least-squares weight | `f32::clamp` **propagates** NaN, and both a NaN and an `±inf` delta lose the `<=`, so the degenerate-delta gate would return `Some(non-finite)` — the "NaN passes a clamp silently" shape | bounded — neither operand can be non-finite: `ffi_types/mod.rs::deserialise_synapse_weight` rejects a non-finite synapse weight at the FFI boundary (Issue #2132), and the proposed delta is a `calculate_optimal_outgoing_weight` return, which `compute_outgoing_weight` has already tested with `is_finite` |
| `weights/adjustment.rs::coordinated_structural_activation_delta` | `noisy_weight.abs() <= EPSILON` before `delta_trusted_weight / noisy_weight` | two synapse weights and two activations | an `±inf` `noisy_weight` loses the `<=`, and `inf / inf` is NaN, so the function would return `Some(NaN)` from a guard whose `None` contract says the value is unusable | bounded — same Issue #2132 gate on both weights; the sole call site (`structural_patterns.rs::detect_noisy_vs_trusted`) additionally drops the sample on `!activation.is_finite()`, so the hazard is closed twice |
| `weights/calculation.rs::compute_outgoing_weight` | `sum_activation_sq <= EPSILON`, then `!raw_weight.is_finite() \|\| raw_weight.abs() <= EPSILON`, then `clamp(-max_outgoing, max_outgoing)` | least-squares sums over finite-filtered samples | a NaN `sum_activation_sq` fails the first `<=` and falls through, but the quotient is then NaN and the explicit `is_finite` test rejects it before the clamp — the ordering that `clamp_weight_update_delta` does not have | bounded — the finitude test sits **between** the divisor guard and the clamp |
| `weights/calculation.rs::calculate_optimal_identity_outgoing_and_bias` | `n <= 0.0`, `std_dev_a < MIN_SOURCE_STD_DEV`, `det.abs() > EPSILON`, `!outgoing_weight_raw.is_finite()`, `!bias.is_finite()` | `f32` sums over samples whose activation and error are both finite | an `f32` sum that overflows to `+inf` makes `var_a` NaN, which `max(0.0)` launders to `0.0` — `f32::max` returns the non-NaN operand — so the low-variance branch is taken and the result still passes `calculate_optimal_outgoing_weight`'s finitude test | bounded — two explicit `is_finite` rejections plus the `sensible_bias_abs_max_for_squash` ceiling |
| `weights/calculation.rs::calculate_optimal_bias` | `total_baseline_error_sq <= EPSILON`, then `error_reduction > best_error_reduction` from an `f32::NEG_INFINITY` sentinel | per-sample squared errors and the searched bias grid | a NaN baseline fails the `<=` and falls through, but every accept is a `>` a NaN loses, so the returned bias stays the `0.0` initialiser; an `+inf` new-error sum makes `error_reduction` `-inf`, which also loses the `>` | bounded — accept-on-`>` from a sentinel, the pattern Issue #2105 recorded for the activation evaluators |
| `confidence.rs::compute_confidence_interval` | `margin = (t_crit * standard_error).clamp(0.0, expected_score_gain.abs().max(0.5))`, and `n > 1.0` before `error_std_dev / n.sqrt()` | the sample error variance and the point estimate | `f32::max` returns `0.5` for a NaN `expected_score_gain`, so the *ceiling* stays finite, and an `+inf` `standard_error` (an `f32`-saturating variance) clamps to it rather than producing an infinite interval. The **margin** is therefore always finite; the emitted bounds are `gain ± margin`, so a non-finite `expected_score_gain` would still reach both of them — this function has no finitude gate on its own point estimate | bounded, but by the callers: every production caller passes an improvement that `scoring/improvement.rs::compute_synapse_improvement_and_count` has already put through `select_finite` (the same invariant Issue #2106 recorded), and `compute_error_variance` skips every non-finite sample and returns `variance.max(0.0)` |
| `confidence.rs::compute_confidence_metrics` | `model_r_squared.map_or(1.0, compute_model_fit_confidence)`, then `.clamp(0.0, 1.0)` on the geometric mean | the optional R² argument | `f32::clamp` propagates NaN, so a NaN R² would make `prediction_confidence` NaN, which serde emits as `null` | bounded — all seven production call sites pass `None`; no path supplies an R² at all |
| `confidence.rs::compute_source_variance_confidence` | `count < 2` guard, then `(std_dev / MIN_CONFIDENT_STD_DEV).clamp(0.0, 1.0)` | source activations, non-finite ones skipped by the accumulator | the divisor is the compile-time `MIN_CONFIDENT_STD_DEV`, never zero, and `variance.max(0.0)` launders a NaN variance from an overflowed sum to `0.0` before the `sqrt` | bounded |
| `confidence.rs::compute_sample_confidence` | `(sample_count as f32 / MIN_CONFIDENT_SAMPLES).min(1.0)` | a slice length | integer-derived, so no NaN is possible on either side of the division | bounded |
| `cross_validation.rs::evaluate_fold` | `activation_sq_sum > 1e-10` before `error_activation_sum / activation_sq_sum`, then `new_error_mag + 1e-8 < old_error_mag` | per-sample activations and errors, both `is_finite`-filtered by the loop's own `continue` | the divisor guard is fail-closed (a NaN loses the `>` and takes the `0.0` branch), and the improvement test is a `<` a NaN loses, so an unclassifiable sample falls to `negative_count` — the conservative side | bounded — the `f64` accumulators are fed only from finite `f32` samples |
| `cross_validation.rs::PerformanceVariance::from_folds` | `fold(f64::NEG_INFINITY, f64::max)` / `fold(f64::INFINITY, f64::min)`, and `variance.max(0.0)` | fold improvement ratios, each a `u32 / u32` quotient | the ratios are integer-derived and the `total == 0` case returns the neutral `0.5` before dividing, so no NaN enters the fold; `variance.max(0.0)` is the documented floating-point-noise clamp | bounded |
| `cross_validation.rs::compute_brittleness_penalty` | `config.variance_threshold <= 0.0` before dividing, then `.clamp(0.0, 1.0)` | the variance above and the configured threshold | a NaN threshold would lose the `<=` and fall through, and the quotient would then be NaN, which `clamp` propagates | bounded — the only production `variance_threshold` is the compile-time `0.04` of `CrossValidationConfig::default()`, the same absent-constructor bound as the capacity row |
| `cross_validation.rs::CrossValidationResult::is_consistent` / `::is_brittle` | `variance < threshold` and `variance >= threshold` | the variance above | the pair is deliberately not a partition under NaN — a NaN variance would answer `false` to both — but neither has a production caller and the variance cannot be NaN | bounded |
| `weights/normalisation.rs::compute_range_aware_sums` | `(sample.activation - sv).abs() <= sentinel_tolerance` | sample activations and the detected sentinel values from `detect_observation_ranges` | the sample is already `is_finite`-filtered when this test runs; a NaN sentinel value would lose the `<=`, so the sample is **kept** rather than excluded, which only widens the fit set and never fabricates a weight | bounded — the verdict is delegated to `calculate_optimal_outgoing_weight`, which rejects a non-finite result whatever the sums contain |
| `weights/calculation.rs::compute_outgoing_weight` and `::calculate_optimal_identity_outgoing_and_bias` | `incoming_weight.abs() > 1.0`, then `ratio = incoming_weight.abs() / (clamped.abs() + EPSILON) < min_ratio` | the candidate's incoming weight against the clamped outgoing weight | the `+ EPSILON` makes the divisor strictly positive, so the ratio is finite whenever the numerator is; a NaN `incoming_weight` would lose the `> 1.0` and skip the reliability gate entirely, passing the candidate | bounded — `incoming_weight` is either the literal `1.0` (every synapse call site) or a creature synapse weight, which Issue #2132 has already proved finite |
<!-- section: recommendation core -->
| `fan_in.rs::detect_fan_in_candidates` | `corr.abs() < INPUT_ERROR_CORRELATION_THRESHOLD`, then `sort_by` on descending `partial_cmp(…).unwrap_or(Equal)` over `corr.abs()`, then `truncate(MAX_INPUTS_PER_TARGET)` | `detection/stats.rs::pearson_correlation` over the input's activations and the target's first errors | a NaN loses the `<` and is **kept**, then compares `Equal` to every finite key while the finite keys order among themselves — not a total order, so `sort_by` may panic and otherwise leaves the order unspecified, which is what the `truncate(15)` then acts on | **unbounded — #2181.** A NaN correlation is reachable from finite records: an activation swing near `±2e30` overflows the `f32` covariance accumulator, `pearson_correlation`'s `denom < f32::EPSILON` guard loses to the resulting NaN and `f32::clamp` propagates it |
| `fan_in.rs::detect_fan_in_candidates` | `sort_by` on descending `partial_cmp(…).unwrap_or(Equal)` over `estimated_improvement`, then `truncate(MAX_FAN_IN_CANDIDATES)` | `evaluate_fan_in_pair`'s scaled two-input regression improvement | no **NaN** can reach it — `compute_two_input_regression` ends in `(ee - residual_sse).max(0.0)` and `f64::max` returns the non-NaN operand, so a NaN improvement is laundered to `0.0` and dropped by `scaled_improvement <= 0.0` — but `+inf` reaches it untouched, and descending `partial_cmp` puts `+inf` first | **unbounded — #2182.** The NaN half is closed by that laundering accident; the `+inf` half is not closed at all. Both producers overflow from finite records (rows below), and `.max(0.0)` passes `+inf` straight through. #2181 additionally asks for `total_cmp` here |
| `fan_in.rs::evaluate_fan_in_pair` | `mutual_corr.abs() > MAX_INPUT_MUTUAL_CORRELATION`, `best_individual <= 0.0`, `combined_improvement < best_individual * MIN_COMBINED_BENEFIT_RATIO`, `scaled_improvement <= 0.0` | the pairwise Pearson and the two least-squares improvements | every one is a skip-on-comparison filter a NaN loses, so a NaN falls **through** each — but `best_individual` is a `.max(0.0)` of two laundered improvements, so a NaN pair always fails `best_individual <= 0.0` and returns `None`. **`+inf` is a different matter:** `inf <= 0.0` is false, `inf < inf * 1.05` is false and `inf * 1.0 * 0.01 <= 0.0` is false, so all four gates pass it | **unbounded — #2182.** Closed for NaN, wide open for `+inf` — the gates test a *sign* and a *ratio*, and neither question has a finite answer for an infinity |
| `fan_in.rs::compute_least_squares_improvement` / `::compute_two_input_regression` | `sum_act_sq < 1e-10`, `det.abs() < 1e-12`, then `(original_sse - residual_sse).max(0.0)` and `(ee - residual_sse).max(0.0)` | least-squares sums over the shared samples, accumulated in `f32` and `f64` respectively | a NaN sum loses the `<` and falls through to the division, but both functions end in `.max(0.0)`, which returns the non-NaN operand. Neither tests for an **infinity**: `original_sse` is an **`f32`** accumulator of `e * e` — sixty errors near `1e19` overflow it while the residual stays small, so `(inf - finite).max(0.0)` is `+inf`, not `0.0`; and `compute_two_input_regression` accumulates in `f64` and then narrows with `improvement as f32`, a saturating cast that yields `+inf` above `f32::MAX` | **unbounded — #2182.** The two producers of the `+inf` in the rows above. `det.abs() < 1e-12` remains fail-open for NaN and fail-closed for a singular system, which is the safe direction |
| `output_bias_drift.rs::detect_output_bias_drift` | `mean_error.abs() < MIN_MEAN_ERROR_MAGNITUDE`, `majority_fraction < MIN_MAJORITY_SIGN_FRACTION`, then `sort_by` on descending `total_cmp` over `estimated_improvement` | `sum_error / n` over the per-record first error | `+inf` **wins** the noise gate (`inf < 0.01` is false), `consistency` is at least `0.2` by the majority gate, so `estimated_improvement` is `+inf`; IEEE-754 totalOrder puts `+inf` above every finite gain, so it heads the list | **unbounded — #2182.** `sum_error` is an `f32` accumulator: twenty finite errors of `1e38` overflow it, so no non-finite value has to cross the FFI boundary. `recommended_bias_delta = -mean_error` then emits a `-inf` bias in the `SetBias` payload |
| `output_bias_drift.rs::output_bias_drift_to_coordinated_candidates` and `::detect_output_bias_drift_with_descriptor` | `sort_by` on descending `total_cmp` over `expected_creature_score_gain` / `estimated_improvement` | the same field, re-ranked on the way out and after the capacity-starvation boost | same totalOrder placement | **unbounded — #2182.** Neither re-sort tests the value it ranks |
| `output_bias_drift.rs::summarise_positive_support` and `::is_capacity_starved` | `target <= POSITIVE_SUPPORT_TARGET_THRESHOLD`, `r.activation > max_activation` from a `f32::NEG_INFINITY` sentinel, `max_activation < SATURATING_ACTIVATION_THRESHOLD` | recorded targets and activations of an output neuron | a NaN target loses the `<=` and is counted; a NaN activation loses the `>` and never displaces the sentinel; the `count == 0` early return makes both divisors at least `1.0` | bounded — and `stats.mean_error.is_finite()` is tested explicitly before the synthesised candidate's bias delta is taken from it, the one finitude test in the file |
| `multi_hop.rs::detect_multi_hop_candidates` | `corr.abs() >= CORRELATION_THRESHOLD`, then `sort_by` on descending `total_cmp` over `corr.abs()` | `pearson_correlation_hashmaps` over the shared observation indices | a NaN loses the `>=` and is **dropped** — the fail-closed direction, and the contrast with `fan_in.rs`'s `<` above | bounded — no NaN key can reach the two-hop intermediate sort |
| `multi_hop.rs::find_three_hop_extensions` | `source_intermediate_corr.abs() < CORRELATION_THRESHOLD` | `compute_activation_activation_correlation`, i.e. the same `pearson_correlation_hashmaps`, over the source and intermediate activations | the **opposite** direction to the two-hop filter in the same file: a NaN loses the `<` and is **kept**, so `combined_corr = (nan * corr).sqrt()` is NaN and `estimated_improvement` is NaN | **unbounded — #2182.** The NaN then wins the `estimated_improvement <= 0.0` gate and enters the same `all_candidates` list `detect_multi_hop_candidates` sorts descending by `total_cmp`, where IEEE-754 totalOrder puts a positive NaN **above** `+inf` — so this path outranks even the `+inf` one |
| `multi_hop.rs::detect_multi_hop_candidates` and `::find_three_hop_extensions` | `estimated_improvement <= 0.0`, then `sort_by` on descending `total_cmp` over `estimated_improvement` (and again in `::multi_hop_to_coordinated_candidates`) | `corr.abs() * compute_mean_abs_error(target_errors) * 0.01` | `+inf` wins the `<= 0.0` gate and heads the descending totalOrder | **unbounded — #2182.** `compute_mean_abs_error` sums `\|e\|` over the target's **whole** error map in `f32`, while the gating correlation sees only the observation indices shared with the source, so the overflow can be placed entirely outside the correlated window |
| `gradient_discovery.rs::compute_synapse_gradient` and `::detect_gradient_candidates` | `!source.activation.is_finite()`, `!error.is_finite()`, `!mean_gradient.is_finite()` | per-sample `activation × error` products | every non-finite sample is skipped and a non-finite mean is rejected outright — the reference guard of this section, and the reason the NaN half of #2182 is closed here | bounded |
| `gradient_discovery.rs::detect_gradient_candidates` | `std_dev > 1e-12` else `f32::MAX`, `consistency < MIN_GRADIENT_CONSISTENCY`, `effective_delta.abs() < 1e-8`, then `sort_by` on descending `total_cmp` over `estimated_improvement` (and again in `::gradient_candidates_to_coordinated`) | `mean_gradient.abs() * effective_delta.abs() * consistency.min(3.0) * 0.01` | an `+inf` variance makes `consistency` `0.0`, which is rejected — but the improvement product itself is never re-tested, and `+inf` heads the descending totalOrder | **unbounded — #2182.** `effective_delta` is `clamp(weight + raw_delta, ±10) - weight`, so a finite synapse weight of `1e38` and a `mean_gradient` of `3e37` multiply to `+inf` past the `is_finite` gate above |
| `sample_weighted.rs::compute_sample_weights` and `::stratify_samples` | `avg_err.is_finite()` per record, `total <= f32::EPSILON`, `easy_mean > f32::EPSILON`, `err <= median`, `select_nth_unstable_by(f32::total_cmp)` | the per-record mean error | every non-finite per-record error is laundered to `0.0` **before** anything is compared, so the median select gets a total order over finite keys and no NaN reaches the ratio; an overflowed `total` renormalises every weight to `e / inf = 0.0` rather than producing an infinity | bounded — the model guard of this section |
| `sample_weighted.rs::detect_high_error_neurons` | `weighted_mean < config.min_weighted_error`, then `sort_by` on descending `total_cmp` over `estimated_improvement` (and again in `::high_error_neurons_to_coordinated_candidates`) | `(weighted_mean * ratio.min(10.0) * 0.01).min(0.1)` | `f32::min` returns the non-NaN operand, so the two caps make the ranked value finite and at most `0.1` whatever the inputs were; `hard_mean / f32::EPSILON` overflowing to `+inf` is capped by `.min(10.0)` | bounded — the only one of the four ranked detectors that cannot be forced to the head of its own list |
| `output_competition.rs::co_activation` and `::detect_output_competition` | `r.activation > CO_ACTIVATION_THRESHOLD`, `count == 0` early return, then `sort_by` on descending `total_cmp` over `estimated_improvement` (and again in `::output_competition_to_coordinated_candidates`) | `sum_min / count` over the co-activated samples | a NaN activation loses the `>` and is excluded; `sum_min` is an `f32` accumulator and can overflow to `+inf`, which would head the descending totalOrder exactly as in #2182 | bounded — by unreachability only: the module has no production caller (#2185). Wiring it up must close the overflow at the same time |
| `activation_recommendation.rs::recommend_activation_function` | `suitability.iter().max_by(a.1.total_cmp(b.1))`, `improvement < MIN_IMPROVEMENT_THRESHOLD` | the suitability map, whose values are compile-time literals scaled by compile-time penalty factors | no input-derived float reaches either comparator, so NaN is not expressible here; the `max_by` was nevertheless order-dependent on a tie, which is #2184 (out of class, since fixed — the comparator now falls through to `a.0.cmp(b.0)`) | bounded |
| `activation_recommendation.rs::recommend_activation_function_for_role` | `family_scores.sort_by(descending total_cmp)` | the same compile-time score space, defaulted to `0.6` for unscored family members | same — constants only | bounded |
| `activation_recommendation.rs::analyse_input_distribution` / `::classify_distribution` / `::apply_gradient_flow_penalty` | `fold(f32::INFINITY, f32::min)` / `fold(f32::NEG_INFINITY, f32::max)`, `std_dev > 1e-6`, `range < BOUNDED_RANGE_THRESHOLD && range > 0.0`, `kurtosis < 2.5`, `(max - min)` then `.clamp(0.0, 1.0)` | recorded activations | `f32::min` / `f32::max` return the non-NaN operand, so the extrema are finite whenever any finite sample exists; `(max - min)` overflowing to `+inf` makes `negative_fraction` `0.0`, and the `max > min` guard keeps the divisor positive | bounded — the classification only selects which literal score map is used, and the one score `apply_gradient_flow_penalty` actually scales by an input-derived value is multiplied by `(1.0 - negative_fraction * 0.5).max(0.3)`, where `negative_fraction` is itself `.clamp(0.0, 1.0)`d — so the multiplier is confined to `[0.3, 1.0]` and cannot move the ranked value off the constant space. The bound is that pair of clamps, not the absence of an input-derived float |
| `activation_recommendation.rs::detect_output_range_requirements` | `(r.activation * 100.0) as i32`, then `\|v - 0.0\| < 0.1` / `\|v - 1.0\| < 0.1` | recorded activations | Rust's float→int `as` cast saturates and maps NaN to `0`, never UB, and the quantised set is only used to pick an enum variant | bounded |
| `activation_recommendation.rs::analyse_gradient_flow_risk` | `r.activation.abs() > 0.95`, `r.activation < 0.05 \|\| > 0.95`, `r.activation <= 0.0`, `r.activation.abs() >= 0.99` — four saturation counters, one per squash family | recorded activations | every one is a count-on-comparison, so a NaN simply is not counted; the returned risk is `count / records.len()`, and the `records.is_empty()` early return makes the divisor at least `1` | bounded — and the value is never ranked: the only callers anywhere are its own `#[cfg(test)]` test and `tests/recommendation/issue_431_activation_recommendation.rs`, so the risk score reaches neither `max_by` nor a sort. The suitability penalties the recommender actually applies are `apply_gradient_flow_penalty`'s, one row above |
| `mod.rs` | none | no float comparator or sort in the file | n/a | n/a |
<!-- section: recommendation batch_successful + epistatic -->
| `detection.rs::detect_individually_successful` | `sort_by` on descending `total_cmp` over `improvement`, then `truncate(MAX_INDIVIDUAL_CANDIDATES)` | the per-pair SSE-reduction ratio `1 - residual_sse / original_sse`, both sums accumulated in `f64` over the shared observation indices | no non-finite key can reach it: `detection.rs::evaluate_individual` returns `None` on `!improvement.is_finite()` **before** the candidate is pushed, so the totalOrder placement of a NaN above `+inf` is never exercised | bounded — the guard ordering the four ranked `recommendation core` detectors do not have. `original_sse < 1e-10` guards the divisor ahead of the division, and the emitted weight is a `calculate_optimal_outgoing_weight` return, which Issue #2043 made the single owner of the finitude and clamp rules |
| `detection.rs::evaluate_individual` | `original_sse < 1e-10`, then `improvement < MIN_INDIVIDUAL_IMPROVEMENT` (`1e-5`) | the same `f64` sums | a NaN `original_sse` loses the `<` and falls through to the division, but the quotient is then NaN and the explicit `is_finite` test rejects it before the `<` accept gate; a NaN improvement also loses the `<`, so **both** orderings reject it | bounded — finitude is tested between the divisor guard and the accept gate |
| `grouping.rs::batch_successful_to_coordinated_candidates` | `retain`-style `filter(combined_improvement > 0.0)` | `Σ improvement × BATCH_IMPROVEMENT_SCALE` over at most `MAX_BATCH_SIZE` (4) candidates | a NaN loses the `>` and the batch is dropped — the fail-closed direction — and it cannot arise anyway: every summand is finite and in `(1e-5, 1]` by the detector's gate, so the sum is at most `4.0` | bounded |
| `candidate_generation.rs::detect_epistatic_pairs` | `sort_by` on descending `total_cmp` over `combined_improvement` | `compute_combined_improvement_on_range`, scaled by `COORDINATED_ESTIMATION_WEIGHT_SCALE` | no non-finite key reaches it: the producer returns `0.0` unless `improvement.is_finite()`, and `evaluate_pair_for_epistasis` then requires `combined_improvement > 0.0`, which a NaN loses. `target_impact` is the compile-time `1.0` / `0.5` of `candidate_selection.rs::detect_epistatic_and_synergistic`, so the post-gate multiply cannot reintroduce an infinity | bounded — the `is_finite` test and the `> 0.0` gate together are a whitelist |
| `candidate_generation.rs::evaluate_pair_for_epistasis` | `complementarity < MIN_COMPLEMENTARITY_RATIO`, `combined_improvement > sum_of_individuals * target_impact`, `improvement_first > 0.0 && improvement_second > 0.0` | the firing-index Jaccard complement, and three `compute_combined_improvement_on_range` passes | `compute_complementarity` divides two `usize` counts after an `is_empty` and a `union_size == 0` guard, so its value is integer-derived and never NaN; the three improvement tests are accept-on-`>`, all of which a NaN loses | bounded — every accept is a `>` a NaN loses, and the two individual improvements are `>= 0.0` by the pre-screen |
| `candidate_generation.rs::has_cross_sample_harm` | `activation.abs() >= ACTIVATION_FIRING_THRESHOLD`, `error.abs() > 1e-10`, `(contribution * error) < 0.0`, then `avg_a_harm > 0.01` | `f64` products of the source weight and the sample activation | every test is a reject-on-comparison a NaN loses, so a NaN falls **through** the harm check rather than triggering it; both divisions are guarded by an `== 0` count test and the counters are whole numbers | bounded — the fall-through is harmless because the super-additivity and cross-validation gates below it are accept-on-`>` |
| `deduplication.rs::deduplicate_by_dominant_neuron` and `::deduplicate_synergistic_by_dominant_neuron` | `individual_improvement_a >= individual_improvement_b` for the group key, then four descending `total_cmp` sorts with `truncate(MAX_PAIRS_PER_DOMINANT_NEURON)` | `combined_improvement` and the two individual improvements of an already-emitted candidate | **NaN suppression is not reachable.** IEEE-754 totalOrder would put a positive NaN above `+inf`, so a NaN "dominant" candidate would head its group and consume the whole 3-slot cap — but no producer can hand it one: both callers gate on `combined_improvement > 0.0` and the `>= MAX_INDIVIDUAL_HARM_FOR_PAIRING` (`0.0`) pre-screen is fail-closed for a NaN dominance key | bounded — `total_cmp` is a total order, so the sort cannot panic either. The open defect is the **tie** order, #2191 |
| `pre_screening.rs::detect_synergistic_candidates` | `max_by` on `total_cmp` over `individual_improvement`, then `sort_by` on descending `total_cmp` over `combined_improvement` | the pre-screened individual improvements, and `evaluate_residual_reduction`'s `f64` ratio | the pre-screen has already dropped every NaN `individual_improvement`; the ranked `combined_improvement` passes a `> 0.0` test in `f64` before the narrowing cast, so it is finite and positive | bounded |
| `pre_screening.rs::evaluate_residual_reduction` | `activation_correlation > 0.9`, `sum_act_squared < 1e-10`, `original_error_sum < 1e-10`, `residual_error_sum > 1e-10`, then `residual_reduction >= 0.1 && synergy_ratio >= 1.1 && combined_improvement > 0.0` | `detection/stats.rs::pearson_correlation_samples` and four `f64` sums of squares | a NaN correlation loses the `>` and the pair is kept, but the correlation cannot be NaN: `pearson_correlation_samples` accumulates in **`f64`** throughout, so the `f32`-overflow path that makes `pearson_correlation` NaN in #2181 has no counterpart here — reaching `f64` overflow would need ~10²³⁰ samples. The three accept tests are all `>=` / `>` a NaN loses, so an unusable candidate is dropped rather than emitted | bounded — the `f64` accumulators are the difference between this file and #2181 / #2182. `synergy_ratio` is deliberately set to `f64::INFINITY` for true synergy, but it is reported, never ranked |
| `scoring.rs` | `correlation >= REDUNDANCY_CORRELATION_THRESHOLD`, `avg_combined > SATURATION_RISK_THRESHOLD`, `saturation_fraction > 0.3`, `(avg_combined / SATURATION_RISK_THRESHOLD).min(1.0)` | the same `f64` Pearson, and an **`f32`** `total_combined` accumulator over the sample loop | a NaN correlation loses the `>=`, so the redundancy filter **keeps** the pair — fail-open, but unreachable for the `f64` Pearson. `total_combined` is `f32` and can overflow to `+inf` or, via `inf + -inf`, to NaN; `avg_combined` then reads `inf` in the emitted reason string, while `severity` stays bounded because `f32::min` returns the non-NaN operand. Both divisors are guarded by the `n < 10` early return | bounded — the two `filter_interfering_*` functions production uses compare only the `f64` Pearson; the `f32` accumulator lives in `check_saturation_risk`, which has no production caller (#2192) |
<!-- section: shared -->
| — | none | `src/analysis/shared/` compares no floats | n/a | n/a |

## Outcome

### synapse pipeline (Issue #2104)

**One finding filed: #2161.** All 14 files were read in full for every defect
class above. What was actually traced:

**Cancellation — the finding.** The pipeline's cancellation contract is
`utils/deadline.rs::deadline_passed`, which also reports the global flag set by
`cancellation::is_cancelled` (Issue #1047). It is consulted per focus target in
`orchestration.rs::analyze_synapses_with_cache_impl` and per source in
`target_analysis/statistics.rs::filter_and_load_sources`, which `break`s out of
record loading. Between those two points sits
`candidate_generation.rs::group_sources_by_locality`, whose grouping loop is
`n(n-1)/2` calls to `candidate_generation.rs::compute_obs_index_overlap` when no
two sources share 80% of their observation indices — and it consults neither the
deadline nor the cancellation flag. `filter_and_load_sources` does not return
early on its `break`, so a pass whose deadline expires mid-load still enters the
quadratic scan with everything it had collected. The only cap on `n`,
`utils/deadline.rs::apply_source_budget`
(`NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET`, Issue #1542), is unset by default.
Filed as #2161 (`severity:medium`, `confidence:high`). Every other loop in these
files is either bounded by a compile-time array (the activation spec scans in
`gpu_evaluation.rs`, `activation_evaluation.rs` and
`activation_subset_evaluation.rs` iterate `spec.orientations × spec.scales`), by
the 50-sample cap in `preparation.rs::compute_constant_source_threshold_from_cache`,
or sits behind the per-target `deadline_passed` check.

**Integer class — the #1906 hypothesis is refuted.**
`holdout_validation.rs::split_samples_holdout` cannot reach
`samples.len() - validate_count` with an empty `samples`: the function returns
`None` while `samples.len() < HOLDOUT_MIN_SAMPLE_COUNT` (20,
`constants/sample_thresholds.rs`). For every surviving `n >= 20`,
`validate_count = round(0.3n)` is at most `0.3n + 0.5`, which is strictly less
than `n`, so the subtraction is positive in both debug and release. The
`max(1.0)` floor only ever raises a value that was already below one, which
cannot happen above the threshold. The sibling division
`seed % (total as u64)` in `holdout_validation.rs::is_validation_sample` has the
same guard, so it can never divide by zero.

**Capacity class.** All seven sites are in the table above and all seven are
bounded. The two that derive from caller data —
`preparation.rs::build_creature_lookups` and
`candidate_generation.rs::build_ordered_neurons` — are bounded by
`MAX_CREATURE_INPUT_NEURONS` (1,000,000), enforced by
`ffi_types/creature_bounds.rs::validate_creature_input_bounds` at every entry
point that accepts a `CreatureJson` (Issues #1867, #2078). Neither sum can
overflow `usize`: the other addends count live `Vec`s, whose lengths are bounded
by the bytes already allocated for them. `gpu_evaluation.rs`'s
`vec![None; plan.specs.len()]` is sized from a compile-time `ACTIVATION_SPECS`
subset, not from input.

**Float class.** The nine comparison sites are in the table above. The load
bearing invariant is `scoring/improvement.rs::finalise_improvement`, which wraps
every result in `scoring/improvement.rs::select_finite` — so `compute_relu_-`,
`compute_activation_-` and `compute_synapse_improvement_and_count` cannot return
a NaN or an infinity, and no gain computed in these files is ever non-finite.
Four guards nonetheless fail open on a NaN — `total_baseline_error_sq <= EPSILON`
in `relu_evaluation.rs` and `filtering.rs`, and `absolute_improvement < 0.001` in
both `activation_evaluation.rs` and `gpu_evaluation.rs` — and one accept path
would skip the comparison entirely (`current_best.is_none() ||` in
`gpu_evaluation.rs::evaluate_all_activation_specs_batched`, the only accept in
these files that does not end in a `>` a NaN loses). None of them is reachable
with a NaN today; they are recorded so a later change to the improvement helpers
is understood to re-open five sites at once, not one.

**Division.** Every denominator in these files is guarded:
`candidate_generation.rs::compute_obs_index_overlap` returns early when either
set is empty, `preparation.rs::compute_constant_source_threshold_from_cache`
divides only when `std_dev_count > 0`, and the two modulo operations in
`holdout_validation.rs::is_validation_sample` sit behind the 20-sample floor.

**Shared-state races.** `orchestration.rs` is the only file here that shares
mutable state. `metadata.rs::AtomicMetadata` is written from the rayon workers
with `Relaxed` stores and `fetch_min` / `fetch_max` — both read-modify-write, so
no update can be lost — and read only in `results.rs::finalise_synapse_results`,
after `par_iter().collect()` has joined. The same join orders the
`TimingCollector`, the `AtomicBool` timeout and saturation flags and the
`completed_count` counter. `orchestration.rs::apply_target_cooldown` takes the
global tracker lock, recovers a poisoned lock with `into_inner` rather than
panicking, and holds it only for the filter.

**Hostile environment values.** None of these files parses an environment
variable. They read `utils::verbose_enabled`, `utils::gpu_timing_enabled`,
`config::max_sources_per_target` and the calibration thresholds, all of which
are total parsers owned by `src/config` (swept separately).

**Deliberately out of scope for this sub-issue:** the 44 rows outside the
`synapse pipeline` section, which belong to the five remaining chunk 8b audit
sub-issues. The two descending `total_cmp` sorts in `post_processing.rs` that
feed `filtering.rs::truncate_combined_synapse_candidate_sets` are #2105's rows.

### synapse post-processing (Issue #2105)

**Four findings filed: #2167, #2168, #2169, #2170.** All four files
(2,614 lines) were read in full for every defect class above. What was actually
traced:

**Float class — the ranking finding (#2167).** `post_processing.rs::apply_post_processing`
holds **three** descending `total_cmp` sorts, not the two the `synapse pipeline`
section above recorded — one each for the helpful, harmful and coordinated
structural buckets. IEEE-754 totalOrder places a positive NaN above `+inf` and
every finite value below both, so a descending sort puts a non-finite gain at
the head of the list the host then adopts. Each bucket is filtered once ahead of
its sort, and all three filters are asymmetric in exactly the wrong direction:
`post_processing.rs::apply_min_expected_gain_floor_for_synapses` runs over the
helpful **and** the harmful bucket keeping `gain >= floor`, and the coordinated
pass keeps `gain > 0.0` (Issues #1110, #1128) — a NaN fails both comparisons and
is dropped, while `+inf` passes both and survives. No bucket is left unfiltered,
which is why the finding is about what the filters keep rather than about a
missing filter. This call
site never runs `candidate_aggregation.rs::reject_non_finite_gains`, the
Issue #1367 gate; only `candidate_aggregation.rs::merge_coordinated_structural_replacements`
does. `+inf` is reachable rather than theoretical:
`post_processing.rs::compute_neuron_error_sq_map` filters its per-sample errors
on `e.is_finite()` but then inserts `e * e`, which overflows to `+inf` for any
error above `f32::MAX.sqrt()` (≈ 1.84e19); `post_processing.rs::scale_by_error_fraction`
then fails its `total_error_sq <= EPSILON` guard on that infinity, divides
`inf / inf` to NaN, and `clamp` propagates NaN unchanged. Filed as #2167
(`severity:medium`, `confidence:high`).

**Cancellation — the quadratic-scan finding (#2169).** Neither detector in
`structural_patterns.rs` consults `utils/deadline.rs::deadline_passed` or the
global flag behind `cancellation::is_cancelled` (Issue #1047), and both run
before `apply_post_processing`. `structural_patterns.rs::detect_noisy_vs_trusted`
walks every unordered pair of incoming inputs, and its per-pair body is not
O(1) — it calls `scoring/improvement.rs::compute_synapse_improvement_and_count`,
which walks the record set, so the work between cancellation points grows as
n²·records. `structural_patterns.rs::detect_collapsible_hidden_neurons` is
O(neurons × records) on the same terms. The only cap on either is the size of
the untrusted creature: `ffi_types/creature_bounds.rs::validate_creature_input_bounds`
bounds input and output width (1,000,000 each, Issues #1867, #2020, #2078) and
says nothing about hidden-neuron or synapse count. Filed as #2169
(`severity:medium`, `confidence:high`) — the same shape as #2161.

**Panic class — the UTF-8 slice (#2168).**
`post_processing.rs::apply_impact_to_helpful` byte-slices a neuron UUID at
index 12 for a verbose log. The `.min(len)` guard covers only a short string;
it does not make index 12 a char boundary, so any UUID whose twelfth byte falls
inside a multi-byte sequence panics with "byte index 12 is not a char
boundary". The panic unwinds out of a rayon worker, and an unwind past an
`extern "C"` frame is an abort rather than a catchable error. The trigger needs
both `utils::verbose_enabled()` and a hidden target neuron. Filed as #2168
(`severity:low`, `confidence:high`).

**Integer class — the untrusted tracker (#2170).**
`add_synapse_gating.rs::should_skip_add_synapse_by_outcome` reads
`module_weights.rs::ModuleStats::success_rate`, whose `(self.attempts -
self.successes)` is a `u32 - u32` **before** the `as f64` cast.
`ModuleStats::record` maintains `successes <= attempts` and
`::record_soft_failures` rejects a non-positive weight, but the derived
`Deserialize` bypasses both: `ffi_types/requests.rs` carries
`module_outcome_tracker: Option<ModuleOutcomeTracker>` on both
`AnalyzeParallelInput` and `AnalyzeAllInput`,
`ffi_internal/analysis.rs::build_analyze_all_input_from_parallel` passes it
through unvalidated, and `analysis/orchestration.rs::analyze_all` hands it to
`add_synapse_gating.rs::gate_add_synapse_candidates`. `Cargo.toml` sets no
`overflow-checks` key in any profile, so the language defaults apply — a
debug or test build panics on the underflow and a release build wraps, giving
a rate near zero that fires the gate and silently drops every add-synapse
candidate. The same field corrupts the divisor: a deserialised
`soft_failures` of `-12.0` makes `beta` negative, `f64::MAX` forces the gate
closed, and a `NaN` makes `rate < success_threshold` false so the gate fails
open. Filed as #2170 (`severity:high`, `confidence:high`) — the highest
severity in this chunk.

**Capacity class.** Five production sites, all in the table above and all
bounded. The four in `structural_patterns.rs` are sized from live collections
the loader has already allocated — `records.len()` on a materialised slice and
`map.len()` on a `HashMap` built earlier in the same pass — so none is a
caller-supplied count. `adaptive_proposal.rs::generate_gaussian_candidates`
answers the issue's question about the origin of its `count`: it is bound from
`constants/candidate_scoring.rs::ADAPTIVE_PROPOSAL_CANDIDATE_COUNT` (12), a
compile-time constant no caller can override. The three vectors sized
`n_samples as usize` live in `structural_patterns.rs`'s `#[cfg(test)]`
`build_collapse_input` fixture builder, are driven by compile-time literals,
and are not compiled into the shipped `cdylib` — recorded as `n/a — test-only`
rather than dropped silently.

**Division.** Every production denominator in these four files is guarded:
`structural_patterns.rs::detect_noisy_vs_trusted::activation_mean_and_variance`
divides `sum / n` and `sum_sq / n` only after an `n <= 0.0` early return, and
the same detector's variance ratio divides by `trusted.var.max(EPSILON)`, which
is positive for every input including a NaN variance; the collapse pass returns
early on
`baseline_sq <= EPSILON`, `add_synapse_gating.rs::should_skip_add_synapse_by_density`
returns early when `total_neurons == 0`, and `adaptive_proposal.rs`'s
acceptance rate divides only when `total > 0`. The one unguarded divisor in
the chunk is `ModuleStats::success_rate`'s, and it is #2170's.

**NaN threads that close by construction — recorded so a later change re-opens
them knowingly.** `structural_patterns.rs`'s filters are all
skip-on-comparison, and they point both ways: the pairing pass skips when
`(a.weight - b.weight).abs() > WEIGHT_EPS` or `(a.mean - b.mean).abs() >
MEAN_EPS` — a `>` on the pairwise **difference**, not a `<` on either magnitude
— while `ratio < MIN_VAR_RATIO`, `improvement <= 0.0` and
`weight.abs() < bypass_floor` skip on the other direction. A NaN fails the
comparison whichever way it points, so in every one of these the candidate
**falls through** rather than being skipped.
Two invariants stop a NaN ever arriving: `scoring/improvement.rs::finalise_improvement`
divides only when `effective_baseline > EPSILON` and passes the quotient
through `scoring/improvement.rs::select_finite`, and
`scoring/weights/calculation.rs::compute_outgoing_weight` returns `None` on a
non-finite raw weight. Two laundering steps are worth naming for the same
reason: `var.max(0.0)` turns a NaN variance into `0.0` silently (`f32::max`
returns the non-NaN operand), and the `n += 1.0` `f32` sample counter stops
incrementing past 2²⁴. Both `add_synapse_gating.rs` gates fail **open** on a
NaN, which is why #2170's divisor corruption matters beyond the underflow.

**Adaptive proposal — clean.** `adaptive_proposal.rs`'s Box-Muller transform
floors its uniform at `1e-15` before `ln`, so `ln(0)` is unreachable;
`deterministic_hash` and `mix_hash` use `wrapping_mul` deliberately as a mixing
function, not as unchecked arithmetic; `record_batch`'s `u32` counters
accumulate once per real candidate batch, so reaching `u32::MAX` would require
more batches than a run can produce; and `sigma_for` divides only behind the
`ADAPTIVE_PROPOSAL_MIN_HISTORY` floor of 15. `generate_fixed_grid` applies no
finiteness guard of its own but is fed by `compute_outgoing_weight`, which has
one.

**Deliberately out of scope for this sub-issue:** the 44 rows outside the
`synapse post-processing` and `synapse pipeline` sections, which belong to the
four remaining chunk 8b audit sub-issues. Fixing any of #2167–#2170 is out of
scope here by the same rule that made #2161 a filing rather than a patch — each
fix ships its own `tests/issue_<n>_*.rs` regression test with its own PR.

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

### synapse scoring + target_analysis (Issue #2106)

**Negative result — no finding filed.** All 10 files were read in full for every
defect class probed. What was actually traced:

**Cancellation — all five uncovered loops are bounded.** The main helpful-path
loop in `evaluation.rs::collect_and_process_helpful_results` (L84 onward) and
the harmful-path loop in `evaluation.rs::process_harmful_batch_from_prepared`
(L688 onward) are both uncovered by `deadline_passed`. Four other loops —
`statistics.rs::filter_and_load_sources` (L49–118),
`statistics.rs::prepare_harmful_samples` (L292–314),
`statistics.rs::build_helpful_work_items` (L188–217), and
`statistics.rs::build_existing_edge_work` (L250–264) — are similarly uncovered
in their source form, though the main loop in `filter_and_load_sources` does
break early on deadline. Crucially, all five materialise their work from live
slices or compile-time constants, so none can loop indefinitely. The uncovered
loops in the work builders are rayon `par_iter()` over finite-length input; the
two main uncovered loops iterate over pre-built `Vec`s materialised at the start
of `analyse_single_target`. No unbounded loop was found.

**Integer class.** `statistics.rs::split_samples_holdout` (L125) subtracts
`results.len() - row_idx` guarded by the loop condition `row_idx < results.len()`,
confirming the subtraction cannot wrap. The two divisions
(`neuron_error_improvement` at L696–706) and (`seed % total`) operate on u32 or
u64 counts bounded by live data, with denominators guarded `> 0`.

**Capacity class.** The four sites in the capacity table above are all bounded.
The first two (`filter_and_load_sources` and `prepare_harmful_samples`)
materialise data from live slices; the third (`process_harmful_batch_from_prepared`)
iterates slices passed in by the caller; the test fixture uses compile-time
constants. No unbounded allocation was found.

**Float class.** The two float-comparison rows added to the table above cover the
main arithmetic path: `process_harmful_batch_from_prepared`'s harmful-path division
operands are u32-cast-to-f32 and thus never NaN; `compute_synapse_improvement_and_count`'s
helpful-path guard returns all-finite before any dispatch. Existing test
`scoring/tests.rs::test_synapse_no_target_branchless_handles_non_finite` passes,
directly verifying NaN-safety on the helpful path.

**Division.** Every division in these files is guarded:
`process_harmful_batch_from_prepared` at L696–706 guards `total_count > 0` before dividing;
divisors in capacity calculations are checked non-zero.

**Shared-state races.** No mutable shared state beyond rayon's own synchronisation
is present in these files. Rayon's `par_iter().collect()` and `.zip()` ordering
is sufficient for all work here.

**Hostile environment values.** None of these files parses an environment
variable. All input derives from FFI-validated creature data, SQL-filtered
source lists, or Parquet records.

**Deliberately out of scope for this sub-issue:** the 44 rows in the
`synapse pipeline` section above, which belong to Issue #2104, and the 54 rows
above that belong to the remaining chunk 8b audit sub-issues.

### scoring (Issue #2107)

**Negative result — no finding filed.** All 10 files were read in full for
every defect class probed. What was actually traced:

**Capacity class — `fold_count` is a constant, not an input.** The issue body
asked whether `cross_validation.rs::compute_cross_validation_score`'s
`Vec::with_capacity(config.fold_count)` is caller-controllable over FFI. It is
not. `CrossValidationConfig` is constructed exactly once in production —
`CrossValidationConfig::default()` in
`neuron/evaluation.rs::apply_cross_validation_penalty` — which sets the
compile-time `fold_count: 5`, and no field of any FFI request type
deserialises into the struct. That absent constructor is the whole bound, and
the function's own preconditions must not be mistaken for a second one: the
`fold_count < 2` early return does hold, but the
`samples.len() / fold_count < min_samples_per_fold` return is never taken
against a `min_samples_per_fold` of `0`, so an in-crate caller that set both
`fold_count: usize::MAX` and `min_samples_per_fold: 0` would reach the
reservation. Any future caller that builds the config itself — rather than
taking `::default()` — therefore needs its own bound on `fold_count`. The only
other sized allocation in these files is
`error_distribution.rs::detect_modes_histogram`'s `vec![Vec::new(); NUM_BINS]`,
sized by a function-local `const usize = 20` and unreachable in production in
any case, since its only caller `detect_error_modes` is itself dead (below).

**Capacity class — the untrusted `failureCache` maps.**
`calibration_correction.rs::from_failure_cache` builds four `HashMap`s keyed by
the `change_type`, `target_squash` and `variant_key` **strings** of the
caller-supplied failure cache, so their cardinality is input-derived. It is not
the #2078 shape: there is no size hint, no multiplier and no per-entry fan-out
— each map holds at most one entry per distinct key of a cache serde has
already materialised from the caller's JSON, so the memory is 1:1 with a
payload the caller was already holding. The same reasoning covers
`disconnect_penalties`, whose key is a triple of those strings.

**Division and cast class.** Every division in these files is guarded ahead of
the arithmetic, and the guards are fail-closed rather than fail-open:

- `sample_creature_disconnect.rs::detect_disconnect` rejects
  `total_count == 0` and the corrupt-counter case `improved_count >
  total_count`, and rejects a non-finite `actual_error_reduction`, before it
  computes `improved_count as f32 / total_count as f32`. This is the
  `improved / total` arithmetic the issue body asked about, and it is the
  reference guard for the chunk.
- `calibration_correction.rs::from_failure_cache` skips
  `expected_error_reduction == 0.0` — which also catches `-0.0`, since the two
  compare equal — and then discards any ratio that is not finite.
- `error_distribution.rs::detect_modes_histogram`'s
  `((error - min) / bin_width).floor() as usize` is safe on all three counts
  the issue body raised: `range < 1e-6` returns before `bin_width` can be zero,
  Rust's float→int `as` cast saturates (a NaN index becomes `0`, never UB), and
  `.min(NUM_BINS - 1)` caps whatever the cast produced, so the `bins[bin_idx]`
  index cannot panic.
- `error_distribution.rs::compute_percentiles`' `idx.floor()` / `idx.ceil()`
  pair is bounded by `if lower == upper || upper >= n { sorted[lower.min(n - 1)] }`,
  so neither index can leave the slice; `n >= 1` is guaranteed by the
  `values.is_empty()` early return.
- `confidence.rs::t_critical_95`'s `df as u32` narrowing is the one cast in
  these files that is lossy in principle: `df` is `samples.len() - 1`, so on a
  64-bit host a sample count of exactly `2³² + 1` would truncate to `0` and
  return the *widest* critical value (12.706) instead of the normal
  approximation. It is unreachable — 2³² `HelpfulSample`s is hundreds of
  gigabytes — and the direction of the error is conservative (a wider interval,
  never a narrower one). Recorded rather than filed.

**Float class — the `total_cmp` sort.**
`error_distribution.rs::compute_percentiles` sorts the error vector with
`f32::total_cmp`, under which a negative NaN sorts **first** and a positive NaN
**last**; a NaN in that vector would therefore surface as `p10` or `p90` and
propagate into `iqr`, and the `skewness` / `kurtosis` moments would be NaN
alongside it. The input cannot contain one: `ErrorDistribution::from_samples`
filters `is_finite`, and both production callers of the `pub fn from_errors`
entry point — `synapse/post_processing.rs::build_metadata` and
`neuron/post_processing.rs::build_neuron_results` — collect their error column
through `.filter(|e| e.is_finite())` at the point of collection. The guarantee
lives at the callers, not in the function, so it is recorded in the float table
above where a future caller will meet it.

**Float class — the `weights/` clamps.** The issue body flagged that a NaN
weight passing a clamp silently would be a finding, and both helpers in
`weights/adjustment.rs` have exactly that shape: `f32::clamp` propagates NaN,
and both `clamp_weight_update_delta`'s `delta_weight.abs() <= EPSILON` gate and
`coordinated_structural_activation_delta`'s `noisy_weight.abs() <= EPSILON`
gate are lost by a NaN, so each would return `Some(non-finite)` from a contract
whose `None` means "unusable". Neither is reachable. Synapse weights are the
only untrusted input either helper takes, and
`ffi_types/mod.rs::deserialise_synapse_weight` rejects a non-finite weight at
deserialisation (Issue #2132) — the `1e39`-saturates-to-`inf` path that
Issue #2133 closed for neuron biases is closed for weights too. The other
operand of
`clamp_weight_update_delta` is a `calculate_optimal_outgoing_weight` return,
and `weights/calculation.rs::compute_outgoing_weight` tests `is_finite`
**between** its divisor guard and its clamp, which is the ordering that makes
the difference. `structural_patterns.rs::detect_noisy_vs_trusted`, the sole
caller of the coordinated helper, additionally drops the sample on
`!activation.is_finite()`, so that hazard is closed twice.

**Integer overflow class.** `cross_validation.rs::FoldResult::improvement_ratio`
adds two `u32` counters before dividing; both are incremented once per sample
of one fold, so their sum is `samples_evaluated` and cannot wrap short of a
4-billion-sample fold. `confidence.rs`'s `count` accumulators are the same
shape. No index arithmetic in these files subtracts without a preceding
comparison.

**Shared-state races.** None of these 10 files holds mutable shared state: every
public function is a pure computation over a borrowed slice or a `&self` read
of an owned `HashMap`. `CalibrationCorrection` is built once per call and
thereafter read-only. Nothing here runs inside a rayon parallel section that
writes a shared collection.

**Panics on hostile environment values.** The only environment readers are
`error_distribution.rs::outlier_analysis_enabled` and
`error_distribution.rs::outlier_percentile_from_env`, which delegate to
`config::outlier_analysis` and `config::outlier_percentile`. Neither can panic:
the percentile is a non-panicking `parse_env::<u8>` with a range filter and a
default (the value is `docs/CONFIGURATION.md`'s to state, not this record's),
and the flag is a boolean presence test.

**Out-of-class observation — a dead operator lever (AGENTS.md § Dead Levers).**
Those two readers have **no callers**, and neither do the four outlier helpers
they exist to configure — `count_outliers`, `filter_outliers`,
`is_likely_bimodal` and `has_significant_outliers` — nor `detect_error_modes`.
`CandidateNeuronJson::outlier_reduction_info` is assigned `None` at every one of
its construction sites, so `OutlierReductionInfo` is never built. That makes
`NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` and
`NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` — both documented in
`docs/CONFIGURATION.md` and in the `src/config/mod.rs` table — levers an
operator can set, and tune during an incident, that change nothing. This is not
one of the five defect classes above, so it is **not** filed as a security
finding; it is filed as #2177 for the ordinary dead-lever cleanup.

**Deliberately out of scope for this sub-issue:** the 48 rows belonging to the
other chunk 8b audit sub-issues.

### recommendation core (Issue #2108)

**Three findings filed — #2181, #2182, #2183.** All eight files were read in
full for every defect class probed. The primary lens was ranking integrity, so
the conclusion is recorded per detector first.

**Ranking integrity — can a crafted input reach rank 1 unchecked?** Every
detector in this section ends in a descending sort that the host reads
head-first, so the question is asked of each one and answered with the path
that gets there.

| Detector | Rank 1 reachable? | Path |
| --- | --- | --- |
| `output_bias_drift.rs::detect_output_bias_drift` | **yes — #2182** | 30 records whose first error is a finite `1e38` overflow the `f32` `sum_error`; `mean_error` is `+inf`, the `mean_error.abs() < MIN_MEAN_ERROR_MAGNITUDE` noise gate is fail-open for it, and `+inf` heads the descending `total_cmp`. Measured: rank 0 with `estimated_improvement = inf` against an honest competitor at `0.045` |
| `multi_hop.rs::detect_multi_hop_candidates` | **yes, by two paths — #2182** | (a) 30 observations shared with the source give a healthy finite correlation; 30 further target-only observations at `1e38` drive `compute_mean_abs_error` to `+inf`, which the `estimated_improvement <= 0.0` gate passes. Measured: rank 0 with `estimated_improvement = inf`. (b) `find_three_hop_extensions` filters on `source_intermediate_corr.abs() < CORRELATION_THRESHOLD` — the **fail-open** direction — so a NaN correlation between a source and an intermediate is kept, `combined_corr` is NaN, and the NaN improvement enters the same descending `total_cmp`, where totalOrder puts a positive NaN **above** `+inf`. That path outranks (a) |
| `gradient_discovery.rs::detect_gradient_candidates` | **yes — #2182** | a finite synapse weight of `1e38` makes `\|effective_delta\|` ≈ `1e38`, which multiplies with a `3e37` `mean_gradient` to `+inf` **after** the `!mean_gradient.is_finite()` gate. Measured: rank 0 with `estimated_improvement = inf` |
| `fan_in.rs::detect_fan_in_candidates` | **yes, by two independent paths — #2182 for rank 1, #2181 for suppression** | (a) **rank 1 — #2182.** Sixty records whose activations sit at `3.3e8 ± 1e3` and whose target error is `3.03e10 ×` that activation overflow the **`f32`** `original_sse` in `compute_least_squares_improvement` while the residual stays ~0, so `(inf - finite).max(0.0)` is `+inf`; `compute_two_input_regression` reaches the same value by narrowing an `f64` `improvement as f32`. All four `evaluate_fan_in_pair` gates are fail-open for it — `inf <= 0.0` is false and `inf < inf * 1.05` is false — so the pair is emitted and heads the descending sort. Measured: rank 0 with `estimated_improvement = inf`, **on honest finite correlations of `1.00` and `0.58`** — this path does not use #2181's NaN filter at all. The spread must stay narrow: `pearson_correlation` is mean-centred, so it survives the error overflow, but a wider spread overflows its `var_x * var_y` and the `denom < f32::EPSILON` guard returns `0.0`, which the `< 0.3` filter then drops. (b) **suppression — #2181.** 20 inputs whose activations swing `±2e30` each score `NaN`, are kept by the fail-open `corr.abs() < 0.3` filter, and compare `Equal` to every finite key under `partial_cmp(…).unwrap_or(Equal)`. Measured: 25 genuine candidates with the poisoned neurons listed after the honest ones, **0** with them listed first — the order is unspecified and the caller picks it |
| `sample_weighted.rs::detect_high_error_neurons` | no | every per-record error is laundered through `is_finite` to `0.0` before anything is compared, an overflowed weight total renormalises to zero, and `.min(10.0)` / `.min(0.1)` cap the ranked value at `0.1` |
| `activation_recommendation.rs::recommend_activation_function` | no | the score space is compile-time literals. One penalty is input-derived — `apply_gradient_flow_penalty` scales `RELU` and `RELU6` by `(1.0 - negative_fraction * 0.5).max(0.3)` — but `negative_fraction` is `.clamp(0.0, 1.0)`d at source and the `.max(0.3)` floors the product, so the multiplier lives in `[0.3, 1.0]`: a crafted record can lower a score, never raise one and never make one non-finite |
| `output_competition.rs::detect_output_competition` | not reachable | the `sum_min` accumulator has the same `+inf` shape as #2182, but the module has no production caller (#2185) |

**The `fan_in.rs` comparators — the issue body's primary question, answered
yes.** Both sorts use `partial_cmp(…).unwrap_or(std::cmp::Ordering::Equal)`,
the only non-total comparators in the chunk, and reachability was **not**
disproved for the first of them:

- `detection/stats.rs::pearson_correlation` has no finitude gate, and its
  `denom < f32::EPSILON` guard is fail-open — a NaN loses the `<`, and
  `f32::clamp` then propagates the NaN out of the function. An activation
  swing near `±2e30` overflows the `f32` covariance accumulator, so both `cov`
  and `var_x * var_y` are `+inf` and `inf / inf` is NaN. **Every input value
  is finite**, so the FFI gates of Issues #2134 / #2135
  (`ffi_types/mod.rs::deserialise_activation`,
  `ffi_types/mod.rs::deserialise_finite_errors`) do not apply: they reject
  `Infinity` and `NaN` on the wire, not a magnitude that overflows later.
- `detect_fan_in_candidates` filters with `corr.abs() < THRESHOLD`, a
  skip-on-comparison test a NaN loses, so the NaN correlation is **kept**.
  `multi_hop.rs` asks the same question of
  `detection/stats.rs::pearson_correlation_hashmaps` — which is the same
  arithmetic over the shared observation indices, and equally capable of
  returning NaN — and answers it **both ways in the same file**:
  `detect_multi_hop_candidates` spells the filter `corr.abs() >= THRESHOLD`,
  which a NaN loses the other way and is therefore dropped, while
  `find_three_hop_extensions` spells it
  `source_intermediate_corr.abs() < THRESHOLD`, which is `fan_in.rs`'s
  fail-open direction exactly. A NaN source-to-intermediate correlation
  survives that second filter, makes `combined_corr` NaN, wins the
  `estimated_improvement <= 0.0` gate, and enters the same descending
  `total_cmp` as every two-hop candidate — where totalOrder ranks a positive
  NaN **above** `+inf`. That is the second half of #2182 for this file, and it
  is a sharper path than the `+inf` one: three detectors differ from each other
  only in the direction of one comparison, and two of the three get it wrong.
- The comparator then reports `Equal` for every NaN-vs-finite pair while the
  finite pairs order among themselves. That is not transitive, so it is not a
  total order; `slice::sort_by` documents that as "may panic", and where it
  does not panic the order is unspecified — which is exactly what the
  `truncate(MAX_INPUTS_PER_TARGET)` on the next line acts on.

The second comparator, on `estimated_improvement`, cannot see a NaN today —
but only because `fan_in.rs::compute_two_input_regression` and
`fan_in.rs::compute_least_squares_improvement` both end in `.max(0.0)`, which
returns the non-NaN operand. That is a laundering accident, not a guard, and
`fan_in.rs::evaluate_fan_in_pair`'s four gates are every one fail-open for
NaN; the pair survives them all and is dropped only by
`best_individual <= 0.0`, which the same laundering guarantees. #2181 asks for
`total_cmp` at both sites.

**The `+inf` half of that comparator is open, and it is #2182 — this sweep's
first pass got this wrong.** `.max(0.0)` launders a NaN and passes an infinity
straight through, and the first pass stopped at the NaN question instead of
applying to this file the `f32`-accumulator reasoning it applied to the four
other detectors. Two sites manufacture the infinity from finite records:

- `fan_in.rs::compute_least_squares_improvement` is **`f32` throughout** — the
  first pass wrongly attributed its `.max(0.0)` to `f64::max`. Its
  `original_sse` sums `e * e` over the shared samples, so sixty errors near
  `1e19` overflow it while `residual_sse` stays small, and
  `(inf - finite).max(0.0)` is `+inf`.
- `fan_in.rs::compute_two_input_regression` accumulates in `f64` and returns
  `Some((wa as f32, wb as f32, improvement as f32))`. Rust's float→float `as`
  cast saturates, so an `f64` improvement above `f32::MAX` narrows to `+inf`.

`fan_in.rs::evaluate_fan_in_pair`'s four gates then pass it — `inf <= 0.0` is
false, `inf < inf * 1.05` is false, `inf * 0.01 <= 0.0` is false — and the
descending sort puts it first. **Measured** against this tree with every
recorded value finite: activations at `3.3e8 ± 1e3`, target errors at
`3.03e10 ×` the activation, rank 0 at `estimated_improvement = inf`, on
correlations of `1.00` and `0.58`. That last number is the point — this path
is admitted on an **honest** correlation, so it is independent of #2181's
fail-open NaN filter and closing #2181 alone would not close it. It is,
however, narrow: `detection/stats.rs::pearson_correlation` is mean-centred, so
it survives the error overflow, but a wider activation spread overflows its
`var_x * var_y` instead, the `denom < f32::EPSILON` guard returns `0.0`, and
the `corr.abs() < 0.3` filter drops the input before the improvement is
computed. The trigger has to thread between the two overflows, and it does.
`tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::a_finite_record_set_still_ranks_a_fan_in_candidate_at_infinity`
pins it.

**The converters re-rank the same field, and none of them re-tests it.** Each
detector has a `*_to_coordinated_candidates` counterpart that sorts
`expected_creature_score_gain` with the same descending `total_cmp` on the way
out — `output_bias_drift.rs::output_bias_drift_to_coordinated_candidates` is
the one that also copies the non-finite `recommended_bias_delta` into a
`SetBias` payload, so the value the host is asked to apply is not a number it
can act on. Fixing only the detector would leave the converter ranking the
same `+inf`; #2182 names both halves.

**Division.** Every divisor in these files is a count, and every count is
guarded before it is used: `output_bias_drift.rs::summarise_positive_support`
returns `None` at `count == 0`, `output_competition.rs::co_activation` returns
`None` at `count == 0`, `gradient_discovery.rs::compute_synapse_gradient`
returns `None` below `MIN_SAMPLES_FOR_GRADIENT`,
`sample_weighted.rs::compute_sample_weights` returns early on an empty record
slice, and `multi_hop.rs::compute_mean_abs_error` returns `0.0` on an empty
map. `sample_weighted.rs::stratify_samples`' `hard_mean / f32::EPSILON` branch
divides by a compile-time constant, never by zero, and its result is capped by
`.min(10.0)`. The hazard in this section is therefore not a zero divisor — it
is the **numerator**, an `f32` accumulator that overflows to `+inf` before the
division ever happens, which is #2182.

**Capacity.** Four sized allocations — the four sites the issue body named, and
the only `with_capacity` / `reserve` / `vec![v; n]` sites in these eight files
— all bounded by a live collection's length and none by a caller-supplied
count. The capacity table carries a fifth row for
`sample_weighted.rs::stratify_samples`' median-scratch `clone()`, which is not
a *sized* allocation and so is not a capacity site under the sweep's own
definition; it is recorded because it is the one remaining per-neuron
allocation proportional to the record count. `sample_weighted.rs::compute_sample_weights`'
`vec![uniform; abs_errors.len()]` — the site the issue body flagged — is one
`f32` per record of a slice the loader has already materialised, so the
reservation is 1:1 with memory the caller was already holding. The `HashMap`s
these detectors build (`record_map`, `activation_by_obs`, `target_error_map`,
`existing_synapses`) carry no size hint, no multiplier and no per-entry
fan-out, so none is the #2078 shape.

**Quadratic and cancellation — #2183.** `fan_in.rs`'s pair loop is genuinely
bounded: `MAX_INPUTS_PER_TARGET` (15) truncates `input_scores` **before** the
`for i` / `for j` loops run, so at most 105 pairs are evaluated per target.
The unbounded work is the scan *above* the truncation — every input scored
against every target, each scoring building a `HashMap` over the input's whole
record set — and the same shape in
`multi_hop.rs::find_three_hop_extensions` (every source, for each of up to ten
intermediates, for every target) and in
`gradient_discovery.rs::detect_gradient_candidates` (every output-targeting
synapse against the source's whole record set). The directory contains **zero**
occurrences of `deadline_passed` or `crate::cancellation::is_cancelled`, and
`discovery_dispatch.rs::detect_discovery_modules_parallel` checks the deadline
only **before** it calls a module's closure (Issue #1029), so the deadline
bounds when a detector may start and not how long it may run. That matters for
all three live detectors and is filed as #2183, the same shape as #2161
and #2169. It does **not** matter for `output_competition.rs`, whose
O(outputs²) pair loop — the sharpest of the four — is unreachable.

The remaining two loops are linear, and the verdict differs for each:
`output_bias_drift.rs::detect_output_bias_drift` walks outputs × that output's
own records once, and `sample_weighted.rs::detect_high_error_neurons` walks
neurons × records once, with no nested rescan and no per-entry fan-out. Both
are O(records) in the data the loader has already materialised, so a creature
cannot make either super-linear in its own topology the way the three
quadratic scans above can. They inherit the same "no deadline once started"
property — the directory has no `deadline_passed` call at all — but the work
they can be made to do is bounded by memory the caller was already holding, so
neither is part of #2183. Re-check that verdict if either grows a nested pass.

**Integer overflow.** No index arithmetic in these files subtracts without a
preceding comparison. `sample_weighted.rs::stratify_samples`' `median_idx =
abs_errors.len() / 2` is below the length for every non-empty slice, so the
`select_nth_unstable_by` cannot panic, and its `split_off(mid)` is taken only
when `easy_samples` is non-empty. `activation_recommendation.rs`'s
`(r.activation * 100.0) as i32` is a saturating float→int cast, not UB.

**Shared-state races.** None of these eight files holds mutable shared state:
every public function is a pure computation over a borrowed slice or map, and
each runs inside a single rayon task whose result is returned by value.

**Panics on hostile environment values.** None of these files reads an
environment variable.

**Out-of-class observations — filed, not swept.** Two things fall outside the
five defect classes and are filed as ordinary issues rather than security
findings:

- **#2184** — `activation_recommendation.rs::recommend_activation_function`
  picks the best activation with `max_by` over a `HashMap`, which returns the
  *last* maximal element in a randomised iteration order, so a tie (`TANH` and
  `HARD_TANH` both score `0.7` for a bimodal distribution) makes the emitted
  recommendation differ between runs for byte-identical input. In the same
  file, `activation_recommendation.rs::apply_gradient_flow_penalty` looks up
  `"ReLU6"` while every insertion in
  `activation_recommendation.rs::classify_activation_suitability` spells the
  key `"RELU6"`, so that penalty branch is dead and RELU6 keeps its full score
  in exactly the negative-heavy case the penalty exists to discourage.
  **Both are fixed** — PR #2187 (commit `3d1b24f`) breaks the tie on the
  activation name and respells the penalty key, and it is merged into this
  branch, so the paragraph above describes the tree as swept, not as it stands
  today. The `clean` row for this file is unchanged: neither defect was in
  class, and neither ranking site could see an input-derived float before or
  after.
- **#2185** — `output_competition.rs` (325 lines, Issue #1321) has no
  production caller: `output_competition.rs::detect_output_competition` and
  `output_competition.rs::co_activation` are reached only from tests, and
  there is no `discovery_spec!` entry for the module. This is the AGENTS.md
  § *Dead Levers* shape, and it is why this file's row reads `clean` on
  unreachability rather than on soundness.

**Deliberately out of scope for this sub-issue:** the 50 rows belonging to the
other chunk 8b audit sub-issues.

### recommendation batch_successful + epistatic (Issue #2109)

**One security finding filed — #2190 — plus two out-of-class
observations, #2191 and #2192.** All eight files were read in full for every
defect class probed. The issue body asked three questions of this section directly; each is
answered below with the code that decides it.

**NaN suppression in the dominant-neuron deduplicator — the issue body's
primary question, answered no.** `deduplication.rs::deduplicate_by_dominant_neuron`
groups pairs by whichever source has the larger individual improvement, sorts
each group descending by `combined_improvement` and keeps
`MAX_PAIRS_PER_DOMINANT_NEURON` (3). Under IEEE-754 totalOrder a positive NaN
sorts **above** `+inf`, so a NaN-scored pair would head its group and consume
the whole three-slot cap, suppressing three real candidates. That is the
hazard the issue body names, and it is **not reachable**, for three
independent reasons:

- **Both producers gate the ranking key.** The epistatic path computes it in
  `candidate_generation.rs::compute_combined_improvement_on_range`, which
  returns `0.0` unless `improvement.is_finite()`, and
  `candidate_generation.rs::evaluate_pair_for_epistasis` then requires
  `combined_improvement > 0.0` — a test a NaN loses. The synergistic path
  applies the same `> 0.0` test in
  `pre_screening.rs::evaluate_residual_reduction`, in `f64`, *before* the
  narrowing cast.
- **The dominance key is pre-screened fail-closed.** Both
  `candidate_generation.rs::detect_epistatic_pairs` and
  `pre_screening.rs::detect_synergistic_candidates` filter sources on
  `c.individual_improvement >= MAX_INDIVIDUAL_HARM_FOR_PAIRING` (`0.0`), and a
  NaN loses a `>=`. So the `individual_improvement_a >= individual_improvement_b`
  test that picks the group key never compares a NaN either.
- **`target_impact` is a constant.** The only production caller,
  `candidate_selection.rs::detect_epistatic_and_synergistic`, passes the
  compile-time `1.0` (output target) or `0.5` (hidden target), so the multiply
  that happens *after* the `is_finite` test in
  `compute_combined_improvement_on_range` cannot reintroduce an infinity. Had
  that argument been caller-derived, the gate ordering would have been wrong.

All four sorts in `deduplication.rs` already use `total_cmp`, so the order is
total and `sort_by` cannot panic. Pinned by
`tests/issue_2109_chunk_08b_batch_successful_epistatic_sweep.rs::neither_producer_can_hand_the_deduplicator_a_non_finite_ranking_key`
and `::the_synergistic_prescreen_drops_a_nan_individual_improvement`, which
drive both producers with finite-but-hostile records rather than asserting the
prose.

**The quadratic pair generation — #2190.**
`candidate_generation.rs::detect_epistatic_pairs` runs
`for i in 0..n { for j in (i+1)..n }` over the source contributions collected
for one target, and each iteration calls
`candidate_generation.rs::evaluate_pair_for_epistasis`, which does a `HashSet`
intersection and union over both firing-index sets, then
`candidate_generation.rs::has_cross_sample_harm` and up to three
`compute_combined_improvement_on_range` passes over the shared samples. There
is no ceiling on `n`, no cap on the emitted candidate vector, and **no
deadline or cancellation check anywhere in either subtree** — `grep` for
`deadline_passed` and `is_cancelled` across
`src/analysis/recommendation/batch_successful/` and
`src/analysis/recommendation/epistatic/` returns zero hits.

This is sharper than #2183, where the deadline had to be threaded from the
dispatch layer: here it is **already in scope one frame up**. The sole
production caller, `candidate_selection.rs::detect_epistatic_and_synergistic`,
holds `ctx: &TargetAnalysisContext` and its `deadline: Option<SystemTime>`,
which the neighbouring calls in the same pipeline already honour —
`target_analysis/statistics.rs::filter_and_load_sources` breaks out of its
source loop on it, and `synapse/candidate_generation.rs::group_sources_by_locality`
takes it as an explicit argument after the #2161 fix. The epistatic scan sits
between those two and is passed nothing, so the pipeline stops and then hands
a truncated source list to an uninterruptible quadratic scan. `n` is bounded
only by the deserialised creature: `creature.input` by
`MAX_CREATURE_INPUT_NEURONS` (1,000,000, Issue #1867), the hidden-neuron count
by the length of a vector the caller sent.

Measured by
`tests/issue_2109_chunk_08b_batch_successful_epistatic_sweep.rs::epistatic_pair_generation_still_scans_every_pair_with_no_ceiling`:
12 fully complementary sources emit 66 candidates and 24 emit 276 — exactly
`n(n-1)/2` at both sizes, with nothing truncating either run. The assertion is
a ratio between two runs of the same code, never a wall-clock threshold
(CONTRIBUTING.md § *Unit Tests vs Benchmarks*).

**Division and the integer class — clean, and the issue body's expectation
about `grouping.rs` was wrong.** There is **no division at all** in
`batch_successful/grouping.rs`; `grouping.rs::group_into_batches` only sums at
most `MAX_BATCH_SIZE` (4) finite improvements and multiplies by a compile-time
`BATCH_IMPROVEMENT_SCALE`. Every division in the section is guarded above the
division itself:

| Divisor | Guard | Where |
| --- | --- | --- |
| `original_sse` | `< 1e-10` early return | `detection.rs::evaluate_individual` |
| `union_size` | `is_empty` on both sets, then `union_size == 0` | `candidate_generation.rs::compute_complementarity` |
| `b_firing_count` / `a_firing_count` | `> 0` before each division | `candidate_generation.rs::has_cross_sample_harm` |
| `original_error_sq` | `< 1e-10` early return | `candidate_generation.rs::compute_combined_improvement_on_range` |
| `sum_act_squared`, `original_error_sum`, `residual_error_sum`, `best_individual` | `< 1e-10` / `> 1e-10` / `> 0.0` before each | `pre_screening.rs::evaluate_residual_reduction` |
| `n` | `n < 10` early return | `scoring.rs::check_saturation_risk` |

The one integer narrowing is `compute_firing_indices`' `i as u32` on the
sample index. Truncation needs a single source with more than `u32::MAX`
samples — over 16 GB of activations for one neuron before the rest of the
`HelpfulSample` is counted — so the bound is the allocation, not the cast, and
the group counts and batch indices are all compile-time constants.

**The shape that saves this section from #2181 and #2182 is `f64`.** Every
accumulator that a hostile record set could overflow is `f64` here:
`detection.rs::evaluate_individual`'s two least-squares sums and its
`original_sse` / `residual_sse`, `compute_combined_improvement_on_range`'s
`original_error_sq` / `combined_error_sq`, `evaluate_residual_reduction`'s
four sums, `has_cross_sample_harm`'s two harm accumulators, and
`detection/stats.rs::pearson_correlation_samples` — which is the epistatic
counterpart of the `f32` `pearson_correlation` that #2181 turns on, and which
accumulates `sum`, `cov`, `var_a` and `var_b` in `f64` throughout. Reaching
`f64` overflow from `f32` inputs would need on the order of 10²³⁰ samples, so
the NaN that #2181 manufactures from `±2e30` activations has no counterpart in
this section. The **one** `f32` accumulator in the section is
`scoring.rs::check_saturation_risk`'s `total_combined`, which does have
the #2182 overflow shape — and it has no production caller, which is #2192.

**Out-of-class observations.**

- **#2191** — `deduplication.rs::deduplicate_by_dominant_neuron` and
  `deduplication.rs::deduplicate_synergistic_by_dominant_neuron` build their
  result by iterating a `HashMap` whose `RandomState` is seeded per instance,
  then stabilise it with a `sort_by` that preserves the order of equal keys.
  Tied `combined_improvement` values therefore come out in a different order
  on every call, the emitted coordinated candidates inherit that order, and
  the downstream per-target cap keeps a different subset. The same shape as
  #2184, which was filed and fixed for `activation_recommendation.rs`. Exact
  ties are cheap to produce: quantised `{0, 1}` activations and errors make
  `compute_combined_improvement_on_range` return bit-identical values for
  structurally different pairs. Measured by
  `::dominant_neuron_dedup_still_orders_tied_candidates_non_deterministically`
  — 64 calls on one 8-group fixture produce more than one ordering.
- **#2192** — `scoring.rs::detect_interfering_pairs` (Issue #415) and its
  helper `scoring.rs::check_saturation_risk` have no production caller: the
  only callers anywhere are the file's own `#[cfg(test)] mod tests` and
  `tests/neuron/issue_415_combo_successful_interference.rs`. Two of the three
  interference types it detects are therefore never computed; the third,
  redundancy, is live only through `filter_interfering_epistatic_pairs` and
  `filter_interfering_synergistic_candidates`, which reimplement the
  correlation test and never call it. This is the AGENTS.md § *Dead Levers*
  shape, and it is why the `epistatic/scoring.rs` row reads `clean` on
  unreachability rather than on soundness.

**Reachability of the section as a whole.** The epistatic path is live and
unconditional — `evaluation.rs::collect_and_process_helpful_results` calls
`candidate_selection.rs::detect_epistatic_and_synergistic` on every target it
processes. The batch-successful path is **operator-gated and off by default**:
`scoring_specs.rs::append_scoring_specs` registers its `discovery_spec!` entry
only when `config::batch_successful_enabled()` is true, i.e. when
`NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL=1` is set (Issue #1059, disabled after
zero production successes). Its two entry points,
`detection.rs::detect_individually_successful` and
`grouping.rs::group_into_batches`, are nevertheless swept on soundness rather
than on that gate, because the gate is an environment variable and not a
deletion.

**Deliberately out of scope for this sub-issue:** the 51 rows belonging to the
other chunk 8b audit sub-issues.

## Issues filed

- `negative-result` — the `shared/` sweep found nothing worth filing.
- `negative-result` — the `scoring` sweep found nothing worth filing; the one
  out-of-class observation (the dead outlier lever) is #2177, not a security
  finding.
- `negative-result` — the `synapse scoring + target_analysis` sweep found nothing worth filing.
- `#2161` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `candidate_generation.rs::group_sources_by_locality` runs an O(n²) pairwise
  source scan with no deadline or cancellation check, so the analysis deadline
  and a host cancel request are both ignored until it finishes. Filed by the
  `synapse pipeline` sweep (Issue #2104).
- `#2167` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `post_processing.rs::apply_post_processing`'s three descending `total_cmp`
  sorts on `expected_creature_score_gain` rank a non-finite gain above `+inf`,
  so a NaN/`inf` candidate that reaches this call site is adopted first by the
  host. Filed by the `synapse post-processing` sweep (Issue #2105).
- `#2168` (`security`, `lang:rust`, `severity:low`, `confidence:high`) —
  `post_processing.rs::apply_impact_to_helpful` slices a UUID string by byte
  index and panics on a multi-byte character boundary. Filed by the
  `synapse post-processing` sweep (Issue #2105).
- `#2169` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `structural_patterns.rs`'s detectors run uncancellable scans over untrusted
  creature topology with no deadline check — `detect_noisy_vs_trusted` is
  O(n²·records) pairwise, `detect_collapsible_hidden_neurons` O(neurons ×
  records) — the same shape as #2161.
  Filed by the `synapse post-processing` sweep (Issue #2105).
- `#2170` (`security`, `lang:rust`, `severity:high`, `confidence:high`) — an
  untrusted `ModuleOutcomeTracker` can underflow `ModuleStats::success_rate`,
  flipping `add_synapse_gating.rs`'s success-rate gate. Filed by the
  `synapse post-processing` sweep (Issue #2105).
- `#2181` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `fan_in.rs::detect_fan_in_candidates` ranks its per-target inputs with a
  `partial_cmp(…).unwrap_or(Equal)` comparator that is not a total order,
  while the `corr.abs() < THRESHOLD` filter above it fails open, so a NaN
  correlation reachable from finite records empties the
  `MAX_INPUTS_PER_TARGET` window and suppresses every genuine fan-in
  recommendation for that target. Filed by the `recommendation core` sweep
  (Issue #2108).
- `#2182` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  the descending sorts in `output_bias_drift.rs`, `multi_hop.rs`,
  `gradient_discovery.rs` **and `fan_in.rs`** rank a `+inf`
  `estimated_improvement` first, and each detector manufactures that `+inf`
  from finite records by overflowing an `f32` accumulator or by narrowing an
  `f64` one with a saturating `as f32`; `output_bias_drift.rs` additionally
  emits a non-finite bias in its `SetBias` payload, and `multi_hop.rs` has a
  second path through a NaN `combined_corr`. The `recommendation core`
  counterpart of #2167. Filed by the `recommendation core` sweep (Issue
  #2108). **The `fan_in.rs` half and the `multi_hop.rs` NaN half were added
  after the issue was filed** — the issue body on GitHub says "three
  detectors"; both are recorded in the comment thread and in full here.
- `#2183` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  the three live recommendation-core detectors run uncancellable scans over
  untrusted creature topology and the record stream; the directory contains no
  `deadline_passed` or `is_cancelled` call and the dispatch layer checks the
  deadline only before a module starts. The same shape as #2161 and #2169.
  Filed by the `recommendation core` sweep (Issue #2108).
- `#2184` and `#2185` — out-of-class observations from the
  `recommendation core` sweep (a non-deterministic activation tie-break with a
  dead penalty key, and the never-dispatched `output_competition` module).
  Ordinary issues, **not** security findings. `#2184` is **closed** — its fix
  landed as PR #2187 (commit `3d1b24f`) and is merged into this branch, so
  `recommend_activation_function` now breaks a score tie on the activation
  name and `apply_gradient_flow_penalty` spells the penalty key `"RELU6"`;
  `#2185` remains open.
- `#2190` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `candidate_generation.rs::detect_epistatic_pairs` runs an O(n²) pair scan
  over the source contributions of one target with no ceiling, no cap on the
  candidate vector it grows, and no deadline or cancellation check; neither
  `batch_successful/` nor `epistatic/` contains a single `deadline_passed` or
  `is_cancelled` call, and the deadline is already in scope in the sole
  production caller. The same shape as #2161, #2169 and #2183. Filed by the
  `recommendation batch_successful + epistatic` sweep (Issue #2109).
- `#2191` and `#2192` — out-of-class observations from the
  `recommendation batch_successful + epistatic` sweep (tied candidates ordered
  by `HashMap` seed in the dominant-neuron deduplicator, and the
  never-called Issue #415 interference detector). Ordinary issues, **not**
  security findings — the same treatment #2108 gave #2184 / #2185 and #2107
  gave #2177. Both remain open.
- The remaining sections list their own findings as they are swept.

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
