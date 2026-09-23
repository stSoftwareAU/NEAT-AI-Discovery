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
  `src/analysis/synapse/holdout_validation.rs`, which add no production code, so
  every outcome below still describes the current tree. **Line counts stay as at
  the baseline commit** — that is what a later reader diffs against.
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

58 files, 20,997 lines. Line counts as at the baseline commit. Every row starts
`pending`; the sub-issue owning the section replaces it with an outcome and a
one-line reason.

### synapse pipeline

3,641 lines. Swept by Issue #2104. The 14 rows are the files that sub-issue
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
| `preparation.rs::build_creature_lookups` | `NeuronIndex::with_capacity(neurons.len() + input + synapses.len() / 10)` | `MAX_CREATURE_INPUT_NEURONS` (1,000,000) caps `input` at every FFI entry point; the other two terms count live `Vec`s, so the sum cannot overflow `usize` | bounded |
| `candidate_generation.rs::build_ordered_neurons` | `Vec::with_capacity(creature.input + creature.neurons.len())` | same cap on `input`; `neurons.len()` counts a live `Vec`, so the sum is at most 1,000,000 + a deserialised vector's length | bounded |
| `candidate_generation.rs::group_sources_by_locality` | `vec![false; sources.len()]` | `sources` is a live slice of per-target sources already materialised by `filter_and_load_sources` | bounded |
| `gpu_evaluation.rs::evaluate_all_activation_specs_batched` | `vec![None; plan.specs.len()]` | `plan.specs` is a subset of the compile-time `ACTIVATION_SPECS` array; no caller-supplied count reaches it | bounded |
| `filtering.rs::truncate_combined_synapse_candidate_sets` | `Vec::with_capacity(total)` | `total` is the sum of three live vector lengths, and the branch is only reached when `total > limit` | bounded |
| `holdout_validation.rs::split_samples_holdout` | `Vec::with_capacity(samples.len() - validate_count)` | the early return rejects fewer than `HOLDOUT_MIN_SAMPLE_COUNT` (20) samples and `validate_count = round(0.3n) < n` for every `n >= 20`, so the subtraction cannot wrap | bounded |
| `holdout_validation.rs::split_samples_holdout` | `Vec::with_capacity(validate_count)` | `validate_count = round(0.3 × samples.len())`, at most 30% of a live slice's length | bounded |
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
<!-- section: recommendation core -->
<!-- section: recommendation batch_successful + epistatic -->
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

## Issues filed

- `negative-result` — the `shared/` sweep found nothing worth filing.
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
