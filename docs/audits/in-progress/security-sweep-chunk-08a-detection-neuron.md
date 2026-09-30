# Security sweep — chunk `8a`: Analysis engine — detection + neuron

This record is **staged** under `docs/audits/in-progress/` while the chunk 8a
audit sub-issues (#2217, #2150, #2218, #2152, #2153) fill their sections;
`record_files()` in `tests/issue_2088_sweep_ledger_contract.rs` reads only the
top level of `docs/audits`, so this staged file is not yet a sweep record, and
the `"8a"` entry in `docs/audits/lib-sweep-coverage.json` stays null.
Finalisation (#2154) `git mv`s it to
`docs/audits/security-sweep-chunk-08a-detection-neuron.md` in the same commit
that sets `last_swept`, `baseline_commit` and `record` for chunk `"8a"`. Rules:
[`README.md`](../README.md).

## Record

- **Chunk id:** `8a`
- **Sweep date:** `2026-09-29` (skeleton staged — sections fill in as their sub-issues land)
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2` (`b85a551`)
- **Audit HEAD:** `90e0c151824847f2153d6c593f0dca25592b7d4a` — the tree the line counts below were taken from
- **Exposure:** `internal`
- **Swept by:** `pending` — one audit sub-issue per section below
- **Tracker issue:** #2092 (part of #2216)

Citation convention: symbol names, not line numbers (CONTRIBUTING.md, "Cite Code by Symbol, Never by Line Number (Issue #1942)").

## Files swept

Line counts from `wc -l` at the audit HEAD `90e0c15`. Each `###` section is
owned by one audit sub-issue, which flips its rows from `pending`; the
section markers keep concurrent PRs in disjoint regions.

### shared

<!-- section: shared -->

Shared detection infrastructure; no dedicated sub-issue.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/detection/mod.rs` | 52 | clean — module declarations only (`pub mod` lines and a doc comment); no allocation, recursion, loop, index, `as` cast, comparator or cache key, so none of the six #2092 classes has a site |
| `src/analysis/detection/helpers.rs` | 223 | clean — no index anywhere: `build_record_map`, `compute_activation_stats`, `compute_activation_range`, `compute_mean_abs_activation`, `sort_candidates_by_score_gain` and `weighted_confidence` only iterate or collect, so no index derives from a creature-supplied value; the two `records.len() as f32` casts (`compute_activation_stats`, `compute_mean_abs_activation`) are `usize`→`f32` precision loss that `#![allow(clippy::cast_precision_loss)]` documents and that cannot wrap; empty input returns zeroed `ActivationStats` / `0.0` (no divide by zero), and `compute_activation_range`'s `INFINITY`/`NEG_INFINITY` result for empty input is its documented caller contract; `sort_candidates_by_score_gain` sorts with `total_cmp`, never `partial_cmp().unwrap()`; `build_record_map` collapses a duplicate UUID last-wins inside one call — a per-call map, not a cross-creature cache key |
| `src/analysis/detection/stats.rs` | 224 | finding #2343 — `pearson_correlation_hashmaps` accumulates in `f32` and returns NaN for finite inputs whose variance overflows (the sibling of the #2304 fix to `pearson_correlation`). Ruled out: `spearman_rank_correlation` indexes `rank_y` with `n = x.len()` and no `y.len()` check, but its sole production caller `monotonicity.rs::detect_non_monotonic_neurons` truncates both series to `activations.len().min(errors.len())` and builds `abs_errors` from that truncated slice, so the lengths are equal; `pearson_correlation_samples` never checks `n_samples`, but `redundant_path.rs::compute_activation_correlation` receives `path_a.samples.len().min(path_b.samples.len())`, `pre_screening.rs::compute_activation_correlation` receives `residuals.len().min(complement.samples.len())` where `compute_residual_errors` is a `filter_map` over the primary's samples (so never longer), and `scoring.rs::compute_sample_correlation` passes `samples_a.len().min(samples_b.len())` — both functions are `pub`, so a direct Rust caller with mismatched lengths panics, a precondition no production path violates. The 12 `as` casts: four `f32`→`f64` widenings in `pearson_correlation_samples` (lossless), two `usize`→`f64` and six `usize`→`f32` (`compute_mean`, `compute_variance`, `pearson_correlation`, `pearson_correlation_hashmaps`, `compute_ranks`, `spearman_rank_correlation`) that lose precision but cannot wrap, which is what `#![allow(clippy::cast_precision_loss)]` documents; `compute_ranks`'s `(i + j)` is at most twice a live `Vec` length and cannot overflow `usize`. Empty input: `compute_mean` returns `0.0` when empty and `compute_variance` below 2 elements, and an `f32` sum overflow saturates both to `inf`, never NaN; `compute_ranks` sorts with `total_cmp` and ranks are always finite, so `spearman_rank_correlation` cannot return NaN; `pearson_correlation_samples` accumulates in `f64`, which a finite `f32` input cannot overflow |
| `src/analysis/detection/topology_cache.rs` | 256 | bounded — the seven `with_capacity` sites in `CreatureTopologyCache::new` are in the capacity table: five sized from `neuron_count` (no numeric cap, bounded by the deserialised `creature.neurons` `Vec`) and `output_uuids` / `input_uuids` sized from `creature.output` / `creature.input`, capped at `1_000_000` by `validate_creature_input_bounds` at the FFI boundary since #2078, not inside `new`, which is `pub`. No recursion and no pairwise loop (one pass over neurons, one over synapses), no index, no `as` cast. Not a poisonable cache: the struct is built once per `analyze_all` call and shared via `Arc` inside that call only, lives outside `src/analysis/cache/`, and is keyed by UUID within one creature, so nothing collides across creatures |
| `src/analysis/detection/activation_properties.rs` | 91 | clean — `is_bounded_squash`, `is_saturating_squash` and `can_have_dead_zone` are pure string matches (`matches!` / `eq_ignore_ascii_case`) on the squash name; no allocation, index, cast, loop or cache, and an unknown squash name returns `false` |

### graph

<!-- section: graph -->

Owned by #2217.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/detection/topology.rs` | 387 | pending |
| `src/analysis/detection/skip_connection.rs` | 320 | pending |
| `src/analysis/detection/dead_neuron.rs` | 269 | pending |
| `src/analysis/detection/compound_degradation.rs` | 502 | pending |
| `src/analysis/detection/redundant_path.rs` | 700 | pending |
| `src/analysis/detection/bottleneck.rs` | 372 | pending |
| `src/analysis/detection/low_impact_neuron.rs` | 255 | pending |
| `src/analysis/detection/cross_detection_synthesis.rs` | 309 | pending |

### pairwise

<!-- section: pairwise -->

Owned by #2150.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/detection/correlated_error.rs` | 558 | finding #2346 — `detect_correlated_error_patterns` allocates a dense `n_outputs × n_outputs` `correlation_matrix` and fills it with an O(n_outputs² · S) Pearson pair loop, with no ceiling on `n_outputs` and no deadline check; `module_tiering.rs::ALWAYS_MODULES` means tiering never skips it. **allocation** — the matrix is #2346: `n_outputs` counts `"output"`-typed neurons with ≥ 20 errored records, and `validate_creature` caps `creature.output` at `1_000_000` without ever tying that neuron count to it; `results` and `assigned` are linear in live `Vec` lengths (capacity table). **recursion** — none. **quadratic** — #2346 covers the pair loop and the `find_predictive_inputs` input scan per group; `cluster_correlated_outputs` is O(n_outputs²) in total (a kept group's members are skipped afterwards, and a rejected singleton costs O(n)); `find_shared_obs_indices`, `count_shared_error_samples` and `compute_mean_abs_error` are O(g · S) per group. **panic** — every `correlation_matrix[i][j]` / `output_neurons_with_errors[i]` index is below `n_outputs` by its loop bound; `group_uuids[0]` and `&group_uuids[1..]` sit behind `is_empty()`; the three divisions are guarded (`corr_count > 0`, `total_sample_count >= MIN_SAMPLES_FOR_CORRELATION`, `count > 0`); both sorts use `total_cmp`. **integer overflow** — `shared_neuron_uuid`'s FNV-1a `wrapping_mul` is intentional hashing and `*b as u64` widens; the `as f32` casts lose precision without wrapping, as `#![allow(clippy::cast_precision_loss)]` documents. **cache poisoning** — no cache key: `build_record_map` and `error_by_obs` are per call (the `shared` `helpers.rs` verdict), and both Pearson calls go through `stats.rs::pearson_correlation_hashmaps`, whose NaN-on-overflow is #2343 (the `shared` verdict, not re-derived) |
| `src/analysis/detection/weight_coherence.rs` | 723 | findings #2347, #2351 — `detect_symmetric_cancellation` runs an i<j pair loop over each `topo.fan_in` target's `incoming_synapses`, and `calculate_correlation` rebuilds two `HashMap` record lookups for every surviving pair. That is O(Σ k² · S), with fan-in `k` up to `creature.input` and no deadline check (#2347); the module is not in `EXPENSIVE_MODULES`. **allocation** — the three `with_capacity` sites are sized from the live `creature.neurons` / `creature.synapses` lengths (capacity table); `detect_symmetric_cancellation`'s pushes can outgrow that, which is part of #2347. **recursion** — none. **quadratic** — #2347; `detect_incoherent_weight_ratios` and `detect_near_constant_paths` are linear: one pass over hidden neurons using `CreatureTopologyCache` `fan_in_for` / `fan_out_for` lookups, bounded per the `shared` `topology_cache.rs` verdict. **panic** — `incoming_synapses[i]` / `[j]` are loop-bounded; no production `unwrap` (`max_by(f32::total_cmp).unwrap_or(0.0)` only); divisions are guarded by `outgoing_sum <= EPSILON`, `.max(EPSILON)` and `activations.len() >= config.min_samples`. An incoming sum that overflows to `+∞` passes `ratio > config.max_weight_ratio` and emits a `+∞` gain, which is #2351; `apply_final_coordinated_gain_floor` → `reject_non_finite_gains` drops it only after per-module truncation. **integer overflow** — only `usize` → `f32` precision casts. **cache poisoning** — `records_map` is per call, and `calculate_correlation` delegates to `stats.rs::pearson_correlation_hashmaps` (#2343, `shared` verdict) |
| `src/analysis/detection/co_adaptation.rs` | 255 | finding #2348 — `detect_co_adapted_neurons` runs an O(E² · S) i<j pair loop over `eligible` hidden neurons, with no ceiling on E or on emitted pairs and no deadline check. `EXPENSIVE_MODULES` skips it above 1000 hidden only on non-escalation passes. **allocation** — `candidates` (`eligible.len()`) and `results` (`candidates.len() * 2`) are sized from live `Vec` lengths (capacity table), but pushes grow `candidates` to E(E−1)/2, which is part of #2348. **recursion** — none. **quadratic** — #2348; `co_adapted_pairs_to_coordinated_candidates` builds its fan-in map in one pass over synapses and clones one fan-in per candidate. **panic** — the `HashMap` indexing `records_map[uuid]` / `activation_maps[uuid_a]` cannot miss, because `eligible` is filtered on `records_map.get(uuid).is_some_and(..)` and `activation_maps` is built from `eligible`; `eligible[i]` / `[j]` are loop-bounded; the two mean divisions follow the `shared.len() >= MIN_DISCOVERY_SAMPLE_COUNT` guard; both sorts use `total_cmp`, and `stats.rs::pearson_correlation` never returns a non-finite value (#2304, `shared` verdict). **integer overflow** — `candidates.len() * 2` cannot wrap, because a live `Vec` of multi-byte elements holds far fewer than `usize::MAX / 2`; the `as f32` casts lose precision only. **cache poisoning** — `build_record_map` and `activation_maps` are per call, with no cache key |
| `src/analysis/detection/symmetry_breaking.rs` | 269 | findings #2349, #2351 — `detect_symmetric_neurons` builds `incoming_weights` with an O(synapses × hidden) `hidden_neurons.iter().any`. Its i<j pair loop over `eligible_neurons` then rebuilds both dense weight vectors per pair through `build_weight_vector`'s linear `find`: O(E² · A · F), with no ceiling and no deadline check (#2349); the module is not in `EXPENSIVE_MODULES`. **allocation** — `candidates` (`eligible_neurons.len()`) and `results` (`candidates.len()`) are sized from live `Vec` lengths (capacity table); the per-pair `all_sources.len()` weight vectors and the per-candidate fan-in clone are part of #2349. **recursion** — none. **quadratic** — #2349. **panic** — `eligible_neurons[i]` / `[j]` are loop-bounded; only `unwrap_or_default` / `unwrap_or(0.0)`; `cosine_similarity` returns `0.0` below `f32::EPSILON` magnitude. Weights ≥ ~1.9e19 overflow `x * x`, so the similarity is NaN, which fails `similarity < COSINE_SIMILARITY_THRESHOLD` open and is emitted with a NaN gain (#2351); both sorts use `total_cmp`. **integer overflow** — none: no `as` cast and no count arithmetic beyond loop indices. **cache poisoning** — `build_record_map` and `incoming_weights` are per call, with no cache key |
| `src/analysis/detection/fanin_polarity_conflict.rs` | 302 | findings #2350, #2351 — `fanin_polarity_conflicts_to_coordinated_candidates` runs a linear `creature.neurons.iter().find` per candidate: O(C · N) with C ≤ hidden count and no deadline check (#2350). The detector `detect_fanin_polarity_conflicts` is linear. **allocation** — the single `with_capacity(candidates.len())` is bounded by the live slice (capacity table); `incoming_by_target` and the per-candidate `synapses_to_move` clones total O(synapses). **recursion** — none. **quadratic** — #2350; the `incoming_by_target` and `synapses_to_move` passes are linear (traversal rows). **panic** — no index, no `unwrap` (`unwrap_or(0)` / `unwrap_or_default` / `map_or_else` only); the `conflict_score` division follows the `positive_sum <= 0.0` or `negative_sum <= 0.0` early `continue`. Sums that overflow to `+∞` make `conflict_score` `inf / inf` = NaN, which fails `conflict_score < MIN_CONFLICT_SCORE` open and emits a NaN gain (#2351); both sorts use `total_cmp`. **integer overflow** — none: counts come from `.count()` and no `as` cast is present. **cache poisoning** — `records_map` and `incoming_synapses` are per call; the `fanin-split-{uuid}` neuron id is a per-candidate proposal the controller validates, not a cache key |
| `src/analysis/detection/opposing_synapse.rs` | 236 | pending |
| `src/analysis/detection/output_conflict.rs` | 341 | pending |
| `src/analysis/detection/hard_sample_cluster.rs` | 432 | pending |
| `src/analysis/detection/sentinel_cluster.rs` | 173 | pending |

### per-neuron-a

<!-- section: per-neuron-a -->

Owned by #2218.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/detection/activation_mismatch.rs` | 332 | pending |
| `src/analysis/detection/bias_perturbation.rs` | 265 | pending |
| `src/analysis/detection/bimodal_neuron.rs` | 306 | pending |
| `src/analysis/detection/bounded_range.rs` | 287 | pending |
| `src/analysis/detection/dormant_synapse.rs` | 200 | pending |
| `src/analysis/detection/error_dispersion.rs` | 115 | pending |
| `src/analysis/detection/error_plateau.rs` | 261 | pending |
| `src/analysis/detection/high_error_squash_exploration.rs` | 358 | pending |
| `src/analysis/detection/input_sensitivity.rs` | 622 | pending |
| `src/analysis/detection/monotonicity.rs` | 281 | pending |
| `src/analysis/detection/noise_signal.rs` | 385 | pending |
| `src/analysis/detection/observation_range.rs` | 196 | pending |
| `src/analysis/detection/observation_utilisation.rs` | 244 | pending |

### per-neuron-b

<!-- section: per-neuron-b -->

Owned by #2152.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/detection/operating_point.rs` | 340 | pending |
| `src/analysis/detection/oscillating_neuron.rs` | 248 | pending |
| `src/analysis/detection/output_range_compression.rs` | 314 | pending |
| `src/analysis/detection/output_squash_mismatch.rs` | 558 | pending |
| `src/analysis/detection/restricted_range.rs` | 320 | pending |
| `src/analysis/detection/saturation.rs` | 352 | pending |
| `src/analysis/detection/sentinel_gating.rs` | 307 | pending |
| `src/analysis/detection/squash_weight_rescale.rs` | 371 | pending |
| `src/analysis/detection/topology_diversification.rs` | 472 | pending |
| `src/analysis/detection/unbounded_capping.rs` | 253 | pending |
| `src/analysis/detection/weight_magnitude_reset.rs` | 273 | pending |
| `src/analysis/detection/weight_polarity_flip.rs` | 280 | pending |

### neuron

<!-- section: neuron -->

Owned by #2153.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/analysis/neuron/evaluation.rs` | 529 | clean — **NaN (refuted):** `passes_neuron_improved_ratio` computes `candidate.improved_count as f32 / candidate.total_count as f32` only in the fall-through after `if candidate.total_count == 0 { return false; }`, so the divisor is never zero and both operands come from `u32` counts: the ratio is finite and the `>= NEURON_MIN_IMPROVED_RATIO` gate cannot be failed open by NaN. **Capacity:** no `with_capacity(`, `vec![_; n]` or `.reserve(` hit in `evaluation.rs`. **Recursion:** none. **Quadratic blowup:** none — `evaluate_neuron_candidates` walks one focus target's `work_results` (one entry per eligible source, already trimmed by `preparation.rs::apply_source_budget`) and checks `deadline_passed(deadline)` and `saturation_aborted` before every source; per source, `evaluate_activation_specs` loops over `batched_candidates`, bounded by the `SquashScanPlan` spec count, and `apply_cross_validation_penalty` is O(samples) per admitted candidate — no pairwise loop. **Panic sites:** no `unwrap()`, `expect(`, `panic!`, `[i]` index or slice; the one division is the guarded ratio above; `lock_or_bail` returns an `Err` instead of panicking, and `maybe_abort_saturated_pass` uses `try_lock`. **Integer overflow:** `formed` is the sum of two `usize::from(bool)` values (at most 2); both `record_target_saturated_drops` narrowings use `u32::try_from(…).unwrap_or(u32::MAX)`; `considered.saturating_sub(within_batch)` cannot wrap; the two `as f32` casts lose precision above 2²⁴ (documented by `#![allow(clippy::cast_precision_loss)]`) and cannot wrap. **Cache poisoning:** no `RecordCache` access — `evaluation.rs` reads only the caller's `HelpfulSample`s and writes the per-pass `helpful_map` through `upsert_candidate`, which is not under `src/analysis/cache/`; the `RecordCache` key and lifetime verdict is in the `neuron/mod.rs` row |
| `src/analysis/neuron/mod.rs` | 630 | finding #2352 — the per-focus-target `par_iter().try_fold` in `analyze_neurons_with_cache_and_gpu_queue` swallows a `RecordCache::get(target_uuid)` error: it logs only under `cfg!(debug_assertions)`, returns `Ok(error_acc)` and never touches `NeuronDiagnostics`, so in release the target is dropped silently and `NeuronDiagnostics::no_candidate_summaries` misreports it as `NoEligibleSources`. **Cache poisoning — no cross-creature collision.** Key: `RecordCache::get(&self, neuron_uuid: &str)` keys its `HashMap<String, Arc<CachedNeuronRecords>>` by the raw neuron UUID string exactly as the record phase wrote it to the parquet file, with no creature or run namespace; input neurons use the synthesised `input-{i}` convention. Neuron call sites: `target_uuid` in `analyze_neurons_with_cache_and_gpu_queue` (from `prep.focus_order`, i.e. `input.focus_neurons` filtered against the creature's own `neuron_type_map`); `source_uuid` in `preparation.rs::load_source_records` (from `build_ordered_neurons(&input.creature)`); the literal `"input-0"` debug probe and `format!("input-{mid_input}")` in `preparation.rs::log_creature_config`. Lifetime: one `RecordCache` serves one creature — `orchestration.rs` builds it once per `analyze_all` call with `RecordCache::new_adaptive_with_deadline_and_budget(&input.parquet_file, …)` and wraps it in an `Arc` as `shared_cache`, shared only by that call's synapse and neuron passes; `analyze_neurons` builds its own with `RecordCache::new_adaptive`; `with_loader`, `with_loader_and_deadline` and `new_tiered` each return a fresh instance bound to one `parquet_file`; there is no `static`, `OnceLock` or `lazy_static` `RecordCache` anywhere in `src/`, so no key is ever looked up against another creature's records. Failed loads: `CachedNeuronRecords` is `OnceLock<Result<Arc<Vec<DiscoverRecord>>, String>>`, and `get` stores a loader `Err` through `get_or_init`, so a failed lazy or tiered load is memoised for the rest of that one analysis (the deadline bail in `get` runs before the cell is created and is not memoised; the pre-loaded loader always returns `Ok(Vec::new())`). Every later `get` returns the `Err` loudly; the source path counts it with `diagnostics.record_load_failure`, and the target path is the silent drop filed as #2352. Synthetic UUIDs: `hard_sample_cluster.rs::hard_sample_neuron_uuid` and `output_conflict.rs::split_neuron_uuid` only name `AddNeuron` operations in emitted candidates; neither file calls `RecordCache::get`, and the neuron call sites above take UUIDs only from the input creature and its focus list, so neither synthetic UUID reaches `RecordCache::get` within an analysis — #2295 owns their collision verdict. **Capacity:** no `with_capacity(`, `vec![_; n]` or `.reserve(` hit in `mod.rs`. **Recursion:** none. **Quadratic blowup:** the `try_fold` over focus targets does O(targets × eligible sources) work, checked per focus target (its capacity and traversal row) and per source inside `load_source_records`. **Panic sites:** no `unwrap()`, `expect(`, `panic!`, `[i]` index or slice; the one division, the verbose-only `avg_group`, is guarded by `!group_sizes.is_empty()`. **Integer overflow:** `completed_count.fetch_add(1, …) + 1` is bounded by the focus count; the two `as f32` casts in `avg_group` lose precision (documented by `#![allow(clippy::cast_precision_loss)]`) and cannot wrap |
| `src/analysis/neuron/post_processing.rs` | 1187 | bounded — **Capacity:** two production hits, both in `apply_distinct_target_spread` and both in the capacity and traversal table: `spread` `Vec::with_capacity(min_distinct)`, where `min_distinct_targets_per_batch()` clamps the `NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH` override to `MIN_DISTINCT_TARGETS_PER_BATCH_CEILING` = `32` and the allocation is reached only after the `distinct_count < min_distinct` early return, so it never exceeds the live candidate count either; and `rest` `Vec::with_capacity(candidates.len())`, sized from the live candidate `Vec` that the following `drain(..)` empties into `spread` and `rest`. **NaN and comparators:** every `sort_by` in the file — three `candidates.sort_by` calls and one `sorted_tail.sort_by` — and both `u32::try_from(dropped).expect("fits in u32")` calls sit after `#[cfg(test)] mod tests`, so all are test-only; production ordering goes only through `ranking_score.rs::sort_candidates_by_rank` (called from `build_neuron_results`, `apply_same_target_squash_diversity` and `apply_per_target_cap_with_priority`), whose total-under-NaN verdict is in the `neuron/ranking_score.rs` row, and every production `u32` narrowing is `u32::try_from(…).unwrap_or(u32::MAX)`. A NaN impact score survives `impact.clamp(0.0, 1.0)` in `apply_impact_discounting` and the purely multiplicative `apply_neuron_pessimism_discount`, `apply_saturation_prediction_discount` and `apply_logistic_prediction_calibration` chain, but `apply_min_expected_gain_floor_for_neurons` keeps only `expected_creature_score_gain >= floor`, which NaN fails, so the candidate is dropped and counted by `global_gain_floor_metrics().record_dropped` rather than emitted — refuted. **Cache poisoning:** `post_processing.rs` reaches `RecordCache` only through `apply_impact_discounting` → `compute_impact_scores_for_discounting(&input.creature, cache.as_ref())` in `src/analysis/diagnostics/mod.rs`, whose `RecordCacheProvider::get` forwards to `RecordCache::get(neuron_uuid)`; `focus/impact.rs::compute_impacts_with_activations` passes only `synapse.from_uuid` and output UUIDs taken from the same `input.creature`, against the same per-analysis `params.cache` — the raw-UUID key and one-creature lifetime recorded in the `neuron/mod.rs` row, so no cross-creature collision; the `Err` fallback `compute_impacts_public(creature)` reads no cache. **Recursion:** none. **Quadratic blowup:** none — `build_neuron_results` runs O(n log n) `sort_candidates_by_rank` passes and linear `retain`, `HashSet` and `HashMap` passes over the candidate list (n is the `helpful_map` entry count); `apply_distinct_target_spread` is two linear passes (its traversal row); the `class_priority` branch (`one_hot_class_allocation::apply_class_priority_spread`) and the pairing, sensible-range and `shuffle_within_top_k` steps live in `src/analysis/one_hot_class_allocation.rs` and `src/analysis/utils`, outside this chunk. **Panic sites:** no production `unwrap()`, `expect(`, `panic!`, `[i]` index, slice or division; `failure_cache.as_deref().unwrap_or(&[])` is a non-panicking default and `lock_or_bail` returns an `Err`. **Integer overflow:** `before - candidates.len()` and `original_len - candidates.len()` follow a `retain` that cannot grow the `Vec`, so neither can underflow; the per-target `*count += 1` stops at `max_add_neuron_candidates_per_target()`, clamped to `MAX_ADD_NEURON_CANDIDATES_PER_TARGET_CEILING` = `32`; both rejection-breakdown narrowings saturate with `unwrap_or(u32::MAX)` |
| `src/analysis/neuron/preparation.rs` | 1051 | bounded — **Capacity:** two production hits, both in the capacity and traversal table: `prepare_neuron_analysis`'s `neuron_type_map` `HashMap::with_capacity(input.creature.input + input.creature.neurons.len())` (an unchecked `usize + usize`; `input` is capped at `MAX_CREATURE_INPUT_NEURONS` = `1_000_000` by `validate_creature_input_bounds` at the FFI boundary, `neurons.len()` only by the deserialised `Vec`), whose follow-on `for input_index in 0..input.creature.input` loop allocates one `Arc<str>` per input; and `load_source_records`'s `sources_to_process` `Vec::with_capacity(eligible_sources.len())`. **Recursion:** none. **Quadratic blowup:** `load_source_records` filters `ordered_neurons_arc` once per focus target (O(neurons) per target, O(targets × neurons) per pass), checks `deadline_passed(deadline)` before every source `RecordCache::get`, and `apply_source_budget` caps the sources that proceed; the caller's `try_fold` checks per focus target; the `for _ in 0..load_failure_count` loop is bounded by the eligible-source count. **Panic sites:** no `unwrap()`, `expect(` or `panic!`; `log_creature_config` indexes `records[0]` and `records[records.len() - 1]` only inside `if !records.is_empty()` and divides by the constant `2` (`input.creature.input / 2`); `compute_target_saturation` divides by `output_range` only after `if output_range <= 0.0` returns, and an overflowing `act_max - act_min` gives an `inf` coverage that `clamp(0.0, 1.0)` saturates, never NaN, because non-finite activations are filtered first. **Integer overflow:** `total_eligible = eligible_sources.len() as u32` truncates above `u32::MAX` sources (diagnostic only); `load_failure_count` and `compute_target_saturation`'s `valid_count` are `u32` counters that would wrap in release only past `u32::MAX` loads or records for one target, which no live `Vec` reaches; `log_creature_config`'s `input.creature.input + non_input_count` is the same unchecked `usize + usize` as the capacity row and cannot wrap under the `1_000_000` cap, but would for a direct Rust caller of the `pub` `analyze_neurons` that skips `validate_creature` (the `topology_cache.rs` convention: recorded, not filed); `build_empty_result` narrows with `u32::try_from(…).unwrap_or(u32::MAX)`. **Cache poisoning:** `RecordCache::get` is called with `source_uuid` in `load_source_records` and with `"input-0"` / `format!("input-{mid_input}")` in the verbose-only `log_creature_config` probe; the key derivation and lifetime verdict is in the `neuron/mod.rs` row. `apply_target_cooldown` reads the process-global `target_failure_tracker::global_tracker()`, which is not a `RecordCache`, lives outside `src/analysis/cache/` and outside this chunk, and recovers a poisoned lock with `into_inner` |
| `src/analysis/neuron/ranking_score.rs` | 243 | clean — **Comparator (total under NaN):** `sort_candidates_by_rank` reads `neuron_ranking_reliability_bands()` once, then `candidates.sort_by` compares `candidate_rank_score(b, bands).total_cmp(&candidate_rank_score(a, bands))` and breaks ties with `.then_with` on `b.expected_creature_score_gain.total_cmp(&a.expected_creature_score_gain)`; `f64::total_cmp` and `f32::total_cmp` order every bit pattern, NaN included, so the comparator is a total order and a NaN gain cannot make `sort_by` panic or scramble. `neuron_rank_score_with_bands` is finite for every input anyway: a non-finite `improved_share` becomes `0.0`, a non-finite or non-positive gain earns no credit, and `bands.max(1)` keeps both divisors (`bands` and `bands + 1.0`) at least 1. **NaN (refuted):** `improved_share` returns `0.0` on `total_count == 0` before its `f64::from(improved_count) / f64::from(total_count)` division, and the gain division by `NEURON_RANKING_GAIN_REFERENCE` is gated by `reference > 0.0`. **Capacity:** no `with_capacity(`, `vec![_; n]` or `.reserve(` hit in `ranking_score.rs`. **Recursion:** none. **Quadratic blowup:** none — one O(n log n) sort computing two O(1) scores per comparison (its traversal row). **Panic sites:** no `unwrap()`, `expect(`, `panic!`, index or slice; both divisions are guarded as above. **Integer overflow:** only lossless `f64::from(u32)` widenings; `bands` is clamped to `MIN_NEURON_RANKING_RELIABILITY_BANDS`..=`MAX_NEURON_RANKING_RELIABILITY_BANDS` (`1`..=`1000`). **Cache poisoning:** no cache access or cache key. Deduplicated against #2181 (the `fan_in.rs` non-total comparator): this comparator is already total, so there is no sibling finding |

## Capacity and traversal table

Every capacity sized from creature input, every recursive or graph traversal
and every pairwise loop found in scope, with its bound (or "unbounded") —
rows added by the section sub-issues.

| Symbol | Kind (capacity / traversal / pairwise loop) | Sized from | Bound | Cancellation-checked? |
| --- | --- | --- | --- | --- |
| `topology_cache.rs::CreatureTopologyCache::new` (`hidden_uuids`) | capacity | `HashSet::with_capacity(neuron_count)`, `neuron_count = creature.neurons.len()` | no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec`, one slot per neuron the payload already materialised | n/a — one linear pass over neurons |
| `topology_cache.rs::CreatureTopologyCache::new` (`fan_in`) | capacity | `HashMap::with_capacity(neuron_count)`, `neuron_count = creature.neurons.len()` | no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec` | n/a — one linear pass over synapses |
| `topology_cache.rs::CreatureTopologyCache::new` (`fan_out`) | capacity | `HashMap::with_capacity(neuron_count)`, `neuron_count = creature.neurons.len()` | no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec` | n/a — one linear pass over synapses |
| `topology_cache.rs::CreatureTopologyCache::new` (`existing_synapse_set`) | capacity | `HashMap::with_capacity(neuron_count)`, `neuron_count = creature.neurons.len()` | no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec` | n/a — one linear pass over synapses |
| `topology_cache.rs::CreatureTopologyCache::new` (`synapse_weight_map`) | capacity | `HashMap::with_capacity(neuron_count)`, `neuron_count = creature.neurons.len()` | no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec` | n/a — one linear pass over synapses |
| `topology_cache.rs::CreatureTopologyCache::new` (`output_uuids`) | capacity | `HashSet::with_capacity(creature.output)` | `MAX_CREATURE_OUTPUT_NEURONS` = `1_000_000`, enforced by `validate_creature_input_bounds` (via `validate_creature`) at every FFI entry point, not inside `new`; `new` is `pub`, so a Rust caller that skips `validate_creature` is unbounded | n/a — no loop |
| `topology_cache.rs::CreatureTopologyCache::new` (`input_uuids`) | capacity | `HashSet::with_capacity(creature.input)` | `MAX_CREATURE_INPUT_NEURONS` = `1_000_000`, enforced by `validate_creature_input_bounds` (via `validate_creature`) at every FFI entry point, not inside `new`; `new` is `pub`, so a Rust caller that skips `validate_creature` is unbounded | n/a — no loop |
| `stats.rs::compute_ranks` (`ranks`) | capacity | `vec![0.0f32; n]`, `n = values.len()` | no numeric cap and no creature count — one `f32` per value in the caller's live slice | n/a — an O(n log n) sort then one linear pass |
| `correlated_error.rs::detect_correlated_error_patterns` (`correlation_matrix`) | capacity | `vec![vec![0.0; n_outputs]; n_outputs]`, `n_outputs = output_neurons_with_errors.len()` — a dense `n_outputs × n_outputs` `f32` matrix | **unbounded — finding #2346.** `n_outputs` counts `"output"`-typed neurons with ≥ 20 errored records. `MAX_CREATURE_OUTPUT_NEURONS` = `1_000_000` caps `creature.output` via `validate_creature_input_bounds`, but nothing ties the `"output"`-typed neuron count to it, so no numeric cap applies. Even at that cap the worst case is 10¹² `f32` ≈ 4 TB, and 50,000 outputs already costs 10 GB. `ALWAYS_MODULES` means tiering never skips it | no (module-level only) |
| `correlated_error.rs::detect_correlated_error_patterns` (`results`) | capacity | `Vec::with_capacity(groups.len())` | bounded — `groups` is a live `Vec` from `cluster_correlated_outputs` of disjoint groups of ≥ 2 members, so at most `n_outputs / 2` | n/a — one pass over groups |
| `correlated_error.rs::cluster_correlated_outputs` (`assigned`) | capacity | `vec![false; n]`, `n = output_uuids.len()` | linear in `n_outputs` — one `bool` per entry of the live `output_neurons_with_errors` slice; the quadratic term is the `correlation_matrix` row (#2346) | n/a — sized once |
| `weight_coherence.rs::detect_incoherent_weight_ratios` (`candidates`) | capacity | `Vec::with_capacity(creature.neurons.len())` | no numeric cap — the hidden-neuron count is uncapped; the bound is the already-deserialised `creature.neurons` `Vec`, with at most one push per hidden neuron | n/a — one linear pass over `topo.hidden_uuids` |
| `weight_coherence.rs::detect_near_constant_paths` (`candidates`) | capacity | `Vec::with_capacity(creature.neurons.len())` | no numeric cap — the hidden-neuron count is uncapped; the bound is the already-deserialised `creature.neurons` `Vec`, with at most one push per hidden neuron | n/a — one linear pass over `creature.neurons` |
| `weight_coherence.rs::detect_symmetric_cancellation` (`candidates`) | capacity | `Vec::with_capacity(creature.synapses.len())` | no numeric cap — the synapse count is uncapped; the bound is the already-deserialised `creature.synapses` `Vec`. The reservation is linear, but pushes can grow the `Vec` to Σ k²/2 pairs, which is part of #2347 | no (module-level only) |
| `co_adaptation.rs::detect_co_adapted_neurons` (`candidates`) | capacity | `Vec::with_capacity(eligible.len())` | no numeric cap — bounded by the live `eligible` `Vec` (hidden neurons with ≥ 20 records); pushes can grow it to E(E−1)/2, which is part of #2348 | no (module-level only) |
| `co_adaptation.rs::co_adapted_pairs_to_coordinated_candidates` (`results`) | capacity | `Vec::with_capacity(candidates.len() * 2)` | bounded — twice the live `candidates` slice, at most two pushes per candidate, and the product cannot wrap; the slice's E(E−1)/2 length is #2348 | n/a — one pass over candidates |
| `symmetry_breaking.rs::detect_symmetric_neurons` (`candidates`) | capacity | `Vec::with_capacity(eligible_neurons.len())` | no numeric cap — bounded by the live `eligible_neurons` `Vec`; pushes can grow it to E(E−1)/2, each cloning neuron B's fan-in, which is part of #2349 | no (module-level only) |
| `symmetry_breaking.rs::symmetric_neurons_to_coordinated_candidates` (`results`) | capacity | `Vec::with_capacity(candidates.len())` | bounded — one push per entry of the live `candidates` slice | n/a — one pass over candidates |
| `fanin_polarity_conflict.rs::fanin_polarity_conflicts_to_coordinated_candidates` (`results`) | capacity | `Vec::with_capacity(candidates.len())` | bounded — at most one push per entry of the live `candidates` slice, itself at most one per hidden neuron | n/a — one pass over candidates; the per-candidate `find` is its own traversal row (#2350) |
| `correlated_error.rs::detect_correlated_error_patterns` (i<j correlation pair loop) | pairwise loop | `for i in 0..n_outputs { for j in (i + 1)..n_outputs }`, one `stats.rs::pearson_correlation_hashmaps` over the shared samples per pair | unbounded — O(n_outputs² · S), finding #2346 | no (module-level only) |
| `correlated_error.rs::detect_correlated_error_patterns` (group mean-correlation pair loop) | pairwise loop | i<j over each group's `group_indices` | O(g²) per group; Σ g² ≤ n_outputs² over disjoint groups, so it is dominated by the correlation pair loop (#2346) | no (module-level only) |
| `correlated_error.rs::cluster_correlated_outputs` (i<j complete-linkage loop) | pairwise loop | `for i in 0..n { for j in (i + 1)..n }`, with an `all` over the current group's members | O(n_outputs²) in total — a kept group's members are skipped afterwards, and a rejected singleton costs O(n); dominated by the correlation pair loop (#2346) | no (module-level only) |
| `correlated_error.rs::find_predictive_inputs` (inputs × shared samples, per group) | pairwise loop | per group: a `shared_obs × group_uuids` average, then one `stats.rs::pearson_correlation_hashmaps` per `"input"`-typed neuron with records | O(groups × inputs × S), with groups ≤ n_outputs / 2 and the `"input"`-typed neuron count uncapped — part of #2346 | no (module-level only) |
| `weight_coherence.rs::detect_symmetric_cancellation` (per-`topo.fan_in`-target i<j loop over `incoming_synapses`) | pairwise loop | fan-in `k` per target; `calculate_correlation` rebuilds two `HashMap`s per surviving pair | unbounded — O(Σ k² · S), with `k` up to `creature.input` (`1_000_000`) for an output fed by every input — finding #2347 | no (module-level only) |
| `co_adaptation.rs::detect_co_adapted_neurons` (i<j loop over `eligible`) | pairwise loop | `eligible` hidden neurons with ≥ 20 records; per pair a `shared` `Vec`, an unzip and a Pearson | unbounded — O(E² · S), finding #2348. `EXPENSIVE_MODULES` skips it above 1000 hidden only on non-escalation passes | no (module-level only) |
| `symmetry_breaking.rs::detect_symmetric_neurons` (`hidden_neurons.iter().any` per synapse) | pairwise loop | a linear scan of `hidden_neurons` for every synapse | unbounded — O(synapses × hidden), finding #2349 | no (module-level only) |
| `symmetry_breaking.rs::detect_symmetric_neurons` (i<j loop over `eligible_neurons`) | pairwise loop | `eligible_neurons` hidden neurons with ≥ 20 records; the squash and bias filters are attacker-controlled | unbounded — O(E²) pairs, finding #2349 | no (module-level only) |
| `symmetry_breaking.rs::build_weight_vector` (linear `find` per source, twice per pair) | pairwise loop | `all_sources.len()` sources × a linear `find` over the neuron's fan-in, rebuilt for every pair | unbounded — O(A · F) per call and O(E² · A · F) in total, finding #2349 | no (module-level only) |
| `fanin_polarity_conflict.rs::detect_fanin_polarity_conflicts` (`incoming_by_target` pass) | traversal | one pass over `creature.synapses` with a `HashSet` membership test, then four passes over each target's weights | linear — O(synapses). Probed as a pairwise candidate and found not pairwise | no (module-level only) |
| `fanin_polarity_conflict.rs::fanin_polarity_conflicts_to_coordinated_candidates` (`synapses_to_move` pass) | traversal | per candidate, a clone and sign filter of that target's fan-in from `incoming_synapses` | linear — each candidate is a distinct target, so the Σ of fan-ins is ≤ synapses. Probed as a pairwise candidate and found not pairwise | no (module-level only) |
| `fanin_polarity_conflict.rs::fanin_polarity_conflicts_to_coordinated_candidates` (`creature.neurons.iter().find` per candidate) | pairwise loop | a linear `find` over `creature.neurons` for every candidate | unbounded — O(C · N), with C ≤ hidden count, finding #2350 | no (module-level only) |
| `src/analysis/neuron/preparation.rs::prepare_neuron_analysis` (`neuron_type_map`) | capacity | `HashMap::with_capacity(input.creature.input + input.creature.neurons.len())` — the `usize + usize` is unchecked | `input` ≤ `MAX_CREATURE_INPUT_NEURONS` = `1_000_000`, enforced by `validate_creature_input_bounds` (`src/ffi_types/creature_bounds.rs`, via `validate_creature`) at every FFI entry point, not inside `prepare_neuron_analysis`; `neurons.len()` has no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec`. Under the cap the sum cannot wrap, but a direct Rust caller of the `pub` `analyze_neurons` that skips `validate_creature` gets a wrapping add in release. The follow-on `for input_index in 0..input.creature.input` loop allocates one `Arc<str>` (`format!("input-{input_index}")`) per input — up to `1_000_000` small allocations under the cap | n/a — linear passes, no deadline check inside the preparation loops |
| `src/analysis/neuron/preparation.rs::load_source_records` (`sources_to_process`) | capacity | `Vec::with_capacity(eligible_sources.len())` | no numeric cap — `eligible_sources` is `ordered_neurons_arc` filtered to `index < target_index`, so at most `creature.input + creature.neurons.len()` entries (the capped input plus the deserialised `Vec`), trimmed by `apply_source_budget` before the allocation | yes — `deadline_passed(deadline)` before every source `RecordCache::get` |
| `src/analysis/neuron/mod.rs::analyze_neurons_with_cache_and_gpu_queue` (`par_iter().try_fold` over `focus_order_arc`) | traversal | one closure per focus target in `prep.focus_order` (`input.focus_neurons` deduplicated by `require_unique_focus` and filtered to the creature's own add-neuron targets); each runs `load_source_records` (O(neurons)) and GPU evaluation | bounded by the focus list and the analysis deadline; no recursion | yes (per focus target) — `analysis_timed_out \|\| saturation_aborted \|\| deadline_passed(&deadline)` at the head of each closure returns before any work |
| `src/analysis/neuron/post_processing.rs::apply_distinct_target_spread` (`spread`) | capacity | `Vec::with_capacity(min_distinct)`, `min_distinct = min_distinct_targets_per_batch()` | `MIN_DISTINCT_TARGETS_PER_BATCH_CEILING` = `32` — the `NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH` override is clamped to `1..=32`, and the allocation is reached only when `distinct_count >= min_distinct`, so it is also no larger than the live candidate count; not sized from a topology count | n/a — sized once |
| `src/analysis/neuron/post_processing.rs::apply_distinct_target_spread` (`rest`) | capacity | `Vec::with_capacity(candidates.len())` | no numeric cap — the live candidate `Vec` (the `helpful_map` candidates that survived the gain floor, pairing, sensible-range and squash-diversity filters); `drain(..)` moves every entry into `spread` or `rest`, so the reservation never exceeds what the caller already holds | n/a — sized once |
| `src/analysis/neuron/post_processing.rs::apply_distinct_target_spread` (`candidates.drain(..)` partition loop) | traversal | one `HashSet` pass counting distinct targets, then one `drain(..)` pass pushing each candidate into `spread` or `rest` | linear — O(n) in the candidate count with O(1) `HashSet` look-ups; no recursion and no pairwise comparison | no — bounded by the candidate count (two linear passes); runs after the per-target GPU work, with no deadline check |
| `src/analysis/neuron/ranking_score.rs::sort_candidates_by_rank` (`candidates.sort_by`) | traversal | one comparison sort over the caller's candidate slice; each comparison computes two `candidate_rank_score`s | O(n log n); total under NaN — `total_cmp` on `candidate_rank_score`, then `total_cmp` on `expected_creature_score_gain` | no — bounded by the candidate count (O(n log n)); `post_processing.rs` calls it three times per pass |

## Defect classes probed

Quoted verbatim from #2092.

- **Unbounded allocation from topology counts** — the #2078 class:
  `with_capacity` / `vec![_; n]` / `reserve` sized from `creature.input`,
  `creature.output`, neuron or synapse counts, or any product of them.
  `src/analysis/detection/topology_cache.rs:44` was the #2078 site; find its
  siblings.
- **Unbounded recursion** — graph traversal over a creature's synapse graph
  with no depth cap: a deep or cyclic topology overflows the stack and aborts.
  Check every recursive walk and every cycle-detection assumption.
- **Quadratic/exponential blowup** — pairwise neuron or synapse comparisons
  over attacker-sized inputs: a wall-clock DoS that no timeout in
  `src/cancellation.rs` / `src/watchdog.rs` necessarily interrupts. Record
  which loops are cancellation-checked.
- **Panic sites** — `unwrap`/`expect`/`[i]` indexing/slicing/division on
  values derived from the creature, and `partial_cmp().unwrap()` on scores
  that could be `NaN`.
- **Integer overflow** — `as` casts and arithmetic on counts/indices;
  wrapping in release builds (the #1906 class).
- **Cache poisoning** — keys in `src/analysis/cache/` reachable from this half
  that collide across distinct creatures.

The `topology_cache.rs:44` citation is #2092's own wording, quoted as filed
against baseline `b85a551`; the #2078 site is `CreatureTopologyCache::new` in
`src/analysis/detection/topology_cache.rs`.

## Outcome

`pending`

## Issues filed

- `#2343` (`security`, `lang:rust`, `severity:low`, `confidence:high`) —
  `stats.rs::pearson_correlation_hashmaps` accumulates in `f32`, so finite
  inputs whose variance overflows return NaN, which every production caller's
  `>=` gate then drops silently. Filed by the `shared` sweep (Issue #2281).
- `#2352` (`security`, `lang:rust`, `severity:low`, `confidence:high`) —
  `neuron/mod.rs::analyze_neurons_with_cache_and_gpu_queue` drops a focus
  target silently in release when `RecordCache::get(target_uuid)` fails (the
  failed load is memoised in `CachedNeuronRecords` for the rest of the
  analysis), and `NeuronDiagnostics` misreports it as `NoEligibleSources`.
  Filed by the `neuron` sweep (Issue #2300).
- #2295 — cross-linked from the `neuron/mod.rs` row: it owns the collision
  verdict for the synthetic `hard_sample_neuron_uuid` / `split_neuron_uuid`
  UUIDs; not a finding filed by this sweep.
- Prior remediations cited by `shared` rows, not filed by this sweep: #2078
  (`creature.output` cap in `validate_creature_input_bounds`) and #2304
  (`pearson_correlation` non-finite hardening).
- #2092 — the tracker issue (part of #2216) whose six defect classes this
  sweep probes; cited from the `mod.rs` and `stats.rs` shared rows, not a
  finding filed by this sweep.
- `#2346` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `correlated_error.rs::detect_correlated_error_patterns` allocates an
  unbounded `n_outputs × n_outputs` `correlation_matrix` and runs an
  uncancellable O(n_outputs² · S) pair scan; `n_outputs` is not tied to the
  capped `creature.output`. Filed by the `pairwise` sweep (Issue #2294).
- `#2347` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `weight_coherence.rs::detect_symmetric_cancellation` runs an uncancellable
  per-target O(k²) fan-in pair scan that rebuilds two record `HashMap`s per
  pair in `calculate_correlation`. Filed by the `pairwise` sweep (Issue #2294).
- `#2348` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `co_adaptation.rs::detect_co_adapted_neurons` runs an uncancellable
  O(E² · S) hidden-neuron pair scan with no ceiling on emitted pairs; tiering
  skips it only on non-escalation passes. Filed by the `pairwise` sweep
  (Issue #2294).
- `#2349` (`security`, `lang:rust`, `severity:medium`, `confidence:high`) —
  `symmetry_breaking.rs::detect_symmetric_neurons` runs an O(synapses × hidden)
  membership scan and an uncancellable O(E²) pair loop that rebuilds both
  weight vectors per pair via `build_weight_vector`'s linear `find`. Filed by
  the `pairwise` sweep (Issue #2294).
- `#2350` (`security`, `lang:rust`, `severity:low`, `confidence:high`) —
  `fanin_polarity_conflict.rs::fanin_polarity_conflicts_to_coordinated_candidates`
  runs a linear `creature.neurons` `find` per candidate, O(C · N). Filed by the
  `pairwise` sweep (Issue #2294).
- `#2351` (`security`, `lang:rust`, `severity:low`, `confidence:high`) —
  finite synapse weights that overflow `f32` sums make
  `symmetry_breaking.rs::cosine_similarity` and
  `fanin_polarity_conflict.rs::detect_fanin_polarity_conflicts` produce a NaN
  that fails their skip-on-`<` gates open, and
  `weight_coherence.rs::detect_incoherent_weight_ratios` emit a `+∞` gain;
  `reject_non_finite_gains` (Issue #1367) drops these only after per-module
  `max_candidates` truncation. Filed by the `pairwise` sweep (Issue #2294).
- #2181 — deduplication reference cited from the `neuron/ranking_score.rs`
  row: the closed `fan_in.rs` non-total comparator finding.
  `sort_candidates_by_rank` already chains `total_cmp`, so there is no sibling
  finding; not a finding filed by this sweep.
- The `neuron` sweep of `evaluation.rs`, `post_processing.rs` and
  `ranking_score.rs` (Issue #2301) filed no new finding: every probe across
  the six #2092 classes was refuted or bounded, as each row records.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- #2078 / #1867 — creature width caps `MAX_CREATURE_INPUT_NEURONS` and
  `MAX_CREATURE_OUTPUT_NEURONS` = `1_000_000`, enforced by
  `validate_creature_input_bounds` in `src/ffi_types/creature_bounds.rs` before
  any detection pass sizes an allocation from `creature.input` /
  `creature.output`.

## Verify this record

`pending` — filled at finalisation, #2154, with the
`git diff <baseline>..HEAD -- src/analysis/detection src/analysis/neuron`
command.
