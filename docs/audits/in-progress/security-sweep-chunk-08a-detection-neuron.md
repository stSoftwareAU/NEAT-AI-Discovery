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
| `src/analysis/neuron/evaluation.rs` | 529 | pending |
| `src/analysis/neuron/mod.rs` | 630 | pending |
| `src/analysis/neuron/post_processing.rs` | 1187 | pending |
| `src/analysis/neuron/preparation.rs` | 1051 | pending |
| `src/analysis/neuron/ranking_score.rs` | 243 | pending |

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
