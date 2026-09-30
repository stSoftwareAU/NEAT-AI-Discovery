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
| `src/analysis/detection/correlated_error.rs` | 558 | pending |
| `src/analysis/detection/weight_coherence.rs` | 723 | pending |
| `src/analysis/detection/co_adaptation.rs` | 255 | pending |
| `src/analysis/detection/symmetry_breaking.rs` | 269 | pending |
| `src/analysis/detection/fanin_polarity_conflict.rs` | 302 | pending |
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
| `src/analysis/neuron/mod.rs` | 630 | finding #2352 — the per-focus-target `par_iter().try_fold` in `analyze_neurons_with_cache_and_gpu_queue` swallows a `RecordCache::get(target_uuid)` error: it logs only under `cfg!(debug_assertions)`, returns `Ok(error_acc)` and never touches `NeuronDiagnostics`, so in release the target is dropped silently and `NeuronDiagnostics::no_candidate_summaries` misreports it as `NoEligibleSources`. **Cache poisoning — no cross-creature collision.** Key: `RecordCache::get(&self, neuron_uuid: &str)` keys its `HashMap<String, Arc<CachedNeuronRecords>>` by the raw neuron UUID string exactly as the record phase wrote it to the parquet file, with no creature or run namespace; input neurons use the synthesised `input-{i}` convention. Neuron call sites: `target_uuid` in `analyze_neurons_with_cache_and_gpu_queue` (from `prep.focus_order`, i.e. `input.focus_neurons` filtered against the creature's own `neuron_type_map`); `source_uuid` in `preparation.rs::load_source_records` (from `build_ordered_neurons(&input.creature)`); the literal `"input-0"` debug probe and `format!("input-{mid_input}")` in `preparation.rs::log_creature_config`. Lifetime: one `RecordCache` serves one creature — `orchestration.rs` builds it once per `analyze_all` call with `RecordCache::new_adaptive_with_deadline_and_budget(&input.parquet_file, …)` and wraps it in an `Arc` as `shared_cache`, shared only by that call's synapse and neuron passes; `analyze_neurons` builds its own with `RecordCache::new_adaptive`; `with_loader`, `with_loader_and_deadline` and `new_tiered` each return a fresh instance bound to one `parquet_file`; there is no `static`, `OnceLock` or `lazy_static` `RecordCache` anywhere in `src/`, so no key is ever looked up against another creature's records. Failed loads: `CachedNeuronRecords` is `OnceLock<Result<Arc<Vec<DiscoverRecord>>, String>>`, and `get` stores a loader `Err` through `get_or_init`, so a failed lazy or tiered load is memoised for the rest of that one analysis (the deadline bail in `get` runs before the cell is created and is not memoised; the pre-loaded loader always returns `Ok(Vec::new())`). Every later `get` returns the `Err` loudly; the source path counts it with `diagnostics.record_load_failure`, and the target path is the silent drop filed as #2352. Synthetic UUIDs: `hard_sample_cluster.rs::hard_sample_neuron_uuid` and `output_conflict.rs::split_neuron_uuid` only name `AddNeuron` operations in emitted candidates; neither file calls `RecordCache::get`, and the neuron call sites above take UUIDs only from the input creature and its focus list, so neither synthetic UUID reaches `RecordCache::get` within an analysis — #2295 owns their collision verdict. **Capacity:** no `with_capacity(`, `vec![_; n]` or `.reserve(` hit in `mod.rs`. **Recursion:** none. **Quadratic blowup:** the `try_fold` over focus targets does O(targets × eligible sources) work, checked per focus target (its capacity and traversal row) and per source inside `load_source_records`. **Panic sites:** no `unwrap()`, `expect(`, `panic!`, `[i]` index or slice; the one division, the verbose-only `avg_group`, is guarded by `!group_sizes.is_empty()`. **Integer overflow:** `completed_count.fetch_add(1, …) + 1` is bounded by the focus count; the two `as f32` casts in `avg_group` lose precision (documented by `#![allow(clippy::cast_precision_loss)]`) and cannot wrap |
| `src/analysis/neuron/post_processing.rs` | 1187 | pending |
| `src/analysis/neuron/preparation.rs` | 1051 | bounded — **Capacity:** two production hits, both in the capacity and traversal table: `prepare_neuron_analysis`'s `neuron_type_map` `HashMap::with_capacity(input.creature.input + input.creature.neurons.len())` (an unchecked `usize + usize`; `input` is capped at `MAX_CREATURE_INPUT_NEURONS` = `1_000_000` by `validate_creature_input_bounds` at the FFI boundary, `neurons.len()` only by the deserialised `Vec`), whose follow-on `for input_index in 0..input.creature.input` loop allocates one `Arc<str>` per input; and `load_source_records`'s `sources_to_process` `Vec::with_capacity(eligible_sources.len())`. **Recursion:** none. **Quadratic blowup:** `load_source_records` filters `ordered_neurons_arc` once per focus target (O(neurons) per target, O(targets × neurons) per pass), checks `deadline_passed(deadline)` before every source `RecordCache::get`, and `apply_source_budget` caps the sources that proceed; the caller's `try_fold` checks per focus target; the `for _ in 0..load_failure_count` loop is bounded by the eligible-source count. **Panic sites:** no `unwrap()`, `expect(` or `panic!`; `log_creature_config` indexes `records[0]` and `records[records.len() - 1]` only inside `if !records.is_empty()` and divides by the constant `2` (`input.creature.input / 2`); `compute_target_saturation` divides by `output_range` only after `if output_range <= 0.0` returns, and an overflowing `act_max - act_min` gives an `inf` coverage that `clamp(0.0, 1.0)` saturates, never NaN, because non-finite activations are filtered first. **Integer overflow:** `total_eligible = eligible_sources.len() as u32` truncates above `u32::MAX` sources (diagnostic only); `load_failure_count` and `compute_target_saturation`'s `valid_count` are `u32` counters that would wrap in release only past `u32::MAX` loads or records for one target, which no live `Vec` reaches; `log_creature_config`'s `input.creature.input + non_input_count` is the same unchecked `usize + usize` as the capacity row and cannot wrap under the `1_000_000` cap, but would for a direct Rust caller of the `pub` `analyze_neurons` that skips `validate_creature` (the `topology_cache.rs` convention: recorded, not filed); `build_empty_result` narrows with `u32::try_from(…).unwrap_or(u32::MAX)`. **Cache poisoning:** `RecordCache::get` is called with `source_uuid` in `load_source_records` and with `"input-0"` / `format!("input-{mid_input}")` in the verbose-only `log_creature_config` probe; the key derivation and lifetime verdict is in the `neuron/mod.rs` row. `apply_target_cooldown` reads the process-global `target_failure_tracker::global_tracker()`, which is not a `RecordCache`, lives outside `src/analysis/cache/` and outside this chunk, and recovers a poisoned lock with `into_inner` |
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
| `src/analysis/neuron/preparation.rs::prepare_neuron_analysis` (`neuron_type_map`) | capacity | `HashMap::with_capacity(input.creature.input + input.creature.neurons.len())` — the `usize + usize` is unchecked | `input` ≤ `MAX_CREATURE_INPUT_NEURONS` = `1_000_000`, enforced by `validate_creature_input_bounds` (`src/ffi_types/creature_bounds.rs`, via `validate_creature`) at every FFI entry point, not inside `prepare_neuron_analysis`; `neurons.len()` has no numeric cap — the bound is the already-deserialised `creature.neurons` `Vec`. Under the cap the sum cannot wrap, but a direct Rust caller of the `pub` `analyze_neurons` that skips `validate_creature` gets a wrapping add in release. The follow-on `for input_index in 0..input.creature.input` loop allocates one `Arc<str>` (`format!("input-{input_index}")`) per input — up to `1_000_000` small allocations under the cap | n/a — linear passes, no deadline check inside the preparation loops |
| `src/analysis/neuron/preparation.rs::load_source_records` (`sources_to_process`) | capacity | `Vec::with_capacity(eligible_sources.len())` | no numeric cap — `eligible_sources` is `ordered_neurons_arc` filtered to `index < target_index`, so at most `creature.input + creature.neurons.len()` entries (the capped input plus the deserialised `Vec`), trimmed by `apply_source_budget` before the allocation | yes — `deadline_passed(deadline)` before every source `RecordCache::get` |
| `src/analysis/neuron/mod.rs::analyze_neurons_with_cache_and_gpu_queue` (`par_iter().try_fold` over `focus_order_arc`) | traversal | one closure per focus target in `prep.focus_order` (`input.focus_neurons` deduplicated by `require_unique_focus` and filtered to the creature's own add-neuron targets); each runs `load_source_records` (O(neurons)) and GPU evaluation | bounded by the focus list and the analysis deadline; no recursion | yes (per focus target) — `analysis_timed_out \|\| saturation_aborted \|\| deadline_passed(&deadline)` at the head of each closure returns before any work |

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
