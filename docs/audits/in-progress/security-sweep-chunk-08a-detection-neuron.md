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
