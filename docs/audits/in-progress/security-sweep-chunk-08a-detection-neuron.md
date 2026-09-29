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
| `src/analysis/detection/mod.rs` | 52 | pending |
| `src/analysis/detection/helpers.rs` | 223 | pending |
| `src/analysis/detection/stats.rs` | 224 | pending |
| `src/analysis/detection/topology_cache.rs` | 256 | pending |
| `src/analysis/detection/activation_properties.rs` | 91 | pending |

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

`pending`

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
