## Summary
Enforce squash diversity within each target neuron in add-neuron post-processing.
Within the candidates for any single `to_neuron_uuid`, only the highest-gain
candidate per distinct `squash` value survives — duplicate `(target, squash)`
pairs are dropped before the per-target cap (#1140) so the remaining cap budget
is spent on genuinely diverse proposals rather than near-identical failing bets
(e.g. the production discovery-cache commit `744ac60d` case where 17 candidates targeting the same
neuron all proposed the same `ReLU6` squash). Closes #1141.

## Changes
- `src/analysis/diagnostics/rejection_reasons.rs` — added the
  `REJECTION_SAME_TARGET_SQUASH_DUPLICATE` reason and a friendly summary
  rendering. Appended it to `ALL_REJECTION_REASONS`.
- `src/analysis/neuron/post_processing.rs` —
  - new `apply_same_target_squash_diversity` function: sorts by gain
    descending (NaN-safe via `total_cmp`), then retains only the highest-gain
    candidate per `(target_neuron_uuid, squash)` pair.
  - call it in `build_neuron_results` immediately before the
    `apply_per_target_cap` (#1140) pass.
  - record the drop count under `REJECTION_SAME_TARGET_SQUASH_DUPLICATE` in
    the structured rejection breakdown.
- `tests/neuron/issue_926_add_neuron_between_hidden.rs` — adjusted to reflect
  the new filter behaviour: the squash-diversity dedup keeps the highest-gain
  candidate per `(target, squash)` regardless of source, so for `hidden-C` the
  surviving candidate is typically `hidden-A → hidden-C` (which has a direct
  synapse and is then converted to a coordinated structural replacement and
  discounted for multi-op gain). The end-to-end coverage that `analyze_neurons`
  evaluates hidden targets is preserved by `test_hidden_neuron_candidate_properties`.
  Also added the mixed-case `ReLU` squash to the valid list — the diversity
  filter allows both `RELU` and `ReLU` to surface as distinct `(target, squash)`
  keys.
- `Cargo.toml` — patch bump to `0.74.19`.

## Evidence
This is a pure backend filter change with no UI surface. Validation is via
unit and integration tests:

- `apply_same_target_squash_diversity` is covered by four new unit tests in
  `src/analysis/neuron/post_processing.rs`:
  - `squash_diversity_keeps_one_candidate_per_distinct_squash` — the
    five-candidate `[ReLU6, ReLU6, SOFTSIGN, ReLU6, ArcTan]` scenario from
    the issue. Asserts the highest-gain `ReLU6` plus `SOFTSIGN` plus `ArcTan`
    are retained in gain-descending order.
  - `squash_diversity_independent_targets_not_collapsed` — same squash on
    different targets must not be deduplicated.
  - `squash_diversity_empty_input_is_safe` — boundary check for empty input.
  - `squash_diversity_breakdown_reports_drop` — verifies the rejection
    breakdown surfaces the drop under `REJECTION_SAME_TARGET_SQUASH_DUPLICATE`.
- Existing per-target-cap tests still pass (helper updated to vary squash
  across test candidates so the cap test exercises only the cap behaviour).
- `./quality.sh` passes cleanly: `fmt`, `clippy`, `cargo check`,
  full test suite, doc build, and release build all green.

## Test Plan
- [x] `cargo test --lib --all-features post_processing` — 8 passed
- [x] `cargo test --test neuron --all-features issue_926` — 3 passed
- [x] `./quality.sh` — all gates green
