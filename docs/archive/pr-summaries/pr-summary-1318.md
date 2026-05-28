# Margin-aware focus ranking for OneHot / Margin costs

## Summary

Focus ranking now reweights per-neuron error by the **per-observation
decision margin** when the task descriptor reports a `OneHot` or `Margin`
topology. Observations where the network's top-1 vs top-2 output activation
gap is small contribute more weight; observations where the gap is wide
contribute less. This targets the plateau where margin-improving changes
that don't yet flip an argmax otherwise look worthless.

For every other topology (`Independent`, `Simplex`, `Unknown`, `OTHER`) and
for `None`, the legacy unweighted ranking is preserved — regression guard.

Closes #1318.

## Design

```mermaid
flowchart LR
    A[TaskDescriptor] -->|OneHot/Margin| B[compute_per_obs_margins]
    A -->|other/None| Z[legacy unweighted ranking]
    B --> C[margin_weights_from_margins<br/>w = 1 / (margin + 0.05)]
    C --> D[weighted_average_absolute_error_from_records]
    D --> E[RankedNeuron.total_error]
    D --> F[max_output_error clamp]
```

Touch points (as scoped in the issue):

- `src/focus/impact.rs` — `compute_per_obs_margins` (top-1 minus top-2 per
  obs from recorded output activations) and `margin_weights_from_margins`
  (`w = 1 / (margin + MARGIN_WEIGHT_EPS)`).
- `src/focus/ranking/score_calculation.rs` —
  `weighted_average_absolute_error_from_records` (falls back to the
  unweighted mean when no weights are supplied).
- `src/focus/ranking/mod.rs` —
  `rank_focus_neurons_with_descriptor` and
  `rank_focus_neurons_with_history_and_descriptor`. Existing
  `rank_focus_neurons` / `rank_focus_neurons_with_history` delegate with
  `None` so external callers see no change.
- `src/ffi_internal/analysis.rs` — forwards `input.task_descriptor` (already
  plumbed by #1314) into the new function.

## Evidence

CLI / library change with no UI. Verified via the new test module
`tests/focus/issue_1318_margin_aware_ranking.rs` (9 tests, all green).
Full `./quality.sh` passes — `cargo fmt`, `cargo clippy -D warnings`,
`cargo test`, doc build, release build.

## Test Plan

New tests in `tests/focus/issue_1318_margin_aware_ranking.rs`:

- `margins_reflect_top1_minus_top2_per_observation` — `compute_per_obs_margins`
  returns the correct top-1 minus top-2 gap.
- `margin_weights_upweight_small_margins` — `margin_weights_from_margins`
  obeys the `1 / (margin + eps)` formula.
- `one_hot_descriptor_promotes_close_margin_error_neuron` — under
  `TargetTopology::OneHot`, the hidden neuron whose error sits on the
  close-margin observation ranks above the mirror neuron whose error sits
  on the wide-margin observation, despite identical unweighted means.
- `margin_descriptor_promotes_close_margin_error_neuron` — same for
  `TargetTopology::Margin` (HINGE).
- `one_hot_descriptor_changes_total_error_for_close_margin_neuron` — direct
  numeric guard that the reweighted `total_error` is strictly larger for the
  close-margin neuron under `OneHot`.
- `no_descriptor_matches_legacy_ranking` — passing `None` matches the legacy
  `rank_focus_neurons` exactly.
- `unknown_descriptor_matches_legacy_ranking` — `TaskDescriptor::neutral()`
  matches legacy (covers `OTHER` and unrecognised cost names).
- `independent_descriptor_matches_legacy_ranking` — `MSE` matches legacy.
- `simplex_descriptor_matches_legacy_ranking` — `CROSS_ENTROPY` matches
  legacy.

The four regression-guard tests provide the acceptance criterion
"`OTHER` / `Unknown` / absent ⇒ existing ranking".
