## Summary

Bottleneck bypass `AddSynapse` candidates and squash-rescale `SetWeight` operations can no longer carry a non-finite (±∞) weight under a finite gain (CWE-682). Closes #2375.

- `bottleneck.rs::bottleneck_neurons_to_coordinated_candidates`: the bypass is skipped when `bypass_weight = upstream_weight * downstream_weight * 0.5` is not finite.
- `squash_weight_rescale.rs::detect_squash_weight_rescale_candidates` (sibling site from the trusted comment on the issue): the candidate is dropped when any rescaled weight is not finite.
- The final candidate pipeline rejects atomic groups containing non-finite operation weights or biases, covering the sibling sites in #2375 before module budgets and target caps.
- `candidate_starvation.rs` classifies `non_finite_operation` as an upstream rejection for candidate-starvation decisions.
- Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.
- Audit record `docs/audits/security-sweep-chunk-08a-detection-neuron.md` marks both findings fixed.

## Spec

### Intent and Rationale

- Two finite synapse weights (e.g. 2e19 × 2e19) overflow `f32`; neither `reject_non_finite_gains` nor the #2167 gate inspects operation weights, so the candidate reached the FFI response as `"weight": null`.
- The shared operation guard drops the whole atomic group and records `non_finite_operation`; it runs before selection limits and again at finalisation.

### Essential Design Decisions

- Skip rather than clamp: a clamped weight would be an invented value with no evidence behind it.
- The parallel-neuron branch is unchanged: it only halves a finite weight, so it cannot overflow.

### Undiscoverable Facts

- The squash-rescale sibling site comes from a trusted comment on #2375, not the issue body.

## Evidence

Backend-only change, no UI. Red-then-green, run with the guard removed from the unfixed production code and restored:

- Bottleneck: `bypass_weight_is_never_non_finite` failed with `emitted AddSynapse weight must be finite, got inf`; passes with the fix.
- Squash rescale: `rescaled_weights_are_never_non_finite` failed with `detected rescaled weight input-1 -> hidden-1 is non-finite: inf`; passes with the fix.

Original trigger closed: any non-finite bypass weight (including NaN) is rejected by `is_finite()` before the operation is built, and any non-finite rescaled weight drops the whole candidate before `SetWeight` operations are produced; the only other `AddSynapse`/`SetWeight` weight sources in these two converters are a halved finite weight or the unchanged finite `old_weight`, so no equivalent bypass remains.

## Test Plan

- `tests/issue_2375_non_finite_operations.rs` exercises the sibling converters and finite controls.
- `tests/issue_2375_bottleneck_bypass_weight_finite.rs::bypass_weight_is_never_non_finite` — 2e19 × 2e19 weights; asserts the bottleneck is detected, the parallel candidate is still emitted, every `AddSynapse` weight is finite and no bypass is emitted. Red on base (above).
- `tests/issue_2375_bottleneck_bypass_weight_finite.rs::bypass_weight_is_emitted_for_ordinary_weights` — positive control (0.8 × 0.6 → bypass weight 0.24).
- `tests/issue_2375_bottleneck_bypass_weight_finite.rs::bypass_weight_is_emitted_below_overflow_boundary` — 1e19 × 1e19 × 0.5 = 5e37 still emitted.
- `tests/issue_2375_squash_rescale_weight_finite.rs::rescaled_weights_are_never_non_finite` — one 1e38 incoming weight, best factor 4.0; asserts no non-finite weight and no candidate. Red on base (above).
- `tests/issue_2375_squash_rescale_weight_finite.rs::ordinary_weight_fixture_reaches_the_rescale` — same fixture with weight 0.5 yields a `SetWeight` of 2.0.
- Existing bottleneck and squash-rescale suites (`analysis`, `detection`, `infrastructure`, `scoring`, `recommendation`, issue 2377/2298/2280/2282/2351/2359 targets) passed.
- No existing assertions were removed or edited.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`: passed.
- `./quality.sh < /dev/null`: passed in full on the pushed head (exit 0, "All quality checks passed!"); the run exceeded the 10-minute foreground cap, so it ran in the background.
