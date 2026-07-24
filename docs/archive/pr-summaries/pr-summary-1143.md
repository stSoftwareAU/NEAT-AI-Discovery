## Summary

Extends the target-saturation pre-check introduced in PR #1116 to the indirect
add-neuron path so it fires regardless of the candidate / intermediate squash.
Previously the gate only suppressed the narrow `compounds_target_clipping`
case (e.g. `ABSOLUTE` feeding into `HARD_TANH`) and otherwise applied a soft
discount; a saturated `HARD_TANH` output could still attract
`ArcTan` or `BENT_IDENTITY` intermediate proposals that produced no usable
gradient (production discovery-cache commit `744ac60d`, candidates
`v2_add-neurons_neuron-921689429_output-0_ArcTan_*` and `…_BENT_IDENTITY_*`,
actual Δerror ≈ `-2.0e-5`). Closes #1143.

The gate now keys solely off the target neuron's observed activation range
versus its bounded squash output range — the candidate/intermediate squash is
irrelevant. The same check also runs at the top of
`analyse_single_target` for synapse analysis so new add-synapse edges into a
saturated target are dropped (existing-edge weight updates are preserved).
Drops surface in the structured rejection breakdown under the new
`REJECTION_TARGET_SATURATED` reason on both `neuronMetadata` and
`synapseMetadata`.

### Changes

- `src/analysis/neuron/preparation.rs` — new `TargetSaturationInfo::rejects_candidates()` helper (squash-agnostic); module re-exported as `pub(crate)` so the synapse path can reuse `compute_target_saturation`.
- `src/analysis/neuron/evaluation.rs` — hard-reject every activation-spec and ReLU-split candidate when `rejects_candidates()` is true, recording the drop count on `NeuronDiagnostics`.
- `src/analysis/diagnostics/neuron_tracking.rs` / `…/rejection.rs` — added `target_saturated_drops` atomic counter plus `record_target_saturated_drops` / `target_saturated_drop_count` accessors on both `NeuronDiagnostics` and `TargetDiagnostics`.
- `src/analysis/diagnostics/rejection_reasons.rs` — added `REJECTION_TARGET_SATURATED` constant, friendly-reason mapping, and entry in `ALL_REJECTION_REASONS`.
- `src/analysis/neuron/post_processing.rs` / `src/analysis/synapse/{post_processing,results}.rs` — surface the per-phase drop counts in `rejection_breakdown`.
- `src/analysis/synapse/target_analysis/mod.rs` — evaluate saturation once per target and clear `sources_to_process` (new edges) when the target is saturated, keeping `existing_sources_to_process` untouched.

## Evidence

Backend/CLI change — no UI. Verification via unit tests in
`src/analysis/neuron/preparation.rs`:

- `test_saturated_target_rejects_arctan_intermediate` — (a) saturated
  `HARD_TANH` + `ArcTan` → rejected.
- `test_saturated_target_rejects_all_intermediates` — (b) saturated
  `HARD_TANH` + any of `ArcTan`, `BENT_IDENTITY`, `IDENTITY`, `RELU`, `TANH`,
  `LOGISTIC`, `GELU`, `SOFTSIGN` → rejected (the gate is squash-independent).
- `test_non_saturated_target_keeps_candidates` — (c) narrow-range `TANH`
  target + any intermediate → kept.
- `test_unbounded_target_never_rejects` — unbounded `IDENTITY` target cannot
  saturate.
- `test_not_saturated_sentinel_keeps_candidates` — default sentinel path.

Full `cargo test --lib` (871 tests) and `cargo test --tests` suites pass.
`cargo clippy --all-targets --all-features -- -D warnings` is clean.

## Test Plan

- Unit: `cargo test --lib analysis::neuron::preparation::tests` — 15 tests pass, including the 5 new Issue #1143 cases.
- Regression: `cargo test --lib` — 871 / 871 pass; integration suites
  (`cargo test --tests`) all green.
- Rejection-breakdown contract: `cargo test --lib analysis::diagnostics::rejection_reasons` passes with the new
  constant listed in `ALL_REJECTION_REASONS`.
- Quality gate: `./quality.sh` (shellcheck → clippy -D warnings → check →
  test → docs → release build).
