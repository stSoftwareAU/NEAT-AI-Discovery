## Summary

Cap `add-neurons` candidates per target within a single discovery batch so
that a clearly-hopeless target cannot consume most of the budget with minor
variants. GRQ-sampler commit `744ac60d` recorded 17 of 19 `add-neurons`
failures in one submission against the **same** neuron; the cross-batch
cooldown added in Issue #1130 cannot fire inside a single batch, so a
within-batch cap is required.

Closes #1140.

### What changed

- New constant `MAX_ADD_NEURON_CANDIDATES_PER_TARGET` (default `3`) and
  accessor `max_add_neuron_candidates_per_target()` in
  `src/analysis/constants/candidate_scoring.rs`. The accessor reads the
  `NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET` environment variable and
  clamps to `[1, 32]`.
- New rejection reason `REJECTION_PER_TARGET_CAP` in
  `src/analysis/diagnostics/rejection_reasons.rs`, added to
  `ALL_REJECTION_REASONS` and mapped in `friendly_reason`.
- `src/analysis/neuron/post_processing.rs::build_neuron_results` now calls
  `apply_per_target_cap` after sort-by-gain and the pairing/filter steps but
  before the shuffle/truncate sweep. The helper:
  1. Re-sorts candidates by `expected_creature_score_gain` descending via
     `f32::total_cmp` (deterministic tie breaking).
  2. Groups by `target_neuron_uuid` and retains only the top-K per target.
  3. Returns the drop count, which is recorded against
     `REJECTION_PER_TARGET_CAP` in `metadata.rejection_breakdown`.
- `AGENTS.md` environment-variable table lists the new override.

### Test plan

- `src/analysis/neuron/post_processing.rs::tests::per_target_cap_*` — four
  new unit tests covering the acceptance criteria:
  - `per_target_cap_truncates_target_over_limit` — target with > K
    candidates is truncated to the K highest-gain ones.
  - `per_target_cap_leaves_targets_under_limit_unchanged` — targets with
    ≤ K candidates are untouched.
  - `per_target_cap_breakdown_reports_drop` — the rejection breakdown
    surfaces `per_target_cap` with the correct drop count.
  - `per_target_cap_env_override_controls_limit` — the env override is
    honoured (`#[serial]` to isolate env state).
- Existing regression tests updated to raise the cap where the test
  exercises a single-target creature (the degenerate case for the cap):
  - `tests/activation/complement_via_identity.rs`
  - `tests/gpu/gpu_activation_shaders.rs`
  - `tests/neuron/issue_926_add_neuron_between_hidden.rs`
  - `tests/synapse/coordinated_structural_replace_synapse_with_relu.rs`

### Evidence

Backend/CLI change only — no UI surface to screenshot. `./quality.sh`
passes cleanly (fmt, clippy, check, full test suite, doc build, release
build).
