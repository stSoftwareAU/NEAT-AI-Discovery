## Summary

Added the missing `// SAFETY:` comments to every `unsafe { std::env::* }` block called out in the issue (`src/focus/ranking/removal_candidates.rs` and `benches/zero_copy_buffer.rs`), and — to enforce the invariant going forward — enabled `clippy::undocumented_unsafe_blocks = "warn"` in `Cargo.toml`'s `[lints.clippy]` section. Enabling the lint surfaced ~50 additional undocumented `unsafe` blocks across the test and source tree; all have been annotated with matching `// SAFETY:` comments naming the serialisation invariant (`#[serial]` / `env_lock()` / single-threaded benchmarks) that makes the call sound. Closes #1240.

## Evidence

This is a documentation/lint change with no UI or runtime-performance impact. The lint is now enforced by `cargo clippy --all-targets --all-features -- -D warnings`, which is the same gate `quality.sh` and CI run.

Verification:

- `cargo clippy --all-targets --all-features -- -D warnings` — passes (previously surfaced 50 `undocumented_unsafe_blocks` errors after enabling the lint).
- `cargo test --lib` — 997 passed, 0 failed.
- `cargo fmt --all --check` — clean.

Files updated (SAFETY comments added):

- `src/focus/ranking/removal_candidates.rs` — three test sites in `env_var_override_changes_effective_floor`.
- `src/analysis/scoring/calibration_correction.rs` — ten test sites under `#[serial]`.
- `benches/zero_copy_buffer.rs` — two cleanup sites (lines 86 and 109).
- `tests/gpu_timing.rs` — four cleanup sites.
- `tests/regression/regression_v0_1_127.rs`, `tests/neuron/issue_132_cost_of_growth.rs`, `tests/neuron/issue_414_remove_neuron_high_error.rs`, `tests/focus/focus.rs`, `tests/focus/issue_1172_focus_ranking_memory_budget.rs`, `tests/focus/issue_156_hidden_focus_neurons_filtered.rs`, `tests/focus/issue_182_focus_unused_observations.rs` — inline match-arm SAFETY comments on the `Drop` guards.
- `tests/scoring/issue_192_error_distribution_analysis.rs` — one cleanup site.
- `tests/infrastructure/issue_228_zero_copy_buffer.rs` — seven sites.
- `tests/infrastructure/observability.rs` — three cleanup sites.
- `tests/analysis/issue_1165_calibration_miss_logging.rs` — four sites.
- `tests/analysis/issue_199_dynamic_constant_source_threshold.rs` — two cleanup sites.
- `tests/analysis/issue_527_diagnostic_tracking.rs` — six sites (two pairs of inline match arms).
- `Cargo.toml` — added `undocumented_unsafe_blocks = "warn"` under `[lints.clippy]`.

## Test Plan

- Lint regression: `cargo clippy --all-targets --all-features -- -D warnings` — enabling `clippy::undocumented_unsafe_blocks = "warn"` together with the workspace's `-D warnings` policy makes the project fail to compile if a future `unsafe` block is added without a `// SAFETY:` comment. This replaces ad-hoc grep checks with a first-class compiler-enforced gate.
- Library regression: `cargo test --lib` — all 997 tests still pass; the targeted `env_var_override_changes_effective_floor` test that was missing SAFETY comments runs green.
