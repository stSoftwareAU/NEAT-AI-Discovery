## Summary

Raised `NEURON_MIN_IMPROVED_RATIO` from `0.4` to `0.55` in
`src/analysis/constants/detection_thresholds.rs`. GRQ-sampler production failure
data (commit [50a2909](https://github.com/stSoftwareAU/GRQ-sampler/commit/50a2909eb04ab912804979d7e056e96350e1d8df))
showed neuron candidates with improved ratios of 52–54% (267–274 out of 510)
consistently produced negative actual error reductions despite positive
predictions. The previous `0.4` threshold let near-random candidates through,
wasting evaluation budget; raising to `0.55` filters these while still staying
below the synapse threshold (`MIN_IMPROVED_RATIO = 0.6`) because neuron
candidates are inherently noisier (two new connections rather than one).

Closes #1109.

## Changes

- `src/analysis/constants/detection_thresholds.rs`: bumped
  `NEURON_MIN_IMPROVED_RATIO` to `0.55` and documented the Issue #1109
  rationale with the production evidence.
- `tests/analysis/issue_938_constants_submodule_organisation.rs`: updated the
  accessibility assertion to the new value.
- `tests/analysis/issue_1109_neuron_min_improved_ratio_raise.rs`: new
  regression tests covering the raised threshold, the 52–54% production
  failure pattern rejection, the 55% boundary acceptance, and the invariant
  that the neuron threshold remains below the synapse threshold.
- `tests/analysis/main.rs`: registered the new test module.
- `Cargo.toml`: bumped patch version `0.74.2` → `0.74.3` (remote machines cache
  the library by version).

## Evidence

CLI/library change only — no UI. `./quality.sh` passes cleanly, including:

- `cargo fmt --all`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --lib --tests --all-features -- --test-threads=2`
- release build

`passes_neuron_improved_ratio()` in `src/analysis/neuron/evaluation.rs` uses the
constant directly and required no code changes — the function's `>=` comparison
still works correctly with the new value, and no temperature scaling is applied
at this filter (temperature scheduling in Issue #1020 operates on acceptance
probabilities downstream, not on this gating threshold).

## Test Plan

- [x] `cargo test --lib --tests --all-features -- --test-threads=2` (via
  `./quality.sh`)
- [x] New tests in `tests/analysis/issue_1109_neuron_min_improved_ratio_raise.rs`
  verify the 0.55 value, rejection of 52–54% production failure ratios,
  acceptance at exactly 55%, and the neuron-below-synapse invariant.
- [x] Existing constant assertion in
  `tests/analysis/issue_938_constants_submodule_organisation.rs` updated.
- [x] Existing Issue #733 tests continue to pass (the "55% good / 20% bad"
  boundary checks remain valid under the new threshold).
