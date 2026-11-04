## Summary

Observation-utilisation `SetBias` candidates now carry the downstream neuron's current bias forward: the emitted value is `current_bias + bias_compensation` instead of the bare compensation, which threw away the learned bias. Closes #2496.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- `SetBias` is absolute (as `error_plateau.rs::compute_bias_adjustment` already treats it), and `docs/discoveries/observation-utilisation.md` documents `bias += −(effective_centre × weight)`. The code now matches that documented behaviour.

### Essential Design Decisions

- The neuron-bias index is built once from `creature.neurons` (first occurrence wins, like the synapse index) and adds no deadline checks, so the issue #2379 deadline-sequence tests are unchanged.
- A target uuid absent from `creature.neurons` is skipped: no absolute bias can be computed, and a made-up default of 0.0 would reintroduce the bug. Assumption, not stated in the issue.

### Undiscoverable Facts

- None.

## Evidence

Backend-only change; no UI. Tests in `src/analysis/detection/observation_utilisation.rs`: `test_set_bias_carries_current_bias_forward` (bias 0.1, weight 0.5, centre 0.6 → −0.2, not the bare −0.3), `test_set_bias_is_per_observation_for_a_shared_target`, `test_target_missing_from_neurons_is_skipped`.

Docs: read `docs/discoveries/observation-utilisation.md` end to end; updated the candidate table row and added the skip sentence. The worked example (H2 0.1 → −0.2, H7 0.0 → 0.18) and the H1 diagram already matched the fixed behaviour.

## Test Plan

- `cargo test --lib observation_utilisation`: 7 passed. `cargo test --test issue_2375_non_finite_operations`: 10 passed. `cargo test --test detection 2379`: 2 passed. `cargo test --test detection 543`: 15 passed. `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- Red check: emitting the bare `bias_compensation` again turns `test_set_bias_carries_current_bias_forward` and `test_set_bias_is_per_observation_for_a_shared_target` red (5 passed, 2 failed); restored. Their fixtures use a non-zero target bias, so the two candidate values differ. `test_target_missing_from_neurons_is_skipped` stays green under that revert by design; it guards the skip branch (a `unwrap_or(0.0)` fallback would turn it red).
- Existing tests edited: none. Existing fixtures all use target bias 0.0, so their expected values are unchanged.
- Callers of `observation_utilisation_to_coordinated_candidates`: `scoring_specs.rs` (via the deadline variant), `benches/issue_2379_synapse_rescans.rs`, and the tests above; all go through the same function and receive the fix.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 560 s foreground cap and was terminated; fmt, clippy and the targeted tests were run separately; CI runs the full gate" -->

## Not checked

- Full `./quality.sh` did not complete locally (see note above).
- No Spec/Standards reviewer sub-agents were dispatched: the issue has no acceptance-criteria section.
