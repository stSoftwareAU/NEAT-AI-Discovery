## Summary

The fan-in, multi-hop relay and topology-diversification converters no longer emit an `AddNeuron` whose UUID names a neuron the creature already has. Each collects `existing_neuron_uuids(creature)` once per call and picks the UUID with `first_free_neuron_uuid`; salt 0 is the unchanged legacy hash UUID (`fan-in-…`, `mh-…`, `td-…`), later salts append `-1`, `-2`, …. Closes #2499.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Same CWE-694 defect class as #2359 and #2412, at the three sites those issues did not cover. A colliding `AddNeuron` costs an evaluation slot and a module success-rate sample, or yields a duplicate UUID that `validate_forward_only_synapses` rejects.
- Reuses the existing helpers in `src/analysis/detection/helpers.rs` and the `-{salt}` suffix convention of `sentinel_gating.rs`, so non-colliding output is byte-for-byte unchanged.

### Essential Design Decisions

- `topology_diversification_to_coordinated_candidates` still skips a candidate when the legacy `(td-uuid, output)` synapse exists. That means the candidate was already applied, and re-salting would add a fresh neuron on every pass.
- As in #2412, `existing` is built once per call and is not updated between candidates in the same call.

### Undiscoverable Facts

- None.

## Evidence

Library change; no UI. Changed call sites: `fan_in_to_coordinated_candidates` (via private `fan_in_to_single_coordinated`, which gained an `existing` parameter; its only caller is updated), `multi_hop_to_coordinated_candidates` (three-hop-or-longer branch only; the two-hop bypass adds no neuron) and `topology_diversification_to_coordinated_candidates`. Callers of these converters outside the diff pass the same arguments as before.

## Test Plan

New `tests/issue_2499_fan_in_multi_hop_topology_add_neuron_uuid_collision.rs`, per converter:

- `fan_in_uses_a_free_uuid_on_each_pass`, `multi_hop_uses_a_free_uuid_on_each_pass`, `topology_diversification_uses_a_free_uuid_on_each_pass`: apply the first pass's neurons, convert again, assert a non-empty second pass whose UUIDs name no existing neuron and whose synapses use the emitted UUID.
- `fan_in_caller_supplied_collisions_skip_multiple_salts`, `multi_hop_caller_supplied_collisions_skip_multiple_salts`, `topology_diversification_caller_supplied_collisions_skip_multiple_salts`: legacy, `-1` and `-2` occupied, so the result is `-3`, deterministic.
- `fan_in_legacy_uuid_is_preserved_when_free`, `multi_hop_legacy_uuid_is_preserved_when_free`, `topology_diversification_legacy_uuid_is_preserved_when_free`: unsalted prefix and repeatable.
- `topology_diversification_skips_candidate_already_applied`: legacy neuron plus its synapse present gives no `AddNeuron`.

Results on the final tree: the new target 10 passed; `issue_2412_detection_add_neuron_uuid_collision` 7 passed; `issue_2182_recommendation_core_non_finite_gain_ranking` 7 passed; `--test recommendation` 249 passed; `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.

Red checks (one converter's fix replaced by the legacy UUID at a time): fan-in, multi-hop and topology each turned that converter's two-pass and caller-supplied-collision tests red; the legacy-preserved and already-applied tests stay green by design. No existing test was edited.

Docs: `CHANGELOG.md` entry. No other manual describes these UUIDs.

Not done / risk: the full `./quality.sh` and `cargo deny` were not completed; no independent reviewers ran, because the issue states no acceptance criteria.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 590 s foreground cap while compiling (exit 124); fmt, clippy -D warnings and the targeted tests above were run; CI runs the full gate" -->
