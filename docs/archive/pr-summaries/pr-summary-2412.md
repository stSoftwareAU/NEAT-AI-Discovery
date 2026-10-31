## Summary

Made five detection converters salt deterministic `AddNeuron` UUIDs around
UUIDs already present in the creature, while preserving each legacy UUID when
free. Threshold conversion now receives topology from its production dispatch.
Closes #2412.

Version bump: `VERSION` updated for this change; the merged commit records the
final version.

## Spec

### Intent and Rationale

- Prevent repeated discovery passes or caller-supplied topology from producing an `AddNeuron` that reuses an existing neuron UUID.
- Reuse `existing_neuron_uuids` and `first_free_neuron_uuid`, already used by five detection converters.

### Essential Design Decisions

- Salt zero preserves the legacy identifier; subsequent candidates append a numeric salt.
- Each converter builds its existing UUID set once per call. Candidate scoring, ordering and topology operations remain unchanged.
- Threshold conversion takes a required `CreatureJson` so it can apply the same collision check.

### Undiscoverable Facts

- Graft tools were not exposed in this run; locations and callers were found by local source inspection.
- The correlated-error legacy identifier is the existing FNV-derived `cs-…` value and remains unchanged when free.

## Evidence

- Backend-only Rust library change; no visual surface changed.
- `tests/issue_2412_detection_add_neuron_uuid_collision.rs` covers two-pass collisions, caller-supplied occupied salts, deterministic output, legacy identifiers and synapse references.
- The threshold-dispatch test inserts the legacy dampening UUID into its creature and observes the salted UUID from the production scoring spec.
- Updated `docs/DISCOVERY_TYPES.md` sections: Monotonicity, Fan-in Polarity Conflict, Correlated Error, Sentinel Gating and Input Sensitivity. Updated converter API comments and the documented UUID-helper caller counts (5 to 10).

## Test Plan

- `cargo test --test issue_2412_detection_add_neuron_uuid_collision -- --test-threads=1`: passed, 7 tests.
- `cargo test --lib issue_2412_threshold_dispatch_passes_creature_topology_to_conversion -- --test-threads=1`: passed, 1 test.
- Existing detection tests passed: sentinel gating (10), fan-in polarity conflict (11), monotonicity (10), input sensitivity (14) and correlated error (13).
- Preserved coverage passed: fan-in conversion growth and insertion anchors (`issue_2350_fanin_polarity_squash_lookup_growth`, 2 tests), and prior synthetic UUID handling (`issue_2359_synthetic_add_neuron_uuid_collision`, 7 tests).
- Individual collision-handling reversions made each corresponding two-pass regression fail for sentinel gating, fan-in polarity conflict, monotonicity, threshold effects and correlated errors. Restoring each converter made the complete new target pass.
- `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings` and `cargo check --all-targets --all-features`: passed.
- `./quality.sh < /dev/null`: passed all quality checks, including 1,751 library tests (2 ignored), integration tests, documentation and release build.
- No existing test assertions were removed; the two threshold-converter test callers now supply a creature and retain their existing operation assertions.
