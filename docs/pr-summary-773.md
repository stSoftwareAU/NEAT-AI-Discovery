## Summary

Introduce `discovery_spec!` macro to eliminate repetitive boilerplate in the
discovery module spec builders (`neuron_specs.rs`, `synapse_specs.rs`,
`structural_specs.rs`, `scoring_specs.rs`). Closes #773.

The macro handles the common clone → guard → detect → convert → wrap pipeline
with four variants:
- **Standard** — optional pre-record `guard` for hidden-neuron emptiness
- **guard_records** — post-record emptiness check (for modules loading all/typed records)
- **guard_min** — minimum-count guard (e.g. `hidden.len() < 2` for pair detectors)
- **custom** — full closure for modules with non-standard logic

### Line reduction

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| `neuron_specs.rs` | 479 | 200 | 58% |
| `synapse_specs.rs` | 274 | 109 | 60% |
| `structural_specs.rs` | 233 | 127 | 45% |
| `scoring_specs.rs` | 331 | 171 | 48% |
| **Spec files total** | **1317** | **607** | **54%** |

New modules can now be added with a single `discovery_spec!()` invocation
instead of ~15 lines of boilerplate.

## Evidence

This is a pure refactoring with no visual or performance changes. All 43
discovery module specs are preserved with identical behaviour, verified by:
- Unit tests confirming correct spec count, unique phase names, and empty-guard
  short-circuiting
- Full `quality.sh` pass (fmt, clippy, check, all tests, doc build, release build)

## Test Plan

- Added `test_build_discovery_module_specs_produces_all_modules` — verifies 43 specs with non-empty names/phases
- Added `test_discovery_module_specs_have_unique_phase_names` — verifies no duplicate phase names
- Added `test_discovery_specs_with_empty_hidden_return_none` — verifies all specs return `None` with empty hidden neurons and no records
- All existing integration tests pass unchanged
