## Summary

Add inline unit tests (`#[cfg(test)]`) to all 9 discovery detection modules, covering
detection criteria, exclusion criteria, edge cases, and coordinated candidate conversion.

Each module previously had zero inline unit tests — all testing was via integration tests
in `tests/`. These new unit tests exercise private helper functions and internal logic
that cannot be reached through the public API alone.

Closes #376.

## Modules and Test Counts

| Module | Tests Added | Coverage |
|--------|-------------|----------|
| `saturation.rs` | 12 | Bounded/unbounded squash, RELU dead zone, variance exclusion, conversion |
| `bottleneck.rs` | 9 | Fan-in/out ratio, output exclusion, topology scoring, conversion |
| `dead_neuron.rs` | 8 | Zero activation, output/input exclusion, connected outputs, conversion |
| `correlated_error.rs` | 7 | Correlated/independent errors, single output skip, conversion |
| `multi_hop.rs` | 8 | Two/three-hop detection, fully-connected exclusion, conversion |
| `oscillating_neuron.rs` | 11 | Alternation, sign balance, dead neuron exclusion, squash recommendation |
| `dormant_synapse.rs` | 7 | Near-zero weight, sole connection protection, conversion |
| `opposing_synapse.rs` | 8 | Correlation detection, hidden target exclusion, removal vs weight flip |
| `output_bias_drift.rs` | 8 | Positive/negative drift, hidden exclusion, noise threshold, conversion |

**Total: 78 new unit tests across 9 modules.**

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

All tests pass via `./quality.sh` (fmt, clippy, check, test, release build).

## Test Plan

- Each of the 9 detection modules has at least 7 unit tests (exceeds the 3-test minimum)
- Tests cover positive detection, negative detection (exclusion), and edge cases
- Tests verify outcomes (what is detected / excluded), not implementation details
- Conversion tests verify `*_to_coordinated_candidates()` produces correct operation types
- Internal helper tests (e.g., `is_bounded_squash`, `can_have_dead_zone`) verify classification logic
- Australian English used in test names and comments
- `./quality.sh` passes cleanly
