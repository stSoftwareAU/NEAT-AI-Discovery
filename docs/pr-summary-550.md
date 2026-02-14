## Summary

Add weight magnitude reset detection for stuck synapses to escape local minima. When a synapse's target neuron shows persistently high error with low variance (a plateau in the error landscape) and the source neuron is actively contributing signal, the module generates multiple exploratory `setWeight` candidates with dramatically different weights — sign flips, zero resets, and scaled magnitudes — to probe beyond the local basin. Closes #550.

This is part of the broader local minimum escape strategy (#545). The approach is analogous to simulated annealing: occasionally making large, seemingly worse moves to escape local minima, relying on NEAT-AI's validation framework to only accept changes that actually improve the score.

## Evidence

This is a backend detection module with no UI component. Evidence is provided through comprehensive test coverage:

- `tests/issue_550_weight_magnitude_reset.rs` — 7 integration tests verifying:
  - Detection identifies stuck synapses with high error contribution
  - Generates `setWeight` candidates with exploratory weight values
  - Weight candidates span a wide range (sign flip, magnitude change)
  - No candidates for healthy (low-error) synapses
  - No candidates with insufficient samples
  - Multiple stuck synapses produce ranked candidates
  - Comment references Issue #550

All tests pass, and `quality.sh` passes cleanly (fmt, clippy, check, test, release build).

## Test Plan

- Added `tests/issue_550_weight_magnitude_reset.rs` with 7 tests
- All 502 unit tests + 97 integration test files pass with `--test-threads=1`
- Full `quality.sh` gate passes
