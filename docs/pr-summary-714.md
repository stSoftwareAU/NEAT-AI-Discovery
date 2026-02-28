## Summary

Refactored `analyze_synapses_with_cache_impl` from ~350 lines down to ~107 lines by extracting logical phases into well-named helper functions. Closes #714.

### Extracted Functions

1. **`preparation::build_creature_lookups()`** — Builds all HashMap/HashSet lookup structures from creature topology (ordered neurons, interned indices, existing synapses, squash/type/bias maps, input neuron tracking). Returns a `CreatureLookups` struct.

2. **`preparation::compute_constant_source_threshold_from_cache()`** — Samples input neuron records to compute average standard deviation and derives the dynamic constant-source effect threshold.

3. **`SharedResultCollectors`** struct with `merge_target_results()` method — Encapsulates all shared mutable state (Mutex-wrapped result vectors, atomic metadata flags) used during parallel target analysis. Replaces 10 separate parameters.

4. **`finalise_synapse_results()`** — Collects results from shared state after the parallel loop, applies post-processing (collapse hidden, impact discounting, sorting), and builds the final `AnalyzeSynapsesResult`.

### No Behavioural Changes

This is a pure refactoring — the same code runs in the same order with the same inputs and outputs.

## Evidence

- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
- All 536 unit tests pass
- No visual/UI changes — backend refactoring only

## Test Plan

- Added 5 unit tests for `build_creature_lookups()` in `preparation.rs`:
  - `test_build_creature_lookups_ordered_neurons` — verifies ordered neurons and order map
  - `test_build_creature_lookups_existing_synapses` — verifies existing synapse sets
  - `test_build_creature_lookups_neuron_maps` — verifies squash, type, input, used inputs, and bias maps
  - `test_build_creature_lookups_synapses_by_target` — verifies synapses grouped by target
  - `test_build_creature_lookups_empty_creature` — verifies behaviour with empty input
- All existing tests continue to pass unchanged
