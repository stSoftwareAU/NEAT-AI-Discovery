## Summary

Further reduce String cloning in synapse preparation by eliminating two
remaining allocation sources: `SynapseJson` cloning in `synapses_by_target`
and `neuron_type` String cloning in `neuron_type_map`. Addresses #808.

This builds on PR #830 which addressed hot-path UUID cloning. These changes
target the one-time setup allocations that were left as future work.

### Changes

**`synapses_by_target` — store references instead of clones:**
- `CreatureLookups.synapses_by_target` changed from `HashMap<u32, Vec<SynapseJson>>`
  to `HashMap<u32, Vec<&'a SynapseJson>>`, borrowing directly from `input.creature.synapses`
- Eliminates cloning entire `SynapseJson` objects (2 String fields + Option<String> each)
- Updated `detect_noisy_vs_trusted` and `prepare_harmful_samples` parameter types
  to accept `&[&SynapseJson]` instead of `&[SynapseJson]`
- Updated `TargetAnalysisContext.synapses_by_target` type accordingly

**`neuron_type_map` — borrow type string values:**
- `CreatureLookups.neuron_type_map` changed from `HashMap<String, String>`
  to `HashMap<String, &'a str>`, borrowing type strings from `NeuronJson.neuron_type`
  and using `"input"` static str for input neurons
- Eliminates `neuron.neuron_type.clone()` and `"input".to_string()` allocations
- Updated all downstream comparison sites to dereference `&&str` values

## Evidence

Benchmark results from `cargo bench --bench synapse_preparation`:

### synapses_by_target construction (one-time setup)

| Neurons | Cloned (before) | Borrowed (after) | Improvement |
|---------|----------------|------------------|-------------|
| 50      | 4.84 µs        | 2.47 µs          | ~49% faster |
| 200     | 19.08 µs       | 9.68 µs          | ~49% faster |
| 500     | 54.69 µs       | 29.65 µs         | ~46% faster |

### neuron_type_map construction (one-time setup)

| Neurons | Owned (before) | Borrowed (after) | Improvement |
|---------|---------------|------------------|-------------|
| 50      | 5.21 µs       | 4.18 µs          | ~20% faster |
| 200     | 21.02 µs      | 17.21 µs         | ~18% faster |
| 500     | 64.12 µs      | 54.04 µs         | ~16% faster |

For a 500-neuron creature, the combined saving is ~25 µs per analysis call
from `synapses_by_target` and ~10 µs from `neuron_type_map`, eliminating
~1000 String allocations from synapse cloning and ~500 from type string cloning.

## Test Plan

- All existing tests pass (verified via `quality.sh`)
- Updated test assertions in `preparation.rs` for new `&str` value type
- Added `synapses_by_target` and `neuron_type_map` benchmark groups to
  `benches/synapse_preparation.rs`
