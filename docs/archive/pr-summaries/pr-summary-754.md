## Summary

Pre-compute shared creature topology cache for detection modules, eliminating redundant `HashMap` / `HashSet` construction across 30+ modules per `analyze_all` call. Closes #754.

### What changed

- Created `CreatureTopologyCache` struct (`src/analysis/detection/topology_cache.rs`) containing pre-computed topology maps: `fan_in`, `fan_out`, `hidden_uuids`, `output_uuids`, `input_uuids`, `existing_synapses`, and `synapse_weights`.
- The cache is built once per `dispatch_and_merge_discovery_modules` call and shared via `Arc` across all parallel detection closures.
- Updated `dead_neuron`, `bottleneck`, and `topology` detection modules to accept an optional `&CreatureTopologyCache` parameter. When provided, they skip local topology construction; when `None`, they build locally for backward compatibility.
- Updated dispatch spec builders to pass the shared cache to these modules.

### Modules updated

| Module | Maps previously rebuilt per call |
|--------|-------------------------------|
| `dead_neuron.rs` | `hidden_uuids`, `output_uuids`, `fan_out_map` |
| `bottleneck.rs` | `fan_in_map`, `fan_out_map`, `hidden_uuids`, `synapse_weights`, `existing_synapses` |
| `topology.rs` | `fan_in_map`, `fan_out_map`, `hidden_uuids`, `output_uuids`, `existing_synapses` |

## Evidence

### Benchmark results (topology_cache)

Simulates 30 modules building topology maps versus 1 shared cache construction:

| Creature size | Per-module rebuild (30×) | Shared cache (1×) | Speedup |
|--------------|------------------------|--------------------|---------|
| 50 neurons | 302 µs | 21.7 µs | **13.9×** |
| 100 neurons | 611 µs | 45.4 µs | **13.5×** |
| 200 neurons | 1,406 µs | 97.3 µs | **14.4×** |
| 500 neurons | 3,658 µs | 247 µs | **14.8×** |

The shared cache approach is consistently ~14× faster than per-module rebuilding across all creature sizes.

## Test Plan

- Added 4 unit tests in `topology_cache.rs` (neuron classification, fan-in/fan-out, synapse lookups, synapse count)
- Added 6 integration tests in `tests/issue_754_topology_cache.rs` verifying cache vs no-cache equivalence for:
  - `detect_dead_neurons` with/without cache
  - `detect_bottleneck_neurons` with/without cache
  - `bottleneck_neurons_to_coordinated_candidates` with/without cache
  - `detect_topology_issues` with/without cache
  - `topology_issues_to_coordinated_candidates` with/without cache
  - Cache construction correctness
- All 67 existing tests for dead_neuron, bottleneck, and topology modules continue to pass
- `quality.sh` passes cleanly
