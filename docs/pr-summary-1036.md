## Summary

Reduce UUID string cloning in analysis hot paths by using `Arc<str>` shared keys
for neuron preparation maps and eliminating double-cloning in cache record-loading
helpers. Closes #1036.

### Changes

**`preparation.rs` — shared `Arc<str>` map keys (Issue #1036)**

The three lookup maps (`neuron_squash_map`, `neuron_type_map`, `order_map`) previously
cloned every neuron UUID string independently into each map. Now a single `Arc<str>`
allocation is shared across all three maps via cheap reference-count increments.

**`cache/mod.rs` — eliminate double-cloning in record loaders**

`load_records_for_all_neurons`, `load_records_for_neuron_types`, and
`load_records_for_synapse_sources` previously collected UUIDs into a temporary
`Vec<String>` (first clone) then delegated to `load_records_for_uuids` which cloned
them again (second clone). These methods now inline the record-loading logic so each
UUID is cloned only once for the output tuple.

**Consumer type updates**

Updated `evaluation.rs`, `post_processing.rs`, `focus_filter.rs`, and `mod.rs` to
accept `HashMap<Arc<str>, V>` map types. All `.get()` calls use `&str` keys which
work transparently with `Arc<str>` via `Borrow<str>`.

## Evidence

Benchmark results (`cargo bench --bench uuid_arc_preparation`):

| Neuron count | `Arc<str>` (new) | `String` clone (old) | Improvement |
|---|---|---|---|
| 100 | 28 µs | 25 µs | ~-12% (Arc overhead at small scale) |
| 500 | 121 µs | 222 µs | **~45% faster** |
| 2000 | 908 µs | 1040 µs | **~13% faster** |

The `Arc<str>` approach shows clear improvement at realistic neuron counts (500+)
where the cost of shared reference counting is amortised across three maps.

## Test Plan

- Added `test_shared_uuid_map_lookup` — verifies `HashMap<Arc<str>, V>` supports
  `&str` lookups used in analysis hot paths
- Added `test_shared_uuid_arc_reuse` — verifies `Arc::ptr_eq` confirms key sharing
  across maps
- Updated existing tests in `diagnostics_tests.rs` and `diagnostics/mod.rs` to use
  `Arc<str>` map keys
- All 700+ existing tests pass via `./quality.sh`
- New benchmark `uuid_arc_preparation` exercises both old and new map-building
  approaches
