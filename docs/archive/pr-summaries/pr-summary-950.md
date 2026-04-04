## Summary

Document and test that `neuron_uuid` / endpoint fields in the FFI boundary may
contain either RFC 4122 UUID strings (from creature exports) or stringified
integers (from TypeScript runtime `neuron.id` after normalisation). Callers
must not assume a single format. Closes #950.

**Changes:**

- Added doc comments to `NeuronJson::uuid`, `SynapseJson::from_uuid`/`to_uuid`,
  `NeuronData::neuron_uuid`, `DiscoverRecord::neuron_uuid`,
  `ReadDiscoveryInput::neuron_uuid`, and `NeuronIndex` clarifying that both
  RFC 4122 UUIDs and stringified integers are valid neuron identity formats.
- Added 12 integration tests covering the full pipeline with numeric neuron IDs.

## Evidence

All 12 new tests pass exercising:
- FFI deserialisation of creatures with purely numeric and mixed-format IDs
- Record-and-read round-trip through Parquet with numeric neuron IDs
- Neuron interning of numeric strings (opaque, no format conversion)
- FFI entry points (`record_discovery_internal`, `read_discovery_records`)
  with numeric neuron identifiers

## Test Plan

- `tests/ffi/issue_950_numeric_neuron_ids.rs` — 12 tests:
  - `deserialise_creature_with_numeric_neuron_ids`
  - `deserialise_creature_with_mixed_uuid_formats`
  - `deserialise_neuron_data_with_numeric_ids`
  - `deserialise_large_numeric_neuron_ids`
  - `record_and_read_with_numeric_neuron_ids`
  - `record_and_read_with_mixed_id_formats`
  - `intern_numeric_neuron_ids`
  - `intern_mixed_format_ids_are_distinct`
  - `discover_record_with_numeric_neuron_id`
  - `discover_record_preserves_id_format`
  - `ffi_record_discovery_with_numeric_ids`
  - `ffi_read_discovery_with_numeric_neuron_id`
