## Summary

Enforce UUID-only FFI discovery contract by rejecting purely numeric integer
neuron/synapse IDs at the FFI boundary. Internal numeric IDs are an optimisation
detail that must never leak into JSON, cache files, or cross-library payloads.
Closes #952.

### Changes

- **UUID validation**: Added `deserialise_neuron_uuid` and `deserialise_synapse_uuid`
  custom deserialisers to `NeuronJson::uuid`, `SynapseJson::from_uuid`/`to_uuid`,
  and `NeuronData::neuron_uuid`. These reject purely numeric strings (e.g. `"0"`,
  `"42"`, `"999999"`) with a `data_validation` error.
- **Schema version**: Added `schemaVersion` field to `RecordDiscoveryOutput`,
  `AnalyzeParallelOutput`, `RankFocusNeuronsOutput`, and `GetVersionOutput`.
  Current schema version is `"2"`. Callers can use this to reject stale cached
  payloads.
- **Wire format**: `RecordDiscoveryOutput` now uses `camelCase` field names
  (`tempDir`, `schemaVersion`, `errorKind`) consistent with other response types.
- **Documentation**: Updated `FFI_API.md` with the neuron identity contract and
  schema version field.
- **Dead code removal**: Removed comments and documentation referencing numeric ID
  acceptance (Issue #950 compatibility).

## Evidence

All 158 tests pass including:
- `tests/ffi/issue_952_uuid_only_ffi_contract.rs` — 11 new tests verifying UUID
  enforcement, numeric ID rejection, and schema version presence
- `tests/ffi/issue_950_numeric_neuron_ids.rs` — Rewritten to test UUID-format IDs
  through the full pipeline (deserialisation, recording, reading, interning)
- Updated existing tests in `tests/integration.rs`,
  `tests/analysis/issue_337_candidate_type_contract.rs`, and
  `tests/infrastructure/issue_874_module_split_backward_compat.rs`

## Test Plan

- `reject_creature_with_numeric_neuron_ids` — Verifies numeric neuron UUIDs are rejected
- `reject_synapse_with_numeric_from_uuid` — Verifies numeric synapse UUIDs are rejected
- `reject_neuron_data_with_numeric_id` — Verifies numeric `NeuronData` UUIDs are rejected
- `reject_large_numeric_neuron_ids` — Verifies large numeric strings are rejected
- `accept_rfc4122_uuids` — Verifies RFC 4122 UUIDs are accepted
- `accept_input_neuron_format` — Verifies `input-N` format is accepted
- `accept_neuron_data_with_uuid` — Verifies UUID `NeuronData` is accepted
- `ffi_record_discovery_rejects_numeric_ids` — Full FFI pipeline rejection test
- `record_discovery_response_includes_schema_version` — Schema version in responses
- `get_version_response_includes_schema_version` — Schema version in version output
- `coordinated_op_serialises_uuid_references` — Coordinated ops use UUID references
