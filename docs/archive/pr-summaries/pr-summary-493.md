## Summary

Extract creature-aware record-loading helpers into `RecordCache` to eliminate UUID-collection boilerplate across 14 discovery module dispatch blocks in `analyze_all()`.

Three new methods added to `RecordCache` (in `cache.rs`):
- `load_records_for_all_neurons(&creature)` — replaces 8 blocks that collected all neuron UUIDs
- `load_records_for_neuron_types(&creature, &["output", "input"])` — replaces 5 blocks that filtered by neuron type
- `load_records_for_synapse_sources(&creature)` — replaces 1 block that collected unique synapse source UUIDs

Net effect: 70 lines of duplicated UUID-collection logic removed from `mod.rs`, replaced by single-line helper calls. Each dispatch block now reads the records in one call instead of building an intermediate UUID vector first.

## Evidence

This is a backend/refactoring change with no UI impact. Verified by:
- All 8 new integration tests pass (see test plan below)
- All 4 existing `issue_481_record_loading_helper` tests still pass
- Full `quality.sh` passes (fmt, clippy, check, 1000+ tests, release build)
- `mod.rs` reduced from ~1,588 to ~1,518 lines

## Test Plan

- Added `tests/issue_493_creature_record_loading.rs` with 8 tests:
  - `test_load_all_neurons_returns_records_for_every_neuron` — all neuron UUIDs loaded
  - `test_load_all_neurons_empty_creature` — empty creature returns empty
  - `test_load_by_type_filters_correctly` — single-type filtering (output, input)
  - `test_load_by_type_multiple_types` — multi-type filtering (input+output, input+hidden)
  - `test_load_by_type_no_matching_types` — no matches returns empty
  - `test_load_synapse_sources_returns_unique_sources` — deduplicates source UUIDs
  - `test_load_synapse_sources_empty_synapses` — empty synapses returns empty
  - `test_load_hidden_returns_correct_records` — existing `load_records_for_hidden` works correctly
