## Summary

Implement incremental analysis that skips unchanged neurons between discovery runs, avoiding redundant GPU computation. Closes #490.

When a caller provides `previousNeuronFingerprints` from a prior run, the library computes structural fingerprints for each neuron (based on activation function, bias, incoming/outgoing synapse weights and sources) and only analyses neurons whose fingerprints have changed. Unchanged neurons are skipped entirely, including both the GPU synapse/neuron analysis and all discovery dispatch modules.

### Key design decisions

- **Fingerprint = topology hash**: The fingerprint captures the structural properties that determine analysis results — squash, bias, incoming synapses, and outgoing synapses. It does not include activation data from parquet, which is the data being analysed rather than the topology.
- **O(n + s) computation**: Fingerprints are computed in a single pass over neurons and synapses, making the overhead negligible compared to GPU analysis.
- **Conservative invalidation**: Any structural change (weight, bias, squash, added/removed synapse) invalidates the neuron's fingerprint. New neurons (no previous fingerprint) are always analysed. This ensures we never miss a change.
- **No changes to the calling interface**: The `previousNeuronFingerprints` field is optional and defaults to `None`. Existing callers see no change in behaviour. The library returns `neuronFingerprints` in the response for callers to store and pass back.
- **Early exit**: When all focus neurons are unchanged, the library skips parquet loading and GPU initialisation entirely.

### New JSON fields

**Input** (`AnalyzeParallelInput` / `AnalyzeAllInput`):
- `previousNeuronFingerprints`: Optional map of `{uuid → fingerprint}` from the previous run.

**Output** (`AnalyzeParallelOutput`):
- `neuronFingerprints`: Current fingerprints for all neurons — store and pass back on the next run.
- `fingerprintCacheHits`: Number of focus neurons skipped (unchanged).
- `fingerprintCacheMisses`: Number of focus neurons analysed (changed or new).

## Evidence

This is a backend/CLI change with no web interface. Evidence is provided by the test suite:

- 15 dedicated tests in `tests/issue_490_incremental_analysis.rs` covering all fingerprint scenarios
- All 1657 existing tests pass with no modifications to business logic
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- `tests/issue_490_incremental_analysis.rs` — 15 tests covering:
  - Fingerprint computation for all neurons
  - Fingerprint changes when synapse weight, activation function, or bias changes
  - Fingerprint changes when synapses are added or removed (incoming and outgoing)
  - Fingerprint stability when topology is unchanged
  - Filtering: unchanged neurons are skipped, changed neurons are included
  - New neurons (no previous fingerprint) are always analysed
  - Removed neurons correctly invalidate downstream fingerprints
  - Cache hit/miss/total counts are reported accurately
  - Empty previous fingerprints means all neurons are analysed (first run)
  - JSON round-trip serialisation/deserialisation of fingerprints
