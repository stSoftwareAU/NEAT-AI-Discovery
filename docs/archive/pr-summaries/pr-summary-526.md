## Summary

Replace String-based HashMap keys with pre-computed FNV-1a hash keys in the `upsert_candidate` hot path, eliminating 3 String heap allocations per candidate insertion. Closes #526.

The `upsert_candidate()` function is called for every neuron candidate in the analysis pipeline. Previously, each call constructed a `(String, String, String, i8, i8)` tuple key by cloning `source_neuron_uuid`, `target_neuron_uuid`, and `squash` — three heap allocations per candidate. With hundreds or thousands of candidates per analysis pass, this was the highest-impact clone site in the hot path.

### Approach

1. **FNV-1a hash key** — `compute_candidate_dedup_key()` computes a `u64` hash from the same five key components (source UUID, target UUID, squash, incoming weight sign, outgoing weight sign) using FNV-1a with separator bytes to prevent cross-field collisions.

2. **HashMap type change** — `HashMap<(String, String, String, i8, i8), CandidateNeuronJson>` → `HashMap<u64, CandidateNeuronJson>`, eliminating all String clones for key construction.

3. **Audit findings** — The remaining `.clone()` calls in the hot path were found to be architecturally necessary:
   - `Vec<HelpfulSample>` clones for GPU work queue: required because the GPU thread takes ownership via crossbeam channel
   - `source_uuid` clone in target_analysis: one clone per source neuron, needed for ownership transfer
   - `SynapseJson` clones for `synapses_by_target`: one-time initialisation cost, not per-candidate

### Benchmark Results

| Scenario | String key (before) | Hash key (after) | Improvement |
|---|---|---|---|
| 100 candidates | 14.30 µs | 8.24 µs | **42.4% faster** |
| 1000 candidates | 169.25 µs | 97.64 µs | **42.3% faster** |

### Files changed

| File | Changes |
|------|---------|
| `src/analysis/synapse/scoring.rs` | Added `weight_sign()`, `compute_candidate_dedup_key()` (FNV-1a hash), updated `upsert_candidate()` to use `HashMap<u64, _>` |
| `src/analysis/synapse/mod.rs` | Updated re-exports for new functions |
| `src/analysis/neuron.rs` | Changed `helpful_map` type to `HashMap<u64, CandidateNeuronJson>` |
| `src/analysis/implementation_tests/mod.rs` | Added `clone_reduction_tests` module, updated common imports |
| `src/analysis/implementation_tests/clone_reduction_tests.rs` | 11 new tests for hash-based deduplication correctness |
| `src/analysis/implementation_tests/relu_evaluation_tests.rs` | Updated to use `HashMap<u64, _>` and `compute_candidate_dedup_key()` |
| `benches/upsert_candidate.rs` | New Criterion benchmark comparing String-key vs hash-key approaches |
| `Cargo.toml` | Added `[[bench]]` entry for `upsert_candidate` |

## Evidence

This is a backend/library change with no UI. Evidence is provided by benchmarks and the test suite:

- **42% throughput improvement** in `upsert_candidate` benchmark (14.30 µs → 8.24 µs for 100 candidates)
- 11 new clone_reduction_tests verify deduplication correctness
- 5 existing relu_evaluation_tests pass with updated map types
- All 489+ unit tests pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Added `src/analysis/implementation_tests/clone_reduction_tests.rs` with 11 tests covering:
  - Identical keys deduplicate to single entry (higher gain wins)
  - Different sources, targets, squash, incoming signs, outgoing signs all produce distinct keys
  - UUID concatenation collision prevention (separator byte verification)
  - 500-candidate stress test confirming zero hash collisions
  - Multiple updates to same key retain best gain
  - `compute_candidate_dedup_key` is deterministic and distinguishes all 5 components
- Updated existing `relu_evaluation_tests` to use new `HashMap<u64, _>` type
- All existing tests continue to pass unchanged
