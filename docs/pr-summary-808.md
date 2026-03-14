## Summary

Reduce String cloning in synapse preparation and candidate generation hot paths
by replacing owned `String` values with `&str` references tied to input lifetimes.
Addresses #808.

### Changes

**Hot path optimisations (per target, per source):**
- `build_samples_for_locality_group` now returns `&str` references instead of
  cloned `String` UUIDs, eliminating one allocation per source per target
- `SourceWorkResult.source_uuid` changed from `String` to `&str`, avoiding
  the double-clone pattern (clone in locality group + clone for HelpfulWork)
- `PreparedHarmfulWork` uses `&str` for `from_uuid`/`to_uuid` instead of
  cloning from `SynapseJson`
- `NeuronWorkResult.source_uuid` changed from `String` to `&str` in the
  neuron analysis path
- Eliminated intermediate `diagnostics_updates: Vec<(String, String, ...)>`
  allocation by updating diagnostics inline

**One-time setup optimisations (per analysis call):**
- `CreatureLookups.neuron_squash_map` changed from `HashMap<String, String>`
  to `HashMap<&str, &str>`, borrowing directly from `NeuronJson` fields
- `CreatureLookups.neuron_bias_map` changed from `HashMap<String, f32>` to
  `HashMap<&str, f32>`
- `CreatureLookups.used_inputs` changed from `HashSet<String>` to `HashSet<&str>`
- `order_eligible_sources` made generic over hash set key type to support
  both `HashSet<String>` (neuron module) and `HashSet<&str>` (synapse module)

## Evidence

Benchmark results from `cargo bench --bench synapse_preparation`:

### Lookup map construction (one-time setup)

| Neurons | Owned (before) | Borrowed (after) | Improvement |
|---------|---------------|------------------|-------------|
| 50      | 4.72 us       | 2.36 us          | ~50% faster |
| 200     | 18.5 us       | 9.21 us          | ~50% faster |
| 500     | 50.6 us       | 27.5 us          | ~46% faster |

### Per-target UUID cloning (hot path)

| Size          | Double clone (before) | Borrow+clone (after) | Improvement |
|---------------|----------------------|---------------------|-------------|
| 50n x 20t     | 19.8 us              | 10.8 us             | ~45% faster |
| 200n x 50t    | 200.6 us             | 104.5 us            | ~48% faster |
| 500n x 100t   | 987.8 us             | 523.0 us            | ~47% faster |

The hot path improvement eliminates one String allocation per source neuron
per target neuron. For a 500-neuron creature with 100 focus targets, this
saves ~50,000 String allocations per analysis call.

## Test Plan

- All existing tests pass (verified via `quality.sh`)
- Added `synapse_preparation` benchmark suite (`benches/synapse_preparation.rs`)
- Updated `EXPECTED_BENCHMARKS` in `tests/issue_576_benchmark_regression_tracking.rs`
- Updated test turbofish annotations for generic `order_eligible_sources`
