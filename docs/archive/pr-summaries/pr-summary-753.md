## Summary

Normalise squash strings to ASCII uppercase at the deserialisation boundary
(`NeuronJson`) so that all downstream detection modules receive pre-normalised
strings. Removes 20 redundant `.to_ascii_uppercase()` allocations from 11
detection and recommendation modules. Closes #753.

## Changes

- **`src/ffi_types/mod.rs`**: Added `deserialize_with = "deserialise_squash"` to
  `NeuronJson.squash` — normalises to uppercase during serde deserialisation.
- **`src/analysis/orchestration.rs`**: Added fallback normalisation when building
  `(uuid, squash, bias)` tuples for programmatically constructed `NeuronJson`.
- **11 detection/recommendation modules**: Removed all `to_ascii_uppercase()`
  calls, replacing `match upper.as_str()` with direct `match squash`.
- **`src/analysis/recommendation/activation_recommendation.rs`**: Normalised
  HashMap keys to uppercase; simplified `get_activation_score` to direct lookup.

## Benchmark Results

| Benchmark | Before | After | Improvement |
|---|---|---|---|
| `saturation_detect_250_neurons` | 84.5 µs | 5.0 µs | **94% faster** |
| `unbounded_capping_250_neurons` | 27.1 µs | 2.8 µs | **90% faster** |
| `combined_detection_250_neurons` | 112.7 µs | 7.8 µs | **93% faster** |

## Evidence

This is a backend performance change with no visual output. Evidence is the
benchmark results above, plus all 561 existing tests passing.

## Test Plan

- Added `tests/issue_753_squash_normalisation.rs` — 5 tests verifying:
  - Mixed-case squash normalised to uppercase on deserialisation
  - Default squash remains `"IDENTITY"`
  - Whitespace squash normalises to empty string
  - Programmatic construction preserves value
  - Saturation detection works with pre-normalised squash
- Added `benches/squash_normalisation.rs` — criterion benchmark suite
- Updated `tests/issue_431_activation_recommendation.rs` — uppercase keys
- Updated `tests/issue_576_benchmark_regression_tracking.rs` — new bench suite
- Updated unit test in `activation_mismatch.rs` — uppercase input
