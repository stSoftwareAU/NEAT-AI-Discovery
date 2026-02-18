# PR Summary — Issue #487: Reduce Unnecessary Clone Allocations in Hot Analysis Paths

## Problem

The analysis pipeline contained ~59 `clone()` calls across hot paths, many executing
per-focus-neuron or per-sample. Key offenders:

1. **Mutex drains** — `helpful_results.lock().clone()` deep-copies entire `Vec` of results
   after parallel analysis completes (4 sites in `synapse/mod.rs`)
2. **HashMap key construction** — `focus_neuron_type_map` cloned every neuron UUID and type
   string into owned `HashMap<String, String>` (1 site)
3. **NeuronStatsJson** — `.clone()` on an all-scalar struct that should be `Copy` (1 site
   in `target_analysis.rs`)
4. **String clones for squash lookup** — `n.squash.clone()` in bottleneck conversion when
   a `&str` borrow suffices (2 sites)
5. **Owned struct copies** — `IncomingInput { from_uuid: String }` cloned repeatedly in
   noisy-vs-trusted detection inner loop (3 sites in `structural_patterns.rs`)

## Changes

### 1. Mutex drain via `std::mem::take` (`synapse/mod.rs`)
Replaced 4 `Mutex::lock().clone()` calls with `std::mem::take()`, which swaps the
Vec/HashMap out in O(1) without allocation:
- `helpful_results`, `harmful_results`, `coordinated_structural_results`
- `error_values_for_distribution`

### 2. Borrow-based HashMap for `focus_neuron_type_map` (`synapse/mod.rs`, `deadline.rs`)
Changed from `HashMap<String, String>` (clones UUID+type per neuron) to
`HashMap<&str, &str>` (borrows from `input.creature.neurons`). Updated
`order_focus_targets` parameter to match.

### 3. `Copy` derive on `NeuronStatsJson` (`lib.rs`, `target_analysis.rs`)
All fields are scalar (`f32`/`u32`/`Option<f32>`), so `Copy` is appropriate. Eliminates
the `.clone()` call in the harmful-synapse loop.

### 4. Borrow squash in bottleneck conversion (`bottleneck.rs`)
Changed squash lookup from `.map(|n| n.squash.clone())` to `.map(|n| n.squash.as_str())`.
Pre-built the comment string before constructing operations, eliminating a second clone.

### 5. Lifetime-parameterised `IncomingInput` (`structural_patterns.rs`)
Changed `IncomingInput { from_uuid: String }` to `IncomingInput<'a> { from_uuid: &'a str }`
with `Copy` derive. Eliminates String allocation per incoming synapse and avoids
`noisy.clone()` / `trusted.clone()` in the inner loop.

### 6. Clippy fixes
- Removed `&` on `&str` arguments to `cache.get()` (needless borrow)
- Replaced `.clone()` on `Copy` type with direct assignment

## Files Changed

| File | Change |
|------|--------|
| `src/analysis/synapse/mod.rs` | Mutex drain, borrow-based type map |
| `src/analysis/synapse/structural_patterns.rs` | Lifetime-parameterised IncomingInput |
| `src/analysis/synapse/target_analysis.rs` | Remove `.clone()` on Copy type |
| `src/analysis/bottleneck.rs` | Borrow squash instead of clone |
| `src/analysis/utils/deadline.rs` | `order_focus_targets` accepts `HashMap<&str, &str>` |
| `src/lib.rs` | Derive `Copy` on `NeuronStatsJson` |
| `Cargo.toml` | New benchmark entry |
| `benches/clone_reduction.rs` | Criterion benchmark for regression tracking |
| `tests/issue_487_reduce_clone_allocations.rs` | 7 correctness tests |
| `tests/issue_468_target_type_prioritisation.rs` | Updated for `HashMap<&str, &str>` |

## Benchmark Results

### Bottleneck path (directly affected)
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| convert_5_bottlenecks | 9.19 µs | 8.89 µs | **-3.1%** |
| detect_20_bottlenecks | 23.26 µs | 22.47 µs | **-2.6%** |
| convert_20_bottlenecks | 37.65 µs | 36.86 µs | **-2.8%** |

### Restricted range / bounded range (no direct changes — control group)
Within noise threshold, confirming no regressions from shared-code changes.

## What Was NOT Changed (Intentional)

- **`apply_target_type_boost` signature** — public API, kept as `HashMap<String, String>`
- **`upsert_candidate` in `scoring.rs`** — the 3 String clones build the HashMap key tuple;
  eliminating them would require changing the key type (breaking internal API for minimal gain)
- **`restricted_range_to_coordinated_candidates`** — String clones construct owned
  `CoordinatedStructuralOpJson` fields; inherent to the data model
- **GPU ownership transfers** in `target_analysis.rs` — `Arc::clone()` and `input.clone()`
  are necessary for thread-safe GPU batch evaluation

## Test Results

All 608 tests pass (463 unit + 145 integration). `quality.sh` passes clean.
