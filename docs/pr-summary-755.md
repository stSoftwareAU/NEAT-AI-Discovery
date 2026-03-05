## Summary

Reduce unnecessary allocations in detection modules by switching BFS visited
sets from `HashSet<String>` to `HashSet<&str>`, adding `Vec::with_capacity`
hints across all 30+ detection functions, and switching `HashSet<(String, String)>`
to `HashSet<(&str, &str)>` for synapse deduplication lookups. Closes #755.

## Changes

### 1. BFS visited set: `HashSet<String>` → `HashSet<&str>` (dead_neuron.rs)

The `find_connected_outputs_cached` BFS function allocated a new `String` for
every node visited. Now borrows `&str` slices from the topology cache, with
pre-allocated capacity.

### 2. Synapse deduplication: `HashSet<(String, String)>` → `HashSet<(&str, &str)>` (skip_connection.rs)

The `skip_connections_to_coordinated_candidates` function cloned two `String`s
per synapse and again per lookup. Now borrows `&str` slices from the creature.

### 3. `Vec::with_capacity` hints (30 detection functions across 30 files)

Every `detect_*` function's main candidate accumulator now uses
`Vec::with_capacity(upper_bound)` instead of `Vec::new()`, avoiding
reallocation during the push loop. The upper bound is derived from the
input collection being iterated (neurons, synapses, or records).

### 4. `.clone()` on Copy types

Clippy's `cloned_instead_of_copied` lint found no violations — `HelpfulSample`
is `Copy` but no iterator chains call `.cloned()` on it.

## Benchmark Results

**BFS visited set** (depth 50, fanout 5):

| Pattern | Time | Speedup |
|---------|------|---------|
| `HashSet<String>` (before) | 61.3 µs | — |
| `HashSet<&str>` (after) | 24.5 µs | **2.5×** |

**Dead neuron detection (detect + convert)**:

| Network | Before | After | Speedup |
|---------|--------|-------|---------|
| depth 10, fan 3 | 79.5 µs | 56.2 µs | **29%** |
| depth 20, fan 3 | 249.4 µs | 160.6 µs | **36%** |

## Test Plan

- All existing tests pass (verified via `quality.sh`)
- No test modifications required — all changes are internal allocation strategy
- New benchmark suite `benches/bfs_allocation.rs` added for regression tracking
- Updated `EXPECTED_BENCHMARKS` in `tests/issue_576_benchmark_regression_tracking.rs`
