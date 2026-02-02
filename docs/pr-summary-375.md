## Summary

Extract the generic discovery module dispatch pattern from `analyze_all()` to eliminate ~495 lines of nearly identical boilerplate (Issue #375).

### What changed

- **New `DiscoveryModule` trait** (`src/analysis/discovery_dispatch/mod.rs`): Defines a standard interface for all discovery detection modules with `name()`, `phase_name()`, and `detect_and_convert()` methods.
- **Generic `dispatch_discovery_module()` function**: Handles the entire dispatch boilerplate — watchdog beats, phase timer, detection, verbose logging, and merging into the synapse result — in a single reusable function.
- **`dispatch_all_discovery_modules()`**: Iterates over all registered modules in sequence.
- **9 concrete module implementations** (`src/analysis/discovery_dispatch/modules.rs`): Each wraps its specific record collection, detection, and candidate conversion logic behind the trait.
- **`analyze_all()` refactored**: The 9 repeated 30–50 line blocks (lines 590–1086) are replaced by a single 12-line dispatch call.
- **Helper functions** `collect_records_for_uuids()` and `collect_records_for_hidden_neurons()` eliminate the repeated cache-lookup pattern.

### Line count impact

| File | Before | After | Delta |
|------|--------|-------|-------|
| `src/analysis/mod.rs` | 1,222 | 743 | -479 |
| `src/analysis/discovery_dispatch/mod.rs` | — | 157 | +157 |
| `src/analysis/discovery_dispatch/modules.rs` | — | 384 | +384 |
| `src/analysis/discovery_dispatch/tests.rs` | — | 233 | +233 |
| `src/analysis/cache.rs` | (existing) | +13 | +13 |

Net: ~495 lines of boilerplate removed from `mod.rs`. The new structured code is split across focused files following the Single Responsibility Principle.

### Adding a new detection module

Before: Copy ~40 lines of boilerplate in `analyze_all()`, adjust variable names.
After: Implement the `DiscoveryModule` trait (~25 lines) and add one line to `all_discovery_modules()`.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- 7 new unit tests in `src/analysis/discovery_dispatch/tests.rs`:
  - `dispatch_module_merges_candidates_into_synapse_result` — verifies candidates are merged
  - `dispatch_empty_module_leaves_synapse_result_unchanged` — verifies empty modules are no-ops
  - `dispatch_all_modules_runs_every_module` — verifies all modules are dispatched
  - `dispatch_respects_max_synapse_candidates` — verifies truncation limit
  - `capitalise_first_works_for_various_inputs` — verifies display helper
  - `collect_records_for_uuids_returns_entries_with_empty_records` — verifies cache helper
  - `collect_records_for_hidden_neurons_returns_entries_with_empty_records` — verifies cache helper
- All 441 existing unit tests pass unchanged
- All 97 integration tests pass unchanged
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
