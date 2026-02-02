## Summary

Extract generic discovery module dispatch pattern (DRY) — Issue #375.

The `analyze_all()` function contained 9 nearly identical code blocks (~500 lines)
for dispatching detection modules (saturation, bottleneck, dead neuron, correlated
error, multi-hop, oscillating neuron, dormant synapse, opposing synapse, output bias
drift). Each block repeated the same watchdog beats, phase timer creation, verbose
logging, and merge-into-synapse-result boilerplate.

This PR extracts the shared pattern into a generic `run_discovery_module()` function
in a new `discovery_dispatch.rs` module. Each detection module now supplies its
specific logic via a closure while the dispatch function handles all common concerns.

**Key changes:**
- New `src/analysis/discovery_dispatch.rs` with `run_discovery_module()` and
  `DiscoveryDetectionResult` type
- Refactored all 9 detection module dispatch blocks in `analyze_all()` to use the
  shared function
- Added local helper closures (`collect_records`, `collect_hidden_records`) to
  eliminate repeated cache-lookup boilerplate
- Net reduction of ~250 lines of duplicated code

**Adding a new detection module now requires:** one call to `run_discovery_module()`
with module-specific detection logic in a closure — no boilerplate for watchdog
beats, phase timers, verbose logging, or merge logic.

## Evidence

Unable to generate screenshot: this is a Rust library with no visual interface.

## Test Plan

- Added 6 unit tests in `src/analysis/discovery_dispatch_tests.rs`:
  - `run_discovery_module_merges_candidates_into_synapse_result` — verifies candidates are merged
  - `run_discovery_module_does_nothing_when_detect_returns_none` — verifies no-op on None
  - `run_discovery_module_does_nothing_when_candidates_empty` — verifies no-op on empty vec
  - `run_discovery_module_respects_max_synapse_candidates` — verifies truncation
  - `run_discovery_module_beats_watchdog_start_and_finish` — verifies watchdog lifecycle
  - `run_discovery_module_accumulates_across_multiple_calls` — verifies multi-module accumulation
- All 440 existing lib tests pass unchanged
- All integration tests pass unchanged
- `./quality.sh` passes cleanly
