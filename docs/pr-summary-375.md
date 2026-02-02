## Summary

Extract the generic discovery module dispatch pattern (DRY) from `analyze_all()` in `src/analysis/mod.rs`.

The function previously contained 9 nearly identical code blocks (~500 lines of boilerplate) for dispatching discovery detection modules (saturation, bottleneck, dead neuron, correlated error, multi-hop, oscillating neuron, dormant synapse, opposing synapse, and output bias drift). Each block repeated the same pipeline: watchdog beat → phase timer → record collection → detection → conversion → verbose logging → merge.

This PR introduces `src/analysis/discovery_dispatch.rs` with:
- `dispatch_discovery_module()` — a generic function handling all dispatch boilerplate
- `DiscoveryDispatchConfig` — configuration struct for module name and phase timer
- `DetectionResult` — standardised detection output (count + candidates)
- `collect_records_for_uuids()` and `collect_hidden_neuron_records()` — shared record collection helpers

The 9 dispatch blocks in `analyze_all()` are replaced with concise calls to `dispatch_discovery_module()`, each providing only module-specific logic via a closure. This reduces ~500 lines of boilerplate to ~200 lines of focused module-specific code.

Adding a new detection module now requires only a single `dispatch_discovery_module()` call with a closure — no boilerplate for watchdog beats, phase timers, verbose logging, or candidate merging.

Closes #375

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Added 5 unit tests in `src/analysis/discovery_dispatch.rs` (inline `#[cfg(test)]` module):
  - `test_dispatch_merges_candidates_into_synapse_result`
  - `test_dispatch_skips_when_closure_returns_none`
  - `test_dispatch_skips_when_candidates_empty`
  - `test_dispatch_respects_max_synapse_candidates_limit`
  - `test_multiple_dispatches_accumulate_candidates`
- Added 6 integration tests in `tests/issue_375_discovery_dispatch_pattern.rs`:
  - `test_dispatch_merges_candidates`
  - `test_dispatch_skips_on_none`
  - `test_dispatch_skips_on_empty_candidates`
  - `test_multiple_dispatches_accumulate`
  - `test_dispatch_respects_max_candidates_limit`
  - `test_dispatch_preserves_candidate_details`
- All 439 existing unit tests pass unchanged
- All existing integration tests pass unchanged
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
