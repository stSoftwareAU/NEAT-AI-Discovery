## Summary

Add bounded range discovery module (Issue #395) to detect neurons whose activations
are concentrated in a narrow sub-range with sentinel values (e.g., -1 or 0) at the
boundary. This addresses the problem where observations like "Debt-to-Equity" use -1
as a null indicator — multiplying -1 by a weight produces a large distorting signal.

The module uses gap-based bimodal distribution analysis to identify the sentinel
cluster and the meaningful active range. When detected, it recommends bias adjustments
to centre the active range around zero, pushing sentinel values into a neutral zone.

### Key design decisions

- **Gap-based detection**: Sorts activations and finds the largest gap to separate
  sentinel values from the active range. This is robust to different sentinel values
  (-1, 0, or any boundary value).
- **DRY**: Uses the existing `discovery_dispatch::run_discovery_module` pattern for
  watchdog, timing, logging, and candidate merging.
- **Reuses existing candidate types**: Emits `SetBias` operations via
  `CoordinatedStructuralCandidateJson` — no new candidate types needed in NEAT-AI.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Added 10 integration tests in `tests/issue_395_bounded_range_detection.rs`:
  1. Detects narrow range with sentinel values at -1
  2. Full-range neuron is NOT flagged
  3. Insufficient samples are rejected
  4. Detects zero-sentinel with positive active range
  5. Mixed neurons — only bounded-range one detected
  6. Coordinated candidate conversion produces valid operations
  7. Bimodal distribution with sentinel cluster detected
  8. No records does not panic or flag
  9. Constant activation is NOT flagged (handled by dead neuron detection)
  10. Sentinel fraction is correctly computed
