## Summary

Disable the batch-successful (combo-successful) discovery module by default and
lower its improvement threshold for when it is re-enabled. Closes #1059.

The module had **zero production successes** across all 43 creatures in the
GRQ-sampler discovery cache. The root cause was `MIN_INDIVIDUAL_IMPROVEMENT =
0.01`, which is 3–5 orders of magnitude above actual production score deltas
(typically 1e-7 to 6e-6).

### Decision: Disable by default, rework threshold

- **Disabled by default** via `NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL` env var
  (defaults to `false`). This redirects compute budget to higher-success modules.
- **Threshold lowered** from 0.01 to 1e-5 so the module has a realistic chance
  of producing candidates when re-enabled for experimentation.
- **Code preserved** — the module is not removed, allowing future re-evaluation.

### Changes

| File | Change |
|------|--------|
| `src/config/user_facing.rs` | Added `batch_successful_enabled()` config accessor |
| `src/config/mod.rs` | Documented new env var |
| `src/analysis/recommendation/batch_successful/detection.rs` | Lowered `MIN_INDIVIDUAL_IMPROVEMENT` from 0.01 to 1e-5 |
| `src/analysis/module_dispatch_specs/scoring_specs.rs` | Gated batch-successful behind config check |
| `src/analysis/module_dispatch_specs/mod.rs` | Updated module count assertion (48 → 47) |
| `tests/recommendation/issue_965_batch_successful_grouping.rs` | Updated low-improvement test for zero-mean error |
| `tests/recommendation/issue_1059_disable_batch_successful.rs` | New test suite for disabled behaviour |
| `README.md` | Documented new env var |

## Evidence

- Zero production successes across 43 creatures with the old threshold (0.01)
- New threshold (1e-5) aligns with actual GRQ-sampler score deltas (1e-7 to 6e-6)
- All 750+ tests pass including 4 new tests and 14 existing batch-successful tests

## Test Plan

- `tests/recommendation/issue_1059_disable_batch_successful.rs`:
  - `batch_successful_disabled_by_default` — verifies module is off by default
  - `lowered_threshold_detects_small_improvements` — verifies detection with tiny signals
  - `strong_signal_still_detected_with_lowered_threshold` — regression test for strong signals
  - `grouping_works_with_small_improvements` — verifies batching with small improvement values
- Updated `test_no_candidates_for_low_improvement` to use zero-mean error (Issue #1059)
- Updated `test_build_discovery_module_specs_produces_all_modules` count (48 → 47)
