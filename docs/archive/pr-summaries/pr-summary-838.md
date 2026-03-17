## Summary

Add concurrency stress tests for deadlock and contention detection in the
analysis pipeline. These tests exercise `analyze_all()` and `compute_impacts`
under high concurrency with parking_lot deadlock detection to identify
deadlocks, lock contention, and stalls. Closes #838.

## Evidence

All 5 stress tests pass within the 60-second threshold (total ~12s):
- Full pipeline with deadlock detection: completes without deadlocks
- 16 concurrent impact computations: deterministic results, no deadlocks
- 3 rapid repeated analyses: no resource exhaustion or deadlocks
- Tight deadline (2s) under contention: respects timeout, no stalls
- Deadlock detection verification: clean state after concurrent work

## Test Plan

- Added `tests/infrastructure/issue_832_deadlock_stress.rs` with 5 tests:
  - `stress_full_pipeline_no_deadlocks_under_high_concurrency` — full synapse + neuron analysis with background deadlock checker
  - `stress_impact_computation_concurrent_determinism` — 16-thread parallel impact computation verifying determinism
  - `stress_rapid_repeated_analyses_no_resource_exhaustion` — 3 rapid sequential analyses
  - `stress_tight_deadline_respects_timeout_without_deadlock` — tight 2s deadline exercise
  - `stress_deadlock_detection_reports_clean_state_after_concurrent_work` — verifies `parking_lot::deadlock::check_deadlock()` mechanism
- Registered module in `tests/infrastructure/main.rs`
- All tests use seeded RNG for reproducibility
- `quality.sh` passes cleanly
