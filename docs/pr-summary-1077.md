## Summary

Fix discovery lockup where the host process waits forever for analysis to
complete after an internal panic. Closes #1077.

The root cause: `mark_analysis_started()` and `mark_analysis_finished()` were
called as a manual pair around each analysis invocation. If the analysis
panicked (e.g., a rayon thread panic propagating through `rayon::join`),
`mark_analysis_finished()` was never called. The host process then waited
indefinitely for `is_analysis_active()` to return false, causing the
"discovery locked up (never completed)" symptom.

The fix introduces an `AnalysisActiveGuard` RAII struct that increments the
counter on creation and decrements it on `Drop` — which runs even during
stack unwinding from a panic. Both `analyze_parallel_internal` and
`rank_focus_neurons_internal` now use this guard instead of manual calls.

## Evidence

- `test_analysis_active_guard_decrements_on_panic` verifies the counter is
  decremented even when the guarded scope panics
- All 739+ existing tests continue to pass
- `./quality.sh` passes cleanly (clippy, fmt, tests, docs, release build)

## Test Plan

- Added `test_analysis_active_guard_lifecycle` — normal creation and drop
- Added `test_analysis_active_guard_decrements_on_panic` — panic safety
- Added `test_analysis_active_guard_multiple_guards` — nested guard correctness
