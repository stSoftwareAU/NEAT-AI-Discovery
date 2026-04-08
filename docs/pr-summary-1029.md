## Summary

Discovery module detection (48 parallel closures via rayon) had no deadline
awareness. When the analysis time budget was exhausted, modules would continue
running indefinitely, causing the process to exceed its `max-task-hours` limit
and get killed by the task controller.

This change threads the analysis deadline into `detect_discovery_modules_parallel`
so that each module checks `deadline_passed()` before executing its detection
closure. Modules that start after the deadline are skipped, and a shared
`AtomicBool` flag propagates the timeout signal across rayon worker threads to
minimise repeated syscalls.

Closes #1029.

## Changes

- `detect_discovery_modules_parallel` now accepts an `Option<SystemTime>` deadline
  parameter. When the deadline is reached, remaining modules are skipped.
- `prepare_and_detect_discovery_modules` accepts and forwards the deadline.
- `analyze_all` (orchestration) builds a deadline from `analysis_deadline_ms`
  and passes it to the discovery detection phase.
- Existing callers (`run_discovery_modules_parallel`, integration tests) pass
  `None` to preserve existing behaviour.

## Evidence

- Tests verify that modules are skipped when the deadline has already passed
- Tests verify that modules run normally with no deadline or a future deadline
- Tests verify that module metadata is preserved even when skipped

## Test Plan

- `parallel_detection_skips_modules_when_deadline_already_passed` — past deadline skips all closures
- `parallel_detection_runs_all_modules_when_no_deadline` — no deadline runs everything
- `parallel_detection_runs_all_modules_when_deadline_is_far_future` — future deadline runs everything
- `parallel_detection_preserves_module_metadata_when_skipped` — metadata intact on skip
- All existing tests pass (171 integration tests, all unit tests)
