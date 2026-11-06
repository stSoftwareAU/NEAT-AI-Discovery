## Summary

Added tracing-event tests for the deadline breakdown's completed, starved and disabled-phase cases. They pin the emitted levels, messages, marker and numeric field contract at the production logging boundary. The inspected orchestration caller supplies timings and completion counts, with no credential-bearing input on this path. Closes #2549.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The finding names secret redaction, but this caller only supplies elapsed milliseconds and phase completion counts. Tests now exercise the real `emit()` events and constrain the data that reaches each logging sink without claiming that numeric-only inputs demonstrate redaction.

### Essential Design Decisions

- INFO events carry the stable marker and numeric timing/completion fields; generated summary text uses `n/a` for disabled phases. WARN events carry the marker and the generated starvation message.
- No redactor was added because this path has no credential input to redact.

### Undiscoverable Facts

- `src/analysis/orchestration.rs` builds the breakdown from `parquet_reload_ms`, dispatch durations and phase completion metadata immediately before calling `emit()`.

## Evidence

Backend-only change; no UI or outbound request body is involved. Event capture tests invoke the real `DeadlineConsumptionBreakdown::emit()` method.

## Test Plan

- Added `tests/deadline_breakdown_test.rs::emit_logs_completed_breakdown_with_numeric_fields_and_summary`, `::emit_logs_starved_breakdown_and_warning`, and `::emit_logs_disabled_phase_as_not_applicable_without_warning`.
- Targeted test command: `cargo test --test deadline_breakdown_test --all-features -- --test-threads=2` — passed, 11 tests.
- Targeted lint: `cargo clippy --test deadline_breakdown_test --all-features -- -D warnings` — passed.
- INFO sink mutation: temporarily added `credential = "synthetic-secret-for-test"`; `emit_logs_completed_breakdown_with_numeric_fields_and_summary` failed on the exact captured field set. Removed the injection.
- WARN sink mutation: temporarily appended `synthetic-secret-for-test` to the warning; `emit_logs_starved_breakdown_and_warning` failed on the exact captured message. Removed the injection.
- Full gate: `./quality.sh < /dev/null` — passed, including formatting, Clippy, type checks, 1,852 library tests, all integration tests, docs and release build.
- No assertions were removed or changed in existing tests.

## Not checked

- Credential redaction in other loggers or outbound bodies is outside this caller and was not tested here. This change establishes no redaction guarantee for those paths.
