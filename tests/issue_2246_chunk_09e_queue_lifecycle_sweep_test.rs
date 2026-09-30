//! Contract tests for the chunk 9e-2 queue-lifecycle sweep (Issue #2246).
//!
//! This file pins what the #2246 slice writes into
//! `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`'s `queue-lifecycle`
//! region — that `staleness.rs`, `heartbeat.rs`, `inflight.rs` and
//! `stale_skip_tests.rs` carry a verdict rather than `pending`, that the
//! five sub-sections and the outcome line are present, and that every test
//! the audit's "Covered by" notes name is actually declared where it says.
//! It also pins one behavioural verdict directly: a missed heartbeat trips
//! the breaker with `HeartbeatStall` and returns a typed `GpuWedged` `Err`.

use std::path::PathBuf;
use std::time::Duration;

/// The chunk 9 prose record.
const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";

/// The four files the #2246 slice sweeps, as `## Files swept` cites them.
const SWEPT_FILES: [&str; 4] = [
    "src/analysis/gpu/queue/staleness.rs",
    "src/analysis/gpu/heartbeat.rs",
    "src/analysis/gpu/inflight.rs",
    "src/analysis/gpu/queue/stale_skip_tests.rs",
];

/// The `#### ` sub-section headings the #2246 slice must carry.
const SUBSECTIONS: [&str; 5] = [
    "#### `src/analysis/gpu/queue/staleness.rs` (Issue #2246)",
    "#### `src/analysis/gpu/heartbeat.rs` (Issue #2246)",
    "#### `src/analysis/gpu/inflight.rs` (Issue #2246)",
    "#### Leaked-thread and wedge lifecycle (Issue #2246)",
    "#### `src/analysis/gpu/queue/stale_skip_tests.rs` (Issue #2246)",
];

/// Every test named in the three "Covered by" notes, except the
/// behavioural test declared in this file.
const COVERAGE: &[(&str, &str)] = &[
    (
        "src/analysis/gpu/queue/staleness.rs",
        "live_caller_with_unbounded_budget_is_not_stale",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "dropped_caller_guard_is_stale",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "expired_budget_is_stale_even_with_live_caller",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "unexpired_budget_with_live_caller_is_not_stale",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "dropped_caller_wins_over_expired_budget",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "shutdown_is_never_stale",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "dropped_caller_detected_for_every_variant",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "expired_budget_detected_for_every_variant",
    ),
    (
        "src/analysis/gpu/queue/staleness.rs",
        "stale_reason_labels_are_distinct",
    ),
    (
        "src/analysis/gpu/queue/stale_skip_tests.rs",
        "stale_request_skipped_without_analysis",
    ),
    (
        "src/analysis/gpu/queue/stale_skip_tests.rs",
        "device_lost_retry_aborts_on_dead_receiver",
    ),
    (
        "src/analysis/gpu/queue/stale_skip_tests.rs",
        "device_lost_retry_still_runs_for_live_caller",
    ),
    (
        "src/analysis/gpu/queue/stale_skip_tests.rs",
        "stale_skip_counted_in_metrics",
    ),
    (
        "src/analysis/gpu/queue/stale_skip_tests.rs",
        "expired_budget_request_fails_loudly_without_analysis",
    ),
    (
        "tests/gpu/issue_1929_stale_request_skip.rs",
        "fresh_metrics_report_no_stale_skips",
    ),
    (
        "tests/gpu/issue_1929_stale_request_skip.rs",
        "stale_skips_accumulate_per_skipped_request",
    ),
    (
        "tests/gpu/issue_1929_stale_request_skip.rs",
        "stale_skips_do_not_inflate_batch_or_sample_counts",
    ),
    (
        "tests/gpu/issue_1929_stale_request_skip.rs",
        "global_metrics_expose_the_stale_skip_counter",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "an_abandoned_request_never_reaches_the_wedged_gpu",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "a_silent_gpu_is_declared_wedged_within_the_stall_window",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "a_device_that_answers_inside_the_window_is_not_flagged",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "a_slow_but_progressing_gpu_is_never_flagged_as_wedged",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "an_endlessly_progressing_gpu_still_ends_at_the_absolute_timeout",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "the_first_wedge_stops_every_later_submission",
    ),
    (
        "src/analysis/gpu/queue/wedge_tests.rs",
        "the_whole_wedge_sequence_fits_inside_a_simulated_run_budget",
    ),
    (
        "tests/issue_1935_wedged_gpu_harness.rs",
        "a_silent_gpu_trips_the_process_wide_breaker_within_the_stall_window",
    ),
    (
        "tests/issue_1935_wedged_gpu_harness.rs",
        "no_second_gpu_thread_is_spawned_after_the_first_wedge",
    ),
    (
        "tests/issue_1935_wedged_gpu_harness.rs",
        "analyze_all_returns_a_signalled_partial_result_after_a_wedge",
    ),
    (
        "tests/issue_1935_wedged_gpu_harness.rs",
        "the_whole_wedge_sequence_fits_inside_a_simulated_run_budget",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "beats_advance_the_counter_monotonically",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "the_device_step_helpers_publish_on_the_global_heartbeat",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "a_silent_heartbeat_is_reported_as_stalled_after_the_window",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "slow_progress_keeps_resetting_the_stall_clock",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "a_zero_window_disables_the_guard",
    ),
    (
        "src/analysis/gpu/heartbeat.rs",
        "the_poll_interval_stays_within_its_bounds",
    ),
    (
        "tests/issue_2096_config_user_facing_a.rs",
        "gpu_stall_window_clamps_disables_and_falls_back",
    ),
    (
        "tests/infrastructure/issue_717_config_env_vars.rs",
        "config_gpu_stall_window_default",
    ),
    (
        "tests/infrastructure/issue_717_config_env_vars.rs",
        "config_gpu_stall_window_custom_value",
    ),
    (
        "src/analysis/gpu/inflight.rs",
        "a_registered_request_is_listed_until_its_guard_drops",
    ),
    (
        "src/analysis/gpu/inflight.rs",
        "dropping_one_guard_leaves_its_sibling_registered",
    ),
    (
        "src/analysis/gpu/inflight.rs",
        "an_outstanding_request_reports_its_age",
    ),
    (
        "src/analysis/gpu/queue/submission.rs",
        "a_waiting_caller_is_listed_as_an_outstanding_request",
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The ATX heading depth of `line`, or `0` if it is not a heading.
///
/// Requires a space (or end of line) after the leading `#`s, so a paragraph
/// that merely starts with a `#issue-number` reference — as the
/// Leaked-thread-and-wedge-lifecycle sub-section's body does, at
/// "#2243 and #2244 hand this question ..." — is not mistaken for a heading.
fn heading_depth(line: &str) -> usize {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    if hashes == 0 {
        return 0;
    }
    match line.as_bytes().get(hashes) {
        None | Some(b' ' | b'\n' | b'\r') => hashes,
        _ => 0,
    }
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let mut offset = 0;
    let mut start = None;
    for line in doc.split_inclusive('\n') {
        if line.trim_end() == heading {
            start = Some(offset + line.len());
            break;
        }
        offset += line.len();
    }
    let start = start.unwrap_or_else(|| panic!("{RECORD} must carry the heading `{heading}`"));
    let level = heading_depth(heading);
    let mut end = doc.len();
    let mut cursor = start;
    for line in doc[start..].split_inclusive('\n') {
        let depth = heading_depth(line);
        if depth > 0 && depth <= level {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &doc[start..end]
}

/// Cells of every Markdown table row in `body` that starts with `prefix`.
fn prefixed_rows(body: &str, prefix: &str) -> Vec<Vec<String>> {
    body.lines()
        .map(str::trim)
        .filter(|line| line.starts_with(prefix))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
}

/// The queue-lifecycle region of `## Audit sections`: the text after
/// `<!-- section: queue-lifecycle -->` within that top-level section.
fn lifecycle_audit(doc: &str) -> &str {
    let audit = section(doc, "## Audit sections");
    let marker = "<!-- section: queue-lifecycle -->";
    let start = audit
        .find(marker)
        .unwrap_or_else(|| panic!("`## Audit sections` must carry `{marker}`"))
        + marker.len();
    &audit[start..]
}

#[test]
fn no_queue_lifecycle_row_for_the_2246_files_is_still_pending() {
    let doc = read(RECORD);
    let files = section(section(&doc, "## Files swept"), "### queue-lifecycle");
    for path in SWEPT_FILES {
        let rows = prefixed_rows(files, &format!("| `{path}` |"));
        assert_eq!(
            rows.len(),
            1,
            "{path} must have exactly one Files swept row"
        );
        let outcome = rows[0].last().expect("outcome cell");
        assert!(
            !outcome.contains("pending"),
            "{path}: the queue-lifecycle row must carry a verdict, not a `pending` outcome: {outcome}"
        );
        assert!(
            outcome.starts_with("audited, no finding")
                || outcome.starts_with("audited, test-only")
                || outcome.starts_with("finding filed — #"),
            "{path}: outcome must open with `audited, no finding`, `audited, test-only` or \
             `finding filed — #N`: {outcome}"
        );
    }
}

#[test]
fn the_queue_lifecycle_region_carries_the_2246_subsections_and_outcome() {
    let doc = read(RECORD);
    let audit = lifecycle_audit(&doc).to_string();
    for heading in SUBSECTIONS {
        // `section` panics with the heading when it is missing.
        assert!(
            !section(&audit, heading).trim().is_empty(),
            "`{heading}` must record a verdict"
        );
    }
    for heading in [SUBSECTIONS[0], SUBSECTIONS[1], SUBSECTIONS[2]] {
        assert!(
            section(&audit, heading).contains("**Covered by"),
            "`{heading}` must carry a `**Covered by` note"
        );
    }
    assert!(
        audit.contains("**Outcome (#2246): no new finding.**"),
        "the queue-lifecycle audit region must state the #2246 outcome explicitly"
    );
}

#[test]
fn every_test_the_coverage_map_names_is_declared_in_its_file() {
    let doc = read(RECORD);
    let audit = lifecycle_audit(&doc);
    for (file, name) in COVERAGE {
        assert!(
            audit.contains(name),
            "the queue-lifecycle audit must name `{name}` in a Covered-by note"
        );
        let source = read(file);
        assert!(
            source.contains(&format!("fn {name}(")),
            "{file} must declare `fn {name}(`"
        );
    }
}

#[test]
fn the_cwe_362_verdict_cites_every_per_request_channel_and_names_the_gap() {
    let doc = read(RECORD);
    let audit = lifecycle_audit(&doc);
    let staleness = section(
        audit,
        "#### `src/analysis/gpu/queue/staleness.rs` (Issue #2246)",
    );
    assert!(
        staleness.contains("submission.rs:202"),
        "the staleness sub-section must cite the first per-request channel site"
    );
    for site in [":263", ":325", ":390", ":449", ":524"] {
        assert!(
            staleness.contains(site),
            "the staleness sub-section must cite the per-request channel site `{site}`"
        );
    }
    assert!(
        staleness.contains("CWE-362"),
        "the staleness sub-section must name CWE-362"
    );
    assert!(
        staleness.contains(
            "Guard dropped after dequeue, during a successful or non-device-lost evaluation"
        ),
        "the staleness sub-section must record the outstanding gap"
    );
}

#[test]
fn the_stall_window_opt_out_links_the_class_issue_instead_of_refiling() {
    let doc = read(RECORD);
    let audit = lifecycle_audit(&doc);
    let heartbeat = section(audit, "#### `src/analysis/gpu/heartbeat.rs` (Issue #2246)");
    assert!(
        heartbeat.contains("#2213"),
        "the heartbeat sub-section must cite #2213"
    );
    assert!(
        heartbeat.contains("#2276"),
        "the heartbeat sub-section must cite #2276"
    );
    assert!(
        heartbeat.contains("Not re-filed"),
        "the heartbeat sub-section must record that the opt-out is not re-filed"
    );

    let ledger = section(&doc, "## Ledger");
    let marker = "<!-- section: queue-lifecycle -->";
    let start = ledger
        .find(marker)
        .unwrap_or_else(|| panic!("`## Ledger` must carry `{marker}`"))
        + marker.len();
    let ledger_region = &ledger[start..];
    assert!(
        !ledger_region.contains("2276"),
        "the queue-lifecycle ledger region must not re-file the #2276 class"
    );
}

#[test]
fn a_missed_heartbeat_trips_the_breaker_and_returns_a_typed_err() {
    use neat_ai_discovery::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
    use neat_ai_discovery::analysis::gpu::queue::submission::heartbeat_stall_error;
    use neat_ai_discovery::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};

    let breaker = GpuCircuitBreaker::new();
    assert_eq!(
        breaker.trip_reason(),
        None,
        "a fresh breaker must not be tripped"
    );

    let err = heartbeat_stall_error(
        &breaker,
        "helpful batch evaluation",
        Duration::from_secs(31),
        Duration::from_secs(30),
    );

    assert_eq!(
        breaker.trip_reason(),
        Some(GpuTripReason::HeartbeatStall),
        "a missed heartbeat must trip the breaker with HeartbeatStall"
    );
    assert_eq!(
        classify_anyhow_error(&err),
        DiscoveryErrorKind::GpuWedged,
        "a missed heartbeat must classify as a typed GpuWedged error"
    );
    assert!(
        breaker.check().is_err(),
        "a tripped breaker must refuse the next submission"
    );
}
