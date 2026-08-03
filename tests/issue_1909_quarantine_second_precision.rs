//! Issue #1909: the quarantine window must be measured in seconds.
//!
//! `bump-deps.sh` used to floor-divide both `now` and `published` to whole
//! hours before subtracting. Truncating each side independently discards up
//! to 59m59s from `published` and credits the same to `now`, so the
//! advertised 24h window could expire after 23h00m01s.
//!
//! These tests drive the real shell helpers with epoch seconds and assert
//! the boundary behaviour. They live in the Rust suite (not just
//! `tests/bump_deps_test.sh`) so the standard `cargo test` CI gate blocks a
//! regression back to hour-floored arithmetic.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// 2025-06-01T00:00:00Z — an exact hour boundary, used as the base epoch.
const BASE: i64 = 1_748_736_000;
const HOUR: i64 = 3600;

fn script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bump-deps.sh")
}

/// Run `bump_deps::is_quarantine_expired NOW PUBLISHED HOURS` and report
/// whether the quarantine is considered expired (helper exit code 0).
fn quarantine_expired(now: i64, published: i64, window_hours: u32) -> bool {
    let status = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "source '{}' && bump_deps::is_quarantine_expired {now} {published} {window_hours}",
            script().display()
        ))
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .stdin(Stdio::null())
        .status()
        .expect("run bump-deps.sh helper");
    status.success()
}

#[test]
fn held_one_second_before_the_window_closes() {
    // Published 23h59m59s ago — inside a 24h window, must be held.
    let published = BASE;
    let now = BASE + 24 * HOUR - 1;
    assert!(
        !quarantine_expired(now, published, 24),
        "a package published 23h59m59s ago must stay quarantined with a 24h window"
    );
}

#[test]
fn released_exactly_on_the_window_boundary() {
    // Published exactly 24h00m00s ago — the window has elapsed.
    let published = BASE;
    let now = BASE + 24 * HOUR;
    assert!(
        quarantine_expired(now, published, 24),
        "a package published exactly 24h ago must be released"
    );
}

#[test]
fn worst_case_hour_straddle_is_still_held() {
    // The failure the hour-floored comparison produced: published at
    // HH:59:59, checked 23h00m01s later at an exact hour boundary. Flooring
    // each side gave `elapsed = 24` hours and released the package early.
    let published = BASE + 10 * HOUR + 3599; // 10:59:59
    let now = published + 23 * HOUR + 1; // exactly 34:00:00 → hour 485794
    assert_eq!(
        now % HOUR,
        0,
        "the straddle case must land on an hour boundary"
    );
    assert_eq!(
        now / HOUR - published / HOUR,
        24,
        "the hour-floored comparison must see a 24h gap for this to be a regression test"
    );
    assert!(
        !quarantine_expired(now, published, 24),
        "23h00m01s of real elapsed time must not satisfy a 24h window"
    );
}

#[test]
fn zero_hour_window_releases_immediately() {
    assert!(
        quarantine_expired(BASE, BASE, 0),
        "a zero-hour window (internal deps) must never quarantine"
    );
}

/// Run `bump_deps::plan_lock_quarantine` over a canned crates.io fixture and
/// return its stdout.
fn plan_lock_quarantine(published_iso: &str, now_epoch: i64) -> String {
    let fixtures = tempfile::tempdir().expect("fixture dir");
    std::fs::write(
        fixtures.path().join("quote-1.0.40.json"),
        format!(r#"{{"version":{{"num":"1.0.40","created_at":"{published_iso}"}}}}"#),
    )
    .expect("write fixture");

    let before = fixtures.path().join("before.tsv");
    let after = fixtures.path().join("after.tsv");
    std::fs::write(&before, "quote\t1.0.35\n").expect("write before");
    std::fs::write(&after, "quote\t1.0.40\n").expect("write after");

    let output = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "source '{}' && bump_deps::plan_lock_quarantine '{}' '{}' {now_epoch} 24",
            script().display(),
            before.display(),
            after.display()
        ))
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .env("BUMP_DEPS_TEST_FIXTURE", fixtures.path())
        .stdin(Stdio::null())
        .output()
        .expect("run plan_lock_quarantine");
    assert!(output.status.success(), "plan_lock_quarantine must exit 0");
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

#[test]
fn lockfile_planner_holds_the_worst_case_straddle() {
    // Published 2025-06-01T10:59:59Z, checked 23h00m01s later at 10:00:00Z
    // the next day — the hour-floored planner released this.
    let published = BASE + 10 * HOUR + 3599;
    let now = published + 23 * HOUR + 1;
    let plan = plan_lock_quarantine("2025-06-01T10:59:59.000000+00:00", now);
    assert!(
        plan.contains("revert\tquote\t1.0.35\t1.0.40\t23h"),
        "an in-quarantine lockfile bump must be reverted and reported in whole hours, got: {plan:?}"
    );
}

#[test]
fn lockfile_planner_releases_at_the_boundary() {
    // Published exactly 24h before the check — outside the window.
    let now = BASE + 10 * HOUR + 3599 + 24 * HOUR;
    let plan = plan_lock_quarantine("2025-06-01T10:59:59.000000+00:00", now);
    assert!(
        plan.is_empty(),
        "a bump published exactly 24h ago must not be gated, got: {plan:?}"
    );
}
