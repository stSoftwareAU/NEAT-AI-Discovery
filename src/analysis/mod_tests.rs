use super::*;

#[test]
fn watchdog_beats_do_not_claim_finished_when_analysis_is_skipped() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    // Ensure a watchdog is active so `beat()` is observable in tests.
    let wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let skipped = "analysis::analyze_all → neuron analysis skipped";
    let finished = "analysis::analyze_all → neuron analysis finished";

    // When disabled, we should record "skipped" and never execute the closure.
    let result: Option<()> =
        run_optional_analysis(false, "starting", finished, skipped, || -> Result<()> {
            unreachable!("disabled analysis closure must not run")
        })
        .expect("should not error");
    assert!(result.is_none());
    assert_eq!(
        crate::watchdog::active_stage_for_test().as_deref(),
        Some(skipped)
    );

    // When enabled, we should end on "finished".
    let result: Option<()> =
        run_optional_analysis(true, "starting", finished, "skipped", || Ok(()))
            .expect("should not error");
    assert!(result.is_some());
    assert_eq!(
        crate::watchdog::active_stage_for_test().as_deref(),
        Some(finished)
    );

    drop(wd);
}

#[test]
fn split_global_deadline_ms_splits_relative_budget_into_two_absolute_deadlines() {
    let now_ms = 1_700_000_000_000u64;
    let raw_relative = 10_000u64;
    let (syn_end, total_end) = split_global_deadline_ms(raw_relative, now_ms, 0.7).expect("split");
    assert_eq!(total_end, now_ms + 10_000);
    assert_eq!(syn_end, now_ms + 7_000);
    assert!(syn_end < total_end);
}

#[test]
fn split_global_deadline_ms_splits_absolute_deadline() {
    let now_ms = 1_700_000_000_000u64;
    let absolute_end = now_ms + 10_000;
    let (syn_end, total_end) = split_global_deadline_ms(absolute_end, now_ms, 0.5).expect("split");
    assert_eq!(total_end, absolute_end);
    assert_eq!(syn_end, now_ms + 5_000);
}

#[test]
fn split_global_deadline_ms_does_not_split_too_small_budgets() {
    // Budgets < 6s cannot be safely split into two 3s windows.
    let now_ms = 1_700_000_000_000u64;
    let raw_relative = 5_000u64;
    let (syn_end, total_end) = split_global_deadline_ms(raw_relative, now_ms, 0.7).expect("ok");
    assert_eq!(syn_end, now_ms + 5_000);
    assert_eq!(total_end, now_ms + 5_000);
}

#[test]
fn split_global_deadline_ms_does_not_intercept_invalid_short_durations() {
    // Invalid (<3s) should be passed through to downstream validation so warnings are emitted.
    let now_ms = 1_700_000_000_000u64;
    let raw_relative = 2_000u64;
    assert!(split_global_deadline_ms(raw_relative, now_ms, 0.7).is_none());
}

#[test]
fn split_global_deadline_ms_does_not_intercept_invalid_long_durations() {
    // Invalid (>1h) should be passed through to downstream validation so warnings are emitted.
    let now_ms = 1_700_000_000_000u64;
    let raw_relative = 3_600_001u64;
    assert!(split_global_deadline_ms(raw_relative, now_ms, 0.7).is_none());
}
