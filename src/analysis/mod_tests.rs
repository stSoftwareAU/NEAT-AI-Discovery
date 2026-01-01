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
fn choose_deadline_order_is_deterministic_for_fixed_inputs() {
    let now_ms = 1_700_000_000_000u64;
    assert_eq!(
        choose_deadline_order_synapse_first(Some(123), now_ms),
        choose_deadline_order_synapse_first(Some(123), now_ms)
    );
}

#[test]
fn choose_deadline_order_varies_over_time() {
    // The chooser mixes in epoch-ms, so adjacent milliseconds should flip the result.
    let now_ms = 1_700_000_000_000u64;
    assert_ne!(
        choose_deadline_order_synapse_first(Some(0), now_ms),
        choose_deadline_order_synapse_first(Some(0), now_ms + 1)
    );
}

#[test]
fn choose_deadline_order_can_be_controlled_by_seed() {
    // With a fixed time, different seeds should be able to flip ordering.
    let now_ms = 1_700_000_000_000u64;
    assert_ne!(
        choose_deadline_order_synapse_first(Some(0), now_ms),
        choose_deadline_order_synapse_first(Some(1), now_ms)
    );
}
