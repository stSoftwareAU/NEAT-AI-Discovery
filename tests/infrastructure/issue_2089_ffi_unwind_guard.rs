//! Issue #2089 (security sweep chunk 2) — the argument-free cancellation and
//! lifecycle entry points delegate correctly through their unwind guard.
//!
//! `cancel_analysis`, `cancel_analysis_memory_pressure`, `reset_cancellation`
//! and `is_analysis_active` were the four `extern "C"` exports with no
//! `panic::catch_unwind`; an unwind out of an `extern "C"` function aborts the
//! host process, which the Deno side cannot catch. Wrapping them must not
//! swallow their effect — these tests pin the delegation the wrapper now sits
//! in front of, so a guard that silently turned an entry point into a no-op
//! fails loudly here.

use neat_ai_discovery::cancellation;
use neat_ai_discovery::ffi;
use serial_test::serial;

#[test]
#[serial]
fn cancel_analysis_entry_point_sets_the_global_flag() {
    cancellation::reset_cancellation();
    ffi::cancel_analysis();
    assert!(
        cancellation::is_cancelled(),
        "the guarded entry point must still set the cancellation flag"
    );
    assert!(
        !cancellation::is_memory_pressure_cancelled(),
        "a plain cancellation must not claim memory pressure"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn memory_pressure_entry_point_sets_both_flags() {
    cancellation::reset_cancellation();
    ffi::cancel_analysis_memory_pressure();
    assert!(cancellation::is_cancelled());
    assert!(
        cancellation::is_memory_pressure_cancelled(),
        "the guarded entry point must still record the memory-pressure reason"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn reset_entry_point_clears_both_flags() {
    cancellation::request_cancellation_memory_pressure();
    ffi::reset_cancellation();
    assert!(!cancellation::is_cancelled());
    assert!(!cancellation::is_memory_pressure_cancelled());
}

#[test]
#[serial]
fn is_analysis_active_entry_point_reports_the_counter() {
    cancellation::reset_analysis_active();
    assert_eq!(
        ffi::is_analysis_active(),
        0,
        "no analysis in flight must report 0"
    );

    cancellation::mark_analysis_started();
    assert_eq!(
        ffi::is_analysis_active(),
        1,
        "an in-flight analysis must report 1 so the host defers temp-dir deletion"
    );

    cancellation::mark_analysis_finished();
    assert_eq!(ffi::is_analysis_active(), 0);
}
