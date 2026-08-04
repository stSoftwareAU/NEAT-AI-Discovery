//! Issue #2006 — numeric `NEAT_AI_DISCOVERY_*` overrides must be
//! whitespace-tolerant everywhere.
//!
//! The rule for turning an environment override into a number — missing →
//! unset, whitespace-tolerant, unparsable → unset — was copy-pasted into 35
//! accessors and the copies had diverged: 25 trimmed before parsing, 10 did
//! not. A value carrying stray whitespace (a trailing newline from a shell
//! heredoc, `VAR: " 5 "` in a YAML env block) therefore tuned most knobs but
//! silently fell back to the compiled default for those ten.
//!
//! These tests pin the reconciled behaviour on every un-cached accessor that
//! used to skip the trim, plus a representative already-trimming accessor to
//! guard against a regression in the other direction.
//!
//! `gpu_batch_size_override` and `gpu_retry_limit` also lost the trim, but both
//! memoise into a process-wide `OnceLock`, so their value is frozen by whichever
//! test touches them first and they cannot be exercised in-process. Their trim
//! is covered by the shared helper's own unit tests in `src/config/helpers.rs`.

use std::time::Duration;

use neat_ai_discovery::config::{
    DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, block_size, dominance_threshold,
    gradient_threshold, max_cached_blocks, noise_signal_threshold, outlier_percentile,
    prefetch_depth, watchdog_abort_delay, watchdog_stall_timeout,
};
use serial_test::serial;

/// Set `name` to `value`, run `body`, then restore the previous value.
///
/// Every caller is `#[serial]`, so the process-wide mutation is safe.
fn with_env<T>(name: &str, value: &str, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();
    // SAFETY: Serialised via #[serial].
    unsafe { std::env::set_var(name, value) };
    let result = body();
    match previous {
        // SAFETY: Serialised via #[serial].
        Some(v) => unsafe { std::env::set_var(name, v) },
        // SAFETY: Serialised via #[serial].
        None => unsafe { std::env::remove_var(name) },
    }
    result
}

// ---------------------------------------------------------------------------
// src/config/detection.rs — the three threshold accessors
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn noise_signal_threshold_tolerates_surrounding_whitespace() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", " 0.25 ", || {
        noise_signal_threshold(0.75)
    });
    assert_eq!(value, 0.25, "padded override must still tune the knob");
}

#[test]
#[serial]
fn noise_signal_threshold_falls_back_when_unparsable() {
    let value = with_env("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD", "abc", || {
        noise_signal_threshold(0.75)
    });
    assert_eq!(value, 0.75, "unparsable override must fall back");
}

#[test]
#[serial]
fn dominance_threshold_tolerates_trailing_newline() {
    let value = with_env("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", "0.4\n", || {
        dominance_threshold(0.9)
    });
    assert_eq!(value, 0.4, "heredoc trailing newline must not be fatal");
}

#[test]
#[serial]
fn gradient_threshold_tolerates_surrounding_whitespace() {
    let value = with_env("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", "\t0.125\t", || {
        gradient_threshold(0.5)
    });
    assert_eq!(value, 0.125);
}

// ---------------------------------------------------------------------------
// src/config/user_facing.rs — watchdog, streaming cache and block sizing
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn watchdog_stall_timeout_tolerates_surrounding_whitespace() {
    let value = with_env("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS", " 45 ", || {
        watchdog_stall_timeout()
    });
    assert_eq!(value, Some(Duration::from_secs(45)));
}

#[test]
#[serial]
fn watchdog_stall_timeout_disabled_for_zero_and_unparsable() {
    let zero = with_env("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS", " 0 ", || {
        watchdog_stall_timeout()
    });
    assert_eq!(zero, None, "an explicit 0 still disables the watchdog");

    let junk = with_env("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS", "soon", || {
        watchdog_stall_timeout()
    });
    assert_eq!(junk, None, "unparsable override keeps the disabled default");
}

#[test]
#[serial]
fn watchdog_abort_delay_tolerates_surrounding_whitespace() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS",
        " 7\n",
        watchdog_abort_delay,
    );
    assert_eq!(value, Duration::from_secs(7));
}

#[test]
#[serial]
fn watchdog_abort_delay_falls_back_when_unparsable() {
    let value = with_env("NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS", "x", || {
        watchdog_abort_delay()
    });
    assert_eq!(value, Duration::from_secs(2), "compiled default is 2s");
}

#[test]
#[serial]
fn max_cached_blocks_tolerates_surrounding_whitespace() {
    let value = with_env("NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS", "  32  ", || {
        max_cached_blocks()
    });
    assert_eq!(value, Some(32));
}

#[test]
#[serial]
fn max_cached_blocks_is_none_when_unparsable() {
    let value = with_env("NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS", "lots", || {
        max_cached_blocks()
    });
    assert_eq!(value, None, "unparsable override keeps adaptive sizing");
}

#[test]
#[serial]
fn prefetch_depth_tolerates_surrounding_whitespace() {
    let value = with_env("NEAT_AI_DISCOVERY_PREFETCH_DEPTH", " 5 ", prefetch_depth);
    assert_eq!(value, 5);
}

#[test]
#[serial]
fn prefetch_depth_falls_back_when_unparsable() {
    let value = with_env("NEAT_AI_DISCOVERY_PREFETCH_DEPTH", "deep", prefetch_depth);
    assert_eq!(value, 2, "compiled default is 2");
}

#[test]
#[serial]
fn block_size_tolerates_surrounding_whitespace_and_still_clamps() {
    let value = with_env("NEAT_AI_DISCOVERY_BLOCK_SIZE", " 5000 ", block_size);
    assert_eq!(value, 5000);

    let clamped = with_env("NEAT_AI_DISCOVERY_BLOCK_SIZE", " 999999 ", block_size);
    assert_eq!(clamped, MAX_BLOCK_SIZE, "per-knob clamp is unchanged");

    let floored = with_env("NEAT_AI_DISCOVERY_BLOCK_SIZE", " 1 ", block_size);
    assert_eq!(floored, MIN_BLOCK_SIZE);
}

#[test]
#[serial]
fn block_size_falls_back_when_unparsable() {
    let value = with_env("NEAT_AI_DISCOVERY_BLOCK_SIZE", "big", block_size);
    assert_eq!(value, DEFAULT_BLOCK_SIZE);
}

// ---------------------------------------------------------------------------
// Regression guard — an accessor that already trimmed must keep its policy
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn outlier_percentile_keeps_trim_and_range_filter() {
    let value = with_env(
        "NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE",
        " 95 ",
        outlier_percentile,
    );
    assert_eq!(value, 95);

    let rejected = with_env(
        "NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE",
        " 0 ",
        outlier_percentile,
    );
    assert_eq!(rejected, 90, "out-of-range values still fall back");
}
