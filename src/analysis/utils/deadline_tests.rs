//! Tests for deadline and logging utilities (Issue #268).
//!
//! These tests were extracted from implementation.rs as part of the refactoring.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::*;
use std::time::{Duration, SystemTime};

// ============================================================================
// shuffle_within_top_k tests
// ============================================================================

#[test]
fn shuffle_within_top_k_deterministic_with_seed() {
    let mut a = [1u32, 2, 3, 4, 5, 6, 7, 8].to_vec();
    let mut b = a.clone();
    shuffle_within_top_k(&mut a[..], Some(123), "ctx", 5);
    shuffle_within_top_k(&mut b[..], Some(123), "ctx", 5);
    assert_eq!(a, b);
}

#[test]
fn shuffle_within_top_k_only_shuffles_prefix() {
    let mut items = [1u32, 2, 3, 4, 5, 6, 7, 8].to_vec();
    shuffle_within_top_k(&mut items[..], Some(123), "ctx", 3);
    // Suffix must remain untouched.
    assert_eq!(items[3..], [4, 5, 6, 7, 8]);
}

#[test]
fn shuffle_within_top_k_preserves_multiset() {
    let mut items = [1u32, 2, 3, 4, 5, 6, 7, 8].to_vec();
    let mut before = items.clone();
    before.sort_unstable();
    shuffle_within_top_k(&mut items[..], Some(999), "ctx", 6);
    items.sort_unstable();
    assert_eq!(items, before);
}

#[test]
fn shuffle_within_top_k_no_op_when_top_k_is_zero() {
    let mut items = [1u32, 2, 3, 4].to_vec();
    let before = items.clone();
    shuffle_within_top_k(&mut items[..], Some(1), "ctx", 0);
    assert_eq!(items, before);
}

// ============================================================================
// deadline_passed tests
// ============================================================================

#[test]
fn deadline_passed_detects_elapsed_wall_clock_deadline() {
    // Use the deadline override mechanism in tests so behaviour is deterministic
    let _guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![true, false]);

    let dummy_deadline = Some(SystemTime::now());
    assert!(
        deadline_passed(&dummy_deadline),
        "past deadlines should be treated as expired immediately"
    );

    assert!(
        !deadline_passed(&dummy_deadline),
        "future deadlines should not be marked as expired"
    );

    assert!(
        !deadline_passed(&None),
        "missing deadlines should behave as if no timeout was requested"
    );
}

// ============================================================================
// build_deadline tests
// ============================================================================

#[test]
fn build_deadline_handles_absolute_timestamps_and_relative_durations() {
    // Verify that build_deadline correctly handles both absolute timestamps
    // (milliseconds since UNIX_EPOCH) and relative durations (milliseconds from now)
    let now = SystemTime::now();
    let now_ms = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_millis() as u64;

    // Test 1: Absolute timestamp (large value, >= year 2000)
    // Create a deadline 10 minutes in the future using absolute timestamp
    let ten_minutes_ms = 10 * 60 * 1000; // 10 minutes in milliseconds
    let future_deadline_ms = now_ms + ten_minutes_ms;
    let deadline = build_deadline(Some(future_deadline_ms));

    assert!(
        deadline.is_some(),
        "deadline should be Some when deadline_ms is provided"
    );

    let deadline_time = deadline.unwrap();

    // The deadline should be approximately 10 minutes in the future
    // Allow for some small timing variance (up to 1 second)
    if let Ok(duration) = deadline_time.duration_since(now) {
        let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
        let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "absolute timestamp deadline should be approximately 10 minutes in the future, got {duration:?}"
        );
    } else {
        panic!("deadline should be in the future");
    }

    // Test 2: Relative duration (small value, < year 2000)
    // Pass 10 minutes as a relative duration
    let relative_deadline_ms = ten_minutes_ms; // 10 minutes as relative duration
    let relative_deadline = build_deadline(Some(relative_deadline_ms));
    assert!(relative_deadline.is_some());
    let relative_time = relative_deadline.unwrap();
    // This should also be approximately 10 minutes in the future
    if let Ok(duration) = relative_time.duration_since(now) {
        let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
        let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "relative duration deadline should be approximately 10 minutes in the future, got {duration:?}"
        );
    } else {
        panic!("relative deadline should be in the future");
    }

    // Test 3: Verify that an absolute timestamp in the past returns None
    // (deadline already passed - no point in creating a deadline)
    let past_timestamp_ms = 1_700_000_000_000u64; // Jan 2024 (in the past)
    let past_deadline = build_deadline(Some(past_timestamp_ms));
    assert!(
        past_deadline.is_none(),
        "Past timestamp should return None (deadline already passed)"
    );

    // Test 4: Verify that a future absolute timestamp is correctly converted to relative duration
    let future_timestamp_ms = now_ms + ten_minutes_ms; // 10 minutes in the future as absolute timestamp
    let future_deadline = build_deadline(Some(future_timestamp_ms));
    assert!(future_deadline.is_some());
    let future_time = future_deadline.unwrap();
    // Should be approximately 10 minutes in the future
    if let Ok(duration) = future_time.duration_since(now) {
        let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
        let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "Future absolute timestamp should be converted to relative duration correctly, got {duration:?}"
        );
    } else {
        panic!("Future deadline should be in the future");
    }
}

#[test]
fn build_deadline_validates_duration_bounds() {
    // Test that build_deadline skips sub-minimum values and clamps over-maximum
    // values rather than inflating either to the 10-minute default (Issue #4139).
    let now = SystemTime::now();

    // Test 1: Duration below minimum (3 seconds) is unusable — skip, do not inflate.
    let too_short_ms = 1_000u64; // 1 second
    let deadline = build_deadline(Some(too_short_ms));
    assert!(
        deadline.is_none(),
        "Duration below 3 seconds must skip analysis, not inflate to 10 minutes"
    );

    // Test 2: Duration above maximum (1 hour) clamps to MAX, not the 10-minute default.
    let too_long_ms = 4_000_000u64; // ~66 minutes
    let deadline = build_deadline(Some(too_long_ms));
    assert!(deadline.is_some());
    let deadline_time = deadline.unwrap();
    if let Ok(duration) = deadline_time.duration_since(now) {
        let expected_min = Duration::from_millis(MAX_DURATION_MS) - Duration::from_secs(1);
        let expected_max = Duration::from_millis(MAX_DURATION_MS) + Duration::from_secs(1);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "Duration above 1 hour should clamp to MAX_DURATION_MS, got {duration:?}"
        );
    } else {
        panic!("Clamped deadline should be in the future");
    }

    // Test 3: Valid duration (10 minutes) should pass through unchanged
    let valid_ms = 10 * 60 * 1000u64; // 10 minutes
    let deadline = build_deadline(Some(valid_ms));
    assert!(deadline.is_some());
    let deadline_time = deadline.unwrap();
    if let Ok(duration) = deadline_time.duration_since(now) {
        let expected_min = Duration::from_millis(valid_ms) - Duration::from_secs(1);
        let expected_max = Duration::from_millis(valid_ms) + Duration::from_secs(1);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "Valid duration should pass through unchanged, got {duration:?}"
        );
    } else {
        panic!("Valid deadline should be in the future");
    }

    // Test 4: Exactly at minimum (3 seconds) should pass through
    let min_ms = 3_000u64; // Exactly 3 seconds
    let deadline = build_deadline(Some(min_ms));
    assert!(deadline.is_some());
    let deadline_time = deadline.unwrap();
    if let Ok(duration) = deadline_time.duration_since(now) {
        let expected_min = Duration::from_millis(min_ms) - Duration::from_millis(100);
        let expected_max = Duration::from_millis(min_ms) + Duration::from_millis(100);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "Duration at minimum should pass through, got {duration:?}"
        );
    } else {
        panic!("Minimum deadline should be in the future");
    }

    // Test 5: Exactly at maximum (1 hour) should pass through
    let max_ms = 3_600_000u64; // Exactly 1 hour
    let deadline = build_deadline(Some(max_ms));
    assert!(deadline.is_some());
    let deadline_time = deadline.unwrap();
    if let Ok(duration) = deadline_time.duration_since(now) {
        let expected_min = Duration::from_millis(max_ms) - Duration::from_millis(1000);
        let expected_max = Duration::from_millis(max_ms) + Duration::from_millis(1000);
        assert!(
            duration >= expected_min && duration <= expected_max,
            "Duration at maximum should pass through, got {duration:?}"
        );
    } else {
        panic!("Maximum deadline should be in the future");
    }
}

// ============================================================================
// calculate_effective_timeout_ms tests
// ============================================================================

/// Test that `calculate_effective_timeout_ms` applies the same logic as `build_deadline`.
/// This is critical for ensuring `log_analysis_start` displays the correct timeout.
#[test]
fn calculate_effective_timeout_ms_matches_build_deadline_logic() {
    let now_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_millis() as u64;

    // Test 1: Relative duration (15 minutes) should pass through unchanged
    let fifteen_minutes_ms = 15 * 60 * 1000u64;
    let result = calculate_effective_timeout_ms(Some(fifteen_minutes_ms));
    assert_eq!(
        result,
        Some(fifteen_minutes_ms),
        "15 minute relative duration should pass through unchanged"
    );

    // Test 2: Absolute timestamp (now + 15 minutes) should convert to ~15 minutes
    let absolute_15min = now_ms + fifteen_minutes_ms;
    let result = calculate_effective_timeout_ms(Some(absolute_15min));
    assert!(
        result.is_some(),
        "Future absolute timestamp should return Some"
    );
    let effective_ms = result.unwrap();
    // Allow 2 second tolerance for timing variance
    assert!(
        effective_ms >= fifteen_minutes_ms - 2000 && effective_ms <= fifteen_minutes_ms + 2000,
        "Absolute timestamp should convert to ~15 minutes, got {effective_ms}ms"
    );

    // Test 3: Duration below minimum (1 second) is skipped, not inflated
    let too_short_ms = 1_000u64;
    let result = calculate_effective_timeout_ms(Some(too_short_ms));
    assert_eq!(
        result, None,
        "Duration below 3 seconds must skip analysis, not inflate to 10 minutes"
    );

    // Test 4: Duration above maximum (2 hours) clamps to MAX, not the 10-minute default
    let too_long_ms = 2 * 3_600_000u64;
    let result = calculate_effective_timeout_ms(Some(too_long_ms));
    assert_eq!(
        result,
        Some(MAX_DURATION_MS),
        "Duration above 1 hour should clamp to MAX_DURATION_MS"
    );

    // Test 5: Past absolute timestamp should return None
    let past_timestamp_ms = 1_700_000_000_000u64; // Circa late 2023
    let result = calculate_effective_timeout_ms(Some(past_timestamp_ms));
    assert!(
        result.is_none(),
        "Past timestamp should return None (deadline already passed)"
    );

    // Test 6: None should default to 10 minutes
    let result = calculate_effective_timeout_ms(None);
    assert_eq!(
        result,
        Some(DEFAULT_DURATION_MS),
        "None should default to 10 minutes"
    );
}

// ============================================================================
// remaining_ms_until tests (Issue #1407)
// ============================================================================

#[test]
fn remaining_ms_until_passes_through_relative_durations() {
    // Values below the year-2000 threshold are relative durations and are
    // returned unchanged regardless of `now_ms`.
    assert_eq!(remaining_ms_until(Some(120_000), 999_999), Some(120_000));
    assert_eq!(remaining_ms_until(Some(0), 42), Some(0));
}

#[test]
fn remaining_ms_until_subtracts_now_for_absolute_timestamps() {
    let now = YEAR_2000_MS + 1_000_000;
    let deadline = now + 30_000; // 30s in the future
    assert_eq!(remaining_ms_until(Some(deadline), now), Some(30_000));
}

#[test]
fn remaining_ms_until_saturates_to_zero_when_deadline_passed() {
    let now = YEAR_2000_MS + 1_000_000;
    let deadline = now - 5_000; // already passed
    assert_eq!(remaining_ms_until(Some(deadline), now), Some(0));
}

#[test]
fn remaining_ms_until_none_without_deadline() {
    assert_eq!(remaining_ms_until(None, 123), None);
}

/// Issue #1407 acceptance criterion: focus selection and analysis bill against
/// ONE shared absolute deadline. Time spent in focus selection must reduce the
/// window left for analysis — it must NOT receive a fresh full window.
#[test]
fn shared_absolute_deadline_reduces_remaining_window_for_later_phase() {
    let discovery_start_ms = YEAR_2000_MS + 10_000_000;
    let budget_ms = 100_000; // 100s total discovery budget
    let shared_deadline = Some(discovery_start_ms + budget_ms);

    // Focus selection starts at the beginning of the cycle — it sees the full
    // budget.
    let focus_window =
        remaining_ms_until(shared_deadline, discovery_start_ms).expect("shared deadline present");
    assert_eq!(
        focus_window, budget_ms,
        "focus should see the full budget at the start of the cycle"
    );

    // Focus selection (plus its parquet load) consumes 80s of wall clock.
    let focus_cost_ms = 80_000;
    let analysis_start_ms = discovery_start_ms + focus_cost_ms;
    let analysis_window =
        remaining_ms_until(shared_deadline, analysis_start_ms).expect("shared deadline present");

    // Analysis must get only the remainder (20s), not a fresh 100s window.
    assert_eq!(analysis_window, budget_ms - focus_cost_ms);
    assert!(
        analysis_window < focus_window,
        "time spent in focus must shrink the analysis window"
    );
    // The phases are coordinated under one budget: focus_cost + remainder == budget.
    assert_eq!(focus_cost_ms + analysis_window, budget_ms);
}

// ============================================================================
// parse_input_index tests
// ============================================================================

#[test]
fn parse_input_index_parses_valid_input_uuids() {
    assert_eq!(parse_input_index("input-0"), Some(0));
    assert_eq!(parse_input_index("input-42"), Some(42));
    assert_eq!(parse_input_index("input-1000"), Some(1000));
}

#[test]
fn parse_input_index_returns_none_for_invalid_uuids() {
    assert_eq!(parse_input_index("hidden-123"), None);
    assert_eq!(parse_input_index("output-0"), None);
    assert_eq!(parse_input_index("input-"), None);
    assert_eq!(parse_input_index("input-abc"), None);
    assert_eq!(parse_input_index(""), None);
}

// ============================================================================
// derive_seed tests
// ============================================================================

#[test]
fn derive_seed_is_deterministic() {
    let seed1 = derive_seed(12345, "context", 0);
    let seed2 = derive_seed(12345, "context", 0);
    assert_eq!(seed1, seed2);
}

#[test]
fn derive_seed_varies_with_context() {
    let seed1 = derive_seed(12345, "context1", 0);
    let seed2 = derive_seed(12345, "context2", 0);
    assert_ne!(seed1, seed2);
}

#[test]
fn derive_seed_varies_with_salt() {
    let seed1 = derive_seed(12345, "context", 0);
    let seed2 = derive_seed(12345, "context", 1);
    assert_ne!(seed1, seed2);
}

#[test]
fn derive_seed_varies_with_base_seed() {
    let seed1 = derive_seed(12345, "context", 0);
    let seed2 = derive_seed(54321, "context", 0);
    assert_ne!(seed1, seed2);
}

// ============================================================================
// shuffle_slice tests
// ============================================================================

#[test]
fn shuffle_slice_is_deterministic_with_seed() {
    let mut a = [1, 2, 3, 4, 5, 6, 7, 8].to_vec();
    let mut b = a.clone();
    shuffle_slice(&mut a[..], Some(42), "test");
    shuffle_slice(&mut b[..], Some(42), "test");
    assert_eq!(a, b);
}

#[test]
fn shuffle_slice_preserves_elements() {
    let mut items = [1, 2, 3, 4, 5].to_vec();
    let mut sorted = items.clone();
    shuffle_slice(&mut items[..], Some(123), "test");
    items.sort();
    sorted.sort();
    assert_eq!(items, sorted);
}

#[test]
fn shuffle_slice_no_op_for_single_element() {
    let mut items = [42].to_vec();
    shuffle_slice(&mut items[..], Some(123), "test");
    assert_eq!(items, [42]);
}

#[test]
fn shuffle_slice_no_op_for_empty() {
    let mut items: Vec<i32> = vec![];
    shuffle_slice(&mut items[..], Some(123), "test");
    assert!(items.is_empty());
}

// ============================================================================
// calculate_gpu_batch_timeout tests
// ============================================================================

#[test]
fn calculate_gpu_batch_timeout_returns_max_when_no_deadline() {
    let timeout = calculate_gpu_batch_timeout(&None);
    assert_eq!(timeout, Duration::from_secs(GPU_QUEUE_TIMEOUT_MAX_SECS));
}

#[test]
fn calculate_gpu_batch_timeout_returns_min_when_deadline_passed() {
    let past_deadline = Some(SystemTime::now() - Duration::from_secs(10));
    let timeout = calculate_gpu_batch_timeout(&past_deadline);
    assert_eq!(timeout, Duration::from_secs(GPU_QUEUE_TIMEOUT_MIN_SECS));
}

#[test]
fn calculate_gpu_batch_timeout_uses_half_remaining_time() {
    let future_deadline = Some(SystemTime::now() + Duration::from_secs(200));
    let timeout = calculate_gpu_batch_timeout(&future_deadline);
    // Half of 200 seconds is 100 seconds, which is between min (60) and max (300)
    assert!(timeout >= Duration::from_secs(95) && timeout <= Duration::from_secs(105));
}

#[test]
fn calculate_gpu_batch_timeout_clamps_to_min() {
    let near_deadline = Some(SystemTime::now() + Duration::from_secs(30));
    let timeout = calculate_gpu_batch_timeout(&near_deadline);
    // Half of 30 seconds is 15, which is below min (60), so should return min
    assert_eq!(timeout, Duration::from_secs(GPU_QUEUE_TIMEOUT_MIN_SECS));
}

#[test]
fn calculate_gpu_batch_timeout_clamps_to_max() {
    let far_deadline = Some(SystemTime::now() + Duration::from_secs(1000));
    let timeout = calculate_gpu_batch_timeout(&far_deadline);
    // Half of 1000 seconds is 500, which is above max (300), so should return max
    assert_eq!(timeout, Duration::from_secs(GPU_QUEUE_TIMEOUT_MAX_SECS));
}

// ============================================================================
// calculate_effective_timeout_ms boundary condition tests (Issue #718)
// ============================================================================

#[test]
fn effective_timeout_just_below_minimum_is_skipped() {
    // 2999ms is just below the 3000ms minimum — skip, do not inflate (Issue #4139).
    let result = calculate_effective_timeout_ms(Some(MIN_DURATION_MS - 1));
    assert_eq!(
        result, None,
        "Duration just below minimum must skip analysis, not fall back to default"
    );
}

#[test]
fn effective_timeout_at_exact_minimum_passes_through() {
    let result = calculate_effective_timeout_ms(Some(MIN_DURATION_MS));
    assert_eq!(
        result,
        Some(MIN_DURATION_MS),
        "Duration at exact minimum should pass through"
    );
}

#[test]
fn effective_timeout_just_above_minimum_passes_through() {
    let result = calculate_effective_timeout_ms(Some(MIN_DURATION_MS + 1));
    assert_eq!(
        result,
        Some(MIN_DURATION_MS + 1),
        "Duration just above minimum should pass through"
    );
}

#[test]
fn effective_timeout_just_below_maximum_passes_through() {
    let result = calculate_effective_timeout_ms(Some(MAX_DURATION_MS - 1));
    assert_eq!(
        result,
        Some(MAX_DURATION_MS - 1),
        "Duration just below maximum should pass through"
    );
}

#[test]
fn effective_timeout_at_exact_maximum_passes_through() {
    let result = calculate_effective_timeout_ms(Some(MAX_DURATION_MS));
    assert_eq!(
        result,
        Some(MAX_DURATION_MS),
        "Duration at exact maximum should pass through"
    );
}

#[test]
fn effective_timeout_just_above_maximum_clamps_to_max() {
    let result = calculate_effective_timeout_ms(Some(MAX_DURATION_MS + 1));
    assert_eq!(
        result,
        Some(MAX_DURATION_MS),
        "Duration just above maximum should clamp to MAX_DURATION_MS, not the 10-minute default"
    );
}

#[test]
fn effective_timeout_zero_is_skipped() {
    let result = calculate_effective_timeout_ms(Some(0));
    assert_eq!(
        result, None,
        "Zero duration must skip analysis, not inflate to 10 minutes"
    );
}

/// Exact GRQ-26 observed remainder: 0.89 s was inflated to 600 s (Issue #4139).
#[test]
fn sub_minimum_deadline_0_89s_is_skipped_not_inflated() {
    let result = calculate_effective_timeout_ms(Some(890));
    assert_eq!(
        result, None,
        "890ms must skip analysis rather than grant DEFAULT_DURATION_MS"
    );
    assert_ne!(
        result,
        Some(DEFAULT_DURATION_MS),
        "must never inflate a sub-minimum remainder to the 10-minute default"
    );
    assert!(
        build_deadline(Some(890)).is_none(),
        "build_deadline must propagate the skip"
    );
    assert!(
        deadline_too_short_to_analyse(Some(890), "test_caller"),
        "skip helper must recognise the GRQ-26 remainder"
    );
}

#[test]
fn over_maximum_deadline_clamps_to_max_not_default() {
    let result = calculate_effective_timeout_ms(Some(MAX_DURATION_MS + 60_000));
    assert_eq!(result, Some(MAX_DURATION_MS));
    assert_ne!(
        result,
        Some(DEFAULT_DURATION_MS),
        "over-maximum must clamp to MAX, not reset to the 10-minute default"
    );
}

// ============================================================================
// cap_deadline_to_wall_clock tests (Issue #1098)
// ============================================================================

#[test]
fn cap_deadline_clamps_when_analysis_exceeds_wall_clock() {
    let start = SystemTime::now();
    let analysis = Some(start + Duration::from_secs(30 * 60)); // 30 min
    let capped = cap_deadline_to_wall_clock(analysis, start, Some(20)); // 20 min cap
    let expected = start + Duration::from_secs(20 * 60);
    let capped_time = capped.expect("should be Some");
    let drift = capped_time
        .duration_since(expected)
        .or_else(|_| expected.duration_since(capped_time))
        .unwrap_or(Duration::ZERO);
    assert!(
        drift < Duration::from_secs(1),
        "Should clamp to wall-clock cap"
    );
}

#[test]
fn cap_deadline_preserves_when_analysis_within_wall_clock() {
    let start = SystemTime::now();
    let analysis = Some(start + Duration::from_secs(10 * 60)); // 10 min
    let capped = cap_deadline_to_wall_clock(analysis, start, Some(20)); // 20 min cap
    let expected = start + Duration::from_secs(10 * 60);
    let capped_time = capped.expect("should be Some");
    let drift = capped_time
        .duration_since(expected)
        .or_else(|_| expected.duration_since(capped_time))
        .unwrap_or(Duration::ZERO);
    assert!(
        drift < Duration::from_secs(1),
        "Should preserve analysis deadline"
    );
}

#[test]
fn cap_deadline_no_cap_leaves_analysis_unchanged() {
    let start = SystemTime::now();
    let analysis_time = start + Duration::from_secs(30 * 60);
    let capped = cap_deadline_to_wall_clock(Some(analysis_time), start, None);
    assert!(capped.is_some());
    let drift = capped
        .unwrap()
        .duration_since(analysis_time)
        .unwrap_or(Duration::ZERO);
    assert!(drift < Duration::from_millis(10));
}

#[test]
fn cap_deadline_no_analysis_applies_wall_clock_only() {
    let start = SystemTime::now();
    let capped = cap_deadline_to_wall_clock(None, start, Some(15));
    let expected = start + Duration::from_secs(15 * 60);
    let capped_time = capped.expect("should produce deadline from cap");
    let drift = capped_time
        .duration_since(expected)
        .or_else(|_| expected.duration_since(capped_time))
        .unwrap_or(Duration::ZERO);
    assert!(drift < Duration::from_secs(1));
}

#[test]
fn cap_deadline_both_none_returns_none() {
    let start = SystemTime::now();
    assert!(cap_deadline_to_wall_clock(None, start, None).is_none());
}

// ============================================================================
// OrderedNeuron tests
// ============================================================================

#[test]
fn ordered_neuron_stores_uuid_and_index() {
    let neuron = OrderedNeuron {
        uuid: "input-42".to_string(),
        index: 42,
    };
    assert_eq!(neuron.uuid, "input-42");
    assert_eq!(neuron.index, 42);
}

// ============================================================================
// order_eligible_sources tests
// ============================================================================

#[test]
fn order_eligible_sources_shuffles_deterministically_with_seed() {
    let neurons: Vec<OrderedNeuron> = (0..10)
        .map(|i| OrderedNeuron {
            uuid: format!("hidden-{i}"),
            index: i,
        })
        .collect();

    let mut sources1: Vec<&OrderedNeuron> = neurons.iter().collect();
    let mut sources2: Vec<&OrderedNeuron> = neurons.iter().collect();

    order_eligible_sources::<String>(&mut sources1, Some(12345), "test", 0, None);
    order_eligible_sources::<String>(&mut sources2, Some(12345), "test", 0, None);

    let uuids1: Vec<&str> = sources1.iter().map(|n| n.uuid.as_str()).collect();
    let uuids2: Vec<&str> = sources2.iter().map(|n| n.uuid.as_str()).collect();

    assert_eq!(uuids1, uuids2);
}

#[test]
fn order_eligible_sources_preserves_all_elements() {
    let neurons: Vec<OrderedNeuron> = (0..5)
        .map(|i| OrderedNeuron {
            uuid: format!("neuron-{i}"),
            index: i,
        })
        .collect();

    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut sources, Some(42), "test", 0, None);

    let mut uuids: Vec<&str> = sources.iter().map(|n| n.uuid.as_str()).collect();
    uuids.sort();
    let expected = ["neuron-0", "neuron-1", "neuron-2", "neuron-3", "neuron-4"];
    assert_eq!(uuids, expected);
}

#[test]
fn order_eligible_sources_no_op_for_single_element() {
    let neurons = [OrderedNeuron {
        uuid: "single".to_string(),
        index: 0,
    }];
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut sources, Some(42), "test", 0, None);
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].uuid, "single");
}

// ============================================================================
// deadline_to_absolute_ms tests (Issue #1097)
// ============================================================================

#[test]
fn deadline_to_absolute_ms_returns_none_for_none() {
    assert!(
        deadline_to_absolute_ms(&None).is_none(),
        "None deadline should return None"
    );
}

#[test]
fn deadline_to_absolute_ms_returns_absolute_timestamp() {
    let now = SystemTime::now();
    let deadline = Some(now + Duration::from_secs(300));
    let abs_ms = deadline_to_absolute_ms(&deadline);
    assert!(abs_ms.is_some(), "Valid deadline should return Some");

    let now_ms = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let result = abs_ms.unwrap();
    // Should be approximately now + 300 seconds
    assert!(
        result >= now_ms + 299_000 && result <= now_ms + 301_000,
        "Absolute ms should be ~300s in the future, got delta {}ms",
        result.saturating_sub(now_ms)
    );
}

/// Issue #1097: Verify that converting a relative deadline to absolute ms and
/// then rebuilding the deadline produces a consistent point in time, rather
/// than re-adding the original duration again.
#[test]
fn shared_deadline_does_not_reset_on_rebuild() {
    // Simulate the original relative duration (10 minutes).
    let relative_ms = 600_000u64;

    // Step 1: Build the overall deadline once (as analyze_all does).
    let overall_deadline = build_deadline(Some(relative_ms));
    assert!(overall_deadline.is_some());

    // Step 2: Convert to absolute ms (the fix from Issue #1097).
    let abs_ms = deadline_to_absolute_ms(&overall_deadline);
    assert!(abs_ms.is_some());
    let abs_ms_value = abs_ms.unwrap();

    // The absolute value must be >= YEAR_2000_MS so that build_deadline treats
    // it as an absolute timestamp rather than a relative duration.
    assert!(
        abs_ms_value >= YEAR_2000_MS,
        "Absolute ms ({abs_ms_value}) should be >= YEAR_2000_MS ({YEAR_2000_MS})"
    );

    // Step 3: Rebuild the deadline from the absolute ms (as sub-phases do).
    let rebuilt_deadline = build_deadline(abs_ms);
    assert!(rebuilt_deadline.is_some());

    // Step 4: The rebuilt deadline should be close to the original, NOT
    // an additional 10 minutes in the future.
    let original_ms = overall_deadline
        .unwrap()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let rebuilt_ms = rebuilt_deadline
        .unwrap()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let drift = rebuilt_ms.abs_diff(original_ms);

    // Allow up to 2 seconds of drift from timing variance.
    assert!(
        drift < 2_000,
        "Rebuilt deadline drifted {drift}ms from the original — the absolute \
         timestamp should NOT re-add the duration (Issue #1097)"
    );
}

// ============================================================================
// Reserved analysis budget tests (Issue #1408)
// ============================================================================

#[test]
fn effective_reserve_uses_absolute_floor_on_generous_budget() {
    // 10-minute window: the 60s floor binds, not the 50% fraction cap (300s).
    let reserve = effective_analysis_reserve_ms(600_000, 60_000, 0.5);
    assert_eq!(reserve, 60_000);
}

#[test]
fn effective_reserve_capped_by_fraction_on_small_budget() {
    // 10s window with a 60s floor: the 50% fraction caps the reserve at 5s so
    // loading is not starved to zero.
    let reserve = effective_analysis_reserve_ms(10_000, 60_000, 0.5);
    assert_eq!(reserve, 5_000);
}

#[test]
fn effective_reserve_zero_floor_disables() {
    assert_eq!(effective_analysis_reserve_ms(600_000, 0, 0.5), 0);
}

#[test]
fn effective_reserve_clamps_fraction_above_one() {
    // A nonsensical fraction > 1.0 is clamped so the reserve never exceeds the
    // whole remaining window.
    let reserve = effective_analysis_reserve_ms(10_000, 60_000, 5.0);
    assert_eq!(reserve, 10_000);
}

#[test]
fn loading_deadline_leaves_reserve_for_analysis() {
    // overall = now + 600s. With a 60s reserve, loading must finish 60s earlier.
    let now = 1_000_000_000_000u64;
    let overall = now + 600_000;
    let loading = loading_deadline_with_reserve_ms(Some(overall), now, 60_000, 0.5).unwrap();
    assert_eq!(loading, overall - 60_000);
    // Analysis therefore keeps the full reserve.
    assert_eq!(overall - loading, 60_000);
}

#[test]
fn loading_deadline_never_before_now() {
    // Even when focus/parquet have consumed almost everything, the loading
    // deadline never precedes `now` (reserve cannot exceed the remaining window).
    let now = 1_000_000_000_000u64;
    let overall = now + 4_000; // only 4s remain
    let loading = loading_deadline_with_reserve_ms(Some(overall), now, 60_000, 0.5).unwrap();
    assert!(
        loading >= now,
        "loading deadline {loading} precedes now {now}"
    );
    // Reserve is fraction-capped to 2s, so loading keeps the other 2s.
    assert_eq!(overall - loading, 2_000);
}

#[test]
fn loading_deadline_none_without_deadline() {
    assert_eq!(
        loading_deadline_with_reserve_ms(None, 1_000, 60_000, 0.5),
        None
    );
}

#[test]
fn reserve_shortfall_none_when_window_is_ample() {
    // Issue #1408 core scenario: focus + parquet consumed most of a 10-minute
    // budget, leaving 90s. The reserve (capped to 45s by the 0.5 fraction) is
    // well above the hard floor, so analysis proceeds — it still gets its window.
    let now = 1_000_000_000_000u64;
    let overall = now + 90_000;
    assert_eq!(
        analysis_reserve_shortfall_ms(Some(overall), now, 60_000, 0.5),
        None
    );
    // And the reserved analysis window is positive (completed targets > 0).
    let loading = loading_deadline_with_reserve_ms(Some(overall), now, 60_000, 0.5).unwrap();
    assert_eq!(overall - loading, 45_000);
}

#[test]
fn reserve_shortfall_some_when_budget_exhausted() {
    // Focus + parquet consumed nearly everything: only 500ms remain. The
    // effective reserve (250ms) is below the 1s hard floor, so the run must
    // fail fast rather than analyse 0/N targets.
    let now = 1_000_000_000_000u64;
    let overall = now + 500;
    assert_eq!(
        analysis_reserve_shortfall_ms(Some(overall), now, 60_000, 0.5),
        Some(500)
    );
}

#[test]
fn reserve_shortfall_disabled_when_reserve_zero() {
    let now = 1_000_000_000_000u64;
    let overall = now + 100; // tiny window, but reserve disabled
    assert_eq!(
        analysis_reserve_shortfall_ms(Some(overall), now, 0, 0.5),
        None
    );
}

#[test]
fn reserve_shortfall_none_without_deadline() {
    assert_eq!(
        analysis_reserve_shortfall_ms(None, 1_000, 60_000, 0.5),
        None
    );
}

#[test]
fn reserved_loading_deadline_systemtime_subtracts_reserve() {
    let now = SystemTime::now();
    let overall_time = now + Duration::from_secs(600);
    let loading =
        reserved_loading_deadline(Some(overall_time), 60_000, 0.5).expect("some deadline");
    // Loading deadline should be ~60s before the overall deadline.
    let gap = overall_time
        .duration_since(loading)
        .expect("overall after loading");
    let gap_ms = gap.as_millis() as u64;
    assert!(
        gap_ms.abs_diff(60_000) < 2_000,
        "expected ~60s reserve gap, got {gap_ms}ms"
    );
}

#[test]
fn reserved_loading_deadline_passthrough_when_disabled() {
    let now = SystemTime::now();
    let overall = Some(now + Duration::from_secs(600));
    // reserve_ms == 0 → overall deadline returned unchanged.
    assert_eq!(reserved_loading_deadline(overall, 0, 0.5), overall);
}

#[test]
fn reserved_loading_deadline_none_without_deadline() {
    assert_eq!(reserved_loading_deadline(None, 60_000, 0.5), None);
}
