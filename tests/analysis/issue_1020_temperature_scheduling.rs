//! Issue #1020: Temperature scheduling for exploration-exploitation balance.
//!
//! Tests verify:
//! 1. Cooling schedule functions (linear and exponential)
//! 2. Temperature scaling of acceptance threshold and improved ratio
//! 3. Default temperature (1.0) preserves existing behaviour
//! 4. High temperature increases acceptance (exploration)
//! 5. Temperature integration with Metropolis-Hastings probability

use neat_ai_discovery::analysis::constants::MIN_IMPROVED_RATIO;
use neat_ai_discovery::analysis::constants::temperature::{
    CoolingSchedule, DEFAULT_EXPONENTIAL_DECAY_RATE, DEFAULT_TEMPERATURE, MAX_TEMPERATURE,
    MIN_TEMPERATURE, compute_scheduled_temperature, scale_mh_temperature,
    scale_ratio_by_temperature, scale_threshold_by_temperature,
};

// =============================================================================
// Cooling Schedule Tests
// =============================================================================

/// Issue #1020: Linear cooling schedule decreases monotonically from initial to minimum.
#[test]
fn linear_schedule_decreases_monotonically() {
    let mut prev = f32::INFINITY;
    for generation in 0..=100 {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 2.0, generation, 100);
        assert!(
            temp <= prev,
            "Linear schedule should decrease: generation {generation}, temp {temp} > prev {prev}"
        );
        assert!(
            temp >= MIN_TEMPERATURE,
            "Temperature must not fall below minimum"
        );
        assert!(
            temp <= MAX_TEMPERATURE,
            "Temperature must not exceed maximum"
        );
        prev = temp;
    }
}

/// Issue #1020: Exponential cooling schedule decreases monotonically.
#[test]
fn exponential_schedule_decreases_monotonically() {
    let mut prev = f32::INFINITY;
    for generation in 0..=1000 {
        let temp =
            compute_scheduled_temperature(CoolingSchedule::Exponential, 2.0, generation, 1000);
        assert!(
            temp <= prev,
            "Exponential schedule should decrease: generation {generation}, temp {temp} > prev {prev}"
        );
        assert!(
            temp >= MIN_TEMPERATURE,
            "Temperature must not fall below minimum"
        );
        assert!(
            temp <= MAX_TEMPERATURE,
            "Temperature must not exceed maximum"
        );
        prev = temp;
    }
}

/// Issue #1020: Both schedules start at the same initial temperature.
#[test]
fn both_schedules_start_at_initial_temperature() {
    let initial = 1.5;
    let linear = compute_scheduled_temperature(CoolingSchedule::Linear, initial, 0, 1000);
    let exponential = compute_scheduled_temperature(CoolingSchedule::Exponential, initial, 0, 1000);
    assert!(
        (linear - initial).abs() < f32::EPSILON,
        "Linear should start at initial"
    );
    assert!(
        (exponential - initial).abs() < f32::EPSILON,
        "Exponential should start at initial"
    );
}

/// Issue #1020: Exponential schedule decays faster initially than linear.
#[test]
fn exponential_cools_faster_initially() {
    let generation = 10;
    let total = 1000;
    let linear = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, generation, total);
    let exponential =
        compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, generation, total);

    // At generation 10/1000, linear = 1.0 * (1 - 0.01) = 0.99
    // exponential = 1.0 * 0.995^10 ≈ 0.951
    assert!(
        exponential < linear,
        "Exponential ({exponential}) should be lower than linear ({linear}) at early generations"
    );
}

/// Issue #1020: Default exponential decay rate is in valid range.
#[test]
fn exponential_decay_rate_is_valid() {
    let rate = DEFAULT_EXPONENTIAL_DECAY_RATE;
    assert!(rate > 0.0 && rate < 1.0, "Decay rate must be in (0, 1)");
}

// =============================================================================
// Threshold Scaling Tests
// =============================================================================

/// Issue #1020: Default temperature (1.0) preserves the original threshold.
#[test]
fn default_temperature_preserves_threshold() {
    let threshold = 0.05;
    let scaled = scale_threshold_by_temperature(threshold, DEFAULT_TEMPERATURE);
    assert!(
        (scaled - threshold).abs() < f32::EPSILON,
        "Default temperature should not change threshold: got {scaled}, expected {threshold}"
    );
}

/// Issue #1020: High temperature (exploration) lowers the effective threshold,
/// accepting more marginal candidates.
#[test]
fn high_temperature_increases_acceptance_via_threshold() {
    let threshold = 0.05;
    let high_temp = 2.0;
    let scaled = scale_threshold_by_temperature(threshold, high_temp);
    assert!(
        scaled < threshold,
        "High temperature should lower threshold: got {scaled}, expected < {threshold}"
    );
    // At temperature 2.0, threshold should be halved
    assert!(
        (scaled - 0.025).abs() < 1e-6,
        "At temp=2.0, threshold 0.05 should become 0.025, got {scaled}"
    );
}

/// Issue #1020: Low temperature (exploitation) raises the effective threshold,
/// only accepting strong candidates.
#[test]
fn low_temperature_decreases_acceptance_via_threshold() {
    let threshold = 0.05;
    let low_temp = 0.5;
    let scaled = scale_threshold_by_temperature(threshold, low_temp);
    assert!(
        scaled > threshold,
        "Low temperature should raise threshold: got {scaled}, expected > {threshold}"
    );
    // At temperature 0.5, threshold should be doubled
    assert!(
        (scaled - 0.1).abs() < 1e-6,
        "At temp=0.5, threshold 0.05 should become 0.1, got {scaled}"
    );
}

// =============================================================================
// Improved Ratio Scaling Tests
// =============================================================================

/// Issue #1020: Default temperature (1.0) preserves `MIN_IMPROVED_RATIO`.
#[test]
fn default_temperature_preserves_improved_ratio() {
    let scaled = scale_ratio_by_temperature(MIN_IMPROVED_RATIO, DEFAULT_TEMPERATURE);
    assert!(
        (scaled - MIN_IMPROVED_RATIO).abs() < f32::EPSILON,
        "Default temperature should not change ratio: got {scaled}, expected {MIN_IMPROVED_RATIO}"
    );
}

/// Issue #1020: High temperature lowers the effective ratio, letting more
/// candidates through (exploration).
#[test]
fn high_temperature_lowers_improved_ratio() {
    let ratio = 0.6;
    let high_temp = 2.0;
    let scaled = scale_ratio_by_temperature(ratio, high_temp);
    assert!(
        scaled < ratio,
        "High temperature should lower ratio: got {scaled}, expected < {ratio}"
    );
    assert!(
        (scaled - 0.3).abs() < 1e-6,
        "At temp=2.0, ratio 0.6 should become 0.3, got {scaled}"
    );
}

/// Issue #1020: Low temperature raises the effective ratio, filtering more
/// aggressively (exploitation). Clamped to 1.0.
#[test]
fn low_temperature_raises_improved_ratio_with_clamp() {
    let ratio = 0.6;
    let very_low_temp = 0.1;
    let scaled = scale_ratio_by_temperature(ratio, very_low_temp);
    assert!(scaled <= 1.0, "Ratio must be clamped to 1.0: got {scaled}");
    assert!(
        scaled > ratio,
        "Low temperature should raise ratio: got {scaled}, expected > {ratio}"
    );
}

// =============================================================================
// MH Temperature Scaling Tests
// =============================================================================

/// Issue #1020: Default temperature preserves MH temperature unchanged.
#[test]
fn default_temperature_preserves_mh_temperature() {
    let base_mh = 0.01;
    let scaled = scale_mh_temperature(base_mh, DEFAULT_TEMPERATURE);
    assert!(
        (scaled - base_mh).abs() < f32::EPSILON,
        "Default temperature should not change MH temp: got {scaled}, expected {base_mh}"
    );
}

/// Issue #1020: High schedule temperature increases effective MH temperature,
/// making probabilistic acceptance more willing to accept marginal candidates.
#[test]
fn high_temperature_increases_mh_acceptance() {
    let base_mh = 0.01;
    let high_temp = 3.0;
    let scaled = scale_mh_temperature(base_mh, high_temp);
    assert!(
        scaled > base_mh,
        "High schedule temp should increase MH temp: got {scaled}, expected > {base_mh}"
    );
    assert!(
        (scaled - 0.03).abs() < 1e-6,
        "At schedule temp=3.0, MH temp 0.01 should become 0.03, got {scaled}"
    );
}

/// Issue #1020: Low schedule temperature decreases effective MH temperature,
/// making probabilistic acceptance more selective.
#[test]
fn low_temperature_decreases_mh_acceptance() {
    let base_mh = 0.01;
    let low_temp = 0.5;
    let scaled = scale_mh_temperature(base_mh, low_temp);
    assert!(
        scaled < base_mh,
        "Low schedule temp should decrease MH temp: got {scaled}, expected < {base_mh}"
    );
}

// =============================================================================
// Temperature Bounds Tests
// =============================================================================

/// Issue #1020: Temperature is clamped to valid range in all scaling functions.
#[test]
fn extreme_temperatures_are_clamped() {
    // Very high temperature is clamped to MAX_TEMPERATURE
    let threshold = 0.05;
    let scaled_high = scale_threshold_by_temperature(threshold, 100.0);
    let expected_max = threshold / MAX_TEMPERATURE;
    assert!(
        (scaled_high - expected_max).abs() < 1e-6,
        "Very high temp should be clamped: got {scaled_high}, expected {expected_max}"
    );

    // Very low temperature is clamped to MIN_TEMPERATURE
    let scaled_low = scale_threshold_by_temperature(threshold, 0.0001);
    let expected_min = threshold / MIN_TEMPERATURE;
    assert!(
        (scaled_low - expected_min).abs() < 1e-3,
        "Very low temp should be clamped: got {scaled_low}, expected {expected_min}"
    );
}

/// Issue #1020: Zero temperature is handled safely (clamped to minimum).
#[test]
fn zero_temperature_does_not_cause_division_by_zero() {
    let threshold = 0.05;
    let scaled = scale_threshold_by_temperature(threshold, 0.0);
    assert!(
        scaled.is_finite(),
        "Zero temperature must not produce infinite threshold"
    );
    assert!(scaled > 0.0, "Scaled threshold must remain positive");
}

/// Issue #1020: Negative temperature is handled safely (clamped to minimum).
#[test]
fn negative_temperature_is_clamped_to_minimum() {
    let threshold = 0.05;
    let scaled = scale_threshold_by_temperature(threshold, -1.0);
    assert!(
        scaled.is_finite(),
        "Negative temperature must not produce NaN"
    );
    let expected = threshold / MIN_TEMPERATURE;
    assert!(
        (scaled - expected).abs() < 1e-3,
        "Negative temp should be clamped to minimum"
    );
}

// =============================================================================
// End-to-End Schedule Integration Tests
// =============================================================================

/// Issue #1020: Full cooling schedule with threshold scaling produces a
/// monotonically increasing effective threshold over generations.
#[test]
fn cooling_schedule_produces_increasing_effective_threshold() {
    let base_threshold = 0.05;
    let total_generations = 500;
    let mut prev_effective = 0.0;

    for generation in 0..=total_generations {
        let temp = compute_scheduled_temperature(
            CoolingSchedule::Exponential,
            2.0,
            generation,
            total_generations,
        );
        let effective = scale_threshold_by_temperature(base_threshold, temp);
        assert!(
            effective >= prev_effective - 1e-6,
            "Effective threshold should increase as temperature cools: generation {generation}, effective {effective} < prev {prev_effective}"
        );
        prev_effective = effective;
    }
}

/// Issue #1020: Full cooling schedule with ratio scaling produces a
/// monotonically increasing effective ratio over generations.
#[test]
fn cooling_schedule_produces_increasing_effective_ratio() {
    let base_ratio = 0.6;
    let total_generations = 500;
    let mut prev_effective = 0.0;

    for generation in 0..=total_generations {
        let temp = compute_scheduled_temperature(
            CoolingSchedule::Linear,
            2.0,
            generation,
            total_generations,
        );
        let effective = scale_ratio_by_temperature(base_ratio, temp);
        assert!(
            effective >= prev_effective - 1e-6,
            "Effective ratio should increase as temperature cools: generation {generation}, effective {effective} < prev {prev_effective}"
        );
        prev_effective = effective;
    }
}

/// Issue #1020: Default temperature constant is exactly 1.0.
#[test]
fn default_temperature_constant_is_one() {
    let default = DEFAULT_TEMPERATURE;
    assert!(
        (default - 1.0).abs() < f32::EPSILON,
        "DEFAULT_TEMPERATURE must be 1.0 for backward compatibility"
    );
}

/// Issue #1020: `MIN_TEMPERATURE` is positive and less than 1.0.
#[test]
fn min_temperature_bounds() {
    let min = MIN_TEMPERATURE;
    assert!(min > 0.0, "MIN_TEMPERATURE must be positive");
    assert!(min < 1.0, "MIN_TEMPERATURE must be less than 1.0");
}

/// Issue #1020: `MAX_TEMPERATURE` is at least 1.0.
#[test]
fn max_temperature_bounds() {
    let max = MAX_TEMPERATURE;
    assert!(max >= 1.0, "MAX_TEMPERATURE must be at least 1.0");
}
