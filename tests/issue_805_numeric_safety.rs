//! Issue #805: Numeric safety guards for division-by-zero and overflow-prone casts.
//!
//! Tests edge cases for:
//! - Activation function f64→f32 casts producing finite results (EXPONENTIAL, SOFTPLUS)
//! - Calibration factor computation with edge-case observations
//! - Bias evaluation step guard (zero/negative step)

// =============================================================================
// Activation function f64→f32 cast safety (activations.rs)
// =============================================================================

/// EXPONENTIAL must return a finite value for extreme negative inputs
/// (f64 `exp()` underflows to 0.0, which must cast to finite f32).
#[test]
fn exponential_returns_finite_for_extreme_negative() {
    let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", f32::MIN)
        .expect("EXPONENTIAL must be a scalar squash");
    assert!(
        y.is_finite(),
        "EXPONENTIAL(f32::MIN) must be finite, got {y}"
    );
}

/// EXPONENTIAL target simulation must also handle extreme negatives.
#[test]
fn exponential_target_sim_returns_finite_for_extreme_negative() {
    let f = neat_ai_discovery::activations::target_simulation_fn("EXPONENTIAL")
        .expect("EXPONENTIAL must have a target simulation function");
    let y = f(f32::MIN);
    assert!(
        y.is_finite(),
        "EXPONENTIAL target_simulation_fn(f32::MIN) must be finite, got {y}"
    );
}

/// EXPONENTIAL must return finite for NaN, +inf, -inf inputs.
#[test]
fn exponential_returns_finite_for_non_finite_inputs() {
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", x)
            .expect("EXPONENTIAL must be a scalar squash");
        assert!(y.is_finite(), "EXPONENTIAL({x}) must be finite, got {y}");

        let f = neat_ai_discovery::activations::target_simulation_fn("EXPONENTIAL").unwrap();
        let y2 = f(x);
        assert!(
            y2.is_finite(),
            "EXPONENTIAL target_simulation_fn({x}) must be finite, got {y2}"
        );
    }
}

/// SOFTPLUS must return a finite value for extreme positive inputs
/// (below the cutoff but still large enough to produce large `exp()` values).
#[test]
fn softplus_returns_finite_for_large_positive() {
    let x = 700.0_f32; // below SOFTPLUS_CUTOFF (709) but large
    let y = neat_ai_discovery::activations::apply_scalar_squash("SOFTPLUS", x)
        .expect("SOFTPLUS must be a scalar squash");
    assert!(y.is_finite(), "SOFTPLUS({x}) must be finite, got {y}");
}

/// SOFTPLUS target simulation must also handle large values.
#[test]
fn softplus_target_sim_returns_finite_for_large_positive() {
    let f = neat_ai_discovery::activations::target_simulation_fn("SOFTPLUS")
        .expect("SOFTPLUS must have a target simulation function");
    let y = f(700.0);
    assert!(
        y.is_finite(),
        "SOFTPLUS target_simulation_fn(700.0) must be finite, got {y}"
    );
}

/// SOFTPLUS must return finite for NaN, +inf, -inf inputs.
#[test]
fn softplus_returns_finite_for_non_finite_inputs() {
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let y = neat_ai_discovery::activations::apply_scalar_squash("SOFTPLUS", x)
            .expect("SOFTPLUS must be a scalar squash");
        assert!(y.is_finite(), "SOFTPLUS({x}) must be finite, got {y}");

        let f = neat_ai_discovery::activations::target_simulation_fn("SOFTPLUS").unwrap();
        let y2 = f(x);
        assert!(
            y2.is_finite(),
            "SOFTPLUS target_simulation_fn({x}) must be finite, got {y2}"
        );
    }
}

// =============================================================================
// Calibration factor edge cases (discovery_history.rs)
// =============================================================================

/// Calibration factor for an unknown module returns 1.0 (no correction).
#[test]
fn calibration_factor_unknown_module_returns_one() {
    let history = neat_ai_discovery::discovery_history::DiscoveryHistory::new();
    let factor = history.calibration_factor("nonexistent", "addSynapse");
    assert!(
        (factor - 1.0).abs() < f64::EPSILON,
        "Calibration factor for unknown module should be 1.0, got {factor}"
    );
}

/// Calibration factor handles near-zero predicted values without division by zero.
#[test]
fn calibration_factor_near_zero_predicted_no_div_by_zero() {
    let mut history = neat_ai_discovery::discovery_history::DiscoveryHistory::new();
    // Record prediction where predicted is near zero
    history.record_calibration("test_module", "addSynapse", 0.0, 0.5);
    let factor = history.calibration_factor("test_module", "addSynapse");
    assert!(
        factor.is_finite(),
        "Calibration factor must be finite when predicted is ~0, got {factor}"
    );
    assert!(
        factor > 0.0,
        "Calibration factor must be positive, got {factor}"
    );
}

/// Calibration factor handles multiple near-zero predictions gracefully.
#[test]
fn calibration_factor_all_near_zero_predictions() {
    let mut history = neat_ai_discovery::discovery_history::DiscoveryHistory::new();
    for i in 0..5 {
        history.record_calibration("test_module", "addSynapse", 1e-15, i as f64 * 0.1);
    }
    let factor = history.calibration_factor("test_module", "addSynapse");
    assert!(
        factor.is_finite(),
        "Calibration factor must be finite with all near-zero predictions, got {factor}"
    );
}
