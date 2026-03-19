//! Regression tests for f32 overflow protection in activation functions.
//!
//! Why this matters:
//! - The implementation operates on `f32` values.
//! - EXPONENTIAL must saturate to a finite value for large inputs.
//! - Issue #323: Behaviour matches NEAT-AI WASM implementation:
//!   - For x >= 36.0, returns `JS_MAX_SAFE_INTEGER` (~9e15) instead of `f32::MAX`.
//!   - This ensures consistency between Discovery and NEAT-AI.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
const JS_MAX_SAFE_INTEGER: f32 = 9_007_199_254_740_992.0;

#[test]
fn exponential_should_saturate_for_large_inputs() {
    let x = 100.0_f32; // Beyond cutoff of 36.0
    let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", x)
        .expect("EXPONENTIAL must be a scalar squash");
    assert!(
        y.is_finite(),
        "Expected EXPONENTIAL({x}) to be finite (saturated), got {y}"
    );
    // Issue #323: Now saturates to JS_MAX_SAFE_INTEGER to match NEAT-AI WASM
    assert!(
        (y - JS_MAX_SAFE_INTEGER).abs() < 1.0,
        "Expected EXPONENTIAL({x}) to saturate to JS_MAX_SAFE_INTEGER (~9e15), got {y}"
    );
}

#[test]
fn exponential_target_simulation_should_saturate_for_large_inputs() {
    let x = 100.0_f32; // Beyond cutoff of 36.0
    let f = neat_ai_discovery::activations::target_simulation_fn("EXPONENTIAL")
        .expect("EXPONENTIAL must have a target simulation function");
    let y = f(x);
    assert!(
        y.is_finite(),
        "Expected EXPONENTIAL target simulation({x}) to be finite (saturated), got {y}"
    );
    // Issue #323: Now saturates to JS_MAX_SAFE_INTEGER to match NEAT-AI WASM
    assert!(
        (y - JS_MAX_SAFE_INTEGER).abs() < 1.0,
        "Expected EXPONENTIAL target simulation({x}) to saturate to JS_MAX_SAFE_INTEGER (~9e15), got {y}"
    );
}

#[test]
fn exponential_at_cutoff_boundary() {
    // At x = 36.0, should return JS_MAX_SAFE_INTEGER
    let x = 36.0_f32;
    let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", x)
        .expect("EXPONENTIAL must be a scalar squash");
    assert!(
        (y - JS_MAX_SAFE_INTEGER).abs() < 1.0,
        "Expected EXPONENTIAL({x}) at cutoff to be JS_MAX_SAFE_INTEGER, got {y}"
    );

    // At x = 35.999, should return actual exp(x)
    let x = 35.999_f32;
    let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", x)
        .expect("EXPONENTIAL must be a scalar squash");
    let expected = ((x as f64).exp()) as f32;
    assert!(
        (y - expected).abs() / expected.abs() < 1e-5,
        "Expected EXPONENTIAL({x}) below cutoff to be exp(x), got {y}, expected {expected}"
    );
}

#[test]
fn logsigmoid_should_approach_x_for_large_negative_inputs() {
    // LOGSIGMOID(x) = -ln(1 + exp(-x)) has asymptotic behaviour LOGSIGMOID(x) → x as x → -∞.
    //
    // We also want overflow protection: a naive implementation that computes exp(-x) for
    // x=-100 would attempt exp(100) which overflows for f32. The stable formulation avoids
    // this and should return a value very close to x.
    let x = -100.0_f32;
    let y = neat_ai_discovery::activations::apply_scalar_squash("LOGSIGMOID", x)
        .expect("LOGSIGMOID must be a scalar squash");
    assert!(
        y.is_finite(),
        "Expected LOGSIGMOID({x}) to be finite, got {y}"
    );
    assert!(
        (y - x).abs() < 1e-3,
        "Expected LOGSIGMOID({x}) ≈ {x} for large negative x, got {y}"
    );
}

#[test]
fn logsigmoid_target_simulation_should_approach_x_for_large_negative_inputs() {
    // Target simulation must match the scalar squash behaviour for large negative inputs.
    let x = -100.0_f32;
    let f = neat_ai_discovery::activations::target_simulation_fn("LOGSIGMOID")
        .expect("LOGSIGMOID must have a target simulation function");
    let y = f(x);
    assert!(
        y.is_finite(),
        "Expected LOGSIGMOID target simulation({x}) to be finite, got {y}"
    );
    assert!(
        (y - x).abs() < 1e-3,
        "Expected LOGSIGMOID target simulation({x}) ≈ {x} for large negative x, got {y}"
    );
}
