//! Regression tests for f32 overflow protection in activation functions.
//!
//! Why this matters (24-Dec-2025):
//! - The implementation operates on `f32` values, where `ln(f32::MAX) ≈ 88.72`.
//! - Using an f64-scale cutoff (eg 709) allows `exp(x)` to overflow to infinity
//!   for inputs in ~[89, 709), producing `inf`/`-inf` instead of saturating to
//!   finite `f32::MAX` / `f32::MIN`.

#[test]
fn exponential_should_saturate_to_f32_max_before_overflow() {
    let x = 100.0_f32; // exp(100) overflows for f32
    let y = neat_ai_discovery::activations::apply_scalar_squash("EXPONENTIAL", x)
        .expect("EXPONENTIAL must be a scalar squash");
    assert!(
        y.is_finite(),
        "Expected EXPONENTIAL({x}) to be finite (saturated), got {y}"
    );
    assert_eq!(
        y,
        f32::MAX,
        "Expected EXPONENTIAL({x}) to saturate to f32::MAX"
    );
}

#[test]
fn exponential_target_simulation_should_saturate_to_f32_max_before_overflow() {
    let x = 100.0_f32; // exp(100) overflows for f32
    let f = neat_ai_discovery::activations::target_simulation_fn("EXPONENTIAL")
        .expect("EXPONENTIAL must have a target simulation function");
    let y = f(x);
    assert!(
        y.is_finite(),
        "Expected EXPONENTIAL target simulation({x}) to be finite (saturated), got {y}"
    );
    assert_eq!(
        y,
        f32::MAX,
        "Expected EXPONENTIAL target simulation({x}) to saturate to f32::MAX"
    );
}

#[test]
fn logsigmoid_should_saturate_to_f32_min_before_overflow() {
    let x = -100.0_f32; // exp(-x)=exp(100) overflows for f32
    let y = neat_ai_discovery::activations::apply_scalar_squash("LOGSIGMOID", x)
        .expect("LOGSIGMOID must be a scalar squash");
    assert!(
        y.is_finite(),
        "Expected LOGSIGMOID({x}) to be finite (saturated), got {y}"
    );
    assert_eq!(
        y,
        f32::MIN,
        "Expected LOGSIGMOID({x}) to saturate to f32::MIN"
    );
}

#[test]
fn logsigmoid_target_simulation_should_saturate_to_f32_min_before_overflow() {
    let x = -100.0_f32; // exp(-x)=exp(100) overflows for f32
    let f = neat_ai_discovery::activations::target_simulation_fn("LOGSIGMOID")
        .expect("LOGSIGMOID must have a target simulation function");
    let y = f(x);
    assert!(
        y.is_finite(),
        "Expected LOGSIGMOID target simulation({x}) to be finite (saturated), got {y}"
    );
    assert_eq!(
        y,
        f32::MIN,
        "Expected LOGSIGMOID target simulation({x}) to saturate to f32::MIN"
    );
}
