//! Issue #323: Compare squash functions with NEAT-AI WASM implementation.
//!
//! This test ensures our squash functions produce results consistent with the
//! NEAT-AI WASM implementation (wasm_activation/src/lib.rs). Differences in
//! squash function behaviour can cause failed candidates during discovery
//! because predictions won't match actual outcomes.
//!
//! Reference: https://github.com/stSoftwareAU/NEAT-AI

use neat_ai_discovery::activations::apply_scalar_squash;

/// Helper to compare f32 values with appropriate tolerance.
/// Uses relative tolerance for larger values and absolute tolerance for smaller values.
fn approx_eq(a: f32, b: f32, rel_tol: f32, abs_tol: f32) -> bool {
    if a == b {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return a.is_nan() && b.is_nan()
            || a.is_infinite() && b.is_infinite() && a.signum() == b.signum();
    }
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs());
    diff <= abs_tol || diff <= rel_tol * max_val
}

/// Assert that squash values are approximately equal.
fn assert_squash_eq(name: &str, x: f32, expected: f32) {
    let actual =
        apply_scalar_squash(name, x).unwrap_or_else(|| panic!("{name} should be a scalar squash"));
    let rel_tol = 1e-5;
    let abs_tol = 1e-6;

    assert!(
        approx_eq(actual, expected, rel_tol, abs_tol),
        "{name}({x}) = {actual}, expected {expected} (diff = {})",
        (actual - expected).abs()
    );
}

// =============================================================================
// NEAT-AI WASM Reference Implementation (from wasm_activation/src/lib.rs)
// =============================================================================
// These functions replicate the exact NEAT-AI WASM implementation for comparison.

const SELU_ALPHA: f32 = 1.673_263_2;
const SELU_LAMBDA: f32 = 1.050_701;
const GELU_COEFF: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
const LEAKY_RELU_ALPHA: f32 = 0.01;
const JS_MAX_SAFE_INTEGER: f32 = 9_007_199_254_740_992.0;

fn neat_ai_identity(x: f32) -> f32 {
    x
}

fn neat_ai_relu(x: f32) -> f32 {
    x.max(0.0)
}

fn neat_ai_relu6(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}

fn neat_ai_leaky_relu(x: f32) -> f32 {
    if x >= 0.0 { x } else { LEAKY_RELU_ALPHA * x }
}

fn neat_ai_selu(x: f32) -> f32 {
    if !x.is_finite() {
        return -JS_MAX_SAFE_INTEGER;
    }
    let safe_x = (x as f64).min(709.0);
    let fx = if safe_x > 0.0 {
        safe_x
    } else {
        (SELU_ALPHA as f64) * safe_x.exp() - (SELU_ALPHA as f64)
    };
    ((SELU_LAMBDA as f64) * fx) as f32
}

fn neat_ai_elu(x: f32) -> f32 {
    // Note: NEAT-AI WASM uses `x > 0.0` not `x >= 0.0`
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

fn neat_ai_logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn neat_ai_tanh(x: f32) -> f32 {
    x.tanh()
}

fn neat_ai_hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

fn neat_ai_softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

fn neat_ai_softplus(x: f32) -> f32 {
    if !x.is_finite() {
        return 1e-15;
    }
    if x >= 709.0 {
        return 100.0;
    }
    ((1.0f64 + (x as f64).exp()).ln()) as f32
}

fn neat_ai_swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn neat_ai_mish(x: f32) -> f32 {
    x * (1.0 + x.exp()).ln().tanh()
}

fn neat_ai_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh())
}

fn neat_ai_sine(x: f32) -> f32 {
    x.sin()
}

fn neat_ai_cosine(x: f32) -> f32 {
    ((x as f64).cos()) as f32
}

fn neat_ai_tan(x: f32) -> f32 {
    ((x as f64).tan()) as f32
}

fn neat_ai_arctan(x: f32) -> f32 {
    x.atan()
}

fn neat_ai_gaussian(x: f32) -> f32 {
    (-x * x).exp()
}

fn neat_ai_bent_identity(x: f32) -> f32 {
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

fn neat_ai_bipolar_sigmoid(x: f32) -> f32 {
    2.0 / (1.0 + (-x).exp()) - 1.0
}

fn neat_ai_bipolar(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { -1.0 }
}

fn neat_ai_step(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn neat_ai_complement(x: f32) -> f32 {
    1.0 - x
}

fn neat_ai_absolute(x: f32) -> f32 {
    x.abs()
}

fn neat_ai_square(x: f32) -> f32 {
    let xf = x as f64;
    (xf * xf) as f32
}

fn neat_ai_cube(x: f32) -> f32 {
    let xf = x as f64;
    (xf * xf * xf) as f32
}

fn neat_ai_sqrt(x: f32) -> f32 {
    if x >= 0.0 { x.sqrt() } else { 0.0 }
}

fn neat_ai_std_inverse(x: f32) -> f32 {
    if x.abs() < 1e-10 {
        if x >= 0.0 { 1e10 } else { -1e10 }
    } else {
        1.0 / x
    }
}

fn neat_ai_exponential(x: f32) -> f32 {
    if !x.is_finite() || x >= 36.0 {
        JS_MAX_SAFE_INTEGER
    } else {
        ((x as f64).exp()) as f32
    }
}

fn neat_ai_logsigmoid(x: f32) -> f32 {
    if !x.is_finite() || x <= -709.0 {
        return -JS_MAX_SAFE_INTEGER;
    }
    let xf = x as f64;
    let exp_neg_x = (-xf).exp();
    (-(1.0f64 + exp_neg_x).ln()) as f32
}

fn neat_ai_isru(x: f32) -> f32 {
    x / (1.0 + x * x).sqrt()
}

// =============================================================================
// Tests comparing NEAT-AI-Discovery vs NEAT-AI WASM implementations
// =============================================================================

#[test]
fn test_identity_consistency() {
    for x in [-100.0, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0] {
        assert_squash_eq("IDENTITY", x, neat_ai_identity(x));
    }
}

#[test]
fn test_relu_consistency() {
    for x in [-100.0, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0] {
        assert_squash_eq("RELU", x, neat_ai_relu(x));
    }
}

#[test]
fn test_relu6_consistency() {
    for x in [-1.0, 0.0, 3.0, 6.0, 7.0, 100.0] {
        assert_squash_eq("RELU6", x, neat_ai_relu6(x));
    }
}

#[test]
fn test_leaky_relu_consistency() {
    for x in [-100.0, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0] {
        assert_squash_eq("LEAKYRELU", x, neat_ai_leaky_relu(x));
    }
}

#[test]
fn test_selu_consistency() {
    for x in [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0] {
        assert_squash_eq("SELU", x, neat_ai_selu(x));
    }
}

#[test]
fn test_elu_consistency() {
    // Note: This test might fail if implementations differ at x=0
    for x in [-10.0, -1.0, -0.5, -0.001, 0.001, 0.5, 1.0, 10.0] {
        assert_squash_eq("ELU", x, neat_ai_elu(x));
    }
}

/// Test ELU at x=0 specifically - this is where implementations may differ.
/// NEAT-AI WASM uses `x > 0.0` (returns exp(0)-1 = 0 for x=0).
/// NEAT-AI-Discovery uses `x >= 0.0` (returns 0 for x=0).
/// Both result in 0, but for different reasons.
#[test]
fn test_elu_at_zero() {
    let x = 0.0;
    let neat_ai_result = neat_ai_elu(x); // exp(0) - 1 = 0
    let discovery_result = apply_scalar_squash("ELU", x).unwrap();

    // Both should return 0 (even if via different code paths)
    assert!(
        (discovery_result - neat_ai_result).abs() < 1e-10,
        "ELU(0): discovery={discovery_result}, neat_ai={neat_ai_result} - must match",
    );
}

#[test]
fn test_logistic_consistency() {
    for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        assert_squash_eq("LOGISTIC", x, neat_ai_logistic(x));
    }
}

#[test]
fn test_tanh_consistency() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        assert_squash_eq("TANH", x, neat_ai_tanh(x));
    }
}

#[test]
fn test_hard_tanh_consistency() {
    for x in [-100.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 100.0] {
        assert_squash_eq("HARD_TANH", x, neat_ai_hard_tanh(x));
    }
    // Test alias
    for x in [-1.5, 0.0, 1.5] {
        assert_squash_eq("CLIPPED", x, neat_ai_hard_tanh(x));
    }
}

#[test]
fn test_softsign_consistency() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        assert_squash_eq("SOFTSIGN", x, neat_ai_softsign(x));
    }
}

#[test]
fn test_softplus_consistency() {
    for x in [-100.0, -10.0, 0.0, 10.0, 20.0] {
        assert_squash_eq("SOFTPLUS", x, neat_ai_softplus(x));
    }
}

/// Test SOFTPLUS at extreme values.
/// Issue #323: This test enforces consistency with NEAT-AI WASM.
#[test]
fn test_softplus_extreme_values() {
    // At x >= 709, NEAT-AI WASM returns 100.0
    let x = 709.0;
    let neat_ai_result = neat_ai_softplus(x);
    assert!(
        (neat_ai_result - 100.0).abs() < 1e-6,
        "NEAT-AI SOFTPLUS(709) should return 100.0, got {neat_ai_result}",
    );

    // Discovery must match NEAT-AI behaviour
    let discovery_result = apply_scalar_squash("SOFTPLUS", x).unwrap();
    assert!(
        approx_eq(discovery_result, neat_ai_result, 1e-5, 1e-6),
        "SOFTPLUS(709): discovery={discovery_result} != neat_ai={neat_ai_result}",
    );

    // Test normal range still works
    let x = 100.0;
    let neat_ai_result = neat_ai_softplus(x);
    let discovery_result = apply_scalar_squash("SOFTPLUS", x).unwrap();
    assert!(
        approx_eq(discovery_result, neat_ai_result, 1e-5, 1e-6),
        "SOFTPLUS(100): discovery={discovery_result} != neat_ai={neat_ai_result}",
    );
}

#[test]
fn test_swish_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("SWISH", x, neat_ai_swish(x));
    }
}

#[test]
fn test_mish_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("MISH", x, neat_ai_mish(x));
    }
}

#[test]
fn test_gelu_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("GELU", x, neat_ai_gelu(x));
    }
}

#[test]
fn test_sine_consistency() {
    for x in [-std::f32::consts::PI, -1.0, 0.0, 1.0, std::f32::consts::PI] {
        assert_squash_eq("SINE", x, neat_ai_sine(x));
    }
    // Test alias
    for x in [0.0, 1.0] {
        assert_squash_eq("SINUSOID", x, neat_ai_sine(x));
    }
}

#[test]
fn test_cosine_consistency() {
    for x in [-std::f32::consts::PI, -1.0, 0.0, 1.0, std::f32::consts::PI] {
        assert_squash_eq("COSINE", x, neat_ai_cosine(x));
    }
}

#[test]
fn test_tan_consistency() {
    // Avoid values near asymptotes (multiples of pi/2)
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        assert_squash_eq("TAN", x, neat_ai_tan(x));
    }
}

#[test]
fn test_arctan_consistency() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        assert_squash_eq("ARCTAN", x, neat_ai_arctan(x));
    }
}

#[test]
fn test_gaussian_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("GAUSSIAN", x, neat_ai_gaussian(x));
    }
}

/// Test GAUSSIAN at extreme values where the clamping may differ.
#[test]
fn test_gaussian_extreme_values() {
    // At x = 100, both should produce a value very close to 0
    let x = 100.0;
    let neat_ai_result = neat_ai_gaussian(x);
    let discovery_result = apply_scalar_squash("GAUSSIAN", x).unwrap();

    // Both should be effectively 0 (underflow)
    assert!(
        discovery_result < 1e-30,
        "GAUSSIAN(100) should underflow to ~0"
    );
    assert!(
        neat_ai_result < 1e-30,
        "NEAT-AI GAUSSIAN(100) should underflow to ~0"
    );
}

#[test]
fn test_bent_identity_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("BENT_IDENTITY", x, neat_ai_bent_identity(x));
    }
}

#[test]
fn test_bipolar_sigmoid_consistency() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        assert_squash_eq("BIPOLAR_SIGMOID", x, neat_ai_bipolar_sigmoid(x));
    }
}

#[test]
fn test_bipolar_consistency() {
    for x in [-100.0, -0.001, 0.0, 0.001, 100.0] {
        assert_squash_eq("BIPOLAR", x, neat_ai_bipolar(x));
    }
}

#[test]
fn test_step_consistency() {
    for x in [-100.0, -0.001, 0.0, 0.001, 100.0] {
        assert_squash_eq("STEP", x, neat_ai_step(x));
    }
}

#[test]
fn test_complement_consistency() {
    for x in [-1.0, 0.0, 0.5, 1.0, 2.0] {
        assert_squash_eq("COMPLEMENT", x, neat_ai_complement(x));
    }
    // Test alias
    for x in [0.0, 0.5, 1.0] {
        assert_squash_eq("INVERSE", x, neat_ai_complement(x));
    }
}

#[test]
fn test_absolute_consistency() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        assert_squash_eq("ABSOLUTE", x, neat_ai_absolute(x));
    }
}

#[test]
fn test_square_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("SQUARE", x, neat_ai_square(x));
    }
}

#[test]
fn test_cube_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("CUBE", x, neat_ai_cube(x));
    }
}

#[test]
fn test_sqrt_consistency() {
    for x in [-1.0, 0.0, 0.25, 1.0, 4.0, 100.0] {
        assert_squash_eq("SQRT", x, neat_ai_sqrt(x));
    }
}

#[test]
fn test_std_inverse_consistency() {
    // Test normal values
    for x in [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0] {
        assert_squash_eq("STDINVERSE", x, neat_ai_std_inverse(x));
    }
}

/// Test STDINVERSE near zero - epsilon must match NEAT-AI (1e-10).
/// Issue #323: This test enforces consistency with NEAT-AI WASM.
#[test]
fn test_std_inverse_near_zero() {
    // NEAT-AI uses 1e-10 epsilon
    // Test values that trigger epsilon handling
    let test_values = [1e-12, -1e-12, 1e-16, -1e-16];

    for x in test_values {
        let neat_ai_result = neat_ai_std_inverse(x);
        let discovery_result = apply_scalar_squash("STDINVERSE", x).unwrap();

        // Must match NEAT-AI behaviour
        assert!(
            approx_eq(discovery_result, neat_ai_result, 1e-5, 1e-6),
            "STDINVERSE({x:e}): discovery={discovery_result:e} != neat_ai={neat_ai_result:e}",
        );
    }
}

#[test]
fn test_exponential_consistency() {
    // Test normal range
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0, 35.0] {
        assert_squash_eq("EXPONENTIAL", x, neat_ai_exponential(x));
    }
}

/// Test EXPONENTIAL at x >= 36 where NEAT-AI clamps to JS_MAX_SAFE_INTEGER.
/// Issue #323: This test enforces consistency with NEAT-AI WASM.
#[test]
fn test_exponential_overflow_handling() {
    // NEAT-AI uses x >= 36.0 → JS_MAX_SAFE_INTEGER
    let x = 36.0;
    let neat_ai_result = neat_ai_exponential(x);
    let discovery_result = apply_scalar_squash("EXPONENTIAL", x).unwrap();

    // Must match NEAT-AI behaviour - clamp at x >= 36.0 to JS_MAX_SAFE_INTEGER
    assert!(
        approx_eq(discovery_result, neat_ai_result, 1e-5, 1e-6),
        "EXPONENTIAL(36): discovery={discovery_result:e} != neat_ai={neat_ai_result:e}",
    );

    // Also test at x = 100 to confirm clamping
    let x = 100.0;
    let neat_ai_result = neat_ai_exponential(x);
    let discovery_result = apply_scalar_squash("EXPONENTIAL", x).unwrap();
    assert!(
        approx_eq(discovery_result, neat_ai_result, 1e-5, 1e-6),
        "EXPONENTIAL(100): discovery={discovery_result:e} != neat_ai={neat_ai_result:e}",
    );
}

#[test]
fn test_logsigmoid_consistency() {
    // Test normal range
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("LOGSIGMOID", x, neat_ai_logsigmoid(x));
    }
}

#[test]
fn test_isru_consistency() {
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert_squash_eq("ISRU", x, neat_ai_isru(x));
    }
}
