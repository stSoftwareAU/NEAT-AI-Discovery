//! Activation (squash) functions and name handling.
//!
//! This module exists to keep squash name handling consistent between:
//! - the TypeScript `NEAT-AI` activation registry (`src/methods/activations`), and
//! - this Rust discovery library.
//!
//! Why this matters (Dec 2025):
//! - Discovery records store `value` (pre-activation) and `activation` (post-squash) for each neuron.
//! - Our analysis uses **value-domain** errors (see README: "VALUE domain error interpretation").
//! - To make meaningful predictions for *any* target squash (not just ReLU/TANH families),
//!   we must be able to re-apply the target squash when simulating candidate contributions.
//!
//! Notes:
//! - Some "activations" in NEAT-AI are actually **aggregate** neurons (eg MINIMUM/MAXIMUM/IF/HYPOT)
//!   which cannot be expressed as a pure `f(value)`; we treat them as non-scalar here.
//! - Name matching is case-insensitive and includes NEAT-AI aliases (eg CLIPPED, RELU, INVERSE, SINUSOID).

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::borrow::Cow;

/// JavaScript's `Number.MAX_SAFE_INTEGER` (~9.007e15).
///
/// Used as an upper bound for several activations (e.g. EXPONENTIAL) to match
/// the NEAT-AI WASM implementation behaviour.
const JS_MAX_SAFE_INTEGER: f32 = 9_007_199_254_740_992.0;

/// Cutoff for EXPONENTIAL to match NEAT-AI WASM.
///
/// At x >= 36.0, EXPONENTIAL returns `JS_MAX_SAFE_INTEGER` to prevent runaway growth
/// and match the TypeScript/WASM behaviour.
const EXPONENTIAL_CUTOFF: f32 = 36.0;

/// Cutoff for SOFTPLUS to match NEAT-AI WASM.
///
/// At x >= 709.0, ln(1+exp(x)) would overflow, so we return a large constant.
/// NEAT-AI WASM uses 100.0 as the large threshold return value.
const SOFTPLUS_CUTOFF: f32 = 709.0;

/// Large threshold return value for SOFTPLUS.
const SOFTPLUS_LARGE_THRESHOLD: f32 = 100.0;

/// Small threshold return value for SOFTPLUS (non-finite inputs).
const SOFTPLUS_SMALL_THRESHOLD: f32 = 1e-15;

/// Epsilon for STDINVERSE to match NEAT-AI WASM.
///
/// Values with |x| < epsilon are clamped to ±epsilon before computing 1/x.
const STDINVERSE_EPSILON: f32 = 1e-10;

/// Uppercase, canonical-ish name for an activation.
///
/// We normalise to an uppercase string so matching is case-insensitive and robust to
/// historical naming differences between projects (eg `Softplus` vs `SOFTPLUS`).
pub fn normalise_squash_name(name: &str) -> Cow<'_, str> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Cow::Borrowed("");
    }
    // Fast path: already uppercase ASCII.
    if trimmed.bytes().all(|b| !b.is_ascii_lowercase()) {
        return Cow::Borrowed(trimmed);
    }
    Cow::Owned(trimmed.to_ascii_uppercase())
}

/// Returns true if this library explicitly recognises the squash name (including aliases).
///
/// This does **not** mean we can perfectly simulate it as `f(value)` (aggregate squashes are
/// recognised but not scalar), only that we won't treat it as "unknown".
pub fn is_known_squash_name(name: &str) -> bool {
    let n = normalise_squash_name(name);
    matches!(
        n.as_ref(),
        // --- Core scalar squashes (types/) ---
        "ABSOLUTE"
            | "ARCTAN"
            | "BENT_IDENTITY"
            | "BIPOLAR"
            | "BIPOLAR_SIGMOID"
            | "COMPLEMENT"
            | "INVERSE" // alias for COMPLEMENT
            | "COSINE"
            | "CUBE"
            | "ELU"
            | "EXPONENTIAL"
            | "GAUSSIAN"
            | "GELU"
            | "HARD_TANH"
            | "CLIPPED" // alias for HARD_TANH
            | "IDENTITY"
            | "ISRU"
            | "LEAKYRELU"
            | "LOGISTIC"
            | "LOGSIGMOID"
            | "MISH"
            | "RELU"
            | "RELU6"
            | "SELU"
            | "SINE"
            | "SINUSOID" // alias for SINE
            | "SOFTPLUS"
            | "SOFTSIGN"
            | "SQRT"
            | "SQUARE"
            | "STDINVERSE"
            | "STEP"
            | "SWISH"
            | "TAN"
            | "TANH"
            // --- Aggregate-ish squashes (aggregate/, deprecated/, etc.) ---
            | "IF"
            | "MAXIMUM"
            | "MINIMUM"
            | "MEAN"
            | "HYPOT"
            | "HYPOTV2"
    )
}

/// Returns true if the squash is an *aggregate* neuron (not a pure `f(value)`).
pub fn is_aggregate_squash(name: &str) -> bool {
    let n = normalise_squash_name(name);
    matches!(
        n.as_ref(),
        "IF" | "MAXIMUM" | "MINIMUM" | "MEAN" | "HYPOT" | "HYPOTV2"
    )
}

/// Apply a scalar squash `f(x)` for the given activation name.
///
/// Returns `None` for aggregate squashes (eg MINIMUM/MAXIMUM/IF/HYPOT) because they are not
/// representable as a simple scalar function of a pre-activation value.
pub fn apply_scalar_squash(name: &str, x: f32) -> Option<f32> {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        "ABSOLUTE" => Some(x.abs()),
        "ARCTAN" => Some(x.atan()),
        "BENT_IDENTITY" => Some(((x * x + 1.0).sqrt() - 1.0) / 2.0 + x),
        "BIPOLAR" => Some(if x > 0.0 { 1.0 } else { -1.0 }),
        "BIPOLAR_SIGMOID" => Some(2.0 / (1.0 + (-x).exp()) - 1.0),
        "COMPLEMENT" | "INVERSE" => Some(1.0 - x),
        "COSINE" => Some(x.cos()),
        "CUBE" => Some(x * x * x),
        "ELU" => Some(if x >= 0.0 { x } else { x.exp() - 1.0 }),
        "EXPONENTIAL" => {
            // Match NEAT-AI WASM behaviour (Issue #323):
            // - For non-finite x or x >= 36.0, return JS_MAX_SAFE_INTEGER.
            if !x.is_finite() || x >= EXPONENTIAL_CUTOFF {
                Some(JS_MAX_SAFE_INTEGER)
            } else {
                let result = ((x as f64).exp()) as f32;
                // Belt-and-suspenders: guard against non-finite f64→f32 cast (Issue #805)
                Some(if result.is_finite() {
                    result
                } else {
                    JS_MAX_SAFE_INTEGER
                })
            }
        }
        "GAUSSIAN" => {
            // Match NEAT-AI: clamp to abs(x) <= 100 to avoid underflow noise.
            let safe_x = x.abs().min(100.0);
            Some((-safe_x * safe_x).exp())
        }
        "GELU" => {
            // GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x^3)))
            let x3 = x * x * x;
            let tanh_arg = 0.797_884_6_f32 * (x + 0.044_715_f32 * x3);
            Some(0.5 * x * (1.0 + tanh_arg.tanh()))
        }
        "HARD_TANH" | "CLIPPED" => Some(x.clamp(-1.0, 1.0)),
        "IDENTITY" => Some(x),
        "ISRU" => {
            // NEAT-AI uses α=1.0: x / sqrt(1 + x^2)
            Some(x / (1.0 + x * x).sqrt())
        }
        "LEAKYRELU" => Some(if x >= 0.0 { x } else { 0.01 * x }),
        "LOGISTIC" => {
            // Numerically stable logistic.
            if x >= 0.0 {
                Some(1.0 / (1.0 + (-x).exp()))
            } else {
                let exp_x = x.exp();
                Some(exp_x / (1.0 + exp_x))
            }
        }
        "LOGSIGMOID" => {
            // LOGSIGMOID(x) = -ln(1 + exp(-x))
            //
            // Numerically-stable formulation (important for f32):
            // - For x >= 0:  -ln(1 + exp(-x))        = -ln1p(exp(-x))  (no overflow; exp(-x) <= 1)
            // - For x < 0:   -ln(1 + exp(-x))
            //              = -ln(exp(-x) * (1 + exp(x)))
            //              = -(-x + ln(1 + exp(x)))
            //              = x - ln1p(exp(x))        (no overflow; exp(x) <= 1)
            //
            // This also preserves the correct asymptote: LOGSIGMOID(x) → x as x → -∞.
            if x >= 0.0 {
                Some(-(-x).exp().ln_1p())
            } else {
                Some(x - x.exp().ln_1p())
            }
        }
        "MISH" => {
            // Mish(x) = x * tanh(softplus(x))
            // Match NEAT-AI WASM behaviour - no cutoff needed since tanh() saturates
            Some(x * (1.0 + x.exp()).ln().tanh())
        }
        "RELU" => Some(x.max(0.0)),
        "RELU6" => Some(x.clamp(0.0, 6.0)),
        "SELU" => {
            // Standard SELU parameters.
            const ALPHA: f32 = 1.673_263_2;
            const LAMBDA: f32 = 1.050_701;
            Some(if x >= 0.0 {
                LAMBDA * x
            } else {
                LAMBDA * ALPHA * (x.exp() - 1.0)
            })
        }
        "SINE" | "SINUSOID" => Some(x.sin()),
        "SOFTPLUS" => {
            // Match NEAT-AI WASM behaviour (Issue #323):
            // - Non-finite x returns SMALL_THRESHOLD (1e-15).
            // - For x >= 709.0, clamp to LARGE_THRESHOLD (100.0).
            if !x.is_finite() {
                Some(SOFTPLUS_SMALL_THRESHOLD)
            } else if x >= SOFTPLUS_CUTOFF {
                Some(SOFTPLUS_LARGE_THRESHOLD)
            } else {
                let result = ((1.0f64 + (x as f64).exp()).ln()) as f32;
                // Belt-and-suspenders: guard against non-finite f64→f32 cast (Issue #805)
                Some(if result.is_finite() {
                    result
                } else {
                    SOFTPLUS_LARGE_THRESHOLD
                })
            }
        }
        "SOFTSIGN" => Some(x / (1.0 + x.abs())),
        "SQRT" => Some(if x.is_finite() && x >= 0.0 {
            x.sqrt()
        } else {
            0.0
        }),
        "SQUARE" => Some(x * x),
        "STDINVERSE" => {
            // Match NEAT-AI WASM behaviour (Issue #323): 1/x with epsilon protection.
            // Uses 1e-10 as epsilon to match WASM implementation.
            let safe_x = if x.abs() < STDINVERSE_EPSILON {
                if x >= 0.0 {
                    STDINVERSE_EPSILON
                } else {
                    -STDINVERSE_EPSILON
                }
            } else {
                x
            };
            Some(1.0 / safe_x)
        }
        "STEP" => Some(if x > 0.0 { 1.0 } else { 0.0 }),
        "SWISH" => {
            // Swish(x) = x * sigmoid(x)
            let sigmoid = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            };
            Some(x * sigmoid)
        }
        "TAN" => Some(x.tan()),
        "TANH" => Some(x.tanh()),

        // Aggregate or unknown.
        _ => None,
    }
}

/// Returns a function pointer for target simulation.
///
/// This is a subset of `apply_scalar_squash` for performance: it returns `fn(f32) -> f32`
/// so hot loops can avoid repeated string matching.
///
/// Notes:
/// - We intentionally return `None` for aggregate squashes (MINIMUM, MAXIMUM, etc.)
///   because they cannot be represented as `f(value)`.
/// - STEP and BIPOLAR both return their threshold functions. While these are discrete,
///   the simulation correctly predicts output flips when a synapse contribution crosses
///   the zero threshold (see README: "Threshold-crossing model for STEP/BIPOLAR").
pub fn target_simulation_fn(name: &str) -> Option<fn(f32) -> f32> {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        "ABSOLUTE" => Some(f32::abs),
        "ARCTAN" => Some(f32::atan),
        "BENT_IDENTITY" => Some(|x| ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x),
        "BIPOLAR" => Some(|x| if x > 0.0 { 1.0 } else { -1.0 }),
        "BIPOLAR_SIGMOID" => Some(|x| 2.0 / (1.0 + (-x).exp()) - 1.0),
        "COMPLEMENT" | "INVERSE" => Some(|x| 1.0 - x),
        "COSINE" => Some(f32::cos),
        "CUBE" => Some(|x| x * x * x),
        "ELU" => Some(|x| if x >= 0.0 { x } else { x.exp() - 1.0 }),
        "EXPONENTIAL" => Some(|x| {
            if !x.is_finite() || x >= EXPONENTIAL_CUTOFF {
                JS_MAX_SAFE_INTEGER
            } else {
                // Belt-and-suspenders: guard against non-finite f64→f32 cast (Issue #805)
                let result = ((x as f64).exp()) as f32;
                if result.is_finite() {
                    result
                } else {
                    JS_MAX_SAFE_INTEGER
                }
            }
        }),
        "GAUSSIAN" => Some(|x| {
            let safe_x = x.abs().min(100.0);
            (-safe_x * safe_x).exp()
        }),
        "GELU" => Some(|x| {
            let x3 = x * x * x;
            let tanh_arg = 0.797_884_6_f32 * (x + 0.044_715_f32 * x3);
            0.5 * x * (1.0 + tanh_arg.tanh())
        }),
        "HARD_TANH" | "CLIPPED" => Some(|x| x.clamp(-1.0, 1.0)),
        "IDENTITY" => Some(|x| x),
        "ISRU" => Some(|x| x / (1.0 + x * x).sqrt()),
        "LEAKYRELU" => Some(|x| if x >= 0.0 { x } else { 0.01 * x }),
        "LOGISTIC" => Some(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        }),
        "LOGSIGMOID" => Some(|x| {
            // See `apply_scalar_squash` for derivation.
            if x >= 0.0 {
                -(-x).exp().ln_1p()
            } else {
                x - x.exp().ln_1p()
            }
        }),
        "MISH" => Some(|x| x * (1.0 + x.exp()).ln().tanh()),
        "RELU" => Some(|x| x.max(0.0)),
        "RELU6" => Some(|x| x.clamp(0.0, 6.0)),
        "SELU" => Some(|x| {
            const ALPHA: f32 = 1.673_263_2;
            const LAMBDA: f32 = 1.050_701;
            if x >= 0.0 {
                LAMBDA * x
            } else {
                LAMBDA * ALPHA * (x.exp() - 1.0)
            }
        }),
        "SINE" | "SINUSOID" => Some(f32::sin),
        "SOFTPLUS" => Some(|x| {
            if !x.is_finite() {
                SOFTPLUS_SMALL_THRESHOLD
            } else if x >= SOFTPLUS_CUTOFF {
                SOFTPLUS_LARGE_THRESHOLD
            } else {
                // Belt-and-suspenders: guard against non-finite f64→f32 cast (Issue #805)
                let result = ((1.0f64 + (x as f64).exp()).ln()) as f32;
                if result.is_finite() {
                    result
                } else {
                    SOFTPLUS_LARGE_THRESHOLD
                }
            }
        }),
        "SOFTSIGN" => Some(|x| x / (1.0 + x.abs())),
        "SQRT" => Some(|x| {
            if x.is_finite() && x >= 0.0 {
                x.sqrt()
            } else {
                0.0
            }
        }),
        "SQUARE" => Some(|x| x * x),
        "STDINVERSE" => Some(|x| {
            let safe_x = if x.abs() < STDINVERSE_EPSILON {
                if x >= 0.0 {
                    STDINVERSE_EPSILON
                } else {
                    -STDINVERSE_EPSILON
                }
            } else {
                x
            };
            1.0 / safe_x
        }),
        "STEP" => Some(|x| if x > 0.0 { 1.0 } else { 0.0 }),
        "SWISH" => Some(|x| {
            let sigmoid = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            };
            x * sigmoid
        }),
        "TAN" => Some(f32::tan),
        "TANH" => Some(f32::tanh),
        _ => None,
    }
}

/// Returns the maximum absolute output magnitude for bounded scalar squashes (Issue #1300).
///
/// This is used by the impact calculation to bound per-synapse contributions so the sum
/// of inbound contributions into a neuron cannot exceed what the squash can actually emit
/// (the "saturation ceiling"). For example, a TANH neuron emits values in `[-1, 1]`, so the
/// magnitude is `1.0`; ten upstream synapses each feeding pre-activation `10` still share
/// the same `±1` emit ceiling rather than each independently contributing `10`.
///
/// # Return value
///
/// - `Some(M)` where `M > 0` is the largest `|f(x)|` the squash can produce.
/// - `None` for unbounded squashes (IDENTITY, RELU, ELU, LEAKYRELU, SELU, EXPONENTIAL,
///   SOFTPLUS, CUBE, SQUARE, etc.) and aggregate squashes (MIN/MAX/IF/HYPOT — handled by
///   selection statistics instead).
///
/// # Notes
///
/// - `STEP` outputs `{0, 1}` → magnitude `1.0`.
/// - `BIPOLAR` outputs `{-1, +1}` → magnitude `1.0`.
/// - `HARD_TANH`/`CLIPPED` clamps to `[-1, 1]` → magnitude `1.0`.
/// - `GAUSSIAN` outputs `(0, 1]` → magnitude `1.0`.
/// - `ARCTAN` outputs `(-π/2, π/2)` → magnitude `π/2 ≈ 1.5708`.
/// - `RELU6` clamps to `[0, 6]` → magnitude `6.0`.
#[must_use]
pub fn squash_emit_magnitude(name: &str) -> Option<f32> {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        // Bounded to ±1
        "TANH" | "LOGISTIC" | "HARD_TANH" | "CLIPPED" | "SOFTSIGN" | "BIPOLAR_SIGMOID" | "ISRU"
        | "STEP" | "BIPOLAR" | "GAUSSIAN" => Some(1.0),
        // ARCTAN: (-π/2, π/2)
        "ARCTAN" => Some(std::f32::consts::FRAC_PI_2),
        // RELU6: [0, 6]
        "RELU6" => Some(6.0),
        // LOGSIGMOID: (-∞, 0]; unbounded below — treat as unbounded.
        // Unbounded scalar squashes: IDENTITY, RELU, LEAKYRELU, ELU, SELU,
        // EXPONENTIAL, SOFTPLUS, CUBE, SQUARE, SINE, COSINE, TAN, MISH, SWISH,
        // GELU, ABSOLUTE, STDINVERSE, SQRT, BENT_IDENTITY, COMPLEMENT/INVERSE.
        // Aggregate squashes (MIN/MAX/IF/HYPOT/MEAN) — handled by selection stats.
        _ => None,
    }
}

/// Returns an approximate inverse function for monotonic activations (Issue #906).
///
/// When `target_value` is missing but `target_activation` is available, the inverse function
/// computes `target_value ≈ inverse(target_activation)` so that saturation-aware simulation
/// can proceed without falling back to the linear model.
///
/// Only activations where a reasonable inverse exists are supported:
/// - Monotonic, bounded activations with closed-form inverses (TANH, LOGISTIC, SOFTSIGN, etc.)
/// - Piecewise-linear activations (`HARD_TANH`, ELU, `LEAKYRELU`, SELU, `RELU6`)
/// - Approximately invertible activations (GELU, MISH, SWISH via Newton's method)
///
/// Non-monotonic activations (SINE, GAUSSIAN, SQUARE, ABSOLUTE) return `None`.
pub fn approximate_inverse_fn(name: &str) -> Option<fn(f32) -> f32> {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        // Piecewise-linear: identity in valid range
        "HARD_TANH" | "CLIPPED" => Some(|y| y.clamp(-1.0, 1.0)),
        "RELU6" => Some(|y| y.clamp(0.0, 6.0)),

        // Closed-form inverses for monotonic bounded activations
        "TANH" => Some(|y| y.clamp(-0.999, 0.999).atanh()),
        "LOGISTIC" => Some(|y| {
            let y = y.clamp(0.001, 0.999);
            (y / (1.0 - y)).ln()
        }),
        "SOFTSIGN" => Some(|y| {
            let y = y.clamp(-0.999, 0.999);
            y / (1.0 - y.abs())
        }),
        "BIPOLAR_SIGMOID" => Some(|y| {
            // bipolar_sigmoid(x) = 2*sigmoid(x) - 1 = tanh(x/2)
            // inverse: 2*atanh(y)
            let y = y.clamp(-0.999, 0.999);
            2.0 * y.atanh()
        }),
        "ISRU" => Some(|y| {
            // ISRU(x) = x/sqrt(1+x^2), inverse: y/sqrt(1-y^2)
            let y = y.clamp(-0.999, 0.999);
            y / (1.0 - y * y).sqrt()
        }),
        "ARCTAN" => Some(f32::tan),
        "LOGSIGMOID" => Some(|y| {
            // logsigmoid(x) = -ln(1+exp(-x)) = y => 1+exp(-x) = exp(-y) => x = -ln(exp(-y)-1)
            // For y in (-inf, 0), exp(-y) > 1 so exp(-y)-1 > 0
            let y = y.min(-0.001);
            -((-y).exp() - 1.0).max(1e-10).ln()
        }),

        // Piecewise-invertible activations
        "ELU" => Some(|y| {
            if y >= 0.0 {
                y
            } else {
                (y + 1.0).max(0.001).ln()
            }
        }),
        "LEAKYRELU" => Some(|y| if y >= 0.0 { y } else { y / 0.01 }),
        "SELU" => Some(|y| {
            const ALPHA: f32 = 1.673_263_2;
            const LAMBDA: f32 = 1.050_701;
            if y >= 0.0 {
                y / LAMBDA
            } else {
                (y / (LAMBDA * ALPHA) + 1.0).max(0.001).ln()
            }
        }),
        "SOFTPLUS" => Some(|y| {
            // softplus(x) = ln(1+exp(x)), inverse: ln(exp(y)-1)
            let y = y.max(0.001);
            (y.exp() - 1.0).max(1e-10).ln()
        }),

        // Newton's method for approximately invertible activations (3 iterations)
        "GELU" => Some(|y| {
            let mut x = if y >= 0.0 {
                y
            } else if y > -0.2 {
                y * 1.5
            } else {
                -1.0
            };
            for _ in 0..4 {
                let x3 = x * x * x;
                let tanh_arg = 0.797_884_6_f32 * (x + 0.044_715_f32 * x3);
                let tanh_val = tanh_arg.tanh();
                let gelu_x = 0.5 * x * (1.0 + tanh_val);
                let sech2 = 1.0 - tanh_val * tanh_val;
                let d_tanh_arg = 0.797_884_6_f32 * (1.0 + 3.0 * 0.044_715_f32 * x * x);
                let derivative = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_tanh_arg;
                if derivative.abs() > 1e-10 {
                    x -= (gelu_x - y) / derivative;
                }
            }
            x
        }),
        "MISH" => Some(|y| {
            let mut x = y;
            for _ in 0..4 {
                let exp_x = x.exp();
                let softplus = (1.0 + exp_x).ln();
                let tanh_sp = softplus.tanh();
                let mish_x = x * tanh_sp;
                let sigmoid = exp_x / (1.0 + exp_x);
                let derivative = tanh_sp + x * (1.0 - tanh_sp * tanh_sp) * sigmoid;
                if derivative.abs() > 1e-10 {
                    x -= (mish_x - y) / derivative;
                }
            }
            x
        }),
        "SWISH" => Some(|y| {
            let mut x = y;
            for _ in 0..4 {
                let sigmoid = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let exp_x = x.exp();
                    exp_x / (1.0 + exp_x)
                };
                let swish_x = x * sigmoid;
                let derivative = sigmoid + x * sigmoid * (1.0 - sigmoid);
                if derivative.abs() > 1e-10 {
                    x -= (swish_x - y) / derivative;
                }
            }
            x
        }),

        // IDENTITY: trivially invertible
        "IDENTITY" => Some(|y| y),
        // COMPLEMENT/INVERSE: self-inverse (1-(1-x) = x)
        "COMPLEMENT" | "INVERSE" => Some(|y| 1.0 - y),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify STEP and BIPOLAR both return simulation functions for consistent handling.
    /// This was a bug fix in v0.2.18 where STEP returned None, causing it to fall back
    /// to the linear error model while BIPOLAR got proper threshold simulation.
    #[test]
    fn step_and_bipolar_both_have_simulation_functions() {
        let step_fn = target_simulation_fn("STEP");
        let bipolar_fn = target_simulation_fn("BIPOLAR");

        assert!(
            step_fn.is_some(),
            "STEP must return a simulation function for consistent handling with BIPOLAR"
        );
        assert!(
            bipolar_fn.is_some(),
            "BIPOLAR must return a simulation function"
        );

        // Verify both functions work correctly
        let step = step_fn.unwrap();
        let bipolar = bipolar_fn.unwrap();

        // STEP: 0 for x <= 0, 1 for x > 0
        assert_eq!(step(-1.0), 0.0);
        assert_eq!(step(0.0), 0.0);
        assert_eq!(step(0.001), 1.0);
        assert_eq!(step(1.0), 1.0);

        // BIPOLAR: -1 for x <= 0, 1 for x > 0
        assert_eq!(bipolar(-1.0), -1.0);
        assert_eq!(bipolar(0.0), -1.0);
        assert_eq!(bipolar(0.001), 1.0);
        assert_eq!(bipolar(1.0), 1.0);
    }

    /// Case-insensitive matching should work for STEP/BIPOLAR.
    #[test]
    fn step_bipolar_case_insensitive() {
        assert!(target_simulation_fn("step").is_some());
        assert!(target_simulation_fn("Step").is_some());
        assert!(target_simulation_fn("STEP").is_some());

        assert!(target_simulation_fn("bipolar").is_some());
        assert!(target_simulation_fn("Bipolar").is_some());
        assert!(target_simulation_fn("BIPOLAR").is_some());
    }

    /// Aggregate squashes should return None (they cannot be simulated as f(x)).
    #[test]
    fn aggregate_squashes_return_none() {
        assert!(target_simulation_fn("MINIMUM").is_none());
        assert!(target_simulation_fn("MAXIMUM").is_none());
        assert!(target_simulation_fn("IF").is_none());
        assert!(target_simulation_fn("MEAN").is_none());
        assert!(target_simulation_fn("HYPOT").is_none());
    }

    // =========================================================================
    // Issue #906: Approximate inverse function tests
    // =========================================================================

    /// Verify inverse functions exist for all expected monotonic activations.
    #[test]
    fn inverse_fn_exists_for_monotonic_activations() {
        let invertible = [
            "TANH",
            "LOGISTIC",
            "SOFTSIGN",
            "BIPOLAR_SIGMOID",
            "ISRU",
            "HARD_TANH",
            "CLIPPED",
            "RELU6",
            "ELU",
            "LEAKYRELU",
            "SELU",
            "SOFTPLUS",
            "ARCTAN",
            "LOGSIGMOID",
            "GELU",
            "MISH",
            "SWISH",
            "IDENTITY",
            "COMPLEMENT",
        ];
        for name in &invertible {
            assert!(
                approximate_inverse_fn(name).is_some(),
                "{name} should have an approximate inverse function"
            );
        }
    }

    /// Non-monotonic activations should not have an inverse.
    #[test]
    fn inverse_fn_none_for_non_monotonic() {
        let non_invertible = ["SINE", "COSINE", "GAUSSIAN", "SQUARE", "ABSOLUTE", "STEP"];
        for name in &non_invertible {
            assert!(
                approximate_inverse_fn(name).is_none(),
                "{name} should not have an approximate inverse function"
            );
        }
    }

    /// Verify round-trip accuracy: `inverse(forward(x)) ≈ x` for closed-form inverses.
    #[test]
    fn inverse_round_trip_closed_form() {
        let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        // TANH
        let fwd = target_simulation_fn("TANH").unwrap();
        let inv = approximate_inverse_fn("TANH").unwrap();
        for &x in &test_values {
            let roundtrip = inv(fwd(x));
            assert!(
                (roundtrip - x).abs() < 0.01,
                "TANH round-trip failed: x={x}, got {roundtrip}"
            );
        }

        // LOGISTIC
        let fwd = target_simulation_fn("LOGISTIC").unwrap();
        let inv = approximate_inverse_fn("LOGISTIC").unwrap();
        for &x in &test_values {
            let roundtrip = inv(fwd(x));
            assert!(
                (roundtrip - x).abs() < 0.01,
                "LOGISTIC round-trip failed: x={x}, got {roundtrip}"
            );
        }

        // SOFTSIGN
        let fwd = target_simulation_fn("SOFTSIGN").unwrap();
        let inv = approximate_inverse_fn("SOFTSIGN").unwrap();
        for &x in &test_values {
            let roundtrip = inv(fwd(x));
            assert!(
                (roundtrip - x).abs() < 0.01,
                "SOFTSIGN round-trip failed: x={x}, got {roundtrip}"
            );
        }

        // ELU
        let fwd = target_simulation_fn("ELU").unwrap();
        let inv = approximate_inverse_fn("ELU").unwrap();
        for &x in &test_values {
            let roundtrip = inv(fwd(x));
            assert!(
                (roundtrip - x).abs() < 0.01,
                "ELU round-trip failed: x={x}, got {roundtrip}"
            );
        }
    }

    /// Verify round-trip accuracy for Newton's method inverses (GELU, MISH, SWISH).
    ///
    /// Note: GELU/MISH/SWISH have small non-monotonic regions for negative x, so
    /// we test in the monotonic region (x >= -0.5) where the inverse is well-defined.
    #[test]
    fn inverse_round_trip_newton() {
        let test_values = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

        for name in &["GELU", "MISH", "SWISH"] {
            let fwd = target_simulation_fn(name).unwrap();
            let inv = approximate_inverse_fn(name).unwrap();
            for &x in &test_values {
                let roundtrip = inv(fwd(x));
                assert!(
                    (roundtrip - x).abs() < 0.05,
                    "{name} round-trip failed: x={x}, got {roundtrip}"
                );
            }
        }
    }
}
