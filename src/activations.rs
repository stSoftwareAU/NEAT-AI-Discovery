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
}
