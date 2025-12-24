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

use std::borrow::Cow;

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
            // Match NEAT-AI's safety behaviour: avoid overflow when exp(x) would blow up.
            if x >= 709.0 {
                Some(f32::MAX)
            } else {
                Some(x.exp())
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
            // -ln(1 + exp(-x)), with overflow protection.
            if x <= -709.0 {
                Some(f32::MIN)
            } else {
                Some(-(1.0 + (-x).exp()).ln())
            }
        }
        "MISH" => {
            // Mish(x) = x * tanh(softplus(x))
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            Some(x * sp.tanh())
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
            // Match NEAT-AI: linearise beyond ~20 to avoid ln(1+exp(x)) overflow.
            Some(if x > 20.0 { x } else { (1.0 + x.exp()).ln() })
        }
        "SOFTSIGN" => Some(x / (1.0 + x.abs())),
        "SQRT" => Some(if x.is_finite() && x >= 0.0 {
            x.sqrt()
        } else {
            0.0
        }),
        "SQUARE" => Some(x * x),
        "STDINVERSE" => {
            // Match NEAT-AI `StdInverse.squash()` behaviour: 1/x with epsilon protection.
            let eps = 1e-15_f32;
            let safe_x = if x.abs() < eps {
                if x >= 0.0 {
                    eps
                } else {
                    -eps
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
/// - We intentionally return `None` for aggregate squashes.
/// - We also return `None` for STEP because Discovery uses a dedicated threshold-crossing model
///   (see README: "Threshold-crossing model for STEP/BIPOLAR"). Treating STEP as smooth is misleading.
pub fn target_simulation_fn(name: &str) -> Option<fn(f32) -> f32> {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        "ABSOLUTE" => Some(|x| x.abs()),
        "ARCTAN" => Some(|x| x.atan()),
        "BENT_IDENTITY" => Some(|x| ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x),
        "BIPOLAR" => Some(|x| if x > 0.0 { 1.0 } else { -1.0 }),
        "BIPOLAR_SIGMOID" => Some(|x| 2.0 / (1.0 + (-x).exp()) - 1.0),
        "COMPLEMENT" | "INVERSE" => Some(|x| 1.0 - x),
        "COSINE" => Some(|x| x.cos()),
        "CUBE" => Some(|x| x * x * x),
        "ELU" => Some(|x| if x >= 0.0 { x } else { x.exp() - 1.0 }),
        "EXPONENTIAL" => Some(|x| if x >= 709.0 { f32::MAX } else { x.exp() }),
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
            if x <= -709.0 {
                f32::MIN
            } else {
                -(1.0 + (-x).exp()).ln()
            }
        }),
        "MISH" => Some(|x| {
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            x * sp.tanh()
        }),
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
        "SINE" | "SINUSOID" => Some(|x| x.sin()),
        "SOFTPLUS" => Some(|x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() }),
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
            let eps = 1e-15_f32;
            let safe_x = if x.abs() < eps {
                if x >= 0.0 {
                    eps
                } else {
                    -eps
                }
            } else {
                x
            };
            1.0 / safe_x
        }),
        // STEP handled via threshold-crossing model.
        "STEP" => None,
        "SWISH" => Some(|x| {
            let sigmoid = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            };
            x * sigmoid
        }),
        "TAN" => Some(|x| x.tan()),
        "TANH" => Some(|x| x.tanh()),
        _ => None,
    }
}
