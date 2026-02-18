//! CPU activation function implementations for candidate evaluation.
//!
//! Each function here implements a specific activation function used during
//! discovery candidate generation and evaluation.

// ============================================================================
// CPU Activation Function Implementations
// ============================================================================

/// GELU activation function.
/// GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
pub fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.797_884_6 * (x + 0.044_715 * x_cubed);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

/// ELU activation function.
/// ELU(x) = x if x >= 0, else exp(x) - 1
pub fn elu_activation(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

/// Softplus activation function.
/// Softplus(x) = ln(1 + exp(x)), linearised for x > 20 to avoid overflow.
pub fn softplus_activation(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Logistic (sigmoid) activation function.
/// Logistic(x) = 1 / (1 + exp(-x)), numerically stable.
pub fn logistic_activation(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Tanh activation function.
pub fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

/// Identity activation function.
pub fn identity_activation(x: f32) -> f32 {
    x
}

/// Bipolar activation function.
/// BIPOLAR(x) = 1 if x > 0, else -1
pub fn bipolar_activation(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { -1.0 }
}

/// Clipped activation function.
/// CLIPPED(x) = clamp(x, -1, 1)
pub fn clipped_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Absolute activation function.
pub fn absolute_activation(x: f32) -> f32 {
    x.abs()
}

// ============================================================================
// NEW ACTIVATION FUNCTIONS (v0.1.139)
// Based on analysis of successful discoveries that evolved TO these activations
// ============================================================================

/// Mish activation function.
/// 2 successful discoveries evolved TO Mish (from ELU and Softplus).
/// Self-regularised activation: x * tanh(softplus(x))
pub fn mish_activation(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    x * sp.tanh()
}

/// Hard tanh activation function.
/// 1 successful discovery evolved CLIPPED → HARD_TANH.
/// Linear in [-1, 1], saturates outside. Same as CLIPPED but named for NEAT-AI.
pub fn hard_tanh_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Softsign activation function.
/// 1 successful discovery neuron with SOFTSIGN.
/// Smooth approximation of sign function: x / (1 + |x|)
pub fn softsign_activation(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// Bent identity activation function.
/// 1 successful discovery evolved LeakyReLU → BENT_IDENTITY.
/// Smooth, nearly linear: (sqrt(x² + 1) - 1) / 2 + x
pub fn bent_identity_activation(x: f32) -> f32 {
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

/// Arctan activation function.
/// Similar to SOFTSIGN, bounded output.
pub fn arctan_activation(x: f32) -> f32 {
    x.atan()
}

/// ReLU6 activation function.
/// Capped ReLU at 6, useful for quantisation.
pub fn relu6_activation(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}
