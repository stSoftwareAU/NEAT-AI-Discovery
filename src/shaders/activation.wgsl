struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct ActivationOutput {
    output: f32,
    output_sq: f32,
    error_output: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct ActivationUniforms {
    sample_count: u32,
    orientation: f32,
    scale: f32,
    activation_type: u32,
    epsilon: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HelpfulSample>;
@group(0) @binding(1)
var<storage, read_write> outputs: array<ActivationOutput>;
@group(0) @binding(2)
var<uniform> uniforms: ActivationUniforms;

const MAX_F32: f32 = 3.402823466e+38;

fn is_finite_value(value: f32) -> bool {
    if (value != value) {
        return false;
    }
    return abs(value) <= MAX_F32;
}

// Activation functions
fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x_cubed);
    return 0.5 * x * (1.0 + tanh(tanh_arg));
}

fn elu_activation(x: f32) -> f32 {
    if (x >= 0.0) {
        return x;
    } else {
        return exp(x) - 1.0;
    }
}

fn selu_activation(x: f32) -> f32 {
    let SELU_ALPHA: f32 = 1.6732632;
    let SELU_LAMBDA: f32 = 1.050701;
    if (x >= 0.0) {
        return SELU_LAMBDA * x;
    } else {
        return SELU_LAMBDA * SELU_ALPHA * (exp(x) - 1.0);
    }
}

fn softplus_activation(x: f32) -> f32 {
    if (x > 20.0) {
        return x;
    } else {
        return log(1.0 + exp(x));
    }
}

fn logistic_activation(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        let exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

fn tanh_activation(x: f32) -> f32 {
    return tanh(x);
}

fn identity_activation(x: f32) -> f32 {
    return x;
}

fn bipolar_activation(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    } else {
        return -1.0;
    }
}

fn clipped_activation(x: f32) -> f32 {
    return clamp(x, -1.0, 1.0);
}

fn absolute_activation(x: f32) -> f32 {
    return abs(x);
}

fn inverse_activation(x: f32) -> f32 {
    return 1.0 - x;
}

fn apply_activation(x: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return gelu_activation(x); }
        case 1u: { return elu_activation(x); }
        case 2u: { return selu_activation(x); }
        case 3u: { return softplus_activation(x); }
        case 4u: { return logistic_activation(x); }
        case 5u: { return tanh_activation(x); }
        case 6u: { return identity_activation(x); }
        case 7u: { return bipolar_activation(x); }
        case 8u: { return clipped_activation(x); }
        case 9u: { return absolute_activation(x); }
        case 10u: { return inverse_activation(x); }
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.sample_count) {
        return;
    }

    let sample = samples[idx];
    var output: ActivationOutput;
    output.output = 0.0;
    output.output_sq = 0.0;
    output.error_output = 0.0;
    output.valid = 0u;
    output.pad0 = 0u;
    output.pad1 = 0u;
    output.pad2 = 0u;

    // Skip invalid samples
    if (!is_finite_value(sample.activation) || !is_finite_value(sample.avg_error)) {
        outputs[idx] = output;
        return;
    }

    // Compute pre-activation: incoming_weight * activation
    let incoming_weight = uniforms.orientation * uniforms.scale;
    let pre_activation = incoming_weight * sample.activation;

    // Apply activation function
    let output_val = apply_activation(pre_activation, uniforms.activation_type);

    // Check if output is valid
    if (is_finite_value(output_val)) {
        output.output = output_val;
        output.output_sq = output_val * output_val;
        output.error_output = output_val * sample.avg_error;
        output.valid = 1u;
    }

    outputs[idx] = output;
}

