// GPU shader for parallel bias grid search
// Tests all bias values in parallel to find optimal bias

struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct BiasResult {
    bias_value: f32,
    error_reduction: f32,
    valid_sample_count: u32,
    pad0: u32,
};

struct BiasUniforms {
    sample_count: u32,
    bias_count: u32,
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_type: u32,
    epsilon: f32,
    min_sample_count: u32,
    pad0: u32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HelpfulSample>;
@group(0) @binding(1)
var<storage, read> bias_candidates: array<f32>;
@group(0) @binding(2)
var<storage, read_write> results: array<BiasResult>;
@group(0) @binding(3)
var<uniform> uniforms: BiasUniforms;

const MAX_F32: f32 = 3.402823466e+38;

fn is_finite_value(value: f32) -> bool {
    if (value != value) {
        return false;
    }
    return abs(value) <= MAX_F32;
}

// Activation functions (same as activation.wgsl)
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

fn relu_activation(x: f32) -> f32 {
    return max(x, 0.0);
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
        case 11u: { return relu_activation(x); }
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bias_idx = global_id.x;
    if (bias_idx >= uniforms.bias_count) {
        return;
    }

    let bias = bias_candidates[bias_idx];
    var result: BiasResult;
    result.bias_value = bias;
    result.error_reduction = 0.0;
    result.valid_sample_count = 0u;
    result.pad0 = 0u;

    // Calculate baseline error (no new neuron)
    var total_baseline_error_sq: f32 = 0.0;
    var total_new_error_sq: f32 = 0.0;
    var valid_samples: u32 = 0u;

    // Test this bias value across all samples
    for (var sample_idx: u32 = 0u; sample_idx < uniforms.sample_count; sample_idx = sample_idx + 1u) {
        let sample = samples[sample_idx];

        // Skip non-finite samples
        if (!is_finite_value(sample.activation) || !is_finite_value(sample.avg_error)) {
            continue;
        }

        // Accumulate baseline error
        total_baseline_error_sq = total_baseline_error_sq + (sample.avg_error * sample.avg_error);

        // Calculate new neuron's activation with this bias
        let pre_activation = uniforms.incoming_weight * sample.activation + bias;
        let new_neuron_activation = apply_activation(pre_activation, uniforms.activation_type);

        if (!is_finite_value(new_neuron_activation)) {
            // If activation is non-finite, this sample keeps baseline error
            total_new_error_sq = total_new_error_sq + (sample.avg_error * sample.avg_error);
            continue;
        }

        // Calculate new error at target neuron
        let correction = uniforms.outgoing_weight * new_neuron_activation;
        let new_error = sample.avg_error - correction;

        if (is_finite_value(new_error)) {
            total_new_error_sq = total_new_error_sq + (new_error * new_error);
            valid_samples = valid_samples + 1u;
        } else {
            // Non-finite new error, keep baseline
            total_new_error_sq = total_new_error_sq + (sample.avg_error * sample.avg_error);
        }
    }

    // Only consider this bias if we have enough valid samples
    if (valid_samples >= uniforms.min_sample_count) {
        // Error reduction is positive when new error is less than baseline
        result.error_reduction = total_baseline_error_sq - total_new_error_sq;
        result.valid_sample_count = valid_samples;
    } else {
        // Not enough samples, mark as invalid with negative error reduction
        result.error_reduction = -1e10;
        result.valid_sample_count = valid_samples;
    }

    results[bias_idx] = result;
}

