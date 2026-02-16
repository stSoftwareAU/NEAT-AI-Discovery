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

// New activations added in v0.1.139 - GPU IDs 11-18 (bias.wgsl uses 11 for ReLU internally)
fn leaky_relu_activation(x: f32) -> f32 {
    if (x >= 0.0) {
        return x;
    } else {
        return 0.01 * x; // Standard leak coefficient
    }
}

fn mish_activation(x: f32) -> f32 {
    // Mish(x) = x * tanh(softplus(x))
    var sp: f32;
    if (x > 20.0) {
        sp = x;
    } else {
        sp = log(1.0 + exp(x));
    }
    return x * tanh(sp);
}

fn swish_activation(x: f32) -> f32 {
    // Swish(x) = x * sigmoid(x)
    var sigmoid: f32;
    if (x >= 0.0) {
        sigmoid = 1.0 / (1.0 + exp(-x));
    } else {
        let exp_x = exp(x);
        sigmoid = exp_x / (1.0 + exp_x);
    }
    return x * sigmoid;
}

fn hard_tanh_activation(x: f32) -> f32 {
    return clamp(x, -1.0, 1.0);
}

fn softsign_activation(x: f32) -> f32 {
    // softsign(x) = x / (1 + |x|)
    return x / (1.0 + abs(x));
}

fn bent_identity_activation(x: f32) -> f32 {
    // bent_identity(x) = (sqrt(x² + 1) - 1) / 2 + x
    return (sqrt(x * x + 1.0) - 1.0) / 2.0 + x;
}

fn arctan_activation(x: f32) -> f32 {
    return atan(x);
}

fn relu6_activation(x: f32) -> f32 {
    return clamp(x, 0.0, 6.0);
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
        // New activations (v0.1.139) - GPU IDs 11-18
        case 11u: { return leaky_relu_activation(x); }
        case 12u: { return mish_activation(x); }
        case 13u: { return swish_activation(x); }
        case 14u: { return hard_tanh_activation(x); }
        case 15u: { return softsign_activation(x); }
        case 16u: { return bent_identity_activation(x); }
        case 17u: { return arctan_activation(x); }
        case 18u: { return relu6_activation(x); }
        default: { return x; }
    }
}

// Issue #567: Workgroup shared memory for tiled sample loading.
// Instead of each thread reading all samples from global memory independently,
// we load tiles of samples into shared memory cooperatively, then all threads
// process them from fast shared memory. This reduces global memory reads by
// the workgroup size factor (256×).
const TILE_SIZE: u32 = 256u;
var<workgroup> shared_samples: array<HelpfulSample, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let bias_idx = global_id.x;
    let local_idx = local_id.x;

    // Each thread processes one bias candidate (if in range)
    let in_range = bias_idx < uniforms.bias_count;

    var bias: f32 = 0.0;
    var total_baseline_error_sq: f32 = 0.0;
    var total_new_error_sq: f32 = 0.0;
    var valid_samples: u32 = 0u;

    if (in_range) {
        bias = bias_candidates[bias_idx];
    }

    // Process samples in tiles using shared memory
    let tile_count = (uniforms.sample_count + TILE_SIZE - 1u) / TILE_SIZE;

    for (var tile: u32 = 0u; tile < tile_count; tile = tile + 1u) {
        // Cooperatively load a tile of samples into shared memory
        let sample_load_idx = tile * TILE_SIZE + local_idx;
        if (sample_load_idx < uniforms.sample_count) {
            shared_samples[local_idx] = samples[sample_load_idx];
        } else {
            // Pad with invalid samples (NaN error so they get skipped)
            var pad_sample: HelpfulSample;
            pad_sample.activation = 0.0;
            pad_sample.avg_error = bitcast<f32>(0x7fc00000u); // NaN
            shared_samples[local_idx] = pad_sample;
        }

        // Synchronise: all threads must finish loading before processing
        workgroupBarrier();

        // Each thread processes the tile for its own bias candidate
        if (in_range) {
            let tile_end = min(TILE_SIZE, uniforms.sample_count - tile * TILE_SIZE);
            for (var i: u32 = 0u; i < tile_end; i = i + 1u) {
                let sample = shared_samples[i];

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
                    total_new_error_sq = total_new_error_sq + (sample.avg_error * sample.avg_error);
                }
            }
        }

        // Synchronise: all threads must finish processing before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (in_range) {
        var result: BiasResult;
        result.bias_value = bias;
        result.pad0 = 0u;

        if (valid_samples >= uniforms.min_sample_count) {
            result.error_reduction = total_baseline_error_sq - total_new_error_sq;
            result.valid_sample_count = valid_samples;
        } else {
            result.error_reduction = -1e10;
            result.valid_sample_count = valid_samples;
        }

        results[bias_idx] = result;
    }
}

