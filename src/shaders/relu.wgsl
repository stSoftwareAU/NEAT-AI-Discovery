struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct ReluContribution {
    positive_activation_sq: f32,
    positive_error_activation: f32,
    positive_count: u32,
    negative_activation_sq: f32,
    negative_error_activation: f32,
    negative_count: u32,
    error_sq: f32,
    pad0: f32,
    pad1: u32,
    pad2: u32,
};

struct ReluUniforms {
    length: u32,
    threshold: f32,
    epsilon: f32,
    pad0: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HelpfulSample>;
@group(0) @binding(1)
var<storage, read_write> contributions: array<ReluContribution>;
@group(0) @binding(2)
var<uniform> uniforms: ReluUniforms;

const MAX_F32: f32 = 3.402823466e+38;

fn is_finite_value(value: f32) -> bool {
    if (value != value) {
        return false;
    }
    return abs(value) <= MAX_F32;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.length) {
        return;
    }

    let sample = samples[idx];
    var contribution: ReluContribution;
    contribution.positive_activation_sq = 0.0;
    contribution.positive_error_activation = 0.0;
    contribution.positive_count = 0u;
    contribution.negative_activation_sq = 0.0;
    contribution.negative_error_activation = 0.0;
    contribution.negative_count = 0u;
    contribution.error_sq = 0.0;
    contribution.pad0 = 0.0;
    contribution.pad1 = 0u;
    contribution.pad2 = 0u;

    // Skip invalid samples
    if (!is_finite_value(sample.activation) || !is_finite_value(sample.avg_error)) {
        contributions[idx] = contribution;
        return;
    }

    // Accumulate error squared for baseline
    contribution.error_sq = sample.avg_error * sample.avg_error;

    // Compute positive ReLU: max(activation, 0)
    let relu_positive = max(sample.activation, 0.0);
    if (relu_positive > uniforms.epsilon) {
        contribution.positive_activation_sq = relu_positive * relu_positive;
        contribution.positive_error_activation = relu_positive * sample.avg_error;
        contribution.positive_count = 1u;
    }

    // Compute negative ReLU: max(-activation, 0)
    let relu_negative = max(-sample.activation, 0.0);
    if (relu_negative > uniforms.epsilon) {
        contribution.negative_activation_sq = relu_negative * relu_negative;
        contribution.negative_error_activation = relu_negative * sample.avg_error;
        contribution.negative_count = 1u;
    }

    contributions[idx] = contribution;
}

