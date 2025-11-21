struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct HelpfulContribution {
    positive_flag: u32,
    negative_flag: u32,
    positive_improvement: f32,
    negative_improvement: f32,
    positive_activation: f32,
    negative_activation: f32,
    error_squared: f32,
    activation_squared: f32,
    error_activation: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

struct HelpfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    pad1: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HelpfulSample>;
@group(0) @binding(1)
var<storage, read_write> contributions: array<HelpfulContribution>;
@group(0) @binding(2)
var<uniform> uniforms: HelpfulUniforms;

fn sign_nonzero(value: f32) -> f32 {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.length) {
        return;
    }
    let sample = samples[idx];
    var contribution: HelpfulContribution;
    contribution.positive_flag = 0u;
    contribution.negative_flag = 0u;
    contribution.positive_improvement = 0.0;
    contribution.negative_improvement = 0.0;
    contribution.positive_activation = 0.0;
    contribution.negative_activation = 0.0;
    contribution.error_squared = 0.0;
    contribution.activation_squared = 0.0;
    contribution.error_activation = 0.0;
    contribution.pad0 = 0.0;
    contribution.pad1 = 0.0;
    contribution.pad2 = 0.0;

    if (abs(sample.activation) > uniforms.epsilon && abs(sample.avg_error) > uniforms.epsilon) {
        let required_sign = -sign_nonzero(sample.avg_error) * sign_nonzero(sample.activation);
        let improvement = abs(sample.avg_error);
        
        contribution.error_squared = sample.avg_error * sample.avg_error;
        contribution.activation_squared = sample.activation * sample.activation;
        contribution.error_activation = sample.avg_error * sample.activation;

        if (required_sign > 0.0) {
            contribution.positive_flag = 1u;
            contribution.positive_improvement = improvement;
            contribution.positive_activation = abs(sample.activation);
        } else if (required_sign < 0.0) {
            contribution.negative_flag = 1u;
            contribution.negative_improvement = improvement;
            contribution.negative_activation = abs(sample.activation);
        }
    } else if (abs(sample.avg_error) > uniforms.epsilon) {
        // Even if activation is zero, we should count the error for total baseline error
        contribution.error_squared = sample.avg_error * sample.avg_error;
    }

    contributions[idx] = contribution;
}
