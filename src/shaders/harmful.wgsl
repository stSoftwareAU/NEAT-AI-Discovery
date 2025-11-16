struct HarmfulSample {
    activation: f32,
    avg_error: f32,
};

struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
};

struct HarmfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    weight: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HarmfulSample>;
@group(0) @binding(1)
var<storage, read_write> contributions: array<HarmfulContribution>;
@group(0) @binding(2)
var<uniform> uniforms: HarmfulUniforms;

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
    var contribution: HarmfulContribution;
    contribution.harmful_flag = 0u;
    contribution.helpful_flag = 0u;
    contribution.error_magnitude = 0.0;
    contribution.pad0 = 0.0;

    if (abs(sample.activation) > uniforms.epsilon && abs(sample.avg_error) > uniforms.epsilon) {
        let signal = sample.activation * uniforms.weight;
        let signal_sign = sign_nonzero(signal);
        let error_sign = sign_nonzero(sample.avg_error);
        if (signal_sign != 0.0 && error_sign != 0.0 && signal_sign == error_sign) {
            contribution.harmful_flag = 1u;
            contribution.error_magnitude = abs(sample.avg_error);
        } else if (signal_sign != 0.0 && error_sign != 0.0) {
            contribution.helpful_flag = 1u;
        }
    }

    contributions[idx] = contribution;
}

