struct TargetRecord {
    obs_index: u32,
    error_start_index: u32,
    error_count: u32,
    pad0: u32,
};

struct FromRecord {
    obs_index: u32,
    activation: f32,
};

struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct MatchingUniforms {
    target_count: u32,
    from_count: u32,
    total_errors: u32,
    pad0: u32,
};

@group(0) @binding(0)
var<storage, read> target_records: array<TargetRecord>;
@group(0) @binding(1)
var<storage, read> errors: array<f32>;
@group(0) @binding(2)
var<storage, read> from_records: array<FromRecord>;
@group(0) @binding(3)
var<storage, read_write> samples: array<HelpfulSample>;
@group(0) @binding(4)
var<uniform> uniforms: MatchingUniforms;

const MAX_F32: f32 = 3.402823466e+38;
const QUIET_NAN_BITS: u32 = 0x7fc00000u;

fn is_finite_value(value: f32) -> bool {
    if (value != value) {
        return false;
    }
    return abs(value) <= MAX_F32;
}

fn quiet_nan() -> f32 {
    return bitcast<f32>(QUIET_NAN_BITS);
}

// Binary search for matching obs_index in sorted target_records
fn find_target_index(search_obs: u32) -> i32 {
    var left: i32 = 0;
    var right: i32 = i32(uniforms.target_count) - 1;

    while (left <= right) {
        let mid = (left + right) / 2;
        let mid_obs = target_records[u32(mid)].obs_index;

        if (mid_obs == search_obs) {
            return mid;
        } else if (mid_obs < search_obs) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1; // Not found
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.from_count) {
        return;
    }

    let from_rec = from_records[idx];

    // Skip if activation is not finite
    if (!is_finite_value(from_rec.activation)) {
        samples[idx] = HelpfulSample(0.0, quiet_nan());
        return;
    }

    // Binary search for matching target record
    let target_idx = find_target_index(from_rec.obs_index);

    if (target_idx >= 0) {
        let target_rec = target_records[u32(target_idx)];

        // Compute avg_error on GPU from error array
        var error_sum: f32 = 0.0;
        var error_count: u32 = 0u;
        
        for (var i: u32 = 0u; i < target_rec.error_count; i++) {
            let error_idx = target_rec.error_start_index + i;
            if (error_idx < uniforms.total_errors) {
                let error_val = errors[error_idx];
                if (is_finite_value(error_val)) {
                    error_sum += error_val;
                    error_count += 1u;
                }
            }
        }

        // Compute average error
        var avg_error: f32;
        if (error_count > 0u) {
            avg_error = error_sum / f32(error_count);
        } else {
            avg_error = quiet_nan();
        }

        // Skip if the averaged error is not finite
        if (is_finite_value(avg_error)) {
            samples[idx] = HelpfulSample(from_rec.activation, avg_error);
        } else {
            samples[idx] = HelpfulSample(0.0, quiet_nan());
        }
    } else {
        // No match found
        samples[idx] = HelpfulSample(0.0, quiet_nan());
    }
}

