struct TargetRecord {
    obs_index: u32,
    avg_error: f32,
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
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<storage, read> target_records: array<TargetRecord>;
@group(0) @binding(1)
var<storage, read> from_records: array<FromRecord>;
@group(0) @binding(2)
var<storage, read_write> samples: array<HelpfulSample>;
@group(0) @binding(3)
var<uniform> uniforms: MatchingUniforms;

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

    // Skip if activation is NaN (NaN != NaN is true)
    if (from_rec.activation != from_rec.activation) {
        samples[idx] = HelpfulSample(0.0, 0.0);
        return;
    }

    // Binary search for matching target record
    let target_idx = find_target_index(from_rec.obs_index);

    if (target_idx >= 0) {
        let target_rec = target_records[u32(target_idx)];

        // Skip if error is NaN
        if (target_rec.avg_error == target_rec.avg_error) {
            samples[idx] = HelpfulSample(from_rec.activation, target_rec.avg_error);
        } else {
            samples[idx] = HelpfulSample(0.0, 0.0);
        }
    } else {
        // No match found
        samples[idx] = HelpfulSample(0.0, 0.0);
    }
}

