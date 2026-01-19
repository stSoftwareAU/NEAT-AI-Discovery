// GPU workgroup reduction shader for HarmfulContribution aggregation (Issue #218).
//
// This shader performs parallel tree reduction within workgroups to reduce
// GPU→CPU data transfer. Instead of transferring one HarmfulContribution per
// sample (16 bytes each), we reduce to one partial sum per workgroup.
//
// For 100K samples with 256-thread workgroups:
// - Before: 100,000 × 16 bytes = 1.6MB transfer
// - After: 391 × 16 bytes = 6.3KB transfer (255× reduction)

struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
};

struct ReductionUniforms {
    // Total number of contributions to reduce
    contribution_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0)
var<storage, read> contributions: array<HarmfulContribution>;

@group(0) @binding(1)
var<storage, read_write> partial_sums: array<HarmfulContribution>;

@group(0) @binding(2)
var<uniform> uniforms: ReductionUniforms;

// Shared memory for workgroup-level reduction
var<workgroup> shared_data: array<HarmfulContribution, 256>;

// Add two HarmfulContribution structs together
fn add_contributions(a: HarmfulContribution, b: HarmfulContribution) -> HarmfulContribution {
    var result: HarmfulContribution;
    result.harmful_flag = a.harmful_flag + b.harmful_flag;
    result.helpful_flag = a.helpful_flag + b.helpful_flag;
    result.error_magnitude = a.error_magnitude + b.error_magnitude;
    result.pad0 = 0.0;
    return result;
}

// Create a zeroed HarmfulContribution
fn zero_contribution() -> HarmfulContribution {
    var result: HarmfulContribution;
    result.harmful_flag = 0u;
    result.helpful_flag = 0u;
    result.error_magnitude = 0.0;
    result.pad0 = 0.0;
    return result;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let local_idx = local_id.x;
    let global_idx = group_id.x * 256u + local_idx;

    // Load contribution into shared memory (or zero if out of bounds)
    if (global_idx < uniforms.contribution_count) {
        shared_data[local_idx] = contributions[global_idx];
    } else {
        shared_data[local_idx] = zero_contribution();
    }

    // Synchronise to ensure all threads have loaded their data
    workgroupBarrier();

    // Tree reduction within workgroup
    // Each iteration halves the number of active threads
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_data[local_idx] = add_contributions(
                shared_data[local_idx],
                shared_data[local_idx + stride]
            );
        }
        workgroupBarrier();
    }

    // First thread writes the workgroup's partial sum
    if (local_idx == 0u) {
        partial_sums[group_id.x] = shared_data[0];
    }
}
