// GPU workgroup reduction shader for ReluContribution aggregation (Issue #567).
//
// This shader performs parallel tree reduction within workgroups to reduce
// GPU→CPU data transfer. Instead of transferring one ReluContribution per
// sample (40 bytes each), we reduce to one partial sum per workgroup.
//
// For 100K samples with 256-thread workgroups:
// - Before: 100,000 × 40 bytes = 4.0MB transfer
// - After: 391 × 40 bytes = 15.6KB transfer (255× reduction)

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

struct ReductionUniforms {
    // Total number of contributions to reduce
    contribution_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0)
var<storage, read> contributions: array<ReluContribution>;

@group(0) @binding(1)
var<storage, read_write> partial_sums: array<ReluContribution>;

@group(0) @binding(2)
var<uniform> uniforms: ReductionUniforms;

// Shared memory for workgroup-level reduction
var<workgroup> shared_data: array<ReluContribution, 256>;

// Add two ReluContribution structs together
fn add_contributions(a: ReluContribution, b: ReluContribution) -> ReluContribution {
    var result: ReluContribution;
    result.positive_activation_sq = a.positive_activation_sq + b.positive_activation_sq;
    result.positive_error_activation = a.positive_error_activation + b.positive_error_activation;
    result.positive_count = a.positive_count + b.positive_count;
    result.negative_activation_sq = a.negative_activation_sq + b.negative_activation_sq;
    result.negative_error_activation = a.negative_error_activation + b.negative_error_activation;
    result.negative_count = a.negative_count + b.negative_count;
    result.error_sq = a.error_sq + b.error_sq;
    result.pad0 = 0.0;
    result.pad1 = 0u;
    result.pad2 = 0u;
    return result;
}

// Create a zeroed ReluContribution
fn zero_contribution() -> ReluContribution {
    var result: ReluContribution;
    result.positive_activation_sq = 0.0;
    result.positive_error_activation = 0.0;
    result.positive_count = 0u;
    result.negative_activation_sq = 0.0;
    result.negative_error_activation = 0.0;
    result.negative_count = 0u;
    result.error_sq = 0.0;
    result.pad0 = 0.0;
    result.pad1 = 0u;
    result.pad2 = 0u;
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
