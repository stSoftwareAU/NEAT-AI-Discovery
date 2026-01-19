// GPU workgroup reduction shader for HelpfulContribution aggregation (Issue #218).
//
// This shader performs parallel tree reduction within workgroups to reduce
// GPU→CPU data transfer. Instead of transferring one HelpfulContribution per
// sample (48 bytes each), we reduce to one partial sum per workgroup.
//
// For 100K samples with 256-thread workgroups:
// - Before: 100,000 × 48 bytes = 4.8MB transfer
// - After: 391 × 48 bytes = 18.8KB transfer (255× reduction)

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

struct ReductionUniforms {
    // Total number of contributions to reduce
    contribution_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0)
var<storage, read> contributions: array<HelpfulContribution>;

@group(0) @binding(1)
var<storage, read_write> partial_sums: array<HelpfulContribution>;

@group(0) @binding(2)
var<uniform> uniforms: ReductionUniforms;

// Shared memory for workgroup-level reduction
var<workgroup> shared_data: array<HelpfulContribution, 256>;

// Add two HelpfulContribution structs together
fn add_contributions(a: HelpfulContribution, b: HelpfulContribution) -> HelpfulContribution {
    var result: HelpfulContribution;
    result.positive_flag = a.positive_flag + b.positive_flag;
    result.negative_flag = a.negative_flag + b.negative_flag;
    result.positive_improvement = a.positive_improvement + b.positive_improvement;
    result.negative_improvement = a.negative_improvement + b.negative_improvement;
    result.positive_activation = a.positive_activation + b.positive_activation;
    result.negative_activation = a.negative_activation + b.negative_activation;
    result.error_squared = a.error_squared + b.error_squared;
    result.activation_squared = a.activation_squared + b.activation_squared;
    result.error_activation = a.error_activation + b.error_activation;
    result.pad0 = 0.0;
    result.pad1 = 0.0;
    result.pad2 = 0.0;
    return result;
}

// Create a zeroed HelpfulContribution
fn zero_contribution() -> HelpfulContribution {
    var result: HelpfulContribution;
    result.positive_flag = 0u;
    result.negative_flag = 0u;
    result.positive_improvement = 0.0;
    result.negative_improvement = 0.0;
    result.positive_activation = 0.0;
    result.negative_activation = 0.0;
    result.error_squared = 0.0;
    result.activation_squared = 0.0;
    result.error_activation = 0.0;
    result.pad0 = 0.0;
    result.pad1 = 0.0;
    result.pad2 = 0.0;
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
