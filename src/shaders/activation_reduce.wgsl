// GPU workgroup reduction shader for ActivationOutput aggregation (Issue #567).
//
// This shader performs parallel tree reduction within workgroups to reduce
// GPU→CPU data transfer. Instead of transferring one ActivationOutput per
// sample (28 bytes each), we reduce to one partial sum per workgroup.
//
// For 100K samples with 256-thread workgroups:
// - Before: 100,000 × 28 bytes = 2.8MB transfer
// - After: 391 × 28 bytes = 10.9KB transfer (255× reduction)

struct ActivationOutput {
    output: f32,
    output_sq: f32,
    error_output: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct ReductionUniforms {
    // Total number of outputs to reduce
    contribution_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0)
var<storage, read> outputs: array<ActivationOutput>;

@group(0) @binding(1)
var<storage, read_write> partial_sums: array<ActivationOutput>;

@group(0) @binding(2)
var<uniform> uniforms: ReductionUniforms;

// Shared memory for workgroup-level reduction
var<workgroup> shared_data: array<ActivationOutput, 256>;

// Add two ActivationOutput structs together
fn add_outputs(a: ActivationOutput, b: ActivationOutput) -> ActivationOutput {
    var result: ActivationOutput;
    result.output = a.output + b.output;
    result.output_sq = a.output_sq + b.output_sq;
    result.error_output = a.error_output + b.error_output;
    result.valid = a.valid + b.valid;
    result.pad0 = 0u;
    result.pad1 = 0u;
    result.pad2 = 0u;
    return result;
}

// Create a zeroed ActivationOutput
fn zero_output() -> ActivationOutput {
    var result: ActivationOutput;
    result.output = 0.0;
    result.output_sq = 0.0;
    result.error_output = 0.0;
    result.valid = 0u;
    result.pad0 = 0u;
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

    // Load output into shared memory (or zero if out of bounds)
    if (global_idx < uniforms.contribution_count) {
        shared_data[local_idx] = outputs[global_idx];
    } else {
        shared_data[local_idx] = zero_output();
    }

    // Synchronise to ensure all threads have loaded their data
    workgroupBarrier();

    // Tree reduction within workgroup
    // Each iteration halves the number of active threads
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_data[local_idx] = add_outputs(
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
