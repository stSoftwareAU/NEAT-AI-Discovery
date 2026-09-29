//! Pre-allocation guard against wgpu device limits (Issue #2314).
//!
//! An oversized sample set previously reached `device.create_buffer*` /
//! `create_bind_group` with a size over the device's wgpu limits; wgpu then
//! panics on the uncaptured validation error and kills the GPU thread. This
//! module is a pure, allocation-free check that must run before any buffer
//! or bind-group is created, so an oversized sample set is rejected with a
//! descriptive `anyhow::Error` instead of reaching wgpu at all.

use anyhow::{Result, bail};

use crate::analysis::gpu::shaders::WORKGROUP_SIZE;
use crate::analysis::samples::{
    ActivationOutput, GpuHelpfulSample, HarmfulContribution, HelpfulContribution, ReluContribution,
};

/// Check that a sample set of `sample_count` elements fits within `limits` for
/// every `(binding name, per-sample element bytes)` pair in `bindings`, and
/// that the resulting workgroup dispatch count fits `max_compute_workgroups_per_dimension`.
///
/// `sample_count` of `0` is always `Ok` — there is nothing to allocate.
pub fn check_sample_set_fits(
    path: &str,
    sample_count: usize,
    bindings: &[(&str, usize)],
    limits: &wgpu::Limits,
) -> Result<()> {
    if sample_count == 0 {
        return Ok(());
    }

    for &(binding_name, element_bytes) in bindings {
        let Some(bytes) = sample_count.checked_mul(element_bytes) else {
            bail!(
                "{path}: binding '{binding_name}' size overflows usize for sample_count={sample_count} \
                 element_bytes={element_bytes}"
            );
        };
        let bytes_u64 = bytes as u64;

        if bytes_u64 > limits.max_storage_buffer_binding_size as u64 {
            bail!(
                "{path}: binding '{binding_name}' needs {bytes_u64} bytes for sample_count={sample_count}, \
                 exceeding max_storage_buffer_binding_size ({})",
                limits.max_storage_buffer_binding_size
            );
        }

        if bytes_u64 > limits.max_buffer_size {
            bail!(
                "{path}: binding '{binding_name}' needs {bytes_u64} bytes for sample_count={sample_count}, \
                 exceeding max_buffer_size ({})",
                limits.max_buffer_size
            );
        }
    }

    let workgroups = sample_count.div_ceil(WORKGROUP_SIZE as usize);
    if workgroups > limits.max_compute_workgroups_per_dimension as usize {
        bail!(
            "{path}: sample_count={sample_count} needs {workgroups} workgroups, exceeding \
             max_compute_workgroups_per_dimension ({})",
            limits.max_compute_workgroups_per_dimension
        );
    }

    Ok(())
}

/// Pre-allocation guard for the helpful-synapse evaluation path.
pub fn check_helpful_set_fits(sample_count: usize, limits: &wgpu::Limits) -> Result<()> {
    check_sample_set_fits(
        "helpful",
        sample_count,
        &[
            ("samples", size_of::<GpuHelpfulSample>()),
            ("contributions", size_of::<HelpfulContribution>()),
        ],
        limits,
    )
}

/// Pre-allocation guard for the harmful-synapse evaluation path.
pub fn check_harmful_set_fits(sample_count: usize, limits: &wgpu::Limits) -> Result<()> {
    check_sample_set_fits(
        "harmful",
        sample_count,
        &[
            ("samples", size_of::<GpuHelpfulSample>()),
            ("contributions", size_of::<HarmfulContribution>()),
        ],
        limits,
    )
}

/// Pre-allocation guard for the `ReLU` activation evaluation path.
pub fn check_relu_set_fits(sample_count: usize, limits: &wgpu::Limits) -> Result<()> {
    check_sample_set_fits(
        "relu",
        sample_count,
        &[
            ("samples", size_of::<GpuHelpfulSample>()),
            ("contributions", size_of::<ReluContribution>()),
        ],
        limits,
    )
}

/// Pre-allocation guard for the generic activation evaluation path.
pub fn check_activation_set_fits(sample_count: usize, limits: &wgpu::Limits) -> Result<()> {
    check_sample_set_fits(
        "activation",
        sample_count,
        &[
            ("samples", size_of::<GpuHelpfulSample>()),
            ("outputs", size_of::<ActivationOutput>()),
        ],
        limits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Struct sizes the boundary tests below depend on; a layout change must
    // fail this assertion loudly rather than silently shifting the boundaries.
    #[test]
    fn gpu_struct_sizes_match_boundary_assumptions() {
        assert_eq!(size_of::<HelpfulContribution>(), 48);
        assert_eq!(size_of::<HarmfulContribution>(), 16);
        assert_eq!(size_of::<ReluContribution>(), 40);
        assert_eq!(size_of::<ActivationOutput>(), 28);
        assert_eq!(size_of::<GpuHelpfulSample>(), 8);
    }

    #[test]
    fn helpful_set_at_binding_limit_boundary() {
        let limits = wgpu::Limits::default();
        assert!(check_helpful_set_fits(2_796_202, &limits).is_ok());
        let err = check_helpful_set_fits(2_796_203, &limits).unwrap_err();
        assert!(err.to_string().contains("max_storage_buffer_binding_size"));
    }

    #[test]
    fn harmful_set_at_binding_limit_boundary() {
        let limits = wgpu::Limits::default();
        assert!(check_harmful_set_fits(8_388_608, &limits).is_ok());
        let err = check_harmful_set_fits(8_388_609, &limits).unwrap_err();
        assert!(err.to_string().contains("max_storage_buffer_binding_size"));
    }

    #[test]
    fn relu_set_at_binding_limit_boundary() {
        let limits = wgpu::Limits::default();
        assert!(check_relu_set_fits(3_355_443, &limits).is_ok());
        let err = check_relu_set_fits(3_355_444, &limits).unwrap_err();
        assert!(err.to_string().contains("max_storage_buffer_binding_size"));
    }

    #[test]
    fn activation_set_at_binding_limit_boundary() {
        let limits = wgpu::Limits::default();
        assert!(check_activation_set_fits(4_793_490, &limits).is_ok());
        let err = check_activation_set_fits(4_793_491, &limits).unwrap_err();
        assert!(err.to_string().contains("max_storage_buffer_binding_size"));
    }

    #[test]
    fn workgroup_limit_is_enforced() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 2,
            ..wgpu::Limits::default()
        };
        // WORKGROUP_SIZE is 256, so 2 workgroups cover 512 samples exactly.
        assert!(check_sample_set_fits("tiny", 512, &[("samples", 1)], &limits).is_ok());
        let err = check_sample_set_fits("tiny", 513, &[("samples", 1)], &limits).unwrap_err();
        assert!(
            err.to_string()
                .contains("max_compute_workgroups_per_dimension")
        );
    }

    #[test]
    fn max_buffer_size_is_enforced() {
        let helpful_binding_bytes =
            2_796_202u64.saturating_mul(size_of::<HelpfulContribution>() as u64);
        let limits = wgpu::Limits {
            max_buffer_size: helpful_binding_bytes - 1,
            ..wgpu::Limits::default()
        };
        let err = check_helpful_set_fits(2_796_202, &limits).unwrap_err();
        assert!(err.to_string().contains("max_buffer_size"));
    }

    #[test]
    fn usize_max_samples_does_not_panic() {
        let limits = wgpu::Limits::default();
        assert!(check_helpful_set_fits(usize::MAX, &limits).is_err());
    }

    #[test]
    fn zero_samples_is_always_ok() {
        let limits = wgpu::Limits::default();
        assert!(check_helpful_set_fits(0, &limits).is_ok());
        assert!(check_harmful_set_fits(0, &limits).is_ok());
        assert!(check_relu_set_fits(0, &limits).is_ok());
        assert!(check_activation_set_fits(0, &limits).is_ok());
    }
}
