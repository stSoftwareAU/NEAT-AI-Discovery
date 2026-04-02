//! Shared GPU compute pipeline builder (Issue #978).
//!
//! Provides a reusable helper for constructing `wgpu` bind group layouts,
//! pipeline layouts, and compute pipelines — eliminating the boilerplate
//! that was previously duplicated across the five GPU evaluation modules.

/// Describes a single buffer binding in a compute pipeline's bind group layout.
pub(crate) struct BufferBindingSpec {
    /// Whether the buffer is read-only storage, read-write storage, or uniform.
    pub ty: wgpu::BufferBindingType,
}

/// Build a complete compute pipeline from a WGSL shader source and binding specifications.
///
/// Returns the `(BindGroupLayout, ComputePipeline)` tuple needed by each evaluation module.
/// All pipelines share a common structure: a single bind group layout, entry point `"main"`,
/// and default compilation options.
pub(crate) fn build_compute_pipeline(
    device: &wgpu::Device,
    shader_label: &str,
    shader_source: &str,
    bind_group_label: &str,
    bindings: &[BufferBindingSpec],
    pipeline_label: &str,
) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(i, spec)| wgpu::BindGroupLayoutEntry {
            #[allow(clippy::cast_possible_truncation)] // Bind group index is always < 5
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: spec.ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(bind_group_label),
        entries: &entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(pipeline_label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (layout, pipeline)
}

/// Standard 3-binding layout: storage read-only, storage read-write, uniform.
///
/// Used by helpful, harmful, `ReLU`, activation, and their reduction pipelines.
pub(crate) const STANDARD_BINDINGS: [BufferBindingSpec; 3] = [
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
    },
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
    },
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Uniform,
    },
];

/// Bias-specific 4-binding layout: two storage read-only, one storage read-write, one uniform.
pub(crate) const BIAS_BINDINGS: [BufferBindingSpec; 4] = [
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
    },
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
    },
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
    },
    BufferBindingSpec {
        ty: wgpu::BufferBindingType::Uniform,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_bindings_has_correct_count() {
        assert_eq!(STANDARD_BINDINGS.len(), 3);
    }

    #[test]
    fn test_bias_bindings_has_correct_count() {
        assert_eq!(BIAS_BINDINGS.len(), 4);
    }

    #[test]
    fn test_standard_bindings_types() {
        // Binding 0: storage read-only
        assert!(matches!(
            STANDARD_BINDINGS[0].ty,
            wgpu::BufferBindingType::Storage { read_only: true }
        ));
        // Binding 1: storage read-write
        assert!(matches!(
            STANDARD_BINDINGS[1].ty,
            wgpu::BufferBindingType::Storage { read_only: false }
        ));
        // Binding 2: uniform
        assert!(matches!(
            STANDARD_BINDINGS[2].ty,
            wgpu::BufferBindingType::Uniform
        ));
    }

    #[test]
    fn test_bias_bindings_types() {
        // Binding 0: storage read-only (samples)
        assert!(matches!(
            BIAS_BINDINGS[0].ty,
            wgpu::BufferBindingType::Storage { read_only: true }
        ));
        // Binding 1: storage read-only (bias candidates)
        assert!(matches!(
            BIAS_BINDINGS[1].ty,
            wgpu::BufferBindingType::Storage { read_only: true }
        ));
        // Binding 2: storage read-write (results)
        assert!(matches!(
            BIAS_BINDINGS[2].ty,
            wgpu::BufferBindingType::Storage { read_only: false }
        ));
        // Binding 3: uniform
        assert!(matches!(
            BIAS_BINDINGS[3].ty,
            wgpu::BufferBindingType::Uniform
        ));
    }
}
