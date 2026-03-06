//! ReLU activation GPU evaluation module.
//!
//! Extracted from `gpu/analyzer.rs` (Issue #520) to reduce file size
//! and improve maintainability. Contains the ReLU activation evaluation
//! pipeline builder and GPU evaluation method.

use anyhow::{Context, Result};
use bytemuck::Zeroable;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::analysis::gpu::device::{GPU_BUFFER_MAP_TIMEOUT_SECS, wait_for_buffer_map};
use crate::analysis::gpu::shaders::{RELU_SHADER, WORKGROUP_SIZE};
use crate::analysis::samples::{
    EPSILON, GpuHelpfulSample, HelpfulSample, ReluContribution, ReluOrientation, ReluStats,
    ReluUniforms,
};

use super::analyzer::GpuAnalyzer;

// =============================================================================
// Pipeline Builder
// =============================================================================

impl GpuAnalyzer {
    pub(super) fn build_relu_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relu-shader"),
            source: wgpu::ShaderSource::Wgsl(RELU_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relu-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        (layout, pipeline)
    }
}

// =============================================================================
// Evaluation Method
// =============================================================================

impl GpuAnalyzer {
    /// GPU-accelerated ReLU evaluation.
    ///
    /// Evaluates ReLU activation for neuron candidates, returning statistics for
    /// both positive and negative orientations plus baseline error.
    pub fn evaluate_relu_gpu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .context("GPU device unavailable for ReLU analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for ReLU analysis")?;
        let relu_layout = self
            .relu_layout
            .as_ref()
            .context("GPU ReLU layout not initialised")?;
        let relu_pipeline = self
            .relu_pipeline
            .as_ref()
            .context("GPU ReLU pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![ReluContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ReluUniforms {
            length: samples.len() as u32,
            threshold,
            epsilon: EPSILON,
            pad0: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: relu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contributions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("relu-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<ReluContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relu-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relu-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(relu_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &contributions_buffer,
            0,
            &staging_buffer,
            0,
            contribution_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("ReLU buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let contributions: &[ReluContribution] = bytemuck::cast_slice(&data);

        let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
        let mut negative_stats = ReluStats::new(ReluOrientation::Negative);
        let mut total_baseline_error_sq = 0.0;

        // Accumulate statistics from GPU contributions
        for (idx, contribution) in contributions.iter().enumerate() {
            if idx < samples.len() {
                total_baseline_error_sq += contribution.error_sq;

                // For positive ReLU, accumulate activation_sq and error_activation
                if contribution.positive_count > 0 {
                    positive_stats.activation_sq_sum += contribution.positive_activation_sq;
                    positive_stats.error_activation_sum += contribution.positive_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.positive_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.positive_error_activation / activation
                    } else {
                        0.0
                    };
                    positive_stats.samples.push((activation, error));
                }

                // For negative ReLU, accumulate activation_sq and error_activation
                if contribution.negative_count > 0 {
                    negative_stats.activation_sq_sum += contribution.negative_activation_sq;
                    negative_stats.error_activation_sum += contribution.negative_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.negative_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.negative_error_activation / activation
                    } else {
                        0.0
                    };
                    negative_stats.samples.push((activation, error));
                }
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok((positive_stats, negative_stats, total_baseline_error_sq))
    }
}
