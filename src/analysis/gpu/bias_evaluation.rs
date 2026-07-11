//! Bias GPU evaluation module.
//!
//! Extracted from `gpu/analyzer.rs` (Issue #520) to reduce file size
//! and improve maintainability. Contains the bias evaluation pipeline
//! builder and GPU evaluation method.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::{Context, Result};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::analysis::gpu::device::{GPU_BUFFER_MAP_TIMEOUT_SECS, wait_for_buffer_map};
use crate::analysis::gpu::pipeline_builder::{BIAS_BINDINGS, build_compute_pipeline};
use crate::analysis::gpu::shaders::{BIAS_SHADER, MIN_NEURON_SAMPLE_COUNT, WORKGROUP_SIZE};
use crate::analysis::samples::{
    BiasResult, BiasUniforms, EPSILON, GpuHelpfulSample, HelpfulSample,
};

use super::analyzer::GpuAnalyzer;

// =============================================================================
// Pipeline Builder
// =============================================================================

impl GpuAnalyzer {
    pub(super) fn build_bias_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        build_compute_pipeline(
            device,
            "bias-shader",
            BIAS_SHADER,
            "bias-bind-group",
            &BIAS_BINDINGS,
            label,
        )
    }
}

// =============================================================================
// Evaluation Method
// =============================================================================

impl GpuAnalyzer {
    /// GPU-accelerated bias grid search.
    /// Tests all bias values in parallel and returns the optimal bias.
    pub fn evaluate_bias_gpu(
        &self,
        samples: &[HelpfulSample],
        incoming_weight: f32,
        outgoing_weight: f32,
        activation_type: u32,
        bias_range: (f32, f32, f32),
    ) -> Result<f32> {
        // Returns: optimal bias value
        if samples.is_empty() {
            return Ok(0.0);
        }

        let (min_bias, max_bias, step) = bias_range;

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .context("GPU device unavailable for bias analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for bias analysis")?;
        let bias_layout = self
            .bias_layout
            .as_ref()
            .context("GPU bias layout not initialised")?;
        let bias_pipeline = self
            .bias_pipeline
            .as_ref()
            .context("GPU bias pipeline not initialised")?;

        // Guard against zero or negative step to prevent division-by-zero / UB (Issue #805)
        if step < f32::EPSILON {
            return Ok(0.0);
        }

        // Generate bias candidates
        let num_steps = ((max_bias - min_bias) / step).ceil() as i32 + 1;
        let bias_candidates: Vec<f32> = (0..num_steps)
            .map(|i| min_bias + (i as f32 * step).min(max_bias - min_bias))
            .collect();

        if bias_candidates.is_empty() {
            return Ok(0.0);
        }

        // Prepare GPU buffers
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let results_zeroed = vec![BiasResult::zeroed(); bias_candidates.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-candidates-buffer"),
            contents: bytemuck::cast_slice(&bias_candidates),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-results-buffer"),
            contents: bytemuck::cast_slice(&results_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = BiasUniforms {
            sample_count: samples.len() as u32,
            bias_count: bias_candidates.len() as u32,
            incoming_weight,
            outgoing_weight,
            activation_type,
            epsilon: EPSILON,
            min_sample_count: MIN_NEURON_SAMPLE_COUNT as u32,
            pad0: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bias_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("bias-bind-group"),
        });

        let output_size = (std::mem::size_of::<BiasResult>() * bias_candidates.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bias-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bias-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bias-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(bias_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (bias_candidates.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, output_size);

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
            .context("Bias buffer mapping failed")?;

        let data = buffer_slice
            .get_mapped_range()
            .context("Bias staging buffer get_mapped_range failed")?;
        let results: &[BiasResult] = bytemuck::cast_slice(&data);

        // Find bias with best error reduction
        let mut best_bias = 0.0;
        let mut best_error_reduction = f32::NEG_INFINITY;

        for result in results {
            if result.valid_sample_count >= MIN_NEURON_SAMPLE_COUNT as u32
                && result.error_reduction > best_error_reduction
            {
                best_error_reduction = result.error_reduction;
                best_bias = result.bias_value;
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(best_bias)
    }
}
