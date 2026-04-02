//! Activation function GPU evaluation module.
//!
//! Extracted from `gpu/analyzer.rs` (Issue #520) to reduce file size
//! and improve maintainability. Contains the activation function evaluation
//! pipeline builder and GPU evaluation methods (single and batched).

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::{Context, Result};
use bytemuck::Zeroable;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::analysis::gpu::device::{
    GPU_BUFFER_MAP_TIMEOUT_SECS, wait_for_buffer_map, wait_for_buffer_maps_batch,
};
use crate::analysis::gpu::pipeline_builder::{STANDARD_BINDINGS, build_compute_pipeline};
use crate::analysis::gpu::shaders::{
    ACTIVATION_REDUCE_SHADER, ACTIVATION_SHADER, GPU_REDUCTION_THRESHOLD, WORKGROUP_SIZE,
};
use crate::analysis::samples::{
    ActivationOutput, ActivationUniforms, EPSILON, GpuHelpfulSample, HelpfulSample,
    ReductionUniforms,
};

use super::analyzer::GpuAnalyzer;

// =============================================================================
// Pipeline Builder
// =============================================================================

impl GpuAnalyzer {
    pub(super) fn build_activation_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        build_compute_pipeline(
            device,
            "activation-shader",
            ACTIVATION_SHADER,
            "activation-bind-group",
            &STANDARD_BINDINGS,
            label,
        )
    }

    /// Build the activation output reduction pipeline (Issue #567).
    ///
    /// This pipeline aggregates `ActivationOutput` data on the GPU using parallel
    /// tree reduction within workgroups, reducing GPU→CPU transfer by ~255×.
    pub(super) fn build_activation_reduce_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        build_compute_pipeline(
            device,
            "activation-reduce-shader",
            ACTIVATION_REDUCE_SHADER,
            "activation-reduce-bind-group",
            &STANDARD_BINDINGS,
            label,
        )
    }
}

// =============================================================================
// Evaluation Methods
// =============================================================================

impl GpuAnalyzer {
    /// GPU-accelerated general activation evaluation.
    ///
    /// Returns (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`).
    ///
    /// For sample sets with >= `GPU_REDUCTION_THRESHOLD` samples, uses GPU-side
    /// workgroup reduction to minimise data transfer (Issue #567).
    pub fn evaluate_activation_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Returns: (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .context("GPU device unavailable for activation analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-outputs-buffer"),
            contents: bytemuck::cast_slice(&outputs_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ActivationUniforms {
            sample_count: samples.len() as u32,
            orientation,
            scale,
            activation_type,
            epsilon: EPSILON,
            pad0: 0.0,
            pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: activation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("activation-bind-group"),
        });

        let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation-command-encoder"),
        });

        // Per-sample computation pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        // Issue #567: Use GPU-side reduction for large sample counts
        let use_reduction = samples.len() >= GPU_REDUCTION_THRESHOLD;

        let (staging_buffer, transfer_size) = if use_reduction {
            let activation_reduce_layout = self
                .activation_reduce_layout
                .as_ref()
                .context("GPU activation reduce layout not initialised")?;
            let activation_reduce_pipeline = self
                .activation_reduce_pipeline
                .as_ref()
                .context("GPU activation reduce pipeline not initialised")?;

            let num_workgroups = workgroups;
            let partial_sums_size =
                (std::mem::size_of::<ActivationOutput>() * num_workgroups as usize) as u64;

            let partial_sums_zeroed = vec![ActivationOutput::zeroed(); num_workgroups as usize];
            let partial_sums_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("activation-partial-sums-buffer"),
                    contents: bytemuck::cast_slice(&partial_sums_zeroed),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

            let reduction_uniforms = ReductionUniforms {
                contribution_count: samples.len() as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let reduction_uniform_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("activation-reduction-uniform-buffer"),
                    contents: bytemuck::bytes_of(&reduction_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let reduction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: activation_reduce_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: outputs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partial_sums_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: reduction_uniform_buffer.as_entire_binding(),
                    },
                ],
                label: Some("activation-reduction-bind-group"),
            });

            // Reduction compute pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("activation-reduction-compute-pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(activation_reduce_pipeline);
                compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("activation-staging-buffer-reduced"),
                size: partial_sums_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(&partial_sums_buffer, 0, &staging, 0, partial_sums_size);

            (staging, partial_sums_size)
        } else {
            let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("activation-staging-buffer"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(&outputs_buffer, 0, &staging, 0, output_size);

            (staging, output_size)
        };

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
            .context("Activation buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

        let mut sum_activation_sq = 0.0f32;
        let mut sum_error_activation = 0.0f32;

        if use_reduction {
            // Reduction path: aggregate partial sums (much fewer elements)
            for output in outputs {
                sum_activation_sq += output.output_sq;
                sum_error_activation += output.error_output;
            }

            if outputs.len() > 1 {
                tracing::trace!(
                    partial_sums = outputs.len(),
                    "GPU reduction (activation): transferred partial sums instead of full outputs"
                );
            }
        } else {
            // Direct path: iterate over all outputs
            for (idx, output) in outputs.iter().enumerate() {
                if idx < samples.len() && output.valid > 0 {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            }
        }

        // Baseline error computed on CPU (same for both paths)
        let total_baseline_error_sq: f32 = samples
            .iter()
            .filter(|s| s.avg_error.is_finite())
            .map(|s| s.avg_error * s.avg_error)
            .sum();

        // Note: improved_count is not calculated here because it requires weight validation
        // that happens in the caller. The caller will calculate improved_count after
        // validating and clamping the outgoing_weight.

        let _ = transfer_size; // Used for buffer sizing
        drop(data);
        staging_buffer.unmap();

        Ok((
            sum_activation_sq,
            sum_error_activation,
            total_baseline_error_sq,
            0, // improved_count calculated by caller after weight validation
        ))
    }

    /// GPU-accelerated batched activation evaluation.
    ///
    /// Issue #201: Evaluates multiple activation function configurations in a single
    /// GPU command buffer submission, reducing CPU-GPU round-trips by 10-20%.
    ///
    /// Issue #567: For sample sets with >= `GPU_REDUCTION_THRESHOLD` samples, uses
    /// GPU-side workgroup reduction to minimise data transfer by ~255×.
    ///
    /// Instead of submitting separate GPU operations for each (`activation_type`, orientation, scale)
    /// combination, this method:
    /// 1. Uploads sample data once
    /// 2. Creates all compute passes in a single command buffer
    /// 3. Optionally reduces outputs on GPU for large sample counts
    /// 4. Maps all result buffers together with a single `device.poll(Wait)`
    /// 5. Processes all results in a single CPU pass
    ///
    /// # Arguments
    /// * `samples` - The sample data to evaluate (uploaded once)
    /// * `activation_configs` - List of (`activation_type`, orientation, scale) tuples
    ///
    /// # Returns
    /// Vector of (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`)
    /// in the same order as the input configs.
    pub fn evaluate_activations_batched_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        // Handle edge cases
        if activation_configs.is_empty() {
            return Ok(Vec::new());
        }

        if samples.is_empty() {
            // Return zero results for each config
            return Ok(vec![(0.0, 0.0, 0.0, 0); activation_configs.len()]);
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .context("GPU device unavailable for activation analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        // Issue #567: Check if we should use GPU reduction
        let use_reduction = samples.len() >= GPU_REDUCTION_THRESHOLD;

        // Pre-convert samples to GPU format once (shared across all configs)
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();

        // Pre-compute baseline error (same for all configs)
        let total_baseline_error_sq: f32 = samples
            .iter()
            .filter(|s| s.avg_error.is_finite())
            .map(|s| s.avg_error * s.avg_error)
            .sum();

        // Create a single sample buffer (shared across all compute passes)
        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batched-activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Calculate sizes
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];
        let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);

        // Create output buffers, staging buffers, uniform buffers, and bind groups for each config
        let mut output_buffers = Vec::with_capacity(activation_configs.len());
        let mut staging_buffers = Vec::with_capacity(activation_configs.len());
        let mut bind_groups = Vec::with_capacity(activation_configs.len());
        // Issue #567: Track partial sums buffers for reduction path
        let mut partial_sums_buffers: Vec<Option<wgpu::Buffer>> =
            Vec::with_capacity(activation_configs.len());

        for (config_idx, &(activation_type, orientation, scale)) in
            activation_configs.iter().enumerate()
        {
            // Output buffer for this config
            let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("batched-activation-outputs-buffer-{config_idx}")),
                contents: bytemuck::cast_slice(&outputs_zeroed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

            // Uniform buffer for this config
            let uniforms = ActivationUniforms {
                sample_count: samples.len() as u32,
                orientation,
                scale,
                activation_type,
                epsilon: EPSILON,
                pad0: 0.0,
                pad1: 0.0,
            };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("batched-activation-uniform-buffer-{config_idx}")),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Bind group for this config (shares sample_buffer)
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: activation_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sample_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outputs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("batched-activation-bind-group-{config_idx}")),
            });

            if use_reduction {
                // Create partial sums buffer and staging for reduced output
                let partial_sums_size =
                    (std::mem::size_of::<ActivationOutput>() * workgroups as usize) as u64;
                let partial_sums_zeroed = vec![ActivationOutput::zeroed(); workgroups as usize];
                let partial_sums_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("batched-activation-partial-sums-{config_idx}")),
                        contents: bytemuck::cast_slice(&partial_sums_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("batched-activation-staging-reduced-{config_idx}")),
                    size: partial_sums_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                partial_sums_buffers.push(Some(partial_sums_buffer));
                staging_buffers.push(staging_buffer);
            } else {
                // Direct transfer path
                let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("batched-activation-staging-buffer-{config_idx}")),
                    size: output_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                partial_sums_buffers.push(None);
                staging_buffers.push(staging_buffer);
            }

            output_buffers.push(outputs_buffer);
            bind_groups.push(bind_group);
        }

        // Create a SINGLE command encoder for ALL compute passes
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched-activation-command-encoder"),
        });

        // Issue all per-sample compute passes in a single command buffer
        for (config_idx, bind_group) in bind_groups.iter().enumerate() {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("batched-activation-compute-pass-{config_idx}")),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        // Issue #567: Add reduction passes and copy commands
        if use_reduction {
            let activation_reduce_layout = self
                .activation_reduce_layout
                .as_ref()
                .context("GPU activation reduce layout not initialised")?;
            let activation_reduce_pipeline = self
                .activation_reduce_pipeline
                .as_ref()
                .context("GPU activation reduce pipeline not initialised")?;

            let reduction_uniforms = ReductionUniforms {
                contribution_count: samples.len() as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let reduction_uniform_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batched-activation-reduction-uniforms"),
                    contents: bytemuck::bytes_of(&reduction_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let partial_sums_size =
                (std::mem::size_of::<ActivationOutput>() * workgroups as usize) as u64;

            for (config_idx, output_buffer) in output_buffers.iter().enumerate() {
                let Some(partial_sums_buffer) = partial_sums_buffers[config_idx].as_ref() else {
                    continue;
                };

                let reduction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: activation_reduce_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: partial_sums_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: reduction_uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some(&format!(
                        "batched-activation-reduction-bind-group-{config_idx}"
                    )),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("batched-activation-reduction-pass-{config_idx}")),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(activation_reduce_pipeline);
                    compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroups, 1, 1);
                }

                // Copy partial sums to staging
                encoder.copy_buffer_to_buffer(
                    partial_sums_buffer,
                    0,
                    &staging_buffers[config_idx],
                    0,
                    partial_sums_size,
                );
            }
        } else {
            // Direct path: copy all output buffers to staging buffers
            let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
            for (output_buffer, staging_buffer) in output_buffers.iter().zip(staging_buffers.iter())
            {
                encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, output_size);
            }
        }

        // Submit the SINGLE command buffer with ALL operations
        queue.submit(Some(encoder.finish()));

        // Issue map_async for ALL staging buffers BEFORE polling
        // This allows the GPU to process the map requests in parallel
        let receivers: Vec<_> = staging_buffers
            .iter()
            .map(|staging_buffer| {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                receiver
            })
            .collect();

        // Wait for ALL buffers to be mapped with a single polling loop
        wait_for_buffer_maps_batch(device, &receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Batched activation buffer mapping failed")?;

        // Process all results
        let mut results = Vec::with_capacity(activation_configs.len());
        for staging_buffer in &staging_buffers {
            let buffer_slice = staging_buffer.slice(..);
            let data = buffer_slice.get_mapped_range();
            let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

            let mut sum_activation_sq = 0.0f32;
            let mut sum_error_activation = 0.0f32;

            if use_reduction {
                // Reduction path: aggregate partial sums
                for output in outputs {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            } else {
                // Direct path: iterate over all outputs
                for (idx, output) in outputs.iter().enumerate() {
                    if idx < samples.len() && output.valid > 0 {
                        sum_activation_sq += output.output_sq;
                        sum_error_activation += output.error_output;
                    }
                }
            }

            drop(data);
            staging_buffer.unmap();

            results.push((
                sum_activation_sq,
                sum_error_activation,
                total_baseline_error_sq,
                0, // improved_count calculated by caller after weight validation
            ));
        }

        Ok(results)
    }
}
