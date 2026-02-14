//! Harmful synapse GPU evaluation module.
//!
//! Extracted from `gpu/analyzer.rs` (Issue #520) to reduce file size
//! and improve maintainability. Contains the harmful synapse evaluation
//! pipeline builder and batch evaluation methods.

use anyhow::{Context, Result};
use bytemuck::Zeroable;
use std::sync::mpsc;
use std::time::Duration;
use wgpu::util::DeviceExt;

use crate::analysis::gpu::device::{
    GPU_BUFFER_MAP_TIMEOUT_SECS, poll_device_until_idle, wait_for_buffer_maps_batch,
};
use crate::analysis::gpu::shaders::{
    GPU_REDUCTION_THRESHOLD, HARMFUL_REDUCE_SHADER, HARMFUL_SHADER, WORKGROUP_SIZE,
};
use crate::analysis::samples::{
    EPSILON, GpuHelpfulSample, HarmfulContribution, HarmfulStats, HarmfulUniforms, HelpfulSample,
    ReductionUniforms,
};
use crate::analysis::utils::{cap_gpu_batch_size_by_bytes, verbose_enabled};

use super::analyzer::{GPU_MAX_BATCH_ALLOC_BYTES, GpuAnalyzer};

// =============================================================================
// Pipeline Builders
// =============================================================================

impl GpuAnalyzer {
    pub(super) fn build_harmful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-synapse-bind-group"),
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

    /// Build the harmful contribution reduction pipeline (Issue #218).
    ///
    /// This pipeline aggregates HarmfulContribution data on the GPU using parallel
    /// tree reduction within workgroups, reducing GPU→CPU transfer by ~250×.
    pub(super) fn build_harmful_reduce_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-reduce-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_REDUCE_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-reduce-bind-group"),
            entries: &[
                // Binding 0: contributions (read-only input)
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
                // Binding 1: partial_sums (read-write output)
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
                // Binding 2: uniforms
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
// Batch Evaluation Methods
// =============================================================================

impl GpuAnalyzer {
    /// Batch evaluate multiple harmful synapse operations to improve GPU utilisation.
    /// Each entry is (samples, weight) pair. Returns stats in the same order as input.
    ///
    /// This reduces CPU-GPU round trips by submitting multiple GPU dispatches in a single
    /// command buffer, significantly improving throughput for harmful synapse analysis.
    ///
    /// For sample sets with >= GPU_REDUCTION_THRESHOLD samples, uses GPU-side
    /// workgroup reduction to minimise data transfer (Issue #218).
    pub fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
    ) -> Result<Vec<HarmfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched harmful analysis")?;
        let harmful_layout = self
            .harmful_layout
            .as_ref()
            .context("GPU layout not initialised for batched harmful analysis")?;
        let harmful_pipeline = self
            .harmful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched harmful analysis")?;
        // Issue #218: Get reduction pipeline for large sample sets
        let harmful_reduce_layout = self
            .harmful_reduce_layout
            .as_ref()
            .context("GPU harmful reduce layout not initialised")?;
        let harmful_reduce_pipeline = self
            .harmful_reduce_pipeline
            .as_ref()
            .context("GPU harmful reduce pipeline not initialised")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch
            .iter()
            .map(|(samples, _)| samples.len())
            .max()
            .unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HarmfulContribution>());
        let effective_batch_size = cap_gpu_batch_size_by_bytes(
            self.batch_size,
            max_sample_len,
            bytes_per_sample,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );

        if verbose_enabled() && effective_batch_size < self.batch_size {
            let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
            let approx_mb = (bytes_per_op as f64 / (1024.0 * 1024.0)).max(0.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Capping GPU harmful batch size from {} to {} due to large sample count. \
                 max_sample_len={}, approx_buffers_per_op\u{2248}{:.1}MB, cap={}MB",
                self.batch_size,
                effective_batch_size,
                max_sample_len,
                approx_mb,
                GPU_MAX_BATCH_ALLOC_BYTES / (1024 * 1024)
            );
        }

        for batch_chunk in samples_batch.chunks(effective_batch_size) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();
            // Issue #218: Track whether each sample set uses reduction
            let mut uses_reduction_flags: Vec<bool> = Vec::with_capacity(batch_chunk.len());

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("harmful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for (samples, weight) in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    uses_reduction_flags.push(false);
                    continue;
                }
                empty_flags.push(false);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HarmfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("harmful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HarmfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    weight: *weight,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: harmful_layout,
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
                    label: Some("harmful-bind-group-batch"),
                });

                // Add compute pass for per-sample contribution calculation
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("harmful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(harmful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                // Issue #218: Use reduction for large sample counts
                let use_reduction = samples.len() >= GPU_REDUCTION_THRESHOLD;
                uses_reduction_flags.push(use_reduction);

                if use_reduction {
                    // Calculate number of workgroups for reduction
                    let num_workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    let partial_sums_size = (std::mem::size_of::<HarmfulContribution>()
                        * num_workgroups as usize)
                        as u64;

                    // Create partial sums buffer for reduction output
                    let partial_sums_zeroed =
                        vec![HarmfulContribution::zeroed(); num_workgroups as usize];
                    let partial_sums_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("harmful-partial-sums-buffer"),
                            contents: bytemuck::cast_slice(&partial_sums_zeroed),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        });

                    // Create reduction uniforms
                    let reduction_uniforms = ReductionUniforms {
                        contribution_count: samples.len() as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    let reduction_uniform_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("harmful-reduction-uniform-buffer"),
                            contents: bytemuck::bytes_of(&reduction_uniforms),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    // Create reduction bind group
                    let reduction_bind_group =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: harmful_reduce_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: contributions_buffer.as_entire_binding(),
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
                            label: Some("harmful-reduction-bind-group"),
                        });

                    // Add reduction compute pass
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("harmful-reduction-compute-pass"),
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(harmful_reduce_pipeline);
                        compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                    }

                    // Staging buffer for partial sums (much smaller than full contributions)
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("harmful-staging-buffer-reduced"),
                        size: partial_sums_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(partial_sums_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push(partial_sums_size);
                } else {
                    // Original path: transfer all contributions to CPU
                    let contribution_size =
                        (std::mem::size_of::<HarmfulContribution>() * samples.len()) as u64;
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("harmful-staging-buffer-batch"),
                        size: contribution_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(contributions_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push(contribution_size);
                }
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let contribution_size = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // OPTIMISATION: Map ALL buffers first, then poll ONCE for all.
            // This reduces GPU-CPU round trips compared to mapping each buffer individually.
            let mut map_receivers = Vec::with_capacity(batch_staging_buffers.len());
            for staging_buffer in &batch_staging_buffers {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                map_receivers.push(receiver);
            }

            // Event-driven wait: poll non-blocking, check all callback channels
            wait_for_buffer_maps_batch(device, &map_receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
                .context("Harmful batch buffer mapping failed")?;

            // Process results - maintain order with empty flags
            // Note: wait_for_buffer_maps_batch already verified all buffers are mapped
            // Issue #218: Filter uses_reduction_flags to only include non-empty samples
            let non_empty_reduction_flags: Vec<bool> = uses_reduction_flags
                .iter()
                .zip(empty_flags.iter())
                .filter(|&(_, &empty)| !empty)
                .map(|(&reduce, _)| reduce)
                .collect();

            let mut buffer_idx = 0;
            for is_empty in empty_flags {
                if is_empty {
                    all_results.push(HarmfulStats::default());
                } else {
                    // Buffer is already mapped and verified by wait_for_buffer_maps_batch
                    let staging_buffer = &batch_staging_buffers[buffer_idx];
                    let buffer_slice = staging_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let contributions: &[HarmfulContribution] = bytemuck::cast_slice(&data);

                    let mut stats = HarmfulStats::default();
                    // Both reduction and non-reduction paths produce HarmfulContribution,
                    // we just iterate over fewer elements when using reduction
                    for contribution in contributions {
                        stats.harmful_count += contribution.harmful_flag;
                        stats.helpful_count += contribution.helpful_flag;
                        stats.harmful_error_sum += contribution.error_magnitude;
                    }

                    // Log reduction usage for verbose output
                    if verbose_enabled()
                        && non_empty_reduction_flags
                            .get(buffer_idx)
                            .copied()
                            .unwrap_or(false)
                    {
                        let num_partial_sums = contributions.len();
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] GPU reduction (harmful): transferred {num_partial_sums} partial sums instead of full contributions"
                        );
                    }

                    drop(data);
                    staging_buffer.unmap();

                    all_results.push(stats);
                    buffer_idx += 1;
                }
            }

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-harmful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }
}
