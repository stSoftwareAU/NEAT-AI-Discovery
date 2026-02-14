//! Helpful synapse GPU evaluation module.
//!
//! Extracted from `gpu/analyzer.rs` (Issue #520) to reduce file size
//! and improve maintainability. Contains the helpful synapse evaluation
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
    GPU_REDUCTION_THRESHOLD, HELPFUL_REDUCE_SHADER, HELPFUL_SHADER, WORKGROUP_SIZE,
};
use crate::analysis::samples::{
    EPSILON, GpuHelpfulSample, HelpfulContribution, HelpfulSample, HelpfulStats, HelpfulUniforms,
    ReductionUniforms,
};
use crate::analysis::utils::{cap_gpu_batch_size_by_bytes, verbose_enabled};

use super::analyzer::{GPU_MAX_BATCH_ALLOC_BYTES, GpuAnalyzer};

// =============================================================================
// Pipeline Builders
// =============================================================================

impl GpuAnalyzer {
    pub(super) fn build_helpful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-synapse-bind-group"),
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

    /// Build the helpful contribution reduction pipeline (Issue #218).
    ///
    /// This pipeline aggregates HelpfulContribution data on the GPU using parallel
    /// tree reduction within workgroups, reducing GPU→CPU transfer by ~250×.
    pub(super) fn build_helpful_reduce_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-reduce-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_REDUCE_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-reduce-bind-group"),
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
    /// Batch evaluate multiple helpful operations to improve GPU utilisation.
    /// Returns a vector of stats in the same order as the input samples.
    ///
    /// For sample sets with >= GPU_REDUCTION_THRESHOLD samples, uses GPU-side
    /// workgroup reduction to minimise data transfer (Issue #218).
    pub fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
    ) -> Result<Vec<HelpfulStats>> {
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
            .context("GPU queue not initialised for batched helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for batched helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched helpful analysis")?;
        // Issue #218: Get reduction pipeline for large sample sets
        let helpful_reduce_layout = self
            .helpful_reduce_layout
            .as_ref()
            .context("GPU helpful reduce layout not initialised")?;
        let helpful_reduce_pipeline = self
            .helpful_reduce_pipeline
            .as_ref()
            .context("GPU helpful reduce pipeline not initialised")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch.iter().map(|s| s.len()).max().unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HelpfulContribution>());
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
                "[NEAT-AI-Discovery][verbose] Capping GPU helpful batch size from {} to {} due to large sample count. \
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
                label: Some("helpful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for samples in batch_chunk {
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
                let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("helpful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HelpfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    pad1: 0.0,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: helpful_layout,
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
                    label: Some("helpful-bind-group-batch"),
                });

                // Add compute pass for per-sample contribution calculation
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("helpful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(helpful_pipeline);
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
                    let partial_sums_size = (std::mem::size_of::<HelpfulContribution>()
                        * num_workgroups as usize)
                        as u64;

                    // Create partial sums buffer for reduction output
                    let partial_sums_zeroed =
                        vec![HelpfulContribution::zeroed(); num_workgroups as usize];
                    let partial_sums_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("helpful-partial-sums-buffer"),
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
                            label: Some("helpful-reduction-uniform-buffer"),
                            contents: bytemuck::bytes_of(&reduction_uniforms),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    // Create reduction bind group
                    let reduction_bind_group =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: helpful_reduce_layout,
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
                            label: Some("helpful-reduction-bind-group"),
                        });

                    // Add reduction compute pass
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("helpful-reduction-compute-pass"),
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(helpful_reduce_pipeline);
                        compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                    }

                    // Staging buffer for partial sums (much smaller than full contributions)
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("helpful-staging-buffer-reduced"),
                        size: partial_sums_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(partial_sums_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push((partial_sums_size, num_workgroups as usize));
                } else {
                    // Original path: transfer all contributions to CPU
                    let contribution_size =
                        (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("helpful-staging-buffer-batch"),
                        size: contribution_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(contributions_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push((contribution_size, samples.len()));
                }
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let (contribution_size, _) = batch_contribution_sizes[i];
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
                .context("Helpful batch buffer mapping failed")?;

            // Now read all the mapped data (buffers are already mapped)
            let mut batch_results = Vec::with_capacity(batch_contribution_sizes.len());
            // Buffer mappings already verified by wait_for_buffer_maps_batch
            // Issue #218: Filter uses_reduction_flags to only include non-empty samples
            let non_empty_reduction_flags: Vec<bool> = uses_reduction_flags
                .iter()
                .zip(empty_flags.iter())
                .filter(|&(_, &empty)| !empty)
                .map(|(&reduce, _)| reduce)
                .collect();

            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let buffer_slice = staging_buffer.slice(..);
                let data = buffer_slice.get_mapped_range();
                let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

                let mut stats = HelpfulStats::default();
                // Both reduction and non-reduction paths produce HelpfulContribution,
                // we just iterate over fewer elements when using reduction
                for contribution in contributions {
                    stats.positive_count += contribution.positive_flag;
                    stats.negative_count += contribution.negative_flag;
                    stats.positive_improvement_sum += contribution.positive_improvement;
                    stats.negative_improvement_sum += contribution.negative_improvement;
                    stats.positive_activation_sum += contribution.positive_activation;
                    stats.negative_activation_sum += contribution.negative_activation;
                    stats.error_sq_sum += contribution.error_squared;
                    stats.activation_sq_sum += contribution.activation_squared;
                    stats.error_activation_sum += contribution.error_activation;
                }

                // Log reduction usage for verbose output
                if verbose_enabled() && non_empty_reduction_flags.get(i).copied().unwrap_or(false) {
                    let (_, count) = batch_contribution_sizes[i];
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] GPU reduction: transferred {count} partial sums instead of full contributions"
                    );
                }

                drop(data);
                staging_buffer.unmap();

                batch_results.push(stats);
            }

            let merged_results = Self::merge_batch_results(&empty_flags, batch_results);
            all_results.extend(merged_results);

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-helpful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }

    /// Merge batch results, inserting default stats for empty sample sets.
    ///
    /// This is a helper for batch processing that maintains ordering when some
    /// input sample sets are empty and were skipped during GPU processing.
    pub fn merge_batch_results(
        empty_flags: &[bool],
        computed_stats: Vec<HelpfulStats>,
    ) -> Vec<HelpfulStats> {
        let expected_non_empty = empty_flags.iter().filter(|flag| !**flag).count();
        debug_assert_eq!(
            expected_non_empty,
            computed_stats.len(),
            "Computed stats should match number of non-empty sample sets"
        );

        let mut results = Vec::with_capacity(empty_flags.len());
        let mut stats_iter = computed_stats.into_iter();

        for &is_empty in empty_flags {
            if is_empty {
                results.push(HelpfulStats::default());
            } else if let Some(stats) = stats_iter.next() {
                results.push(stats);
            } else {
                // Safety guard: if counts mismatch, preserve ordering by inserting default.
                results.push(HelpfulStats::default());
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::HelpfulStats;

    #[test]
    fn test_merge_batch_results_empty() {
        let empty_flags: Vec<bool> = vec![];
        let computed: Vec<HelpfulStats> = vec![];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_batch_results_all_empty() {
        let empty_flags = vec![true, true, true];
        let computed: Vec<HelpfulStats> = vec![];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert_eq!(result.len(), 3);
        for stats in result {
            assert_eq!(stats.positive_count, 0);
        }
    }

    #[test]
    fn test_merge_batch_results_mixed() {
        let empty_flags = vec![true, false, true, false];
        let computed = vec![
            HelpfulStats {
                positive_count: 5,
                ..Default::default()
            },
            HelpfulStats {
                positive_count: 10,
                ..Default::default()
            },
        ];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].positive_count, 0); // empty
        assert_eq!(result[1].positive_count, 5); // first computed
        assert_eq!(result[2].positive_count, 0); // empty
        assert_eq!(result[3].positive_count, 10); // second computed
    }
}
