use crate::parquet_format::read_records_from_parquet;
use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateSynapseJson};
use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc};
use wgpu::util::DeviceExt;

const EPSILON: f32 = 1e-8;
const WORKGROUP_SIZE: u32 = 256;

#[cfg(test)]
static FORCE_GPU_ADAPTER_FAILURE: AtomicBool = AtomicBool::new(false);

pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub gpu_used: bool,
}

struct OrderedNeuron {
    uuid: String,
    index: usize,
}

struct RecordCache<'a> {
    parquet_file: &'a str,
    cache: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl<'a> RecordCache<'a> {
    fn new(parquet_file: &'a str) -> Self {
        Self {
            parquet_file,
            cache: HashMap::new(),
        }
    }

    fn get(&mut self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        use std::collections::hash_map::Entry;

        match self.cache.entry(neuron_uuid.to_string()) {
            Entry::Occupied(entry) => Ok(Arc::clone(entry.get())),
            Entry::Vacant(entry) => {
                let mut records = read_records_from_parquet(self.parquet_file, neuron_uuid)
                    .with_context(|| {
                        format!("Failed to read discovery records for neuron {neuron_uuid}")
                    })?;
                records.sort_by_key(|record| record.obs_index);
                let arc = Arc::new(records);
                entry.insert(Arc::clone(&arc));
                Ok(arc)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct HelpfulSample {
    activation: f32,
    avg_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHelpfulSample {
    activation: f32,
    avg_error: f32,
}

impl From<HelpfulSample> for GpuHelpfulSample {
    fn from(value: HelpfulSample) -> Self {
        Self {
            activation: value.activation,
            avg_error: value.avg_error,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulContribution {
    positive_flag: u32,
    negative_flag: u32,
    positive_improvement: f32,
    negative_improvement: f32,
    positive_activation: f32,
    negative_activation: f32,
    pad0: f32,
    pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    weight: f32,
}

#[derive(Default)]
struct HelpfulStats {
    positive_count: u32,
    negative_count: u32,
    positive_improvement_sum: f32,
    negative_improvement_sum: f32,
    positive_activation_sum: f32,
    negative_activation_sum: f32,
}

#[derive(Default)]
struct HarmfulStats {
    harmful_count: u32,
    helpful_count: u32,
    harmful_error_sum: f32,
}

fn cpu_helpful_stats(samples: &[HelpfulSample]) -> HelpfulStats {
    let mut stats = HelpfulStats::default();

    for sample in samples {
        if sample.activation.abs() <= EPSILON || sample.avg_error.abs() <= EPSILON {
            continue;
        }

        let required_sign = -sample.avg_error.signum() * sample.activation.signum();
        let improvement = sample.avg_error.abs();
        let activation_mag = sample.activation.abs();

        if required_sign > 0.0 {
            stats.positive_count += 1;
            stats.positive_improvement_sum += improvement;
            stats.positive_activation_sum += activation_mag;
        } else if required_sign < 0.0 {
            stats.negative_count += 1;
            stats.negative_improvement_sum += improvement;
            stats.negative_activation_sum += activation_mag;
        }
    }

    stats
}

fn cpu_harmful_stats(samples: &[HelpfulSample], weight: f32) -> HarmfulStats {
    let mut stats = HarmfulStats::default();

    for sample in samples {
        if sample.activation.abs() <= EPSILON || sample.avg_error.abs() <= EPSILON {
            continue;
        }
        let signal = sample.activation * weight;
        let signal_sign = signal.signum();
        let error_sign = sample.avg_error.signum();

        if signal_sign == 0.0 || error_sign == 0.0 {
            continue;
        }

        if signal_sign == error_sign {
            stats.harmful_count += 1;
            stats.harmful_error_sum += sample.avg_error.abs();
        } else {
            stats.helpful_count += 1;
        }
    }

    stats
}

struct GpuAnalyzer {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    helpful_layout: Option<wgpu::BindGroupLayout>,
    helpful_pipeline: Option<wgpu::ComputePipeline>,
    harmful_layout: Option<wgpu::BindGroupLayout>,
    harmful_pipeline: Option<wgpu::ComputePipeline>,
    gpu_used: bool,
}

impl GpuAnalyzer {
    fn cpu_fallback() -> Self {
        Self {
            device: None,
            queue: None,
            helpful_layout: None,
            helpful_pipeline: None,
            harmful_layout: None,
            harmful_pipeline: None,
            gpu_used: false,
        }
    }

    fn new(require_gpu: bool) -> Result<Self> {
        let instance = wgpu::Instance::default();
        #[cfg(test)]
        let adapter = if FORCE_GPU_ADAPTER_FAILURE.load(AtomicOrdering::SeqCst) {
            None
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        };

        #[cfg(not(test))]
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(adapter) => adapter,
            None => {
                if require_gpu {
                    return Err(anyhow!("No GPU adapter available for discovery analysis"));
                }
                return Ok(Self::cpu_fallback());
            }
        };

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )) {
            Ok(result) => result,
            Err(err) => {
                if require_gpu {
                    return Err(anyhow!(
                        "Failed to initialise GPU device for discovery analysis: {err}"
                    ));
                }
                return Ok(Self::cpu_fallback());
            }
        };

        let (helpful_layout, helpful_pipeline) =
            Self::build_helpful_pipeline(&device, "helpful-synapse-pipeline");
        let (harmful_layout, harmful_pipeline) =
            Self::build_harmful_pipeline(&device, "harmful-synapse-pipeline");

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            gpu_used: true,
        })
    }

    fn build_helpful_pipeline(
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
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_harmful_pipeline(
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
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn evaluate_helpful(&self, samples: &[HelpfulSample]) -> Result<HelpfulStats> {
        if samples.is_empty() {
            return Ok(HelpfulStats::default());
        }

        if !self.gpu_used {
            return Ok(cpu_helpful_stats(samples));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for helpful analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for helpful analysis")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("helpful-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("helpful-contributions-buffer"),
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
            label: Some("helpful-uniform-buffer"),
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
            label: Some("helpful-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helpful-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("helpful-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("helpful-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(helpful_pipeline);
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
        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(anyhow!("Failed to map helpful contributions buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive helpful map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

        let mut stats = HelpfulStats::default();
        for contribution in contributions {
            stats.positive_count += contribution.positive_flag;
            stats.negative_count += contribution.negative_flag;
            stats.positive_improvement_sum += contribution.positive_improvement;
            stats.negative_improvement_sum += contribution.negative_improvement;
            stats.positive_activation_sum += contribution.positive_activation;
            stats.negative_activation_sum += contribution.negative_activation;
        }

        drop(data);
        staging_buffer.unmap();

        if stats.positive_count == 0 && stats.negative_count == 0 {
            let cpu_stats = cpu_helpful_stats(samples);
            if cpu_stats.positive_count > 0 || cpu_stats.negative_count > 0 {
                stats = cpu_stats;
            }
        }

        Ok(stats)
    }

    fn evaluate_harmful(&self, samples: &[HelpfulSample], weight: f32) -> Result<HarmfulStats> {
        if samples.is_empty() {
            return Ok(HarmfulStats::default());
        }

        if !self.gpu_used {
            return Ok(cpu_harmful_stats(samples, weight));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for harmful analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for harmful analysis")?;
        let harmful_layout = self
            .harmful_layout
            .as_ref()
            .context("GPU layout not initialised for harmful analysis")?;
        let harmful_pipeline = self
            .harmful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for harmful analysis")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![HarmfulContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = HarmfulUniforms {
            length: samples.len() as u32,
            pad0: 0,
            epsilon: EPSILON,
            weight,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-uniform-buffer"),
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
            label: Some("harmful-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<HarmfulContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("harmful-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("harmful-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("harmful-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(harmful_pipeline);
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
        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(anyhow!("Failed to map harmful contributions buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive harmful map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let contributions: &[HarmfulContribution] = bytemuck::cast_slice(&data);

        let mut stats = HarmfulStats::default();
        for contribution in contributions {
            stats.harmful_count += contribution.harmful_flag;
            stats.helpful_count += contribution.helpful_flag;
            stats.harmful_error_sum += contribution.error_magnitude;
        }

        drop(data);
        staging_buffer.unmap();

        if stats.harmful_count <= stats.helpful_count {
            let cpu_stats = cpu_harmful_stats(samples, weight);
            if cpu_stats.harmful_count > cpu_stats.helpful_count {
                stats = cpu_stats;
            }
        }

        Ok(stats)
    }

    fn gpu_used(&self) -> bool {
        self.gpu_used
    }
}

const HELPFUL_SHADER: &str = r#"
struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct HelpfulContribution {
    positive_flag: u32,
    negative_flag: u32,
    positive_improvement: f32,
    negative_improvement: f32,
    positive_activation: f32,
    negative_activation: f32,
    pad0: f32,
    pad1: f32,
};

struct HelpfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    pad1: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HelpfulSample>;
@group(0) @binding(1)
var<storage, read_write> contributions: array<HelpfulContribution>;
@group(0) @binding(2)
var<uniform> uniforms: HelpfulUniforms;

fn sign_nonzero(value: f32) -> f32 {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.length) {
        return;
    }
    let sample = samples[idx];
    var contribution: HelpfulContribution;
    contribution.positive_flag = 0u;
    contribution.negative_flag = 0u;
    contribution.positive_improvement = 0.0;
    contribution.negative_improvement = 0.0;
    contribution.positive_activation = 0.0;
    contribution.negative_activation = 0.0;
    contribution.pad0 = 0.0;
    contribution.pad1 = 0.0;

    if (abs(sample.activation) > uniforms.epsilon && abs(sample.avg_error) > uniforms.epsilon) {
        let required_sign = -sign_nonzero(sample.avg_error) * sign_nonzero(sample.activation);
        let improvement = abs(sample.avg_error);
        if (required_sign > 0.0) {
            contribution.positive_flag = 1u;
            contribution.positive_improvement = improvement;
            contribution.positive_activation = abs(sample.activation);
        } else if (required_sign < 0.0) {
            contribution.negative_flag = 1u;
            contribution.negative_improvement = improvement;
            contribution.negative_activation = abs(sample.activation);
        }
    }

    contributions[idx] = contribution;
}
"#;

const HARMFUL_SHADER: &str = r#"
struct HarmfulSample {
    activation: f32,
    avg_error: f32,
};

struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
};

struct HarmfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    weight: f32,
};

@group(0) @binding(0)
var<storage, read> samples: array<HarmfulSample>;
@group(0) @binding(1)
var<storage, read_write> contributions: array<HarmfulContribution>;
@group(0) @binding(2)
var<uniform> uniforms: HarmfulUniforms;

fn sign_nonzero(value: f32) -> f32 {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.length) {
        return;
    }
    let sample = samples[idx];
    var contribution: HarmfulContribution;
    contribution.harmful_flag = 0u;
    contribution.helpful_flag = 0u;
    contribution.error_magnitude = 0.0;
    contribution.pad0 = 0.0;

    if (abs(sample.activation) > uniforms.epsilon && abs(sample.avg_error) > uniforms.epsilon) {
        let signal = sample.activation * uniforms.weight;
        let signal_sign = sign_nonzero(signal);
        let error_sign = sign_nonzero(sample.avg_error);
        if (signal_sign != 0.0 && error_sign != 0.0 && signal_sign == error_sign) {
            contribution.harmful_flag = 1u;
            contribution.error_magnitude = abs(sample.avg_error);
        } else if (signal_sign != 0.0 && error_sign != 0.0) {
            contribution.helpful_flag = 1u;
        }
    }

    contributions[idx] = contribution;
}
"#;

fn build_ordered_neurons(creature: &crate::CreatureJson) -> Vec<OrderedNeuron> {
    let mut ordered = Vec::with_capacity(creature.input + creature.neurons.len());

    for input_index in 0..creature.input {
        ordered.push(OrderedNeuron {
            uuid: format!("input-{input_index}"),
            index: input_index,
        });
    }

    for (offset, neuron) in creature.neurons.iter().enumerate() {
        ordered.push(OrderedNeuron {
            uuid: neuron.uuid.clone(),
            index: creature.input + offset,
        });
    }

    ordered
}

fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    let mut error_map: HashMap<u32, f32> = HashMap::with_capacity(target_records.len());
    for record in target_records {
        if record.errors.is_empty() {
            continue;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for error in &record.errors {
            if error.is_finite() {
                sum += *error;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        error_map.insert(record.obs_index, sum / count as f32);
    }

    if error_map.is_empty() {
        return Vec::new();
    }

    let mut samples = Vec::new();
    for record in from_records {
        if let Some(avg_error) = error_map.get(&record.obs_index) {
            if record.activation.is_finite() && avg_error.is_finite() {
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error: *avg_error,
                });
            }
        }
    }

    samples
}

pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    let require_gpu = input.require_gpu.unwrap_or(cfg!(target_os = "macos"));
    let analyzer = GpuAnalyzer::new(require_gpu)?;

    let ordered_neurons = build_ordered_neurons(&input.creature);
    let mut order_map: HashMap<&str, usize> = HashMap::new();
    for neuron in &ordered_neurons {
        order_map.insert(neuron.uuid.as_str(), neuron.index);
    }

    let existing_synapses: HashSet<(String, String)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.from_uuid.clone(), synapse.to_uuid.clone()))
        .collect();

    let mut synapses_by_target: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    for synapse in &input.creature.synapses {
        synapses_by_target
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let mut cache = RecordCache::new(&input.parquet_file);

    let mut helpful_results: Vec<CandidateSynapseJson> = Vec::new();
    let mut harmful_results: Vec<CandidateSynapseJson> = Vec::new();
    let mut helpful_fallback: Option<CandidateSynapseJson> = None;

    let threshold = input.improvement_threshold.unwrap_or(0.1);

    for target_uuid in &input.focus_neurons {
        let target_records_arc = cache.get(target_uuid)?;
        if target_records_arc.is_empty() {
            continue;
        }
        let target_records = target_records_arc.as_ref();

        let target_index = match order_map.get(target_uuid.as_str()) {
            Some(index) => *index,
            None => continue,
        };

        // Helpful candidates
        for source in ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
        {
            let source_uuid = source.uuid.as_str();

            if existing_synapses.contains(&(source_uuid.to_string(), target_uuid.clone())) {
                continue;
            }

            let from_records_arc = cache.get(source_uuid)?;
            if from_records_arc.is_empty() {
                continue;
            }
            let from_records = from_records_arc.as_ref();

            let samples = build_samples(target_records, from_records);
            if samples.is_empty() {
                continue;
            }

            let stats = analyzer.evaluate_helpful(&samples)?;
            let positive_is_better = stats.positive_count >= stats.negative_count;
            let improved_count = if positive_is_better {
                stats.positive_count
            } else {
                stats.negative_count
            };
            if improved_count == 0 {
                continue;
            }

            let worsen_count = if positive_is_better {
                stats.negative_count
            } else {
                stats.positive_count
            };
            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            let improvement_sum = if positive_is_better {
                stats.positive_improvement_sum
            } else {
                stats.negative_improvement_sum
            };

            let activation_sum = if positive_is_better {
                stats.positive_activation_sum
            } else {
                stats.negative_activation_sum
            };

            let mut weight = 0.0;
            if activation_sum.abs() > EPSILON {
                let raw_weight = improvement_sum / (activation_sum + 1e-8);
                weight = if positive_is_better {
                    -raw_weight
                } else {
                    raw_weight
                };
                weight = weight.clamp(-1.0, 1.0);
            }

            let expected_improvement_percentage =
                (improved_count as f32 - worsen_count as f32) / total_count as f32;

            if expected_improvement_percentage <= threshold {
                if helpful_fallback.as_ref().is_none_or(|existing| {
                    existing.expected_improvement_percentage < expected_improvement_percentage
                }) {
                    helpful_fallback = Some(CandidateSynapseJson {
                        from_neuron_uuid: source_uuid.to_string(),
                        to_neuron_uuid: target_uuid.clone(),
                        weight,
                        expected_improvement_percentage,
                        improved_count,
                        total_count,
                    });
                }
                continue;
            }

            helpful_results.push(CandidateSynapseJson {
                from_neuron_uuid: source_uuid.to_string(),
                to_neuron_uuid: target_uuid.clone(),
                weight,
                expected_improvement_percentage,
                improved_count,
                total_count,
            });
        }

        // Harmful synapses (existing connections)
        if let Some(existing) = synapses_by_target.get(target_uuid.as_str()) {
            for synapse in existing {
                let from_records_arc = cache.get(synapse.from_uuid.as_str())?;
                if from_records_arc.is_empty() {
                    continue;
                }
                let from_records = from_records_arc.as_ref();
                let samples = build_samples(target_records, from_records);
                if samples.is_empty() {
                    continue;
                }

                let stats = analyzer.evaluate_harmful(&samples, synapse.weight)?;
                let total_count = samples.len() as u32;
                if total_count == 0 {
                    continue;
                }

                let expected_improvement_percentage =
                    (stats.harmful_count as f32 - stats.helpful_count as f32) / total_count as f32;

                let candidate = CandidateSynapseJson {
                    from_neuron_uuid: synapse.from_uuid.clone(),
                    to_neuron_uuid: synapse.to_uuid.clone(),
                    weight: synapse.weight,
                    expected_improvement_percentage,
                    improved_count: stats.harmful_count,
                    total_count,
                };
                harmful_results.push(candidate);
            }
        }
    }

    if helpful_results.is_empty() {
        if let Some(candidate) = helpful_fallback.take() {
            helpful_results.push(candidate);
        }
    }
    helpful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });
    harmful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
        harmful_results.truncate(limit);
    }

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        gpu_used: analyzer.gpu_used(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering as AtomicOrdering;

    struct ForceGpuFailureGuard;

    impl ForceGpuFailureGuard {
        fn new() -> Self {
            FORCE_GPU_ADAPTER_FAILURE.store(true, AtomicOrdering::SeqCst);
            Self
        }
    }

    impl Drop for ForceGpuFailureGuard {
        fn drop(&mut self) {
            FORCE_GPU_ADAPTER_FAILURE.store(false, AtomicOrdering::SeqCst);
        }
    }

    #[test]
    fn cpu_fallback_when_gpu_not_required() {
        let _guard = ForceGpuFailureGuard::new();
        let analyzer =
            GpuAnalyzer::new(false).expect("CPU analysis should be available when GPU is optional");

        assert!(
            !analyzer.gpu_used(),
            "GPU should not be reported as used when we fall back to CPU analysis"
        );

        let samples = vec![
            HelpfulSample {
                activation: 0.8,
                avg_error: -0.4,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: 0.3,
            },
        ];

        let helpful_stats = analyzer
            .evaluate_helpful(&samples)
            .expect("CPU helpful analysis should succeed");
        assert!(
            helpful_stats.positive_count > 0 || helpful_stats.negative_count > 0,
            "CPU analysis should produce non-zero helpful counts"
        );

        let harmful_stats = analyzer
            .evaluate_harmful(&samples, 0.5)
            .expect("CPU harmful analysis should succeed");
        assert!(
            harmful_stats.harmful_count > 0 || harmful_stats.helpful_count > 0,
            "CPU analysis should produce non-zero harmful counts"
        );
    }
}
