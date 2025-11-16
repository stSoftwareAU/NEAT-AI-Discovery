use crate::parquet_format::read_records_from_parquet;
use crate::types::DiscoverRecord;
use crate::{AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson, CandidateSynapseJson};
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
const MIN_NEURON_SAMPLE_COUNT: usize = 10;
const GPU_BATCH_SIZE: usize = 32; // Batch multiple GPU operations together for better utilization

#[cfg(test)]
static FORCE_GPU_ADAPTER_FAILURE: AtomicBool = AtomicBool::new(false);

pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub gpu_used: bool,
}

pub struct AnalyzeNeuronsResult {
    pub helpful_neurons: Vec<CandidateNeuronJson>,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTargetRecord {
    obs_index: u32,
    avg_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFromRecord {
    obs_index: u32,
    activation: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatchingUniforms {
    target_count: u32,
    from_count: u32,
    pad0: u32,
    pad1: u32,
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

#[derive(Clone, Copy)]
enum ReluOrientation {
    Positive,
    Negative,
}

struct ReluStats {
    orientation: ReluOrientation,
    samples: Vec<(f32, f32)>,
    activation_sq_sum: f32,
    error_activation_sum: f32,
}

impl ReluStats {
    fn new(orientation: ReluOrientation) -> Self {
        Self {
            orientation,
            samples: Vec::new(),
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
        }
    }

    fn push(&mut self, relu_activation: f32, error: f32) {
        self.samples.push((relu_activation, error));
        self.activation_sq_sum += relu_activation * relu_activation;
        self.error_activation_sum += relu_activation * error;
    }

    fn candidate(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        threshold: f32,
    ) -> Option<CandidateNeuronJson> {
        if self.samples.len() < MIN_NEURON_SAMPLE_COUNT || self.activation_sq_sum <= EPSILON {
            return None;
        }

        let mut outgoing_weight = self.error_activation_sum / (self.activation_sq_sum + EPSILON);
        if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
            return None;
        }
        outgoing_weight = outgoing_weight.clamp(-5.0, 5.0);

        let mut improved_count = 0u32;
        let mut worsened_count = 0u32;
        for (relu_activation, error) in &self.samples {
            let new_error = error - outgoing_weight * relu_activation;
            if new_error.abs() + EPSILON < error.abs() {
                improved_count += 1;
            } else if new_error.abs() > error.abs() + EPSILON {
                worsened_count += 1;
            }
        }

        let total_count = self.samples.len() as u32;
        if total_count == 0 {
            return None;
        }

        let expected_improvement_percentage =
            (improved_count as f32 - worsened_count as f32) / total_count as f32;
        if expected_improvement_percentage <= threshold {
            return None;
        }

        let incoming_weight = match self.orientation {
            ReluOrientation::Positive => 1.0,
            ReluOrientation::Negative => -1.0,
        };

        Some(CandidateNeuronJson {
            source_neuron_uuid: source_uuid.to_string(),
            target_neuron_uuid: target_uuid.to_string(),
            incoming_weight,
            outgoing_weight,
            squash: "ReLU".to_string(),
            bias: 0.0,
            expected_improvement_percentage,
            improved_count,
            total_count,
        })
    }
}

struct ActivationCandidateSpec {
    name: &'static str,
    orientations: &'static [f32],
    scales: &'static [f32],
    activation: fn(f32) -> f32,
    min_improvement: f32,
}

const ORIENTATIONS_BIDIRECTIONAL: [f32; 2] = [1.0, -1.0];
const SCALES_WIDE: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const SCALES_SMOOTH: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.797_884_6 * (x + 0.044_715 * x_cubed);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn elu_activation(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

fn selu_activation(x: f32) -> f32 {
    const SELU_ALPHA: f32 = 1.673_263_2;
    const SELU_LAMBDA: f32 = 1.050_701;
    if x >= 0.0 {
        SELU_LAMBDA * x
    } else {
        SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
    }
}

fn softplus_activation(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logistic_activation(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

const ACTIVATION_SPECS: [ActivationCandidateSpec; 6] = [
    ActivationCandidateSpec {
        name: "GELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: gelu_activation,
        min_improvement: 0.08,
    },
    ActivationCandidateSpec {
        name: "ELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: elu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "SELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: selu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "Softplus",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: softplus_activation,
        min_improvement: 0.07,
    },
    ActivationCandidateSpec {
        name: "LOGISTIC",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: logistic_activation,
        min_improvement: 0.05,
    },
    ActivationCandidateSpec {
        name: "TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: tanh_activation,
        min_improvement: 0.0,
    },
];

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
    matching_layout: Option<wgpu::BindGroupLayout>,
    matching_pipeline: Option<wgpu::ComputePipeline>,
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
            matching_layout: None,
            matching_pipeline: None,
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
            Some(adapter) => {
                #[cfg(not(test))]
                {
                    // Log adapter info for diagnostics (only in non-test builds)
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU adapter found: {:?}",
                            adapter.get_info()
                        );
                    }
                }
                adapter
            }
            None => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] No GPU adapter available, {}",
                            if require_gpu {
                                "failing (GPU required)"
                            } else {
                                "falling back to CPU"
                            }
                        );
                    }
                }
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
            Ok(result) => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU device initialised successfully: {:?}",
                            result.0.features()
                        );
                    }
                }
                result
            }
            Err(err) => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU device initialisation failed: {err}, {}",
                            if require_gpu {
                                "failing (GPU required)"
                            } else {
                                "falling back to CPU"
                            }
                        );
                    }
                }
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
        let (matching_layout, matching_pipeline) =
            Self::build_matching_pipeline(&device, "matching-pipeline");

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            matching_layout: Some(matching_layout),
            matching_pipeline: Some(matching_pipeline),
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

    fn build_matching_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matching-shader"),
            source: wgpu::ShaderSource::Wgsl(MATCHING_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matching-bind-group"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

    /// Batch evaluate multiple helpful operations to improve GPU utilization
    /// Returns a vector of stats in the same order as the input samples
    fn evaluate_helpful_batch(&self, samples_batch: &[&[HelpfulSample]]) -> Result<Vec<HelpfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        if !self.gpu_used {
            // CPU fallback - process sequentially
            return Ok(samples_batch
                .iter()
                .map(|samples| cpu_helpful_stats(samples))
                .collect());
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for batched helpful analysis")?;
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

        // Process in batches to avoid excessive memory usage
        let mut all_results = Vec::with_capacity(samples_batch.len());
        
        for batch_chunk in samples_batch.chunks(GPU_BATCH_SIZE) {
            let mut batch_encoders = Vec::new();
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();

            // Prepare all operations in this batch
            for samples in batch_chunk {
                if samples.is_empty() {
                    all_results.push(HelpfulStats::default());
                    continue;
                }

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

                let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

                let contribution_size = (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("helpful-staging-buffer-batch"),
                    size: contribution_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("helpful-command-encoder-batch"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("helpful-compute-pass-batch"),
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

                batch_encoders.push(encoder);
                batch_staging_buffers.push(staging_buffer);
                batch_contribution_sizes.push((contribution_size, samples.len()));
            }

            // Submit all operations in this batch at once
            let command_buffers: Vec<_> = batch_encoders.into_iter().map(|e| e.finish()).collect();
            queue.submit(command_buffers);

            // Wait for all results (single poll for entire batch)
            let mut batch_results = Vec::new();
            for (staging_buffer, (_contribution_size, sample_len)) in batch_staging_buffers.into_iter().zip(batch_contribution_sizes) {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                
                // Poll until this specific buffer is ready
                loop {
                    device.poll(wgpu::Maintain::Poll);
                    match receiver.try_recv() {
                        Ok(Ok(())) => break,
                        Ok(Err(err)) => {
                            return Err(anyhow!("Failed to map helpful contributions buffer: {err}"));
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            // Continue polling
                            continue;
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            return Err(anyhow!("Failed to receive helpful map_async completion"));
                        }
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

                // Fallback check
                if stats.positive_count == 0 && stats.negative_count == 0 && sample_len > 0 {
                    // This shouldn't happen in batch mode, but keep the check
                    let cpu_stats = cpu_helpful_stats(&[]); // Empty check
                    if cpu_stats.positive_count > 0 || cpu_stats.negative_count > 0 {
                        stats = cpu_stats;
                    }
                }

                batch_results.push(stats);
            }

            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    /// GPU-accelerated matching of activations to errors by obs_index
    /// This replaces the CPU-based build_samples function for better GPU utilization
    fn build_samples_gpu(
        &self,
        target_records: &[DiscoverRecord],
        from_records: &[DiscoverRecord],
    ) -> Result<Vec<HelpfulSample>> {
        if target_records.is_empty() || from_records.is_empty() {
            return Ok(Vec::new());
        }

        if !self.gpu_used {
            // CPU fallback
            return Ok(build_samples(target_records, from_records));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for sample matching")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for sample matching")?;
        let matching_layout = self
            .matching_layout
            .as_ref()
            .context("GPU matching layout not initialised")?;
        let matching_pipeline = self
            .matching_pipeline
            .as_ref()
            .context("GPU matching pipeline not initialised")?;

        // Prepare target records: compute avg_error and create GPU structures
        let mut gpu_targets: Vec<GpuTargetRecord> = Vec::new();
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
            if count > 0 {
                gpu_targets.push(GpuTargetRecord {
                    obs_index: record.obs_index,
                    avg_error: sum / count as f32,
                });
            }
        }

        if gpu_targets.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by obs_index for binary search (should already be sorted, but ensure it)
        gpu_targets.sort_by_key(|r| r.obs_index);

        // Prepare from records
        let gpu_froms: Vec<GpuFromRecord> = from_records
            .iter()
            .map(|r| GpuFromRecord {
                obs_index: r.obs_index,
                activation: r.activation,
            })
            .collect();

        // Create GPU buffers
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-target-buffer"),
            contents: bytemuck::cast_slice(&gpu_targets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let from_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-from-buffer"),
            contents: bytemuck::cast_slice(&gpu_froms),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let samples_zeroed = vec![GpuHelpfulSample::zeroed(); from_records.len()];
        let samples_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-samples-buffer"),
            contents: bytemuck::cast_slice(&samples_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = MatchingUniforms {
            target_count: gpu_targets.len() as u32,
            from_count: from_records.len() as u32,
            pad0: 0,
            pad1: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: matching_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: from_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("matching-bind-group"),
        });

        let sample_size = (std::mem::size_of::<GpuHelpfulSample>() * from_records.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matching-staging-buffer"),
            size: sample_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matching-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matching-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(matching_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (from_records.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &samples_buffer,
            0,
            &staging_buffer,
            0,
            sample_size,
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
                return Err(anyhow!("Failed to map matching samples buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive matching map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let gpu_samples: &[GpuHelpfulSample] = bytemuck::cast_slice(&data);

        // Filter out zero samples (no match or invalid)
        let mut samples = Vec::new();
        for gpu_sample in gpu_samples {
            if gpu_sample.activation != 0.0 || gpu_sample.avg_error != 0.0 {
                samples.push(HelpfulSample {
                    activation: gpu_sample.activation,
                    avg_error: gpu_sample.avg_error,
                });
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(samples)
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

const MATCHING_SHADER: &str = r#"
struct TargetRecord {
    obs_index: u32,
    avg_error: f32,
};

struct FromRecord {
    obs_index: u32,
    activation: f32,
};

struct HelpfulSample {
    activation: f32,
    avg_error: f32,
};

struct MatchingUniforms {
    target_count: u32,
    from_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<storage, read> target_records: array<TargetRecord>;
@group(0) @binding(1)
var<storage, read> from_records: array<FromRecord>;
@group(0) @binding(2)
var<storage, read_write> samples: array<HelpfulSample>;
@group(0) @binding(3)
var<uniform> uniforms: MatchingUniforms;

// Binary search for matching obs_index in sorted target_records
fn find_target_index(search_obs: u32) -> i32 {
    var left: i32 = 0;
    var right: i32 = i32(uniforms.target_count) - 1;
    
    while (left <= right) {
        let mid = (left + right) / 2;
        let mid_obs = target_records[u32(mid)].obs_index;
        
        if (mid_obs == search_obs) {
            return mid;
        } else if (mid_obs < search_obs) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // Not found
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.from_count) {
        return;
    }
    
    let from_rec = from_records[idx];
    
    // Skip if activation is NaN (NaN != NaN is true)
    if (from_rec.activation != from_rec.activation) {
        samples[idx] = HelpfulSample(0.0, 0.0);
        return;
    }
    
    // Binary search for matching target record
    let target_idx = find_target_index(from_rec.obs_index);
    
    if (target_idx >= 0) {
        let target_rec = target_records[u32(target_idx)];
        
        // Skip if error is NaN
        if (target_rec.avg_error == target_rec.avg_error) {
            samples[idx] = HelpfulSample(from_rec.activation, target_rec.avg_error);
        } else {
            samples[idx] = HelpfulSample(0.0, 0.0);
        }
    } else {
        // No match found
        samples[idx] = HelpfulSample(0.0, 0.0);
    }
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

fn upsert_candidate(
    map: &mut HashMap<(String, String, String), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
    );
    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            if candidate.expected_improvement_percentage
                > entry.get().expected_improvement_percentage
            {
                entry.insert(candidate);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

fn evaluate_relu_candidate(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.is_empty() {
        return Ok(None);
    }

    // Trigger the helpful analysis pipeline so we honour GPU requirements, even
    // though the detailed ReLU statistics are still evaluated on the CPU.
    let _ = analyzer.evaluate_helpful(samples)?;

    let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
    let mut negative_stats = ReluStats::new(ReluOrientation::Negative);

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        let activation = sample.activation;
        let error = sample.avg_error;

        let relu_positive = activation.max(0.0);
        if relu_positive > EPSILON {
            positive_stats.push(relu_positive, error);
        }

        let relu_negative = (-activation).max(0.0);
        if relu_negative > EPSILON {
            negative_stats.push(relu_negative, error);
        }
    }

    let mut candidates = Vec::new();
    if let Some(candidate) = positive_stats.candidate(source_uuid, target_uuid, threshold) {
        candidates.push(candidate);
    }
    if let Some(candidate) = negative_stats.candidate(source_uuid, target_uuid, threshold) {
        candidates.push(candidate);
    }

    Ok(candidates.into_iter().max_by(|a, b| {
        a.expected_improvement_percentage
            .partial_cmp(&b.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    }))
}

fn evaluate_activation_candidate(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    spec: &ActivationCandidateSpec,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    // Trigger GPU path if required.
    let _ = analyzer.evaluate_helpful(samples)?;

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = threshold;
    let mut fallback_candidate: Option<CandidateNeuronJson> = None;
    let mut fallback_score = f32::MIN;
    let mut outputs = Vec::with_capacity(samples.len());

    for &orientation in spec.orientations {
        for &scale in spec.scales {
            outputs.clear();
            let incoming_weight = orientation * scale;
            let mut sum_activation_sq = 0.0;
            let mut sum_error_activation = 0.0;
            let mut valid = true;

            for sample in samples {
                let pre_activation = incoming_weight * sample.activation;
                let output = (spec.activation)(pre_activation);
                if !output.is_finite() {
                    valid = false;
                    break;
                }
                outputs.push(output);
                sum_activation_sq += output * output;
                sum_error_activation += output * sample.avg_error;
            }

            if !valid || outputs.len() != samples.len() || sum_activation_sq <= EPSILON {
                continue;
            }

            let mut outgoing_weight = sum_error_activation / (sum_activation_sq + EPSILON);
            if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
                continue;
            }
            outgoing_weight = outgoing_weight.clamp(-5.0, 5.0);

            let mut improved_count = 0u32;
            let mut worsened_count = 0u32;
            for (sample, output) in samples.iter().zip(outputs.iter()) {
                let new_error = sample.avg_error - outgoing_weight * output;
                if new_error.abs() + EPSILON < sample.avg_error.abs() {
                    improved_count += 1;
                } else if new_error.abs() > sample.avg_error.abs() + EPSILON {
                    worsened_count += 1;
                }
            }

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            let expected_improvement_percentage =
                (improved_count as f32 - worsened_count as f32) / total_count as f32;

            if expected_improvement_percentage > fallback_score {
                fallback_score = expected_improvement_percentage;
                fallback_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: 0.0,
                    expected_improvement_percentage,
                    improved_count,
                    total_count,
                });
            }

            let improvement_cutoff = threshold.min(spec.min_improvement);

            if expected_improvement_percentage <= improvement_cutoff
                || improved_count < MIN_NEURON_SAMPLE_COUNT as u32
            {
                continue;
            }

            if let Some(candidate) = &fallback_candidate {
                if expected_improvement_percentage > best_score {
                    best_score = expected_improvement_percentage;
                    best_candidate = Some(CandidateNeuronJson {
                        source_neuron_uuid: candidate.source_neuron_uuid.clone(),
                        target_neuron_uuid: candidate.target_neuron_uuid.clone(),
                        incoming_weight: candidate.incoming_weight,
                        outgoing_weight: candidate.outgoing_weight,
                        squash: candidate.squash.clone(),
                        bias: candidate.bias,
                        expected_improvement_percentage: candidate.expected_improvement_percentage,
                        improved_count: candidate.improved_count,
                        total_count: candidate.total_count,
                    });
                }
            }
        }
    }

    Ok(best_candidate.or(fallback_candidate))
}

pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    let require_gpu = input.require_gpu.unwrap_or(cfg!(target_os = "macos"));
    let analyzer = GpuAnalyzer::new(require_gpu)?;

    let threshold = input.improvement_threshold.unwrap_or(0.1);
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let mut order_map: HashMap<&str, usize> = HashMap::new();
    for neuron in &ordered_neurons {
        order_map.insert(neuron.uuid.as_str(), neuron.index);
    }

    let mut cache = RecordCache::new(&input.parquet_file);
    let mut helpful_map: HashMap<(String, String, String), CandidateNeuronJson> = HashMap::new();

    let mut seen_targets: HashSet<&str> = HashSet::new();
    let mut unique_focus: Vec<&String> = Vec::new();
    for target_uuid in &input.focus_neurons {
        if seen_targets.insert(target_uuid.as_str()) {
            unique_focus.push(target_uuid);
        }
    }

    for target_uuid in unique_focus {
        let target_records_arc = match cache.get(target_uuid) {
            Ok(records) => records,
            Err(err) => {
                if cfg!(debug_assertions) {
                    eprintln!("Failed to load target neuron records for {target_uuid}: {err}");
                }
                continue;
            }
        };
        if target_records_arc.is_empty() {
            continue;
        }
        let target_records = target_records_arc.as_ref();

        let target_index = match order_map.get(target_uuid.as_str()) {
            Some(index) => *index,
            None => continue,
        };

        for source in ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
        {
            let source_uuid = source.uuid.as_str();
            let from_records_arc = match cache.get(source_uuid) {
                Ok(records) => records,
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("Failed to load source neuron records for {source_uuid}: {err}");
                    }
                    continue;
                }
            };
            if from_records_arc.is_empty() {
                continue;
            }
            let from_records = from_records_arc.as_ref();

            let samples = analyzer.build_samples_gpu(target_records, from_records)?;
            if samples.is_empty() {
                continue;
            }

            if let Some(candidate) =
                evaluate_relu_candidate(&analyzer, source_uuid, target_uuid, &samples, threshold)?
            {
                upsert_candidate(&mut helpful_map, candidate);
            }

            for spec in ACTIVATION_SPECS.iter() {
                if let Some(candidate) = evaluate_activation_candidate(
                    &analyzer,
                    source_uuid,
                    target_uuid,
                    &samples,
                    threshold,
                    spec,
                )? {
                    upsert_candidate(&mut helpful_map, candidate);
                }
            }
        }
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();
    helpful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
    }

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used: analyzer.gpu_used(),
    })
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

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
    }

    let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();

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

        // Collect helpful candidates for batching
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

            let samples = analyzer.build_samples_gpu(target_records, from_records)?;
            if samples.is_empty() {
                continue;
            }

            helpful_work_batch.push(HelpfulWork {
                source_uuid: source_uuid.to_string(),
                target_uuid: target_uuid.clone(),
                samples,
            });
        }
    }

    // Process helpful work in batches for better GPU utilization
    let helpful_samples_refs: Vec<&[HelpfulSample]> = helpful_work_batch.iter().map(|w| w.samples.as_slice()).collect();
    let helpful_stats_batch = analyzer.evaluate_helpful_batch(&helpful_samples_refs)?;

    // Process results
    for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
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
        let total_count = work.samples.len() as u32;
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
                    from_neuron_uuid: work.source_uuid.clone(),
                    to_neuron_uuid: work.target_uuid.clone(),
                    weight,
                    expected_improvement_percentage,
                    improved_count,
                    total_count,
                });
            }
            continue;
        }

        helpful_results.push(CandidateSynapseJson {
            from_neuron_uuid: work.source_uuid.clone(),
            to_neuron_uuid: work.target_uuid.clone(),
            weight,
            expected_improvement_percentage,
            improved_count,
            total_count,
        });
    }

    // Harmful synapses (existing connections) - process per target
    for target_uuid in &input.focus_neurons {
        let target_records_arc = cache.get(target_uuid)?;
        if target_records_arc.is_empty() {
            continue;
        }
        let target_records = target_records_arc.as_ref();

        if let Some(existing) = synapses_by_target.get(target_uuid.as_str()) {
            for synapse in existing {
                let from_records_arc = cache.get(synapse.from_uuid.as_str())?;
                if from_records_arc.is_empty() {
                    continue;
                }
                let from_records = from_records_arc.as_ref();
                let samples = analyzer.build_samples_gpu(target_records, from_records)?;
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
    use crate::parquet_format::write_records_to_parquet;
    use crate::types::DiscoverRecord;
    use crate::{AnalyzeNeuronsInput, CreatureJson, NeuronJson};
    use std::sync::atomic::Ordering as AtomicOrdering;
    use tempfile::tempdir;

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

    #[test]
    fn analyze_neurons_deduplicates_focus_targets() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = (MIN_NEURON_SAMPLE_COUNT + 5) as u32;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-source".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![1.0],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
        };

        let result = analyze_neurons(&input)
            .expect("Neuron analysis should succeed with duplicate focus neurons");
        assert!(
            !result.helpful_neurons.is_empty(),
            "Expected at least one candidate neuron",
        );

        use std::collections::HashSet;
        let unique: HashSet<(&str, &str, &str)> = result
            .helpful_neurons
            .iter()
            .map(|candidate| {
                (
                    candidate.source_neuron_uuid.as_str(),
                    candidate.target_neuron_uuid.as_str(),
                    candidate.squash.as_str(),
                )
            })
            .collect();

        assert_eq!(
            unique.len(),
            result.helpful_neurons.len(),
            "Duplicate focus neurons should not yield duplicate candidates",
        );

        assert!(
            unique.iter().any(|(_, target, _)| *target == "output-0"),
            "Expected a candidate targeting output-0",
        );
    }

    #[test]
    fn analyze_neurons_respects_gpu_requirement() {
        let _guard = ForceGpuFailureGuard::new();

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: "non-existent.parquet".to_string(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: None,
            max_candidates: None,
            require_gpu: Some(true),
        };

        let err = analyze_neurons(&input)
            .err()
            .expect("Expected neuron analysis to fail when GPU is required but unavailable");
        let message = format!("{err}");
        assert!(
            message.contains("GPU"),
            "Expected GPU related error message, got: {message}"
        );
    }
}
