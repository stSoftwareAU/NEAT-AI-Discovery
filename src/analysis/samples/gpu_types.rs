//! GPU-compatible data formats for the analysis pipeline.
//!
//! All types in this module use `#[repr(C)]` and implement `bytemuck::Pod`
//! for direct GPU buffer compatibility.

use bytemuck::{Pod, Zeroable};

use super::HelpfulSample;

/// GPU buffer format for helpful samples.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuHelpfulSample {
    pub activation: f32,
    pub avg_error: f32,
}

impl From<HelpfulSample> for GpuHelpfulSample {
    fn from(value: HelpfulSample) -> Self {
        Self {
            activation: value.activation,
            avg_error: value.avg_error,
        }
    }
}

/// GPU contribution data for helpful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HelpfulContribution {
    pub positive_flag: u32,
    pub negative_flag: u32,
    pub positive_improvement: f32,
    pub negative_improvement: f32,
    pub positive_activation: f32,
    pub negative_activation: f32,
    pub error_squared: f32,
    pub activation_squared: f32,
    pub error_activation: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

/// GPU shader uniforms for helpful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HelpfulUniforms {
    pub length: u32,
    pub pad0: u32,
    pub epsilon: f32,
    pub pad1: f32,
}

/// GPU contribution data for harmful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HarmfulContribution {
    pub harmful_flag: u32,
    pub helpful_flag: u32,
    pub error_magnitude: f32,
    pub pad0: f32,
}

/// GPU shader uniforms for harmful synapse evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HarmfulUniforms {
    pub length: u32,
    pub pad0: u32,
    pub epsilon: f32,
    pub weight: f32,
}

/// GPU contribution data for `ReLU` activation evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReluContribution {
    pub positive_activation_sq: f32,
    pub positive_error_activation: f32,
    pub positive_count: u32,
    pub negative_activation_sq: f32,
    pub negative_error_activation: f32,
    pub negative_count: u32,
    pub error_sq: f32,
    pub pad0: f32,
    pub pad1: u32,
    pub pad2: u32,
}

/// GPU shader uniforms for `ReLU` activation evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReluUniforms {
    pub length: u32,
    pub threshold: f32,
    pub epsilon: f32,
    pub pad0: f32,
}

/// GPU result data for bias optimisation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BiasResult {
    pub bias_value: f32,
    pub error_reduction: f32,
    pub valid_sample_count: u32,
    pub pad0: u32,
}

impl BiasResult {
    /// Create a zeroed result.
    pub fn zeroed() -> Self {
        Self {
            bias_value: 0.0,
            error_reduction: 0.0,
            valid_sample_count: 0,
            pad0: 0,
        }
    }
}

/// GPU shader uniforms for bias optimisation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BiasUniforms {
    pub sample_count: u32,
    pub bias_count: u32,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub activation_type: u32,
    pub epsilon: f32,
    pub min_sample_count: u32,
    pub pad0: u32,
}

/// GPU output data for activation function evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ActivationOutput {
    pub output: f32,
    pub output_sq: f32,
    pub error_output: f32,
    pub valid: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// GPU shader uniforms for activation function evaluation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ActivationUniforms {
    pub sample_count: u32,
    pub orientation: f32,
    pub scale: f32,
    pub activation_type: u32,
    pub epsilon: f32,
    pub pad0: f32,
    pub pad1: f32,
}

/// GPU shader uniforms for workgroup reduction (Issue #218).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReductionUniforms {
    /// Total number of contributions to reduce
    pub contribution_count: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}
