//! Candidate-related FFI types.
//!
//! Structs representing mutation candidates (synapse, neuron, coordinated)
//! returned from analysis to the FFI boundary.

use serde::Serialize;

use crate::analysis;

use super::NeuronStatsJson;

/// Candidate to add a new synapse.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSynapseJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
    /// Index of `from_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    /// Provided for debugging and ranking analysis (eg why candidates cluster near the end).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_neuron_index: Option<usize>,
    /// Index of `to_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_neuron_index: Option<usize>,
    pub weight: f32,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    /// Output neurons have impact = 1.0, hidden neurons have discounted impact.
    pub target_neuron_impact: f32,
    /// Expected reduction in creature's error from adding this synapse.
    /// Formula: `neuron_error_reduction` × `target_neuron_impact`
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature's score from adding this synapse.
    /// Since score = 1 - error, this equals `expected_creature_error_reduction`.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    /// Magnitude-weighted improvement ratio in `[0, 1]` (Issue #1161).
    ///
    /// Whereas `improved_count / total_count` only counts whether a sample
    /// improved (binary), this ratio measures *by how much* each sample
    /// improved relative to its baseline error magnitude. A population of
    /// noise-level reductions yields a near-zero ratio, while a population
    /// of substantial reductions approaches 1.0.
    ///
    /// Internal-only: skipped from the FFI/JSON wire schema. `None` means the
    /// metric was not computed for this candidate (legacy paths).
    #[serde(skip)]
    pub improvement_magnitude_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
    /// Information about how this candidate affects outlier samples (Issue #192).
    ///
    /// Only populated when outlier analysis is enabled via
    /// `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS=1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outlier_reduction_info: Option<analysis::scoring::error_distribution::OutlierReductionInfo>,
    /// Overall confidence score for this prediction (Issue #194).
    ///
    /// A value between 0.0 and 1.0 indicating how reliable the prediction is.
    /// Higher values mean more reliable predictions. Computed from:
    /// - Sample size (more samples = higher confidence)
    /// - Source variance (higher variance = more reliable correlation)
    /// - Model fit (better fit = higher confidence)
    pub prediction_confidence: f32,
    /// 95% confidence interval for expectedCreatureScoreGain (Issue #194).
    ///
    /// The first element is the lower bound, the second is the upper bound.
    /// The point estimate (expectedCreatureScoreGain) should fall within this interval.
    pub expected_score_gain_confidence_interval: [f32; 2],
    /// Human-readable label identifying weight variants (Issue #513).
    ///
    /// When a synapse candidate is paired with conservative/gentle-nudge/micro-nudge
    /// variants, each variant gets a descriptive comment. The original candidate's
    /// comment lists which variants were included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// Candidate to update the weight of an existing synapse (delta-based).
///
/// This represents a *weight adjustment* (not a new connection). The prediction logic treats
/// `delta_weight` as an additive correction to the existing synapse weight.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SynapseWeightUpdateCandidateJson {
    pub from_neuron_uuid: String,
    pub to_neuron_uuid: String,
    /// Index of `from_neuron_uuid` in the creature's forward-only evaluation order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_neuron_index: Option<usize>,
    /// Index of `to_neuron_uuid` in the creature's forward-only evaluation order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_neuron_index: Option<usize>,
    pub old_weight: f32,
    pub new_weight: f32,
    /// The proposed additive change: `new_weight - old_weight`.
    pub delta_weight: f32,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    pub target_neuron_impact: f32,
    /// Expected reduction in creature error from applying `delta_weight`.
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature score from applying `delta_weight`.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
}

/// A single atomic operation inside a coordinated (grouped) candidate.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CoordinatedStructuralOpJson {
    RemoveSynapse {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
    },
    AddSynapse {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
        weight: f32,
    },
    /// Add a neuron as part of a coordinated structural candidate.
    ///
    /// Notes (7-Jan-2026):
    /// - `neuronUuid` is emitted by Rust and must be deterministic so coordinated candidates are replayable.
    /// - For forward-only creatures, `insertBeforeNeuronUuid` provides a placement hint so subsequent
    ///   `addSynapse(newNeuron -> target)` can satisfy the forward-only ordering constraints.
    AddNeuron {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        #[serde(rename = "neuronType")]
        neuron_type: String,
        squash: String,
        bias: f32,
        #[serde(
            rename = "insertBeforeNeuronUuid",
            skip_serializing_if = "Option::is_none"
        )]
        insert_before_neuron_uuid: Option<String>,
    },
    /// Remove a neuron and any attached synapses.
    RemoveNeuron {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
    },
    /// Change a neuron's squash/activation function.
    ChangeSquash {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        squash: String,
    },
    /// Set a neuron's bias.
    SetBias {
        #[serde(rename = "neuronUuid")]
        neuron_uuid: String,
        bias: f32,
    },
    /// Set an existing synapse's weight (Issue #180).
    ///
    /// This replaces the previous `removeSynapse` + `addSynapse` pattern for weight adjustments,
    /// providing a simpler and more direct representation of the intended change.
    SetWeight {
        #[serde(rename = "fromNeuronUuid")]
        from_neuron_uuid: String,
        #[serde(rename = "toNeuronUuid")]
        to_neuron_uuid: String,
        weight: f32,
    },
}

/// A grouped candidate that must be applied as a single unit.
///
/// This supports "Coordinated Structural Discovery" (Issue #165): beneficial changes that are
/// epistatic (no single edit improves fitness in isolation).
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CoordinatedStructuralCandidateJson {
    pub operations: Vec<CoordinatedStructuralOpJson>,
    pub expected_creature_score_gain: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandidateNeuronJson {
    pub source_neuron_uuid: String,
    pub target_neuron_uuid: String,
    /// Index of `source_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_index: Option<usize>,
    /// Index of `target_neuron_uuid` in the creature's forward-only evaluation order.
    ///
    /// This includes input neurons (`input-0..`) followed by `creature.neurons[]` in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_index: Option<usize>,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub squash: String,
    pub bias: f32,
    /// Optional human-readable comment for diagnostics and production experiments.
    ///
    /// This is intentionally optional to maintain backwards compatibility with older
    /// consumers that don't expect the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Impact of the target neuron on the creature's output (0.0 to 1.0).
    /// Output neurons have impact = 1.0, hidden neurons have discounted impact.
    pub target_neuron_impact: f32,
    /// Expected reduction in creature's error from adding this neuron.
    /// Formula: `neuron_error_reduction` × `target_neuron_impact`
    pub expected_creature_error_reduction: f32,
    /// Expected improvement in creature's score from adding this neuron.
    /// Since score = 1 - error, this equals `expected_creature_error_reduction`.
    pub expected_creature_score_gain: f32,
    pub improved_count: u32,
    pub total_count: u32,
    /// Magnitude-weighted improvement ratio in `[0, 1]` (Issue #1161).
    ///
    /// See [`CandidateSynapseJson::improvement_magnitude_ratio`] for the full
    /// description. Internal-only: skipped from the FFI/JSON wire schema.
    #[serde(skip)]
    pub improvement_magnitude_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_neuron_stats: Option<NeuronStatsJson>,
    /// Overall confidence score for this prediction (Issue #194).
    ///
    /// A value between 0.0 and 1.0 indicating how reliable the prediction is.
    /// Higher values mean more reliable predictions. Computed from:
    /// - Sample size (more samples = higher confidence)
    /// - Source variance (higher variance = more reliable correlation)
    /// - Model fit (better fit = higher confidence)
    pub prediction_confidence: f32,
    /// 95% confidence interval for expectedCreatureScoreGain (Issue #194).
    ///
    /// The first element is the lower bound, the second is the upper bound.
    /// The point estimate (expectedCreatureScoreGain) should fall within this interval.
    pub expected_score_gain_confidence_interval: [f32; 2],
    /// Target saturation factor (0.0–1.0) for downstream scoring (Issue #1111).
    ///
    /// Indicates how close the target neuron's activation range is to spanning
    /// its full output range. 0.0 = not saturated; 1.0 = fully saturated.
    /// Candidates targeting near-saturated neurons have reduced expected gains.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_saturation_factor: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankedNeuronJson {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = `structural_impact` × `mean_activation`
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
}

/// A neuron with activation-weighted impact below costOfGrowth threshold - candidate for removal.
/// Removing such neurons improves score because complexity reduction outweighs contribution.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovalCandidateJson {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = `structural_impact` × `mean_activation`
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
    /// Number of synapses pointing TO this neuron
    pub incoming_synapses: usize,
    /// Number of synapses pointing FROM this neuron
    pub outgoing_synapses: usize,
    /// The complexity savings from removing this neuron (based on NEAT-AI Score.ts formula)
    pub removal_savings: f32,
    /// Expected creature-level error reduction from removing this neuron.
    /// Issue #117: This is based on `activation_weighted_impact`, NOT `total_error`.
    /// For low-impact removal candidates, this will be very small (as it should be).
    pub expected_error_reduction: f32,
    /// Explains why removal improves score
    pub reason: String,
}
