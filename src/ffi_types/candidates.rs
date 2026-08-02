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
    /// Stable variant identifier (Issue #1163).
    ///
    /// When a synapse candidate is produced by `variant_generation`, this
    /// field carries the canonical variant key (e.g., `gentle-nudge`,
    /// `micro-nudge`, `conservative`) — the canonical identifier the
    /// failure-cache calibrator groups by. The `comment` string remains
    /// human-readable; `variant_key` is the machine-readable label.
    /// `None` for original (non-variant) candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_key: Option<String>,
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

/// Variance-aware weight-redistribution compensation attached to a bare
/// remove-neuron candidate (Issue #1689, wiring Issue #1559).
///
/// When a candidate's sole operation removes a **variance-carrying** neuron, the
/// live dispatch path evaluates counterfactual (d) — folding the removed
/// neuron's per-sample downstream signal into a correlated survivor's weight —
/// and attaches the result here so the applier redistributes the survivor's
/// weight instead of applying a mean-only bias fold. The compact covariance
/// sufficient statistic travels alongside the remedy so the applier can reason
/// about the removal without re-reading per-sample activations.
///
/// This is absent (`None`) for candidates with no variance-carrying survivor and
/// for constant-neuron candidates, which route to the bias-fold path
/// (Issue #1623) rather than being given a redistribution here.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RemoveNeuronCompensationJson {
    /// Downstream neuron that both the removed candidate and the chosen survivor
    /// feed — the shared target whose inbound weight absorbs the signal.
    pub target_neuron_uuid: String,
    /// Surviving neuron whose weight into `target_neuron_uuid` is bumped to
    /// absorb the removed neuron's per-sample signal.
    pub survivor_neuron_uuid: String,
    /// Optimal least-squares weight bump to add to the survivor's weight into the
    /// shared target: `Δw = w_c · cov / var(a_s)`.
    pub delta_weight: f32,
    /// Number of aligned per-sample activation pairs behind the statistic.
    pub sample_count: u64,
    /// Population variance of the removed neuron's per-sample activation.
    pub candidate_variance: f32,
    /// Population variance of the survivor's per-sample activation.
    pub survivor_variance: f32,
    /// Population covariance of the candidate and survivor activations.
    pub covariance: f32,
    /// Candidate/survivor activation correlation in `[-1, 1]`.
    pub correlation: f32,
    /// Per-sample residual variance under NEAT-AI's mean-only bias fold — the
    /// cost that survives the current lever.
    pub bias_only_residual_variance: f32,
    /// Per-sample residual variance after weight redistribution (d).
    pub redistributed_residual_variance: f32,
    /// Variance recovered by redistribution over the bias-only fold (`≥ 0`).
    pub variance_recovered: f32,
    /// `true` when redistribution drives the residual variance to ~0 — the
    /// removal becomes non-regressive, which the mean-only bias lever can never
    /// achieve.
    pub fully_compensable: bool,
}

/// A single downstream target that absorbs a folded constant-neuron bias delta
/// (Issue #1690, emitting the #1623 fold).
///
/// When a functionally-constant neuron is removed, its fixed contribution
/// `outgoing_weight × constant_activation` to each downstream target is folded
/// into that target's bias so the removal is behaviour-preserving on the recorded
/// window.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FoldedBiasDeltaJson {
    /// Downstream neuron whose bias absorbs the constant contribution.
    pub target_neuron_uuid: String,
    /// Bias delta to add to the target: `outgoing_weight × constant_activation`.
    pub bias_delta: f32,
}

/// Constant-neuron bias-fold compensation attached to a bare remove-neuron
/// candidate (Issue #1690, wiring Issue #1623).
///
/// When a candidate's sole operation removes a **functionally-constant** neuron —
/// one whose recorded activations are constant within the evaluate-before-accept
/// gate tolerance — the live dispatch path evaluates the #1623 bias fold and
/// attaches the folded per-target bias deltas here so the applier folds the
/// constant contribution into downstream biases rather than applying a mean-only
/// fold. A constant neuron carries no per-sample variance, so a plain bias fold
/// is fully compensable and no survivor redistribution is needed.
///
/// This is absent (`None`) for variance-carrying candidates, which route to the
/// #1559 weight-redistribution remedy ([`RemoveNeuronCompensationJson`]) instead,
/// and for candidates the gate rejects (records vary over tolerance or none are
/// recorded — rejected fail-loud, never folded blind).
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConstantNeuronBiasFoldJson {
    /// The mean activation used as the folded constant `c`.
    pub constant_activation: f32,
    /// The population variance of the neuron's activation over the recorded
    /// window — (near-)zero for a genuinely constant neuron.
    pub activation_variance: f32,
    /// The maximum per-sample residual `|w·(a_i − c)|` across every outgoing
    /// connection and observation, at or below the gate tolerance on acceptance.
    pub max_residual: f32,
    /// Per-target bias deltas the applier adds before deleting the neuron.
    pub folded_targets: Vec<FoldedBiasDeltaJson>,
}

/// A grouped candidate that must be applied as a single unit.
///
/// This supports "Coordinated Structural Discovery" (Issue #165): beneficial changes that are
/// epistatic (no single edit improves fitness in isolation).
#[derive(Debug, Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CoordinatedStructuralCandidateJson {
    pub operations: Vec<CoordinatedStructuralOpJson>,
    pub expected_creature_score_gain: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Variance-aware compensation for a bare remove-neuron candidate
    /// (Issue #1689). Populated by the live dispatch path for sole-op
    /// `RemoveNeuron` candidates whose removed neuron carries per-sample variance
    /// and has a correlated downstream survivor; `None` otherwise (including for
    /// constant-neuron candidates, which route to the #1623 bias fold).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remove_neuron_compensation: Option<RemoveNeuronCompensationJson>,
    /// Constant-neuron bias-fold compensation for a bare remove-neuron candidate
    /// (Issue #1690). Populated by the live dispatch path for sole-op
    /// `RemoveNeuron` candidates whose removed neuron is functionally constant
    /// (recorded activations constant within the gate tolerance); `None`
    /// otherwise (including for variance-carrying candidates, which route to the
    /// #1559 weight-redistribution remedy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constant_neuron_bias_fold: Option<ConstantNeuronBiasFoldJson>,
}

/// JSON-serialisable `addNeuron` candidate: a proposed hidden neuron bridging a
/// source and target, with its weights, activation, and expected-gain metrics.
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
    /// Stable variant identifier (Issue #1163).
    ///
    /// When an add-neuron candidate is produced by `variant_generation`, this
    /// field carries the canonical variant key (e.g., `gentle-nudge`,
    /// `micro-nudge`, `conservative`, `feather-touch`, `whisper`) — the
    /// canonical identifier the failure-cache calibrator groups by. The
    /// `comment` string remains human-readable; `variant_key` is the
    /// machine-readable label. `None` for original (non-variant) candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_key: Option<String>,
}

/// JSON-serialisable view of a ranked focus neuron returned to NEAT-AI: its
/// error, impact, activation metrics, and combined ranking score.
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
    /// Issue #1445: Combined impact-weighted ranking score (error × impact^γ ×
    /// gradient × frequency × history) — the same value the focus list is
    /// ordered by and the roulette weight used for diversity-aware focus
    /// selection. Surfaced so callers can reuse the Rust weight directly rather
    /// than recomputing a (squared) weight that re-concentrates onto one neuron.
    pub weighted_score: f32,
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
    /// Mean absolute activation value from recorded samples.
    ///
    /// Issue #1923: measured on **both** producing paths. When the discovery
    /// records cannot be read — no parquet, an I/O error, or a neuron with no
    /// recorded rows — this stays `0.0` and [`Self::reason`] says so explicitly.
    /// A `0.0` here is therefore only a measurement when the reason string
    /// reports the gate as resolved; consumers must not treat an unmeasured
    /// zero as "measured and inactive".
    pub mean_activation: f32,
    /// Activation-weighted impact = `structural_impact` × `mean_activation`
    /// This reflects the actual contribution the neuron makes during inference,
    /// and is the value candidates are ranked by (Issue #1923).
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
