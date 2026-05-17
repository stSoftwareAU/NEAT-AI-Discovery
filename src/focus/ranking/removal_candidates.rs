//! Removal candidate identification and constant neuron removal.
//!
//! Contains `RemovalCandidate`, `SynapseCounts`, `calculate_removal_savings`,
//! and detection of constant-value neurons for coordinated structural removal.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::record_providers::get_records_or_error;
use super::score_calculation::activation_mean_and_variance_from_records;
use crate::analysis::constants::{
    REMOVAL_CANDIDATE_BOOST, REMOVAL_MEAN_ACTIVATION_THRESHOLD, remove_low_impact_noise_floor,
};
use crate::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
};
use rayon::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

use super::record_providers::RecordProvider;
use super::score_calculation::RankedNeuron;

/// A neuron with activation-weighted impact below removal savings threshold - candidate for removal.
/// Removing such neurons improves score because complexity reduction outweighs contribution.
#[derive(Debug)]
pub struct RemovalCandidate {
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
    /// This is based on `activation_weighted_impact`, NOT the neuron's error.
    ///
    /// Issue #117: Previously, `total_error` was incorrectly used as expected error reduction,
    /// leading to predictions like 27% when actual reduction was ~0%.
    pub expected_error_reduction: f32,
    pub reason: String,
}

/// Calculate the complexity savings from removing a neuron.
///
/// Based on NEAT-AI's Score.ts formula:
/// ```typescript
/// const complexityPenalty = hiddenNeuronCount * growthCost +
///     creature.synapses.length * growthCost / 10 + penalty * growthCost / 100;
/// ```
///
/// So removing a neuron with N incoming and M outgoing synapses saves:
/// - `growth_cost` for the neuron itself
/// - `(N + M) × growth_cost / 10` for the synapses
///
/// Total: `growth_cost × (1 + (N + M) / 10)`
///
/// # Arguments
/// * `incoming_synapses` - Number of synapses pointing TO this neuron
/// * `outgoing_synapses` - Number of synapses pointing FROM this neuron
/// * `growth_cost` - The cost per hidden neuron (typically 1e-7)
///
/// # Returns
/// The total complexity savings from removing this neuron and its synapses
pub fn calculate_removal_savings(
    incoming_synapses: usize,
    outgoing_synapses: usize,
    growth_cost: f32,
) -> f32 {
    let total_synapses = incoming_synapses + outgoing_synapses;
    growth_cost * (1.0 + total_synapses as f32 / 10.0)
}

/// Pre-computed synapse counts for efficient O(1) lookup.
///
/// Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
///
/// Previously, `count_synapses_for_neuron` performed a linear O(m) scan through ALL synapses
/// twice (incoming + outgoing) for each neuron being ranked. With n neurons and m synapses,
/// this was O(n × m) complexity.
///
/// This struct pre-builds two `HashMaps` during initialisation in O(m) time, then provides
/// O(1) lookup for any neuron's synapse counts. Total complexity is O(n + m).
///
/// # Example performance improvement
///
/// For a creature with 500 neurons and 10,000 synapses:
/// - Previous: 500 × 10,000 × 2 = **10 million** iterations
/// - With `SynapseCounts`: 10,000 + 500 = **10,500** iterations
/// - **~1000x improvement**
///
/// Issue #983: Uses borrowed `&str` keys to avoid cloning every synapse UUID
/// during construction.
#[derive(Debug)]
pub struct SynapseCounts<'a> {
    /// Map from neuron UUID to count of synapses pointing TO that neuron
    incoming: HashMap<&'a str, usize>,
    /// Map from neuron UUID to count of synapses pointing FROM that neuron
    outgoing: HashMap<&'a str, usize>,
}

impl<'a> SynapseCounts<'a> {
    /// Create a new `SynapseCounts` by scanning all synapses once.
    ///
    /// Time complexity: O(m) where m is the number of synapses.
    /// Space complexity: O(n) where n is the number of unique neurons with synapses.
    pub fn new(creature: &'a CreatureJson) -> Self {
        let mut incoming: HashMap<&'a str, usize> = HashMap::new();
        let mut outgoing: HashMap<&'a str, usize> = HashMap::new();

        for synapse in &creature.synapses {
            *incoming.entry(synapse.to_uuid.as_str()).or_default() += 1;
            *outgoing.entry(synapse.from_uuid.as_str()).or_default() += 1;
        }

        Self { incoming, outgoing }
    }

    /// Get the synapse counts for a neuron in O(1) time.
    ///
    /// # Arguments
    /// * `neuron_uuid` - The UUID of the neuron to look up
    ///
    /// # Returns
    /// A tuple of (`incoming_count`, `outgoing_count`). Returns (0, 0) if the neuron
    /// has no synapses or doesn't exist in the creature.
    pub fn get(&self, neuron_uuid: &str) -> (usize, usize) {
        (
            self.incoming.get(neuron_uuid).copied().unwrap_or(0),
            self.outgoing.get(neuron_uuid).copied().unwrap_or(0),
        )
    }
}

/// Outcome of [`identify_removal_candidates`] — the surviving removal
/// candidates plus rejection counts for diagnostic surfacing (Issue #1142).
#[derive(Debug, Default)]
pub(super) struct RemovalCandidateOutcome {
    /// Removal candidates that passed every filter.
    pub candidates: Vec<RemovalCandidate>,
    /// Number of candidates dropped because `boosted_savings - impact` fell
    /// below [`remove_low_impact_noise_floor`] (Issue #1142).
    pub noise_floor_rejections: u32,
}

/// Identify removal candidates from ranked neurons.
///
/// Issue #235: Return ALL neurons where removal improves the creature's score.
/// A removal improves score when: `removal_savings` > `activation_weighted_impact`
///
/// Issue #892: Apply stricter filtering based on GRQ-sampler cache evidence.
/// Successful removals (21.5% success rate) have low mean activation (≤ 0.04)
/// and low structural impact (≤ 6e-5). Candidates passing these thresholds
/// receive a scoring boost to prioritise them over other candidate types.
///
/// Issue #1142: Apply a `REMOVE_LOW_IMPACT_NOISE_FLOOR` gate on the
/// post-boost `net_improvement = boosted_savings - activation_weighted_impact`.
/// The `REMOVAL_CANDIDATE_BOOST` multiplier is applied to raw savings **before**
/// the savings-vs-impact comparison, so boost-inflated noise (net improvements
/// in the 1e-8 range) would otherwise survive the pipeline. The dropped-count
/// is surfaced under [`REJECTION_REMOVAL_BELOW_NOISE_FLOOR`] in
/// `metadata.rejection_breakdown`.
pub(super) fn identify_removal_candidates(
    neurons: &[RankedNeuron],
    synapse_counts: &SynapseCounts,
    cost_of_growth_threshold: f32,
) -> RemovalCandidateOutcome {
    let noise_floor = remove_low_impact_noise_floor();

    // Each neuron maps to `Option<Result<RemovalCandidate, NoiseFloorReject>>`
    // so we can collect both surviving candidates and rejection counts in a
    // single parallel pass.
    #[derive(Debug)]
    enum Emit {
        Candidate(Box<RemovalCandidate>),
        NoiseFloorReject,
    }

    let emitted: Vec<Emit> = neurons
        .par_iter()
        .filter_map(|n| {
            // Issue #208: Use pre-computed synapse counts for O(1) lookup
            let (incoming, outgoing) = synapse_counts.get(&n.neuron_uuid);
            let savings = calculate_removal_savings(incoming, outgoing, cost_of_growth_threshold);

            // Issue #892: Apply boost to savings BEFORE the savings-vs-impact check.
            // The 21.5% success rate justifies prioritising these candidates, but
            // see Issue #1142 — we must re-check against a noise floor below.
            let boosted_savings = savings * REMOVAL_CANDIDATE_BOOST;

            // Issue #235: Filter on boosted savings > impact (removal improves score)
            // instead of impact < threshold (may miss valid candidates).
            if boosted_savings <= n.activation_weighted_impact {
                return None;
            }

            // Issue #892: Filter out neurons with high mean activation when they
            // also have non-negligible structural impact. Cache evidence shows
            // failed removals have high mean activation (up to 57.8) combined with
            // meaningful impact — these neurons are actually contributing.
            // Disconnected neurons (impact ≈ 0) are always safe to remove regardless
            // of activation level.
            let has_meaningful_impact = n.impact > f32::EPSILON;
            if has_meaningful_impact
                && n.mean_activation > REMOVAL_MEAN_ACTIVATION_THRESHOLD
            {
                return None;
            }

            // Net score improvement = boosted_savings - impact
            let net_improvement = boosted_savings - n.activation_weighted_impact;

            // Issue #1142: Gate on a noise floor to drop boost-inflated candidates
            // whose net improvement is indistinguishable from numerical noise
            // (e.g. 6.64e-8 in GRQ-sampler commit 744ac60d).
            if net_improvement < noise_floor {
                return Some(Emit::NoiseFloorReject);
            }

            // Issue #117: expected_error_reduction should be based on activation_weighted_impact,
            // NOT total_error.
            let expected_error_reduction = n.activation_weighted_impact;

            Some(Emit::Candidate(Box::new(RemovalCandidate {
                neuron_uuid: n.neuron_uuid.clone(),
                total_error: n.total_error,
                impact: n.impact,
                mean_activation: n.mean_activation,
                activation_weighted_impact: n.activation_weighted_impact,
                incoming_synapses: incoming,
                outgoing_synapses: outgoing,
                removal_savings: boosted_savings,
                expected_error_reduction,
                reason: format!(
                    "Removal improves score: saves {:.2e} (boosted {:.1}×) > impact {:.2e} (net +{:.2e}), {} synapses, costOfGrowth={:.2e}",
                    savings,
                    REMOVAL_CANDIDATE_BOOST,
                    n.activation_weighted_impact,
                    net_improvement,
                    incoming + outgoing,
                    cost_of_growth_threshold,
                ),
            })))
        })
        .collect();

    let mut removal_candidates: Vec<RemovalCandidate> = Vec::with_capacity(emitted.len());
    let mut noise_floor_rejections: u32 = 0;
    for emit in emitted {
        match emit {
            Emit::Candidate(c) => removal_candidates.push(*c),
            Emit::NoiseFloorReject => {
                noise_floor_rejections = noise_floor_rejections.saturating_add(1);
            }
        }
    }

    // Issue #235: Sort by net improvement (removal_savings - activation_weighted_impact).
    // Higher net improvement = better candidate (removing it saves more than its contribution).
    removal_candidates.sort_by(|a, b| {
        let a_net = a.removal_savings - a.activation_weighted_impact;
        let b_net = b.removal_savings - b.activation_weighted_impact;

        b_net
            .total_cmp(&a_net)
            .then_with(|| {
                a.activation_weighted_impact
                    .total_cmp(&b.activation_weighted_impact)
            })
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    RemovalCandidateOutcome {
        candidates: removal_candidates,
        noise_floor_rejections,
    }
}

/// Threshold for considering a neuron as "constant" (near-zero variance).
/// Issue #217/306: Neurons with variance below this threshold are treated as constant
/// and can be removed with bias adjustments for downstream neurons.
///
/// Value 1e-10 is from Issue #217's proposal for `DEAD_VARIANCE_THRESHOLD`.
const CONSTANT_VARIANCE_THRESHOLD: f32 = 1e-10;

/// Detect constant-value neurons and create coordinated structural candidates
/// that remove the neuron and adjust downstream biases.
///
/// A neuron with near-zero activation variance is "constant" - it always outputs roughly
/// the same value regardless of input. Removing it is equivalent to adjusting the biases
/// of downstream neurons by: `bias_adjustment` = `synapse_weight` × `mean_activation`
pub(super) fn detect_constant_neuron_removals(
    selectable: &[&NeuronJson],
    records_provider: &Arc<dyn RecordProvider>,
    synapse_counts: &SynapseCounts,
    creature: &CreatureJson,
    cost_of_growth_threshold: f32,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let neuron_types: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();

    let neuron_biases: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();

    // Build outgoing synapse map: from_uuid -> [(to_uuid, weight)]
    let outgoing_synapses: HashMap<&str, Vec<(&str, f32)>> =
        creature
            .synapses
            .iter()
            .fold(HashMap::new(), |mut map, syn| {
                map.entry(syn.from_uuid.as_str())
                    .or_default()
                    .push((syn.to_uuid.as_str(), syn.weight));
                map
            });

    // Find constant hidden neurons and create coordinated removal candidates
    selectable
        .par_iter()
        .filter_map(|neuron| {
            // Only consider hidden neurons for constant removal
            let neuron_type = neuron_types.get(neuron.uuid.as_str())?;
            if *neuron_type != "hidden" {
                return None;
            }

            // Get records and compute variance
            let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid).ok()?;
            let (mean_activation, variance) = activation_mean_and_variance_from_records(&records);

            // Check if variance is below threshold (constant neuron)
            if variance > CONSTANT_VARIANCE_THRESHOLD {
                return None;
            }

            // Get outgoing synapses for this neuron
            let outgoing = outgoing_synapses.get(neuron.uuid.as_str())?;
            if outgoing.is_empty() {
                return None;
            }

            // Build coordinated structural candidate:
            // 1. SetBias operations for all downstream neurons
            // 2. RemoveNeuron operation
            let mut operations = Vec::with_capacity(outgoing.len() + 1);

            // Add SetBias operations for all downstream neurons
            for (to_uuid, weight) in outgoing {
                let old_bias = neuron_biases.get(to_uuid).copied().unwrap_or(0.0);
                let bias_adjustment = weight * mean_activation;
                let new_bias = old_bias + bias_adjustment;

                if new_bias.is_finite() {
                    operations.push(CoordinatedStructuralOpJson::SetBias {
                        neuron_uuid: to_uuid.to_string(),
                        bias: new_bias,
                    });
                }
            }

            // Add RemoveNeuron operation (must be last so bias adjustments happen first)
            operations.push(CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: neuron.uuid.clone(),
            });

            // Calculate expected improvement: removal savings (complexity reduction)
            // Issue #208: Use pre-computed synapse counts for O(1) lookup
            let (incoming_count, outgoing_count) = synapse_counts.get(&neuron.uuid);
            let removal_savings =
                calculate_removal_savings(incoming_count, outgoing_count, cost_of_growth_threshold);

            Some(CoordinatedStructuralCandidateJson {
                operations,
                expected_creature_score_gain: removal_savings,
                comment: Some(format!(
                    "Issue #306: Constant neuron removal with bias adjustments. \
                     mean_activation={mean_activation:.6}, variance={variance:.2e}, \
                     {} downstream neurons, savings={removal_savings:.2e}",
                    outgoing.len()
                )),
            })
        })
        .collect()
}

// =============================================================================
// Unit tests for Issue #1142 noise-floor gating
// =============================================================================

#[cfg(test)]
mod noise_floor_tests {
    //! Issue #1142: Remove-low-impact candidates with net improvement below
    //! `REMOVE_LOW_IMPACT_NOISE_FLOOR` must be dropped before emission and the
    //! drop count must be surfaced under `REJECTION_REMOVAL_BELOW_NOISE_FLOOR`.

    use super::*;
    use crate::focus::gradient::GradientFlowStats;
    use crate::{CreatureJson, NeuronJson, SynapseJson};
    use std::sync::{Mutex, OnceLock};

    /// Module-level lock protecting tests that read or write
    /// `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR`. Acquire this before
    /// any call to `identify_removal_candidates` in tests that depend on the
    /// env-var being at its default, or before mutating it.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    /// Craft a [`RankedNeuron`] with the exact impact/activation values needed
    /// to produce a targeted `activation_weighted_impact` without touching
    /// the production record-derived code paths.
    fn ranked_neuron(uuid: &str, impact: f32, activation_weighted_impact: f32) -> RankedNeuron {
        // mean_activation is derived so that structural impact × mean_activation
        // equals the desired activation_weighted_impact. Tests use impact ≈ 0
        // (disconnected neurons) so the `mean_activation` filter never fires.
        let mean_activation = if impact > 0.0 {
            activation_weighted_impact / impact
        } else {
            0.0
        };
        RankedNeuron {
            neuron_uuid: uuid.to_string(),
            total_error: 0.0,
            raw_error: 0.0,
            impact,
            mean_activation,
            activation_weighted_impact,
            gradient_flow: GradientFlowStats::default(),
            activation_frequency: 0.5,
        }
    }

    /// Build a minimal `CreatureJson` with the given (incoming, outgoing)
    /// synapse counts for the neuron named `uuid`.
    fn creature_with_synapse_counts(uuid: &str, incoming: usize, outgoing: usize) -> CreatureJson {
        let mut neurons: Vec<NeuronJson> = vec![NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }];
        let mut synapses: Vec<SynapseJson> = Vec::new();

        for i in 0..incoming {
            let source = format!("in-{i}");
            neurons.push(NeuronJson {
                uuid: source.clone(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
            synapses.push(SynapseJson {
                from_uuid: source,
                to_uuid: uuid.to_string(),
                weight: 0.1,
                synapse_type: None,
            });
        }
        for i in 0..outgoing {
            let target = format!("out-{i}");
            neurons.push(NeuronJson {
                uuid: target.clone(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
            synapses.push(SynapseJson {
                from_uuid: uuid.to_string(),
                to_uuid: target,
                weight: 0.1,
                synapse_type: None,
            });
        }

        let input = neurons.iter().filter(|n| n.neuron_type == "input").count();
        let output = neurons.iter().filter(|n| n.neuron_type == "output").count();
        CreatureJson {
            neurons,
            synapses,
            input,
            output,
        }
    }

    /// Scenario from Issue #1142 evidence (`v2_remove-low-impact_0ce92a87…`):
    /// raw `savings = 1.20e-7`, boosted to `1.8e-7` by the 1.5× multiplier,
    /// `impact = 1.14e-7`, giving net `+6.6e-8` — indistinguishable from
    /// floating-point noise and therefore dropped.
    ///
    /// With cost-of-growth `1e-7` and 2 synapses (1 incoming, 1 outgoing):
    ///   savings  = 1e-7 × (1 + 2/10)         = 1.2e-7   ✓ matches evidence
    ///   boosted  = 1.2e-7 × `REMOVAL_CANDIDATE_BOOST`(1.5) = 1.8e-7
    ///   net      = 1.8e-7 − 1.14e-7           ≈ 6.6e-8
    #[test]
    fn issue_1142_evidence_candidate_is_dropped() {
        let _guard = env_lock();

        let growth = 1e-7_f32;
        let creature = creature_with_synapse_counts("h1", 1, 1);
        let synapse_counts = SynapseCounts::new(&creature);

        let neuron = ranked_neuron("h1", 0.0, 1.14e-7);
        let outcome = identify_removal_candidates(&[neuron], &synapse_counts, growth);

        assert!(
            outcome.candidates.is_empty(),
            "net-improvement ≈ 6.6e-8 is indistinguishable from noise and must be dropped; got {} candidates",
            outcome.candidates.len()
        );
        assert_eq!(
            outcome.noise_floor_rejections, 1,
            "the dropped candidate must be counted under noise_floor_rejections"
        );
    }

    /// Candidates well above the noise floor (net improvement ≈ 1.5e-5)
    /// must survive.
    ///
    /// Targets the issue's second scenario: boosted `savings ≈ 2e-5`,
    /// `impact = 5e-6`, `net ≈ 1.5e-5`.
    ///
    /// With 2 synapses (1 in + 1 out) the savings multiplier is `1.2`, and
    /// `REMOVAL_CANDIDATE_BOOST`(1.5) gives an effective multiplier of `1.8`,
    /// so pick `growth = 2e-5 / 1.8 ≈ 1.111e-5`.
    #[test]
    fn well_above_noise_floor_candidate_is_kept() {
        let _guard = env_lock();

        let growth = 2e-5_f32 / 1.8;
        let creature = creature_with_synapse_counts("h1", 1, 1);
        let synapse_counts = SynapseCounts::new(&creature);

        let neuron = ranked_neuron("h1", 0.0, 5e-6);
        let outcome = identify_removal_candidates(&[neuron], &synapse_counts, growth);

        assert_eq!(
            outcome.candidates.len(),
            1,
            "net-improvement ≈ 1.5e-5 is well above noise floor and must be kept"
        );
        assert_eq!(
            outcome.noise_floor_rejections, 0,
            "no candidates should be rejected by the noise floor"
        );
        assert_eq!(outcome.candidates[0].neuron_uuid, "h1");
    }

    /// Environment-variable override lets operators loosen the noise floor
    /// (or tighten it) without recompiling (Issue #1142 acceptance criterion).
    ///
    /// This test locks the env-var around the two `identify_removal_candidates`
    /// calls so concurrent tests are not affected.
    #[test]
    fn env_var_override_changes_effective_floor() {
        let _guard = env_lock();

        let growth = 1e-7_f32 / 1.5;
        let creature = creature_with_synapse_counts("h1", 1, 1);
        let synapse_counts = SynapseCounts::new(&creature);

        let neuron = ranked_neuron("h1", 0.0, 1.14e-7);

        // Default floor (1e-5) → dropped.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        let dropped =
            identify_removal_candidates(std::slice::from_ref(&neuron), &synapse_counts, growth);
        assert!(dropped.candidates.is_empty());
        assert_eq!(dropped.noise_floor_rejections, 1);

        // Loosened floor (1e-10) → kept.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR", "1e-10");
        }
        let kept =
            identify_removal_candidates(std::slice::from_ref(&neuron), &synapse_counts, growth);
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        assert_eq!(
            kept.candidates.len(),
            1,
            "candidate should survive when the env-var lowers the noise floor"
        );
        assert_eq!(kept.noise_floor_rejections, 0);
    }
}
