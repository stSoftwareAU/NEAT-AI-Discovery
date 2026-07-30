//! Removal candidate identification and constant neuron removal.
//!
//! Contains `RemovalCandidate`, `SynapseCounts`, `calculate_removal_savings`,
//! and detection of constant-value neurons for coordinated structural removal.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::DEFAULT_COST_OF_GROWTH;
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
///
/// # Conservation invariant (Issue #1808)
///
/// `candidates.len() + total_rejections() == considered`. Every neuron entering
/// triage is either emitted as a candidate or counted under a named rejection
/// reason; a future gate that drops one with a bare `continue` breaks the
/// equality and fails the conservation tests rather than vanishing silently.
#[derive(Debug, Default)]
pub(crate) struct RemovalCandidateOutcome {
    /// Removal candidates that passed every filter.
    pub candidates: Vec<RemovalCandidate>,
    /// Number of candidates dropped because `boosted_savings - impact` fell
    /// below [`remove_low_impact_noise_floor`] (Issue #1142).
    pub noise_floor_rejections: u32,
    /// Number of neurons dropped because the boosted savings did not exceed
    /// their contribution at all (Issue #1808) — the dominant rejection class
    /// at shipped defaults, previously counted nowhere.
    pub savings_below_impact_rejections: u32,
    /// Number of neurons dropped because they were still actively contributing
    /// (mean activation above [`REMOVAL_MEAN_ACTIVATION_THRESHOLD`] with
    /// meaningful impact — Issue #892, counted from Issue #1808). Always `0` on
    /// the structure-only path, which has no activations to measure.
    pub active_neuron_rejections: u32,
    /// Number of neurons that entered triage, whatever their verdict.
    pub considered: u32,
}

impl RemovalCandidateOutcome {
    /// Total neurons rejected across every named reason.
    pub(crate) fn total_rejections(&self) -> u32 {
        self.noise_floor_rejections
            .saturating_add(self.savings_below_impact_rejections)
            .saturating_add(self.active_neuron_rejections)
    }

    /// Enforce the conservation invariant at every construction site, so a new
    /// drop path that forgets its counter trips immediately in dev and test
    /// rather than under-reporting to consumers (Issue #1808).
    fn assert_conserved(&self) {
        debug_assert_eq!(
            u32::try_from(self.candidates.len()).unwrap_or(u32::MAX) + self.total_rejections(),
            self.considered,
            "removal triage dropped a neuron without counting it: {self:?}"
        );
    }

    /// Build a stable-keyed rejection breakdown for this outcome (Issue #1142,
    /// reused by the structure-only focus path in #1767, extended to every
    /// rejection class in #1808).
    ///
    /// Reuses the Issue #1129 rejection-reason vocabulary so downstream tooling
    /// (FFI consumers, dashboards) can merge these counts into the existing
    /// `metadata.rejection_breakdown` map without special-casing.
    pub(crate) fn rejection_breakdown(&self) -> HashMap<String, u32> {
        use crate::analysis::diagnostics::rejection_reasons::{
            REJECTION_REMOVAL_ACTIVE_NEURON, REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
            REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
        };
        let mut map = HashMap::new();
        for (reason, count) in [
            (
                REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
                self.noise_floor_rejections,
            ),
            (
                REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
                self.savings_below_impact_rejections,
            ),
            (
                REJECTION_REMOVAL_ACTIVE_NEURON,
                self.active_neuron_rejections,
            ),
        ] {
            if count > 0 {
                map.insert(reason.to_string(), count);
            }
        }
        map
    }
}

/// Identify removal candidates from ranked neurons.
///
/// Issue #235: Return ALL neurons where removal improves the creature's score.
/// A removal improves score when: `removal_savings` > `activation_weighted_impact`
///
/// Issue #892: Apply stricter filtering based on production discovery-cache evidence.
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
    // Issue #1814: the floor is denominated in units of `costOfGrowth`, the
    // same scale the savings term lives on, so the screen cannot drift out of
    // reach when the host changes `costOfGrowth`.
    let noise_floor = remove_low_impact_noise_floor(cost_of_growth_threshold);

    // Issue #1808: every neuron maps to exactly one verdict, so surviving
    // candidates and each rejection class are collected in a single parallel
    // pass and none can be dropped silently.
    #[derive(Debug)]
    enum Emit {
        Candidate(Box<RemovalCandidate>),
        NoiseFloorReject,
        SavingsBelowImpactReject,
        ActiveNeuronReject,
    }

    let emitted: Vec<Emit> = neurons
        .par_iter()
        .map(|n| {
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
                return Emit::SavingsBelowImpactReject;
            }

            // Issue #892: Filter out neurons with high mean activation when they
            // also have non-negligible structural impact. Cache evidence shows
            // failed removals have high mean activation (up to 57.8) combined with
            // meaningful impact — these neurons are actually contributing.
            // Disconnected neurons (impact ≈ 0) are always safe to remove regardless
            // of activation level.
            let has_meaningful_impact = n.impact > f32::EPSILON;
            if has_meaningful_impact && n.mean_activation > REMOVAL_MEAN_ACTIVATION_THRESHOLD {
                return Emit::ActiveNeuronReject;
            }

            // Net score improvement = boosted_savings - impact
            let net_improvement = boosted_savings - n.activation_weighted_impact;

            // Issue #1142: Gate on a noise floor to drop boost-inflated candidates
            // whose net improvement is indistinguishable from numerical noise
            // (e.g. 6.64e-8 in production discovery-cache analysis).
            if net_improvement < noise_floor {
                return Emit::NoiseFloorReject;
            }

            // Issue #117: expected_error_reduction should be based on activation_weighted_impact,
            // NOT total_error.
            let expected_error_reduction = n.activation_weighted_impact;

            Emit::Candidate(Box::new(RemovalCandidate {
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
            }))
        })
        .collect();

    let considered = u32::try_from(emitted.len()).unwrap_or(u32::MAX);
    let mut removal_candidates: Vec<RemovalCandidate> = Vec::with_capacity(emitted.len());
    let mut noise_floor_rejections: u32 = 0;
    let mut savings_below_impact_rejections: u32 = 0;
    let mut active_neuron_rejections: u32 = 0;
    for emit in emitted {
        match emit {
            Emit::Candidate(c) => removal_candidates.push(*c),
            Emit::NoiseFloorReject => {
                noise_floor_rejections = noise_floor_rejections.saturating_add(1);
            }
            Emit::SavingsBelowImpactReject => {
                savings_below_impact_rejections = savings_below_impact_rejections.saturating_add(1);
            }
            Emit::ActiveNeuronReject => {
                active_neuron_rejections = active_neuron_rejections.saturating_add(1);
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

    let outcome = RemovalCandidateOutcome {
        candidates: removal_candidates,
        noise_floor_rejections,
        savings_below_impact_rejections,
        active_neuron_rejections,
        considered,
    };
    outcome.assert_conserved();
    outcome
}

/// Resolve the effective cost-of-growth, rejecting values that would produce
/// nonsense savings. A non-finite or non-positive input is a caller bug, so it
/// is logged at WARN (never silently accepted) and the default substituted.
///
/// Issue #1783: this validation used to live only on the
/// [`triage_removal_candidates`](super::triage_removal_candidates) adapter, so
/// the shipped FFI path took `costOfGrowth` raw. It now guards the single
/// criterion, which is what makes the two entry points provably identical.
fn effective_cost_of_growth(cost_of_growth: Option<f32>) -> f32 {
    match cost_of_growth {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(invalid) => {
            tracing::warn!(
                target: "neat_ai_discovery::focus::removal_candidates",
                invalid_cost_of_growth = invalid,
                default_cost_of_growth = DEFAULT_COST_OF_GROWTH,
                "removal triage received a non-finite or non-positive costOfGrowth; using the default",
            );
            DEFAULT_COST_OF_GROWTH
        }
        None => DEFAULT_COST_OF_GROWTH,
    }
}

/// One hidden neuron's verdict under the structural removal criterion.
///
/// Issue #1808: every hidden neuron yields exactly one variant, so the triage
/// cannot drop one without accounting for it.
#[derive(Debug)]
enum StructuralEmit {
    Candidate(Box<RemovalCandidate>),
    NoiseFloorReject,
    SavingsBelowImpactReject,
}

/// The **single** structural removal criterion (Issue #1783).
///
/// Savings-vs-contribution, the [`REMOVAL_CANDIDATE_BOOST`] application point,
/// the non-finite-impact policy (#1804) and the noise-floor gate (#1142) exist
/// here and nowhere else. Both entry points —
/// [`identify_structural_removal_candidates`] and the
/// [`triage_removal_candidates`](super::triage_removal_candidates) adapter —
/// run this function, so they cannot drift apart again.
///
/// Returns [`StructuralEmit::SavingsBelowImpactReject`] when the neuron is not
/// worth pruning (savings do not exceed its structural contribution) — a
/// counted verdict, not a silent drop (Issue #1808).
fn structural_removal_verdict(
    neuron: &NeuronJson,
    impacts: &HashMap<String, f32>,
    synapse_counts: &SynapseCounts,
    growth_cost: f32,
    noise_floor: f32,
) -> StructuralEmit {
    let (incoming, outgoing) = synapse_counts.get(&neuron.uuid);
    let savings = calculate_removal_savings(incoming, outgoing, growth_cost);

    // Issue #892: boost applied to raw savings BEFORE the savings-vs-
    // contribution comparison (mirrors the record-derived path), re-gated
    // by the noise floor below (Issue #1142).
    let boosted_savings = savings * REMOVAL_CANDIDATE_BOOST;

    // Structural contribution = the neuron's path-weight impact on the outputs.
    //
    // Issue #1804: a **non-finite** impact cannot be reasoned about (a NaN
    // synapse weight propagates NaN through `(|weight| / total) * child_impact`),
    // so it maps to `f32::INFINITY` — the neuron is never pruned. Bad numbers
    // must not license a destructive edit; mapping them to `0.0` made them the
    // *strongest* removal candidate.
    //
    // A genuine `0.0` impact stays prunable. A **negative** impact is
    // unreachable by construction — every term is an `abs()` times a
    // non-negative child impact — so the `> 0.0` arm is defensive only and is
    // deliberately not folded in with the non-finite case.
    let raw_impact = impacts.get(&neuron.uuid).copied().unwrap_or(0.0);
    let contribution = if !raw_impact.is_finite() {
        f32::INFINITY
    } else if raw_impact > 0.0 {
        raw_impact
    } else {
        0.0
    };

    // Removal only improves the score when the complexity savings exceed the
    // structural contribution — the near-opposite of the high-impact focus draw.
    if boosted_savings <= contribution {
        return StructuralEmit::SavingsBelowImpactReject;
    }

    let net_improvement = boosted_savings - contribution;

    // Issue #1142: drop boost-inflated candidates whose net improvement is
    // indistinguishable from numerical noise.
    if net_improvement < noise_floor {
        return StructuralEmit::NoiseFloorReject;
    }

    StructuralEmit::Candidate(Box::new(RemovalCandidate {
        neuron_uuid: neuron.uuid.clone(),
        // Record-derived fields are not measured on the focus path.
        total_error: 0.0,
        impact: contribution,
        mean_activation: 0.0,
        activation_weighted_impact: 0.0,
        incoming_synapses: incoming,
        outgoing_synapses: outgoing,
        removal_savings: boosted_savings,
        // Structural first-pass estimate; the activation-weighted value is
        // refined in the analysis phase.
        expected_error_reduction: contribution,
        reason: format!(
            "Structural removal (Issue #1767): saves {savings:.2e} (boosted {REMOVAL_CANDIDATE_BOOST:.1}×) > structural impact {contribution:.2e} (net +{net_improvement:.2e}), {} synapses, costOfGrowth={growth_cost:.2e}; activation-weighted gate deferred to analysis",
            incoming + outgoing,
        ),
    }))
}

/// Identify removal candidates from creature **structure alone** — the
/// near-opposite axis to focus selection (Issue #1767).
///
/// Focus prefers **high** structural impact (weighted-random draw, #1766).
/// Removal is the near-opposite: a hidden neuron is a candidate when its
/// structural **contribution** (path-weight impact on the outputs) is **low**
/// relative to the complexity **savings** of pruning it and its synapses. This
/// is "low contribution vs savings", *not* "negate the focus score" (the #414
/// "high error ≠ remove" philosophy is untouched — error is never read here).
///
/// Nothing in this function opens or decodes discovery parquet: the impact map
/// comes from [`compute_impacts_public`](crate::focus::impact::compute_impacts_public)
/// (topology only) and synapse counts from [`SynapseCounts`]. It therefore never
/// reintroduces the focus-time parquet dependency that caused the ~2h focus stall
/// (#1766), and is `O(neurons + synapses)` — comfortably under the seconds bar.
///
/// The activation-weighted gates that genuinely need records — the
/// mean-activation guard and constant-neuron variance folding — are deliberately
/// **not** applied here; they run later in the analysis phase after the focus set
/// is fixed. To mark them as not-yet-measured this triage sets `mean_activation`
/// and `activation_weighted_impact` to `0.0`; `expected_error_reduction` carries
/// the structural contribution as a first-pass estimate, refined in analysis.
///
/// Only **hidden** neurons are considered: outputs (seeded at impact `1.0`) and
/// inputs / constants are never removal targets.
///
/// # Non-finite impact policy (Issue #1804)
///
/// A **non-finite** structural impact (NaN or infinite — a NaN synapse weight
/// propagates NaN through the impact product) means the neuron's contribution
/// **cannot be reasoned about**, so it is treated as `f32::INFINITY`: the
/// savings can never exceed it and the neuron is **never** emitted as a removal
/// candidate. Bad numbers must not license a destructive edit — the previous
/// mapping to `0.0` made a NaN-impact neuron the *strongest* candidate. A
/// genuine `0.0` impact remains prunable (a zero-contribution neuron *should*
/// be removable). This matches [`triage_removal_candidates`](super::triage_removal_candidates).
///
/// # Cost of growth
///
/// `cost_of_growth` is validated by [`effective_cost_of_growth`]: `None`, a
/// non-finite value, or a non-positive value falls back to
/// [`DEFAULT_COST_OF_GROWTH`] with a WARN, never producing nonsense savings
/// (Issue #1783 — the shipped path previously took the value raw).
pub(crate) fn identify_structural_removal_candidates(
    creature: &CreatureJson,
    cost_of_growth: Option<f32>,
) -> RemovalCandidateOutcome {
    let growth_cost = effective_cost_of_growth(cost_of_growth);
    let impacts = crate::focus::impact::compute_impacts_public(creature);
    let synapse_counts = SynapseCounts::new(creature);
    // Issue #1814: denominated in units of the *validated* `costOfGrowth`, so
    // the screen scales with the savings term it screens.
    let noise_floor = remove_low_impact_noise_floor(growth_cost);

    let emitted: Vec<StructuralEmit> = creature
        .neurons
        .par_iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| structural_removal_verdict(n, &impacts, &synapse_counts, growth_cost, noise_floor))
        .collect();

    // Issue #1808: one verdict per hidden neuron, so this *is* the number of
    // hidden neurons considered.
    let considered = u32::try_from(emitted.len()).unwrap_or(u32::MAX);
    let mut candidates: Vec<RemovalCandidate> = Vec::with_capacity(emitted.len());
    let mut noise_floor_rejections: u32 = 0;
    let mut savings_below_impact_rejections: u32 = 0;
    for emit in emitted {
        match emit {
            StructuralEmit::Candidate(c) => candidates.push(*c),
            StructuralEmit::NoiseFloorReject => {
                noise_floor_rejections = noise_floor_rejections.saturating_add(1);
            }
            StructuralEmit::SavingsBelowImpactReject => {
                savings_below_impact_rejections = savings_below_impact_rejections.saturating_add(1);
            }
        }
    }

    // Sort by net improvement (savings − contribution) descending: the biggest
    // structural wins first. Deterministic tie-break on the uuid.
    candidates.sort_by(|a, b| {
        let a_net = a.removal_savings - a.impact;
        let b_net = b.removal_savings - b.impact;
        b_net
            .total_cmp(&a_net)
            .then_with(|| a.impact.total_cmp(&b.impact))
            .then_with(|| a.neuron_uuid.cmp(&b.neuron_uuid))
    });

    if noise_floor_rejections > 0 || savings_below_impact_rejections > 0 {
        tracing::debug!(
            target: "neat_ai_discovery::focus::removal_candidates",
            noise_floor_rejections,
            savings_below_impact_rejections,
            noise_floor,
            considered,
            surviving = candidates.len(),
            "structural removal triage rejected hidden neurons",
        );
    }

    let outcome = RemovalCandidateOutcome {
        candidates,
        noise_floor_rejections,
        savings_below_impact_rejections,
        // Never measured on the structure-only path — the mean-activation gate
        // needs records and stays in the analysis phase.
        active_neuron_rejections: 0,
        considered,
    };
    outcome.assert_conserved();
    outcome
}

/// Threshold for considering a neuron as "constant" (near-zero variance).
/// Issue #217/306: Neurons with variance below this threshold are treated as constant
/// and can be removed with bias adjustments for downstream neurons.
///
/// Value 1e-10 is from Issue #217's proposal for `DEAD_VARIANCE_THRESHOLD`.
const CONSTANT_VARIANCE_THRESHOLD: f32 = 1e-10;

/// Collect the UUIDs of functionally-constant hidden neurons — those whose
/// recorded activation variance is at or below [`CONSTANT_VARIANCE_THRESHOLD`]
/// (Issue #1624).
///
/// These are exactly the neurons the constant-removal path
/// ([`detect_constant_neuron_removals`], Issue #306) folds into downstream
/// biases. A neuron whose output never varies cannot yield a successful
/// add-synapse / add-neuron focus candidate — no structural change feeding a
/// constant signal moves the network — so every focus slot it occupies is
/// wasted. This set lets the ranker make such neurons ineligible for focus
/// slots while leaving them fully available to the removal path.
///
/// Only `hidden` neurons are considered: output neurons are always
/// focus-eligible (they are the add-neuron targets) and input / constant-type
/// neurons are never selectable in the first place. This mirrors the hidden-only
/// gate in [`detect_constant_neuron_removals`], so the same neurons are made
/// focus-ineligible and offered as removal candidates.
pub(super) fn functionally_constant_focus_uuids(
    selectable: &[&NeuronJson],
    records_provider: &Arc<dyn RecordProvider>,
    creature: &CreatureJson,
) -> std::collections::HashSet<String> {
    let neuron_types: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();

    selectable
        .par_iter()
        .filter_map(|neuron| {
            if neuron_types.get(neuron.uuid.as_str()).copied() != Some("hidden") {
                return None;
            }
            let records = get_records_or_error(records_provider.as_ref(), &neuron.uuid).ok()?;
            let (_mean, variance) = activation_mean_and_variance_from_records(&records);
            (variance <= CONSTANT_VARIANCE_THRESHOLD).then(|| neuron.uuid.clone())
        })
        .collect()
}

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
                remove_neuron_compensation: None,
                constant_neuron_bias_fold: None,
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
// Shared test env-lock (Issue #1142, #1767)
// =============================================================================

/// Lock protecting every test that reads or writes
/// `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR`. Both the record-derived
/// (#1142) and structure-only (#1767) removal tests share it, so a test that
/// mutates the env-var never races one that depends on the default floor.
///
/// Issue #1814: the lock itself now lives beside the constant so the
/// constants-module tests share it too.
#[cfg(test)]
pub(super) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::analysis::constants::noise_floor_env_lock()
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
            weighted_score: 0.0,
            reconstruction_mismatch: 0.0,
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
        // SAFETY: env access is serialised via `env_lock()` for the duration of this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        let dropped =
            identify_removal_candidates(std::slice::from_ref(&neuron), &synapse_counts, growth);
        assert!(dropped.candidates.is_empty());
        assert_eq!(dropped.noise_floor_rejections, 1);

        // Loosened floor (1e-10) → kept.
        // SAFETY: env access is serialised via `env_lock()` for the duration of this test.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR", "1e-10");
        }
        let kept =
            identify_removal_candidates(std::slice::from_ref(&neuron), &synapse_counts, growth);
        // SAFETY: env access is serialised via `env_lock()` for the duration of this test.
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

    /// Issue #1808: the record-derived path must also account for every neuron
    /// it triages — one verdict each, across all three gates.
    ///
    /// With `growth = 2e-5 / 1.8` and 1 in + 1 out synapse per neuron the
    /// boosted savings are `2e-5`, which places each fixture neuron in exactly
    /// one class.
    #[test]
    fn every_triaged_neuron_lands_in_exactly_one_class() {
        let _guard = env_lock();
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }

        let growth = 2e-5_f32 / 1.8;
        let creature = creature_with_synapse_counts("h-keep", 1, 1);
        let synapse_counts = SynapseCounts::new(&creature);

        let neurons = [
            // Contribution above the 2e-5 boosted savings ⇒ savings-vs-impact.
            ranked_neuron("h-contributing", 0.0, 1e-3),
            // Tiny-but-real impact with mean activation 0.05 > the 0.04
            // threshold ⇒ still actively contributing.
            ranked_neuron("h-active", 2e-7, 1e-8),
            // Net improvement 5e-6, under the default 1e-5 floor ⇒ noise.
            ranked_neuron("h-noise", 0.0, 1.5e-5),
            // Net improvement 1.5e-5 ⇒ survives as a candidate.
            ranked_neuron("h-keep", 0.0, 5e-6),
        ];
        let outcome = identify_removal_candidates(&neurons, &synapse_counts, growth);

        assert_eq!(outcome.considered, 4);
        assert_eq!(outcome.savings_below_impact_rejections, 1, "{outcome:?}");
        assert_eq!(outcome.active_neuron_rejections, 1, "{outcome:?}");
        assert_eq!(outcome.noise_floor_rejections, 1, "{outcome:?}");
        assert_eq!(outcome.candidates.len(), 1, "{outcome:?}");
        assert_eq!(
            outcome.candidates.len() as u32 + outcome.total_rejections(),
            outcome.considered,
            "candidates + rejections must equal the neurons considered: {outcome:?}"
        );

        let breakdown = outcome.rejection_breakdown();
        assert_eq!(
            breakdown.values().sum::<u32>(),
            outcome.total_rejections(),
            "every rejection must be named in the breakdown: {breakdown:?}"
        );
    }
}

// =============================================================================
// Unit tests for Issue #1767 structure-only removal triage
// =============================================================================

#[cfg(test)]
mod structural_removal_tests {
    //! Issue #1767: removal triage on the near-opposite axis to focus, computed
    //! from creature **structure alone** — no discovery records, no parquet.

    use super::*;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    fn neuron(uuid: &str, ntype: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }
    }

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    /// A creature with **no output**: every hidden neuron has zero structural
    /// impact (nothing to seed the path-weight products at 1.0), so hidden
    /// neurons are pure low-contribution removal candidates.
    fn outputless_chain() -> CreatureJson {
        // i0 → h1 → h2  (h1: 1 in + 1 out, h2: 1 in + 0 out)
        CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("h2", "hidden"),
            ],
            synapses: vec![synapse("i0", "h1", 0.5), synapse("h1", "h2", 0.5)],
            input: 1,
            output: 0,
        }
    }

    /// Low structural contribution + savings above the noise floor ⇒ removal
    /// candidates, ordered biggest structural win first. Uses a large
    /// `costOfGrowth` so the boosted savings clear the default 1e-5 floor.
    #[test]
    fn low_impact_hidden_neurons_are_removal_candidates() {
        let _guard = env_lock();
        let creature = outputless_chain();
        // growth 1e-4 → h1 boosted savings 1.8e-4, h2 1.65e-4; both net > 1e-5.
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-4));

        assert_eq!(outcome.candidates.len(), 2, "both hidden neurons qualify");
        // h1 has more synapses (higher savings) so it sorts first.
        assert_eq!(outcome.candidates[0].neuron_uuid, "h1");
        assert_eq!(outcome.candidates[1].neuron_uuid, "h2");
        for c in &outcome.candidates {
            assert_eq!(c.impact, 0.0, "no path to output ⇒ zero structural impact");
            // Record-derived fields are unmeasured on the structure-only path.
            assert_eq!(c.mean_activation, 0.0);
            assert_eq!(c.activation_weighted_impact, 0.0);
            assert!(c.removal_savings > 0.0);
        }
        assert_eq!(outcome.noise_floor_rejections, 0);
    }

    /// A hidden neuron that drives the output has HIGH structural contribution —
    /// the near-opposite of a removal candidate — and must never be flagged.
    #[test]
    fn high_impact_hidden_neuron_is_not_a_removal_candidate() {
        let _guard = env_lock();
        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![synapse("i0", "h1", 1.0), synapse("h1", "o0", 1.0)],
            input: 1,
            output: 1,
        };
        // Even with a large costOfGrowth, boosted savings ≪ the ~1.0 impact.
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-3));
        assert!(
            outcome.candidates.is_empty(),
            "high-impact neuron must not be a removal candidate, got {:?}",
            outcome.candidates
        );
    }

    /// Output neurons are never removal candidates even when they carry
    /// synapses — the triage is hidden-only.
    #[test]
    fn outputs_are_never_removal_candidates() {
        let _guard = env_lock();
        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![synapse("i0", "h1", 1.0), synapse("h1", "o0", 1.0)],
            input: 1,
            output: 1,
        };
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-3));
        assert!(
            outcome.candidates.iter().all(|c| c.neuron_uuid != "o0"),
            "output neuron must never appear as a removal candidate"
        );
    }

    /// Savings below the noise floor are counted as rejections rather than
    /// emitted (Issue #1142 gate, reused).
    ///
    /// **Issue #1814 changed this test's setup, not its contract.** It used to
    /// rely on the default floor being an absolute `1e-5` that `costOfGrowth =
    /// 1e-7` could never reach — the very defect #1814 fixes, so at the default
    /// these neurons are now (correctly) candidates. The absolute env override
    /// pins the historical `1e-5` so the *counting* behaviour this test exists
    /// for is still exercised; reachability at the default is covered by
    /// `noise_floor_denomination_tests`.
    #[test]
    fn below_noise_floor_savings_are_rejected_and_counted() {
        let _guard = env_lock();
        // Absolute floor pinned at 1e-5; growth 1e-7 → boosted savings ~1.8e-7 ≪ floor.
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR", "1e-5");
        }
        let creature = outputless_chain();
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-7));
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        assert!(
            outcome.candidates.is_empty(),
            "sub-noise-floor structural savings must be dropped"
        );
        assert_eq!(
            outcome.noise_floor_rejections, 2,
            "both dropped neurons must be counted under the noise-floor gate"
        );
        // The rejection breakdown surfaces the drop under the stable reason key.
        let breakdown = outcome.rejection_breakdown();
        use crate::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR;
        assert_eq!(breakdown.get(REJECTION_REMOVAL_BELOW_NOISE_FLOOR), Some(&2));
    }

    /// Issue #1804: a creature whose `h-nan` hidden neuron carries a NaN
    /// structural impact, alongside the two non-regression cases.
    ///
    /// * `h-nan` — NaN weight into `o0`, so `compute_impacts_public` evaluates
    ///   `(NaN / NaN) * 1.0` = NaN for it.
    /// * `h-zero` — weight `0.0` into `o1` ⇒ a genuine `0.0` impact.
    /// * `h-keep` — weight `1.0` into `o1` ⇒ impact `1.0`, far above savings.
    ///
    /// A NaN weight cannot cross the JSON FFI boundary (serde rejects it), so
    /// this policy is only reachable — and therefore only testable — here.
    fn creature_with_nan_weight() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h-nan", "hidden"),
                neuron("h-zero", "hidden"),
                neuron("h-keep", "hidden"),
                neuron("o0", "output"),
                neuron("o1", "output"),
            ],
            synapses: vec![
                synapse("i0", "h-nan", 0.5),
                synapse("i0", "h-zero", 0.5),
                synapse("i0", "h-keep", 0.5),
                synapse("h-nan", "o0", f32::NAN),
                synapse("h-zero", "o1", 0.0),
                synapse("h-keep", "o1", 1.0),
            ],
            input: 1,
            output: 2,
        }
    }

    /// Issue #1804 acceptance 2: a neuron whose structural impact is NaN is
    /// never offered for removal — bad numbers must not license a destructive
    /// edit. Before the fix, NaN mapped to a `0.0` contribution and made this
    /// neuron the *strongest* candidate.
    #[test]
    fn nan_impact_neuron_absent_from_candidates() {
        let _guard = env_lock();
        let creature = creature_with_nan_weight();
        // Sanity: the fixture really does produce a non-finite impact.
        let impacts = crate::focus::impact::compute_impacts_public(&creature);
        assert!(
            !impacts.get("h-nan").copied().unwrap_or(0.0).is_finite(),
            "fixture must yield a non-finite impact for h-nan, or the test proves nothing"
        );

        let outcome = identify_structural_removal_candidates(&creature, Some(1e-4));
        assert!(
            outcome.candidates.iter().all(|c| c.neuron_uuid != "h-nan"),
            "a neuron with a non-finite structural impact must never be a removal candidate, got {:?}",
            outcome
                .candidates
                .iter()
                .map(|c| c.neuron_uuid.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// Issue #1804 acceptance 3: the fix must not over-correct — a genuinely
    /// zero-contribution neuron is still prunable, and a high-impact one is
    /// still safe.
    #[test]
    fn zero_impact_neuron_still_candidate() {
        let _guard = env_lock();
        let creature = creature_with_nan_weight();
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-4));

        assert!(
            outcome.candidates.iter().any(|c| c.neuron_uuid == "h-zero"),
            "a genuine 0.0-impact neuron must remain a removal candidate"
        );
        assert!(
            outcome.candidates.iter().all(|c| c.neuron_uuid != "h-keep"),
            "a high-impact neuron must never be a removal candidate"
        );
    }

    /// Issue #1804 acceptance 1: the invariant asserted directly over the
    /// returned list — no emitted candidate carries a non-finite impact.
    #[test]
    fn no_candidate_has_nonfinite_raw_impact() {
        let _guard = env_lock();
        let creature = creature_with_nan_weight();
        let outcome = identify_structural_removal_candidates(&creature, Some(1e-4));

        for candidate in &outcome.candidates {
            assert!(
                candidate.impact.is_finite(),
                "candidate {} has a non-finite impact {}",
                candidate.neuron_uuid,
                candidate.impact
            );
            assert!(
                candidate.expected_error_reduction.is_finite(),
                "candidate {} has a non-finite expected error reduction",
                candidate.neuron_uuid
            );
        }
    }

    /// The triage is a pure function of structure — identical creature in,
    /// identical candidates out — and needs no records/parquet to run.
    #[test]
    fn triage_is_deterministic_and_record_free() {
        let _guard = env_lock();
        let creature = outputless_chain();
        let a = identify_structural_removal_candidates(&creature, Some(1e-4));
        let b = identify_structural_removal_candidates(&creature, Some(1e-4));
        let ids_a: Vec<&str> = a
            .candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();
        let ids_b: Vec<&str> = b
            .candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();
        assert_eq!(ids_a, ids_b);
    }
}

// =============================================================================
// Unit tests for Issue #1814 — noise floor denominated in costOfGrowth
// =============================================================================

#[cfg(test)]
mod noise_floor_denomination_tests {
    //! Issue #1814: `REMOVE_LOW_IMPACT_NOISE_FLOOR` was an absolute `1e-5`
    //! screening `boostedSavings − contribution`, a term **linear in the
    //! host-supplied `costOfGrowth`**. At the shipped
    //! [`DEFAULT_COST_OF_GROWTH`] a
    //! zero-contribution neuron needed 657 synapses to clear it, so the gate
    //! rejected every neuron the production population contains.
    //!
    //! These tests pin the three properties of the re-denominated screen: a
    //! realistic neuron survives, the #1142 numerical-noise class is still
    //! rejected, and the verdict no longer moves when `costOfGrowth` does.

    use super::*;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    /// NEAT-AI's shipped default — the value the host actually sends. Resolved
    /// through the single definition (Issue #1807), never restated as a literal.
    const HOST_COST_OF_GROWTH: f32 = crate::focus::DEFAULT_COST_OF_GROWTH;

    fn neuron(uuid: &str, ntype: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }
    }

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    /// A creature whose `h-low` hidden neuron is *attenuated* rather than
    /// disconnected: it reaches the single output through a weight of
    /// `out_weight` while `h-dom` reaches it through `1.0`, so under the linear
    /// (IDENTITY) impact rule
    /// `impact(h-low) = out_weight / (out_weight + 1) ≈ out_weight`.
    ///
    /// `incoming` inputs feed `h-low`, giving it a degree of `incoming + 1` —
    /// the realistic-degree dial the acceptance criteria ask for.
    fn attenuated_hidden(incoming: usize, out_weight: f32) -> CreatureJson {
        let mut neurons = vec![
            neuron("h-low", "hidden"),
            neuron("h-dom", "hidden"),
            neuron("o0", "output"),
        ];
        let mut synapses = vec![
            synapse("h-low", "o0", out_weight),
            synapse("h-dom", "o0", 1.0),
        ];
        for i in 0..incoming {
            let source = format!("i-{i}");
            neurons.push(neuron(&source, "input"));
            synapses.push(synapse(&source, "h-low", 0.5));
        }
        neurons.push(neuron("i-dom", "input"));
        synapses.push(synapse("i-dom", "h-dom", 0.5));

        let input = neurons.iter().filter(|n| n.neuron_type == "input").count();
        CreatureJson {
            neurons,
            synapses,
            input,
            output: 1,
        }
    }

    /// Clear both overrides so the shipped default governs.
    fn use_shipped_defaults() {
        // SAFETY: every caller holds `env_lock()`, so no other test touches the
        // environment concurrently.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS");
        }
    }

    /// Acceptance 1: a low-contribution hidden neuron with a **realistic**
    /// degree (≤ 20 synapses) at the host's [`DEFAULT_COST_OF_GROWTH`] survives
    /// the screen.
    ///
    /// 12 synapses (11 in + 1 out) gives boosted savings
    /// `1.5 × 1e-7 × 2.2 = 3.3e-7` against a `1e-8` contribution — a net of
    /// `3.2e-7`, comfortably over the `1e-7` floor. Against the old absolute
    /// `1e-5` floor this net was **30× short**, so this test also proves the
    /// fix landed.
    #[test]
    fn realistic_neuron_survives_noise_floor_at_default_cost_of_growth() {
        let _guard = env_lock();
        use_shipped_defaults();

        let creature = attenuated_hidden(11, 1e-8);
        let outcome = identify_structural_removal_candidates(&creature, Some(HOST_COST_OF_GROWTH));

        assert!(
            outcome.candidates.iter().any(|c| c.neuron_uuid == "h-low"),
            "a 12-synapse, 1e-8-contribution hidden neuron must survive at \
             the default cost-of-growth; got candidates {:?} with {} noise-floor rejections",
            outcome
                .candidates
                .iter()
                .map(|c| c.neuron_uuid.as_str())
                .collect::<Vec<_>>(),
            outcome.noise_floor_rejections,
        );
        assert_eq!(
            outcome.noise_floor_rejections, 0,
            "nothing in this fixture is numerical noise: {outcome:?}"
        );

        // Every realistic degree in the population clears the floor, not just 12.
        for degree in [0_usize, 1, 4, 19] {
            let creature = attenuated_hidden(degree, 1e-8);
            let outcome =
                identify_structural_removal_candidates(&creature, Some(HOST_COST_OF_GROWTH));
            assert!(
                outcome.candidates.iter().any(|c| c.neuron_uuid == "h-low"),
                "degree {} must clear the floor at the default cost-of-growth",
                degree + 1
            );
        }
    }

    /// Acceptance 2: the #1142 numerical-noise class is still rejected.
    ///
    /// Reproduces production discovery-cache entry
    /// `v2_remove-low-impact_0ce92a87…`: a 2-synapse neuron at
    /// the default cost-of-growth has boosted savings `1.8e-7`, and an attenuation of
    /// `1.136e-7` puts its contribution at the cached `1.14e-7`, leaving the
    /// cached `+6.64e-8` net improvement. That is `0.664` units of
    /// `costOfGrowth`, below the `1.0`-unit floor, so it is dropped — and
    /// counted, never silently swallowed.
    #[test]
    fn numerical_noise_class_still_rejected() {
        let _guard = env_lock();
        use_shipped_defaults();

        let creature = attenuated_hidden(1, 1.136e-7);
        let impacts = crate::focus::impact::compute_impacts_public(&creature);
        let contribution = impacts.get("h-low").copied().unwrap_or(0.0);
        let boosted =
            calculate_removal_savings(1, 1, HOST_COST_OF_GROWTH) * REMOVAL_CANDIDATE_BOOST;
        let net = boosted - contribution;
        assert!(
            (5e-8..8e-8).contains(&net),
            "fixture must reproduce the 6.64e-8-class net improvement, got {net:e}"
        );

        let outcome = identify_structural_removal_candidates(&creature, Some(HOST_COST_OF_GROWTH));
        assert!(
            outcome.candidates.iter().all(|c| c.neuron_uuid != "h-low"),
            "a {net:e} net improvement is numerical noise and must be rejected (#1142)"
        );
        assert_eq!(
            outcome.noise_floor_rejections, 1,
            "the drop must be counted under the noise-floor gate: {outcome:?}"
        );
    }

    /// Acceptance 3: the screen's strictness scales with `costOfGrowth`.
    ///
    /// Both terms of a zero-contribution neuron's net improvement are linear in
    /// `costOfGrowth`, so its verdict must be **invariant** across the whole
    /// ladder. Under the old absolute `1e-5` floor it was not: the same neuron
    /// was rejected at `1e-7` and accepted at `1e-4`.
    #[test]
    fn screen_strictness_scales_with_cost_of_growth() {
        let _guard = env_lock();
        use_shipped_defaults();

        let creature = attenuated_hidden(11, 0.0);
        let verdicts: Vec<(f32, bool, u32)> = [1e-7_f32, 1e-6, 1e-5, 1e-4]
            .into_iter()
            .map(|cost_of_growth| {
                let outcome =
                    identify_structural_removal_candidates(&creature, Some(cost_of_growth));
                (
                    cost_of_growth,
                    outcome.candidates.iter().any(|c| c.neuron_uuid == "h-low"),
                    outcome.noise_floor_rejections,
                )
            })
            .collect();

        for (cost_of_growth, accepted, rejections) in &verdicts {
            assert!(
                *accepted && *rejections == 0,
                "verdict must not depend on costOfGrowth; at {cost_of_growth:e} the \
                 neuron was accepted={accepted} with {rejections} noise-floor rejections \
                 (full ladder: {verdicts:?})"
            );
        }
    }

    /// The `costOfGrowth` coupling holds for the **rejection** verdict too: the
    /// noise-class neuron of acceptance 2 scaled up by 10× — contribution and
    /// all — is still rejected at 10× `costOfGrowth`.
    #[test]
    fn noise_class_rejection_also_scales_with_cost_of_growth() {
        let _guard = env_lock();
        use_shipped_defaults();

        for (cost_of_growth, out_weight) in [
            (HOST_COST_OF_GROWTH, 1.136e-7_f32),
            (HOST_COST_OF_GROWTH * 10.0, 1.136e-6),
        ] {
            let creature = attenuated_hidden(1, out_weight);
            let outcome = identify_structural_removal_candidates(&creature, Some(cost_of_growth));
            assert_eq!(
                outcome.noise_floor_rejections, 1,
                "the noise class must stay rejected at costOfGrowth {cost_of_growth:e}: {outcome:?}"
            );
        }
    }
}
