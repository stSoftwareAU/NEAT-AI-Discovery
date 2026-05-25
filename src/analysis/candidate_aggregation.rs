//! Candidate aggregation and post-processing (Issue #562).
//!
//! This module handles the post-processing steps after synapse and neuron analysis:
//! - Converting add-neuron candidates to coordinated structural replacements
//!   when a direct synapse already exists (Issue #173)
//! - Merging coordinated structural replacements into synapse results
//! - Truncating candidate sets to respect `max_synapse_candidates`

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use crate::{
    AnalyzeAllInput, CandidateNeuronJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson,
};

use super::constants::{
    MIN_COORDINATED_MULTI_OP_GAIN, coordinated_empirical_discount,
    coordinated_post_discount_noise_floor,
};
use super::{cache, shared, synapse};

/// Filter coordinated-structural candidates below the post-discount noise
/// floor (Issue #1110, #1128, #1272).
///
/// Intended as the **final** filter in the pipeline — applied after all
/// pessimism, calibration, module-boost, and ensemble discounts so that gains
/// discounted into the documented noise range (1e-7 to 1e-8 per Issue #1127
/// failure evidence) cannot reach the FFI response.
///
/// The floor is applied **per operation count** via
/// [`coordinated_post_discount_noise_floor`] — 5e-7 for 1-op, 1e-6 for 2-op,
/// 2e-6 for 3-op, and 5e-6 for 4+-op (Issue #1272). Higher-op candidates
/// carry materially higher implementation risk (see GRQ-sampler `bcbca347`
/// 4-op failures that just cleared the legacy single 5e-7 floor yet harmed
/// the network by 1000–6000× the predicted magnitude), so a per-tier floor
/// is enforced in step with the empirical-discount tiers from #1058.
///
/// Issue #1129: Returns the number of candidates removed so callers can
/// record the drop in the structured rejection breakdown.
pub fn apply_coordinated_gain_floor(
    candidates: &mut Vec<CoordinatedStructuralCandidateJson>,
) -> u32 {
    apply_coordinated_gain_floor_with_multiplier(candidates, 1.0)
}

/// Variant of [`apply_coordinated_gain_floor`] that multiplies each
/// candidate's per-op-count floor by a caller-supplied factor before
/// filtering (Issue #1132, #1272).
///
/// The `multiplier` must be `>= 1.0`; values below 1.0 are clamped to 1.0 so
/// the floor can never become looser than the default. Conservative discovery
/// mode passes a multiplier above 1.0 to bias away from borderline structural
/// candidates when the creature has a low recent success rate. The multiplier
/// is applied to the per-tier floor returned by
/// [`coordinated_post_discount_noise_floor`], so the relative tier ordering
/// is preserved.
pub fn apply_coordinated_gain_floor_with_multiplier(
    candidates: &mut Vec<CoordinatedStructuralCandidateJson>,
    multiplier: f32,
) -> u32 {
    let factor = multiplier.max(1.0);
    let before = candidates.len();
    candidates.retain(|c| {
        let floor = coordinated_post_discount_noise_floor(c.operations.len()) * factor;
        c.expected_creature_score_gain >= floor
    });
    u32::try_from(before.saturating_sub(candidates.len())).unwrap_or(u32::MAX)
}

/// Apply the final coordinated-structural gain floor to a synapse result and
/// refresh dependent metadata (Issue #1139).
///
/// This is the FFI-facing safety net: it must always run after variant
/// generation (`pair_coordinated_structural_with_weight_variants`) because the
/// `0.75×`/`0.5×`/`0.25×`/`0.1×` expected-gain multipliers can pull a variant
/// below its per-op-count noise floor (see
/// `coordinated_post_discount_noise_floor`, Issue #1272) even when its base
/// candidate is above the floor. Production evidence (GRQ-sampler discoveryVersion
/// 0.74.16) captured `Gentle Nudge` variants with gains of ~1.3e-7 damaging
/// creatures when tested.
///
/// The floor was previously applied only inside the
/// `!memory_budget_exceeded && !post_processing_deadline_passed` fast-path
/// guard in `analyze_all`, so sub-floor variants leaked whenever the memory
/// budget or deadline was exceeded. Moving the call out of that guard means:
///
/// - `expected_creature_score_gain` is screened in both the fast-path and the
///   skipped-post-processing fallback.
/// - `metadata.rejection_breakdown[REJECTION_BELOW_EXPECTED_GAIN_FLOOR]` is
///   updated with the drop count.
/// - `metadata.candidates_returned` is refreshed so the FFI caller sees an
///   accurate total after filtering.
///
/// Returns the number of coordinated-structural candidates removed.
pub fn apply_final_coordinated_gain_floor(
    synapse: &mut shared::AnalyzeSynapsesResult,
    discovery_mode: super::discovery_mode::DiscoveryMode,
    conservative_multiplier: f32,
) -> u32 {
    use super::diagnostics::rejection_reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR;

    let gain_multiplier = super::discovery_mode::coordinated_gain_multiplier_for_mode(
        discovery_mode,
        conservative_multiplier,
    );
    let removed = apply_coordinated_gain_floor_with_multiplier(
        &mut synapse.coordinated_structural_candidates,
        gain_multiplier,
    );
    synapse
        .metadata
        .rejection_breakdown
        .record_many_u32(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, removed);
    synapse.metadata.candidates_returned = synapse.helpful_synapses.len()
        + synapse.harmful_synapses.len()
        + synapse.coordinated_structural_candidates.len();
    removed
}

/// Compute the operation-count discount for a coordinated candidate (Issue #732, #1058).
///
/// Issue #1058: Replaced the three-layer compound discount (per-op exponential ×
/// flat pessimism) with a single empirical lookup per operation count, derived
/// from GRQ-sampler success rates.
///
/// Single-operation candidates receive no discount (returns original gain).
pub fn apply_operation_count_discount(candidate: &CoordinatedStructuralCandidateJson) -> f32 {
    let op_count = candidate.operations.len();
    candidate.expected_creature_score_gain * coordinated_empirical_discount(op_count)
}

/// Validate whether a coordinated candidate's gain exceeds the minimum threshold (Issue #732).
///
/// Single-operation candidates only require positive gain — the pipeline's
/// final post-discount noise sweep (`apply_coordinated_gain_floor`, Issue #1128)
/// catches sub-noise gains after all downstream pessimism/calibration stages.
/// Multi-operation candidates (>= 2 operations) must exceed
/// `MIN_COORDINATED_MULTI_OP_GAIN` after per-op discounting to filter out
/// near-zero predictions that almost never succeed in practice.
pub fn validate_coordinated_candidate_gain(candidate: &CoordinatedStructuralCandidateJson) -> bool {
    if candidate.operations.len() <= 1 {
        return candidate.expected_creature_score_gain > 0.0;
    }
    let discounted = apply_operation_count_discount(candidate);
    discounted > MIN_COORDINATED_MULTI_OP_GAIN
}

/// Merge coordinated structural replacement candidates into the synapse result.
///
/// Filters out candidates with non-positive expected gain (Issue #557),
/// applies operation-count discount for multi-operation candidates (Issue #732),
/// filters multi-op candidates below `MIN_COORDINATED_MULTI_OP_GAIN` **after**
/// discounting (Issue #732, #1110), sorts by expected gain (unless
/// diversified), and truncates to the combined synapse candidate limit. The
/// final post-discount noise sweep happens in `analyze_all` via
/// `apply_coordinated_gain_floor` (Issue #1128).
///
/// Issue #1129: Records a structured breakdown of removed candidates in
/// `synapse.metadata.rejection_breakdown` so the FFI caller can root-cause
/// "no candidates found" failures without re-running analysis.
pub(crate) fn merge_coordinated_structural_replacements(
    synapse: &mut shared::AnalyzeSynapsesResult,
    mut replacements: Vec<CoordinatedStructuralCandidateJson>,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
) {
    use crate::analysis::diagnostics::rejection_reasons::{
        REJECTION_BELOW_MULTI_OP_FLOOR, REJECTION_NON_POSITIVE_GAIN,
    };

    if replacements.is_empty() {
        return;
    }

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    // Only candidates predicted to improve the creature's score should be returned.
    let before_positive = replacements.len();
    replacements.retain(|c| c.expected_creature_score_gain > 0.0);
    let dropped_non_positive =
        u32::try_from(before_positive.saturating_sub(replacements.len())).unwrap_or(u32::MAX);
    synapse
        .metadata
        .rejection_breakdown
        .record_many_u32(REJECTION_NON_POSITIVE_GAIN, dropped_non_positive);
    if replacements.is_empty() {
        return;
    }

    // Issue #732: Apply operation-count discount and minimum gain validation.
    // Multi-operation candidates have compounding prediction uncertainty.
    for c in &mut replacements {
        if c.operations.len() > 1 {
            c.expected_creature_score_gain = apply_operation_count_discount(c);
        }
    }
    // Issue #732, #1110: After discounting, filter multi-op candidates below
    // the multi-op minimum gain threshold. Single-op candidates are kept on
    // `> 0.0` (filtered above) — the final post-discount noise sweep in
    // `analyze_all` (Issue #1128) catches sub-noise gains that survived
    // downstream pessimism/calibration stages.
    let before_multi = replacements.len();
    replacements.retain(|c| {
        if c.operations.len() <= 1 {
            c.expected_creature_score_gain > 0.0
        } else {
            c.expected_creature_score_gain > MIN_COORDINATED_MULTI_OP_GAIN
        }
    });
    let dropped_multi =
        u32::try_from(before_multi.saturating_sub(replacements.len())).unwrap_or(u32::MAX);
    synapse
        .metadata
        .rejection_breakdown
        .record_many_u32(REJECTION_BELOW_MULTI_OP_FLOOR, dropped_multi);
    if replacements.is_empty() {
        return;
    }

    synapse
        .coordinated_structural_candidates
        .append(&mut replacements);

    // Keep deterministic ordering for non-deadline runs.
    //
    // Deadline-diversified runs rely on upstream per-bucket ordering and round-robin selection,
    // so we avoid resorting here when `diversify` is enabled.
    if !diversify {
        synapse.coordinated_structural_candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .total_cmp(&a.expected_creature_score_gain)
        });
    }

    if let Some(limit) = max_synapse_candidates {
        let (helpful, harmful, coordinated) = synapse::truncate_combined_synapse_candidate_sets(
            mem::take(&mut synapse.helpful_synapses),
            mem::take(&mut synapse.harmful_synapses),
            mem::take(&mut synapse.coordinated_structural_candidates),
            limit,
            diversify,
        );
        synapse.helpful_synapses = helpful;
        synapse.harmful_synapses = harmful;
        synapse.coordinated_structural_candidates = coordinated;
    }

    // Ensure metadata reflects what we actually return to callers.
    synapse.metadata.candidates_returned = synapse.helpful_synapses.len()
        + synapse.harmful_synapses.len()
        + synapse.coordinated_structural_candidates.len();
}

/// Update neuron result after filtering out candidates converted to coordinated replacements.
pub(crate) fn apply_kept_neuron_candidates(
    neuron: &mut shared::AnalyzeNeuronsResult,
    kept_neurons: Vec<CandidateNeuronJson>,
) {
    // Regression fix (7-Jan-2026):
    // Post-processing may convert some add-neuron candidates into coordinated structural
    // replacements. When we filter those out of `helpful_neurons`, we must also update the
    // metadata so JSON output remains consistent with the returned arrays.
    neuron.helpful_neurons = kept_neurons;
    neuron.metadata.candidates_returned = neuron.helpful_neurons.len();
}

/// Convert eligible add-neuron candidates into coordinated structural replacements.
///
/// Issue #173 (7-Jan-2026): When a direct synapse already exists between (source -> target),
/// applying an add-neuron candidate without removing the synapse is often not the intended edit.
/// We instead emit a coordinated group that removes the synapse and inserts the hidden neuron
/// path atomically.
///
/// Normal add-neurons discovery works as-is for cases where no direct synapse exists.
pub(crate) fn convert_neurons_to_coordinated_replacements(
    input: &AnalyzeAllInput,
    syn: &mut shared::AnalyzeSynapsesResult,
    neuron: &mut shared::AnalyzeNeuronsResult,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Issue #943: Borrow UUID strings from input instead of cloning.
    let mut direct_synapse_weight: HashMap<(&str, &str), f32> = HashMap::new();
    for s in &input.creature.synapses {
        direct_synapse_weight.insert((s.from_uuid.as_str(), s.to_uuid.as_str()), s.weight);
    }

    let mut kept_neurons: Vec<CandidateNeuronJson> =
        Vec::with_capacity(neuron.helpful_neurons.len());
    let mut replacements: Vec<CoordinatedStructuralCandidateJson> = Vec::new();

    for candidate in &neuron.helpful_neurons {
        let key = (
            candidate.source_neuron_uuid.as_str(),
            candidate.target_neuron_uuid.as_str(),
        );
        let Some(&old_weight) = direct_synapse_weight.get(&key) else {
            kept_neurons.push(candidate.clone());
            continue;
        };

        let new_neuron_uuid = synapse::deterministic_coordinated_neuron_uuid(
            &candidate.source_neuron_uuid,
            &candidate.target_neuron_uuid,
            &candidate.squash,
            candidate.incoming_weight,
            candidate.outgoing_weight,
            candidate.bias,
        );

        let replace_params = synapse::ReplaceSynapseParams {
            cache: shared_cache.as_ref(),
            source_uuid: &candidate.source_neuron_uuid,
            target_uuid: &candidate.target_neuron_uuid,
            old_weight,
            incoming_weight: candidate.incoming_weight,
            outgoing_weight: candidate.outgoing_weight,
            bias: candidate.bias,
            squash: &candidate.squash,
        };
        let mut expected_gain =
            synapse::expected_gain_replace_synapse_with_hidden_neuron(&replace_params)
                .unwrap_or(candidate.expected_creature_score_gain);

        // Apply a conservative impact discount when the target is hidden.
        // This mirrors the synapse/neurons discounting semantics without requiring deep graph analysis.
        let is_target_output = input
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == candidate.target_neuron_uuid)
            .is_some_and(|n| n.neuron_type == "output");
        if !is_target_output {
            expected_gain *= 0.1;
        }

        replacements.push(CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                    to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                },
                CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: new_neuron_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: candidate.squash.clone(),
                    bias: candidate.bias,
                    insert_before_neuron_uuid: Some(candidate.target_neuron_uuid.clone()),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                    to_neuron_uuid: new_neuron_uuid.clone(),
                    weight: candidate.incoming_weight,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: new_neuron_uuid,
                    to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                    weight: candidate.outgoing_weight,
                },
            ],
            expected_creature_score_gain: expected_gain,
            comment: Some(format!(
                "Coordinated replacement: remove synapse and insert {} hidden neuron",
                candidate.squash
            )),
        });
    }

    // Keep non-replacement add-neurons unchanged.
    apply_kept_neuron_candidates(neuron, kept_neurons);

    merge_coordinated_structural_replacements(
        syn,
        replacements,
        input.max_synapse_candidates,
        input.analysis_deadline_ms.is_some(),
    );
}
