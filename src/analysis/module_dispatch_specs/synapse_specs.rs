//! Synapse-focused discovery module dispatch specs.
//!
//! Covers: dormant synapse, opposing synapse, weight coherence (ratio,
//! near-constant path, symmetric cancellation), noisy synapses, and
//! weight magnitude reset detection.

use std::sync::Arc;
use std::time::SystemTime;

use super::super::detection::{
    dormant_synapse, fanin_polarity_conflict, noise_signal, opposing_synapse,
    topology_cache::CreatureTopologyCache, weight_coherence, weight_magnitude_reset,
    weight_polarity_flip,
};
use super::super::{cache, discovery_dispatch};
use crate::types::SharedRecords;

/// Run the symmetric-cancellation pair scan for the production dispatch
/// closure (Issue #2347) and return the resulting candidates. Extracted from
/// the `discovery_spec!` closure so the deadline-forwarding and
/// partial-scan-detection logic are independently testable; a regression that
/// silently drops `deadline` (e.g. passing `&None` instead) would otherwise
/// only be caught by the absence of a timeout in production.
fn run_symmetric_cancellation(
    creature: &crate::CreatureJson,
    records: &[(String, SharedRecords)],
    topo: &CreatureTopologyCache,
    deadline: &Option<SystemTime>,
) -> Vec<weight_coherence::SymmetricCancellationCandidate> {
    let config = weight_coherence::WeightCoherenceConfig::default();
    let scan = weight_coherence::detect_symmetric_cancellation_with_deadline(
        creature,
        records,
        &config,
        Some(topo),
        deadline,
    );
    if symmetric_cancellation_scan_is_partial(&scan) {
        tracing::warn!(
            reason = ?scan.truncation,
            returned = scan.candidates.len(),
            skipped_high_fanin_targets = scan.skipped_high_fanin_targets,
            "Symmetric cancellation scan stopped early or skipped high fan-in targets; returning partial candidates."
        );
    }
    scan.candidates
}

/// True when a [`weight_coherence::SymmetricCancellationScan`] stopped before
/// visiting every target, or skipped a target entirely for exceeding the
/// fan-in cap — i.e. the candidates returned are a partial result, not the
/// complete scan.
fn symmetric_cancellation_scan_is_partial(
    scan: &weight_coherence::SymmetricCancellationScan,
) -> bool {
    scan.truncation.is_some() || scan.skipped_high_fanin_targets > 0
}

/// Append synapse-focused discovery module specs to the provided vector.
///
/// `deadline` (Issue #2347) is forwarded into the symmetric-cancellation pair
/// scan so it stops at the discovery deadline or on global cancellation.
pub(crate) fn append_synapse_specs(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
    deadline: Option<SystemTime>,
) {
    // Issue #359: Dormant synapse detection
    discovery_spec!(modules, "dormant synapse detection", "dormant_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_synapse_sources(&creature),
        detect: |records| dormant_synapse::detect_dormant_synapses(&creature, &records),
        convert: |detected| dormant_synapse::dormant_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #360: Opposing synapse detection
    discovery_spec!(modules, "opposing synapse detection", "opposing_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| opposing_synapse::detect_opposing_synapses(&creature, &records),
        convert: |detected| opposing_synapse::opposing_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #437: Weight coherence validation - incoherent weight ratios
    discovery_spec!(modules, "weight coherence ratio detection", "weight_coherence_ratio_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = weight_coherence::WeightCoherenceConfig::default();
            weight_coherence::detect_incoherent_weight_ratios(&creature, &records, &config, Some(&topo))
        },
        convert: |detected| weight_coherence::incoherent_ratios_to_coordinated_candidates(&detected),
    );

    // Issue #437: Weight coherence validation - near-constant output paths
    discovery_spec!(modules, "near-constant path detection", "near_constant_path_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature, topo = topo =>
        guard: hidden,
        records: cache.load_records_for_hidden(&hidden),
        detect: |records| {
            let config = weight_coherence::WeightCoherenceConfig::default();
            weight_coherence::detect_near_constant_paths(&creature, &records, &config, Some(&topo))
        },
        convert: |detected| weight_coherence::near_constant_paths_to_coordinated_candidates(&detected),
    );

    // Issue #437 / #2347: Weight coherence validation - symmetric weight
    // cancellation. The O(fan_in^2) pair scan is bounded by a fan-in cap, a
    // candidate ceiling and the discovery deadline; a truncated or partially
    // skipped scan is logged, not hidden.
    discovery_spec!(modules, "symmetric cancellation detection", "symmetric_cancellation_detection",
        cache = shared_cache, creature = creature, topo = topo =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| run_symmetric_cancellation(&creature, &records, &topo, &deadline),
        convert: |detected| weight_coherence::symmetric_cancellation_to_coordinated_candidates(&detected),
    );

    // Issue #434: Noise-to-signal ratio detection for synapses
    discovery_spec!(modules, "noisy synapse detection", "noisy_synapse_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| noise_signal::detect_noisy_synapses(&creature, &records),
        convert: |detected| noise_signal::noisy_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #550: Weight magnitude reset for stuck synapses (local minimum escape)
    discovery_spec!(modules, "weight magnitude reset detection", "weight_magnitude_reset_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| weight_magnitude_reset::detect_stuck_synapse_weight_resets(&creature, &records),
        convert: |detected| weight_magnitude_reset::stuck_synapses_to_coordinated_candidates(&detected),
    );

    // Issue #641: Fan-in weight polarity conflict detection
    discovery_spec!(modules, "fan-in polarity conflict detection", "fanin_polarity_conflict_detection",
        cache = shared_cache, hidden = hidden_neurons, creature = creature =>
        guard: hidden,
        records: cache.load_records_for_all_neurons(&creature),
        detect: |records| fanin_polarity_conflict::detect_fanin_polarity_conflicts(&creature, &records),
        convert: |detected| fanin_polarity_conflict::fanin_polarity_conflicts_to_coordinated_candidates(&detected, &creature),
    );

    // Issue #644: Weight polarity flip detection (gradient-weight sign disagreement)
    discovery_spec!(modules, "weight polarity flip detection", "weight_polarity_flip_detection",
        cache = shared_cache, creature = creature =>
        records: cache.load_records_for_all_neurons(&creature),
        guard_records,
        detect: |records| weight_polarity_flip::detect_weight_polarity_flip_candidates(&creature, &records),
        convert: |detected| weight_polarity_flip::polarity_flip_candidates_to_coordinated(&detected),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::recommendation::epistatic::ScanTruncation;
    use crate::types::DiscoverRecord;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    /// A single target ("target-1") with two opposite-sign, equal-magnitude
    /// incoming weights from perfectly correlated sources — the only pattern
    /// `detect_symmetric_cancellation_with_deadline` flags.
    fn cancellation_fixture() -> (Arc<CreatureJson>, Arc<CreatureTopologyCache>) {
        let creature = CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "target-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "input-1".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "target-1".to_string(),
                    weight: 5.0,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "input-1".to_string(),
                    to_uuid: "target-1".to_string(),
                    weight: -5.0,
                    synapse_type: None,
                },
            ],
            input: 2,
            output: 1,
        };
        let creature = Arc::new(creature);
        let topo = Arc::new(CreatureTopologyCache::new(&creature));
        (creature, topo)
    }

    /// Every neuron shares the same 25-sample activation pattern, so the two
    /// opposite-weighted sources above are perfectly correlated.
    #[allow(clippy::cast_precision_loss)] // Test fixture only; precision is irrelevant here.
    fn cancellation_cache() -> cache::RecordCache {
        cache::RecordCache::with_loader(
            "test.parquet",
            Arc::new(
                |_file: &str, uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> {
                    Ok((0..25u32)
                        .map(|idx| {
                            let activation = ((idx as f32) * 0.1).sin().tanh();
                            DiscoverRecord::new(
                                idx,
                                uuid.to_string(),
                                Some(activation),
                                activation,
                                vec![0.1],
                            )
                        })
                        .collect())
                },
            ),
        )
    }

    /// Issue #2347: the "symmetric cancellation detection" spec must stop
    /// before finding any candidate when the deadline has already elapsed —
    /// proving `deadline` (not `&None`) is actually forwarded into the scan.
    #[test]
    fn symmetric_cancellation_spec_returns_none_when_deadline_already_elapsed() {
        let (creature, topo) = cancellation_fixture();
        let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(vec![]);
        let shared_cache = Arc::new(cancellation_cache());

        let mut modules = Vec::new();
        append_synapse_specs(
            &mut modules,
            &creature,
            &hidden,
            &shared_cache,
            &topo,
            Some(SystemTime::UNIX_EPOCH),
        );
        let spec = modules
            .into_iter()
            .find(|s| s.module_name == "symmetric cancellation detection")
            .expect("symmetric cancellation detection spec must be present");
        let result = (spec.detect_fn)();
        assert!(
            result.is_none(),
            "an already-elapsed deadline must stop the scan before any candidate is \
             produced; the spec is dropping the deadline it was given"
        );
    }

    /// Same fixture, no deadline: the scan must run to completion and find
    /// the symmetric-cancellation candidate, so the elapsed-deadline
    /// assertion above can actually fail when the deadline is dropped.
    #[test]
    fn symmetric_cancellation_spec_returns_candidates_without_deadline() {
        let (creature, topo) = cancellation_fixture();
        let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(vec![]);
        let shared_cache = Arc::new(cancellation_cache());

        let mut modules = Vec::new();
        append_synapse_specs(&mut modules, &creature, &hidden, &shared_cache, &topo, None);
        let spec = modules
            .into_iter()
            .find(|s| s.module_name == "symmetric cancellation detection")
            .expect("symmetric cancellation detection spec must be present");
        let result = (spec.detect_fn)();
        assert!(
            result.is_some(),
            "fixture is a symmetric-cancellation pattern and must be detected when \
             nothing bounds the scan"
        );
        assert!(result.unwrap().detected_count > 0);
    }

    fn complete_scan() -> weight_coherence::SymmetricCancellationScan {
        weight_coherence::SymmetricCancellationScan {
            candidates: Vec::new(),
            truncation: None,
            skipped_high_fanin_targets: 0,
            pairs_correlated: 0,
            activation_maps_built: 0,
        }
    }

    #[test]
    fn scan_is_not_partial_when_complete() {
        assert!(!symmetric_cancellation_scan_is_partial(&complete_scan()));
    }

    #[test]
    fn scan_is_partial_when_deadline_passed() {
        let mut scan = complete_scan();
        scan.truncation = Some(ScanTruncation::DeadlinePassed);
        assert!(symmetric_cancellation_scan_is_partial(&scan));
    }

    #[test]
    fn scan_is_partial_when_candidate_ceiling_hit() {
        let mut scan = complete_scan();
        scan.truncation = Some(ScanTruncation::CandidateCeiling);
        assert!(symmetric_cancellation_scan_is_partial(&scan));
    }

    #[test]
    fn scan_is_partial_when_high_fanin_targets_skipped() {
        let mut scan = complete_scan();
        scan.skipped_high_fanin_targets = 1;
        assert!(symmetric_cancellation_scan_is_partial(&scan));
    }
}
