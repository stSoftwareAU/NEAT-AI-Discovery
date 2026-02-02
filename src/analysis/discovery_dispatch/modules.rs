//! Issue #375: Concrete `DiscoveryModule` implementations for all 9 detection modules.
//!
//! Each struct wraps the module-specific detection and conversion logic behind
//! the generic `DiscoveryModule` trait, eliminating the boilerplate in `analyze_all()`.

use super::{collect_records_for_hidden_neurons, collect_records_for_uuids, DiscoveryModule};
use crate::analysis::cache::RecordCache;
use crate::analysis::{
    bottleneck, correlated_error, dead_neuron, dormant_synapse, multi_hop, opposing_synapse,
    oscillating_neuron, output_bias_drift, saturation,
};
use crate::{CoordinatedStructuralCandidateJson, CreatureJson};
use std::sync::Arc;

/// Helper: collect hidden neurons as `(uuid, squash, bias)` tuples from a creature.
fn collect_hidden_neurons(creature: &CreatureJson) -> Vec<(String, String, f32)> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Saturation Detection (Issue #342)
// ---------------------------------------------------------------------------

pub struct SaturationModule;

impl DiscoveryModule for SaturationModule {
    fn name(&self) -> &'static str {
        "saturation"
    }

    fn phase_name(&self) -> &'static str {
        "saturation_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let hidden_neurons = collect_hidden_neurons(creature);
        if hidden_neurons.is_empty() {
            return (0, Vec::new());
        }

        let neuron_records = collect_records_for_hidden_neurons(&hidden_neurons, shared_cache);
        let saturated = saturation::detect_saturated_neurons(&hidden_neurons, &neuron_records);
        if saturated.is_empty() {
            return (0, Vec::new());
        }

        let candidates = saturation::saturated_neurons_to_coordinated_candidates(&saturated);
        (saturated.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 2. Bottleneck Detection (Issue #343)
// ---------------------------------------------------------------------------

pub struct BottleneckModule;

impl DiscoveryModule for BottleneckModule {
    fn name(&self) -> &'static str {
        "bottleneck"
    }

    fn phase_name(&self) -> &'static str {
        "bottleneck_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let hidden_neurons = collect_hidden_neurons(creature);
        if hidden_neurons.is_empty() {
            return (0, Vec::new());
        }

        let neuron_records = collect_records_for_hidden_neurons(&hidden_neurons, shared_cache);
        let detected = bottleneck::detect_bottleneck_neurons(creature, &neuron_records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates =
            bottleneck::bottleneck_neurons_to_coordinated_candidates(&detected, creature);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 3. Dead Neuron Detection (Issue #341)
// ---------------------------------------------------------------------------

pub struct DeadNeuronModule;

impl DiscoveryModule for DeadNeuronModule {
    fn name(&self) -> &'static str {
        "dead neuron"
    }

    fn phase_name(&self) -> &'static str {
        "dead_neuron_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let hidden_neurons = collect_hidden_neurons(creature);
        if hidden_neurons.is_empty() {
            return (0, Vec::new());
        }

        let neuron_records = collect_records_for_hidden_neurons(&hidden_neurons, shared_cache);
        let detected = dead_neuron::detect_dead_neurons(creature, &neuron_records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates = dead_neuron::dead_neurons_to_coordinated_candidates(&detected);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 4. Correlated Error Detection (Issue #344)
// ---------------------------------------------------------------------------

pub struct CorrelatedErrorModule;

impl DiscoveryModule for CorrelatedErrorModule {
    fn name(&self) -> &'static str {
        "correlated error"
    }

    fn phase_name(&self) -> &'static str {
        "correlated_error_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let output_count = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .count();

        if output_count < 2 {
            return (0, Vec::new());
        }

        let neuron_uuids: Vec<String> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output" || n.neuron_type == "input")
            .map(|n| n.uuid.clone())
            .collect();

        let records = collect_records_for_uuids(&neuron_uuids, shared_cache);
        let groups = correlated_error::detect_correlated_error_patterns(creature, &records);
        if groups.is_empty() {
            return (0, Vec::new());
        }

        let candidates =
            correlated_error::correlated_errors_to_coordinated_candidates(&groups, creature);
        (groups.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 5. Multi-Hop Analysis (Issue #230)
// ---------------------------------------------------------------------------

pub struct MultiHopModule;

impl DiscoveryModule for MultiHopModule {
    fn name(&self) -> &'static str {
        "multi-hop"
    }

    fn phase_name(&self) -> &'static str {
        "multi_hop_analysis"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let hidden_neurons = collect_hidden_neurons(creature);
        if hidden_neurons.is_empty() {
            return (0, Vec::new());
        }

        let neuron_uuids: Vec<String> = creature.neurons.iter().map(|n| n.uuid.clone()).collect();
        let records = collect_records_for_uuids(&neuron_uuids, shared_cache);
        let detected = multi_hop::detect_multi_hop_candidates(creature, &records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates = multi_hop::multi_hop_to_coordinated_candidates(&detected, creature);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 6. Oscillating Neuron Detection (Issue #358)
// ---------------------------------------------------------------------------

pub struct OscillatingNeuronModule;

impl DiscoveryModule for OscillatingNeuronModule {
    fn name(&self) -> &'static str {
        "oscillating neuron"
    }

    fn phase_name(&self) -> &'static str {
        "oscillating_neuron_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let hidden_neurons = collect_hidden_neurons(creature);
        if hidden_neurons.is_empty() {
            return (0, Vec::new());
        }

        let neuron_records = collect_records_for_hidden_neurons(&hidden_neurons, shared_cache);
        let detected =
            oscillating_neuron::detect_oscillating_neurons(&hidden_neurons, &neuron_records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates =
            oscillating_neuron::oscillating_neurons_to_coordinated_candidates(&detected);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 7. Dormant Synapse Detection (Issue #359)
// ---------------------------------------------------------------------------

pub struct DormantSynapseModule;

impl DiscoveryModule for DormantSynapseModule {
    fn name(&self) -> &'static str {
        "dormant synapse"
    }

    fn phase_name(&self) -> &'static str {
        "dormant_synapse_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let source_uuids: Vec<String> = creature
            .synapses
            .iter()
            .map(|s| s.from_uuid.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let records = collect_records_for_uuids(&source_uuids, shared_cache);
        let detected = dormant_synapse::detect_dormant_synapses(creature, &records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates = dormant_synapse::dormant_synapses_to_coordinated_candidates(&detected);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 8. Opposing Synapse Detection (Issue #360)
// ---------------------------------------------------------------------------

pub struct OpposingSynapseModule;

impl DiscoveryModule for OpposingSynapseModule {
    fn name(&self) -> &'static str {
        "opposing synapse"
    }

    fn phase_name(&self) -> &'static str {
        "opposing_synapse_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let neuron_uuids: Vec<String> = creature.neurons.iter().map(|n| n.uuid.clone()).collect();
        let records = collect_records_for_uuids(&neuron_uuids, shared_cache);
        let detected = opposing_synapse::detect_opposing_synapses(creature, &records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates = opposing_synapse::opposing_synapses_to_coordinated_candidates(&detected);
        (detected.len(), candidates)
    }
}

// ---------------------------------------------------------------------------
// 9. Output Bias Drift Detection (Issue #361)
// ---------------------------------------------------------------------------

pub struct OutputBiasDriftModule;

impl DiscoveryModule for OutputBiasDriftModule {
    fn name(&self) -> &'static str {
        "output bias drift"
    }

    fn phase_name(&self) -> &'static str {
        "output_bias_drift_detection"
    }

    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        let output_uuids: Vec<String> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .map(|n| n.uuid.clone())
            .collect();

        let records = collect_records_for_uuids(&output_uuids, shared_cache);
        let detected = output_bias_drift::detect_output_bias_drift(creature, &records);
        if detected.is_empty() {
            return (0, Vec::new());
        }

        let candidates = output_bias_drift::output_bias_drift_to_coordinated_candidates(&detected);
        (detected.len(), candidates)
    }
}

/// Returns all 9 discovery modules in the standard dispatch order.
///
/// This order matches the original `analyze_all()` sequence:
/// saturation → bottleneck → dead neuron → correlated error → multi-hop →
/// oscillating neuron → dormant synapse → opposing synapse → output bias drift.
pub fn all_discovery_modules() -> Vec<Box<dyn DiscoveryModule>> {
    vec![
        Box::new(SaturationModule),
        Box::new(BottleneckModule),
        Box::new(DeadNeuronModule),
        Box::new(CorrelatedErrorModule),
        Box::new(MultiHopModule),
        Box::new(OscillatingNeuronModule),
        Box::new(DormantSynapseModule),
        Box::new(OpposingSynapseModule),
        Box::new(OutputBiasDriftModule),
    ]
}
