//! Dead neuron detection module (Issue #341).
//!
//! Identifies neurons that have become effectively dead (always outputting zero or
//! near-zero activation) and recommends their removal. Dead neurons waste computation
//! without contributing to the network's output.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Dead Neuron Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A neuron is "dead" if:
//! 1. **Near-zero activation**: Mean absolute activation < threshold (e.g., 1e-6) across
//!    all samples.
//! 2. **Zero variance**: Standard deviation of activation ≈ 0 (always outputs the same value).
//! 3. **Only hidden neurons**: Output and input neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When a dead neuron is detected, we recommend:
//! 1. **Remove the neuron**: Emit a `RemoveNeuron` operation via `CoordinatedStructuralCandidateJson`.
//!
//! Dead neurons consume GPU resources during both training and inference without contributing
//! useful information, so removal is the primary recommendation.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable dead neuron detection.
const MIN_SAMPLES_FOR_DEAD_DETECTION: usize = 20;

/// Threshold for mean absolute activation to consider a neuron dead.
/// Activations below this are effectively zero.
const DEAD_ACTIVATION_THRESHOLD: f32 = 1e-6;

/// Maximum activation standard deviation for a dead neuron.
/// If output varies significantly, the neuron is not truly dead.
const MAX_DEAD_STD_DEV: f32 = 1e-6;

/// Minimum fraction of samples where a neuron must be active to avoid being
/// flagged as dead. A neuron active on even a small fraction of samples with
/// meaningful activation is not dead.
const MIN_ACTIVE_FRACTION: f32 = 0.01;

/// Activation magnitude threshold for considering a single sample "active".
const ACTIVE_SAMPLE_THRESHOLD: f32 = 0.01;

/// Result of detecting a dead neuron.
#[derive(Debug, Clone)]
pub struct DeadNeuronCandidate {
    /// UUID of the dead neuron.
    pub neuron_uuid: String,
    /// Mean absolute activation across all samples.
    pub mean_abs_activation: f32,
    /// Standard deviation of activation across samples.
    pub activation_std_dev: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// UUIDs of output neurons reachable from this neuron.
    pub connected_outputs: Vec<String>,
    /// Confidence that removing this neuron is safe (0.0 to 1.0).
    pub removal_confidence: f32,
    /// Estimated creature score improvement from removing this neuron.
    pub estimated_improvement: f32,
}

/// Detect dead neurons from the creature topology and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `DeadNeuronCandidate` for neurons that are dead,
/// sorted by removal confidence (highest first).
pub fn detect_dead_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<DeadNeuronCandidate> {
    // Identify hidden neurons only
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    // Identify output neuron UUIDs
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build fan-out map for finding connected outputs
    let mut fan_out_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for synapse in &creature.synapses {
        fan_out_map
            .entry(synapse.from_uuid.as_str())
            .or_default()
            .push(synapse.to_uuid.as_str());
    }

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for uuid in &hidden_uuids {
        let Some(records) = records_map.get(uuid) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_DEAD_DETECTION {
            continue;
        }

        let n = records.len() as f32;

        // Compute mean absolute activation
        let sum_abs_activation: f32 = records.iter().map(|r| r.activation.abs()).sum();
        let mean_abs_activation = sum_abs_activation / n;

        // Compute activation standard deviation
        let mean_activation: f32 = records.iter().map(|r| r.activation).sum::<f32>() / n;
        let variance: f32 = records
            .iter()
            .map(|r| {
                let diff = r.activation - mean_activation;
                diff * diff
            })
            .sum::<f32>()
            / n;
        let activation_std_dev = variance.sqrt();

        // Check if neuron is dead: near-zero mean absolute activation AND low variance
        if mean_abs_activation >= DEAD_ACTIVATION_THRESHOLD {
            continue;
        }

        if activation_std_dev >= MAX_DEAD_STD_DEV {
            continue;
        }

        // Check active fraction: if the neuron fires meaningfully on even a small
        // fraction of samples, it is not dead (prevents false positives).
        let active_count = records
            .iter()
            .filter(|r| r.activation.abs() >= ACTIVE_SAMPLE_THRESHOLD)
            .count();
        let active_fraction = active_count as f32 / n;

        if active_fraction >= MIN_ACTIVE_FRACTION {
            continue;
        }

        // Find output neurons reachable from this neuron (BFS)
        let connected_outputs = find_connected_outputs(uuid, &fan_out_map, &output_uuids);

        // Compute removal confidence based on how dead the neuron is
        let confidence = compute_removal_confidence(mean_abs_activation, activation_std_dev, n);

        // Estimated improvement: removing a dead neuron saves computation.
        // The improvement is small but positive (reduced overhead).
        let estimated_improvement = confidence * 0.001;

        candidates.push(DeadNeuronCandidate {
            neuron_uuid: uuid.to_string(),
            mean_abs_activation,
            activation_std_dev,
            sample_count: records.len(),
            connected_outputs,
            removal_confidence: confidence,
            estimated_improvement,
        });
    }

    // Sort by removal confidence (highest first)
    candidates.sort_by(|a, b| {
        b.removal_confidence
            .partial_cmp(&a.removal_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Find output neurons reachable from a given neuron via BFS through the fan-out map.
fn find_connected_outputs<'a>(
    start_uuid: &str,
    fan_out_map: &HashMap<&'a str, Vec<&'a str>>,
    output_uuids: &HashSet<&str>,
) -> Vec<String> {
    let mut visited = HashSet::new();
    let mut queue = vec![start_uuid];
    let mut connected = Vec::new();

    while let Some(current) = queue.pop() {
        if !visited.insert(current.to_string()) {
            continue;
        }

        if let Some(downstream) = fan_out_map.get(current) {
            for &next in downstream {
                if output_uuids.contains(next) {
                    connected.push(next.to_string());
                }
                if !visited.contains(next) {
                    queue.push(next);
                }
            }
        }
    }

    connected.sort();
    connected.dedup();
    connected
}

/// Compute removal confidence based on activation statistics.
///
/// Higher confidence when:
/// - Mean absolute activation is closer to zero
/// - Standard deviation is closer to zero
/// - More samples were analysed
fn compute_removal_confidence(
    mean_abs_activation: f32,
    activation_std_dev: f32,
    sample_count: f32,
) -> f32 {
    // Base confidence from how dead the neuron is (closer to zero = higher confidence)
    let activation_factor = 1.0 - (mean_abs_activation / DEAD_ACTIVATION_THRESHOLD).min(1.0);

    // Variance factor (lower variance = higher confidence)
    let variance_factor = 1.0 - (activation_std_dev / MAX_DEAD_STD_DEV).min(1.0);

    // Sample size factor (more samples = higher confidence, plateaus at 1000)
    let sample_factor = (sample_count / 1000.0).min(1.0);

    // Combine: all factors contribute to confidence
    let raw_confidence = activation_factor * 0.4 + variance_factor * 0.4 + sample_factor * 0.2;

    // Scale to [0.5, 1.0] range since we already passed the threshold checks
    0.5 + raw_confidence * 0.5
}

/// Convert dead neuron candidates into coordinated structural candidates.
///
/// Each dead neuron produces a `RemoveNeuron` coordinated candidate.
/// The NEAT-AI controller will validate the removal through ablation testing
/// before actually applying it.
pub fn dead_neurons_to_coordinated_candidates(
    candidates: &[DeadNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: c.neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Dead neuron {}: mean abs activation {:.2e}, std dev {:.2e}, {} samples → remove to reduce wasted computation",
                c.neuron_uuid, c.mean_abs_activation, c.activation_std_dev, c.sample_count
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use crate::{NeuronJson, SynapseJson};

    fn make_neuron(uuid: &str, ntype: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        }
    }

    fn make_synapse(from: &str, to: &str) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight: 1.0,
            synapse_type: None,
        }
    }

    fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
        CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: 1,
        }
    }

    fn make_record(uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
        DiscoverRecord::new(obs_index, uuid.to_string(), None, activation, vec![0.01])
    }

    // ── compute_removal_confidence ─────────────────────────────────────

    #[test]
    fn perfect_dead_neuron_has_high_confidence() {
        // Zero activation, zero std dev, many samples → high confidence
        let confidence = compute_removal_confidence(0.0, 0.0, 1000.0);
        assert!(
            confidence > 0.8,
            "Perfectly dead neuron should have high confidence, got {confidence}"
        );
    }

    #[test]
    fn more_samples_increase_confidence() {
        let c_few = compute_removal_confidence(0.0, 0.0, 20.0);
        let c_many = compute_removal_confidence(0.0, 0.0, 500.0);
        assert!(
            c_many > c_few,
            "More samples should increase confidence: {c_many} vs {c_few}"
        );
    }

    #[test]
    fn confidence_never_exceeds_one() {
        let confidence = compute_removal_confidence(0.0, 0.0, 100_000.0);
        assert!(
            confidence <= 1.0,
            "Confidence should not exceed 1.0, got {confidence}"
        );
    }

    // ── find_connected_outputs ─────────────────────────────────────────

    #[test]
    fn finds_directly_connected_output() {
        let fan_out_map: HashMap<&str, Vec<&str>> = [("h-1", vec!["out-1"])].into_iter().collect();
        let output_uuids: HashSet<&str> = ["out-1"].into_iter().collect();

        let connected = find_connected_outputs("h-1", &fan_out_map, &output_uuids);
        assert_eq!(connected, vec!["out-1".to_string()]);
    }

    #[test]
    fn finds_transitively_connected_output() {
        let fan_out_map: HashMap<&str, Vec<&str>> = [("h-1", vec!["h-2"]), ("h-2", vec!["out-1"])]
            .into_iter()
            .collect();
        let output_uuids: HashSet<&str> = ["out-1"].into_iter().collect();

        let connected = find_connected_outputs("h-1", &fan_out_map, &output_uuids);
        assert_eq!(connected, vec!["out-1".to_string()]);
    }

    #[test]
    fn no_connected_outputs_returns_empty() {
        let fan_out_map: HashMap<&str, Vec<&str>> = HashMap::new();
        let output_uuids: HashSet<&str> = ["out-1"].into_iter().collect();

        let connected = find_connected_outputs("h-1", &fan_out_map, &output_uuids);
        assert!(connected.is_empty());
    }

    // ── detect_dead_neurons ────────────────────────────────────────────

    #[test]
    fn active_neuron_not_flagged_as_dead() {
        let creature = make_creature(
            vec![make_neuron("h-1", "hidden"), make_neuron("out-1", "output")],
            vec![make_synapse("h-1", "out-1")],
        );

        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "h-1".to_string(),
            (0..30).map(|i| make_record("h-1", i, 0.5)).collect(),
        )];

        let result = detect_dead_neurons(&creature, &records);
        assert!(
            result.is_empty(),
            "Active neuron should not be flagged as dead"
        );
    }

    #[test]
    fn output_neuron_never_flagged() {
        let creature = make_creature(vec![make_neuron("out-1", "output")], vec![]);

        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
            "out-1".to_string(),
            (0..30).map(|i| make_record("out-1", i, 0.0)).collect(),
        )];

        let result = detect_dead_neurons(&creature, &records);
        assert!(
            result.is_empty(),
            "Output neurons should never be flagged as dead"
        );
    }

    // ── dead_neurons_to_coordinated_candidates ─────────────────────────

    #[test]
    fn conversion_produces_remove_neuron_op() {
        let candidate = DeadNeuronCandidate {
            neuron_uuid: "dead-1".to_string(),
            mean_abs_activation: 0.0,
            activation_std_dev: 0.0,
            sample_count: 100,
            connected_outputs: vec!["out-1".to_string()],
            removal_confidence: 0.95,
            estimated_improvement: 0.001,
        };

        let coordinated = dead_neurons_to_coordinated_candidates(&[candidate]);
        assert_eq!(coordinated.len(), 1);
        assert!(matches!(
            &coordinated[0].operations[0],
            CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "dead-1"
        ));
    }
}
