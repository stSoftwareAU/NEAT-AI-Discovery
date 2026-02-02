//! Bottleneck neuron detection module (Issue #343).
//!
//! See `docs/DISCOVERY_TYPES.md` § "Bottleneck Neuron Detection" for full documentation.
//!
//! Identifies hidden neurons that form information bottlenecks — single points where
//! many input signals converge through one neuron before reaching outputs. A bottleneck
//! limits the network's ability to represent complex input combinations because one
//! neuron's activation range must encode all upstream information.
//!
//! ## Detection Criteria
//!
//! A neuron is a "bottleneck" if:
//! 1. **High fan-in**: Many incoming connections (≥ `MIN_FAN_IN_FOR_BOTTLENECK`).
//! 2. **High fan-in / fan-out ratio**: Significantly more inputs than outputs.
//! 3. **Error concentration**: The neuron carries a disproportionate share of output error.
//! 4. **Not an output neuron**: Output neurons are natural convergence points and are excluded.
//!
//! ## Recommended Actions
//!
//! When a bottleneck is detected, we recommend:
//! 1. **Add parallel neuron**: Create a new hidden neuron sharing a subset of inputs/outputs
//!    to increase capacity at the bottleneck.
//! 2. **Add bypass synapse**: Add a direct connection from an upstream neuron to a downstream
//!    neuron, reducing dependency on the bottleneck.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `AddNeuron` and/or
//! `AddSynapse` operations.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable bottleneck detection.
const MIN_SAMPLES_FOR_BOTTLENECK: usize = 20;

/// Minimum fan-in to consider a hidden neuron as a potential bottleneck.
/// Neurons with fewer incoming connections are unlikely to be true bottlenecks.
const MIN_FAN_IN_FOR_BOTTLENECK: usize = 3;

/// Minimum fan-in / fan-out ratio to consider the neuron a bottleneck.
/// A ratio of 2.0 means the neuron has at least twice as many inputs as outputs.
const MIN_FAN_IN_FAN_OUT_RATIO: f32 = 2.0;

/// Result of detecting a bottleneck neuron.
#[derive(Debug, Clone)]
pub struct BottleneckNeuronCandidate {
    /// UUID of the bottleneck neuron.
    pub neuron_uuid: String,
    /// Number of incoming synapses (fan-in).
    pub fan_in: usize,
    /// Number of outgoing synapses (fan-out).
    pub fan_out: usize,
    /// Fraction of total error flowing through this neuron (0.0 to 1.0).
    pub error_contribution_ratio: f32,
    /// Overall bottleneck severity score (higher = worse bottleneck).
    pub bottleneck_score: f32,
    /// Estimated creature score improvement from resolving the bottleneck.
    pub estimated_improvement: f32,
    /// Recommended structural actions (e.g., "addParallelNeuron", "addBypassSynapse").
    pub recommended_actions: Vec<String>,
    /// UUIDs of neurons connected upstream (inputs to the bottleneck).
    pub upstream_uuids: Vec<String>,
    /// UUIDs of neurons connected downstream (outputs from the bottleneck).
    pub downstream_uuids: Vec<String>,
}

/// Detect bottleneck neurons from the creature topology and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `BottleneckNeuronCandidate` for neurons that are bottlenecks,
/// sorted by estimated improvement (best first).
pub fn detect_bottleneck_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BottleneckNeuronCandidate> {
    // Build topology maps
    let mut fan_in_map: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut fan_out_map: HashMap<&str, Vec<&str>> = HashMap::new();

    for synapse in &creature.synapses {
        fan_in_map
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse.from_uuid.as_str());
        fan_out_map
            .entry(synapse.from_uuid.as_str())
            .or_default()
            .push(synapse.to_uuid.as_str());
    }

    // Identify hidden neurons only
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    // Compute total error across all recorded neurons for normalisation
    let total_error: f32 = neuron_records
        .iter()
        .flat_map(|(_, records)| records.iter())
        .flat_map(|r| r.errors.iter())
        .map(|e| e.abs())
        .sum();

    let mut candidates = Vec::new();

    for uuid in &hidden_uuids {
        let fan_in_list = fan_in_map.get(uuid).cloned().unwrap_or_default();
        let fan_out_list = fan_out_map.get(uuid).cloned().unwrap_or_default();

        let fan_in = fan_in_list.len();
        let fan_out = fan_out_list.len();

        // Skip if fan-in is too low — not a bottleneck
        if fan_in < MIN_FAN_IN_FOR_BOTTLENECK {
            continue;
        }

        // Skip if fan-out is zero (dead end, different problem)
        if fan_out == 0 {
            continue;
        }

        // Check fan-in / fan-out ratio
        let ratio = fan_in as f32 / fan_out as f32;
        if ratio < MIN_FAN_IN_FAN_OUT_RATIO {
            continue;
        }

        // Check records
        let records = match records_map.get(uuid) {
            Some(r) if r.len() >= MIN_SAMPLES_FOR_BOTTLENECK => *r,
            _ => continue,
        };

        // Compute error contribution ratio for this neuron
        let neuron_error: f32 = records
            .iter()
            .flat_map(|r| r.errors.iter())
            .map(|e| e.abs())
            .sum();

        let error_contribution_ratio = if total_error > 0.0 {
            neuron_error / total_error
        } else {
            0.0
        };

        // Compute bottleneck score: combines topology and error concentration
        //
        // Topology component: how much information is squeezed
        //   - fan_in / fan_out gives the compression ratio
        //   - We use log2 to dampen extreme ratios
        let topology_score = (ratio).ln_1p() / (10.0_f32).ln_1p();

        // Error component: how much error flows through this neuron
        let error_score = error_contribution_ratio;

        // Combined score (weighted average)
        let bottleneck_score = 0.6 * topology_score + 0.4 * error_score;

        // Estimated improvement: conservative estimate based on bottleneck severity
        // Widening a bottleneck with fan-in=5 and fan-out=1 could allow the network
        // to represent more complex combinations.
        let estimated_improvement = bottleneck_score * 0.01;

        if estimated_improvement <= 0.0 {
            continue;
        }

        // Determine recommended actions
        let mut recommended_actions = Vec::new();
        recommended_actions.push("addParallelNeuron".to_string());

        // If the bottleneck connects to outputs, bypass synapses can help
        if fan_in > 3 {
            recommended_actions.push("addBypassSynapse".to_string());
        }

        candidates.push(BottleneckNeuronCandidate {
            neuron_uuid: uuid.to_string(),
            fan_in,
            fan_out,
            error_contribution_ratio,
            bottleneck_score,
            estimated_improvement,
            recommended_actions,
            upstream_uuids: fan_in_list.iter().map(|s| s.to_string()).collect(),
            downstream_uuids: fan_out_list.iter().map(|s| s.to_string()).collect(),
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Generate a deterministic UUID for a parallel bottleneck neuron.
fn bottleneck_parallel_neuron_uuid(bottleneck_uuid: &str, index: usize) -> String {
    let key = format!("bottleneck-parallel|{bottleneck_uuid}|{index}");
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("bp-{hash:016x}")
}

/// Convert bottleneck neuron candidates into coordinated structural candidates.
///
/// Each bottleneck neuron may produce one or more candidates:
/// 1. **Add parallel neuron**: A new hidden neuron sharing a subset of inputs and outputs
///    to widen the bottleneck.
/// 2. **Add bypass synapse**: A direct connection from an upstream neuron to a downstream
///    neuron, reducing dependency on the bottleneck.
///
/// # Arguments
/// * `candidates` - Detected bottleneck neurons.
/// * `creature` - The creature topology (needed for existing synapse weights).
pub fn bottleneck_neurons_to_coordinated_candidates(
    candidates: &[BottleneckNeuronCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Build synapse weight lookup
    let synapse_weights: HashMap<(&str, &str), f32> = creature
        .synapses
        .iter()
        .map(|s| ((s.from_uuid.as_str(), s.to_uuid.as_str()), s.weight))
        .collect();

    // Build existing synapse set for checking if bypass already exists
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    let mut results = Vec::new();

    for c in candidates {
        // Candidate 1: Add parallel neuron
        // Create a new hidden neuron that receives a subset of the bottleneck's inputs
        // and feeds the same outputs. This widens the information channel.
        if c.recommended_actions
            .contains(&"addParallelNeuron".to_string())
        {
            let new_uuid = bottleneck_parallel_neuron_uuid(&c.neuron_uuid, 0);

            // Find the bottleneck neuron's squash function
            let squash = creature
                .neurons
                .iter()
                .find(|n| n.uuid == c.neuron_uuid)
                .map(|n| n.squash.clone())
                .unwrap_or_else(|| "TANH".to_string());

            let mut operations = Vec::new();

            // Add the new parallel neuron (placed before the bottleneck)
            operations.push(CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: new_uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: squash.clone(),
                bias: 0.0,
                insert_before_neuron_uuid: Some(c.neuron_uuid.clone()),
            });

            // Connect a subset of upstream neurons to the new parallel neuron
            // Use the top half of upstream connections (by weight magnitude)
            let mut upstream_with_weights: Vec<(&str, f32)> = c
                .upstream_uuids
                .iter()
                .map(|u| {
                    let w = synapse_weights
                        .get(&(u.as_str(), c.neuron_uuid.as_str()))
                        .copied()
                        .unwrap_or(0.1);
                    (u.as_str(), w)
                })
                .collect();
            upstream_with_weights.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take the top half of upstream connections
            let half = (upstream_with_weights.len() / 2).max(1);
            for (upstream_uuid, weight) in upstream_with_weights.iter().take(half) {
                operations.push(CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: upstream_uuid.to_string(),
                    to_neuron_uuid: new_uuid.clone(),
                    weight: weight * 0.5, // Start with scaled-down weights
                });
            }

            // Connect the new neuron to all downstream neurons
            for downstream_uuid in &c.downstream_uuids {
                let existing_weight = synapse_weights
                    .get(&(c.neuron_uuid.as_str(), downstream_uuid.as_str()))
                    .copied()
                    .unwrap_or(0.1);
                operations.push(CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: new_uuid.clone(),
                    to_neuron_uuid: downstream_uuid.clone(),
                    weight: existing_weight * 0.5, // Start with scaled-down weight
                });
            }

            results.push(CoordinatedStructuralCandidateJson {
                operations,
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Bottleneck neuron {}: fan-in={}, fan-out={} (ratio {:.1}) → add parallel {} neuron to widen information flow",
                    c.neuron_uuid, c.fan_in, c.fan_out, c.fan_in as f32 / c.fan_out as f32, squash
                )),
            });
        }

        // Candidate 2: Add bypass synapses
        // Connect upstream neurons directly to downstream neurons to reduce bottleneck dependency.
        // Only add bypasses that don't already exist.
        if c.recommended_actions
            .contains(&"addBypassSynapse".to_string())
        {
            // Pick the upstream neuron with highest weighted connection to the bottleneck
            let best_upstream = c.upstream_uuids.iter().max_by(|a, b| {
                let wa = synapse_weights
                    .get(&(a.as_str(), c.neuron_uuid.as_str()))
                    .copied()
                    .unwrap_or(0.0)
                    .abs();
                let wb = synapse_weights
                    .get(&(b.as_str(), c.neuron_uuid.as_str()))
                    .copied()
                    .unwrap_or(0.0)
                    .abs();
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(upstream_uuid) = best_upstream {
                for downstream_uuid in &c.downstream_uuids {
                    // Skip if bypass already exists
                    if existing_synapses
                        .contains(&(upstream_uuid.as_str(), downstream_uuid.as_str()))
                    {
                        continue;
                    }

                    let upstream_weight = synapse_weights
                        .get(&(upstream_uuid.as_str(), c.neuron_uuid.as_str()))
                        .copied()
                        .unwrap_or(0.1);
                    let downstream_weight = synapse_weights
                        .get(&(c.neuron_uuid.as_str(), downstream_uuid.as_str()))
                        .copied()
                        .unwrap_or(0.1);
                    let bypass_weight = upstream_weight * downstream_weight * 0.5;

                    results.push(CoordinatedStructuralCandidateJson {
                        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                            from_neuron_uuid: upstream_uuid.clone(),
                            to_neuron_uuid: downstream_uuid.clone(),
                            weight: bypass_weight,
                        }],
                        expected_creature_score_gain: c.estimated_improvement * 0.7,
                        comment: Some(format!(
                            "Bottleneck bypass: {} → {} (bypassing bottleneck {}), fan-in={}, fan-out={}",
                            upstream_uuid, downstream_uuid, c.neuron_uuid, c.fan_in, c.fan_out
                        )),
                    });
                }
            }
        }
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
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    fn neuron(uuid: &str, ntype: &str, squash: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: squash.to_string(),
            bias: 0.0,
        }
    }

    fn syn(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    fn rec(uuid: &str, obs: u32, activation: f32) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: obs,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation,
            errors: vec![0.5],
        }
    }

    fn records_for(uuid: &str, count: usize) -> (String, Vec<DiscoverRecord>) {
        let recs = (0..count).map(|i| rec(uuid, i as u32, 0.5)).collect();
        (uuid.to_string(), recs)
    }

    /// Build a creature with a clear bottleneck: 4 inputs → h1 → 1 output.
    fn bottleneck_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
        let neurons = vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("i3", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o0", "output", "IDENTITY"),
        ];
        let synapses = vec![
            syn("i0", "h1", 1.0),
            syn("i1", "h1", 1.0),
            syn("i2", "h1", 1.0),
            syn("i3", "h1", 1.0),
            syn("h1", "o0", 1.0),
        ];
        let creature = CreatureJson {
            neurons,
            synapses,
            input: 4,
            output: 1,
        };
        let records = vec![records_for("h1", 30), records_for("o0", 30)];
        (creature, records)
    }

    // -----------------------------------------------------------------------
    // Detection criteria
    // -----------------------------------------------------------------------

    #[test]
    fn high_fan_in_low_fan_out_detected_as_bottleneck() {
        let (creature, records) = bottleneck_creature();
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].neuron_uuid, "h1");
        assert_eq!(candidates[0].fan_in, 4);
        assert_eq!(candidates[0].fan_out, 1);
    }

    #[test]
    fn bottleneck_recommends_add_parallel_neuron() {
        let (creature, records) = bottleneck_creature();
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(candidates[0]
            .recommended_actions
            .contains(&"addParallelNeuron".to_string()));
    }

    #[test]
    fn higher_fan_in_scores_higher() {
        // 5 inputs → h1 → 1 output should score higher than 3 inputs → h2 → 1 output
        let neurons = vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("i3", "input", "IDENTITY"),
            neuron("i4", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("h2", "hidden", "TANH"),
            neuron("o0", "output", "IDENTITY"),
        ];
        let synapses = vec![
            syn("i0", "h1", 1.0),
            syn("i1", "h1", 1.0),
            syn("i2", "h1", 1.0),
            syn("i3", "h1", 1.0),
            syn("i4", "h1", 1.0),
            syn("h1", "o0", 1.0),
            syn("i0", "h2", 1.0),
            syn("i1", "h2", 1.0),
            syn("i2", "h2", 1.0),
            syn("h2", "o0", 1.0),
        ];
        let creature = CreatureJson {
            neurons,
            synapses,
            input: 5,
            output: 1,
        };
        let records = vec![
            records_for("h1", 30),
            records_for("h2", 30),
            records_for("o0", 30),
        ];
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(candidates.len() >= 2);
        // First candidate (best) should be h1 with fan-in=5
        assert_eq!(candidates[0].neuron_uuid, "h1");
    }

    // -----------------------------------------------------------------------
    // Exclusion criteria
    // -----------------------------------------------------------------------

    #[test]
    fn output_neurons_excluded() {
        let neurons = vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("o0", "output", "IDENTITY"),
        ];
        let synapses = vec![
            syn("i0", "o0", 1.0),
            syn("i1", "o0", 1.0),
            syn("i2", "o0", 1.0),
        ];
        let creature = CreatureJson {
            neurons,
            synapses,
            input: 3,
            output: 1,
        };
        let records = vec![records_for("o0", 30)];
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(candidates.is_empty(), "output neurons should be excluded");
    }

    #[test]
    fn low_fan_in_excluded() {
        // fan-in=2, fan-out=1 — below MIN_FAN_IN_FOR_BOTTLENECK (3)
        let neurons = vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o0", "output", "IDENTITY"),
        ];
        let synapses = vec![
            syn("i0", "h1", 1.0),
            syn("i1", "h1", 1.0),
            syn("h1", "o0", 1.0),
        ];
        let creature = CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: 1,
        };
        let records = vec![records_for("h1", 30)];
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(candidates.is_empty(), "fan-in < 3 should be excluded");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn insufficient_samples_returns_empty() {
        let (creature, _) = bottleneck_creature();
        let records = vec![records_for("h1", 5)];
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(
            candidates.is_empty(),
            "fewer than MIN_SAMPLES should return empty"
        );
    }

    #[test]
    fn no_synapses_returns_empty() {
        let creature = CreatureJson {
            neurons: vec![neuron("h1", "hidden", "TANH")],
            synapses: vec![],
            input: 0,
            output: 0,
        };
        let records = vec![records_for("h1", 30)];
        let candidates = detect_bottleneck_neurons(&creature, &records);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    #[test]
    fn coordinated_candidates_contain_add_neuron_operations() {
        let (creature, records) = bottleneck_creature();
        let candidates = detect_bottleneck_neurons(&creature, &records);
        let coordinated = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature);
        assert!(
            !coordinated.is_empty(),
            "should produce coordinated candidates"
        );
        let has_add_neuron = coordinated.iter().any(|c| {
            c.operations
                .iter()
                .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }))
        });
        assert!(has_add_neuron, "should contain AddNeuron operation");
    }
}
