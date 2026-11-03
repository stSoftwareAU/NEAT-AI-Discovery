//! Activation-error monotonicity detection module (Issue #643).
//!
//! For a well-functioning hidden neuron, the relationship between activation and
//! output error should be monotonic — higher activation consistently correlates
//! with either lower or higher error. A non-monotonic relationship (where activation
//! increases produce **both** error improvements and error worsenings) indicates the
//! neuron is encoding contradictory information and should be split or restructured.
//!
//! ## Detection Method
//!
//! For each hidden neuron:
//! 1. Sort observations by activation value
//! 2. Compute Spearman's rank correlation between activation rank and error
//! 3. Flag neurons where |rho| is below the monotonicity threshold
//!
//! ## Distinction from Related Modules
//!
//! - `noise_signal.rs` — measures variance ratio (noise), not directional consistency
//! - `gradient_discovery.rs` — analyses gradients for existing synapses, not
//!   neuron activation-error relationships
//!
//! ## Recommended Actions
//!
//! - `addNeuron`: Split the non-monotonic neuron so each sub-neuron handles one
//!   direction of the activation-error mapping
//! - `changeSquash`: Change the activation function to better fit the data
//!
//! ## Quantised `{0, 1}` Error Regime (Issue #1247)
//!
//! Spearman's rank correlation requires the error series to carry
//! ordinal information. Under `CATEGORICAL_ERROR`, output errors are
//! quantised misclassification flags (`0` or `1`) and the rank vector
//! collapses to two tied groups, producing a near-zero rho regardless
//! of any real activation-error relationship. Detecting "non-monotonic"
//! on that signal would flag every `CATEGORICAL_ERROR`-driven hidden
//! neuron.
//!
//! When [`is_quantised_zero_one`] reports the regime, this detector
//! skips the affected neuron and emits no candidate. The decision is
//! made per-neuron — a creature trained under a continuous cost still
//! exercises the full detector — and is documented as the "skip
//! cleanly with a diagnostic" branch from Issue #1247's acceptance
//! criteria.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::stats::spearman_rank_correlation;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DETECTION;
use crate::analysis::detection::helpers::{existing_neuron_uuids, first_free_neuron_uuid};
use crate::analysis::quantised_error::is_quantised_zero_one;
use crate::analysis::utils::deadline_passed;

/// Minimum |rho| below which a neuron is considered non-monotonic.
/// Spearman's rho ranges from -1.0 (perfectly decreasing) to +1.0 (perfectly increasing).
/// Values near zero indicate no monotonic relationship.
const MONOTONICITY_THRESHOLD: f32 = 0.3;

/// Result of detecting a non-monotonic neuron.
#[derive(Debug, Clone)]
pub struct MonotonicityCandidate {
    /// UUID of the neuron with non-monotonic activation-error relationship.
    pub neuron_uuid: String,
    /// Spearman's rank correlation between activation and error.
    /// Values near zero indicate strong non-monotonicity.
    pub monotonicity_score: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from restructuring this neuron.
    pub estimated_improvement: f32,
}

/// Detect neurons with non-monotonic activation-error relationships.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `MonotonicityCandidate` for neurons with non-monotonic activation-error
/// relationships, sorted by estimated improvement (best first).
pub fn detect_non_monotonic_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<MonotonicityCandidate> {
    // Only consider hidden neurons
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        let records = records.as_ref();
        if !hidden_uuids.contains(uuid.as_str()) {
            continue;
        }

        if records.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        // Extract activation and error pairs
        let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
        let errors: Vec<f32> = records
            .iter()
            .filter_map(|r| r.errors.first().copied())
            .collect();

        if errors.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        // Use the minimum length if activations and errors differ in count
        let n = activations.len().min(errors.len());
        let activations = &activations[..n];
        let errors = &errors[..n];

        // Use absolute error values for monotonicity analysis
        let abs_errors: Vec<f32> = errors.iter().map(|e| e.abs()).collect();

        // Skip neurons whose errors are quantised `{0, 1}` flags
        // (CATEGORICAL_ERROR regime, Issue #1247). Rank-correlation on
        // two tied groups is meaningless and would mass-flag every
        // hidden neuron as non-monotonic.
        if is_quantised_zero_one(&abs_errors) {
            continue;
        }

        let rho = spearman_rank_correlation(activations, &abs_errors);

        // Flag if |rho| is below threshold (non-monotonic)
        if rho.abs() >= MONOTONICITY_THRESHOLD {
            continue;
        }

        // Estimated improvement: lower |rho| and more samples → higher confidence
        let non_monotonicity = 1.0 - rho.abs();
        let sample_factor = (n as f32 / 1000.0).min(1.0);
        let estimated_improvement = 0.001 * (non_monotonicity * 0.7 + sample_factor * 0.3);

        candidates.push(MonotonicityCandidate {
            neuron_uuid: uuid.clone(),
            monotonicity_score: rho,
            sample_count: n,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert non-monotonic neuron candidates into coordinated structural candidates.
///
/// Each non-monotonic neuron produces either:
/// - An `AddNeuron` candidate (to split the neuron so each handles one direction)
/// - A `ChangeSquash` candidate (to try a different activation function)
///
/// Which candidate is produced depends on the creature's topology — if the neuron
/// has downstream synapses, we prefer adding a neuron to split the workload;
/// otherwise we suggest changing the activation function. Its synthetic UUID
/// is salted when needed to avoid existing neuron UUIDs (Issue #2412).
///
/// Issue #2479: no deadline — equivalent to calling
/// [`non_monotonic_neurons_to_coordinated_candidates_with_deadline`] with a `None`
/// deadline.
pub fn non_monotonic_neurons_to_coordinated_candidates(
    candidates: &[MonotonicityCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    non_monotonic_neurons_to_coordinated_candidates_with_deadline(candidates, creature, &None)
}

/// Deadline-aware variant of [`non_monotonic_neurons_to_coordinated_candidates`]
/// (Issue #2479).
///
/// Builds the uuid → neuron lookup and the from-/to-adjacency maps once up front
/// (rather than scanning `creature.neurons` and `creature.synapses` twice per
/// candidate) and checks `deadline` once per candidate, so a hung or oversized
/// conversion can be stopped mid-flight, returning the coordinated candidates
/// built so far.
pub fn non_monotonic_neurons_to_coordinated_candidates_with_deadline(
    candidates: &[MonotonicityCandidate],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    non_monotonic_neurons_to_coordinated_candidates_with_observer(
        candidates,
        creature,
        deadline,
        &mut || {},
    )
}

fn non_monotonic_neurons_to_coordinated_candidates_with_observer(
    candidates: &[MonotonicityCandidate],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
    work_observer: &mut impl FnMut(),
) -> Vec<CoordinatedStructuralCandidateJson> {
    let existing = existing_neuron_uuids(creature);

    // Issue #2479: build the uuid -> squash lookup once rather than scanning
    // `creature.neurons` per candidate (was O(C·N)). `entry(..).or_insert(..)`
    // preserves the original `find`'s first-match semantics when several
    // neurons share a uuid.
    let mut squash_by_uuid: HashMap<&str, &str> = HashMap::new();
    for n in &creature.neurons {
        work_observer();
        squash_by_uuid
            .entry(n.uuid.as_str())
            .or_insert(n.squash.as_str());
    }

    // Issue #2479: from- and to-adjacency maps, filled in synapse order so each
    // per-candidate list keeps the order the original `filter` produced.
    let mut outgoing_by_uuid: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    let mut incoming_by_uuid: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    for synapse in &creature.synapses {
        work_observer();
        outgoing_by_uuid
            .entry(synapse.from_uuid.as_str())
            .or_default()
            .push(synapse);
        incoming_by_uuid
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        if deadline_passed(deadline) {
            break;
        }

        // Find the neuron's current squash function (Issue #2479: O(1) map lookup)
        let current_squash = lookup_squash(&squash_by_uuid, c.neuron_uuid.as_str(), work_observer)
            .unwrap_or("LOGISTIC");

        // Find synapses going out of this neuron
        let outgoing: &[&crate::SynapseJson] =
            lookup_synapses(&outgoing_by_uuid, c.neuron_uuid.as_str(), work_observer);

        // Find synapses coming into this neuron
        let incoming: &[&crate::SynapseJson] =
            lookup_synapses(&incoming_by_uuid, c.neuron_uuid.as_str(), work_observer);

        // Strategy: if the neuron has outgoing connections, add a parallel neuron
        // to split the workload. Otherwise, try changing the activation function.
        if !outgoing.is_empty() && !incoming.is_empty() {
            // AddNeuron: create a new neuron inserted before the target of the
            // first outgoing synapse, with a complementary activation function
            let target_uuid = &outgoing[0].to_uuid;
            let new_uuid = first_free_neuron_uuid(&existing, |salt| {
                if salt == 0 {
                    format!("split-{}", c.neuron_uuid)
                } else {
                    format!("split-{}-{salt}", c.neuron_uuid)
                }
            });
            let new_squash = suggest_complementary_squash(current_squash);

            let mut operations = vec![CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: new_uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: new_squash,
                bias: 0.0,
                insert_before_neuron_uuid: Some(target_uuid.clone()),
            }];

            // Connect the first incoming source to the new neuron
            let source_uuid = &incoming[0].from_uuid;
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: source_uuid.clone(),
                to_neuron_uuid: new_uuid.clone(),
                weight: incoming[0].weight * 0.5,
            });

            // Connect new neuron to the first outgoing target
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: new_uuid,
                to_neuron_uuid: target_uuid.clone(),
                weight: outgoing[0].weight * 0.5,
            });

            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                operations,
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Non-monotonic neuron {}: rho {:.3}, {} samples → add parallel neuron to split contradictory activation-error mapping",
                    c.neuron_uuid, c.monotonicity_score, c.sample_count
                )),
            });
        } else {
            // ChangeSquash: try a different activation function
            let new_squash = suggest_complementary_squash(current_squash);
            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: c.neuron_uuid.clone(),
                    squash: new_squash,
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Non-monotonic neuron {}: rho {:.3}, {} samples → change squash to improve activation-error mapping",
                    c.neuron_uuid, c.monotonicity_score, c.sample_count
                )),
            });
        }
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}

/// Suggest a complementary activation function for a non-monotonic neuron.
///
/// The idea is to try a function with different saturation and linearity
/// characteristics that may better fit the data.
fn suggest_complementary_squash(current: &str) -> String {
    match current {
        "LOGISTIC" => "TANH".to_string(),
        "TANH" => "RELU".to_string(),
        "RELU" => "LOGISTIC".to_string(),
        "IDENTITY" => "TANH".to_string(),
        "SOFTSIGN" => "LOGISTIC".to_string(),
        "SOFTPLUS" => "TANH".to_string(),
        _ => "TANH".to_string(),
    }
}

fn lookup_squash<'a>(
    squash_by_uuid: &'a HashMap<&'a str, &'a str>,
    neuron_uuid: &str,
    work_observer: &mut impl FnMut(),
) -> Option<&'a str> {
    work_observer();
    squash_by_uuid.get(neuron_uuid).copied()
}

fn lookup_synapses<'a>(
    adjacency: &'a HashMap<&'a str, Vec<&'a crate::SynapseJson>>,
    neuron_uuid: &str,
    work_observer: &mut impl FnMut(),
) -> &'a [&'a crate::SynapseJson] {
    work_observer();
    match adjacency.get(neuron_uuid) {
        Some(synapses) => synapses.as_slice(),
        None => &[],
    }
}

#[cfg(test)]
mod issue_2479_tests {
    use super::*;

    /// An input `in`, `count` hidden neurons `h{i}` each fed by `in` and feeding
    /// `out`, and an output `out`. Every hidden neuron has both incoming and
    /// outgoing synapses, so each yields an `AddNeuron` candidate.
    fn split_candidate_fixture(count: usize) -> (CreatureJson, Vec<MonotonicityCandidate>) {
        let mut neurons = vec![crate::NeuronJson {
            uuid: "in".to_string(),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }];
        let mut synapses = Vec::new();
        let mut candidates = Vec::new();
        for i in 0..count {
            let uuid = format!("h{i}");
            neurons.push(crate::NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: 0.0,
            });
            synapses.push(crate::SynapseJson {
                from_uuid: "in".to_string(),
                to_uuid: uuid.clone(),
                weight: 0.5,
                synapse_type: None,
            });
            synapses.push(crate::SynapseJson {
                from_uuid: uuid.clone(),
                to_uuid: "out".to_string(),
                weight: 1.0,
                synapse_type: None,
            });
            candidates.push(MonotonicityCandidate {
                neuron_uuid: uuid,
                monotonicity_score: 0.0,
                sample_count: 30,
                estimated_improvement: 0.001,
            });
        }
        neurons.push(crate::NeuronJson {
            uuid: "out".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        let creature = CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: 1,
        };
        (creature, candidates)
    }

    #[test]
    fn adjacency_and_squash_work_scales_linearly_at_n_and_4n() {
        const N: usize = 128;

        let measured_work = |count: usize| {
            let (creature, candidates) = split_candidate_fixture(count);
            let mut work = 0;
            let result = non_monotonic_neurons_to_coordinated_candidates_with_observer(
                &candidates,
                &creature,
                &None,
                &mut || work += 1,
            );
            assert_eq!(result.len(), count);
            work
        };

        // Index build visits every neuron (count + 2) once and every synapse (2 *
        // count) once; each candidate performs one squash lookup and two adjacency
        // lookups.
        let small_work = measured_work(N);
        let large_work = measured_work(N * 4);
        assert_eq!(small_work, (N + 2) + 2 * N + 3 * N);
        assert_eq!(large_work, (N * 4 + 2) + 2 * N * 4 + 3 * N * 4);
        assert_eq!(large_work - 2, (small_work - 2) * 4);
    }

    /// The deadline is checked once per candidate, and on expiry the candidates
    /// already built are returned rather than discarded. The scripted sequence lets
    /// the first candidate through and stops at the second, so hoisting the check
    /// out of the loop (all three built) or replacing `break` with an empty return
    /// (none built) both fail this test.
    #[test]
    #[serial_test::serial]
    fn deadline_mid_conversion_returns_the_candidates_built_so_far() {
        use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;

        let (creature, candidates) = split_candidate_fixture(3);

        // Positive control: with no override every candidate converts.
        let control = non_monotonic_neurons_to_coordinated_candidates_with_deadline(
            &candidates,
            &creature,
            &None,
        );
        assert_eq!(
            control.len(),
            3,
            "fixture must yield one candidate per non-monotonic neuron"
        );

        let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
        let partial = non_monotonic_neurons_to_coordinated_candidates_with_deadline(
            &candidates,
            &creature,
            &None,
        );

        assert_eq!(
            partial.len(),
            1,
            "the deadline must stop the conversion at the second candidate and keep the first"
        );
        let split_uuid = partial[0].operations.iter().find_map(|op| match op {
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => Some(neuron_uuid.clone()),
            _ => None,
        });
        assert_eq!(
            split_uuid.as_deref(),
            Some("split-h0"),
            "the kept candidate must be the first non-monotonic neuron's"
        );
    }
}
