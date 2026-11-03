//! Hidden neuron operating-point analysis module (Issue #401).
//!
//! Hidden neurons may have a bias or incoming weight configuration that places
//! their operating point outside the "active zone" of their squash function.
//! For example:
//! - A LOGISTIC neuron with a very negative bias always outputs ~0 (wasting
//!   the sigmoid's S-curve).
//! - A TANH neuron where the pre-activation (`value`) is always in [-0.1, 0.1]
//!   — only using the linear region.
//!
//! ## Relationship to Other Detection Modules
//!
//! - **Restricted range** (Issue #399): analyses *activation* (post-squash)
//!   output range as a fraction of the theoretical bounds.
//! - **Saturation** (Issue #342): neurons stuck at the *bounds* of their
//!   activation function.
//! - **This module** (Issue #401): analyses the *pre-activation* (`value`)
//!   distribution against the squash function's active zone to measure how
//!   much of the dynamic range is actually being utilised.
//!
//! ## Detection Criteria
//!
//! A hidden neuron has an operating-point issue if:
//! 1. It uses a bounded squash function with a known active zone.
//! 2. `dynamic_range_utilisation` — the fraction of the squash function's
//!    dynamic range actually exercised by the observed pre-activation
//!    distribution — is below a configurable threshold (default: 20%).
//! 3. Sufficient samples with pre-activation values are available (≥ 20).
//!
//! ## Recommended Actions
//!
//! - `setBias` — shift the operating point into the active zone.
//! - `changeSquash` — switch to a function whose active zone matches the
//!   current operating point.
//! - `setWeight` — scale incoming weights to expand the pre-activation range.

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::analysis::utils::deadline_passed;

/// Configuration for operating-point analysis.
#[derive(Debug, Clone)]
pub struct OperatingPointConfig {
    /// Maximum dynamic range utilisation below which a neuron is flagged
    /// (default: 0.20 = 20%).
    pub utilisation_threshold: f32,
    /// Minimum samples with valid pre-activation values.
    pub min_samples: usize,
}

impl Default for OperatingPointConfig {
    fn default() -> Self {
        Self {
            utilisation_threshold: 0.20,
            min_samples: MIN_SAMPLES,
        }
    }
}

/// Result of detecting an operating-point issue on a hidden neuron.
#[derive(Debug, Clone)]
pub struct OperatingPointIssue {
    /// UUID of the neuron with the operating-point issue.
    pub neuron_uuid: String,
    /// Current squash (activation) function name.
    pub squash: String,
    /// Current bias of the neuron.
    pub bias: f32,
    /// Minimum observed pre-activation value.
    pub value_min: f32,
    /// Maximum observed pre-activation value.
    pub value_max: f32,
    /// Active zone lower bound for the squash function.
    pub active_zone_min: f32,
    /// Active zone upper bound for the squash function.
    pub active_zone_max: f32,
    /// Fraction of the squash function's dynamic range being utilised (0.0–1.0).
    pub dynamic_range_utilisation: f32,
    /// Number of samples with valid pre-activation values.
    pub sample_count: usize,
}

/// Returns the "active zone" `(min, max)` for a squash function — the
/// pre-activation range over which the function transitions most of its
/// output dynamic range.
///
/// Narrows to the region where the function is most sensitive.
///
/// Returns `None` for unbounded or discrete activations where the concept
/// of an active zone does not apply.
fn active_zone(squash: &str) -> Option<(f32, f32)> {
    match squash {
        // TANH: tanh(x) transitions from ~-0.96 to ~0.96 over x ∈ [-2, 2]
        "TANH" | "HARD_TANH" | "CLIPPED" => Some((-2.0, 2.0)),
        // LOGISTIC: sigmoid transitions from ~0.02 to ~0.98 over x ∈ [-4, 4]
        "LOGISTIC" => Some((-4.0, 4.0)),
        // SOFTSIGN: x/(1+|x|) transitions over a wider range
        "SOFTSIGN" => Some((-4.0, 4.0)),
        // ARCTAN: transitions over roughly [-3, 3]
        "ARCTAN" => Some((-3.0, 3.0)),
        // RELU6: active in [0, 6]
        "RELU6" => Some((0.0, 6.0)),
        // Discrete or unbounded activations — no meaningful active zone
        _ => None,
    }
}

/// Compute what fraction of a squash function's dynamic range is utilised
/// by the given pre-activation range.
///
/// Uses `target_simulation_fn()` to evaluate the squash function at the
/// observed min/max pre-activation values and at the active zone bounds,
/// then computes the ratio of observed output range to theoretical output
/// range.
fn compute_dynamic_range_utilisation(
    squash: &str,
    value_min: f32,
    value_max: f32,
    zone_min: f32,
    zone_max: f32,
) -> f32 {
    let activation_fn = match crate::activations::target_simulation_fn(squash) {
        Some(f) => f,
        None => return 0.0,
    };

    // Output range at the active zone bounds (theoretical dynamic range)
    let zone_out_min = activation_fn(zone_min);
    let zone_out_max = activation_fn(zone_max);
    let theoretical_range = (zone_out_max - zone_out_min).abs();

    if theoretical_range < 1e-9 {
        return 0.0;
    }

    // Output range at the observed pre-activation bounds
    let obs_out_min = activation_fn(value_min);
    let obs_out_max = activation_fn(value_max);
    let observed_range = (obs_out_max - obs_out_min).abs();

    (observed_range / theoretical_range).clamp(0.0, 1.0)
}

/// Detect hidden neurons with operating-point issues.
///
/// Analyses pre-activation (`value`) distribution per hidden neuron and
/// compares it against the squash function's active zone.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples.
/// * `config` - Detection configuration.
///
/// # Returns
/// A list of `OperatingPointIssue` sorted by utilisation (lowest first).
pub fn detect_operating_point_issues(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    config: &OperatingPointConfig,
) -> Vec<OperatingPointIssue> {
    // Build map: UUID → (squash, bias) for hidden neurons only
    let neuron_map: std::collections::HashMap<&str, (&str, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.as_str(), (n.squash.as_str(), n.bias)))
        .collect();

    let mut results = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        let records = records.as_ref();
        let Some(&(squash, bias)) = neuron_map.get(uuid.as_str()) else {
            continue;
        };

        // Get the active zone for this squash function
        let Some((zone_min, zone_max)) = active_zone(squash) else {
            continue;
        };

        // Collect pre-activation values (only records that have `value`)
        let values: Vec<f32> = records
            .iter()
            .filter_map(|r| r.value)
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < config.min_samples {
            continue;
        }

        let value_min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let value_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let utilisation =
            compute_dynamic_range_utilisation(squash, value_min, value_max, zone_min, zone_max);

        if utilisation >= config.utilisation_threshold {
            continue;
        }

        results.push(OperatingPointIssue {
            neuron_uuid: uuid.clone(),
            squash: squash.to_string(),
            bias,
            value_min,
            value_max,
            active_zone_min: zone_min,
            active_zone_max: zone_max,
            dynamic_range_utilisation: utilisation,
            sample_count: values.len(),
        });
    }

    // Sort by utilisation (lowest first — worst offenders first)
    results.sort_by(|a, b| {
        a.dynamic_range_utilisation
            .total_cmp(&b.dynamic_range_utilisation)
    });

    results
}

/// Convert operating-point issues into coordinated structural candidates.
///
/// Each issue may produce up to three candidates:
/// 1. **setBias** — shift the operating point into the active zone.
/// 2. **changeSquash** — switch to a function whose active zone matches.
/// 3. **setWeight** — scale incoming weights to expand the pre-activation range.
///
/// Issue #2479: no deadline — equivalent to calling
/// [`operating_point_to_coordinated_candidates_with_deadline`] with a `None`
/// deadline.
pub fn operating_point_to_coordinated_candidates(
    detected: &[OperatingPointIssue],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    operating_point_to_coordinated_candidates_with_deadline(detected, creature, &None)
}

/// Deadline-aware variant of [`operating_point_to_coordinated_candidates`]
/// (Issue #2479).
///
/// Builds the target-uuid → incoming-synapses lookup once up front (rather than
/// scanning `creature.synapses` per candidate, which was O(C·S)) and checks
/// `deadline` once per candidate so a hung or oversized conversion can be
/// stopped mid-flight, returning the coordinated candidates built so far.
pub fn operating_point_to_coordinated_candidates_with_deadline(
    detected: &[OperatingPointIssue],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    operating_point_to_coordinated_candidates_with_observer(
        detected,
        creature,
        deadline,
        &mut || {},
    )
}

fn operating_point_to_coordinated_candidates_with_observer(
    detected: &[OperatingPointIssue],
    creature: &CreatureJson,
    deadline: &Option<std::time::SystemTime>,
    work_observer: &mut impl FnMut(),
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Issue #2479: build the to_uuid -> incoming synapses lookup once, in synapse
    // order, so each candidate's list keeps the order of the former per-candidate filter.
    let mut incoming_by_target: std::collections::HashMap<&str, Vec<&crate::SynapseJson>> =
        std::collections::HashMap::new();
    for synapse in &creature.synapses {
        work_observer();
        incoming_by_target
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let mut results = Vec::new();

    for issue in detected {
        if deadline_passed(deadline) {
            break;
        }

        let base_improvement = (1.0 - issue.dynamic_range_utilisation) * 0.005;

        // Candidate 1: setBias — shift the operating point to the centre of the active zone
        let observed_centre = (issue.value_min + issue.value_max) / 2.0;
        let active_zone_centre = (issue.active_zone_min + issue.active_zone_max) / 2.0;
        let bias_delta = active_zone_centre - observed_centre;
        let new_bias = issue.bias + bias_delta;

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: issue.neuron_uuid.clone(),
                bias: new_bias,
            }],
            expected_creature_score_gain: base_improvement,
            comment: Some(format!(
                "Operating point on {}: {} using {:.0}% of dynamic range, pre-activation [{:.2}, {:.2}] vs active zone [{:.1}, {:.1}] → setBias from {:.3} to {:.3}",
                issue.neuron_uuid,
                issue.squash,
                issue.dynamic_range_utilisation * 100.0,
                issue.value_min,
                issue.value_max,
                issue.active_zone_min,
                issue.active_zone_max,
                issue.bias,
                new_bias,
            )),
        });

        // Candidate 2: changeSquash — switch to IDENTITY (no bounding)
        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: issue.neuron_uuid.clone(),
                squash: "IDENTITY".to_string(),
            }],
            expected_creature_score_gain: base_improvement * 0.7,
            comment: Some(format!(
                "Operating point on {}: {} using {:.0}% of dynamic range → changeSquash to IDENTITY",
                issue.neuron_uuid,
                issue.squash,
                issue.dynamic_range_utilisation * 100.0,
            )),
        });

        // Candidate 3: setWeight — scale incoming weights to expand range into active zone
        work_observer();
        let incoming_synapses: Vec<&crate::SynapseJson> = incoming_by_target
            .get(issue.neuron_uuid.as_str())
            .cloned()
            .unwrap_or_default();

        if !incoming_synapses.is_empty() {
            let observed_range = (issue.value_max - issue.value_min).abs().max(0.001);
            let active_zone_range = issue.active_zone_max - issue.active_zone_min;
            // Target 80% of active zone coverage
            let target_range = active_zone_range * 0.80;
            let scale_factor = target_range / observed_range;

            let mut weight_ops: Vec<CoordinatedStructuralOpJson> = Vec::new();
            for s in &incoming_synapses {
                weight_ops.push(CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: s.from_uuid.clone(),
                    to_neuron_uuid: s.to_uuid.clone(),
                    weight: s.weight * scale_factor,
                });
            }

            // Also adjust bias proportionally
            weight_ops.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: issue.neuron_uuid.clone(),
                bias: issue.bias * scale_factor,
            });

            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                operations: weight_ops,
                expected_creature_score_gain: base_improvement * 0.5,
                comment: Some(format!(
                    "Operating point on {}: {} using {:.0}% → setWeight scale {:.2}× to expand pre-activation range",
                    issue.neuron_uuid,
                    issue.squash,
                    issue.dynamic_range_utilisation * 100.0,
                    scale_factor,
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

#[cfg(test)]
mod issue_2479_tests {
    use super::*;

    fn issue(uuid: &str) -> OperatingPointIssue {
        OperatingPointIssue {
            neuron_uuid: uuid.to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
            value_min: -0.5,
            value_max: 0.5,
            active_zone_min: -2.0,
            active_zone_max: 2.0,
            dynamic_range_utilisation: 0.1,
            sample_count: 30,
        }
    }

    /// One hidden neuron per uuid, each fed by two incoming synapses.
    fn creature_with_hidden(uuids: &[String]) -> CreatureJson {
        let mut neurons = vec![
            crate::NeuronJson {
                uuid: "input-a".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            crate::NeuronJson {
                uuid: "input-b".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ];
        let mut synapses = Vec::new();
        for uuid in uuids {
            neurons.push(crate::NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            });
            for from in ["input-a", "input-b"] {
                synapses.push(crate::SynapseJson {
                    from_uuid: from.to_string(),
                    to_uuid: uuid.clone(),
                    weight: 1.0,
                    synapse_type: None,
                });
            }
        }
        CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: 1,
        }
    }

    #[test]
    fn incoming_synapse_work_scales_linearly_at_n_and_4n() {
        const N: usize = 128;

        let measured_work = |count: usize| {
            let uuids: Vec<String> = (0..count).map(|i| format!("h{i}")).collect();
            let creature = creature_with_hidden(&uuids);
            let detected: Vec<OperatingPointIssue> =
                uuids.iter().map(|u| issue(u.as_str())).collect();
            let mut work = 0;
            let result = operating_point_to_coordinated_candidates_with_observer(
                &detected,
                &creature,
                &None,
                &mut || work += 1,
            );
            assert_eq!(result.len(), 3 * count);
            work
        };

        // Index build: one step per synapse (two per hidden neuron), then one
        // lookup per issue.
        let small_work = measured_work(N);
        let large_work = measured_work(N * 4);
        assert_eq!(small_work, 2 * N + N);
        assert_eq!(large_work, 2 * (N * 4) + N * 4);
        assert_eq!(large_work, small_work * 4);
    }

    /// The deadline is checked once per issue, and on expiry the candidates
    /// already built are returned rather than discarded. The scripted sequence lets
    /// the first issue through and stops at the second, so hoisting the check out of
    /// the loop (all nine built) or replacing `break` with an empty return (none
    /// built) both fail this test.
    #[test]
    #[serial_test::serial]
    fn deadline_mid_conversion_returns_the_candidates_built_so_far() {
        use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;

        let uuids = [
            "hidden-alpha".to_string(),
            "hidden-beta".to_string(),
            "hidden-gamma".to_string(),
        ];
        let creature = creature_with_hidden(&uuids);
        let detected: Vec<OperatingPointIssue> = uuids.iter().map(|u| issue(u.as_str())).collect();

        // Positive control: with no override every issue converts to three candidates.
        let control =
            operating_point_to_coordinated_candidates_with_deadline(&detected, &creature, &None);
        assert_eq!(
            control.len(),
            9,
            "fixture must yield three candidates per issue"
        );

        let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
        let partial =
            operating_point_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

        assert_eq!(
            partial.len(),
            3,
            "the deadline must stop the conversion at the second issue and keep the first's three candidates"
        );
        for candidate in &partial {
            let comment = candidate.comment.as_deref().unwrap_or_default();
            assert!(
                comment.contains("hidden-alpha"),
                "every kept candidate must belong to the first issue, got: {comment}"
            );
        }
    }
}
