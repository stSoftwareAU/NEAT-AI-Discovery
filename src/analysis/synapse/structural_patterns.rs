//! Structural pattern discovery for synapse analysis
//!
//! This module detects coordinated structural changes in the creature topology:
//! - Noisy vs trusted input folding (Issue #165)
//! - Collapse 1-in/1-out hidden neurons into direct synapses (Issue #425)
//!
//! Extracted from mod.rs as part of Issue #482.

use super::scoring::compute_synapse_improvement_and_count;
use crate::analysis::cache::RecordCache;
use crate::analysis::constants::{MIN_NEURON_SAMPLE_COUNT, min_bypass_weight_for_collapse};
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::samples::{EPSILON, HelpfulSample};
use crate::analysis::scoring::weights::calculate_optimal_outgoing_weight;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, SynapseJson};
use std::collections::{HashMap, HashSet};

/// Output of [`detect_collapsible_hidden_neurons`] paired with the count of
/// candidates rejected by the bypass-weight floor (Issue #1270).
///
/// Callers should record `bypass_weight_below_floor_drops` against
/// [`crate::analysis::diagnostics::rejection_reasons::REJECTION_COORDINATED_COLLAPSE_BYPASS_WEIGHT_BELOW_FLOOR`]
/// on the synapse metadata's `rejection_breakdown` so the rejection surfaces
/// in the drought diagnostic.
#[derive(Debug, Default)]
pub(crate) struct CollapseDetectionOutcome {
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
    /// Number of 1-in/1-out chains rejected because the computed bypass
    /// weight had `|weight| < MIN_BYPASS_WEIGHT_FOR_COLLAPSE` (Issue #1270).
    pub bypass_weight_below_floor_drops: u32,
}

// =============================================================================
// Noisy vs Trusted Input Folding (Issue #165)
// =============================================================================

/// Detect noisy vs trusted input pairs feeding the same target neuron.
///
/// When two input signals have the same mean and the same starting synapse weight,
/// but one is much noisier (higher activation variance), the intended coordinated fix is:
/// - Remove the noisy synapse
/// - Remove the trusted synapse
/// - Add the trusted synapse back with a higher weight (typically doubled)
///
/// Returns `Some(candidate)` if a beneficial noisy-vs-trusted pair is found.
pub(crate) fn detect_noisy_vs_trusted(
    target_uuid: &str,
    synapses_by_target: &[&SynapseJson],
    cache: &RecordCache,
    target_map: &TargetMap,
    neuron_squash_map: &HashMap<&str, &str>,
) -> Option<CoordinatedStructuralCandidateJson> {
    fn activation_mean_and_variance(records: &[DiscoverRecord]) -> Option<(f32, f32)> {
        let mut n = 0.0f32;
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for r in records {
            if r.activation.is_finite() {
                n += 1.0;
                sum += r.activation;
                sum_sq += r.activation * r.activation;
            }
        }
        if n <= 0.0 {
            return None;
        }
        let mean = sum / n;
        let var = (sum_sq / n) - (mean * mean);
        Some((mean, var.max(0.0)))
    }

    fn activation_map(records: &[DiscoverRecord]) -> HashMap<u32, f32> {
        let mut map = HashMap::with_capacity(records.len());
        for r in records {
            if r.activation.is_finite() {
                map.insert(r.obs_index, r.activation);
            }
        }
        map
    }

    #[derive(Clone, Copy)]
    struct IncomingInput<'a> {
        from_uuid: &'a str,
        weight: f32,
        mean: f32,
        var: f32,
    }

    let mut incoming_inputs: Vec<IncomingInput<'_>> = Vec::new();
    for syn in synapses_by_target {
        if !syn.from_uuid.starts_with("input-") {
            continue;
        }
        let Ok(from_records_arc) = cache.get(&syn.from_uuid) else {
            continue;
        };
        if from_records_arc.is_empty() {
            continue;
        }
        let Some((mean, var)) = activation_mean_and_variance(from_records_arc.as_ref()) else {
            continue;
        };
        incoming_inputs.push(IncomingInput {
            from_uuid: &syn.from_uuid,
            weight: syn.weight,
            mean,
            var,
        });
    }

    if incoming_inputs.len() < 2 || target_map.map.is_empty() {
        return None;
    }

    // Strict matching for the simple-case test: same weights and same means.
    const WEIGHT_EPS: f32 = 1e-6;
    const MEAN_EPS: f32 = 1e-3;
    const MIN_VAR_RATIO: f32 = 10.0;

    let target_squash = neuron_squash_map.get(target_uuid).copied();

    let mut best: Option<(IncomingInput<'_>, IncomingInput<'_>, f32)> = None; // (noisy, trusted, gain)

    for i in 0..incoming_inputs.len() {
        for j in (i + 1)..incoming_inputs.len() {
            let a = &incoming_inputs[i];
            let b = &incoming_inputs[j];

            if (a.weight - b.weight).abs() > WEIGHT_EPS {
                continue;
            }
            if (a.mean - b.mean).abs() > MEAN_EPS {
                continue;
            }

            let (noisy, trusted) = if a.var >= b.var { (a, b) } else { (b, a) };
            let ratio = noisy.var / trusted.var.max(EPSILON);
            if ratio < MIN_VAR_RATIO {
                continue;
            }

            let Ok(noisy_records_arc) = cache.get(noisy.from_uuid) else {
                continue;
            };
            let Ok(trusted_records_arc) = cache.get(trusted.from_uuid) else {
                continue;
            };

            let noisy_map = activation_map(noisy_records_arc.as_ref());
            let trusted_map = activation_map(trusted_records_arc.as_ref());

            let mut delta_samples: Vec<HelpfulSample> = Vec::with_capacity(target_map.map.len());
            for (obs_index, target) in &target_map.map {
                let Some(noisy_act) = noisy_map.get(obs_index) else {
                    continue;
                };
                let Some(trusted_act) = trusted_map.get(obs_index) else {
                    continue;
                };
                let Some(activation) =
                    crate::analysis::scoring::weights::coordinated_structural_activation_delta(
                        *trusted_act,
                        *noisy_act,
                        noisy.weight,
                        trusted.weight,
                    )
                else {
                    continue;
                };
                if !activation.is_finite() || !target.avg_error.is_finite() {
                    continue;
                }
                delta_samples.push(HelpfulSample {
                    activation,
                    avg_error: target.avg_error,
                    target_value: target.value,
                    target_activation: Some(target.activation),
                });
            }

            if delta_samples.is_empty() {
                continue;
            }

            let baseline_sq: f32 = delta_samples
                .iter()
                .map(|s| s.avg_error * s.avg_error)
                .sum();

            // Move the noisy weight onto the trusted input:
            // Δoutput = w_noisy * (trusted - noisy)
            let moved_weight = noisy.weight;
            let (improvement, _, _, _, _) = compute_synapse_improvement_and_count(
                delta_samples.as_slice(),
                moved_weight,
                baseline_sq,
                target_squash,
            );

            if improvement <= 0.0 {
                continue;
            }

            match &best {
                Some((_, _, best_gain)) if *best_gain >= improvement => {}
                _ => best = Some((*noisy, *trusted, improvement)),
            }
        }
    }

    best.map(|(noisy, trusted, gain)| {
        let new_weight = trusted.weight + noisy.weight;
        CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: noisy.from_uuid.to_string(),
                    to_neuron_uuid: target_uuid.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: trusted.from_uuid.to_string(),
                    to_neuron_uuid: target_uuid.to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: trusted.from_uuid.to_string(),
                    to_neuron_uuid: target_uuid.to_string(),
                    weight: new_weight,
                },
            ],
            expected_creature_score_gain: gain,
            comment: Some(
                "Coordinated: prune noisy input (high variance), strengthen trusted input"
                    .to_string(),
            ),
        }
    })
}

// =============================================================================
// Collapse Hidden Neuron (Issue #425)
// =============================================================================

/// Detect hidden neurons that form simple 1-in/1-out chains and propose collapsing
/// them into direct synapses.
///
/// When a hidden neuron `h` has exactly one incoming synapse (a → h) and one
/// outgoing synapse (h → b), we propose removing `h` and replacing the chain
/// with a direct synapse (a → b). This is emitted as a coordinated-structural
/// candidate group with 4 operations: RemoveSynapse(a→h), RemoveSynapse(h→b),
/// RemoveNeuron(h), AddSynapse(a→b).
pub(crate) fn detect_collapsible_hidden_neurons(
    input: &crate::AnalyzeSynapsesInput,
    cache: &RecordCache,
) -> CollapseDetectionOutcome {
    let mut results = Vec::new();
    let mut bypass_weight_below_floor_drops: u32 = 0;
    let bypass_floor = min_bypass_weight_for_collapse();

    // Build incoming/outgoing synapse lists per neuron (using references to avoid cloning).
    let mut incoming: HashMap<&str, Vec<&SynapseJson>> = HashMap::new();
    let mut outgoing: HashMap<&str, Vec<&SynapseJson>> = HashMap::new();
    for s in &input.creature.synapses {
        incoming.entry(s.to_uuid.as_str()).or_default().push(s);
        outgoing.entry(s.from_uuid.as_str()).or_default().push(s);
    }

    // Quick neuron-type lookup (only creature.neurons; inputs are not here).
    let neuron_type_map_local: HashMap<&str, &str> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();

    // Precompute existing direct synapses so we don't propose duplicates.
    let mut existing_edges: HashSet<(&str, &str)> = HashSet::new();
    for s in &input.creature.synapses {
        existing_edges.insert((s.from_uuid.as_str(), s.to_uuid.as_str()));
    }

    for neuron in &input.creature.neurons {
        if neuron.neuron_type != "hidden" {
            continue;
        }
        let h = neuron.uuid.as_str();
        let Some(ins) = incoming.get(h) else { continue };
        let Some(outs) = outgoing.get(h) else {
            continue;
        };
        if ins.len() != 1 || outs.len() != 1 {
            continue;
        }

        let a_syn = &ins[0];
        let b_syn = &outs[0];
        let a = a_syn.from_uuid.as_str();
        let b = b_syn.to_uuid.as_str();

        // Skip degenerate / non-actionable cases.
        if a == b || a == h || b == h {
            continue;
        }
        if existing_edges.contains(&(a, b)) {
            // A direct synapse already exists; collapsing would need additional ops (future work).
            continue;
        }

        // Ensure the target exists (either input-* or a neuron) so the op is not stale.
        let target_is_known = b.starts_with("input-") || neuron_type_map_local.contains_key(b);
        if !target_is_known {
            continue;
        }

        // Build samples: correlate a's activation to b's adjusted error after removing h→b.
        let Ok(a_records) = cache.get(a) else {
            continue;
        };
        let Ok(h_records) = cache.get(h) else {
            continue;
        };
        let Ok(b_records) = cache.get(b) else {
            continue;
        };
        if a_records.is_empty() || h_records.is_empty() || b_records.is_empty() {
            continue;
        }

        let target_map_b = TargetMap::from_records(b_records.as_ref());
        if target_map_b.map.is_empty() {
            continue;
        }
        let build_act_map = |records: &[DiscoverRecord]| -> HashMap<u32, f32> {
            let mut map: HashMap<u32, f32> = HashMap::with_capacity(records.len());
            for r in records {
                if r.activation.is_finite() {
                    map.insert(r.obs_index, r.activation);
                }
            }
            map
        };
        let a_map = build_act_map(a_records.as_ref());
        let h_map = build_act_map(h_records.as_ref());

        let mut samples: Vec<HelpfulSample> = Vec::with_capacity(target_map_b.map.len());
        for (obs_index, target) in &target_map_b.map {
            let Some(a_act) = a_map.get(obs_index) else {
                continue;
            };
            let Some(h_act) = h_map.get(obs_index) else {
                continue;
            };
            if !a_act.is_finite() || !h_act.is_finite() || !target.avg_error.is_finite() {
                continue;
            }
            let adjusted_error = target.avg_error + b_syn.weight * (*h_act);
            if !adjusted_error.is_finite() {
                continue;
            }
            samples.push(HelpfulSample {
                activation: *a_act,
                avg_error: adjusted_error,
                target_value: None,
                target_activation: None,
            });
        }

        if samples.len() < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        let mut sum_act_sq = 0.0f32;
        let mut sum_err_act = 0.0f32;
        let mut baseline_sq = 0.0f32;
        for s in &samples {
            sum_act_sq += s.activation * s.activation;
            sum_err_act += s.activation * s.avg_error;
            baseline_sq += s.avg_error * s.avg_error;
        }
        if baseline_sq <= EPSILON {
            continue;
        }

        let Some(weight) = calculate_optimal_outgoing_weight(sum_err_act, sum_act_sq, 1.0) else {
            continue;
        };

        // Issue #1270: skip 1-in/1-out collapse candidates whose computed
        // bypass weight is below the meaningful-weight floor. At near-zero
        // bypass weights the chain `a→h→b` was contributing essentially
        // nothing through `h`, so the 4-op coordinated change is functionally
        // equivalent to a 1-op `remove-neuron` but carries the much higher
        // implementation-risk profile of a 4-op coordinated candidate. The
        // dropped count is propagated to the caller via
        // [`CollapseDetectionOutcome`] for the rejection breakdown.
        if weight.abs() < bypass_floor {
            bypass_weight_below_floor_drops = bypass_weight_below_floor_drops.saturating_add(1);
            continue;
        }

        let (improvement, _improved, _worsened, _total, _magnitude) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);
        if improvement <= 0.0 {
            continue;
        }

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: a.to_string(),
                    to_neuron_uuid: h.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: h.to_string(),
                    to_neuron_uuid: b.to_string(),
                },
                CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: h.to_string(),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: a.to_string(),
                    to_neuron_uuid: b.to_string(),
                    weight,
                },
            ],
            expected_creature_score_gain: improvement,
            comment: Some(
                "Coordinated collapse: remove 1-in/1-out hidden neuron and add bypass synapse"
                    .to_string(),
            ),
        });
    }

    CollapseDetectionOutcome {
        candidates: results,
        bypass_weight_below_floor_drops,
    }
}

// =============================================================================
// Tests (Issue #1270)
// =============================================================================

#[cfg(test)]
mod tests {
    //! Unit tests for the 1-in/1-out hidden neuron collapse detector.
    //!
    //! These exercise the bypass-weight floor introduced in Issue #1270
    //! directly via a synthetic [`RecordCache`], avoiding the GPU / parquet
    //! setup required by the integration tests in
    //! `tests/synapse/issue_522_synapse_structural_patterns.rs`.
    use super::*;
    use crate::AnalyzeSynapsesInput;
    use crate::ffi_types::{CreatureJson, NeuronJson, SynapseJson};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    /// Serialise tests that mutate the `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE`
    /// env variable so they do not race with other tests in the same process.
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RAII guard that overrides `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE`
    /// during a test and restores the previous value on drop.
    struct BypassFloorGuard {
        previous: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl BypassFloorGuard {
        fn new(value: &str) -> Self {
            let lock = env_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let previous = std::env::var("NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE").ok();
            // SAFETY: env mutation is serialised via the static lock above.
            unsafe {
                std::env::set_var("NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE", value);
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for BypassFloorGuard {
        fn drop(&mut self) {
            // SAFETY: env mutation is serialised via the held lock.
            unsafe {
                match &self.previous {
                    Some(prev) => {
                        std::env::set_var("NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE", prev);
                    }
                    None => {
                        std::env::remove_var("NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE");
                    }
                }
            }
        }
    }

    /// Build a minimal `input-0 → hidden-0 → output-0` chain whose collapse
    /// would emit a bypass synapse `input-0 → output-0` with computed
    /// `weight ≈ target_bypass_weight`.
    ///
    /// Constructs `n_samples` observations with activations evenly spaced in
    /// `[-1, 1]`. The chain is configured as a pure passthrough
    /// (`h_act = a_act`, intermediate squash IDENTITY, bias 0). With the
    /// existing `h → b` synapse weight fixed at 0.5, the target neuron's
    /// average error is set to `(target_bypass_weight - 0.5) * a_act`, which
    /// makes the function's least-squares fit produce
    /// `weight ≈ target_bypass_weight`.
    fn build_collapse_input(
        target_bypass_weight: f32,
        n_samples: u32,
    ) -> (AnalyzeSynapsesInput, RecordCache) {
        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                SynapseJson {
                    from_uuid: "hidden-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.5,
                    synapse_type: None,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: "<in-memory>".to_string(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: None,
            analysis_deadline_ms: None,
            random_seed: Some(42),
            temperature: 1.0,
            failure_cache: None,
            discovery_outcome_log: None,
        };

        // Build per-neuron record vectors and bind them into the synthetic
        // record cache loader closure.
        let mut input0 = Vec::with_capacity(n_samples as usize);
        let mut hidden0 = Vec::with_capacity(n_samples as usize);
        let mut output0 = Vec::with_capacity(n_samples as usize);
        let b_syn_weight = 0.5_f32;
        for i in 0..n_samples {
            let a_act = -1.0
                + (2.0 * f32::from(u16::try_from(i).unwrap_or(0)))
                    / f32::from(u16::try_from(n_samples - 1).unwrap_or(1));
            let h_act = a_act; // pure passthrough
            let target_err = (target_bypass_weight - b_syn_weight) * a_act;
            input0.push(DiscoverRecord::new(
                i,
                "input-0".to_string(),
                Some(a_act),
                a_act,
                Vec::new(),
            ));
            hidden0.push(DiscoverRecord::new(
                i,
                "hidden-0".to_string(),
                Some(h_act),
                h_act,
                Vec::new(),
            ));
            output0.push(DiscoverRecord::new(
                i,
                "output-0".to_string(),
                Some(0.5 * h_act),
                0.5 * h_act,
                vec![target_err],
            ));
        }

        let cache = RecordCache::with_loader(
            "<in-memory>",
            Arc::new(move |_path: &str, uuid: &str| match uuid {
                "input-0" => Ok(input0.clone()),
                "hidden-0" => Ok(hidden0.clone()),
                "output-0" => Ok(output0.clone()),
                other => Err(anyhow::anyhow!("unexpected uuid {other}")),
            }),
        );

        (input, cache)
    }

    /// Acceptance-criterion test: a synthetic 1-in/1-out chain whose computed
    /// bypass weight is `0.005` must be rejected and the rejection counter
    /// incremented.
    #[test]
    fn rejects_collapse_when_bypass_weight_below_default_floor() {
        let _guard = BypassFloorGuard::new("0.01");

        let (input, cache) = build_collapse_input(0.005, 12);
        let outcome = detect_collapsible_hidden_neurons(&input, &cache);

        assert!(
            outcome.candidates.is_empty(),
            "Expected no collapse candidates with bypass weight 0.005 under a 0.01 floor; got: {:?}",
            outcome.candidates
        );
        assert_eq!(
            outcome.bypass_weight_below_floor_drops, 1,
            "Expected exactly one bypass-weight-floor rejection"
        );
    }

    /// Regression test for the GRQ-sampler creature `bcbca347` failure-cache
    /// pattern: bypass weight `0.0021` must not produce a collapse candidate.
    #[test]
    fn reproduces_bcbca347_failure_cache_pattern() {
        let _guard = BypassFloorGuard::new("0.01");

        let (input, cache) = build_collapse_input(0.0021, 12);
        let outcome = detect_collapsible_hidden_neurons(&input, &cache);

        assert!(
            outcome.candidates.is_empty(),
            "Bypass weight 0.0021 (bcbca347 failure pattern) should be rejected"
        );
        assert_eq!(
            outcome.bypass_weight_below_floor_drops, 1,
            "bcbca347-style bypass weight must increment the rejection counter"
        );
    }

    /// A bypass weight above the floor should still be emitted. This guards
    /// against the floor accidentally rejecting legitimately useful
    /// candidates after the Issue #1270 change.
    #[test]
    fn emits_candidate_when_bypass_weight_above_floor() {
        let _guard = BypassFloorGuard::new("0.01");

        // Computed weight ~0.05, comfortably above the 0.01 default floor.
        let (input, cache) = build_collapse_input(0.05, 12);
        let outcome = detect_collapsible_hidden_neurons(&input, &cache);

        assert_eq!(
            outcome.bypass_weight_below_floor_drops, 0,
            "Above-floor bypass weight must not trigger the rejection counter"
        );
        assert_eq!(
            outcome.candidates.len(),
            1,
            "Expected exactly one collapse candidate, got {} (candidates: {:?})",
            outcome.candidates.len(),
            outcome.candidates,
        );
    }

    /// The env-var override (`NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE
    /// = 0`) must disable the floor entirely so legacy callers that depend on
    /// the pre-#1270 behaviour can opt out for tests.
    #[test]
    fn env_var_override_disables_floor() {
        let _guard = BypassFloorGuard::new("0");

        let (input, cache) = build_collapse_input(0.005, 12);
        let outcome = detect_collapsible_hidden_neurons(&input, &cache);

        assert_eq!(
            outcome.bypass_weight_below_floor_drops, 0,
            "Disabling the floor must suppress the rejection counter"
        );
        // The candidate may or may not survive the `improvement > 0` check
        // — that path is unaffected by Issue #1270. We only assert the
        // dispatch-side counter stays zero when the floor is disabled.
    }
}
