//! Add-synapse candidate gating (Issue #1057).
//!
//! Provides filtering logic to skip add-synapse candidate generation when
//! historical data or structural properties indicate near-zero success rates.
//!
//! ## Gating criteria
//!
//! 1. **Module outcome tracker gate**: When the `ModuleOutcomeTracker` shows a
//!    historical success rate below a configurable threshold (default 1%) for
//!    the `"add-synapse"` module, add-synapse candidates are skipped entirely
//!    to save compute.
//!
//! 2. **Synapse density gate**: When the network's synapse-to-neuron ratio
//!    exceeds a threshold (default 14.0), adding a single synapse has minimal
//!    structural impact and is unlikely to succeed. Candidates are skipped.
//!
//! Both gates require sufficient historical data (`MIN_BOOST_SAMPLES` attempts)
//! before they activate, ensuring new creatures get a fair chance.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use crate::analysis::module_weights::ModuleOutcomeTracker;
use crate::ffi_types::CreatureJson;

/// Module name used to track add-synapse outcomes in the `ModuleOutcomeTracker`.
pub const ADD_SYNAPSE_MODULE_NAME: &str = "add-synapse";

/// Default success rate threshold below which add-synapse generation is skipped.
///
/// GRQ-sampler data shows add-synapse candidates have ~0.1-0.3% success rates.
/// A 1% threshold skips generation when the tracker confirms this pattern for
/// a given creature, saving wasted compute.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values above 0.05 may skip add-synapse too eagerly
/// for creatures that genuinely benefit from connectivity changes.
pub const DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD: f64 = 0.01;

/// Default synapse density ratio above which add-synapse candidates are skipped.
///
/// When a creature has more than this many synapses per neuron, adding a single
/// synapse has negligible structural impact. The signal-to-noise ratio is too
/// low for meaningful improvement.
///
/// A creature with 1,400 neurons and 19,000 synapses has a ratio of ~13.6.
///
/// ## Valid Range
/// Must be > 1.0. Values below 5.0 may skip too eagerly for sparse networks.
pub const DEFAULT_SYNAPSE_DENSITY_THRESHOLD: f32 = 14.0;

/// Determines whether add-synapse candidates should be skipped based on
/// the `ModuleOutcomeTracker` historical success rate.
///
/// Returns `true` if add-synapse generation should be **skipped** (gated out).
///
/// The gate activates only when:
/// 1. The tracker has at least `MIN_BOOST_SAMPLES` attempts for `"add-synapse"`
/// 2. The Bayesian success rate is below `success_threshold`
///
/// When the tracker has insufficient data, the gate is inactive (returns `false`)
/// to allow the system to gather data before making gating decisions.
pub fn should_skip_add_synapse_by_outcome(
    tracker: &ModuleOutcomeTracker,
    success_threshold: f64,
) -> bool {
    let stats = tracker.stats(ADD_SYNAPSE_MODULE_NAME);

    // Require sufficient data before gating.
    if (stats.attempts as usize) < crate::analysis::constants::MIN_BOOST_SAMPLES {
        return false;
    }

    let rate = stats.success_rate();
    rate < success_threshold
}

/// Determines whether add-synapse candidates should be skipped based on
/// synapse density (synapses-per-neuron ratio).
///
/// Returns `true` if add-synapse generation should be **skipped** (gated out).
///
/// The total neuron count includes input neurons (which are valid synapse
/// sources) plus all neurons in the creature definition.
pub fn should_skip_add_synapse_by_density(creature: &CreatureJson, density_threshold: f32) -> bool {
    let total_neurons = creature.input + creature.neurons.len();
    if total_neurons == 0 {
        return false;
    }

    let synapse_count = creature.synapses.len();
    let density = synapse_count as f32 / total_neurons as f32;

    density > density_threshold
}

/// Combined gating check for add-synapse candidate generation (Issue #1057).
///
/// Returns `true` if add-synapse candidates should be **skipped**.
///
/// Checks both the outcome tracker gate and the density gate. Either condition
/// being true is sufficient to skip generation.
pub fn should_skip_add_synapse(
    tracker: &ModuleOutcomeTracker,
    creature: &CreatureJson,
    success_threshold: f64,
    density_threshold: f32,
) -> bool {
    should_skip_add_synapse_by_outcome(tracker, success_threshold)
        || should_skip_add_synapse_by_density(creature, density_threshold)
}

/// Apply add-synapse gating to analysis results (Issue #1057).
///
/// If the gate determines that add-synapse candidates should be skipped,
/// clears the `helpful_synapses` vector from the results, preserving
/// harmful synapse and coordinated structural candidates.
///
/// Returns the number of helpful synapse candidates that were removed.
pub fn gate_add_synapse_candidates(
    helpful_synapses: &mut Vec<crate::CandidateSynapseJson>,
    tracker: &ModuleOutcomeTracker,
    creature: &CreatureJson,
) -> usize {
    if should_skip_add_synapse(
        tracker,
        creature,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
    ) {
        let removed = helpful_synapses.len();
        if removed > 0 {
            tracing::info!(
                removed,
                module = ADD_SYNAPSE_MODULE_NAME,
                "add-synapse candidates gated out (Issue #1057)"
            );
            helpful_synapses.clear();
        }
        removed
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::constants::MIN_BOOST_SAMPLES;
    use crate::analysis::module_weights::ModuleOutcomeTracker;
    use crate::ffi_types::{CreatureJson, NeuronJson, SynapseJson};

    // =========================================================================
    // Outcome tracker gate tests
    // =========================================================================

    #[test]
    fn test_skip_by_outcome_insufficient_data_returns_false() {
        let mut tracker = ModuleOutcomeTracker::new();
        // Record fewer than MIN_BOOST_SAMPLES attempts
        for _ in 0..(MIN_BOOST_SAMPLES - 1) {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        assert!(
            !should_skip_add_synapse_by_outcome(&tracker, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD),
            "should not skip when insufficient data"
        );
    }

    #[test]
    fn test_skip_by_outcome_empty_tracker_returns_false() {
        let tracker = ModuleOutcomeTracker::new();
        assert!(
            !should_skip_add_synapse_by_outcome(&tracker, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD),
            "should not skip with empty tracker"
        );
    }

    #[test]
    fn test_skip_by_outcome_low_success_rate_returns_true() {
        let mut tracker = ModuleOutcomeTracker::new();
        // Record many failures and very few successes (< 1% success rate)
        for _ in 0..100 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        // 0 successes out of 100 → Bayesian rate = 1/102 ≈ 0.0098 < 0.01
        assert!(
            should_skip_add_synapse_by_outcome(&tracker, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD),
            "should skip with near-zero success rate"
        );
    }

    #[test]
    fn test_skip_by_outcome_high_success_rate_returns_false() {
        let mut tracker = ModuleOutcomeTracker::new();
        // Record high success rate (50%)
        for _ in 0..50 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, true);
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        assert!(
            !should_skip_add_synapse_by_outcome(&tracker, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD),
            "should not skip with healthy success rate"
        );
    }

    #[test]
    fn test_skip_by_outcome_borderline_success_rate() {
        let mut tracker = ModuleOutcomeTracker::new();
        // 2 successes out of 100 → Bayesian rate = 3/102 ≈ 0.0294 > 0.01
        for _ in 0..98 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        for _ in 0..2 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, true);
        }
        assert!(
            !should_skip_add_synapse_by_outcome(&tracker, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD),
            "should not skip when success rate is above threshold"
        );
    }

    #[test]
    fn test_skip_by_outcome_respects_custom_threshold() {
        let mut tracker = ModuleOutcomeTracker::new();
        // 5 successes out of 100 → Bayesian rate = 6/102 ≈ 0.0588
        for _ in 0..95 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        for _ in 0..5 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, true);
        }
        // With 10% threshold, should skip (5.88% < 10%)
        assert!(should_skip_add_synapse_by_outcome(&tracker, 0.10));
        // With 1% threshold, should not skip (5.88% > 1%)
        assert!(!should_skip_add_synapse_by_outcome(&tracker, 0.01));
    }

    #[test]
    fn test_skip_by_outcome_only_checks_add_synapse_module() {
        let mut tracker = ModuleOutcomeTracker::new();
        // Record failures for a different module
        for _ in 0..100 {
            tracker.record("saturation", false);
        }
        // add-synapse has no data, so should not skip
        assert!(!should_skip_add_synapse_by_outcome(
            &tracker,
            DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
        ),);
    }

    // =========================================================================
    // Density gate tests
    // =========================================================================

    fn make_creature(
        neuron_count: usize,
        synapse_count: usize,
        input_count: usize,
    ) -> CreatureJson {
        let neurons: Vec<NeuronJson> = (0..neuron_count)
            .map(|i| NeuronJson {
                uuid: format!("neuron-{i}"),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            })
            .collect();
        let synapses: Vec<SynapseJson> = (0..synapse_count)
            .map(|i| SynapseJson {
                from_uuid: format!("input-{}", i % input_count.max(1)),
                to_uuid: format!("neuron-{}", i % neuron_count.max(1)),
                weight: 0.5,
                synapse_type: None,
            })
            .collect();
        CreatureJson {
            neurons,
            synapses,
            input: input_count,
            output: 1,
        }
    }

    #[test]
    fn test_skip_by_density_sparse_network_returns_false() {
        // 10 neurons + 5 inputs = 15 total, 20 synapses → ratio 1.33
        let creature = make_creature(10, 20, 5);
        assert!(
            !should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "should not skip for sparse network"
        );
    }

    #[test]
    fn test_skip_by_density_dense_network_returns_true() {
        // 100 neurons + 10 inputs = 110 total, 2000 synapses → ratio 18.18
        let creature = make_creature(100, 2000, 10);
        assert!(
            should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "should skip for dense network"
        );
    }

    #[test]
    fn test_skip_by_density_at_threshold() {
        // 100 neurons + 0 inputs = 100 total, 1400 synapses → ratio 14.0 (not >)
        let creature = make_creature(100, 1400, 0);
        assert!(
            !should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "should not skip when exactly at threshold (uses >)"
        );
    }

    #[test]
    fn test_skip_by_density_just_above_threshold() {
        // 100 neurons + 0 inputs = 100 total, 1401 synapses → ratio 14.01
        let creature = make_creature(100, 1401, 0);
        assert!(
            should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "should skip when just above threshold"
        );
    }

    #[test]
    fn test_skip_by_density_empty_creature_returns_false() {
        let creature = make_creature(0, 0, 0);
        assert!(
            !should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "should not skip for empty creature"
        );
    }

    #[test]
    fn test_skip_by_density_respects_custom_threshold() {
        // 10 neurons + 5 inputs = 15 total, 100 synapses → ratio 6.67
        let creature = make_creature(10, 100, 5);
        // With threshold 5.0, should skip (6.67 > 5.0)
        assert!(should_skip_add_synapse_by_density(&creature, 5.0));
        // With default threshold 14.0, should not skip (6.67 < 14.0)
        assert!(!should_skip_add_synapse_by_density(
            &creature,
            DEFAULT_SYNAPSE_DENSITY_THRESHOLD
        ));
    }

    #[test]
    fn test_skip_by_density_counts_input_neurons() {
        // 10 neurons + 100 inputs = 110 total, 1000 synapses → ratio 9.09
        let creature = make_creature(10, 1000, 100);
        assert!(
            !should_skip_add_synapse_by_density(&creature, DEFAULT_SYNAPSE_DENSITY_THRESHOLD),
            "input neurons count toward total, lowering density"
        );
    }

    // =========================================================================
    // Combined gate tests
    // =========================================================================

    #[test]
    fn test_combined_gate_both_inactive() {
        let tracker = ModuleOutcomeTracker::new();
        let creature = make_creature(10, 20, 5);
        assert!(!should_skip_add_synapse(
            &tracker,
            &creature,
            DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
            DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
        ),);
    }

    #[test]
    fn test_combined_gate_outcome_active_only() {
        let mut tracker = ModuleOutcomeTracker::new();
        for _ in 0..100 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }
        let creature = make_creature(10, 20, 5); // sparse
        assert!(
            should_skip_add_synapse(
                &tracker,
                &creature,
                DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
                DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
            ),
            "outcome gate alone should trigger skip"
        );
    }

    #[test]
    fn test_combined_gate_density_active_only() {
        let tracker = ModuleOutcomeTracker::new(); // no data
        let creature = make_creature(100, 2000, 10); // dense
        assert!(
            should_skip_add_synapse(
                &tracker,
                &creature,
                DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
                DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
            ),
            "density gate alone should trigger skip"
        );
    }

    // =========================================================================
    // gate_add_synapse_candidates integration test
    // =========================================================================

    fn make_test_candidate() -> crate::CandidateSynapseJson {
        crate::CandidateSynapseJson {
            from_neuron_uuid: "input-0".to_string(),
            to_neuron_uuid: "neuron-1".to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: 0.5,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.01,
            expected_creature_score_gain: 0.01,
            improved_count: 5,
            total_count: 10,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.5,
            expected_score_gain_confidence_interval: [0.005, 0.015],
            comment: None,
        }
    }

    #[test]
    fn test_gate_removes_helpful_when_gated() {
        let mut tracker = ModuleOutcomeTracker::new();
        for _ in 0..100 {
            tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
        }

        let mut helpful = vec![make_test_candidate()];
        let creature = make_creature(10, 20, 5); // sparse, but tracker gates

        let removed = gate_add_synapse_candidates(&mut helpful, &tracker, &creature);
        assert_eq!(removed, 1);
        assert!(helpful.is_empty());
    }

    #[test]
    fn test_gate_preserves_helpful_when_not_gated() {
        let tracker = ModuleOutcomeTracker::new(); // no data
        let creature = make_creature(10, 20, 5); // sparse

        let mut helpful = vec![make_test_candidate()];

        let removed = gate_add_synapse_candidates(&mut helpful, &tracker, &creature);
        assert_eq!(removed, 0);
        assert_eq!(helpful.len(), 1);
    }
}
