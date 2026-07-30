//! Structure-only removal triage (Issue #1767).
//!
//! ## Opposite axes
//!
//! Focus and removal answer near-**opposite** questions and must therefore be
//! driven by opposite criteria over the *same* structural impact map:
//!
//! | Concern | Criteria | Records needed |
//! |---------|----------|----------------|
//! | Focus (#1766) | **High** structural impact, weighted-random (outputs seed at 1.0) | Never for choosing the set |
//! | Removal (#1767) | **Low** contribution vs the complexity savings from pruning | Structure only for triage |
//!
//! Removal is *not* "negate the focus score" — high error never means remove
//! (see the Issue #414 disablement). It is **low contribution vs savings**.
//!
//! ## Why this path exists
//!
//! [`super::removal_candidates::identify_removal_candidates`] leans on the same
//! axis, but it consumes [`super::RankedNeuron`]s whose
//! `activation_weighted_impact` is derived from recorded discovery data. That
//! coupled removal triage to a parquet decode at focus-selection time — a
//! production incident burned ~2 h warming a 12.5 GB discovery file before any
//! useful discovery work began.
//!
//! [`triage_removal_candidates`] needs only the creature topology: it derives
//! structural impact from path weights
//! ([`compute_impacts_public`](crate::focus::impact::compute_impacts_public)) and
//! complexity savings from synapse counts. No parquet file is opened, decoded,
//! or even named. Activation-weighted gates that still need records belong in
//! the **analysis** phase, after the focus set is fixed.
//!
//! ## One criterion, two shapes (Issue #1783)
//!
//! This module holds **no decision logic**. The savings-vs-impact criterion, the
//! [`REMOVAL_CANDIDATE_BOOST`](crate::analysis::constants::REMOVAL_CANDIDATE_BOOST)
//! application point, the non-finite-impact policy (#1804), the noise-floor
//! re-gate (#1142), the hidden-only filter and the net-improvement-descending
//! sort all live in
//! [`identify_structural_removal_candidates`](super::removal_candidates::identify_structural_removal_candidates)
//! — the path the FFI ships. A near-identical copy used to live here and had
//! already drifted once (non-finite impact, `costOfGrowth` validation), so
//! [`triage_removal_candidates`] is now a thin **adapter** over that single
//! implementation: it only reshapes [`super::RemovalCandidate`] into the
//! record-free [`StructuralRemovalCandidate`].

use super::removal_candidates::identify_structural_removal_candidates;
use crate::CreatureJson;
use std::collections::HashMap;

/// A hidden neuron whose complexity savings outweigh its **structural**
/// contribution — safe to prune on topology evidence alone (Issue #1767).
///
/// Deliberately carries no record-derived fields (no `mean_activation`, no
/// `activation_weighted_impact`, no `total_error`): every value here is
/// computable from the creature JSON, which is what keeps focus-time triage
/// free of parquet I/O. The activation-weighted view lives on
/// [`super::RemovalCandidate`] and is produced later, in the analysis phase.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralRemovalCandidate {
    pub neuron_uuid: String,
    /// Structural impact magnitude — the weighted path product to the outputs.
    /// Low impact is what makes the neuron a removal candidate.
    pub impact: f32,
    /// Number of synapses pointing TO this neuron.
    pub incoming_synapses: usize,
    /// Number of synapses pointing FROM this neuron.
    pub outgoing_synapses: usize,
    /// Complexity savings from pruning the neuron and its synapses, after the
    /// [`REMOVAL_CANDIDATE_BOOST`](crate::analysis::constants::REMOVAL_CANDIDATE_BOOST)
    /// multiplier.
    pub removal_savings: f32,
    /// `removal_savings - impact`: how much better the creature is expected to
    /// score once the neuron is gone.
    pub net_improvement: f32,
    pub reason: String,
}

/// Outcome of [`triage_removal_candidates`] — surviving candidates plus every
/// rejection class, surfaced rather than silently dropped.
///
/// # Conservation invariant (Issue #1808)
///
/// `candidates.len() + rejection_breakdown().values().sum() ==
/// hidden_neurons_considered`. Every hidden neuron entering triage is either
/// emitted as a candidate or counted under a named rejection reason, so a
/// future gate that drops one without a counter fails
/// `removal_triage_accounts_for_every_hidden_neuron` rather than vanishing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructuralRemovalTriage {
    /// Candidates that cleared both the savings-vs-impact test and the
    /// noise floor, best net improvement first.
    pub candidates: Vec<StructuralRemovalCandidate>,
    /// Candidates dropped because `net_improvement` fell below
    /// [`remove_low_impact_noise_floor`](crate::analysis::constants::remove_low_impact_noise_floor)
    /// (Issue #1142).
    pub noise_floor_rejections: u32,
    /// Hidden neurons dropped because the boosted savings never exceeded their
    /// structural contribution — the first gate, and the dominant rejection
    /// class at shipped defaults (Issue #1808).
    pub savings_below_impact_rejections: u32,
    /// Hidden neurons that entered triage, whatever their verdict (Issue #1808).
    pub hidden_neurons_considered: u32,
}

impl StructuralRemovalTriage {
    /// Stable-keyed rejection breakdown, the same shape the shipped FFI path
    /// merges into `metadata.rejection_breakdown` (Issue #1808).
    ///
    /// Callers no longer have to hard-code a reason constant to report why
    /// hidden neurons were dropped.
    #[must_use]
    pub fn rejection_breakdown(&self) -> HashMap<String, u32> {
        use crate::analysis::diagnostics::rejection_reasons::{
            REJECTION_REMOVAL_BELOW_NOISE_FLOOR, REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
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
        ] {
            if count > 0 {
                map.insert(reason.to_string(), count);
            }
        }
        map
    }

    /// Total hidden neurons rejected across every named reason.
    #[must_use]
    pub fn total_rejections(&self) -> u32 {
        self.noise_floor_rejections
            .saturating_add(self.savings_below_impact_rejections)
    }
}

/// Triage removal candidates from the creature **topology alone** (Issue #1767).
///
/// A hidden neuron is a removal candidate when the complexity savings from
/// pruning it exceed its structural contribution:
///
/// ```text
/// savings          = costOfGrowth × (1 + (incoming + outgoing) / 10)
/// boostedSavings   = savings × REMOVAL_CANDIDATE_BOOST
/// candidate  ⟺  boostedSavings > |impact|  and  boostedSavings − |impact| ≥ noiseFloor
/// ```
///
/// This is the near-**opposite** of the focus criterion (high structural
/// impact, weighted-random — Issue #1766) evaluated over the same impact map,
/// so the two never compete for the same neurons.
///
/// Only `hidden` neurons are triaged. Outputs seed at impact `1.0` and are the
/// add-neuron targets, inputs and constants are not prunable — mirroring the
/// hidden-only gate in the constant-neuron removal path (Issue #306).
///
/// # Performance
///
/// Topology-only: no parquet file is opened or decoded, so the seconds bar
/// from Issue #1766 holds whether or not multi-GB discovery data exists. The
/// underlying pass is `rayon`-parallel over the creature's neurons.
///
/// # Arguments
/// * `creature` — the creature topology to triage.
/// * `cost_of_growth` — per-hidden-neuron growth cost; `None`, non-finite, or
///   non-positive falls back to the crate default (logged at WARN).
#[must_use]
pub fn triage_removal_candidates(
    creature: &CreatureJson,
    cost_of_growth: Option<f32>,
) -> StructuralRemovalTriage {
    let outcome = identify_structural_removal_candidates(creature, cost_of_growth);

    // Already sorted best-net-improvement-first by the single implementation;
    // reshaping preserves that order.
    let candidates = outcome
        .candidates
        .into_iter()
        .map(|c| StructuralRemovalCandidate {
            net_improvement: c.removal_savings - c.impact,
            impact: c.impact,
            incoming_synapses: c.incoming_synapses,
            outgoing_synapses: c.outgoing_synapses,
            removal_savings: c.removal_savings,
            neuron_uuid: c.neuron_uuid,
            reason: c.reason,
        })
        .collect();

    StructuralRemovalTriage {
        candidates,
        noise_floor_rejections: outcome.noise_floor_rejections,
        savings_below_impact_rejections: outcome.savings_below_impact_rejections,
        hidden_neurons_considered: outcome.considered,
    }
}

// =============================================================================
// Unification parity tests (Issue #1783 / #1805)
// =============================================================================

#[cfg(test)]
mod unification_parity_tests {
    //! Issue #1805: the savings-vs-impact criterion now exists exactly once, in
    //! [`identify_structural_removal_candidates`]. These tests prove the two
    //! surviving entry points cannot silently re-diverge: for the same creature
    //! and cost of growth they must return the **same candidate set in the same
    //! order**, with identical impacts, savings, net improvements and
    //! noise-floor rejection counts.
    //!
    //! They live here rather than in `tests/` because the shipped path is
    //! `pub(crate)` — and because the non-finite-impact case (where the two
    //! copies previously disagreed, `INFINITY` vs `0.0`) needs a NaN synapse
    //! weight, which serde refuses to carry across the JSON FFI boundary.

    use super::super::removal_candidates::env_lock;
    use super::*;
    use crate::{NeuronJson, SynapseJson};

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

    /// Hidden neurons with wildly different synapse counts and impact levels, so
    /// the candidate set, the ordering and the noise-floor rejections all carry
    /// real signal.
    fn mixed_creature() -> CreatureJson {
        let mut neurons = vec![neuron("in-0", "input"), neuron("out", "output")];
        let mut synapses = vec![];
        for i in 0..6 {
            let uuid = format!("h-{i}");
            neurons.push(neuron(&uuid, "hidden"));
            synapses.push(synapse("in-0", &uuid, 0.5));
            // Even neurons dominate the output; odd ones are negligible.
            let weight = if i % 2 == 0 { 1.0 } else { 1e-8 };
            synapses.push(synapse(&uuid, "out", weight));
            // Vary the synapse count so savings — and therefore the sort order —
            // differ between candidates.
            for j in 0..i {
                let extra = format!("in-{i}-{j}");
                neurons.push(neuron(&extra, "input"));
                synapses.push(synapse(&extra, &uuid, 0.5));
            }
        }
        let input = neurons.iter().filter(|n| n.neuron_type == "input").count();
        CreatureJson {
            neurons,
            synapses,
            input,
            output: 1,
        }
    }

    /// A creature whose `h-nan` hidden neuron carries a NaN structural impact —
    /// the exact input on which the two copies used to disagree.
    fn creature_with_nan_impact() -> CreatureJson {
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

    /// Assert the adapter and the shipped path agree exactly, field by field and
    /// in order, for one (creature, cost-of-growth) pair.
    fn assert_parity(label: &str, creature: &CreatureJson, cost_of_growth: Option<f32>) {
        let shipped = identify_structural_removal_candidates(creature, cost_of_growth);
        let adapted = triage_removal_candidates(creature, cost_of_growth);

        let shipped_ids: Vec<&str> = shipped
            .candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();
        let adapted_ids: Vec<&str> = adapted
            .candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();
        assert_eq!(
            adapted_ids, shipped_ids,
            "[{label}] both entry points must return the same candidates in the same order"
        );
        assert_eq!(
            adapted.noise_floor_rejections, shipped.noise_floor_rejections,
            "[{label}] noise-floor rejection counts must match"
        );
        // Issue #1808: the two paths must also agree on the newer counters, or
        // they have re-diverged on rejection accounting.
        assert_eq!(
            adapted.savings_below_impact_rejections, shipped.savings_below_impact_rejections,
            "[{label}] savings-vs-impact rejection counts must match"
        );
        assert_eq!(
            adapted.hidden_neurons_considered, shipped.considered,
            "[{label}] hidden-neurons-considered must match"
        );
        assert_eq!(
            adapted.rejection_breakdown(),
            shipped.rejection_breakdown(),
            "[{label}] both paths must report the same rejection breakdown"
        );

        for (a, s) in adapted.candidates.iter().zip(&shipped.candidates) {
            assert_eq!(a.impact.to_bits(), s.impact.to_bits(), "[{label}] impact");
            assert_eq!(
                a.removal_savings.to_bits(),
                s.removal_savings.to_bits(),
                "[{label}] removal savings (boost application point)"
            );
            assert_eq!(
                a.net_improvement.to_bits(),
                (s.removal_savings - s.impact).to_bits(),
                "[{label}] net improvement"
            );
            assert_eq!(
                (a.incoming_synapses, a.outgoing_synapses),
                (s.incoming_synapses, s.outgoing_synapses),
                "[{label}] synapse counts"
            );
            assert_eq!(a.reason, s.reason, "[{label}] reason");
        }
    }

    /// Acceptance: identical candidate sets and ordering for the same creature
    /// and cost of growth.
    #[test]
    fn entry_points_agree_on_candidates_and_ordering() {
        let _guard = env_lock();
        let creature = mixed_creature();
        assert_parity("mixed 1e-4", &creature, Some(1e-4));

        // Sanity: the fixture must actually produce candidates, or parity is vacuous.
        let triage = triage_removal_candidates(&creature, Some(1e-4));
        assert!(
            triage.candidates.len() >= 2,
            "fixture must yield several candidates to make the ordering assertion meaningful, got {:?}",
            triage.candidates
        );
    }

    /// Acceptance: parity on a **non-finite** impact — the input on which the two
    /// copies previously disagreed (`f32::INFINITY` vs `0.0`).
    #[test]
    fn entry_points_agree_on_a_non_finite_impact() {
        let _guard = env_lock();
        let creature = creature_with_nan_impact();

        // Sanity: the fixture really does yield a non-finite impact.
        let impacts = crate::focus::impact::compute_impacts_public(&creature);
        assert!(
            !impacts.get("h-nan").copied().unwrap_or(0.0).is_finite(),
            "fixture must yield a non-finite impact for h-nan, or the test proves nothing"
        );

        assert_parity("nan impact", &creature, Some(1e-4));

        let triage = triage_removal_candidates(&creature, Some(1e-4));
        let ids: Vec<&str> = triage
            .candidates
            .iter()
            .map(|c| c.neuron_uuid.as_str())
            .collect();
        assert!(
            !ids.contains(&"h-nan"),
            "a non-finite impact must never license a prune on either path, got {ids:?}"
        );
        assert!(
            ids.contains(&"h-zero"),
            "a genuine 0.0-impact neuron must stay prunable, got {ids:?}"
        );
    }

    /// Acceptance: the `costOfGrowth` validation now guards the single criterion,
    /// so both entry points fall back to the default identically.
    #[test]
    fn entry_points_agree_on_an_invalid_cost_of_growth() {
        let _guard = env_lock();
        let creature = mixed_creature();

        for cost in [
            None,
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(-1.0),
            Some(0.0),
        ] {
            assert_parity("invalid cost", &creature, cost);
        }

        // …and the fallback is the default, not a silently different result.
        let default_run = identify_structural_removal_candidates(&creature, None);
        for invalid in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
            let actual = identify_structural_removal_candidates(&creature, Some(invalid));
            let actual_ids: Vec<&str> = actual
                .candidates
                .iter()
                .map(|c| c.neuron_uuid.as_str())
                .collect();
            let expected_ids: Vec<&str> = default_run
                .candidates
                .iter()
                .map(|c| c.neuron_uuid.as_str())
                .collect();
            assert_eq!(
                actual_ids, expected_ids,
                "costOfGrowth {invalid} must fall back to the default on the shipped path too"
            );
            assert_eq!(
                actual.noise_floor_rejections, default_run.noise_floor_rejections,
                "costOfGrowth {invalid} must produce the default rejection count"
            );
        }
    }

    /// A creature that trips **both** removal gates plus the accept path, so
    /// the conservation assertion below has all three verdicts to balance.
    ///
    /// Each hidden neuron has one inbound and one outbound synapse, so its
    /// boosted savings are `costOfGrowth × 1.2 × 1.5`. Structural impact is the
    /// neuron's share of the output's total inbound weight, which is what the
    /// outbound weights below dial in:
    ///
    /// * `h-dominant` — impact ≈ 1.0, far above the savings ⇒ savings-vs-impact
    ///   rejection.
    /// * `h-marginal` — impact just under the savings but within the noise
    ///   floor ⇒ noise-floor rejection.
    /// * `h-idle-a` / `h-idle-b` — negligible impact ⇒ candidates.
    fn creature_tripping_both_gates() -> CreatureJson {
        let hidden = [
            ("h-dominant", 1.0_f32),
            ("h-marginal", 1.75e-4),
            ("h-idle-a", 1e-9),
            ("h-idle-b", 1e-9),
        ];
        let mut neurons = vec![neuron("in-0", "input")];
        let mut synapses = vec![];
        for (uuid, _) in hidden {
            neurons.push(neuron(uuid, "hidden"));
        }
        neurons.push(neuron("out", "output"));
        for (uuid, weight) in hidden {
            synapses.push(synapse("in-0", uuid, 0.5));
            synapses.push(synapse(uuid, "out", weight));
        }
        CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: 1,
        }
    }

    /// Acceptance (Issue #1808): every hidden neuron entering triage is either
    /// emitted as a candidate or counted under a named rejection reason.
    ///
    /// A future gate that drops a neuron with a bare `continue` breaks this
    /// equality, so the silent-drop class the issue describes cannot come back.
    #[test]
    fn removal_triage_accounts_for_every_hidden_neuron() {
        let _guard = env_lock();
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        let creature = creature_tripping_both_gates();
        let triage = triage_removal_candidates(&creature, Some(1e-4));

        let hidden = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        assert_eq!(
            triage.hidden_neurons_considered as usize, hidden,
            "every hidden neuron must be reported as considered"
        );

        // The fixture must exercise both gates, or the conservation assertion
        // proves nothing about the newly-counted class.
        assert!(
            triage.savings_below_impact_rejections > 0,
            "fixture must trip the savings-vs-impact gate, got {triage:?}"
        );
        assert!(
            triage.noise_floor_rejections > 0,
            "fixture must trip the noise floor, got {triage:?}"
        );
        assert!(
            !triage.candidates.is_empty(),
            "fixture must also emit candidates, got {triage:?}"
        );

        let counted: u32 = triage.rejection_breakdown().values().sum();
        assert_eq!(
            counted,
            triage.total_rejections(),
            "the breakdown must report every rejection the counters hold"
        );
        assert_eq!(
            triage.candidates.len() as u32 + counted,
            triage.hidden_neurons_considered,
            "candidates + rejections must equal the hidden neurons considered: {triage:?}"
        );
    }

    /// The rejection breakdown is keyed by the shared reason vocabulary, so a
    /// caller can merge it into `metadata.rejection_breakdown` without
    /// hard-coding a reason constant (Issue #1808).
    #[test]
    fn triage_reports_rejections_under_named_reasons() {
        use crate::analysis::diagnostics::rejection_reasons::{
            REJECTION_REMOVAL_BELOW_NOISE_FLOOR, REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
        };
        let _guard = env_lock();
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        let triage = triage_removal_candidates(&creature_tripping_both_gates(), Some(1e-4));
        let breakdown = triage.rejection_breakdown();

        assert_eq!(
            breakdown
                .get(REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT)
                .copied(),
            Some(triage.savings_below_impact_rejections)
        );
        assert_eq!(
            breakdown.get(REJECTION_REMOVAL_BELOW_NOISE_FLOOR).copied(),
            Some(triage.noise_floor_rejections)
        );
    }

    /// Sub-noise-floor savings are rejected — and counted — identically on both
    /// paths (Issue #1142 contract preserved through the merge).
    ///
    /// **Issue #1814 changed this test's setup, not its contract.** At the
    /// production [`DEFAULT_COST_OF_GROWTH`](crate::focus::DEFAULT_COST_OF_GROWTH)
    /// these ~2e-7 net improvements are no longer noise: the floor is now
    /// `REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS × costOfGrowth`, so they are
    /// correctly emitted as candidates — that is the defect #1814 fixes. The
    /// absolute env override pins the historical `1e-5` so both entry points are
    /// still checked for agreement on a *rejecting* floor, and parity at the
    /// shipped default is asserted first.
    #[test]
    fn entry_points_agree_on_noise_floor_rejections() {
        let _guard = env_lock();
        let creature = mixed_creature();
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }
        assert_parity("default growth", &creature, Some(1e-7));

        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR", "1e-5");
        }
        assert_parity("pinned absolute floor", &creature, Some(1e-7));
        let triage = triage_removal_candidates(&creature, Some(1e-7));
        // SAFETY: env access is serialised via `env_lock()` for this test.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR");
        }

        assert!(
            triage.candidates.is_empty(),
            "1.8e-7 net improvements are below a pinned 1e-5 floor and must be dropped, got {:?}",
            triage.candidates
        );
        assert!(
            triage.noise_floor_rejections > 0,
            "the drop must be reported, never silently swallowed"
        );
    }
}
