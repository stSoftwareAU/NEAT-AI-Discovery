//! Creature-scale discovery module tiering (Issue #1547).
//!
//! After synapse/neuron dispatch, `analyze_all` runs ~48 discovery modules in
//! parallel. On a **large** creature (many hidden neurons) some of those modules
//! are inherently expensive — multi-hop correlation maps, co-adaptation and
//! weight-coherence pairwise scans, topology diversification — because their cost
//! grows super-linearly with the hidden-neuron count. Spending post-processing
//! budget on them every pass competes with the next discovery cycle's wall-clock.
//!
//! This module classifies each discovery module into a cost [`ModuleTier`] and
//! decides — purely, so it is trivially unit-testable — whether an `Expensive`
//! module should be skipped for a given pass. The gate fires **only** when:
//!
//! 1. tiering is enabled (`threshold > 0`),
//! 2. the creature is large (`hidden_neuron_count > threshold`), and
//! 3. no drought / novelty escalation is active for this pass (#1422 / #1423).
//!
//! When drought or novelty escalation engages, [`should_skip_module`] returns
//! `false` for every module, so the full set is re-enabled for that pass — the
//! escalation path must be free to try everything to escape the drought.
//!
//! Modules are **never** removed from the codebase; gating happens at dispatch
//! time. On small creatures (at or below the threshold) tiering is a no-op and
//! the dispatched set is identical to today's.

/// Cost tier for a discovery module (Issue #1547).
///
/// Only [`ModuleTier::Expensive`] modules are ever skipped by tiering; `Always`
/// and `Standard` both always run. The distinction between `Always` and
/// `Standard` is documentary — it records which modules are known to be cheap or
/// near-no-ops at GRQ scale (e.g. correlated-error early-returns when the
/// creature has a single output) versus ordinary-cost modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleTier {
    /// Cheap or near-no-op modules — always dispatched.
    Always,
    /// Ordinary-cost modules — always dispatched.
    Standard,
    /// Expensive modules — skipped on large creatures outside escalation.
    Expensive,
}

/// Default hidden-neuron count above which expensive modules are tiered out on
/// non-escalation passes (Issue #1547).
///
/// A value of `0` disables tiering entirely (every module always runs).
pub const DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD: usize = 1000;

/// Discovery modules classified as [`ModuleTier::Expensive`] (Issue #1547).
///
/// These perform pairwise or multi-hop scans whose cost grows super-linearly
/// with the hidden-neuron count, so they dominate the post-processing budget on
/// GRQ-scale creatures. Names must match the `module_name` set in
/// `analysis::discovery_dispatch` exactly.
pub const EXPENSIVE_MODULES: &[&str] = &[
    "multi-hop analysis",
    "topology structure analysis",
    "topology diversification detection",
    "co-adaptation detection",
    "merge redundant neuron detection",
    "weight coherence ratio detection",
    "skip connection discovery",
];

/// Discovery modules classified as [`ModuleTier::Always`] (Issue #1547).
///
/// Cheap or near-no-op at GRQ scale (e.g. correlated-error detection
/// early-returns when the creature has a single output), so they are documented
/// as always-run even though — like `Standard` — they are never tiered out.
pub const ALWAYS_MODULES: &[&str] = &["correlated error detection"];

/// Classify a discovery module by name into its cost [`ModuleTier`].
///
/// Unknown module names default to [`ModuleTier::Standard`] so a new module is
/// never accidentally skipped until it is explicitly classified as expensive.
#[must_use]
pub fn classify_module(module_name: &str) -> ModuleTier {
    if EXPENSIVE_MODULES.contains(&module_name) {
        ModuleTier::Expensive
    } else if ALWAYS_MODULES.contains(&module_name) {
        ModuleTier::Always
    } else {
        ModuleTier::Standard
    }
}

/// Whether creature-scale tiering applies for this pass at all (Issue #1547).
///
/// Returns `true` only when tiering is enabled (`threshold > 0`), the creature is
/// larger than the threshold, and no escalation is active. When it returns
/// `false`, every module is dispatched exactly as today.
#[must_use]
pub fn tiering_applies(
    hidden_neuron_count: usize,
    threshold: usize,
    escalation_active: bool,
) -> bool {
    threshold > 0 && !escalation_active && hidden_neuron_count > threshold
}

/// Decide whether a single discovery module should be skipped for this pass.
///
/// Skips only [`ModuleTier::Expensive`] modules, and only when
/// [`tiering_applies`] holds. On escalation passes, below the threshold, or with
/// tiering disabled, this always returns `false`.
#[must_use]
pub fn should_skip_module(
    module_name: &str,
    hidden_neuron_count: usize,
    threshold: usize,
    escalation_active: bool,
) -> bool {
    tiering_applies(hidden_neuron_count, threshold, escalation_active)
        && classify_module(module_name) == ModuleTier::Expensive
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expensive_modules_classify_as_expensive() {
        for name in EXPENSIVE_MODULES {
            assert_eq!(
                classify_module(name),
                ModuleTier::Expensive,
                "{name} must be Expensive"
            );
        }
    }

    #[test]
    fn always_modules_classify_as_always() {
        for name in ALWAYS_MODULES {
            assert_eq!(
                classify_module(name),
                ModuleTier::Always,
                "{name} must be Always"
            );
        }
    }

    #[test]
    fn unknown_module_defaults_to_standard() {
        assert_eq!(
            classify_module("some brand new module"),
            ModuleTier::Standard
        );
    }

    #[test]
    fn large_creature_without_escalation_skips_expensive_only() {
        let hidden = 1662; // GRQ-scale
        let threshold = 1000;
        assert!(should_skip_module(
            "multi-hop analysis",
            hidden,
            threshold,
            false
        ));
        assert!(should_skip_module(
            "co-adaptation detection",
            hidden,
            threshold,
            false
        ));
        // Standard / Always modules are never skipped.
        assert!(!should_skip_module(
            "dead neuron detection",
            hidden,
            threshold,
            false
        ));
        assert!(!should_skip_module(
            "correlated error detection",
            hidden,
            threshold,
            false
        ));
    }

    #[test]
    fn escalation_re_enables_full_set_regardless_of_scale() {
        let hidden = 100_000;
        for name in EXPENSIVE_MODULES {
            assert!(
                !should_skip_module(name, hidden, 1000, true),
                "escalation must re-enable {name}"
            );
        }
    }

    #[test]
    fn below_threshold_is_a_no_op() {
        // Small creature: nothing is skipped, tiering does not apply.
        assert!(!tiering_applies(999, 1000, false));
        for name in EXPENSIVE_MODULES {
            assert!(!should_skip_module(name, 999, 1000, false));
        }
        // Exactly at the threshold is still a no-op (strictly greater required).
        assert!(!tiering_applies(1000, 1000, false));
    }

    #[test]
    fn zero_threshold_disables_tiering() {
        assert!(!tiering_applies(1_000_000, 0, false));
        assert!(!should_skip_module(
            "multi-hop analysis",
            1_000_000,
            0,
            false
        ));
    }
}
