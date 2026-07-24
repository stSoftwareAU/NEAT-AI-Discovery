//! Issue #1547 — creature-scale tiering / skipping of expensive detection modules.
//!
//! Acceptance criteria from the issue, expressed as behaviour tests against the
//! real tiering predicate [`should_skip_module`] (the same function that gates
//! production dispatch in `analysis::module_dispatch_specs`):
//!
//! (a) With `hidden_neuron_count > N` and no escalation active, `expensive`-tier
//!     modules are absent from the dispatched set while `always` / `standard`
//!     modules still run.
//! (b) With drought / novelty escalation active, the full module set is
//!     dispatched regardless of scale.
//! (c) Below the threshold, tiering is a no-op (dispatched set identical to
//!     today).

use neat_ai_discovery::analysis::module_tiering::{
    ALWAYS_MODULES, DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD, EXPENSIVE_MODULES, ModuleTier,
    classify_module, should_skip_module, tiering_applies,
};
use neat_ai_discovery::config::module_tiering_hidden_neuron_threshold;
use serial_test::serial;

const THRESHOLD_ENV: &str = "NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD";

/// The ~48 discovery modules dispatched by `analyze_all`, as a representative
/// mix spanning all three tiers. Expensive + always names must match the
/// production `module_name` strings; the rest stand in for the standard bulk.
fn all_module_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    names.extend_from_slice(EXPENSIVE_MODULES);
    names.extend_from_slice(ALWAYS_MODULES);
    // A handful of real standard-tier modules.
    names.extend_from_slice(&[
        "dead neuron detection",
        "dormant synapse detection",
        "gradient-based discovery",
        "saturation detection",
        "bottleneck detection",
    ]);
    names
}

/// Compute the set of modules that would actually be dispatched, mirroring the
/// production `apply_module_tiering` retain step.
fn dispatched(
    all: &[&'static str],
    hidden: usize,
    threshold: usize,
    escalation: bool,
) -> Vec<&'static str> {
    all.iter()
        .copied()
        .filter(|name| !should_skip_module(name, hidden, threshold, escalation))
        .collect()
}

/// AC(a): large creature, no escalation → expensive modules absent, others kept.
#[test]
fn ac_a_large_creature_skips_expensive_modules_only() {
    let all = all_module_names();
    let threshold = DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD;
    let hidden = 1662; // large production creature

    assert!(tiering_applies(hidden, threshold, false));

    let dispatched = dispatched(&all, hidden, threshold, false);

    // Every expensive module is absent.
    for name in EXPENSIVE_MODULES {
        assert!(
            !dispatched.contains(name),
            "expensive module {name} must be tiered out on a large creature"
        );
    }
    // Every non-expensive module still runs.
    for name in &all {
        if classify_module(name) != ModuleTier::Expensive {
            assert!(
                dispatched.contains(name),
                "non-expensive module {name} must still be dispatched"
            );
        }
    }
    // The dispatched set is strictly smaller than the full set.
    assert!(dispatched.len() < all.len());
}

/// AC(b): escalation active → full set dispatched regardless of scale.
#[test]
fn ac_b_escalation_re_enables_full_module_set() {
    let all = all_module_names();
    let threshold = DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD;
    let hidden = 100_000; // absurdly large — scale must not matter under escalation

    assert!(!tiering_applies(hidden, threshold, true));

    let dispatched = dispatched(&all, hidden, threshold, true);
    assert_eq!(
        dispatched.len(),
        all.len(),
        "escalation must dispatch the full module set"
    );
    for name in &all {
        assert!(
            dispatched.contains(name),
            "{name} must run under escalation"
        );
    }
}

/// AC(c): below the threshold → no-op, dispatched set identical to today.
#[test]
fn ac_c_below_threshold_is_a_no_op() {
    let all = all_module_names();
    let threshold = DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD;

    for hidden in [0_usize, 1, 500, threshold] {
        assert!(!tiering_applies(hidden, threshold, false));
        let dispatched = dispatched(&all, hidden, threshold, false);
        assert_eq!(
            dispatched, all,
            "below/at threshold ({hidden}) tiering must be a no-op"
        );
    }
}

/// Tiering disabled via a `0` threshold keeps the full set even on huge creatures.
#[test]
fn zero_threshold_disables_tiering() {
    let all = all_module_names();
    let dispatched = dispatched(&all, 1_000_000, 0, false);
    assert_eq!(dispatched, all, "threshold 0 must disable tiering");
}

/// The env-var override drives the threshold; unset falls back to the default;
/// `0` disables tiering.
#[test]
#[serial]
fn config_threshold_reads_env_var() {
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var(THRESHOLD_ENV) };
    assert_eq!(
        module_tiering_hidden_neuron_threshold(),
        DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
        "unset must fall back to the default"
    );

    // SAFETY: Serialised via #[serial].
    unsafe { std::env::set_var(THRESHOLD_ENV, "250") };
    assert_eq!(module_tiering_hidden_neuron_threshold(), 250);

    // SAFETY: Serialised via #[serial].
    unsafe { std::env::set_var(THRESHOLD_ENV, "0") };
    assert_eq!(module_tiering_hidden_neuron_threshold(), 0);
    assert!(!tiering_applies(1_000_000, 0, false), "0 disables tiering");

    // SAFETY: Serialised via #[serial] — restore clean state.
    unsafe { std::env::remove_var(THRESHOLD_ENV) };
}

/// The published tier lists are internally consistent: no name appears in both
/// the expensive and always lists, and every expensive name classifies as
/// expensive.
#[test]
fn tier_lists_are_consistent() {
    for name in EXPENSIVE_MODULES {
        assert_eq!(classify_module(name), ModuleTier::Expensive);
        assert!(
            !ALWAYS_MODULES.contains(name),
            "{name} must not be in both tier lists"
        );
    }
    for name in ALWAYS_MODULES {
        assert_eq!(classify_module(name), ModuleTier::Always);
    }
}
