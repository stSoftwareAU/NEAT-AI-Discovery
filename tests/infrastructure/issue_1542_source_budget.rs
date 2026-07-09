//! Issue #1542: two-stage per-target source budget.
//!
//! Stage 1 is the existing cheap CPU pre-score (`order_eligible_sources`), which
//! places the highest-priority sources first. Stage 2 (`apply_source_budget`)
//! caps how many of those ordered sources proceed to the expensive
//! sample-building + GPU evaluation, controlled by
//! `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET`.
//!
//! These tests exercise the real config accessor and budget helper — they set
//! the env var, call the functions, and assert on the returned values/mutation.
//! All env access is serialised via `#[serial]`, so each `unsafe` set/remove is
//! safe (no concurrent env access) — hence the per-block SAFETY comments.

#![allow(clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::utils::{
    OrderedNeuron, apply_source_budget, order_eligible_sources,
};
use neat_ai_discovery::config::max_sources_per_target;
use serial_test::serial;

const ENV: &str = "NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET";

/// Set the budget env var. SAFETY per call: serialised via `#[serial]`.
fn set_env(value: &str) {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var(ENV, value) };
}

/// Clear the budget env var. SAFETY per call: serialised via `#[serial]`.
fn clear_env() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var(ENV) };
}

fn make_neurons(n: usize) -> Vec<OrderedNeuron> {
    (0..n)
        .map(|i| OrderedNeuron {
            uuid: format!("hidden-{i}"),
            index: i,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// config: max_sources_per_target()
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn config_unset_is_unlimited() {
    clear_env();
    assert_eq!(max_sources_per_target(), None);
}

#[test]
#[serial]
fn config_positive_value_parsed() {
    set_env("128");
    assert_eq!(max_sources_per_target(), Some(128));
    clear_env();
}

#[test]
#[serial]
fn config_zero_is_unlimited() {
    // `0` explicitly means "no cap" for back-compat.
    set_env("0");
    assert_eq!(max_sources_per_target(), None);
    clear_env();
}

#[test]
#[serial]
fn config_empty_and_invalid_are_unlimited() {
    set_env("");
    assert_eq!(max_sources_per_target(), None, "empty string");
    set_env("   ");
    assert_eq!(max_sources_per_target(), None, "whitespace only");
    set_env("not-a-number");
    assert_eq!(max_sources_per_target(), None, "non-numeric");
    set_env("-5");
    assert_eq!(max_sources_per_target(), None, "negative");
    clear_env();
}

// ---------------------------------------------------------------------------
// apply_source_budget()
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn budget_unset_is_noop_backcompat() {
    clear_env();
    let neurons = make_neurons(1000);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    let dropped = apply_source_budget(&mut sources);

    assert_eq!(dropped, 0, "unlimited budget drops nothing");
    assert_eq!(sources.len(), 1000, "all sources retained (back-compat)");
}

#[test]
#[serial]
fn budget_larger_than_list_is_noop() {
    set_env("5000");
    let neurons = make_neurons(1000);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    let dropped = apply_source_budget(&mut sources);

    assert_eq!(dropped, 0, "budget above list length drops nothing");
    assert_eq!(sources.len(), 1000);
    clear_env();
}

#[test]
#[serial]
fn budget_truncates_to_top_k() {
    set_env("128");
    let neurons = make_neurons(1000);
    let mut sources: Vec<&OrderedNeuron> = neurons.iter().collect();

    let dropped = apply_source_budget(&mut sources);

    assert_eq!(dropped, 872, "1000 - 128 dropped");
    assert_eq!(sources.len(), 128, "capped to K");
    clear_env();
}

#[test]
#[serial]
fn budget_keeps_leading_window_of_ordering() {
    // The retained set must be exactly the first K of the priority ordering —
    // the budget only drops the low-priority tail, it does not reorder.
    let neurons = make_neurons(500);

    // Full ordering (stage 1) with no budget.
    clear_env();
    let mut full: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut full, Some(999), "issue-1542", 0, None);
    let expected_top_k: Vec<&str> = full.iter().take(64).map(|n| n.uuid.as_str()).collect();

    // Same ordering, then apply the budget (stage 2).
    set_env("64");
    let mut budgeted: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut budgeted, Some(999), "issue-1542", 0, None);
    let dropped = apply_source_budget(&mut budgeted);
    let got: Vec<&str> = budgeted.iter().map(|n| n.uuid.as_str()).collect();

    assert_eq!(dropped, 436);
    assert_eq!(
        got, expected_top_k,
        "budget keeps the top-K of the ordering"
    );
    clear_env();
}

#[test]
#[serial]
fn budget_selection_is_deterministic_under_seed() {
    // Same seed + same K ⇒ identical retained subset across runs.
    let neurons = make_neurons(400);

    set_env("100");

    let mut run1: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut run1, Some(12345), "det", 0, None);
    apply_source_budget(&mut run1);
    let uuids1: Vec<&str> = run1.iter().map(|n| n.uuid.as_str()).collect();

    let mut run2: Vec<&OrderedNeuron> = neurons.iter().collect();
    order_eligible_sources::<String>(&mut run2, Some(12345), "det", 0, None);
    apply_source_budget(&mut run2);
    let uuids2: Vec<&str> = run2.iter().map(|n| n.uuid.as_str()).collect();

    assert_eq!(uuids1.len(), 100);
    assert_eq!(uuids1, uuids2, "deterministic selection under fixed seed");
    clear_env();
}
