//! Integration tests for adaptive discovery module timeout allocation (Issue #603).
//!
//! Tests that time budgets are allocated to discovery modules based on their
//! historical acceptance rates, with decay and fallback behaviour.

use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;

/// Helper to create a tracker with known module history.
fn tracker_with_history(entries: &[(&str, u32, u32)]) -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    for &(name, attempts, successes) in entries {
        for i in 0..attempts {
            tracker.record(name, i < successes);
        }
    }
    tracker
}

// =============================================================================
// Proportional allocation tests
// =============================================================================

#[test]
fn test_equal_history_gives_equal_budgets() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 10),
        ("dead_neuron", 20, 10),
        ("bottleneck", 20, 10),
    ]);

    let module_names = vec![
        "saturation".to_string(),
        "dead_neuron".to_string(),
        "bottleneck".to_string(),
    ];

    let budgets = allocate_time_budgets(&module_names, &tracker, 3000, 1.0);

    // With equal success rates, budgets should be approximately equal.
    // Each should get roughly 1000ms (3000 / 3).
    assert_eq!(budgets.len(), 3);
    let total: u64 = budgets.values().sum();
    assert!(total <= 3000, "total budget should not exceed total_ms");

    for budget in budgets.values() {
        // Allow some tolerance for rounding.
        assert!(
            *budget >= 900 && *budget <= 1100,
            "expected ~1000ms, got {budget}ms"
        );
    }
}

#[test]
fn test_high_yield_module_gets_more_time() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 18), // 90% success rate
        ("dead_neuron", 20, 2), // 10% success rate
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];

    let budgets = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);

    assert_eq!(budgets.len(), 2);

    let sat_budget = budgets["saturation"];
    let dead_budget = budgets["dead_neuron"];

    // The high-yield module should receive more time than the low-yield module.
    assert!(
        sat_budget > dead_budget,
        "saturation ({sat_budget}ms) should get more time than dead_neuron ({dead_budget}ms)"
    );
}

#[test]
fn test_zero_success_module_still_gets_minimum_budget() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 20), // 100% success
        ("dead_neuron", 20, 0), // 0% success
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];

    let budgets = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);

    let dead_budget = budgets["dead_neuron"];

    // Even a zero-success module should get a minimum allocation (not zero).
    assert!(
        dead_budget > 0,
        "zero-success module should still get a non-zero budget, got {dead_budget}ms"
    );
}

// =============================================================================
// Fallback to proportional allocation
// =============================================================================

#[test]
fn test_sparse_history_falls_back_to_proportional() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    // Fewer than MIN_BOOST_SAMPLES attempts — should fall back to equal allocation.
    let tracker = tracker_with_history(&[
        ("saturation", 3, 3),  // too few attempts
        ("dead_neuron", 3, 0), // too few attempts
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];

    let budgets = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);

    assert_eq!(budgets.len(), 2);

    let sat_budget = budgets["saturation"];
    let dead_budget = budgets["dead_neuron"];

    // With insufficient history, should be approximately equal.
    let diff = (sat_budget as i64 - dead_budget as i64).unsigned_abs();
    assert!(
        diff <= 100,
        "expected roughly equal budgets with sparse history, got saturation={sat_budget}ms dead_neuron={dead_budget}ms"
    );
}

#[test]
fn test_unknown_modules_get_proportional_allocation() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 18), // known high-yield
    ]);

    let module_names = vec![
        "saturation".to_string(),
        "unknown_module".to_string(), // not in tracker
    ];

    let budgets = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);

    assert_eq!(budgets.len(), 2);

    // Unknown module should get a neutral allocation (not zero).
    let unknown_budget = budgets["unknown_module"];
    assert!(
        unknown_budget > 0,
        "unknown module should get a non-zero budget, got {unknown_budget}ms"
    );
}

#[test]
fn test_empty_tracker_gives_equal_budgets() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = ModuleOutcomeTracker::new();

    let module_names = vec![
        "saturation".to_string(),
        "dead_neuron".to_string(),
        "bottleneck".to_string(),
    ];

    let budgets = allocate_time_budgets(&module_names, &tracker, 3000, 1.0);

    assert_eq!(budgets.len(), 3);

    for budget in budgets.values() {
        assert!(
            *budget >= 900 && *budget <= 1100,
            "expected ~1000ms with empty tracker, got {budget}ms"
        );
    }
}

// =============================================================================
// Decay factor tests
// =============================================================================

#[test]
fn test_decay_factor_reduces_historical_influence() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 18), // 90% success
        ("dead_neuron", 20, 2), // 10% success
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];

    // With full decay (factor = 0.0), history has no influence — equal budgets.
    let budgets_no_decay = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);
    let budgets_full_decay = allocate_time_budgets(&module_names, &tracker, 2000, 0.0);

    let sat_no_decay = budgets_no_decay["saturation"];
    let dead_no_decay = budgets_no_decay["dead_neuron"];
    let sat_full_decay = budgets_full_decay["saturation"];
    let dead_full_decay = budgets_full_decay["dead_neuron"];

    // With no decay, difference should be large.
    let diff_no_decay = (sat_no_decay as i64 - dead_no_decay as i64).unsigned_abs();
    // With full decay, difference should be small (equal allocation).
    let diff_full_decay = (sat_full_decay as i64 - dead_full_decay as i64).unsigned_abs();

    assert!(
        diff_no_decay > diff_full_decay,
        "decay should reduce the difference: no_decay_diff={diff_no_decay}, full_decay_diff={diff_full_decay}"
    );
}

#[test]
fn test_partial_decay_is_intermediate() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 18), // 90% success
        ("dead_neuron", 20, 2), // 10% success
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];

    let budgets_full = allocate_time_budgets(&module_names, &tracker, 2000, 1.0);
    let budgets_half = allocate_time_budgets(&module_names, &tracker, 2000, 0.5);
    let budgets_none = allocate_time_budgets(&module_names, &tracker, 2000, 0.0);

    let diff_full =
        (budgets_full["saturation"] as i64 - budgets_full["dead_neuron"] as i64).unsigned_abs();
    let diff_half =
        (budgets_half["saturation"] as i64 - budgets_half["dead_neuron"] as i64).unsigned_abs();
    let diff_none =
        (budgets_none["saturation"] as i64 - budgets_none["dead_neuron"] as i64).unsigned_abs();

    // Full influence > half influence > no influence (equal)
    assert!(
        diff_full >= diff_half,
        "full influence diff ({diff_full}) should be >= half influence diff ({diff_half})"
    );
    assert!(
        diff_half >= diff_none,
        "half influence diff ({diff_half}) should be >= no influence diff ({diff_none})"
    );
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_single_module_gets_full_budget() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[("saturation", 20, 10)]);

    let module_names = vec!["saturation".to_string()];

    let budgets = allocate_time_budgets(&module_names, &tracker, 5000, 1.0);

    assert_eq!(budgets.len(), 1);
    assert_eq!(budgets["saturation"], 5000);
}

#[test]
fn test_empty_module_list_returns_empty_budgets() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = ModuleOutcomeTracker::new();
    let module_names: Vec<String> = vec![];

    let budgets = allocate_time_budgets(&module_names, &tracker, 5000, 1.0);

    assert!(budgets.is_empty());
}

#[test]
fn test_budgets_sum_to_total() {
    use neat_ai_discovery::analysis::module_weights::allocate_time_budgets;

    let tracker = tracker_with_history(&[
        ("saturation", 20, 18),
        ("dead_neuron", 20, 2),
        ("bottleneck", 20, 10),
        ("oscillation", 20, 15),
    ]);

    let module_names = vec![
        "saturation".to_string(),
        "dead_neuron".to_string(),
        "bottleneck".to_string(),
        "oscillation".to_string(),
    ];

    let budgets = allocate_time_budgets(&module_names, &tracker, 10000, 1.0);

    let total: u64 = budgets.values().sum();
    assert!(
        total <= 10000,
        "total budget ({total}ms) should not exceed 10000ms"
    );
    // Should use most of the budget (allow small rounding loss).
    assert!(
        total >= 9900,
        "total budget ({total}ms) should use most of the available time"
    );
}

// =============================================================================
// ModuleOutcomeTracker decay method tests
// =============================================================================

#[test]
fn test_apply_decay_reduces_counts() {
    let mut tracker = tracker_with_history(&[("saturation", 20, 10), ("dead_neuron", 40, 20)]);

    tracker.apply_decay(0.5);

    let sat_stats = tracker.stats("saturation");
    let dead_stats = tracker.stats("dead_neuron");

    // After 0.5 decay: attempts should be halved (rounded).
    assert!(
        sat_stats.attempts <= 11,
        "expected ~10 attempts after decay, got {}",
        sat_stats.attempts
    );
    assert!(
        dead_stats.attempts <= 21,
        "expected ~20 attempts after decay, got {}",
        dead_stats.attempts
    );

    // Success rate should remain approximately the same.
    let sat_rate = sat_stats.success_rate();
    assert!(
        (sat_rate - 0.5).abs() < 0.1,
        "success rate should be preserved after decay, got {sat_rate}"
    );
}

#[test]
fn test_apply_decay_zero_clears_history() {
    let mut tracker = tracker_with_history(&[("saturation", 20, 10)]);

    tracker.apply_decay(0.0);

    let stats = tracker.stats("saturation");
    assert_eq!(stats.attempts, 0, "zero decay should clear all counts");
    assert_eq!(stats.successes, 0, "zero decay should clear all counts");
}

#[test]
fn test_apply_decay_one_preserves_history() {
    let mut tracker = tracker_with_history(&[("saturation", 20, 10)]);

    tracker.apply_decay(1.0);

    let stats = tracker.stats("saturation");
    assert_eq!(stats.attempts, 20, "decay=1.0 should preserve all counts");
    assert_eq!(stats.successes, 10, "decay=1.0 should preserve all counts");
}
