//! Integration tests for adaptive candidate budget allocation by module success rate (Issue #967).
//!
//! Tests that candidate generation budgets are allocated to discovery modules based
//! on their historical Bayesian success rates, with a minimum floor to maintain
//! exploration and a global cap to prevent candidate explosion.

use neat_ai_discovery::analysis::module_weights::{
    CandidateBudgetConfig, ModuleOutcomeTracker, allocate_candidate_budgets,
};

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
// Budget allocation scales with success rate
// =============================================================================

#[test]
fn test_high_success_module_gets_more_candidates() {
    let tracker = tracker_with_history(&[
        ("saturation", 20, 18), // 90% success
        ("dead_neuron", 20, 2), // 10% success
    ]);

    let module_names = vec!["saturation".to_string(), "dead_neuron".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    let sat_budget = budgets["saturation"];
    let dead_budget = budgets["dead_neuron"];

    assert!(
        sat_budget > dead_budget,
        "High-success module ({sat_budget}) should get more candidates than low-success ({dead_budget})"
    );
}

#[test]
fn test_equal_success_gives_equal_budgets() {
    let tracker = tracker_with_history(&[("mod_a", 20, 10), ("mod_b", 20, 10), ("mod_c", 20, 10)]);

    let module_names = vec![
        "mod_a".to_string(),
        "mod_b".to_string(),
        "mod_c".to_string(),
    ];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert_eq!(budgets["mod_a"], budgets["mod_b"]);
    assert_eq!(budgets["mod_b"], budgets["mod_c"]);
}

// =============================================================================
// Budget bounds (0.5x-2.0x range)
// =============================================================================

#[test]
fn test_max_budget_is_two_times_base() {
    let tracker = tracker_with_history(&[
        ("perfect", 20, 20), // 100% success
    ]);

    let module_names = vec!["perfect".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert!(
        budgets["perfect"] <= 20,
        "Budget should not exceed 2x base (20), got {}",
        budgets["perfect"]
    );
}

#[test]
fn test_min_budget_is_half_base() {
    let tracker = tracker_with_history(&[
        ("failure", 20, 0), // 0% success
    ]);

    let module_names = vec!["failure".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert!(
        budgets["failure"] >= 5,
        "Budget should not drop below 0.5x base (5), got {}",
        budgets["failure"]
    );
}

// =============================================================================
// Global candidate cap
// =============================================================================

#[test]
fn test_global_cap_limits_total_candidates() {
    let tracker = tracker_with_history(&[
        ("mod_a", 20, 18),
        ("mod_b", 20, 16),
        ("mod_c", 20, 14),
        ("mod_d", 20, 12),
    ]);

    let module_names = vec![
        "mod_a".to_string(),
        "mod_b".to_string(),
        "mod_c".to_string(),
        "mod_d".to_string(),
    ];
    let config = CandidateBudgetConfig {
        base_candidates: 20,
        global_max_candidates: 50,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    let total: usize = budgets.values().sum();
    assert!(
        total <= 50,
        "Total candidates ({total}) should not exceed global cap (50)"
    );
}

#[test]
fn test_global_cap_preserves_proportions() {
    let tracker = tracker_with_history(&[
        ("high", 20, 18), // ~90% success
        ("low", 20, 2),   // ~10% success
    ]);

    let module_names = vec!["high".to_string(), "low".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 20,
        global_max_candidates: 20, // Force scaling down
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    // High-success module should still get more than low-success even after capping.
    assert!(
        budgets["high"] >= budgets["low"],
        "Proportions should be preserved after capping: high={}, low={}",
        budgets["high"],
        budgets["low"]
    );

    let total: usize = budgets.values().sum();
    assert!(
        total <= 20,
        "Total ({total}) should not exceed global cap (20)"
    );
}

// =============================================================================
// Sparse/unknown modules get base allocation
// =============================================================================

#[test]
fn test_sparse_history_gets_base_allocation() {
    let tracker = tracker_with_history(&[
        ("sparse", 3, 3), // Too few attempts for adaptive
    ]);

    let module_names = vec!["sparse".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert_eq!(
        budgets["sparse"], 10,
        "Sparse-history module should get base allocation"
    );
}

#[test]
fn test_unknown_module_gets_base_allocation() {
    let tracker = ModuleOutcomeTracker::new();

    let module_names = vec!["unknown".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 10,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert_eq!(
        budgets["unknown"], 10,
        "Unknown module should get base allocation"
    );
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_empty_module_list_returns_empty() {
    let tracker = ModuleOutcomeTracker::new();
    let module_names: Vec<String> = vec![];
    let config = CandidateBudgetConfig::default();

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert!(budgets.is_empty());
}

#[test]
fn test_every_module_gets_at_least_one_candidate() {
    let tracker = tracker_with_history(&[
        ("terrible", 20, 0), // 0% success
    ]);

    let module_names = vec!["terrible".to_string()];
    let config = CandidateBudgetConfig {
        base_candidates: 2,
        ..Default::default()
    };

    let budgets = allocate_candidate_budgets(&module_names, &tracker, &config);

    assert!(
        budgets["terrible"] >= 1,
        "Every module must get at least 1 candidate, got {}",
        budgets["terrible"]
    );
}

#[test]
fn test_default_config_has_sensible_values() {
    let config = CandidateBudgetConfig::default();

    assert!(
        config.base_candidates > 0,
        "Base candidates must be positive"
    );
    assert!(
        config.global_max_candidates >= config.base_candidates,
        "Global cap must be >= base"
    );
    assert!(
        (0.0..=1.0).contains(&config.min_allocation_factor),
        "Min factor must be in [0.0, 1.0]"
    );
    assert!(
        config.max_allocation_factor >= 1.0,
        "Max factor must be >= 1.0"
    );
}

// =============================================================================
// DiscoveryModuleSpec max_candidates field
// =============================================================================

#[test]
fn test_module_spec_includes_max_candidates() {
    use neat_ai_discovery::analysis::discovery_dispatch::{
        DiscoveryDetectionResult, DiscoveryModuleSpec,
    };

    let spec = DiscoveryModuleSpec {
        module_name: "test".to_string(),
        phase_name: "test_phase",
        max_candidates: 15,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![],
            })
        }),
    };

    assert_eq!(spec.max_candidates, 15);
}
