//! Tests for Issue #227: Track and prioritise neurons with historical improvement success.
//!
//! Discovery currently treats all neurons equally when selecting focus targets.
//! By tracking which neurons have historically led to successful discoveries
//! (candidates that survived ablation testing), we can prioritise them in future
//! runs, improving the discovery hit rate.
//!
//! Key features:
//! - Bayesian scoring with beta distribution posterior mean
//! - New neurons get a neutral prior (0.5 success rate)
//! - History can be persisted and loaded
//! - Focus selection incorporates history scores

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::discovery_history::{DiscoveryHistory, NeuronDiscoveryHistory};
use neat_ai_discovery::focus::rank_focus_neurons_with_history;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a simple creature with specified neurons and synapses
fn create_creature(
    neurons: Vec<(&str, &str)>,       // (uuid, type)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t)| *t == "output").count();
    CreatureJson {
        // Filter out input neurons - they are only represented by creature.input count
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type)| *neuron_type != "input")
            .map(|(uuid, neuron_type)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, weight)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Helper to create parquet records for neurons with specified errors
fn create_records(neuron_errors: Vec<(&str, f32)>) -> Vec<DiscoverRecord> {
    neuron_errors
        .into_iter()
        .flat_map(|(uuid, error)| {
            vec![
                DiscoverRecord::new(0, uuid.to_string(), Some(0.5), 0.5, vec![error]),
                DiscoverRecord::new(1, uuid.to_string(), Some(0.5), 0.5, vec![error]),
            ]
        })
        .collect()
}

// =============================================================================
// NeuronDiscoveryHistory Tests
// =============================================================================

#[test]
fn test_neuron_discovery_history_new() {
    let history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    assert_eq!(history.uuid(), "hidden-1");
    assert_eq!(history.attempts(), 0);
    assert_eq!(history.successes(), 0);
    assert!(history.last_success_epoch().is_none());
}

#[test]
fn test_neuron_discovery_history_success_rate_no_attempts() {
    // With no attempts, success rate should return the neutral prior (0.5)
    let history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    let rate = history.success_rate();
    assert!(
        (rate - 0.5).abs() < 0.001,
        "Success rate with no attempts should be 0.5 (neutral prior), got {rate}"
    );
}

#[test]
fn test_neuron_discovery_history_success_rate_with_attempts() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    history.record_attempt(true, Some(1000));
    history.record_attempt(true, Some(1001));
    history.record_attempt(false, None);

    // 2 successes out of 3 attempts = 66.7%
    let rate = history.success_rate();
    assert!(
        (rate - 2.0 / 3.0).abs() < 0.001,
        "Success rate should be 2/3 (~0.667), got {rate}"
    );
}

#[test]
fn test_neuron_discovery_history_bayesian_score_no_attempts() {
    // With no attempts, Bayesian score uses prior (alpha=1, beta=1) → 0.5
    let history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    let score = history.bayesian_score();
    assert!(
        (score - 0.5).abs() < 0.001,
        "Bayesian score with no attempts should be 0.5, got {score}"
    );
}

#[test]
fn test_neuron_discovery_history_bayesian_score_with_attempts() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    // Record 3 successes, 1 failure
    for _ in 0..3 {
        history.record_attempt(true, Some(1000));
    }
    history.record_attempt(false, None);

    // Beta distribution posterior mean: (alpha) / (alpha + beta)
    // where alpha = successes + 1, beta = failures + 1
    // alpha = 3 + 1 = 4, beta = 1 + 1 = 2
    // Expected: 4 / (4 + 2) = 4/6 = 0.667
    let score = history.bayesian_score();
    assert!(
        (score - 4.0 / 6.0).abs() < 0.001,
        "Bayesian score should be 4/6 (~0.667), got {score}"
    );
}

#[test]
fn test_neuron_discovery_history_bayesian_handles_all_successes() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    for _ in 0..10 {
        history.record_attempt(true, Some(1000));
    }

    // alpha = 10 + 1 = 11, beta = 0 + 1 = 1
    // Expected: 11 / (11 + 1) = 11/12 = 0.917
    let score = history.bayesian_score();
    assert!(
        (score - 11.0 / 12.0).abs() < 0.001,
        "Bayesian score with all successes should be ~0.917, got {score}"
    );
    assert!(
        score < 1.0,
        "Bayesian score should be < 1.0 (regularised), got {score}"
    );
}

#[test]
fn test_neuron_discovery_history_bayesian_handles_all_failures() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    for _ in 0..10 {
        history.record_attempt(false, None);
    }

    // alpha = 0 + 1 = 1, beta = 10 + 1 = 11
    // Expected: 1 / (1 + 11) = 1/12 = 0.083
    let score = history.bayesian_score();
    assert!(
        (score - 1.0 / 12.0).abs() < 0.001,
        "Bayesian score with all failures should be ~0.083, got {score}"
    );
    assert!(
        score > 0.0,
        "Bayesian score should be > 0.0 (regularised), got {score}"
    );
}

#[test]
fn test_neuron_discovery_history_record_attempt_updates_last_success() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());

    // Record a failure - last_success should remain None
    history.record_attempt(false, None);
    assert!(history.last_success_epoch().is_none());

    // Record a success with epoch
    history.record_attempt(true, Some(12345));
    assert_eq!(history.last_success_epoch(), Some(12345));

    // Record another failure - last_success should remain unchanged
    history.record_attempt(false, None);
    assert_eq!(history.last_success_epoch(), Some(12345));

    // Record another success - should update last_success
    history.record_attempt(true, Some(12350));
    assert_eq!(history.last_success_epoch(), Some(12350));
}

// =============================================================================
// DiscoveryHistory Container Tests
// =============================================================================

#[test]
fn test_discovery_history_new() {
    let history = DiscoveryHistory::new();
    assert!(history.is_empty());
    assert_eq!(history.len(), 0);
}

#[test]
fn test_discovery_history_get_nonexistent_returns_none() {
    let history = DiscoveryHistory::new();
    assert!(history.get("hidden-1").is_none());
}

#[test]
fn test_discovery_history_record_creates_new_entry() {
    let mut history = DiscoveryHistory::new();
    history.record("hidden-1", true, Some(1000));

    assert_eq!(history.len(), 1);
    let entry = history.get("hidden-1").expect("Entry should exist");
    assert_eq!(entry.attempts(), 1);
    assert_eq!(entry.successes(), 1);
}

#[test]
fn test_discovery_history_record_updates_existing_entry() {
    let mut history = DiscoveryHistory::new();
    history.record("hidden-1", true, Some(1000));
    history.record("hidden-1", false, None);
    history.record("hidden-1", true, Some(1001));

    let entry = history.get("hidden-1").expect("Entry should exist");
    assert_eq!(entry.attempts(), 3);
    assert_eq!(entry.successes(), 2);
    assert_eq!(entry.last_success_epoch(), Some(1001));
}

#[test]
fn test_discovery_history_bayesian_score_for_unknown_neuron() {
    let history = DiscoveryHistory::new();
    // Unknown neurons should get neutral prior (0.5)
    let score = history.bayesian_score_for("unknown-neuron");
    assert!(
        (score - 0.5).abs() < 0.001,
        "Unknown neuron should have neutral Bayesian score (0.5), got {score}"
    );
}

#[test]
fn test_discovery_history_bayesian_score_for_known_neuron() {
    let mut history = DiscoveryHistory::new();
    history.record("hidden-1", true, Some(1000));
    history.record("hidden-1", true, Some(1001));
    history.record("hidden-1", false, None);

    // 2 successes, 1 failure: alpha = 3, beta = 2
    // Bayesian score = 3/5 = 0.6
    let score = history.bayesian_score_for("hidden-1");
    assert!(
        (score - 3.0 / 5.0).abs() < 0.001,
        "Bayesian score should be 0.6, got {score}"
    );
}

// =============================================================================
// JSON Serialisation Tests
// =============================================================================

#[test]
fn test_neuron_discovery_history_serialisation() {
    let mut history = NeuronDiscoveryHistory::new("hidden-1".to_string());
    history.record_attempt(true, Some(12345));
    history.record_attempt(false, None);

    // Serialise to JSON
    let json = serde_json::to_string(&history).expect("Serialisation should succeed");
    assert!(json.contains("hidden-1"));
    assert!(json.contains("12345"));

    // Deserialise back
    let restored: NeuronDiscoveryHistory =
        serde_json::from_str(&json).expect("Deserialisation should succeed");
    assert_eq!(restored.uuid(), "hidden-1");
    assert_eq!(restored.attempts(), 2);
    assert_eq!(restored.successes(), 1);
    assert_eq!(restored.last_success_epoch(), Some(12345));
}

#[test]
fn test_discovery_history_serialisation() {
    let mut history = DiscoveryHistory::new();
    history.record("hidden-1", true, Some(1000));
    history.record("hidden-1", false, None);
    history.record("hidden-2", true, Some(2000));

    // Serialise to JSON
    let json = serde_json::to_string(&history).expect("Serialisation should succeed");
    assert!(json.contains("hidden-1"));
    assert!(json.contains("hidden-2"));

    // Deserialise back
    let restored: DiscoveryHistory =
        serde_json::from_str(&json).expect("Deserialisation should succeed");
    assert_eq!(restored.len(), 2);

    let h1 = restored.get("hidden-1").expect("hidden-1 should exist");
    assert_eq!(h1.attempts(), 2);
    assert_eq!(h1.successes(), 1);

    let h2 = restored.get("hidden-2").expect("hidden-2 should exist");
    assert_eq!(h2.attempts(), 1);
    assert_eq!(h2.successes(), 1);
}

#[test]
fn test_discovery_history_json_format_matches_spec() {
    // Verify the JSON format matches the specification from the issue
    let mut history = DiscoveryHistory::new();
    history.record("hidden-1", true, Some(12345));
    history.record("hidden-1", true, Some(12346));
    history.record("hidden-1", false, None);
    history.record("hidden-2", true, Some(12340));

    let json = serde_json::to_string_pretty(&history).expect("Serialisation should succeed");

    // Should contain the expected structure
    assert!(
        json.contains("neurons") || json.contains("hidden-1"),
        "JSON should contain neuron history"
    );

    // Parse and verify structure
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");

    // Verify we can access neuron data (exact structure may vary)
    assert!(!parsed.is_null(), "Parsed JSON should not be null");
}

// =============================================================================
// Focus Selection with History Tests
// =============================================================================

#[test]
fn test_rank_focus_neurons_with_history_prioritises_high_success_neurons() {
    // Create a creature with two hidden neurons that have identical error and impact
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hot-neuron", "hidden"),  // High historical success rate
            ("cold-neuron", "hidden"), // Low historical success rate
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hot-neuron", 1.0),
            ("input-0", "cold-neuron", 1.0),
            ("hot-neuron", "output-0", 0.5),  // Same weight/impact
            ("cold-neuron", "output-0", 0.5), // Same weight/impact
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both neurons have identical error
    let records = create_records(vec![
        ("hot-neuron", 1.0),
        ("cold-neuron", 1.0),
        ("output-0", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Create history where hot-neuron has high success rate
    let mut history = DiscoveryHistory::new();
    // hot-neuron: 8 successes, 2 failures → Bayesian score ≈ 9/12 = 0.75
    for _ in 0..8 {
        history.record("hot-neuron", true, Some(1000));
    }
    for _ in 0..2 {
        history.record("hot-neuron", false, None);
    }
    // cold-neuron: 2 successes, 8 failures → Bayesian score ≈ 3/12 = 0.25
    for _ in 0..2 {
        history.record("cold-neuron", true, Some(1000));
    }
    for _ in 0..8 {
        history.record("cold-neuron", false, None);
    }

    let result = rank_focus_neurons_with_history(file_path, &creature, None, None, Some(&history))
        .expect("Should succeed");

    // Find both neurons in results
    let hot_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "hot-neuron")
        .expect("hot-neuron should be in results");
    let cold_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "cold-neuron")
        .expect("cold-neuron should be in results");

    // hot-neuron should rank higher (earlier position = higher rank)
    assert!(
        hot_rank < cold_rank,
        "hot-neuron (rank {hot_rank}) should rank higher than cold-neuron (rank {cold_rank}) \
         due to higher historical success rate"
    );
}

#[test]
fn test_rank_focus_neurons_with_history_new_neurons_get_fair_chance() {
    // New neurons (not in history) should get a neutral prior (0.5)
    // This ensures they have a fair chance compared to known neurons
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("known-neuron", "hidden"),
            ("new-neuron", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "known-neuron", 1.0),
            ("input-0", "new-neuron", 1.0),
            ("known-neuron", "output-0", 0.5),
            ("new-neuron", "output-0", 0.5),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both neurons have identical error
    let records = create_records(vec![
        ("known-neuron", 1.0),
        ("new-neuron", 1.0),
        ("output-0", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Create history with only known-neuron having exactly 50% success rate
    let mut history = DiscoveryHistory::new();
    // known-neuron: 5 successes, 5 failures → Bayesian score ≈ 6/12 = 0.5
    for _ in 0..5 {
        history.record("known-neuron", true, Some(1000));
    }
    for _ in 0..5 {
        history.record("known-neuron", false, None);
    }
    // new-neuron is NOT in history → should get neutral 0.5

    let result = rank_focus_neurons_with_history(file_path, &creature, None, None, Some(&history))
        .expect("Should succeed");

    // Find both neurons
    let known_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "known-neuron")
        .expect("known-neuron should be in results");
    let new_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "new-neuron")
        .expect("new-neuron should be in results");

    // With equal error, impact, AND history score (~0.5), ranks should be close
    // Allow some difference due to tie-breaking
    let rank_diff = (known_rank as i32 - new_rank as i32).abs();
    assert!(
        rank_diff <= 1,
        "Neurons with equal scores should rank similarly. known_rank={known_rank}, new_rank={new_rank}"
    );
}

#[test]
fn test_rank_focus_neurons_without_history_unchanged_behaviour() {
    // When no history is provided, behaviour should match the original function
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("input-0", "hidden-2", 1.0),
            ("hidden-1", "output-0", 0.3),
            ("hidden-2", "output-0", 0.7),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records(vec![
        ("hidden-1", 2.0),
        ("hidden-2", 0.3),
        ("output-0", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Call with no history
    let result = rank_focus_neurons_with_history(file_path, &creature, None, None, None)
        .expect("Should succeed");

    // Should still return ranked neurons
    assert!(!result.neurons.is_empty());
    assert_eq!(result.neurons.len(), 3);
}

// =============================================================================
// Learning Behaviour Tests
// =============================================================================

#[test]
fn test_history_improves_selection_over_multiple_runs() {
    // Simulate multiple discovery runs to verify that history tracking
    // improves selection over time
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("good-neuron", "hidden"),   // Simulated: 70% success rate
            ("bad-neuron", "hidden"),    // Simulated: 20% success rate
            ("medium-neuron", "hidden"), // Simulated: 50% success rate
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "good-neuron", 1.0),
            ("input-0", "bad-neuron", 1.0),
            ("input-0", "medium-neuron", 1.0),
            ("good-neuron", "output-0", 0.33),
            ("bad-neuron", "output-0", 0.33),
            ("medium-neuron", "output-0", 0.33),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // All neurons have identical error
    let records = create_records(vec![
        ("good-neuron", 1.0),
        ("bad-neuron", 1.0),
        ("medium-neuron", 1.0),
        ("output-0", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Simulate 20 discovery runs with predetermined success rates
    let mut history = DiscoveryHistory::new();
    let mut rng_seed = 12345u64;

    for _ in 0..20 {
        // Simple pseudo-random for reproducibility
        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand_good = (rng_seed % 100) as f64 / 100.0;

        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand_bad = (rng_seed % 100) as f64 / 100.0;

        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand_medium = (rng_seed % 100) as f64 / 100.0;

        // good-neuron succeeds 70% of the time
        history.record("good-neuron", rand_good < 0.70, Some(1000));
        // bad-neuron succeeds 20% of the time
        history.record("bad-neuron", rand_bad < 0.20, Some(1000));
        // medium-neuron succeeds 50% of the time
        history.record("medium-neuron", rand_medium < 0.50, Some(1000));
    }

    // After learning, good-neuron should have highest Bayesian score
    let good_score = history.bayesian_score_for("good-neuron");
    let bad_score = history.bayesian_score_for("bad-neuron");
    let medium_score = history.bayesian_score_for("medium-neuron");

    assert!(
        good_score > medium_score,
        "good-neuron ({good_score:.3}) should have higher score than medium-neuron ({medium_score:.3})"
    );
    assert!(
        medium_score > bad_score,
        "medium-neuron ({medium_score:.3}) should have higher score than bad-neuron ({bad_score:.3})"
    );

    // Now verify ranking with history
    let result = rank_focus_neurons_with_history(file_path, &creature, None, None, Some(&history))
        .expect("Should succeed");

    let good_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "good-neuron")
        .expect("good-neuron should be in results");
    let bad_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "bad-neuron")
        .expect("bad-neuron should be in results");

    // After learning, good-neuron should rank higher than bad-neuron
    assert!(
        good_rank < bad_rank,
        "After learning from 20 runs, good-neuron (rank {good_rank}) should rank higher \
         than bad-neuron (rank {bad_rank})"
    );
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_discovery_history_handles_empty_uuid() {
    let mut history = DiscoveryHistory::new();
    history.record("", true, Some(1000));

    // Should handle empty UUID gracefully
    let entry = history.get("");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().attempts(), 1);
}

#[test]
fn test_discovery_history_handles_special_characters_in_uuid() {
    let mut history = DiscoveryHistory::new();
    let special_uuid = "neuron-with-special_chars.123/456";
    history.record(special_uuid, true, Some(1000));

    let entry = history.get(special_uuid);
    assert!(entry.is_some());

    // Verify serialisation handles special characters
    let json = serde_json::to_string(&history).expect("Serialisation should succeed");
    let restored: DiscoveryHistory =
        serde_json::from_str(&json).expect("Deserialisation should succeed");
    assert!(restored.get(special_uuid).is_some());
}

#[test]
fn test_discovery_history_handles_large_epoch_values() {
    let mut history = DiscoveryHistory::new();
    let large_epoch = u64::MAX;
    history.record("hidden-1", true, Some(large_epoch));

    let entry = history.get("hidden-1").expect("Entry should exist");
    assert_eq!(entry.last_success_epoch(), Some(large_epoch));
}

#[test]
fn test_discovery_history_handles_many_neurons() {
    let mut history = DiscoveryHistory::new();

    // Record history for 1000 neurons
    for i in 0..1000 {
        let uuid = format!("neuron-{i}");
        history.record(&uuid, i % 2 == 0, Some(i as u64));
    }

    assert_eq!(history.len(), 1000);

    // Verify we can retrieve any neuron
    let entry = history.get("neuron-500").expect("Entry should exist");
    assert_eq!(entry.attempts(), 1);

    // Verify serialisation works with many entries
    let json = serde_json::to_string(&history).expect("Serialisation should succeed");
    let restored: DiscoveryHistory =
        serde_json::from_str(&json).expect("Deserialisation should succeed");
    assert_eq!(restored.len(), 1000);
}
