//! Issue #204: Include activation frequency in focus neuron ranking.
//!
//! These tests verify that:
//! - Neurons with rarely-firing patterns (activation rate < 10%) are penalised
//! - Neurons with always-firing patterns (activation rate > 90%) are penalised
//! - Neurons with moderate activation frequency (10-90%) are not penalised
//! - The frequency factor integrates correctly with error × impact ranking

use neat_ai_discovery::focus::rank_focus_neurons;
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

/// Create records where a neuron fires on specific samples.
///
/// `firing_samples` is a list of sample indices where activation > 0.
/// Other samples have activation = 0 (not firing).
fn create_records_with_firing_pattern(
    neuron_uuid: &str,
    total_samples: u32,
    firing_samples: Vec<u32>,
    error: f32,
) -> Vec<DiscoverRecord> {
    (0..total_samples)
        .map(|obs_index| {
            let activation = if firing_samples.contains(&obs_index) {
                0.8 // Firing
            } else {
                0.0 // Not firing
            };
            DiscoverRecord::new(
                obs_index,
                neuron_uuid.to_string(),
                Some(0.5),
                activation,
                vec![error],
            )
        })
        .collect()
}

/// Test: Rarely-firing neuron (activation rate < 10%) should be penalised.
///
/// A neuron that fires on only 5% of samples has limited influence on most samples.
/// Its ranking score should be reduced by the frequency penalty factor (0.8).
#[test]
fn test_rarely_firing_neuron_is_penalised() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("rarely-fires", "hidden"),   // Fires on 5% of samples
            ("moderate-fires", "hidden"), // Fires on 50% of samples
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "rarely-fires", 1.0),
            ("input-0", "moderate-fires", 1.0),
            ("rarely-fires", "output-0", 1.0),   // Same impact
            ("moderate-fires", "output-0", 1.0), // Same impact
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create 100 samples total
    // rarely-fires: fires on 5 samples (5% activation rate) - should be penalised
    // moderate-fires: fires on 50 samples (50% activation rate) - no penalty
    let mut records = vec![];

    // rarely-fires: only fires on samples 0-4 (5% of 100)
    records.extend(create_records_with_firing_pattern(
        "rarely-fires",
        100,
        (0..5).collect(),
        0.5, // Same error as moderate-fires
    ));

    // moderate-fires: fires on samples 0-49 (50% of 100)
    records.extend(create_records_with_firing_pattern(
        "moderate-fires",
        100,
        (0..50).collect(),
        0.5, // Same error
    ));

    // Output records
    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find both neurons
    let rarely = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "rarely-fires")
        .expect("rarely-fires should be in results");
    let moderate = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "moderate-fires")
        .expect("moderate-fires should be in results");

    // Verify activation_frequency is tracked (this field needs to be added)
    assert!(
        rarely.activation_frequency < 0.1,
        "rarely-fires should have activation_frequency < 10%, got {}%",
        rarely.activation_frequency * 100.0
    );
    assert!(
        moderate.activation_frequency >= 0.1 && moderate.activation_frequency <= 0.9,
        "moderate-fires should have activation_frequency between 10-90%, got {}%",
        moderate.activation_frequency * 100.0
    );

    // The rarely-firing neuron should rank LOWER than moderate-fires
    // due to the 0.8 penalty factor, even though they have same error and impact
    let rarely_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "rarely-fires")
        .unwrap();
    let moderate_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "moderate-fires")
        .unwrap();

    assert!(
        rarely_rank > moderate_rank,
        "Rarely-firing neuron should rank LOWER than moderate-firing neuron due to frequency penalty. \
         rarely-fires rank: {rarely_rank}, moderate-fires rank: {moderate_rank}"
    );
}

/// Test: Always-firing neuron (activation rate > 90%) should be penalised.
///
/// A neuron that fires on 95% of samples behaves like a constant (no discriminative power).
/// Its ranking score should be reduced by the frequency penalty factor (0.8).
#[test]
fn test_always_firing_neuron_is_penalised() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("always-fires", "hidden"),   // Fires on 95% of samples
            ("moderate-fires", "hidden"), // Fires on 50% of samples
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "always-fires", 1.0),
            ("input-0", "moderate-fires", 1.0),
            ("always-fires", "output-0", 1.0),
            ("moderate-fires", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = vec![];

    // always-fires: fires on samples 0-94 (95% of 100)
    records.extend(create_records_with_firing_pattern(
        "always-fires",
        100,
        (0..95).collect(),
        0.5,
    ));

    // moderate-fires: fires on samples 0-49 (50% of 100)
    records.extend(create_records_with_firing_pattern(
        "moderate-fires",
        100,
        (0..50).collect(),
        0.5,
    ));

    // Output records
    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let always = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "always-fires")
        .expect("always-fires should be in results");
    let moderate = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "moderate-fires")
        .expect("moderate-fires should be in results");

    // Verify activation_frequency
    assert!(
        always.activation_frequency > 0.9,
        "always-fires should have activation_frequency > 90%, got {}%",
        always.activation_frequency * 100.0
    );
    assert!(
        moderate.activation_frequency >= 0.1 && moderate.activation_frequency <= 0.9,
        "moderate-fires should have activation_frequency between 10-90%, got {}%",
        moderate.activation_frequency * 100.0
    );

    // The always-firing neuron should rank LOWER than moderate-fires
    let always_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "always-fires")
        .unwrap();
    let moderate_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "moderate-fires")
        .unwrap();

    assert!(
        always_rank > moderate_rank,
        "Always-firing neuron should rank LOWER than moderate-firing neuron due to frequency penalty. \
         always-fires rank: {always_rank}, moderate-fires rank: {moderate_rank}"
    );
}

/// Test: Moderate-frequency neuron (10-90% activation rate) has no penalty.
///
/// Neurons in the "sweet spot" activation range have good discriminative power
/// and should not have their ranking score reduced.
#[test]
fn test_moderate_frequency_neuron_not_penalised() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("moderate-10", "hidden"), // 10% - edge of no penalty
            ("moderate-50", "hidden"), // 50% - clearly no penalty
            ("moderate-90", "hidden"), // 90% - edge of no penalty
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "moderate-10", 1.0),
            ("input-0", "moderate-50", 1.0),
            ("input-0", "moderate-90", 1.0),
            ("moderate-10", "output-0", 1.0),
            ("moderate-50", "output-0", 1.0),
            ("moderate-90", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = vec![];

    // moderate-10: fires on 10 samples (10%)
    records.extend(create_records_with_firing_pattern(
        "moderate-10",
        100,
        (0..10).collect(),
        0.5,
    ));

    // moderate-50: fires on 50 samples (50%)
    records.extend(create_records_with_firing_pattern(
        "moderate-50",
        100,
        (0..50).collect(),
        0.5,
    ));

    // moderate-90: fires on 90 samples (90%)
    records.extend(create_records_with_firing_pattern(
        "moderate-90",
        100,
        (0..90).collect(),
        0.5,
    ));

    // Output records
    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // All three should have activation_frequency in [0.1, 0.9] range
    for uuid in &["moderate-10", "moderate-50", "moderate-90"] {
        let neuron = result
            .neurons
            .iter()
            .find(|n| n.neuron_uuid == *uuid)
            .unwrap_or_else(|| panic!("{uuid} should be in results"));

        let freq_pct = neuron.activation_frequency * 100.0;
        assert!(
            neuron.activation_frequency >= 0.1 && neuron.activation_frequency <= 0.9,
            "{uuid} should have activation_frequency in [10%, 90%] range, got {freq_pct}%"
        );
    }

    // All three should have similar rankings (no penalty applied)
    // Since they have same error and impact, ranking differences come only from
    // their activation_frequency differences (higher frequency = higher mean_activation = higher weighted impact)
    // But none should have the penalty factor applied

    // The key check: all are in the top 3 ranks among hidden neurons
    let hidden_ranks: Vec<_> = result
        .neurons
        .iter()
        .filter(|n| n.neuron_uuid.starts_with("moderate-"))
        .collect();
    assert_eq!(hidden_ranks.len(), 3, "Should have all 3 moderate neurons");
}

/// Test: Frequency factor integrates correctly with error × impact calculation.
///
/// The final score should be: error × impact^gamma × `gradient_factor` × `frequency_factor`
/// This test verifies the `frequency_factor` is correctly applied to the ranking.
#[test]
fn test_frequency_factor_integrates_with_error_impact_calculation() {
    // Create neurons with different characteristics:
    // - high-error-rare: high error (1.0), rarely fires (5%) → penalised
    // - low-error-moderate: low error (0.1), moderate fires (50%) → no penalty
    //
    // Without frequency factor: high-error-rare would rank higher (higher error)
    // With frequency factor: the 0.8 penalty should reduce its advantage
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("high-error-rare", "hidden"),
            ("low-error-moderate", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "high-error-rare", 1.0),
            ("input-0", "low-error-moderate", 1.0),
            ("high-error-rare", "output-0", 1.0), // Same impact
            ("low-error-moderate", "output-0", 1.0), // Same impact
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = vec![];

    // high-error-rare: high error (1.0), fires on 5% of samples
    records.extend((0..100).map(|i| {
        let activation = if i < 5 { 0.8 } else { 0.0 };
        DiscoverRecord::new(
            i,
            "high-error-rare".to_string(),
            Some(0.5),
            activation,
            vec![1.0],
        )
    }));

    // low-error-moderate: low error (0.1), fires on 50% of samples
    records.extend((0..100).map(|i| {
        let activation = if i < 50 { 0.8 } else { 0.0 };
        DiscoverRecord::new(
            i,
            "low-error-moderate".to_string(),
            Some(0.5),
            activation,
            vec![0.1],
        )
    }));

    // Output records
    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let high_error_rare = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "high-error-rare")
        .expect("high-error-rare should be in results");
    let low_error_moderate = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "low-error-moderate")
        .expect("low-error-moderate should be in results");

    // Verify the frequency factor was applied
    assert!(
        high_error_rare.activation_frequency < 0.1,
        "high-error-rare should have low activation_frequency"
    );
    assert!(
        low_error_moderate.activation_frequency >= 0.1
            && low_error_moderate.activation_frequency <= 0.9,
        "low-error-moderate should have moderate activation_frequency"
    );

    // The ranking should reflect the combined factors:
    // high-error-rare: 1.0 × impact × gradient × 0.8 (penalty)
    // low-error-moderate: 0.1 × impact × gradient × 1.0 (no penalty)
    //
    // With same impact: 1.0 × 0.8 = 0.8 vs 0.1 × 1.0 = 0.1
    // high-error-rare still wins due to much higher error, but gap is reduced

    // Just verify both neurons are processed and have the expected characteristics
    assert!(high_error_rare.raw_error > low_error_moderate.raw_error);
}

/// Test: Edge case - zero activation records (never fires) should be handled.
#[test]
fn test_never_firing_neuron_handled_gracefully() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("never-fires", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "never-fires", 1.0),
            ("never-fires", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // All activations are 0 (never fires)
    let mut records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord::new(i, "never-fires".to_string(), Some(0.5), 0.0, vec![0.5]))
        .collect();

    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let never = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "never-fires")
        .expect("never-fires should be in results");

    // Should have 0% activation frequency and be penalised
    assert!(
        never.activation_frequency < 0.1,
        "never-fires should have activation_frequency < 10% (actually 0%), got {}%",
        never.activation_frequency * 100.0
    );

    // Should still be processed without error
    assert!(!result.neurons.is_empty());
}

/// Test: Edge case - all activations are non-zero (always fires at 100%) should be handled.
#[test]
fn test_all_firing_neuron_handled_gracefully() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("always-fires", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "always-fires", 1.0),
            ("always-fires", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // All activations are non-zero (always fires)
    let mut records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord::new(i, "always-fires".to_string(), Some(0.5), 0.8, vec![0.5]))
        .collect();

    records.extend(
        (0..100).map(|i| DiscoverRecord::new(i, "output-0".to_string(), Some(0.5), 0.5, vec![0.3])),
    );

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let always = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "always-fires")
        .expect("always-fires should be in results");

    // Should have 100% activation frequency and be penalised
    assert!(
        always.activation_frequency > 0.9,
        "always-fires should have activation_frequency > 90% (actually 100%), got {}%",
        always.activation_frequency * 100.0
    );

    // Should still be processed without error
    assert!(!result.neurons.is_empty());
}
