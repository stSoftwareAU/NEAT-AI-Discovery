//! Tests for Issue #210: Neuron UUID string interning
//!
//! These tests verify that the NeuronIndex interning mechanism works correctly
//! in the context of synapse analysis, ensuring no functional regressions while
//! providing memory efficiency improvements.

use neat_ai_discovery::intern::NeuronIndex;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::{HashMap, HashSet};

/// Helper to create a test creature with specified neuron and synapse counts.
fn create_test_creature(neuron_count: usize, synapse_count: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(neuron_count);
    let mut synapses = Vec::with_capacity(synapse_count);

    // Create neurons
    for i in 0..neuron_count {
        let neuron_type = if i < neuron_count / 2 {
            "hidden"
        } else {
            "output"
        };
        neurons.push(NeuronJson {
            uuid: format!("neuron-{i}"),
            neuron_type: neuron_type.to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Create synapses connecting inputs to neurons and neurons to each other
    let input_count = 10;
    for i in 0..synapse_count {
        let from_uuid = if i % 3 == 0 {
            format!("input-{}", i % input_count)
        } else {
            format!("neuron-{}", i % neuron_count)
        };
        let to_uuid = format!("neuron-{}", (i + 1) % neuron_count);
        synapses.push(SynapseJson {
            from_uuid,
            to_uuid,
            weight: 0.5,
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: neuron_count / 2,
    }
}

#[test]
fn test_neuron_index_basic_functionality() {
    let mut index = NeuronIndex::new();

    // Test basic interning
    let idx1 = index.intern("neuron-1");
    let idx2 = index.intern("neuron-2");
    let idx3 = index.intern("neuron-1"); // Same as idx1

    assert_eq!(idx1, idx3, "Same UUID should return same index");
    assert_ne!(
        idx1, idx2,
        "Different UUIDs should return different indices"
    );

    // Test round-trip
    assert_eq!(index.get_uuid(idx1), Some("neuron-1"));
    assert_eq!(index.get_uuid(idx2), Some("neuron-2"));
    assert_eq!(index.get_index("neuron-1"), Some(idx1));
    assert_eq!(index.get_index("neuron-2"), Some(idx2));
}

#[test]
fn test_neuron_index_with_creature_uuids() {
    let creature = create_test_creature(100, 500);
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Intern all input UUIDs
    for i in 0..creature.input {
        index.intern(&format!("input-{i}"));
    }

    // Intern all neuron UUIDs
    for neuron in &creature.neurons {
        index.intern(&neuron.uuid);
    }

    // Verify all UUIDs are interned correctly
    assert_eq!(index.len(), creature.input + creature.neurons.len());

    // Verify round-trip for a sample of neurons
    for neuron in creature.neurons.iter().take(10) {
        let idx = index
            .get_index(&neuron.uuid)
            .expect("UUID should be interned");
        let uuid = index.get_uuid(idx).expect("Index should be valid");
        assert_eq!(uuid, neuron.uuid);
    }
}

#[test]
fn test_interned_synapse_lookup() {
    let creature = create_test_creature(50, 200);
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Intern all UUIDs from creature
    for i in 0..creature.input {
        index.intern(&format!("input-{i}"));
    }
    for neuron in &creature.neurons {
        index.intern(&neuron.uuid);
    }
    for synapse in &creature.synapses {
        index.intern(&synapse.from_uuid);
        index.intern(&synapse.to_uuid);
    }

    // Build existing synapses set using interned indices
    let existing_synapses: HashSet<(u32, u32)> = creature
        .synapses
        .iter()
        .map(|s| {
            (
                index.get_index(&s.from_uuid).unwrap(),
                index.get_index(&s.to_uuid).unwrap(),
            )
        })
        .collect();

    // Verify lookups work correctly
    for synapse in &creature.synapses {
        let from_idx = index.get_index(&synapse.from_uuid).unwrap();
        let to_idx = index.get_index(&synapse.to_uuid).unwrap();
        assert!(
            existing_synapses.contains(&(from_idx, to_idx)),
            "Synapse ({}, {}) should be in the set",
            synapse.from_uuid,
            synapse.to_uuid
        );
    }
}

#[test]
fn test_interned_synapse_weights_lookup() {
    let creature = create_test_creature(50, 200);
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Intern all UUIDs
    for synapse in &creature.synapses {
        index.intern(&synapse.from_uuid);
        index.intern(&synapse.to_uuid);
    }

    // Build weights map using interned indices
    let existing_weights: HashMap<(u32, u32), f32> = creature
        .synapses
        .iter()
        .map(|s| {
            (
                (
                    index.get_index(&s.from_uuid).unwrap(),
                    index.get_index(&s.to_uuid).unwrap(),
                ),
                s.weight,
            )
        })
        .collect();

    // Verify weight lookups work correctly
    for synapse in &creature.synapses {
        let from_idx = index.get_index(&synapse.from_uuid).unwrap();
        let to_idx = index.get_index(&synapse.to_uuid).unwrap();
        let weight = existing_weights.get(&(from_idx, to_idx)).copied();
        assert_eq!(
            weight,
            Some(synapse.weight),
            "Weight lookup should match for synapse ({}, {})",
            synapse.from_uuid,
            synapse.to_uuid
        );
    }
}

#[test]
fn test_synapses_by_target_interned() {
    let creature = create_test_creature(50, 200);
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Intern all UUIDs
    for synapse in &creature.synapses {
        index.intern(&synapse.from_uuid);
        index.intern(&synapse.to_uuid);
    }

    // Build synapses_by_target using interned index as key
    let synapses_by_target: HashMap<u32, Vec<SynapseJson>> = creature
        .synapses
        .iter()
        .map(|s| (index.get_index(&s.to_uuid).unwrap(), s.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Verify lookups work correctly
    for neuron in &creature.neurons {
        let idx = index.get_index(&neuron.uuid).unwrap();
        let synapses_for_target = synapses_by_target.get(&idx);

        // Count expected synapses to this target
        let expected_count = creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == neuron.uuid)
            .count();

        match synapses_for_target {
            Some(synapses) => assert_eq!(
                synapses.len(),
                expected_count,
                "Synapse count should match for target {}",
                neuron.uuid
            ),
            None => assert_eq!(
                expected_count, 0,
                "Should have no synapses for target {}",
                neuron.uuid
            ),
        }
    }
}

#[test]
fn test_memory_efficiency_comparison() {
    // This test demonstrates the memory savings from using interned indices
    // instead of String-based keys.

    let synapse_count = 10_000;
    let uuid_length = 36; // Typical UUID v4 length

    // Calculate memory for String-based approach
    // HashSet<(String, String)> = 10,000 × (2 × String overhead + 2 × 36 bytes data)
    // String on 64-bit: 24 bytes (ptr + len + cap) + heap data
    let string_overhead_per_pair = 2 * (24 + uuid_length);
    let string_total_bytes = synapse_count * string_overhead_per_pair;

    // Calculate memory for interned approach
    // HashSet<(u32, u32)> = 10,000 × 8 bytes
    // Plus one-time cost of interning unique UUIDs
    let interned_per_pair = 8; // 2 × u32
    let interned_total_bytes = synapse_count * interned_per_pair;

    let savings_percent =
        ((string_total_bytes - interned_total_bytes) as f64 / string_total_bytes as f64) * 100.0;

    // Memory for synapse pairs should be significantly reduced
    assert!(
        savings_percent > 85.0,
        "Expected >85% memory reduction for synapse pairs, got {savings_percent:.1}%"
    );

    // Print statistics for documentation
    let string_kb = string_total_bytes / 1024;
    let interned_kb = interned_total_bytes / 1024;
    eprintln!("Memory comparison for {synapse_count} synapses:");
    eprintln!("  String-based: {string_kb} KB ({string_overhead_per_pair} bytes per pair)");
    eprintln!("  Interned:     {interned_kb} KB ({interned_per_pair} bytes per pair)");
    eprintln!("  Savings:      {savings_percent:.1}%");
}

#[test]
fn test_large_creature_interning_performance() {
    // Test with a creature size mentioned in the issue (500 neurons, 10,000 synapses)
    let creature = create_test_creature(500, 10_000);
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Intern all UUIDs
    for i in 0..creature.input {
        index.intern(&format!("input-{i}"));
    }
    for neuron in &creature.neurons {
        index.intern(&neuron.uuid);
    }
    for synapse in &creature.synapses {
        index.intern(&synapse.from_uuid);
        index.intern(&synapse.to_uuid);
    }

    // Build the same data structures as in implementation.rs
    let existing_synapses: HashSet<(u32, u32)> = creature
        .synapses
        .iter()
        .map(|s| {
            (
                index.get_index(&s.from_uuid).unwrap(),
                index.get_index(&s.to_uuid).unwrap(),
            )
        })
        .collect();

    let existing_weights: HashMap<(u32, u32), f32> = creature
        .synapses
        .iter()
        .map(|s| {
            (
                (
                    index.get_index(&s.from_uuid).unwrap(),
                    index.get_index(&s.to_uuid).unwrap(),
                ),
                s.weight,
            )
        })
        .collect();

    let synapses_by_target: HashMap<u32, Vec<SynapseJson>> = creature
        .synapses
        .iter()
        .map(|s| (index.get_index(&s.to_uuid).unwrap(), s.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Verify data integrity
    // Note: We may have fewer unique synapses than total if there are duplicates
    // in the generated test data (same from->to pair appears multiple times)
    assert!(
        existing_synapses.len() <= creature.synapses.len(),
        "Unique synapses ({}) should be <= total synapses ({})",
        existing_synapses.len(),
        creature.synapses.len()
    );
    assert!(
        existing_weights.len() <= creature.synapses.len(),
        "Unique weights ({}) should be <= total synapses ({})",
        existing_weights.len(),
        creature.synapses.len()
    );

    // Total targets should be <= number of neurons
    assert!(synapses_by_target.len() <= creature.neurons.len());

    eprintln!("Large creature test completed:");
    eprintln!("  Neurons: {}", creature.neurons.len());
    eprintln!("  Synapses: {}", creature.synapses.len());
    eprintln!("  Unique interned UUIDs: {}", index.len());
    eprintln!("  Unique targets: {}", synapses_by_target.len());
}
