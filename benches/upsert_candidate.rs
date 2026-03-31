//! Benchmark for Issue #526: Clone reduction in `upsert_candidate` hot path.
//!
//! Measures the cost of `upsert_candidate()` which is called for every neuron
//! candidate in the analysis pipeline. The key optimisation is replacing
//! String-based `HashMap` keys with pre-computed hash keys to avoid 3 String
//! clones per candidate insertion.

#![allow(clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::CandidateNeuronJson;
use std::collections::HashMap;
use std::hint::black_box;

/// Create a realistic candidate for benchmarking.
fn make_candidate(source_idx: usize, target_idx: usize, squash: &str) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: format!("input-{source_idx}"),
        target_neuron_uuid: format!("hidden-{target_idx}"),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 0.5,
        squash: squash.to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.05,
        expected_creature_score_gain: 0.05,
        improved_count: 80,
        total_count: 100,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.01, 0.09],
    }
}

/// Benchmark: `upsert_candidate` with String-based keys (current approach).
fn bench_upsert_candidate(c: &mut Criterion) {
    let mut group = c.benchmark_group("upsert_candidate");

    // Typical: 100 candidates from 10 sources × 10 targets × 1 squash
    let candidates_100: Vec<CandidateNeuronJson> = (0..10)
        .flat_map(|src| (0..10).map(move |tgt| make_candidate(src, tgt, "ReLU")))
        .collect();

    group.bench_function("100_candidates", |b| {
        b.iter(|| {
            let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
                HashMap::new();
            for candidate in &candidates_100 {
                let key = (
                    candidate.source_neuron_uuid.clone(),
                    candidate.target_neuron_uuid.clone(),
                    candidate.squash.clone(),
                    if candidate.incoming_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                    if candidate.outgoing_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                );
                map.entry(key).or_insert_with(|| candidate.clone());
            }
            black_box(&map);
        });
    });

    // Larger: 1000 candidates from 100 sources × 10 targets
    let candidates_1000: Vec<CandidateNeuronJson> = (0..100)
        .flat_map(|src| (0..10).map(move |tgt| make_candidate(src, tgt, "ReLU")))
        .collect();

    group.bench_function("1000_candidates", |b| {
        b.iter(|| {
            let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
                HashMap::new();
            for candidate in &candidates_1000 {
                let key = (
                    candidate.source_neuron_uuid.clone(),
                    candidate.target_neuron_uuid.clone(),
                    candidate.squash.clone(),
                    if candidate.incoming_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                    if candidate.outgoing_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                );
                map.entry(key).or_insert_with(|| candidate.clone());
            }
            black_box(&map);
        });
    });

    // Hash-based key approach (the optimisation)
    group.bench_function("100_candidates_hash_key", |b| {
        b.iter(|| {
            let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();
            for candidate in &candidates_100 {
                let key = compute_candidate_hash_key(
                    &candidate.source_neuron_uuid,
                    &candidate.target_neuron_uuid,
                    &candidate.squash,
                    if candidate.incoming_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                    if candidate.outgoing_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                );
                map.entry(key).or_insert_with(|| candidate.clone());
            }
            black_box(&map);
        });
    });

    group.bench_function("1000_candidates_hash_key", |b| {
        b.iter(|| {
            let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();
            for candidate in &candidates_1000 {
                let key = compute_candidate_hash_key(
                    &candidate.source_neuron_uuid,
                    &candidate.target_neuron_uuid,
                    &candidate.squash,
                    if candidate.incoming_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                    if candidate.outgoing_weight > 0.0 {
                        1i8
                    } else {
                        -1i8
                    },
                );
                map.entry(key).or_insert_with(|| candidate.clone());
            }
            black_box(&map);
        });
    });

    group.finish();
}

/// FNV-1a hash for candidate deduplication key (same as production code uses for UUIDs).
fn compute_candidate_hash_key(
    source_uuid: &str,
    target_uuid: &str,
    squash: &str,
    incoming_sign: i8,
    outgoing_sign: i8,
) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in source_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Separator to avoid key collisions between "input-1" + "hidden-2" and "input-12" + "hidden-"
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    for b in target_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    for b in squash.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= incoming_sign as u8 as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= outgoing_sign as u8 as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    hash
}

criterion_group!(benches, bench_upsert_candidate);
criterion_main!(benches);
