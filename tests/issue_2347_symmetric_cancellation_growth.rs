//! Issue #2347: the symmetric-cancellation pair scan must not grow quadratically
//! in the fan-in of a single target.
//!
//! This file deliberately uses only the legacy public surface
//! (`detect_symmetric_cancellation`, `WeightCoherenceConfig`) so it compiles and
//! can be shown red at the base branch, before the fan-in cap exists.
//!
//! The fixture is one target with `n` sources alternating `+5.0` / `-5.0`, all
//! sharing one activation pattern. At base the 512-source target yields
//! 256 * 256 = 65,536 candidates (quadratic, with no ceiling), so the output
//! bound fails. After the fix the over-cap target (fan-in cap 256) is skipped
//! entirely and yields 0 candidates, so the bound holds.
//!
//! The check is deterministic and output-based: it counts the candidates the
//! scan emits, never wall-clock time.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)
mod common;

use common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::weight_coherence::{
    WeightCoherenceConfig, detect_symmetric_cancellation,
};
use neat_ai_discovery::types::DiscoverRecord;

const RECORDS_PER_SOURCE: u32 = 25;

/// A single target neuron ("target-1") with `n` input sources, alternating
/// `+5.0` / `-5.0` weights so every cross-parity pair is an opposite-sign,
/// equal-magnitude pair (the only pairs symmetric cancellation can flag).
/// Every source shares the same activation pattern, so every cross-parity
/// pair is perfectly correlated.
fn fanin_fixture(
    n: usize,
) -> (
    neat_ai_discovery::CreatureJson,
    Vec<(String, Vec<DiscoverRecord>)>,
) {
    let mut neurons = vec![neuron("target-1", "output", "IDENTITY")];
    let mut synapses = Vec::with_capacity(n);
    let mut records = Vec::with_capacity(n);

    for i in 0..n {
        let uuid = format!("input-{i}");
        neurons.push(neuron(&uuid, "input", "IDENTITY"));
        let weight = if i % 2 == 0 { 5.0 } else { -5.0 };
        synapses.push(synapse(&uuid, "target-1", weight));

        let source_records: Vec<DiscoverRecord> = (0..RECORDS_PER_SOURCE)
            .map(|idx| {
                let activation = ((idx as f32 * 0.1).sin()).tanh();
                DiscoverRecord {
                    obs_index: idx,
                    neuron_uuid: uuid.clone(),
                    value: Some(activation),
                    activation,
                    errors: vec![0.1],
                }
            })
            .collect();
        records.push((uuid, source_records));
    }

    (make_creature(neurons, synapses), records)
}

// =============================================================================
// Work grows no faster than the fan-in cap allows.
// =============================================================================

#[test]
fn symmetric_cancellation_work_grows_no_faster_than_the_cap_allows() {
    let config = WeightCoherenceConfig::default();

    // Under the fan-in cap (256): scanned, and bounded by the candidate ceiling.
    let n: usize = 128;
    let (creature_n, records_n) = fanin_fixture(n);
    let candidates_n = detect_symmetric_cancellation(&creature_n, &records_n, &config, None).len();

    // Four times the fan-in: over the cap (256), so the target must be skipped.
    let four_n: usize = 512;
    let (creature_4n, records_4n) = fanin_fixture(four_n);
    let candidates_4n =
        detect_symmetric_cancellation(&creature_4n, &records_4n, &config, None).len();

    // Mirrors MAX_SYMMETRIC_CANCELLATION_CANDIDATES (1,024): the output for a
    // single target is capped regardless of how many sources it has.
    assert!(
        candidates_4n <= 1024,
        "output must stay within the candidate ceiling of 1024 at fan-in {four_n}: \
         got {candidates_4n} candidates (fan-in {n} gave {candidates_n})"
    );
    assert!(
        candidates_4n <= 4 * candidates_n,
        "output must not grow faster than 4x when fan-in grows 4x: \
         fan-in {four_n} gave {candidates_4n} candidates vs 4x{candidates_n} at fan-in {n}"
    );
}

// =============================================================================
// Control: the skip is not just returning nothing everywhere.
// =============================================================================

#[test]
fn fanin_under_the_cap_still_yields_candidates() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let candidates = detect_symmetric_cancellation(&creature, &records, &config, None);

    assert!(
        !candidates.is_empty(),
        "a target under the fan-in cap must still yield candidates, got none"
    );
}
