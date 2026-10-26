//! Issue #2368 (CWE-400, Uncontrolled Resource Consumption): the skip-connection
//! detector's depth computation is a longest-path BFS that re-pushes a node on
//! every depth rise and keeps no visited set. On a cyclic creature the depth of
//! the cycle grows without bound, so `detect_skip_connection_candidates` never
//! returns. On unfixed code the call never returns for the creature below.
//!
//! This test is deliberately written against only the pre-existing public API
//! (`detect_skip_connection_candidates`) so that it compiles against the unfixed
//! base code. The detector runs on a detached thread and reports its result over
//! a channel, so a hang fails the test with a timeout instead of blocking the
//! whole test binary. After the fix, a cyclic creature has no well-defined
//! depths, so the detector returns promptly with no candidates.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activation waveforms (Issue #873)

mod common;

use std::sync::mpsc;
use std::time::Duration;

use common::{hidden, make_creature, neuron, output, synapse};
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::detection::skip_connection::{
    SkipConnectionCandidate, detect_skip_connection_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Build `count` records for a neuron with a single error value that is
/// modulated per sample, so the mean absolute error is `error` on average.
fn make_records(uuid: &str, count: u32, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: (i as f32 * 0.1).sin(),
            errors: vec![error * (1.0 + (i as f32 * 0.05).sin())],
        })
        .collect()
}

/// Creature with a back edge: in-0 -> h0 -> h1 -> h2 -> h3 -> h4 -> out-0,
/// plus h4 -> h3 closing a cycle between h3 and h4.
fn cyclic_creature() -> CreatureJson {
    make_creature(
        vec![
            neuron("in-0", "input", "IDENTITY"),
            hidden("h0", "TANH"),
            hidden("h1", "TANH"),
            hidden("h2", "TANH"),
            hidden("h3", "TANH"),
            hidden("h4", "TANH"),
            output("out-0", "TANH"),
        ],
        vec![
            synapse("in-0", "h0", 0.5),
            synapse("h0", "h1", 0.5),
            synapse("h1", "h2", 0.5),
            synapse("h2", "h3", 0.5),
            synapse("h3", "h4", 0.5),
            synapse("h4", "h3", 0.5),
            synapse("h4", "out-0", 0.5),
        ],
    )
}

/// Guards against CWE-400: a cyclic creature must not make the skip-connection
/// detector loop forever. Cyclic graphs have no well-defined depths, so the
/// detector must return promptly and propose nothing.
#[test]
fn cyclic_creature_returns_without_hanging_and_proposes_nothing() {
    let creature = cyclic_creature();
    let errors = [
        ("h0", 0.8_f32),
        ("h1", 0.6),
        ("h2", 0.08),
        ("h3", 0.05),
        ("h4", 0.03),
    ];
    let records: Vec<(String, Vec<DiscoverRecord>)> = errors
        .iter()
        .map(|&(uuid, error)| (uuid.to_string(), make_records(uuid, 30, error)))
        .collect();

    let (tx, rx) = mpsc::channel::<Vec<SkipConnectionCandidate>>();
    std::thread::spawn(move || {
        let candidates = detect_skip_connection_candidates(&creature, &records);
        let _ = tx.send(candidates);
    });

    let candidates = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("detect_skip_connection_candidates must return on a cyclic creature");
    assert!(
        candidates.is_empty(),
        "a cyclic creature has no well-defined depths, so no skip connections may be proposed, got {} candidates",
        candidates.len()
    );
}
