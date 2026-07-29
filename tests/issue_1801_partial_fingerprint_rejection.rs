//! Issue #1801: a **partial** fingerprint-cache skip must be counted too.
//!
//! Issue #1781 made the *whole-pass* skip visible, but only when every focus
//! neuron was unchanged. The common case — the cache drops some focus neurons
//! and the pass proceeds with the rest — returned a `pass_rejection_breakdown`
//! built by `RejectionBreakdown::new()`, so the skipped neurons never reached
//! `candidate_starvation::classify` and their absence read as a gate rejection.
//!
//! These tests drive the real `analyze_all` entry point (never a hand-built
//! breakdown) so a refactor that reverts the wiring in
//! `src/analysis/orchestration.rs` goes red:
//!
//! 1. `partial_cache_hits_recorded_as_fingerprint_unchanged` — hits > 0 and
//!    misses > 0 surfaces `fingerprint_unchanged: hits`.
//! 2. `whole_pass_skip_counts_hits_exactly_once` — the #1781 early return still
//!    reports exactly `fingerprint_cache_hits`, never twice.
//! 3. `zero_cache_hits_yields_no_fingerprint_entry` — no spurious zero entry.

use std::collections::HashMap;

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_FINGERPRINT_UNCHANGED;
use neat_ai_discovery::analysis::neuron_fingerprint::{
    NeuronFingerprint, compute_neuron_fingerprints,
};
use neat_ai_discovery::analysis::shared::AnalyzeAllResult;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::TempDir;

/// The two focus neurons: `HIDDEN` is the one the cache is primed to skip.
const HIDDEN: &str = "hidden-0";
const OUTPUT: &str = "output-0";

fn build_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: HIDDEN.to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: OUTPUT.to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: HIDDEN.to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: HIDDEN.to_string(),
                to_uuid: OUTPUT.to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn build_records(creature: &CreatureJson, count: u32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count {
        let t = f32::from(u16::try_from(obs).expect("observation index fits u16"))
            / f32::from(u16::try_from(count).expect("record count fits u16"));
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![0.05 * (t * 3.0).cos()])
                }
                "output" => {
                    let act = 0.5 + 0.2 * t;
                    (
                        act,
                        Some(act),
                        vec![0.05 * (t * std::f32::consts::PI).cos()],
                    )
                }
                _ => continue,
            };
            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }
    records
}

/// Write the fixture recording once; the `TempDir` must outlive the pass.
fn write_fixture_parquet() -> (TempDir, String) {
    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let parquet_file = temp_dir
        .path()
        .join("issue_1801.parquet")
        .to_str()
        .expect("parquet path is valid UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records).expect("write parquet");
    (temp_dir, parquet_file)
}

/// Both hidden and output neurons are focus targets, so priming the cache with
/// a subset of them yields a genuine partial skip.
fn make_input(
    parquet_file: String,
    previous_fingerprints: Option<HashMap<String, NeuronFingerprint>>,
) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file,
        creature: build_creature(),
        focus_neurons: vec![HIDDEN.to_string(), OUTPUT.to_string()],
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: previous_fingerprints,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: None,
    }
}

/// Current fingerprints for `uuids` only — every other focus neuron is a miss.
fn fingerprints_for(uuids: &[&str]) -> HashMap<String, NeuronFingerprint> {
    let all = compute_neuron_fingerprints(&build_creature());
    uuids
        .iter()
        .map(|uuid| {
            let fingerprint = all
                .get(*uuid)
                .expect("fixture neuron has a fingerprint")
                .clone();
            ((*uuid).to_string(), fingerprint)
        })
        .collect()
}

fn fingerprint_unchanged_count(result: &AnalyzeAllResult) -> Option<u32> {
    result
        .pass_rejection_breakdown
        .counts()
        .get(REJECTION_FINGERPRINT_UNCHANGED)
        .copied()
}

/// A pass with one unchanged and one changed focus neuron must report the
/// unchanged one, not stay silent.
#[test]
fn partial_cache_hits_recorded_as_fingerprint_unchanged() {
    let (_temp_dir, parquet_file) = write_fixture_parquet();
    let input = make_input(parquet_file, Some(fingerprints_for(&[HIDDEN])));

    let result = analyze_all(&input).expect("analyze_all");

    assert_eq!(
        (
            result.fingerprint_cache_hits,
            result.fingerprint_cache_misses
        ),
        (1, 1),
        "the fixture must produce a genuinely partial skip — one hit, one miss"
    );
    assert_eq!(
        fingerprint_unchanged_count(&result),
        Some(1),
        "Issue #1801: a partial fingerprint skip must be counted as \
         `{REJECTION_FINGERPRINT_UNCHANGED}` on the pass breakdown. Without it the skipped focus \
         neuron never reaches `candidate_starvation::classify`, so its absence is misread as a \
         gate rejection."
    );
}

/// The #1781 whole-pass early return must still report exactly
/// `fingerprint_cache_hits` — the two return paths are mutually exclusive, so
/// the same hits can never be accumulated twice.
#[test]
fn whole_pass_skip_counts_hits_exactly_once() {
    let (_temp_dir, parquet_file) = write_fixture_parquet();
    let input = make_input(parquet_file, Some(fingerprints_for(&[HIDDEN, OUTPUT])));

    let result = analyze_all(&input).expect("analyze_all");

    assert_eq!(
        (
            result.fingerprint_cache_hits,
            result.fingerprint_cache_misses
        ),
        (2, 0),
        "every focus neuron must be a cache hit for the whole-pass early return"
    );
    assert_eq!(
        fingerprint_unchanged_count(&result),
        Some(2),
        "Issue #1781 regression: the whole-pass skip must report exactly \
         `fingerprint_cache_hits` — a larger count means the early return and the normal path \
         both accumulated the same hits."
    );
    assert_eq!(
        result.pass_rejection_breakdown.counts().len(),
        1,
        "the whole-pass skip records only the fingerprint reason"
    );
}

/// No cache hits must leave the key absent entirely — a zero-count entry would
/// shift the ratios `candidate_starvation::classify` reads.
#[test]
fn zero_cache_hits_yields_no_fingerprint_entry() {
    let (_temp_dir, parquet_file) = write_fixture_parquet();
    let input = make_input(parquet_file, None);

    let result = analyze_all(&input).expect("analyze_all");

    assert_eq!(
        result.fingerprint_cache_hits, 0,
        "with no previous fingerprints nothing can be skipped"
    );
    assert_eq!(
        fingerprint_unchanged_count(&result),
        None,
        "Issue #1801: a pass with zero cache hits must not record a \
         `{REJECTION_FINGERPRINT_UNCHANGED}` entry at all."
    );
}
