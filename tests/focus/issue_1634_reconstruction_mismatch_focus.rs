//! Issue #1634: reconstruction-mismatch focus signal.
//!
//! Focus selection should prioritise neurons the current model of the creature
//! fails to explain — those whose recorded activation cannot be reconstructed
//! from their inbound synapses (`squash(bias + Σ from_activation × weight)`).
//!
//! These tests build a synthetic creature with two hidden neurons that are
//! byte-identical in everything the legacy focus score sees (records, error,
//! impact, gradient, frequency) and differ **only** in bias — so their
//! reconstruction deltas differ while their base scores tie exactly. This
//! isolates the new additive signal:
//!
//! 1. With the signal enabled the high-mismatch neuron ranks first (it ties on
//!    the old key today, so the old path orders them by UUID — a genuine flip).
//! 2. The well-reconstructed neuron gains ~no boost (mismatch ≈ 0).
//! 3. Under a focus budget of one, the high-mismatch neuron is the survivor.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::config::{
    DEFAULT_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT, resolve_focus_reconstruction_mismatch_weight,
};
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const ENABLE_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH";

/// RAII guard that sets/restores an env var for a single (serialised) test.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

const OBS: u32 = 4;
/// Constant input activation feeding both hidden neurons.
const INPUT_ACTIVATION: f32 = 0.5;

fn neuron(uuid: &str, ntype: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: ntype.to_string(),
        squash: squash.to_string(),
        bias,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Two hidden neurons, identical in every legacy-score input, differing only in
/// bias so their reconstruction deltas differ:
/// - `h-good` bias 0.0 → reconstruct tanh(0.5) == its recorded activation (Δ≈0)
/// - `h-poor` bias 1.5 → reconstruct tanh(2.0) ≠ its recorded activation (Δ large)
fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input", "IDENTITY", 0.0),
            neuron("h-good", "hidden", "TANH", 0.0),
            neuron("h-poor", "hidden", "TANH", 1.5),
            neuron("out", "output", "LOGISTIC", 0.0),
        ],
        synapses: vec![
            synapse("in-0", "h-good", 1.0),
            synapse("in-0", "h-poor", 1.0),
            synapse("h-good", "out", 1.0),
            synapse("h-poor", "out", 1.0),
        ],
        input: 1,
        output: 1,
    }
}

fn records() -> Vec<DiscoverRecord> {
    // Both hidden neurons carry IDENTICAL recordings so their base focus scores
    // tie exactly; only bias (hence reconstruction) differs.
    let hidden_activation = INPUT_ACTIVATION.tanh(); // tanh(0.5) ≈ 0.4621
    let mut out = Vec::new();
    for obs in 0..OBS {
        out.push(DiscoverRecord {
            obs_index: obs,
            neuron_uuid: "in-0".to_string(),
            value: Some(INPUT_ACTIVATION),
            activation: INPUT_ACTIVATION,
            errors: vec![],
        });
        for hid in ["h-good", "h-poor"] {
            out.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: hid.to_string(),
                value: Some(INPUT_ACTIVATION),
                activation: hidden_activation,
                errors: vec![0.1],
            });
        }
        out.push(DiscoverRecord {
            obs_index: obs,
            neuron_uuid: "out".to_string(),
            value: Some(0.3),
            activation: 0.6,
            errors: vec![0.2],
        });
    }
    out
}

fn write_parquet() -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), &records()).expect("write parquet");
    tmp
}

fn rank_index(stats: &neat_ai_discovery::focus::RankFocusStats, uuid: &str) -> usize {
    stats
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == uuid)
        .unwrap_or_else(|| panic!("{uuid} must be present in the ranked pool"))
}

/// TDD 1: with the signal enabled, the high-mismatch neuron ranks above the
/// well-reconstructed one — a flip relative to the disabled path, where the two
/// tie on the legacy score and fall back to UUID order (`h-good` < `h-poor`).
#[test]
#[serial]
fn high_mismatch_neuron_ranks_first_when_enabled() {
    let creature = make_creature();
    let parquet = write_parquet();
    let path = parquet.path().to_str().unwrap();

    // Disabled (default): the two hidden neurons tie on the legacy score, so the
    // UUID tie-break puts h-good ahead of h-poor. Force the env off for this run.
    let disabled = {
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(ENABLE_ENV) };
        rank_focus_neurons(path, &creature, None, None).expect("ranking should succeed")
    };
    assert!(
        rank_index(&disabled, "h-good") < rank_index(&disabled, "h-poor"),
        "with the signal disabled the tie must fall back to UUID order (h-good first)",
    );

    // Enabled: the high-mismatch neuron (h-poor) must overtake h-good.
    let enabled = {
        let _guard = EnvVarGuard::set(ENABLE_ENV, "1");
        rank_focus_neurons(path, &creature, None, None).expect("ranking should succeed")
    };
    assert!(
        rank_index(&enabled, "h-poor") < rank_index(&enabled, "h-good"),
        "with the signal enabled the high-mismatch neuron (h-poor) must rank first",
    );
}

/// TDD 2: the well-reconstructed neuron is not boosted — its mismatch is ~0.
#[test]
#[serial]
fn well_reconstructed_neuron_not_boosted() {
    let creature = make_creature();
    let parquet = write_parquet();
    let path = parquet.path().to_str().unwrap();

    let _guard = EnvVarGuard::set(ENABLE_ENV, "1");
    let stats = rank_focus_neurons(path, &creature, None, None).expect("ranking should succeed");

    let good = &stats.neurons[rank_index(&stats, "h-good")];
    let poor = &stats.neurons[rank_index(&stats, "h-poor")];

    assert!(
        good.reconstruction_mismatch < 0.01,
        "well-reconstructed neuron must have ~0 mismatch, got {}",
        good.reconstruction_mismatch,
    );
    assert!(
        poor.reconstruction_mismatch > 0.4,
        "poorly-reconstructed neuron must have a large mismatch, got {}",
        poor.reconstruction_mismatch,
    );
    assert!(
        poor.weighted_score > good.weighted_score,
        "the high-mismatch neuron must score above the well-reconstructed one",
    );
}

/// TDD 3: under a limited focus budget, the well-reconstructed neuron is the one
/// dropped — the high-mismatch neuron is retained. The output neuron ("out")
/// legitimately ranks above both hidden neurons (higher error and impact), so we
/// budget for the top two and assert the *surviving hidden* neuron is h-poor,
/// with the well-reconstructed h-good dropped.
#[test]
#[serial]
fn budget_drops_well_reconstructed_neuron_first() {
    let creature = make_creature();
    let parquet = write_parquet();
    let path = parquet.path().to_str().unwrap();

    let _guard = EnvVarGuard::set(ENABLE_ENV, "1");
    // Budget of two: keeps the output neuron and exactly one hidden neuron.
    let stats = rank_focus_neurons(path, &creature, Some(2), None).expect("ranking should succeed");

    assert_eq!(
        stats.neurons.len(),
        2,
        "a budget of two must retain exactly two ranked neurons",
    );
    let retained: Vec<&str> = stats
        .neurons
        .iter()
        .map(|n| n.neuron_uuid.as_str())
        .collect();
    assert!(
        retained.contains(&"h-poor"),
        "the high-mismatch neuron must be retained under budget, got {retained:?}",
    );
    assert!(
        !retained.contains(&"h-good"),
        "the well-reconstructed neuron must be dropped first, got {retained:?}",
    );
}

/// The pure weight resolver falls back to the default for absent, invalid, and
/// negative inputs, accepts valid non-negative values, and treats zero as a
/// disabled-but-present weight (Issue #1634).
#[test]
fn weight_resolver_validates_input() {
    let default = DEFAULT_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT;

    // Absent / empty / non-numeric → default.
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(None, default),
        default
    );
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some(""), default),
        default
    );
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some("abc"), default),
        default
    );

    // Negative and non-finite → default (a negative weight would invert the signal).
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some("-0.5"), default),
        default
    );
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some("inf"), default),
        default
    );
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some("NaN"), default),
        default
    );

    // Valid non-negative values (including zero) are honoured.
    assert!(
        (resolve_focus_reconstruction_mismatch_weight(Some("0.25"), default) - 0.25).abs() < 1e-9
    );
    assert_eq!(
        resolve_focus_reconstruction_mismatch_weight(Some("0.0"), default),
        0.0
    );
    assert!(
        (resolve_focus_reconstruction_mismatch_weight(Some(" 1.5 "), default) - 1.5).abs() < 1e-9
    );
}
