//! Issue #1635: near-zero-impact neurons must not consume focus slots.
//!
//! Beyond the functionally-constant hidden neurons already excluded by #1624, a
//! large fraction of neurons have a **near-zero downstream impact magnitude**
//! while still being non-constant (their activation varies across samples), so
//! the constant filter leaves them in the focus pool. No add-synapse /
//! add-neuron change feeding such a neuron can move the output, so every focus
//! slot it occupies is wasted.
//!
//! When `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE` is enabled, focus ranking drops
//! neurons whose structural impact magnitude is below
//! `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD` (default `1e-6`). These tests
//! assert the gate skips the low-impact neurons, keeps the high-impact one,
//! reports the gated count (never silently swallowed), and — with the gate off —
//! confirms the low-impact neurons are genuinely non-constant yet sub-gate.

use neat_ai_discovery::config::{
    DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD, resolve_focus_impact_gate_threshold,
};
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const FLAG_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE";
const THRESHOLD_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD";

/// RAII guard that sets/restores an env var for a single test. All env
/// manipulation is serialised via `#[serial]`.
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

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(key) };
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

/// Creature: one input feeds a high-impact hidden neuron (large weight to the
/// output) and three low-impact hidden neurons (tiny weight to the output).
/// Structural impact is the normalised weight fraction, so the low-weight
/// neurons carry `|impact|` far below the `1e-6` gate while still varying across
/// samples.
fn make_creature() -> CreatureJson {
    let neuron = |uuid: &str, ntype: &str| NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: ntype.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    };
    let synapse = |from: &str, to: &str, w: f32| SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: w,
        synapse_type: None,
    };
    CreatureJson {
        neurons: vec![
            neuron("h-high", "hidden"),
            neuron("h-low-a", "hidden"),
            neuron("h-low-b", "hidden"),
            neuron("h-low-c", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-high", 0.5),
            synapse("in-0", "h-low-a", 0.5),
            synapse("in-0", "h-low-b", 0.5),
            synapse("in-0", "h-low-c", 0.5),
            // High-impact path: dominant weight into the output.
            synapse("h-high", "out", 1.0),
            // Near-zero-impact paths: tiny weight into the output. Their
            // normalised impact fraction is ~1e-8 ≪ 1e-6 gate.
            synapse("h-low-a", "out", 1e-8),
            synapse("h-low-b", "out", 1e-8),
            synapse("h-low-c", "out", 1e-8),
        ],
        input: 1,
        output: 1,
    }
}

/// Records: every hidden neuron and the output vary across observations, so none
/// are functionally constant — the constant filter (#1624) would leave them all
/// in the focus pool. Only the impact gate removes the low-impact ones.
fn make_records() -> Vec<DiscoverRecord> {
    let count = 12u32;
    let mut records = Vec::new();
    for i in 0..count {
        let vary = f32::from(u16::try_from(i).unwrap()) * 0.1 + 0.05;
        for uuid in ["h-high", "h-low-a", "h-low-b", "h-low-c", "out"] {
            records.push(DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.to_string(),
                value: Some(vary),
                activation: vary,
                errors: vec![0.4],
            });
        }
    }
    records
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

fn ranks(ranked: &[String], uuid: &str) -> bool {
    ranked.iter().any(|u| u == uuid)
}

/// With the gate enabled, the three near-zero-impact hidden neurons are dropped
/// from the ranked focus list, the high-impact neuron and the output remain, and
/// the gated count reports exactly three exclusions.
#[test]
#[serial]
fn low_impact_neurons_gated_when_flag_enabled() {
    let _flag = EnvVarGuard::set(FLAG_ENV, "1");
    let _threshold = EnvVarGuard::unset(THRESHOLD_ENV); // use default 1e-6

    let creature = make_creature();
    let tmp = write_temp_parquet(&make_records());
    let stats = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let ranked: Vec<String> = stats
        .neurons
        .iter()
        .map(|n| n.neuron_uuid.clone())
        .collect();

    for low in ["h-low-a", "h-low-b", "h-low-c"] {
        assert!(
            !ranks(&ranked, low),
            "near-zero-impact neuron {low} must not consume a focus slot; ranked = {ranked:?}"
        );
    }
    assert!(
        ranks(&ranked, "h-high"),
        "high-impact neuron must remain focus-eligible; ranked = {ranked:?}"
    );
    assert_eq!(
        stats.focus_ineligible_low_impact, 3,
        "exactly three near-zero-impact neurons should have been gated"
    );
}

/// Baseline (gate disabled): the low-impact neurons *do* consume focus slots —
/// the wasted-slot behaviour this issue documents — the gated count is zero, and
/// their structural impact is genuinely below the gate while they remain
/// non-constant. This is the "before" half of the slot-waste measurement and
/// proves the constant filter cannot catch them.
#[test]
#[serial]
fn low_impact_neurons_consume_focus_slot_by_default() {
    let _flag = EnvVarGuard::unset(FLAG_ENV);

    let creature = make_creature();
    let tmp = write_temp_parquet(&make_records());
    let stats = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let ranked: Vec<String> = stats
        .neurons
        .iter()
        .map(|n| n.neuron_uuid.clone())
        .collect();

    for low in ["h-low-a", "h-low-b", "h-low-c"] {
        assert!(
            ranks(&ranked, low),
            "without the gate the low-impact neuron {low} wastes a focus slot; ranked = {ranked:?}"
        );
    }
    assert_eq!(
        stats.focus_ineligible_low_impact, 0,
        "no gating should occur while the gate is disabled"
    );

    // The low-impact neurons are genuinely sub-gate yet non-constant, so only
    // the impact gate — not the constant filter — can remove them.
    for low in ["h-low-a", "h-low-b", "h-low-c"] {
        let n = stats
            .neurons
            .iter()
            .find(|n| n.neuron_uuid == low)
            .expect("low-impact neuron present in ranked list");
        assert!(
            n.impact.abs() < DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD,
            "{low} impact {} should be below the {DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD:e} gate",
            n.impact
        );
    }
}

/// A custom threshold is honoured: raising the gate above the high-impact
/// neuron's impact gates *everything*, and the count reflects the full pool.
#[test]
#[serial]
fn custom_threshold_is_honoured() {
    let _flag = EnvVarGuard::set(FLAG_ENV, "1");
    // A gate of 100 sits above every neuron's normalised impact (max ~1.0).
    let _threshold = EnvVarGuard::set(THRESHOLD_ENV, "100");

    let creature = make_creature();
    let tmp = write_temp_parquet(&make_records());
    let stats = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    assert!(
        stats.neurons.is_empty(),
        "a gate above every impact should leave no focus-eligible neurons; ranked = {:?}",
        stats
            .neurons
            .iter()
            .map(|n| &n.neuron_uuid)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        stats.focus_ineligible_low_impact, 5,
        "all five selectable neurons should be gated"
    );
}

/// The threshold resolver is pure and defends against invalid configuration:
/// missing, empty, non-numeric, non-finite, zero, and negative values fall back
/// to the default; valid positive values (including whitespace-padded) parse.
#[test]
fn resolve_threshold_rejects_invalid_and_non_positive() {
    let default = DEFAULT_FOCUS_IMPACT_GATE_THRESHOLD;
    assert_eq!(resolve_focus_impact_gate_threshold(None, default), default);
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some(""), default),
        default
    );
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some("abc"), default),
        default
    );
    // Zero would gate nothing (every |impact| >= 0) → reject, keep default.
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some("0"), default),
        default
    );
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some("-1e-6"), default),
        default
    );
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some("inf"), default),
        default
    );
    assert_eq!(
        resolve_focus_impact_gate_threshold(Some("NaN"), default),
        default
    );
    // Valid positive overrides parse (whitespace tolerated).
    assert!((resolve_focus_impact_gate_threshold(Some("1e-4"), default) - 1e-4).abs() < 1e-12);
    assert!((resolve_focus_impact_gate_threshold(Some(" 5e-6 "), default) - 5e-6).abs() < 1e-12);
}
