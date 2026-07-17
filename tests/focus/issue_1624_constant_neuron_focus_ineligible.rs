//! Issue #1624: functionally-constant hidden neurons must not consume focus slots.
//!
//! The fast focus path ranks every selectable (non-input, non-constant-type)
//! neuron. `is_selectable_type` does *not* exclude functionally-constant hidden
//! neurons — those with zero recorded activation variance — so they land in the
//! ranked focus list and displace productive neurons. A constant neuron can
//! never yield a successful add-synapse / add-neuron candidate (its output never
//! varies), so every focus slot it occupies is wasted.
//!
//! When `NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS` is enabled, such
//! neurons are dropped from the focus list while remaining fully available to
//! the constant-neuron *removal* path (`constant_neuron_removals`). These tests
//! assert both halves and record the before/after slot-waste measurement
//! (`focus_ineligible_constant`).

use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const FLAG_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS";

/// RAII guard that sets/restores `FLAG_ENV` for a single test. All env
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

/// Creature: one input feeds a functionally-constant hidden neuron and a
/// varying hidden neuron, both wired to a single output.
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
            neuron("h-const", "hidden"),
            neuron("h-vary", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-const", 0.5),
            synapse("in-0", "h-vary", 0.5),
            synapse("h-const", "out", 0.5),
            synapse("h-vary", "out", 0.5),
        ],
        input: 1,
        output: 1,
    }
}

/// Records: `h-const` holds a single activation (zero variance); `h-vary` and
/// `out` vary across observations (non-trivial variance). No input records are
/// required — inputs are not selectable for ranking.
fn make_records() -> Vec<DiscoverRecord> {
    let count = 12u32;
    let mut records = Vec::new();
    for i in 0..count {
        let vary = f32::from(u16::try_from(i).unwrap()) * 0.1;
        // Constant hidden neuron: identical activation on every observation.
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "h-const".to_string(),
            value: Some(1.0),
            activation: 1.0,
            errors: vec![0.4],
        });
        // Varying hidden neuron: activation changes every observation.
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "h-vary".to_string(),
            value: Some(vary),
            activation: vary,
            errors: vec![0.4],
        });
        // Output neuron varies so it is never clamped to zero error.
        records.push(DiscoverRecord {
            obs_index: i,
            neuron_uuid: "out".to_string(),
            value: Some(vary),
            activation: vary,
            errors: vec![0.5],
        });
    }
    records
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

fn ranks(stats_neuron_uuids: &[String], uuid: &str) -> bool {
    stats_neuron_uuids.iter().any(|u| u == uuid)
}

/// With the filter enabled, a functionally-constant hidden neuron never lands in
/// the ranked focus list, the varying neuron still does, and the slot-waste
/// count reports exactly one exclusion. The constant neuron remains a
/// constant-removal candidate — the removal path is untouched.
#[test]
#[serial]
fn constant_neuron_is_focus_ineligible_when_flag_enabled() {
    let _guard = EnvVarGuard::set(FLAG_ENV, "1");

    let creature = make_creature();
    let tmp = write_temp_parquet(&make_records());
    let stats = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let ranked: Vec<String> = stats
        .neurons
        .iter()
        .map(|n| n.neuron_uuid.clone())
        .collect();

    assert!(
        !ranks(&ranked, "h-const"),
        "constant neuron must not consume a focus slot; ranked = {ranked:?}"
    );
    assert!(
        ranks(&ranked, "h-vary"),
        "varying neuron must remain focus-eligible; ranked = {ranked:?}"
    );
    assert_eq!(
        stats.focus_ineligible_constant, 1,
        "exactly one constant neuron should have been excluded"
    );

    // The removal path is preserved: h-const is still offered for bias-fold removal.
    let const_removal = stats.constant_neuron_removals.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(op, CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "h-const")
        })
    });
    assert!(
        const_removal,
        "constant neuron must still be a constant-removal candidate"
    );
}

/// Baseline (default, flag disabled): the constant neuron *does* consume a focus
/// slot — the wasted-slot behaviour this issue documents — and the exclusion
/// count is zero. This is the "before" half of the slot-waste measurement.
#[test]
#[serial]
fn constant_neuron_consumes_focus_slot_by_default() {
    let _guard = EnvVarGuard::unset(FLAG_ENV);

    let creature = make_creature();
    let tmp = write_temp_parquet(&make_records());
    let stats = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let ranked: Vec<String> = stats
        .neurons
        .iter()
        .map(|n| n.neuron_uuid.clone())
        .collect();

    assert!(
        ranks(&ranked, "h-const"),
        "without the filter the constant neuron wastes a focus slot; ranked = {ranked:?}"
    );
    assert_eq!(
        stats.focus_ineligible_constant, 0,
        "no exclusion should occur while the filter is disabled"
    );
}
