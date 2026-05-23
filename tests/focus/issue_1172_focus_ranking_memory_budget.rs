//! Issue #1172: Configurable focus-ranking memory budget.
//!
//! `focus::ranking` previously sampled system memory and silently downgraded
//! to lazy-loading mode under pressure. This test suite covers the new
//! `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` env var which lets
//! callers cap the eager pre-load size deterministically (no system memory
//! sampling) and verifies the chosen mode is surfaced via `RankFocusStats`.

#![allow(clippy::cast_precision_loss)] // Numeric casts in test setup.

use neat_ai_discovery::focus::{
    FocusLazyReason, FocusLoadingMode, decide_loading_mode_for_budget, rank_focus_neurons,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const BUDGET_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB";

/// RAII guard that sets/restores `BUDGET_ENV` for a single test.
///
/// The budget env var has no per-process cache so we can mutate it freely
/// from tests as long as we serialise via `#[serial]`.
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

/// Build a tiny creature: one input -> one hidden -> one output.
fn make_simple_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "h1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "out".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "in-0".to_string(),
                to_uuid: "h1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".to_string(),
                to_uuid: "out".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

fn make_records(uuid: &str, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: Some(0.5),
            activation: 0.3,
            errors: vec![0.1],
        })
        .collect()
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

// ---------------------------------------------------------------------------
// Pure decision helper — exercises the budget logic without needing a parquet
// large enough to exceed the multi-MB threshold after the 3× memory factor.
// AC: "A unit test forces a tiny budget and asserts lazy mode is chosen."
// ---------------------------------------------------------------------------

#[test]
fn tiny_budget_forces_lazy_mode() {
    // 64 MB projected vs 1 MB budget → must select lazy mode.
    let projected_bytes: u64 = 64 * 1024 * 1024;
    let (mode, reason) = decide_loading_mode_for_budget(projected_bytes, 1);
    assert_eq!(
        mode,
        FocusLoadingMode::Lazy,
        "projection above budget must force lazy mode"
    );
    assert_eq!(
        reason,
        FocusLazyReason::Budget,
        "lazy reason must indicate the configured budget triggered the fallback"
    );
}

// AC: "A unit test with a generous budget asserts pre-load mode is chosen."
#[test]
fn generous_budget_selects_preload() {
    // 100 KB projected vs 4096 MB budget → preload.
    let projected_bytes: u64 = 100 * 1024;
    let (mode, reason) = decide_loading_mode_for_budget(projected_bytes, 4096);
    assert_eq!(mode, FocusLoadingMode::Preload);
    assert_eq!(reason, FocusLazyReason::None);
}

#[test]
fn projection_equal_to_budget_keeps_preload() {
    // Strict greater-than: projection at the budget boundary must NOT trigger
    // lazy mode (matches the spirit of "exceeds the budget").
    let budget_mb: u64 = 8;
    let projected_bytes = budget_mb * 1024 * 1024;
    let (mode, _reason) = decide_loading_mode_for_budget(projected_bytes, budget_mb);
    assert_eq!(mode, FocusLoadingMode::Preload);
}

// ---------------------------------------------------------------------------
// Integration: end-to-end propagation of mode/budget into RankFocusStats.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn generous_budget_end_to_end_selects_preload_mode() {
    // 64 GB easily exceeds any plausible parquet projection so preload must
    // be selected without sampling system memory at all.
    let _guard = EnvVarGuard::set(BUDGET_ENV, "65536");

    let creature = make_simple_creature();
    let mut records = make_records("h1", 200);
    records.extend(make_records("out", 200));
    let parquet = write_temp_parquet(&records);

    let stats = rank_focus_neurons(parquet.path().to_str().unwrap(), &creature, None, None)
        .expect("ranking should succeed with a generous budget");

    assert_eq!(
        stats.loading_mode,
        FocusLoadingMode::Preload,
        "generous budget must allow preload mode"
    );
    assert_eq!(
        stats.lazy_reason,
        FocusLazyReason::None,
        "lazy_reason must be None when preload is chosen"
    );
    assert_eq!(stats.budget_mb, Some(65_536));
    assert!(!stats.neurons.is_empty());
}

/// AC: "If the budget is unset, behaviour matches today (auto-detect + WARN
/// on fallback)."
#[test]
#[serial]
fn unset_budget_falls_back_to_auto_detect() {
    let _guard = EnvVarGuard::unset(BUDGET_ENV);

    let creature = make_simple_creature();
    let mut records = make_records("h1", 200);
    records.extend(make_records("out", 200));
    let parquet = write_temp_parquet(&records);

    let stats = rank_focus_neurons(parquet.path().to_str().unwrap(), &creature, None, None)
        .expect("ranking should succeed with default auto-detect");

    // budget_mb is None when no explicit budget was set.
    assert_eq!(stats.budget_mb, None);
    // Whatever mode auto-detect picks, the lazy_reason must be coherent:
    //   Preload → None
    //   Lazy → MemoryPressure (we never get Budget without a configured budget)
    match stats.loading_mode {
        FocusLoadingMode::Preload => {
            assert_eq!(stats.lazy_reason, FocusLazyReason::None);
        }
        FocusLoadingMode::Lazy => {
            assert_eq!(
                stats.lazy_reason,
                FocusLazyReason::MemoryPressure,
                "without a configured budget, lazy mode must always be attributed \
                 to memory pressure",
            );
        }
    }
}

/// Invalid budget values are ignored so we fall back to auto-detect rather
/// than aborting the run.
#[test]
#[serial]
fn invalid_budget_value_is_ignored() {
    let _guard = EnvVarGuard::set(BUDGET_ENV, "not-a-number");

    let creature = make_simple_creature();
    let mut records = make_records("h1", 100);
    records.extend(make_records("out", 100));
    let parquet = write_temp_parquet(&records);

    let stats = rank_focus_neurons(parquet.path().to_str().unwrap(), &creature, None, None)
        .expect("invalid budget should not abort ranking");

    assert_eq!(
        stats.budget_mb, None,
        "invalid budget values must be treated as unset"
    );
}

/// Budget of zero is documented as unset (avoids accidentally disabling
/// pre-load entirely on misconfiguration).
#[test]
#[serial]
fn zero_budget_is_treated_as_unset() {
    let _guard = EnvVarGuard::set(BUDGET_ENV, "0");

    let creature = make_simple_creature();
    let mut records = make_records("h1", 100);
    records.extend(make_records("out", 100));
    let parquet = write_temp_parquet(&records);

    let stats = rank_focus_neurons(parquet.path().to_str().unwrap(), &creature, None, None)
        .expect("zero budget should not abort ranking");

    assert_eq!(stats.budget_mb, None);
}
