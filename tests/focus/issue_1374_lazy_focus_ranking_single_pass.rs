//! Issue #1374: Lazy focus ranking must materialise each neuron's records at
//! most once per run.
//!
//! Previously, lazy mode re-read the entire parquet file on every cache miss
//! and the ranking pipeline swept the selectable set ~5 times against an
//! 8-entry cache, so a working set larger than 8 neurons thrashed into
//! `O(passes × neurons)` full-file decodes. The fix sizes the cache to the
//! working set and warms it with a single grouped pass.
//!
//! These tests guard the **numerical parity** acceptance criterion: lazy-mode
//! ranking must produce identical results to preload mode. The per-neuron
//! loader-invocation count (`O(neurons)`, not `O(passes × neurons)`) is
//! asserted directly against the provider in `src/focus/tests.rs`.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::focus::{FocusLoadingMode, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

const BUDGET_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB";

/// RAII guard that sets/restores `BUDGET_ENV` for a single test.
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

/// Build a creature with `hidden_count` hidden neurons (deliberately larger
/// than the lazy cache's default 8-entry capacity) feeding one output.
fn make_wide_creature(hidden_count: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    for i in 0..hidden_count {
        let uuid = format!("h{i}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: "in-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 1.0,
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: uuid,
            to_uuid: "out".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "out".to_string(),
        neuron_type: "output".to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    });

    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Distinct per-neuron records so ranking scores (and ordering) are non-trivial.
fn make_varied_records(uuid: &str, seed: usize, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let phase = (seed + i as usize) as f32;
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.to_string(),
                value: Some(0.1 + 0.01 * phase),
                activation: 0.2 + 0.03 * (seed as f32),
                errors: vec![0.05 + 0.02 * (seed as f32)],
            }
        })
        .collect()
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

/// Lazy-mode ranking must produce the *same* ranked neurons and scores as
/// preload mode (numerical parity guard, AC #3).
#[test]
#[serial]
fn lazy_mode_matches_preload_mode_numerically() {
    const HIDDEN: usize = 16; // > default cache capacity (8) so the bug would thrash.
    // Enough records that the projected in-memory size (file × 3) exceeds the
    // 1 MB budget below, forcing the end-to-end lazy path.
    const PER_NEURON: u32 = 6000;

    let creature = make_wide_creature(HIDDEN);
    let mut records = Vec::new();
    for i in 0..HIDDEN {
        records.extend(make_varied_records(&format!("h{i}"), i + 1, PER_NEURON));
    }
    records.extend(make_varied_records("out", HIDDEN + 1, PER_NEURON));
    let parquet = write_temp_parquet(&records);
    let path = parquet.path().to_str().unwrap();

    // Preload: a generous budget keeps everything in memory.
    let preload = {
        let _guard = EnvVarGuard::set(BUDGET_ENV, "65536");
        rank_focus_neurons(path, &creature, None, None).expect("preload ranking should succeed")
    };
    assert_eq!(preload.loading_mode, FocusLoadingMode::Preload);

    // Lazy: a tiny budget forces the lazy path (cache warm + sized cache).
    let lazy = {
        let _guard = EnvVarGuard::set(BUDGET_ENV, "1");
        rank_focus_neurons(path, &creature, None, None).expect("lazy ranking should succeed")
    };
    assert_eq!(
        lazy.loading_mode,
        FocusLoadingMode::Lazy,
        "tiny budget must force lazy mode for this dataset"
    );

    // Numerical parity: identical ranking, identical scores.
    assert_eq!(
        preload.neurons.len(),
        lazy.neurons.len(),
        "both modes must rank the same number of neurons"
    );
    assert_eq!(
        preload.max_output_error, lazy.max_output_error,
        "max output error must be identical across modes"
    );

    for (p, l) in preload.neurons.iter().zip(lazy.neurons.iter()) {
        assert_eq!(p.neuron_uuid, l.neuron_uuid, "ranking order must match");
        assert_eq!(p.total_error, l.total_error, "total_error must match");
        assert_eq!(p.raw_error, l.raw_error, "raw_error must match");
        assert_eq!(p.impact, l.impact, "impact must match");
        assert_eq!(
            p.mean_activation, l.mean_activation,
            "mean_activation must match"
        );
        assert_eq!(
            p.activation_frequency, l.activation_frequency,
            "activation_frequency must match"
        );
    }
}
