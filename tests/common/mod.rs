//! Shared test utilities for integration tests.
//!
//! This module provides common helpers to eliminate boilerplate across test files:
//! - **GPU skip macro** — `skip_without_gpu!()` for tests requiring GPU access
//! - **Fixture builders** — `neuron()`, `hidden()`, `output()`, `synapse()`, `make_creature()`
//! - **Record helpers** — `record()` for creating `DiscoverRecord` instances

#[allow(dead_code)]
use std::path::PathBuf;

use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Get the path to test data directory.
#[allow(dead_code)]
pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("data");
    path
}

// ---------------------------------------------------------------------------
// GPU skip macro
// ---------------------------------------------------------------------------

/// Macro to skip tests that require a GPU when no GPU is available.
/// Place this at the start of any test that calls GPU-accelerated analysis functions.
#[macro_export]
macro_rules! skip_without_gpu {
    () => {
        if !neat_ai_discovery::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// Issue #1191 — minimum-expected-gain floor RAII guard.
//
// Many integration tests build small synthetic fixtures whose post-discount
// candidate gains fall below the new 1e-5 production noise floor. Holding a
// `GainFloorDisableGuard` for the duration of such a test sets
// `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN=0` on construction and restores the
// previous value on drop, so the contract under test is observable
// independently of the noise filter.
// ---------------------------------------------------------------------------

/// RAII guard that disables the Issue #1191 minimum-expected-gain floor for
/// the duration of a test, restoring the prior env-var value on drop.
///
/// Tests that construct synthetic fixtures whose post-discount gains fall
/// below `MIN_EXPECTED_CREATURE_SCORE_GAIN` should hold one of these for the
/// scope of the assertion. Use with `#[serial]` (or another env-serialisation
/// strategy) to avoid concurrent env access.
#[allow(dead_code)]
pub struct GainFloorDisableGuard {
    previous: Option<String>,
}

impl GainFloorDisableGuard {
    /// Create a guard, setting `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN=0`.
    #[allow(dead_code)]
    pub fn new() -> Self {
        let previous = std::env::var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN").ok();
        // SAFETY: callers serialise env access (e.g. via `#[serial]`).
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN", "0");
        }
        Self { previous }
    }
}

impl Default for GainFloorDisableGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GainFloorDisableGuard {
    fn drop(&mut self) {
        // SAFETY: callers serialise env access (e.g. via `#[serial]`).
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN", v),
                None => std::env::remove_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Neuron builders
// ---------------------------------------------------------------------------

/// Build a `NeuronJson` with the given type and activation function.
#[allow(dead_code)]
pub fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Build a hidden `NeuronJson` with zero bias.
#[allow(dead_code)]
pub fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    neuron(uuid, "hidden", squash)
}

/// Build a hidden `NeuronJson` with a specified bias.
#[allow(dead_code)]
pub fn hidden_with_bias(uuid: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias,
    }
}

/// Build an output `NeuronJson` with zero bias.
#[allow(dead_code)]
pub fn output(uuid: &str, squash: &str) -> NeuronJson {
    neuron(uuid, "output", squash)
}

// ---------------------------------------------------------------------------
// Synapse builders
// ---------------------------------------------------------------------------

/// Build a `SynapseJson` with `synapse_type` defaulting to `None`.
#[allow(dead_code)]
pub fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Build a `SynapseJson` with a specific synapse type.
/// Used for IF neurons with "condition", "positive", or "negative" synapses.
#[allow(dead_code)]
pub fn synapse_typed(from: &str, to: &str, weight: f32, stype: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: Some(stype.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Creature builders
// ---------------------------------------------------------------------------

/// Build a minimal `CreatureJson` from neurons and synapses.
///
/// Sets `input` to 2 and `output` to 1 — suitable for most unit-level
/// detection tests that only need a small topology.
#[allow(dead_code)]
pub fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

// ---------------------------------------------------------------------------
// Record helpers
// ---------------------------------------------------------------------------

/// Create a `DiscoverRecord` with a default single-element error vector.
#[allow(dead_code)]
pub fn record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation,
        errors: vec![0.01],
    }
}
