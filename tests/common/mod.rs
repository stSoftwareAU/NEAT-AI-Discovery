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

// ---------------------------------------------------------------------------
// Cost-compatibility fixtures (Issue #1246)
//
// Helpers for building deterministic per-cost output-neuron error fixtures
// and a small toy creature shared by the cost-compatibility integration
// tests. Discovery is cost-agnostic by construction — these fixtures
// reproduce each built-in cost's per-record residual shape so the
// `record_discovery` → `analyze_parallel` pipeline can be exercised
// end-to-end against every cost without depending on the NEAT-AI runtime.
// ---------------------------------------------------------------------------

/// The seven built-in NEAT-AI cost names that discovery must remain
/// compatible with. Mirrors `BUILT_IN_COST_NAMES` in NEAT-AI's `Costs.ts`.
#[allow(dead_code)]
pub const BUILT_IN_COST_NAMES: [&str; 7] = [
    "MSE",
    "MAE",
    "MAPE",
    "MSLE",
    "HINGE",
    "CROSS_ENTROPY",
    "CATEGORICAL_ERROR",
];

/// Produce a deterministic output-neuron error value shaped like a
/// per-record residual under `cost`.
///
/// The shapes follow §1 of `docs/COST_FUNCTION_NOTES.md`:
/// - `MSE`/`MAE`: continuous signed.
/// - `MAPE`: continuous signed, smaller scale (percentage-like).
/// - `MSLE`: continuous signed (log-space residual).
/// - `HINGE`: non-negative, sparse (frequent zeros for margined samples).
/// - `CROSS_ENTROPY`: continuous signed in `[-1, 1]` (soft-max gradient).
/// - `CATEGORICAL_ERROR`: quantised misclassification flag in `{0, 1}`.
///
/// Deterministic in `(cost, obs_index, output_index)` so tests are
/// reproducible without an RNG.
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn cost_shaped_error(cost: &str, obs_index: u32, output_index: usize) -> f32 {
    let phase = (obs_index as f32) * 0.37 + (output_index as f32) * 1.7;
    match cost {
        "MSE" | "MAE" => phase.sin() * 0.3,
        "MAPE" => phase.sin() * 0.15,
        "MSLE" => phase.sin() * 0.2,
        "HINGE" => {
            // Sparse: zero on roughly half of samples ("correctly margined"),
            // positive magnitude on the rest. Sign convention follows
            // NEAT-AI's chain rule which preserves the sign of the residual.
            if phase.sin() > 0.0 {
                (1.0 - phase.cos() * 0.6).clamp(0.0, 1.5)
            } else {
                0.0
            }
        }
        "CROSS_ENTROPY" => (phase.sin() * 0.5).clamp(-0.95, 0.95),
        "CATEGORICAL_ERROR" => {
            if phase.sin() > 0.3 {
                1.0
            } else {
                0.0
            }
        }
        _ => phase.sin() * 0.3,
    }
}

/// Build a small 2-input, 3-hidden, 2-output toy creature used by the
/// cost-compatibility tests. Returns the `CreatureJson` together with the
/// list of hidden + output neuron UUIDs (in forward-only order).
///
/// Topology (forward-only):
/// - `input-0`, `input-1` → `h0`, `h1`
/// - `h0`, `h1` → `h2`
/// - `h2` → `o0`, `o1`
#[allow(dead_code)]
pub fn toy_cost_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 2,
        neurons: vec![
            hidden("h0", "TANH"),
            hidden("h1", "LOGISTIC"),
            hidden("h2", "TANH"),
            output("o0", "IDENTITY"),
            output("o1", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "h0", 0.7),
            synapse("input-1", "h1", -0.5),
            synapse("h0", "h2", 0.6),
            synapse("h1", "h2", 0.4),
            synapse("h2", "o0", 0.8),
            synapse("h2", "o1", -0.3),
        ],
    }
}
