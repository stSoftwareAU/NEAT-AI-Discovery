//! Common test utilities

#[allow(dead_code)]
use std::path::PathBuf;

use neat_ai_discovery::SynapseJson;

/// Get the path to test data directory
#[allow(dead_code)]
pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("data");
    path
}

/// Helper function to create a SynapseJson with synapse_type defaulting to None.
/// This avoids having to specify synapse_type: None in every test.
#[allow(dead_code)]
pub fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper function to create a SynapseJson with a specific synapse type.
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
