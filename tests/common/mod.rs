//! Common test utilities

#[allow(dead_code)]
use std::path::PathBuf;

/// Get the path to test data directory
#[allow(dead_code)]
pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("data");
    path
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
