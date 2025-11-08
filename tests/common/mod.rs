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
