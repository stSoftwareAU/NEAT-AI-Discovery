//! Integration tests for classifying wgpu 30 memory exhaustion (Issue #2363).
//!
//! wgpu-core reports `DeviceError::OutOfMemory` as "Not enough memory left.",
//! which the classifiers must route to device-lost recovery and to the #1083
//! batch-halving path.

use neat_ai_discovery::analysis::gpu::{is_device_lost_error, is_memory_exhaustion_error};

/// Issue #2363: wgpu-core's "Not enough memory left." must route to device-lost recovery.
#[test]
fn not_enough_memory_left_is_device_lost() {
    let err =
        anyhow::anyhow!("Not enough memory left.").context("Helpful batch buffer creation failed");
    assert!(
        is_device_lost_error(&err),
        "expected device-lost classification for: {err:#}"
    );
}

/// Issue #2363: wgpu-core's "Not enough memory left." must trigger #1083 batch halving.
#[test]
fn not_enough_memory_left_is_memory_exhaustion() {
    let err =
        anyhow::anyhow!("Not enough memory left.").context("Helpful batch buffer creation failed");
    assert!(
        is_memory_exhaustion_error(&err),
        "expected memory-exhaustion classification for: {err:#}"
    );
}

/// Issue #2363: the opaque map error is routed by the device-lost flag, not by text.
#[test]
fn opaque_map_error_alone_is_not_classified_by_text() {
    let err = anyhow::anyhow!(
        "Buffer 0 mapping failed: Error occurred when trying to async map a buffer"
    );
    assert!(
        !is_device_lost_error(&err),
        "opaque map error must not match device-lost text: {err:#}"
    );
    assert!(
        !is_memory_exhaustion_error(&err),
        "opaque map error must not match memory-exhaustion text: {err:#}"
    );
}
