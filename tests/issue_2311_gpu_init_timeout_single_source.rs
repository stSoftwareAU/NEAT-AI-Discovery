//! Regression test for Issue #2311: `GPU_INIT_TIMEOUT_SECS` must have a single
//! definition. `device.rs` re-exports the `shaders.rs` item instead of
//! declaring its own literal.
//!
//! The pin is structural: both modules are glob-imported into one scope. If
//! they ever carry two distinct items again, the name becomes ambiguous and
//! this file fails to compile — even when both literals happen to agree.

mod single_source {
    #[allow(unused_imports)]
    use neat_ai_discovery::analysis::gpu::device::*;
    #[allow(unused_imports)]
    use neat_ai_discovery::analysis::gpu::shaders::*;

    pub const PINNED_GPU_INIT_TIMEOUT_SECS: u64 = GPU_INIT_TIMEOUT_SECS;
}

#[test]
fn gpu_init_timeout_has_a_single_source() {
    assert_eq!(
        single_source::PINNED_GPU_INIT_TIMEOUT_SECS,
        neat_ai_discovery::analysis::gpu::GPU_INIT_TIMEOUT_SECS
    );
    assert_eq!(
        single_source::PINNED_GPU_INIT_TIMEOUT_SECS,
        neat_ai_discovery::analysis::gpu::SHADER_GPU_INIT_TIMEOUT_SECS
    );
}

#[test]
fn gpu_init_timeout_is_within_documented_range() {
    assert!((5..=60).contains(&single_source::PINNED_GPU_INIT_TIMEOUT_SECS));
}
