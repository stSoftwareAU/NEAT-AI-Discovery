//! Pins how `is_device_lost_error`/`is_memory_exhaustion_error` classify the
//! real wgpu 30.0.1 error text recorded in the chunk-9 ledger's
//! queue-lifecycle section (Issue #2245). CPU-only: no GPU device needed.

use neat_ai_discovery::analysis::gpu::{is_device_lost_error, is_memory_exhaustion_error};

/// Wrap an inner error text with an anyhow context, mirroring production
/// `.context(...)` call sites; the classifier formats `{error:#}`, so the
/// context chain is included in the matched text.
fn wrapped(inner: String, context: &'static str) -> anyhow::Error {
    anyhow::anyhow!("{inner}").context(context)
}

struct Case {
    row: &'static str,
    error: anyhow::Error,
    device_lost: bool,
    memory: bool,
}

// Rows 2, 3, 7, 9, 10 and M2 have no wgpu producer that can reach either
// classifier, so they are not represented in this table.
#[test]
fn every_producer_bearing_ledger_row_classifies_as_recorded() {
    let cases = vec![
        Case {
            // Device-lost text, but wgpu 30 never delivers it as an `Err` (finding #2363).
            row: "row 1 device is lost",
            error: anyhow::anyhow!("{}", wgpu::wgc::device::DeviceError::Lost),
            device_lost: true,
            memory: false,
        },
        Case {
            // Over-broad: this is a wait timeout, not device loss.
            row: "row 4 gpu device poll error",
            error: anyhow::anyhow!(
                "GPU device poll error ({label}): {e}",
                label = "GPU warm-up after pipeline creation",
                e = wgpu::PollError::Timeout
            ),
            device_lost: true,
            memory: false,
        },
        Case {
            // Over-broad: reaches only the uncaptured-error default handler.
            row: "row 5 internal error",
            error: anyhow::anyhow!(
                "{}",
                wgpu::wgc::pipeline::CreateComputePipelineError::Internal(
                    "shader translation failed".to_string()
                )
            ),
            device_lost: true,
            memory: false,
        },
        Case {
            // Device-lost verdict, but delivered only to the default handler, which panics (finding #2363).
            row: "row 6 out of memory",
            error: anyhow::anyhow!(
                "{}",
                wgpu::Error::OutOfMemory {
                    source: Box::new(wgpu::wgc::device::DeviceError::OutOfMemory),
                }
            ),
            device_lost: true,
            memory: true,
        },
        Case {
            // Over-broad: an encoder-state error, not device loss.
            row: "row 8 command buffer (wgpu)",
            error: anyhow::anyhow!("{}", wgpu::wgc::command::EncoderStateError::Submitted),
            device_lost: true,
            memory: false,
        },
        Case {
            // The crate label forges a "command buffer" match; over-broad.
            row: "row 8 command buffer (crate poll label)",
            error: anyhow::anyhow!(
                "GPU device poll timed out after {:.1}s ({label}). The GPU driver may be unresponsive.",
                5.0_f64,
                label = "post-helpful-batch command buffer release"
            ),
            device_lost: true,
            memory: false,
        },
        Case {
            // Over-broad; a budget expiry is re-initialised — finding #2365; the fix must update this case.
            row: "row 11/12 gpu driver (map timeout)",
            error: wrapped(
                "GPU buffer mapping timed out after 0.0s. The GPU driver may be unresponsive."
                    .to_string(),
                "ReLU buffer mapping failed",
            ),
            device_lost: true,
            memory: false,
        },
    ];

    for case in cases {
        let formatted = format!("{:#}", case.error);
        assert_eq!(
            is_device_lost_error(&case.error),
            case.device_lost,
            "{}: is_device_lost_error mismatch for {formatted:?}",
            case.row
        );
        assert_eq!(
            is_memory_exhaustion_error(&case.error),
            case.memory,
            "{}: is_memory_exhaustion_error mismatch for {formatted:?}",
            case.row
        );
    }
}

// This is current behaviour recorded as finding #2363 (SEC-19ddcad53b91); the
// fix must flip these assertions once device loss is correctly propagated.
#[test]
fn device_loss_reported_through_the_map_callback_is_not_classified() {
    let multi = wrapped(
        format!("Buffer 0 mapping failed: {}", wgpu::BufferAsyncError),
        "Helpful batch buffer mapping failed",
    );
    assert!(!is_device_lost_error(&multi));
    assert!(!is_memory_exhaustion_error(&multi));

    let single = wrapped(
        format!("Buffer mapping failed: {}", wgpu::BufferAsyncError),
        "ReLU buffer mapping failed",
    );
    assert!(!is_device_lost_error(&single));
    assert!(!is_memory_exhaustion_error(&single));

    let core_oom = anyhow::anyhow!("{}", wgpu::wgc::device::DeviceError::OutOfMemory);
    assert!(!is_device_lost_error(&core_oom));
    assert!(!is_memory_exhaustion_error(&core_oom));
}

// No `MapRangeError` producer reaches either classifier, so it correctly
// never matches — the wgpu 29 -> 30 text stayed clear of both patterns.
#[test]
fn map_range_error_texts_never_match() {
    let literals = vec![
        // wgpu-30.0.1/src/api/buffer.rs:815-817
        "Buffer view error: tried to call get_mapped_range(_mut) on an unmapped buffer",
        // wgpu-30.0.1/src/api/buffer.rs:821-827 (example offsets)
        "Buffer view error: tried to call get_mapped_range(_mut) on a range that is not \
         entirely mapped. Attempted to get range 0..64, but the mapped range is 0..32",
        // wgpu-30.0.1/src/api/buffer.rs:836-842 (example offsets)
        "Buffer view error: tried to call get_mapped_range(_mut) on a range that has already \
         been mapped and would break Rust memory aliasing rules. Attempted to get range \
         0..64, and the conflicting range is 0..32",
    ];

    for text in literals {
        let err = wrapped(
            text.to_string(),
            "Helpful staging buffer get_mapped_range failed",
        );
        assert!(!is_device_lost_error(&err), "unexpected match for {text:?}");
        assert!(
            !is_memory_exhaustion_error(&err),
            "unexpected match for {text:?}"
        );
    }

    let core_cases = vec![
        wgpu::wgc::resource::BufferAccessError::NotMapped,
        wgpu::wgc::resource::BufferAccessError::UnalignedOffset { offset: 3 },
        wgpu::wgc::resource::BufferAccessError::MapStartOffsetOverrun {
            offset: 128,
            buffer_size: 64,
        },
    ];
    for e in core_cases {
        let text = format!("Buffer view error: Validation Error\n\nCaused by:\n  {e}\n");
        let err = wrapped(
            text.clone(),
            "Helpful staging buffer get_mapped_range failed",
        );
        assert!(!is_device_lost_error(&err), "unexpected match for {text:?}");
        assert!(
            !is_memory_exhaustion_error(&err),
            "unexpected match for {text:?}"
        );
    }
}

// Ledger verdict "unreachable — validation errors take the uncaptured-error
// path"; the text also matches no pattern.
#[test]
fn workgroup_limit_validation_error_is_not_matched() {
    let dispatch_error = wgpu::wgc::command::DispatchError::InvalidGroupSize {
        current: [65_536, 1, 1],
        limit: 65_535,
    };
    let err = wgpu::Error::Validation {
        source: Box::new(dispatch_error.clone()),
        description: format!("Validation Error\n\nCaused by:\n  {dispatch_error}\n"),
    };
    // Default uncaptured-error handler text (wgpu-30.0.1/src/backend/wgpu_core.rs:694).
    let err = anyhow::anyhow!("wgpu error: {err}\n");
    assert!(!is_device_lost_error(&err));
}

// The classifier is text-only; the ledger's forged-match verdict is refuted
// because no interpolating error site in src/analysis/gpu can carry caller text.
#[test]
fn caller_text_containing_a_pattern_matches_but_is_unreachable() {
    let err = anyhow::anyhow!("user label: internal error");
    assert!(is_device_lost_error(&err));
}
