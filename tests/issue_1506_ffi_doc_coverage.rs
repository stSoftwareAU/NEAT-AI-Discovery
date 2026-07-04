//! Issue #1506 — DOC-README-API-GAP: verify every public FFI entry point is
//! documented in both the README FFI API Summary and `docs/FFI_API.md`.
//!
//! `neat_ai_discovery` is consumed exclusively across its FFI boundary, so the
//! `pub extern "C"` symbol set *is* the public surface. A consumer following
//! the documented discovery path (README summary → `docs/FFI_API.md`) must be
//! able to learn that each entry point exists. These tests assert the five
//! previously-undocumented symbols are present in both documents, and that they
//! are genuine, callable FFI entry points (not stale doc references).

/// The FFI entry points that Issue #1506 identified as undocumented in both the
/// README and `docs/FFI_API.md`.
const NEWLY_DOCUMENTED_SYMBOLS: &[&str] = &[
    "get_calibration_summary",
    "cleanup_discovery_lib",
    "cleanup_discovery_dir",
    "clean_orphaned_discovery_dirs",
    "cancel_analysis_memory_pressure",
];

const README: &str = include_str!("../README.md");
const FFI_API: &str = include_str!("../docs/FFI_API.md");

#[test]
fn readme_documents_all_previously_missing_ffi_entry_points() {
    for symbol in NEWLY_DOCUMENTED_SYMBOLS {
        assert!(
            README.contains(symbol),
            "README.md must document the `{symbol}` FFI entry point (Issue #1506)"
        );
    }
}

#[test]
fn ffi_api_doc_documents_all_previously_missing_ffi_entry_points() {
    for symbol in NEWLY_DOCUMENTED_SYMBOLS {
        assert!(
            FFI_API.contains(symbol),
            "docs/FFI_API.md must document the `{symbol}` FFI entry point (Issue #1506)"
        );
    }
}

/// The most material omission called out in the issue: `cleanup_discovery_lib`
/// is a correctness-relevant call a host must make before process exit. Ensure
/// the README flags it as such rather than merely naming it.
#[test]
fn readme_flags_cleanup_before_exit() {
    assert!(
        README.contains("cleanup_discovery_lib"),
        "README.md must name `cleanup_discovery_lib` (Issue #1506)"
    );
    assert!(
        README.to_lowercase().contains("before process exit"),
        "README.md must flag that cleanup is called before process exit (Issue #1506)"
    );
}

/// `cleanup_discovery_lib` takes no arguments and is safe to call multiple times
/// from any thread, so calling it here proves the documented symbol is a real,
/// callable FFI entry point — the docs are not referencing a phantom function.
#[test]
fn documented_cleanup_symbol_is_a_real_ffi_entry_point() {
    neat_ai_discovery::ffi::cleanup_discovery_lib();
    // Idempotent: a second call must also succeed without panicking.
    neat_ai_discovery::ffi::cleanup_discovery_lib();
}

/// `cancel_analysis_memory_pressure` mutates global cancellation state, so we
/// must not call it from a parallel test. Binding it as a function pointer with
/// its exact `extern "C"` signature still proves the documented symbol resolves
/// to a genuine, correctly-typed FFI entry point without touching that state.
#[test]
fn documented_memory_pressure_symbol_is_a_real_ffi_entry_point() {
    let symbol: extern "C" fn() = neat_ai_discovery::ffi::cancel_analysis_memory_pressure;
    // Reference the pointer so the binding is genuinely used.
    assert!(symbol as usize != 0);
}
