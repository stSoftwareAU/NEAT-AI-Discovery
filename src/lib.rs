//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

// Global tracking allocator — wraps the system allocator to report Rust-side
// memory usage via FFI (Issue #1027). Overhead is a single atomic add/sub per
// allocation, which is negligible for polling every 5-30 seconds.
#[global_allocator]
static ALLOCATOR: cap::Cap<std::alloc::System> = cap::Cap::new(std::alloc::System, usize::MAX);

pub mod activations;
pub mod analysis;
pub mod cancellation;
pub mod config;
pub mod debug;
pub mod discovery_history;
pub mod export;
pub mod ffi;
mod ffi_internal;
pub mod ffi_types;
pub mod focus;
pub mod intern;
pub mod observability;
pub mod parquet_format;
pub mod record;
pub mod streaming;
pub mod types;
mod watchdog;

// Re-export all FFI boundary types so that `crate::TypeName` and
// `neat_ai_discovery::TypeName` continue to work without any change
// to existing code.
pub use ffi_types::*;

// Re-export all internal business-logic functions so that existing
// integration tests (`neat_ai_discovery::*_internal`) continue to work.
pub use ffi_internal::*;

use std::sync::OnceLock;

// Library version from Cargo.toml
const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");

// Static flag to ensure version is logged only once
static VERSION_LOGGED: OnceLock<()> = OnceLock::new();

/// Log library version on first initialisation.
///
/// Initialises the tracing subscriber (Issue #575) and debug handlers, then
/// logs the compiled library version.
pub(crate) fn log_version_once() {
    VERSION_LOGGED.get_or_init(|| {
        // Initialise structured logging subscriber before any tracing calls.
        observability::init_tracing();

        tracing::info!(
            version = LIB_VERSION,
            "NEAT-AI-Discovery library initialised"
        );

        // Initialise debug handlers (deadlock detection + kill -3 thread dump)
        debug::init_debug_handlers();
    });
}
