//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

pub mod activations;
pub mod analysis;
pub mod debug;
pub mod discovery_history;
pub mod export;
mod ffi;
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

use once_cell::sync::OnceCell;

// Library version from Cargo.toml
const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");

// Static flag to ensure version is logged only once
static VERSION_LOGGED: OnceCell<()> = OnceCell::new();

/// Log library version on first initialization
/// This shows the ACTUAL compiled version embedded in the binary at build time
pub(crate) fn log_version_once() {
    VERSION_LOGGED.get_or_init(|| {
        eprintln!("[NEAT-AI-Discovery] Library version {LIB_VERSION} initialized (compiled version embedded in binary)");
        // Initialise debug handlers (deadlock detection + kill -3 thread dump)
        debug::init_debug_handlers();
    });
}
