//! FFI boundary types — JSON request/response structs.
//!
//! All types in this module are used to serialise and deserialise data at the
//! FFI boundary between this Rust library and the TypeScript/Deno controller.

mod candidates;
mod conversions;
mod creature;
mod diagnostics;
mod requests;
mod responses;
mod session;

// Re-export all public types at the `ffi_types::` level for backward compatibility.
pub use candidates::*;
pub(crate) use conversions::*;
pub use creature::*;
pub use diagnostics::*;
pub use requests::*;
pub use responses::*;
pub use session::*;
