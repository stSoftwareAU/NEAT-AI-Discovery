//! FFI entry points — `#[no_mangle] pub extern "C"` functions.
//!
//! Every symbol exposed to the Deno FFI layer lives here. Each function
//! catches panics, converts between C strings and Rust strings, and delegates
//! to the corresponding `*_internal` function in the crate root.

// Sub-modules are public so that `#[no_mangle]` symbols are visible in the
// compiled cdylib. The functions themselves are only called via FFI, not from
// Rust code, so the imports appear "unused" to the compiler.
#[allow(unused_imports)]
pub use analysis::*;
#[allow(unused_imports)]
pub use gpu::*;
#[allow(unused_imports)]
pub use recording::*;
#[allow(unused_imports)]
pub use utilities::*;

mod analysis;
mod gpu;
mod recording;
mod utilities;

// ============================================================================
// Memory management
// ============================================================================

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn free_discovery_result(ptr: *mut std::ffi::c_char) {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This is unlikely to panic, but we protect it anyway for safety
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if !ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(ptr);
            }
        }
    }));
}
