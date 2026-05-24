## Summary

Replaced the two glob re-exports in `src/lib.rs` (`pub use ffi_types::*;` and
`pub use ffi_internal::*;`) with explicit, alphabetically-sorted `pub use`
lists. The crate's public API surface is now visible at a glance from the
root module, so any new `pub` item added under `ffi_types/` or
`ffi_internal/` is now an intentional API decision rather than an
accidental leak. Closes #1256.

The set of re-exported names is unchanged — `crate::TypeName` and
`neat_ai_discovery::TypeName` paths used by integration tests continue to
work. The list was enumerated from every `pub` item currently visible under
`ffi_types` (incl. nested `responses::{analysis,export,gpu}`) and
`ffi_internal`.

## Evidence

This is a backend / API-surface change with no UI to screenshot.

- **`./quality.sh` passes** — fmt, clippy (`-D warnings`), `cargo check
  --all-targets --all-features`, full test suite, doc build, and release
  build all succeed.
- **New regression test** — `tests/issue_1256_public_api_surface.rs`
  imports every name on the explicit re-export list via the
  `neat_ai_discovery::*` path. If anything is dropped from the explicit
  list (or renamed) the test fails to compile, so the list is now
  guarded by CI.
- **Existing call sites still compile** — every `crate::TypeName` and
  `neat_ai_discovery::TypeName` reference under `src/` and `tests/` was
  enumerated before the change to confirm the explicit list covers it.

## Test Plan

- `cargo test --test issue_1256_public_api_surface --all-features --
  --test-threads=2` — 3 passed, 0 failed.
- `./quality.sh < /dev/null` — passes cleanly (fmt, clippy with
  `-D warnings`, check, all tests, doc, release build).
