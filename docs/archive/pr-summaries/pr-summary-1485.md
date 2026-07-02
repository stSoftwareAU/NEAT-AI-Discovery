## Summary

Documented the `*_internal` business-logic functions re-exported at the crate
root and added a re-detection guard so future undocumented re-exports fail the
quality gate. Closes #1485.

The crate's `rlib` re-exports a set of `*_internal` entry points from
`ffi_internal` onto its public surface (`src/lib.rs`). Five of them shipped with
no `///` item documentation, so an rlib consumer saw blank rustdoc for them:

- `analyze_parallel_internal` (`src/ffi_internal/analysis.rs`)
- `rank_focus_neurons_internal` (`src/ffi_internal/analysis.rs`)
- `check_gpu_available_internal` (`src/ffi_internal/gpu.rs`)
- `get_library_version_internal` (`src/ffi_internal/gpu.rs`)
- `merge_discovery_parquet_internal` (`src/ffi_internal/utilities.rs`)

Each now carries a one-line `///` summary matching the style of the already
documented siblings (`get_calibration_summary_internal`, `read_discovery_records`,
`record_discovery_internal`, `export_visualisation_snapshot_internal`).

### Scoped `missing_docs` guard

The issue suggested `missing_docs = "warn"` under `[lints.rust]` in
`Cargo.toml`. A crate-wide enable is **not** viable: the wider crate exposes
**485** undocumented public items (mostly `#[repr(C)]` GPU struct fields and FFI
JSON DTO fields), so a crate-root lint would fail CI's `-D warnings` gate and
pull ~485 unrelated items into this low-severity docs fix.

Instead the guard is scoped to exactly the surface the issue is about — the
re-exported public API — via a module-level inner attribute in
`src/ffi_internal/mod.rs`:

```rust
#![warn(missing_docs)]
```

`ffi_internal` is now fully documented (0 warnings), so under CI's `-D warnings`
this promotes to an error the moment any new `*_internal` re-export is added
without rustdoc — re-detection-safe, and no unrelated churn.

## Evidence

Backend/library change — no web interface to screenshot. Verified via the Rust
toolchain:

- `cargo clippy --all-targets --all-features -- -D warnings` — clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` — clean.
- `cargo fmt --all -- --check` — clean.
- Guard proven: temporarily deleting the doc from `get_library_version_internal`
  reproduces `warning: missing documentation for a function`, which `-D warnings`
  turns into a build failure. Restored immediately.

Note: the repo's `quality.sh` runs `cargo upgrade --incompatible`, which bumps
`wgpu` 29→30 (a breaking mapped-range API migration touching several GPU
evaluation files). That migration is unrelated to this docs fix and out of
scope, so the dependency bump was reverted and the change was validated against
the committed `wgpu = "29"` baseline.

```mermaid
flowchart LR
    A[crate root re-export<br/>src/lib.rs] --> B[ffi_internal::*_internal]
    B --> C{missing_docs guard<br/>#![warn] in mod.rs}
    C -->|documented| D[cargo doc OK]
    C -->|undocumented| E[-D warnings → build fails]
```

## Test Plan

- No new runtime test: documentation presence is a compile-time property, not a
  runtime behaviour, so the enforcement is the `#![warn(missing_docs)]` guard
  under CI's `-D warnings`, verified above by removing and restoring a doc.
- Existing `ffi_internal` lib tests still pass and exercise the documented
  functions:
  `cargo test --lib --all-features ffi_internal -- --test-threads=2`
  → 21 passed, 0 failed (includes
  `get_library_version_internal_returns_well_formed_json`,
  `check_gpu_available_internal_returns_well_formed_json`,
  `analyze_parallel_internal_returns_combined_payload`).
