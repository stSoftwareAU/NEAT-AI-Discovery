## Summary

Added a `[lints.rust]` section to `Cargo.toml` that denies
`unsafe_op_in_unsafe_fn` at the crate root, so CI fails fast if a
contributor introduces a bare `unsafe` op inside an `unsafe fn` instead
of wrapping it in its own `unsafe { ... }` block with a `// SAFETY:`
comment. The codebase already follows that convention (e.g.
`src/ffi/utilities.rs`); this change makes the project-wide style
mechanically enforceable. Closes #1255.

`missing_docs` is intentionally **not** added in this PR. The issue
suggests starting it at `warn` with a later promotion to `deny`, but
`quality.sh` exports `RUSTFLAGS="-D warnings"` globally, which would
immediately escalate any `warn`-level lint to a build failure. Enabling
`missing_docs` therefore requires a back-fill of the undocumented `pub`
items in `src/ffi_types/responses/` and elsewhere first, which is out of
scope for this lint-policy change and should be tracked as a follow-up.

## Evidence

This is a build-configuration change with no UI surface. Verified by:

- `./quality.sh` ran to completion with all checks green (Bash syntax,
  `cargo deny`, build, `cargo fmt`, `cargo clippy --all-targets
  --all-features -- -D warnings`, `cargo check`, `cargo test`,
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, release build).
- The new test `tests/issue_1255_rustc_lints.rs` reads `Cargo.toml` and
  asserts that `[lints.rust].unsafe_op_in_unsafe_fn = "deny"` is
  present. Run in isolation: `1 passed; 0 failed`.

```mermaid
flowchart LR
    A[Contributor adds bare unsafe op<br/>inside unsafe fn] --> B[cargo build / clippy]
    B --> C{[lints.rust]<br/>unsafe_op_in_unsafe_fn<br/>= deny}
    C -->|violated| D[CI fails fast]
    C -->|honoured| E[Build green]
```

## Test Plan

- Added `tests/issue_1255_rustc_lints.rs::cargo_toml_denies_unsafe_op_in_unsafe_fn`
  which parses `Cargo.toml` and asserts the rustc lint is declared and
  set to `deny`. Fails the build if the section is removed or weakened.
- Ran the full `./quality.sh` gate locally — all stages pass on the
  branch with the lint enabled, confirming the existing codebase already
  satisfies `unsafe_op_in_unsafe_fn = "deny"`.
