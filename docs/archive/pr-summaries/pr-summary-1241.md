## Summary

Added the missing `description`, `repository`, and `readme` fields to the
`[package]` table in `Cargo.toml` so the `neat_ai_discovery` crate meets
the Rust API Guidelines (C-METADATA). This unblocks `cargo publish`,
gives `cargo metadata` consumers (SBOM tools, doc generators) the
upstream source, and brings Discovery into line with the sibling
`rust_scorer` crate. Closes #1241.

## Evidence

This is a metadata-only change with no runtime or UI surface. Verified
by:

- New regression test `tests/issue_1241_cargo_toml_metadata.rs` reads
  `Cargo.toml` and asserts the three fields are present, that
  `repository` points at the canonical GitHub repo, and that the
  referenced `README.md` exists on disk.
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release
  build).

Before the fix the new test failed with:

```
Cargo.toml [package] must declare `description` (Issue #1241)
```

After applying the manifest edit the test passes.

## Test Plan

- Added `tests/issue_1241_cargo_toml_metadata.rs::cargo_toml_declares_required_package_metadata`
  which fails against the unfixed manifest and passes after the fix.
- Ran `cargo test --test issue_1241_cargo_toml_metadata` — 1 passed.
- Ran `./quality.sh` — all quality checks passed.
