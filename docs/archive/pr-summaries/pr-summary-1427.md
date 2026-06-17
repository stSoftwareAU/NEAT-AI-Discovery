# SCR-LOCKFILE: commit a Cargo.lock for the fuzz crate (Issue #1427)

## Summary

The `fuzz/` binary crate shipped a `Cargo.toml` with declared `[[bin]]`
targets but **no committed `Cargo.lock`**. Because the fuzz crate is a
**separate** crate — the root `Cargo.toml` declares no spanning
`[workspace]` and `fuzz/Cargo.toml` carries its own `[workspace]` — the
root `Cargo.lock` does not pin the fuzz crate's dependency tree
(`libfuzzer-sys`, `serde_json`, the local path dep, and transitives).
Those dependencies floated to whatever the registry resolved at build
time and escaped the `cargo audit` / `cargo deny` posture the root tree
enjoys, leaving a supply-chain gap when the (template-only) fuzzing
workflow is activated.

This change closes that posture gap by committing a lockfile for the fuzz
crate and enforcing the invariant the same way the root lockfile is
enforced.

- Added `fuzz/Cargo.lock` via `cargo generate-lockfile` (240 packages
  pinned).
- Added a regression guard `tests/issue_1427_fuzz_cargo_lock_committed.rs`
  mirroring the root guard (`issue_1266_cargo_lock_committed.rs`):
  asserts `fuzz/Cargo.lock` exists and that `.gitignore` does not ignore
  it.
- Updated the fuzzing workflow template (`docs/ci-fuzzing-workflow.yml`)
  to run `cargo +nightly fuzz run --locked …` so CI honours the committed
  lockfile rather than re-resolving.

The root `Cargo.lock` change in this PR is the standard output of the
`quality.sh` dependency-refresh step (AGENTS.md quality gate step 2), not
a manual edit.

Closes #1427.

### Deno regression avoided

Not applicable — this is a Rust crate; no Node/Deno tooling involved.

## Evidence

Backend/CLI change with no web interface — no screenshot applicable.

Verified via the new regression test. Before generating the lockfile the
presence test fails; after, it passes:

```
# before (TDD red)
test fuzz_cargo_lock_exists ... FAILED
test gitignore_does_not_ignore_fuzz_cargo_lock ... ok

# after generating fuzz/Cargo.lock (TDD green)
test fuzz_cargo_lock_exists ... ok
test gitignore_does_not_ignore_fuzz_cargo_lock ... ok
```

`./quality.sh` passes cleanly (fmt, clippy `-D warnings`, check, full test
suite, doc build, release build).

```mermaid
flowchart LR
    A[fuzz/Cargo.toml<br/>own [workspace]] -->|no committed lock| B[deps float at build time]
    B -.escape.-> C[cargo audit / deny<br/>on root tree]
    A2[fuzz/Cargo.toml] -->|fuzz/Cargo.lock committed| D[deps pinned & reviewable]
    D --> E[cargo fuzz run --locked<br/>honours the lockfile]
```

## Test Plan

- Added `tests/issue_1427_fuzz_cargo_lock_committed.rs`:
  - `fuzz_cargo_lock_exists` — asserts `fuzz/Cargo.lock` is a committed
    file (reproduces the issue against the unfixed tree).
  - `gitignore_does_not_ignore_fuzz_cargo_lock` — asserts no active
    `.gitignore` rule would ignore `fuzz/Cargo.lock`.
- Confirmed `git check-ignore fuzz/Cargo.lock` reports the file is **not**
  ignored, so it is tracked.
- Existing `tests/issue_1266_cargo_lock_committed.rs` continues to pass
  (root lockfile invariant unchanged).
