//! Issue #1427: the `fuzz/` binary crate must ship a committed `Cargo.lock`.
//!
//! The repository enforces lockfile commitment for reproducible builds
//! (Issue #1266 covers the root `cdylib` crate). The `fuzz/` crate is a
//! **separate** crate — the root `Cargo.toml` declares no `[workspace]`
//! that spans it and `fuzz/Cargo.toml` carries its own `[workspace]` — so
//! the root `Cargo.lock` does not pin `fuzz/`'s dependency tree
//! (`libfuzzer-sys`, `serde_json`, etc.).
//!
//! Without a committed `fuzz/Cargo.lock` those dependencies float to
//! whatever the registry resolves at build time and escape the
//! `cargo audit` / `cargo deny` posture that the root tree enjoys. This
//! test enforces the same invariant for the fuzz crate as
//! `issue_1266_cargo_lock_committed.rs` enforces for the root crate:
//!   1. `fuzz/Cargo.lock` exists.
//!   2. `.gitignore` does not carry an active rule that would ignore it.

use std::fs;
use std::path::Path;

/// Decide whether a `.gitignore` pattern would ignore the fuzz crate's
/// `Cargo.lock`. Kept narrow — only the literal patterns that would match
/// `fuzz/Cargo.lock` count as a violation. Comment lines and negations are
/// skipped by the caller.
fn pattern_ignores_fuzz_cargo_lock(pattern: &str) -> bool {
    let pattern = pattern.trim();
    matches!(
        pattern,
        "Cargo.lock"
            | "/Cargo.lock"
            | "**/Cargo.lock"
            | "fuzz/Cargo.lock"
            | "/fuzz/Cargo.lock"
            | "fuzz/"
            | "/fuzz/"
            | "fuzz"
            | "/fuzz"
    )
}

#[test]
fn fuzz_cargo_lock_exists() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = manifest_dir.join("fuzz").join("Cargo.lock");
    assert!(
        lock.is_file(),
        "fuzz/Cargo.lock must be committed so the fuzz crate's dependency \
         tree is pinned and auditable (Issue #1427); expected {lock:?}. \
         Generate it with `cd fuzz && cargo generate-lockfile`."
    );
}

#[test]
fn gitignore_does_not_ignore_fuzz_cargo_lock() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gitignore = manifest_dir.join(".gitignore");
    let contents =
        fs::read_to_string(&gitignore).unwrap_or_else(|e| panic!("read {gitignore:?}: {e}"));

    let mut violations: Vec<(usize, String)> = Vec::new();
    for (idx, raw_line) in contents.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw_line.trim();
        // Skip blanks and comments.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Negation rules (`!fuzz/Cargo.lock`) re-include — not a violation.
        if line.starts_with('!') {
            continue;
        }
        // Strip an inline trailing comment if present.
        let pattern = line.split('#').next().unwrap_or(line).trim();
        if pattern_ignores_fuzz_cargo_lock(pattern) {
            violations.push((lineno, raw_line.to_string()));
        }
    }

    assert!(
        violations.is_empty(),
        "`.gitignore` must not ignore fuzz/Cargo.lock — the fuzz crate needs \
         a committed lockfile for a pinned, auditable dependency tree \
         (Issue #1427). Offending lines:\n{}",
        violations
            .iter()
            .map(|(n, l)| format!("  {n}: {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
