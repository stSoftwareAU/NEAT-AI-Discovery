//! Issue #1266: `Cargo.lock` must be committed for reproducible builds of
//! the `cdylib` artefact consumed by NEAT-AI via FFI.
//!
//! The historical "library = no lockfile" advice was reversed by the
//! Cargo team in 2023 (see
//! <https://blog.rust-lang.org/2023/08/29/committing-lockfiles.html>).
//! For this crate it matters extra because:
//!
//! * `[lib] crate-type = ["cdylib", "rlib"]` — the build produces a
//!   shared-object artefact loaded by the NEAT-AI controller. Without a
//!   committed lockfile, two CI runs at the same commit can resolve to
//!   different transitive dependency versions.
//! * `cargo-deny` / `cargo-audit` operate on the resolved dependency
//!   graph in `Cargo.lock`. With the lockfile ignored, the scan in CI
//!   advises on a freshly-resolved graph that no human ever reviewed.
//! * Dependency-bump PRs raised by `bump-deps.sh` must produce a
//!   reviewable lockfile diff so transitive changes are visible during
//!   the `VIBE_BUMP_QUARANTINE_HOURS` cooling window.
//!
//! This test enforces both halves of the rule:
//!   1. `Cargo.lock` exists at the repo root.
//!   2. `.gitignore` does not carry an active rule that would ignore it.

use std::fs;
use std::path::Path;

/// Decide whether a `.gitignore` pattern (left of any trailing comment)
/// would ignore a top-level `Cargo.lock`. We intentionally keep this
/// narrow — only the literal patterns we know match the repo's root
/// `Cargo.lock` count as a violation. Comment lines and negations are
/// skipped by the caller.
fn pattern_ignores_root_cargo_lock(pattern: &str) -> bool {
    let pattern = pattern.trim();
    matches!(pattern, "Cargo.lock" | "/Cargo.lock" | "**/Cargo.lock")
}

#[test]
fn cargo_lock_exists_at_repo_root() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = manifest_dir.join("Cargo.lock");
    assert!(
        lock.is_file(),
        "Cargo.lock must be committed at the repo root for reproducible \
         cdylib builds (Issue #1266); expected {lock:?}"
    );
}

#[test]
fn gitignore_does_not_ignore_cargo_lock() {
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
        // Negation rules (`!Cargo.lock`) re-include — they are not a
        // violation; skip them.
        if line.starts_with('!') {
            continue;
        }
        // Strip an inline trailing comment if present.
        let pattern = line.split('#').next().unwrap_or(line).trim();
        if pattern_ignores_root_cargo_lock(pattern) {
            violations.push((lineno, raw_line.to_string()));
        }
    }

    assert!(
        violations.is_empty(),
        "`.gitignore` must not ignore Cargo.lock — the cdylib build needs \
         a committed lockfile (Issue #1266). Offending lines:\n{}",
        violations
            .iter()
            .map(|(n, l)| format!("  {n}: {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
