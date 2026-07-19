//! Issue #1658: the `quality` job in `.github/workflows/ci.yml` type-checked
//! the whole workspace twice — once via the "Run linter" clippy step and once
//! via a standalone "Check types" (`cargo check --all-targets --all-features`)
//! step immediately after it.
//!
//! `cargo clippy` drives the same compilation as `cargo check` (clippy is a
//! superset), over the identical `--all-targets --all-features` scope, with the
//! same `RUSTFLAGS: "-D warnings"` job environment. The "Check types" step could
//! therefore never fail unless the clippy step before it had already failed — it
//! added a full re-check pass to the heaviest job for zero added signal.
//!
//! The redundant "Check types" step is removed, leaving clippy as the single
//! broad compile gate. This test parses `ci.yml` as plain text and asserts:
//!   1. There is no standalone "Check types" `cargo check` step.
//!   2. The `quality` job retains no bare `cargo check --all-targets` invocation.
//!   3. The clippy "Run linter" step remains the broad compile gate.
//!   4. The `validation` job comment cross-references the surviving step
//!      ("Run linter"), not the removed one.

use std::fs;
use std::path::{Path, PathBuf};

fn read_ci() -> String {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".github/workflows")
        .join("ci.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn no_standalone_check_types_step() {
    let body = read_ci();
    assert!(
        !body.contains("- name: Check types"),
        "the redundant `Check types` step must be removed from ci.yml — clippy \
         (`Run linter`) already drives the same compilation (Issue #1658)",
    );
}

#[test]
fn quality_job_has_no_redundant_cargo_check() {
    let body = read_ci();
    assert!(
        !body.contains("cargo check --all-targets --all-features"),
        "ci.yml must not run `cargo check --all-targets --all-features` — it \
         duplicates the clippy `Run linter` compile gate over the same scope \
         (Issue #1658)",
    );
}

#[test]
fn linter_step_is_the_broad_compile_gate() {
    let body = read_ci();
    assert!(
        body.contains("cargo clippy --all-targets --all-features"),
        "the `Run linter` clippy step must remain as the single broad compile \
         gate over `--all-targets --all-features` (Issue #1658)",
    );
}

#[test]
fn validation_comment_points_at_surviving_step() {
    let body = read_ci();
    assert!(
        body.contains("the `quality` job (Run linter)"),
        "the `validation` job comment must cross-reference the surviving \
         `Run linter` step, not the removed `Check types` step (Issue #1658)",
    );
    assert!(
        !body.contains("the `quality` job (Check types)"),
        "the `validation` job comment must not reference the removed \
         `Check types` step (Issue #1658)",
    );
}
