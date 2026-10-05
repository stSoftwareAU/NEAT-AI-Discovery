//! Issue #2395: `AtomicUsize::try_update` (src/cancellation.rs) is stable
//! only from Rust 1.95, but `Cargo.toml` declared no `rust-version`. That
//! left `scripts/runlib.sh`'s toolchain gate — which upgrades a host to the
//! max of the crate's own `rust-version` and the dependency graph's — with
//! nothing to read, so hosts still on an older rustc (then 1.93.1) never
//! upgraded and the build broke on every consuming host.
//!
//! This test parses `Cargo.toml` and `.github/workflows/msrv.yml` as plain
//! text (no YAML/TOML parser is pulled in just for this) and asserts:
//!   - `Cargo.toml`'s `[package] rust-version` is declared and is at least
//!     1.95, the floor `AtomicUsize::try_update` needs.
//!   - `msrv.yml`'s `toolchain:` input matches that declared `rust-version`
//!     exactly, so the two cannot silently drift apart.
//!   - `msrv.yml` carries the hardening the sibling workflow tests
//!     (Issues #1288, #1287, #1891) check per-workflow: a `pull_request:`
//!     trigger with a `milestone/*` branch filter, read-only `contents`
//!     permissions, a non-persisted checkout, the shared `setup-rust`
//!     composite action, and the actual MSRV build command.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &str) -> String {
    let full = repo_root().join(path);
    fs::read_to_string(&full).unwrap_or_else(|e| panic!("read {}: {e}", full.display()))
}

/// Return the `rust-version` value from the `[package]` table only. Stops
/// at the next `[` table header (e.g. `[dependencies]`,
/// `[workspace.package]`) and ignores comment lines, so a `rust-version`
/// mentioned elsewhere in the manifest (a dependency table, a workspace
/// table) is not mistaken for the crate's own declaration.
fn package_rust_version(manifest: &str) -> Option<String> {
    let start = manifest.find("[package]")?;
    let after = &manifest[start + "[package]".len()..];
    for line in after.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            // Reached the next table header; [package] is exhausted.
            break;
        }
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("rust-version") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let value = rest.trim().trim_matches('"');
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Return the quoted or unquoted value of the `toolchain:` line in a
/// workflow's body.
fn msrv_toolchain(workflow: &str) -> Option<String> {
    for line in workflow.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("toolchain:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            return Some(value.to_string());
        }
    }
    None
}

/// Parse a dotted version string ("1.95", "1.93.1") into numeric
/// components, defaulting missing trailing components to zero so "1.95"
/// and "1.95.0" compare equal.
fn version_components(version: &str) -> [u64; 3] {
    let mut parts = [0u64; 3];
    for (i, segment) in version.split('.').take(3).enumerate() {
        parts[i] = segment.parse().unwrap_or(0);
    }
    parts
}

fn version_at_least(version: &str, floor: &str) -> bool {
    version_components(version) >= version_components(floor)
}

// --- Cargo.toml declares a sufficient rust-version ------------------------

#[test]
fn cargo_toml_declares_rust_version_at_or_above_the_try_update_floor() {
    let manifest = read("Cargo.toml");
    let declared = package_rust_version(&manifest).unwrap_or_else(|| {
        panic!(
            "Cargo.toml [package] must declare rust-version (Issue #2395): \
             AtomicUsize::try_update (src/cancellation.rs) needs Rust 1.95"
        )
    });
    assert!(
        version_at_least(&declared, "1.95"),
        "Cargo.toml [package] rust-version {declared} must be >= 1.95, the floor \
         AtomicUsize::try_update needs (Issue #2395)"
    );
}

// --- msrv.yml stays pinned to Cargo.toml's declared rust-version ----------

#[test]
fn msrv_workflow_toolchain_matches_cargo_toml_rust_version() {
    let manifest = read("Cargo.toml");
    let declared = package_rust_version(&manifest)
        .expect("Cargo.toml [package] must declare rust-version (Issue #2395)");

    let workflow = read(".github/workflows/msrv.yml");
    let toolchain =
        msrv_toolchain(&workflow).expect("msrv.yml must declare a toolchain: input (Issue #2395)");

    assert_eq!(
        toolchain, declared,
        "msrv.yml's toolchain ({toolchain}) must equal Cargo.toml's rust-version \
         ({declared}) exactly, so the two cannot drift apart (Issue #2395)"
    );
}

// --- msrv.yml carries the usual workflow hardening ------------------------

#[test]
fn msrv_workflow_has_expected_shape() {
    let workflow = read(".github/workflows/msrv.yml");

    assert!(
        workflow.contains("pull_request:"),
        "msrv.yml must run on pull_request (Issue #2395)"
    );
    assert!(
        workflow.contains("milestone/*"),
        "msrv.yml's branch filter must include milestone/* (Issue #2395)"
    );
    assert!(
        workflow.contains("contents: read"),
        "msrv.yml must declare read-only contents permissions (Issue #2395)"
    );
    assert!(
        workflow.contains("persist-credentials: false"),
        "msrv.yml's checkout must not persist credentials (Issue #2395)"
    );
    assert!(
        workflow.contains("uses: ./.github/actions/setup-rust"),
        "msrv.yml must bootstrap Rust via the shared setup-rust action (Issue #2395)"
    );
    assert!(
        workflow.contains("cargo check --locked --all-targets --all-features"),
        "msrv.yml must build every target with --locked --all-targets --all-features \
         (Issue #2395)"
    );
}

// --- Helper unit tests -----------------------------------------------------

#[test]
fn package_rust_version_returns_none_when_absent() {
    let manifest = "[package]\nname = \"x\"\nedition = \"2024\"\n\n[dependencies]\nfoo = \"1\"\n";
    assert_eq!(package_rust_version(manifest), None);
}

#[test]
fn package_rust_version_ignores_dependency_table_rust_version() {
    let manifest = "[package]\nname = \"x\"\n\n\
                     [dependencies.foo]\nversion = \"1\"\nrust-version = \"1.50\"\n";
    assert_eq!(package_rust_version(manifest), None);
}

#[test]
fn package_rust_version_ignores_workspace_package_rust_version() {
    let manifest = "[package]\nname = \"x\"\n\n\
                     [workspace.package]\nrust-version = \"1.40\"\n";
    assert_eq!(package_rust_version(manifest), None);
}

#[test]
fn package_rust_version_reads_the_package_table_value() {
    let manifest = "[package]\nname = \"x\"\nrust-version = \"1.95\"\nedition = \"2024\"\n";
    assert_eq!(package_rust_version(manifest), Some("1.95".to_string()));
}

#[test]
fn version_1_94_is_below_1_95() {
    assert!(!version_at_least("1.94", "1.95"));
}

#[test]
fn version_1_95_0_equals_1_95() {
    assert_eq!(version_components("1.95.0"), version_components("1.95"));
    assert!(version_at_least("1.95.0", "1.95"));
    assert!(version_at_least("1.95", "1.95.0"));
}

#[test]
fn version_1_100_is_above_1_95_numerically_not_lexically() {
    assert!(version_at_least("1.100", "1.95"));
}
