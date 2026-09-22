//! Issue #2097 (chunk 16): the two rustup pin sets must not drift apart.
//!
//! The digest-verified rustup bootstrap of Issue #1911 exists twice in this
//! repository, because two scripts bootstrap rustup independently:
//!
//! * `scripts/install-rustup.sh` reads its digests from the committed manifest
//!   `scripts/rustup-init.sha256` and its version from `RUSTUP_VERSION`;
//! * `scripts/runlib.sh` — the canonical NEAT-AI-core helper, byte-synced into
//!   this repository — inlines the same six digests in
//!   `_runlib_pinned_rustup_digest` and the same version in
//!   `_RUNLIB_RUSTUP_VERSION`.
//!
//! Both files say in prose that the pin must move as one unit
//! (`scripts/runlib.sh` "To bump, change `_RUNLIB_RUSTUP_VERSION` and every
//! digest … together"; `scripts/rustup-init.sha256` "Both files must move
//! together"). Nothing asserted it. A bump applied to one file alone leaves the
//! other pinning a digest for a version it will never be served: the stale side
//! then fails closed on a mismatch that looks like tampering, and — worse for a
//! reviewer — a *wrong* digest copied into one side is invisible until the
//! bootstrap actually runs on a host with no rustc.
//!
//! These tests read the three committed files and compare them. No network, no
//! subprocess, no digest is recomputed: provenance for the pins is documentary
//! (the upstream-published `.sha256` values, recorded in
//! `scripts/rustup-init.sha256`), so re-downloading to "confirm" a pin would
//! verify the download against itself. What is checked here is the one property
//! no single file can hold on its own — that the two copies agree.
//!
//! `scripts/runlib.sh` is under the copy contract (Issue #2072): these tests
//! read it and never write it.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    let path = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// Every target triple both pin sets are expected to carry. Pinned here so a
/// parser that silently read nothing cannot make the comparisons vacuous.
const EXPECTED_TARGETS: [&str; 6] = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
];

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// `target -> digest` as committed in `scripts/rustup-init.sha256`.
///
/// Format is `<sha256>  <target-triple>`; comments and blank lines are ignored,
/// exactly as `install-rustup.sh::_pinned_digest` reads it.
fn manifest_pins() -> BTreeMap<String, String> {
    let mut pins = BTreeMap::new();
    for line in read("scripts/rustup-init.sha256").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let (Some(sha), Some(target)) = (fields.next(), fields.next()) else {
            panic!("malformed pin line in scripts/rustup-init.sha256: {line}");
        };
        assert!(
            is_sha256_hex(sha),
            "scripts/rustup-init.sha256 must pin a lower-case 64-hex SHA-256 for \
             {target}, found: {sha}"
        );
        assert!(
            pins.insert(target.to_owned(), sha.to_owned()).is_none(),
            "scripts/rustup-init.sha256 pins {target} twice — one target, one digest"
        );
    }
    pins
}

/// `target -> digest` as inlined in `scripts/runlib.sh::_runlib_pinned_rustup_digest`.
///
/// The arms read
/// ```text
///     x86_64-unknown-linux-gnu)
///       printf '%s' '<64 hex>' ;;
/// ```
/// so a bare `<word>)` line opens an arm and the following `printf` carries the
/// digest. The catch-all `*)` arm returns non-zero and pins nothing.
fn runlib_pins() -> BTreeMap<String, String> {
    let body = read("scripts/runlib.sh");
    let start = body
        .find("_runlib_pinned_rustup_digest() {")
        .expect("scripts/runlib.sh must define _runlib_pinned_rustup_digest (Issue #1911/#699)");

    let mut pins = BTreeMap::new();
    let mut target: Option<String> = None;
    for line in body[start..].lines().skip(1) {
        let line = line.trim();
        if line == "}" {
            break;
        }
        if let Some(digest) = single_quoted_sha256(line) {
            let Some(target) = target.take() else {
                panic!("scripts/runlib.sh inlines the digest {digest} under no target arm");
            };
            assert!(
                pins.insert(target.clone(), digest).is_none(),
                "scripts/runlib.sh pins {target} twice — one target, one digest"
            );
            continue;
        }
        // A bare `<target>)` opens an arm; `*)` is the refusal arm, and
        // `case "$1" in` / `esac` carry no closing parenthesis of their own.
        if let Some(name) = line.strip_suffix(')') {
            if !name.is_empty() && !name.contains('*') && !name.contains('"') {
                target = Some(name.to_owned());
            }
        }
    }
    pins
}

/// The 64-hex digest inside the first pair of single quotes on `line`, if any.
fn single_quoted_sha256(line: &str) -> Option<String> {
    let mut parts = line.split('\'');
    parts.next()?; // before the opening quote
    parts.find(|part| is_sha256_hex(part)).map(str::to_owned)
}

/// The value of a `NAME="value"` assignment, read as the first such line.
fn shell_assignment(relative: &str, name: &str) -> String {
    let needle = format!("{name}=\"");
    read(relative)
        .lines()
        .map(str::trim)
        .find_map(|line| {
            let rest = line.strip_prefix(needle.as_str())?;
            rest.split('"').next().map(str::to_owned)
        })
        .unwrap_or_else(|| panic!("{relative} must assign {name}"))
}

#[test]
fn both_pin_sets_cover_every_supported_target() {
    let manifest = manifest_pins();
    let runlib = runlib_pins();
    for target in EXPECTED_TARGETS {
        assert!(
            manifest.contains_key(target),
            "scripts/rustup-init.sha256 pins no digest for {target} — \
             install-rustup.sh fails closed on that host (Issue #1911)"
        );
        assert!(
            runlib.contains_key(target),
            "scripts/runlib.sh::_runlib_pinned_rustup_digest pins no digest for \
             {target} — the canonical bootstrap fails closed on that host"
        );
    }
}

#[test]
fn every_manifest_digest_is_inlined_in_runlib() {
    let manifest = manifest_pins();
    let runlib = runlib_pins();
    let script = read("scripts/runlib.sh");

    for (target, digest) in &manifest {
        let inlined = runlib.get(target).unwrap_or_else(|| {
            panic!(
                "scripts/rustup-init.sha256 pins {target} but \
                 scripts/runlib.sh does not — the two rustup pins have drifted \
                 (Issue #2097). Move both together."
            )
        });
        assert_eq!(
            inlined, digest,
            "rustup-init digest for {target} differs between \
             scripts/rustup-init.sha256 ({digest}) and scripts/runlib.sh \
             ({inlined}) — the two rustup pins have drifted (Issue #2097). \
             One of them will refuse a genuine download as tampering."
        );
        assert!(
            script.contains(digest.as_str()),
            "the digest {digest} pinned for {target} does not appear verbatim in \
             scripts/runlib.sh (Issue #2097)"
        );
    }
}

#[test]
fn runlib_pins_no_target_the_manifest_omits() {
    let manifest = manifest_pins();
    for target in runlib_pins().keys() {
        assert!(
            manifest.contains_key(target),
            "scripts/runlib.sh pins {target} but scripts/rustup-init.sha256 \
             does not — install-rustup.sh would refuse to install on a host \
             the canonical helper supports (Issue #2097)"
        );
    }
}

#[test]
fn the_two_rustup_version_pins_are_equal() {
    let install = shell_assignment("scripts/install-rustup.sh", "RUSTUP_VERSION");
    let runlib = shell_assignment("scripts/runlib.sh", "_RUNLIB_RUSTUP_VERSION");
    assert_eq!(
        install, runlib,
        "scripts/install-rustup.sh pins rustup {install} and scripts/runlib.sh \
         pins {runlib} — a version moved without the other side's digests can \
         only fail closed (Issue #2097). Move version and digests together."
    );
}
