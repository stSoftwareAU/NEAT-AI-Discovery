//! Issue #2097 (chunk 16): the two rustup pin sets must not drift apart.
//!
//! The digest-verified rustup bootstrap of Issue #1911 exists twice in this
//! repository, because two scripts bootstrap rustup independently:
//!
//! * `scripts/install-rustup.sh` — `_pinned_digest` reads the committed
//!   manifest `scripts/rustup-init.sha256`, and `RUSTUP_VERSION` names the
//!   release those digests belong to;
//! * `scripts/runlib.sh` — the canonical NEAT-AI-core helper, byte-synced into
//!   this repository — inlines the same six digests in
//!   `_runlib_pinned_rustup_digest` and the same release in
//!   `_RUNLIB_RUSTUP_VERSION`.
//!
//! Both files say in prose that the pin must move as one unit
//! (`scripts/runlib.sh` "To bump, change `_RUNLIB_RUSTUP_VERSION` and every
//! digest … together"; `scripts/rustup-init.sha256` "Both files must move
//! together"). Nothing asserted it. A bump applied to one side alone leaves the
//! other pinning a digest for a release it will never be served: that side then
//! fails closed on a mismatch indistinguishable from tampering, and a *wrong*
//! digest copied into one side stays invisible until the bootstrap actually
//! runs on a host with no rustc.
//!
//! These tests source each script and call its own pin reader, so they assert
//! on what the scripts answer rather than on how they are written — the house
//! pattern `tests/issue_1911_rustup_digest_verification.rs` established for
//! `_pinned_digest`. Both scripts guard their entry point on
//! `BASH_SOURCE[0] == $0`, so sourcing defines the helpers and runs nothing.
//!
//! No network and no digest is recomputed. Provenance for the pins is
//! documentary — the upstream-published `.sha256` values, recorded in
//! `scripts/rustup-init.sha256` — so re-downloading to "confirm" one would
//! verify the artefact against itself. What is checked here is the property no
//! single file can hold on its own: that the two copies agree.
//!
//! `scripts/runlib.sh` is under the copy contract (Issue #2072): these tests
//! read it and never write it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every target triple both pin sets are expected to answer for. Held here so a
/// reader that silently returned nothing cannot make the comparisons vacuous;
/// `the_committed_manifest_declares_exactly_the_supported_targets` keeps this
/// list honest against the manifest.
const SUPPORTED_TARGETS: [&str; 6] = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
];

/// Target triples neither pin set supports, used to prove a refusal is a real
/// refusal rather than an empty answer the comparison would accept.
const UNSUPPORTED_TARGETS: [&str; 3] = [
    "sparc64-unknown-linux-gnu",
    "i686-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
];

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// Run `script` in bash with the script sourced first, and return the result.
fn sourced(script: &str, body: &str) -> Output {
    let program = format!("source './{script}'\n{body}\n");
    Command::new("bash")
        .arg("-c")
        .arg(program)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|error| panic!("run bash with {script} sourced: {error}"))
}

/// `target -> digest` as the sourced `script` answers for every supported
/// target, by calling `reader` — the script's own pin function.
///
/// A target the reader refuses is recorded as an empty digest, so a refusal is
/// visible to the caller rather than silently dropped.
fn pins_answered_by(script: &str, reader: &str) -> BTreeMap<String, String> {
    let targets = SUPPORTED_TARGETS.join(" ");
    let out = sourced(
        script,
        &format!(
            "for target in {targets}; do \
               printf '%s\\t%s\\n' \"$target\" \"$({reader} \"$target\" 2>/dev/null || true)\"; \
             done"
        ),
    );
    assert!(
        out.status.success(),
        "sourcing {script} and calling {reader} must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mut pins = BTreeMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((target, digest)) = line.split_once('\t') else {
            panic!("{reader} produced an unreadable line: {line}");
        };
        assert!(
            pins.insert(target.to_owned(), digest.to_owned()).is_none(),
            "{reader} answered for {target} twice"
        );
    }
    pins
}

/// The value the sourced `script` holds in shell variable `name`.
fn pinned_version(script: &str, name: &str) -> String {
    let out = sourced(script, &format!("printf '%s' \"${name}\""));
    assert!(
        out.status.success(),
        "sourcing {script} to read {name} must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    assert!(
        !value.is_empty(),
        "{script} must pin a rustup release in {name}"
    );
    value
}

/// The target column of the committed digest manifest — the data file
/// `install-rustup.sh::_pinned_digest` itself reads.
fn manifest_targets() -> Vec<String> {
    let path = repo_root().join("scripts/rustup-init.sha256");
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut targets: Vec<String> = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split_whitespace();
            let (Some(_sha), Some(target)) = (fields.next(), fields.next()) else {
                panic!("malformed pin line in scripts/rustup-init.sha256: {line}");
            };
            target.to_owned()
        })
        .collect();
    targets.sort();
    targets
}

#[test]
fn the_committed_manifest_declares_exactly_the_supported_targets() {
    let mut expected: Vec<String> = SUPPORTED_TARGETS.iter().map(|t| (*t).to_owned()).collect();
    expected.sort();
    assert_eq!(
        manifest_targets(),
        expected,
        "scripts/rustup-init.sha256 no longer declares exactly the targets this \
         parity check covers — add the new target to both pin sets and to \
         SUPPORTED_TARGETS here, so the comparison keeps covering all of them \
         (Issue #2097)"
    );
}

#[test]
fn both_pin_sets_answer_for_every_supported_target() {
    let manifest = pins_answered_by("scripts/install-rustup.sh", "_pinned_digest");
    let runlib = pins_answered_by("scripts/runlib.sh", "_runlib_pinned_rustup_digest");

    for target in SUPPORTED_TARGETS {
        let from_manifest = manifest.get(target).map(String::as_str).unwrap_or("");
        let from_runlib = runlib.get(target).map(String::as_str).unwrap_or("");
        assert!(
            is_sha256_hex(from_manifest),
            "scripts/install-rustup.sh::_pinned_digest gave no lower-case 64-hex \
             SHA-256 for {target} — the bootstrap fails closed on that host \
             (Issue #1911). Got: {from_manifest:?}"
        );
        assert!(
            is_sha256_hex(from_runlib),
            "scripts/runlib.sh::_runlib_pinned_rustup_digest gave no lower-case \
             64-hex SHA-256 for {target} — the canonical bootstrap fails closed \
             on that host. Got: {from_runlib:?}"
        );
    }
}

#[test]
fn every_digest_matches_between_the_two_pin_sets() {
    let manifest = pins_answered_by("scripts/install-rustup.sh", "_pinned_digest");
    let runlib = pins_answered_by("scripts/runlib.sh", "_runlib_pinned_rustup_digest");

    for target in SUPPORTED_TARGETS {
        let from_manifest = manifest
            .get(target)
            .unwrap_or_else(|| panic!("install-rustup.sh answered nothing for {target}"));
        let from_runlib = runlib
            .get(target)
            .unwrap_or_else(|| panic!("runlib.sh answered nothing for {target}"));
        assert_eq!(
            from_manifest, from_runlib,
            "the rustup-init digest for {target} differs between \
             scripts/install-rustup.sh ({from_manifest}) and scripts/runlib.sh \
             ({from_runlib}) — the two rustup pins have drifted (Issue #2097). \
             One of them will refuse a genuine download as tampering; move both \
             together."
        );
    }
}

#[test]
fn both_pin_sets_refuse_a_target_neither_supports() {
    for target in UNSUPPORTED_TARGETS {
        for (script, reader) in [
            ("scripts/install-rustup.sh", "_pinned_digest"),
            ("scripts/runlib.sh", "_runlib_pinned_rustup_digest"),
        ] {
            let out = sourced(script, &format!("{reader} '{target}'"));
            assert!(
                !out.status.success(),
                "{script}::{reader} must refuse the unpinned target {target} — \
                 no pinned digest means no install (Issue #1911)"
            );
            assert!(
                String::from_utf8_lossy(&out.stdout).trim().is_empty(),
                "{script}::{reader} must print nothing for the unpinned target \
                 {target}, so a refusal cannot be mistaken for a digest"
            );
        }
    }
}

#[test]
fn the_two_rustup_version_pins_are_equal() {
    let install = pinned_version("scripts/install-rustup.sh", "RUSTUP_VERSION");
    let runlib = pinned_version("scripts/runlib.sh", "_RUNLIB_RUSTUP_VERSION");
    assert_eq!(
        install, runlib,
        "scripts/install-rustup.sh pins rustup {install} and scripts/runlib.sh \
         pins {runlib} — a release moved without the other side's digests can \
         only fail closed (Issue #2097). Move release and digests together."
    );
}
