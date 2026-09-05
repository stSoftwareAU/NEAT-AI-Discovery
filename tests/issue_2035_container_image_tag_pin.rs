//! Issue #2035: the Semgrep job pinned its container to a **bare digest** —
//! `semgrep/semgrep@sha256:7cad2bc…` with no release tag beside it.
//!
//! A bare digest is immutable but untrackable. Both Renovate (`github-actions`
//! manager) and Dependabot (`docker` ecosystem) resolve a bump from the *tag*
//! and then rewrite the digest beside it; with no tag there is nothing to
//! resolve, so the image freezes at whatever the tag pointed at on the day it
//! was pinned and no updater — nor the repo's own `bump-deps.sh` — has a
//! version string to key a bump off.
//!
//! The fix carries both: `name:<tag>@sha256:<digest>`. The digest keeps the
//! image byte-for-byte immutable; the tag gives an updater the version to
//! resolve from, and the existing `github-actions` packageRule in
//! `renovate.json` then applies the same 24h quarantine as every other
//! external dependency (Issue #1234).
//!
//! These tests parse the workflow files as plain text (no YAML parser is in
//! the dependency tree, matching the sibling workflow tests) and enforce the
//! contract for **every** container image in **every** workflow, not just the
//! Semgrep one.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `*.yml` / `*.yaml` file under `.github/workflows/`.
fn workflow_files() -> Vec<PathBuf> {
    let dir = root().join(".github/workflows");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|entry| entry.expect("workflow dir entry").path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("yml" | "yaml")))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no workflow files found under .github/workflows — this gate must not \
         report success while scanning nothing (Issue #2035)"
    );
    files
}

/// Strip a YAML comment so an image reference quoted in prose is not mistaken
/// for a declaration.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// A container image declared by a workflow: file name, 1-based line, and the
/// image reference itself.
struct ImageRef {
    file: String,
    line: usize,
    image: String,
}

/// Collect every `image:` value declared across the workflow files. In GitHub
/// Actions the key appears only under `jobs.<job>.container` and
/// `jobs.<job>.services.<id>`, both of which take a container image reference.
fn container_images() -> Vec<ImageRef> {
    let mut found = Vec::new();
    for path in workflow_files() {
        let file = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("workflow file name")
            .to_string();
        for (idx, line) in read(&path).lines().enumerate() {
            let code = strip_comment(line);
            let Some(rest) = code.trim_start().strip_prefix("image:") else {
                continue;
            };
            let image = rest.trim().trim_matches(['"', '\''].as_slice()).to_string();
            if image.is_empty() {
                continue;
            }
            found.push(ImageRef {
                file: file.clone(),
                line: idx + 1,
                image,
            });
        }
    }
    found
}

/// Validate a container image reference: it must carry an explicit,
/// immutable release tag **and** a `sha256` content digest.
///
/// Returns `Err(reason)` describing the first violation found.
fn validate_image_pin(image: &str) -> Result<(), String> {
    if image.contains("${{") {
        return Err("image is a workflow expression, so its pin cannot be verified".to_string());
    }

    let Some((name_and_tag, digest)) = image.split_once('@') else {
        return Err("no `@sha256:` content digest — the image is mutable".to_string());
    };

    let Some(hex) = digest.strip_prefix("sha256:") else {
        return Err(format!("digest `{digest}` is not a `sha256:` digest"));
    };
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("`{digest}` is not a 64-character sha256 digest"));
    }

    // A registry host may carry a port (`registry:5000/name`), so only the
    // final path segment can hold the tag separator.
    let last_segment = name_and_tag.rsplit('/').next().unwrap_or(name_and_tag);
    let Some((_, tag)) = last_segment.split_once(':') else {
        return Err(
            "bare digest with no release tag — no updater can resolve a bump from it".to_string(),
        );
    };
    if tag.is_empty() {
        return Err("empty release tag".to_string());
    }
    if tag == "latest" {
        return Err("`latest` is not a release tag — it names no version to bump from".to_string());
    }
    Ok(())
}

#[test]
fn every_workflow_container_image_carries_a_tag_and_a_digest() {
    for image in container_images() {
        if let Err(reason) = validate_image_pin(&image.image) {
            panic!(
                "{}:{}: container image `{}` must be pinned as `name:<tag>@sha256:<digest>` \
                 (Issue #2035) — {reason}",
                image.file, image.line, image.image,
            );
        }
    }
}

#[test]
fn container_image_gate_scans_at_least_one_image() {
    // A gate that silently scans nothing reports success while enforcing
    // nothing — the repository declares a container image, so finding none
    // means the parser broke, not that the tree is clean.
    let images = container_images();
    assert!(
        !images.is_empty(),
        "no `image:` declaration found in any workflow — the parser in this \
         test has stopped matching the workflow syntax (Issue #2035)"
    );
}

#[test]
fn semgrep_job_pins_the_semgrep_image_by_tag_and_digest() {
    let images = container_images();
    let semgrep: Vec<&ImageRef> = images
        .iter()
        .filter(|i| i.file == "semgrep.yml" && i.image.starts_with("semgrep/semgrep"))
        .collect();
    assert_eq!(
        semgrep.len(),
        1,
        "semgrep.yml must declare exactly one `semgrep/semgrep` container image \
         (Issue #2035); found {}",
        semgrep.len()
    );
    let image = &semgrep[0].image;
    validate_image_pin(image).unwrap_or_else(|reason| {
        panic!("semgrep.yml container image `{image}` is not tag+digest pinned (Issue #2035) — {reason}")
    });
}

#[test]
fn a_bare_digest_pin_is_rejected() {
    // The exact pre-fix reference from semgrep.yml: immutable, but with no
    // tag for any updater to resolve a bump from.
    let reason = validate_image_pin(
        "semgrep/semgrep@sha256:7cad2bc2d1e44f87f0bf4be6d1fa23aa90fb72015bebc89fb91385d813987a03",
    )
    .expect_err("a bare-digest pin must be rejected (Issue #2035)");
    assert!(
        reason.contains("no release tag"),
        "the rejection must name the missing tag; got: {reason}"
    );
}

#[test]
fn a_tagged_digest_pin_is_accepted() {
    assert_eq!(
        validate_image_pin(
            "semgrep/semgrep:1.163.0@sha256:7cad2bc2d1e44f87f0bf4be6d1fa23aa90fb72015bebc89fb91385d813987a03"
        ),
        Ok(())
    );
    // A registry host with a port must not be mistaken for a tag.
    assert_eq!(
        validate_image_pin(
            "registry.example.com:5000/team/tool:2.1.0@sha256:0000000000000000000000000000000000000000000000000000000000000000"
        ),
        Ok(())
    );
}

#[test]
fn a_tagged_but_undigested_pin_is_rejected() {
    // The mutable shape the digest pin replaced (Issue #1258) must stay
    // rejected — a tag alone is not immutable.
    let reason = validate_image_pin("semgrep/semgrep:1.163.0")
        .expect_err("a tag-only pin must be rejected (Issue #2035)");
    assert!(
        reason.contains("digest"),
        "the rejection must name the missing digest; got: {reason}"
    );
}

#[test]
fn a_latest_tag_is_rejected() {
    let reason = validate_image_pin(
        "semgrep/semgrep:latest@sha256:7cad2bc2d1e44f87f0bf4be6d1fa23aa90fb72015bebc89fb91385d813987a03",
    )
    .expect_err("`latest` is not a release tag (Issue #2035)");
    assert!(
        reason.contains("latest"),
        "the rejection must name the mutable tag; got: {reason}"
    );
}

#[test]
fn a_malformed_digest_is_rejected() {
    for bad in [
        "semgrep/semgrep:1.163.0@sha256:deadbeef",
        "semgrep/semgrep:1.163.0@md5:7cad2bc2d1e44f87f0bf4be6d1fa23aa",
    ] {
        validate_image_pin(bad).expect_err(&format!("`{bad}` must be rejected (Issue #2035)"));
    }
}
