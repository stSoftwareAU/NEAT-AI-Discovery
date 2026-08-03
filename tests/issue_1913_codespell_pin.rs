//! Issue #1913: the CI spell-check job installed codespell with a bare
//! `pip install --user codespell`, the last unpinned and unhashed tool
//! install in the repository. It resolved the newest release — and whatever
//! build backend its sdist shipped — fresh on every pull request, on a
//! runner holding a `GITHUB_TOKEN` (class A03, Software Supply Chain
//! Failures).
//!
//! Every sibling install is pinned: `cargo install --locked --version …`
//! (enforced by `quality/cargo_install_pinning.sh`, Issue #1912) and the
//! inline `markdownlint-cli2@<x.y.z>` npm pin (Issue #1484).
//!
//! This test enforces the configuration-level contract for the fix:
//!
//!   1. No workflow under `.github/workflows/` runs an unpinned `pip install`.
//!   2. The codespell install reads a `--require-hashes` requirements file,
//!      and that file pins an exact version with SHA-256 hashes.
//!   3. `renovate.json` tracks the requirements file via the
//!      `pip_requirements` manager, under the same 24h quarantine window as
//!      every other external dependency.
//!   4. The spell-check job still invokes the installed binary.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn requirements_path() -> PathBuf {
    root().join(".github/requirements/codespell-requirements.txt")
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
         report success while scanning nothing (Issue #1913)"
    );
    files
}

/// Strip a YAML/shell comment so a `pip install` mentioned in prose is not
/// mistaken for an invocation.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// True when the install command pins its inputs: either a `--require-hashes`
/// requirements file, or an inline exact `package==X.Y.Z` pin.
fn is_pinned_pip_install(command: &str) -> bool {
    if command.contains("--require-hashes") && command.contains("-r ") {
        return true;
    }
    command.split_whitespace().any(|token| {
        let Some((name, version)) = token.split_once("==") else {
            return false;
        };
        let parts: Vec<&str> = version.split('.').collect();
        !name.is_empty()
            && parts.len() >= 3
            && parts[..3]
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
    })
}

#[test]
fn no_unpinned_pip_install_in_workflows() {
    let mut offenders = Vec::new();
    let mut checked = 0usize;

    for file in workflow_files() {
        let contents = read(&file);
        for (idx, raw) in contents.lines().enumerate() {
            let line = strip_comment(raw);
            if !(line.contains("pip install") || line.contains("pip3 install")) {
                continue;
            }
            checked += 1;
            if !is_pinned_pip_install(line) {
                offenders.push(format!(
                    "{}:{} — {}",
                    file.file_name().unwrap().to_string_lossy(),
                    idx + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        checked > 0,
        "expected at least one `pip install` in .github/workflows (the \
         codespell install) — if it moved, update this gate rather than \
         letting it pass vacuously (Issue #1913)"
    );
    assert!(
        offenders.is_empty(),
        "every `pip install` in .github/workflows must pin its inputs — use \
         `--require-hashes -r <requirements file>` (preferred) or an exact \
         `package==X.Y.Z` pin (Issue #1913). Unpinned: {offenders:?}"
    );
}

#[test]
fn codespell_install_uses_the_hash_pinned_requirements_file() {
    let ci = read(&root().join(".github/workflows/ci.yml"));
    let line = ci
        .lines()
        .map(strip_comment)
        .find(|line| line.contains("pip install"))
        .unwrap_or_else(|| panic!("ci.yml must install codespell via pip install (Issue #1913)"));

    assert!(
        line.contains("--require-hashes"),
        "the codespell install must pass `--require-hashes` so pip refuses any \
         artefact whose SHA-256 does not match (Issue #1913) — found: {line}"
    );
    assert!(
        line.contains(".github/requirements/codespell-requirements.txt"),
        "the codespell install must read \
         `.github/requirements/codespell-requirements.txt` (Issue #1913) — \
         found: {line}"
    );
    assert!(
        requirements_path().is_file(),
        "the requirements file referenced by ci.yml must exist at {} \
         (Issue #1913)",
        requirements_path().display()
    );
}

#[test]
fn requirements_file_pins_an_exact_version_with_hashes() {
    let contents = read(&requirements_path());
    let body: String = contents
        .lines()
        .map(strip_comment)
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        is_pinned_pip_install(&body),
        "the requirements file must pin codespell to an exact \
         `codespell==X.Y.Z` release (Issue #1913)"
    );
    assert!(
        body.contains("codespell=="),
        "the requirements file must pin codespell itself (Issue #1913)"
    );

    let hashes: Vec<&str> = body
        .split_whitespace()
        .filter(|t| t.starts_with("--hash=sha256:"))
        .collect();
    assert!(
        !hashes.is_empty(),
        "the requirements file must carry `--hash=sha256:…` digests so \
         `--require-hashes` can verify the download (Issue #1913)"
    );
    for hash in &hashes {
        let digest = hash.trim_start_matches("--hash=sha256:");
        assert!(
            digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit()),
            "`{hash}` is not a 64-character hexadecimal SHA-256 digest \
             (Issue #1913)"
        );
    }
}

#[test]
fn renovate_tracks_the_codespell_requirements_file() {
    let contents = read(&root().join("renovate.json"));
    let config: serde_json::Value =
        serde_json::from_str(&contents).expect("renovate.json must be valid JSON");

    let patterns = config["pip_requirements"]["managerFilePatterns"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "renovate.json must configure `pip_requirements.managerFilePatterns` \
                 so the codespell pin receives managed bumps (Issue #1913)"
            )
        });
    assert!(
        patterns
            .iter()
            .any(|p| p.as_str().is_some_and(|s| s.contains("requirements"))),
        "the `pip_requirements` file patterns must cover the codespell \
         requirements file (Issue #1913) — found: {patterns:?}"
    );

    let quarantined = config["packageRules"]
        .as_array()
        .expect("renovate.json must declare packageRules")
        .iter()
        .any(|rule| {
            let matches_manager = rule["matchManagers"]
                .as_array()
                .is_some_and(|m| m.iter().any(|v| v.as_str() == Some("pip_requirements")));
            matches_manager && rule["minimumReleaseAge"].as_str() == Some("24h")
        });
    assert!(
        quarantined,
        "renovate.json must hold `pip_requirements` bumps for the same 24h \
         quarantine window as every other external dependency (Issue #1913)"
    );
}

#[test]
fn spell_check_job_still_runs_codespell() {
    let ci = read(&root().join(".github/workflows/ci.yml"));
    assert!(
        ci.contains("codespell \\") || ci.contains("bin/codespell"),
        "the spell-check job must still invoke the installed codespell binary \
         (Issue #1913)"
    );
    assert!(
        ci.contains("--check-filenames") && ci.contains("--check-hidden"),
        "the codespell invocation must keep its existing flags so it reports \
         the same findings (Issue #1913)"
    );
}
