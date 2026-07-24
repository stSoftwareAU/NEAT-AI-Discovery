//! Committed fixtures must be self-contained and synthetic (Issue #1722).
//!
//! This public repository previously committed test fixtures captured from
//! private repositories — including a full production creature topology
//! (~3.2 MB) and recorded discovery-cache failure records. Anyone who cloned the
//! repository received that private data, and nobody outside the private
//! organisation could verify or refresh it against its stated source.
//!
//! Every fixture under `tests/fixtures/` is now hand-authored and synthetic. This
//! suite is the regression gate that keeps it that way: it fails loudly (Issue
//! #3234) the moment a fixture names a private source repository or grows to the
//! size that only a bulk production capture reaches.
//!
//! The two guards are deliberately independent — a private capture that is
//! renamed still trips the size guard, and a small private-derived record that
//! documents its private origin still trips the provenance guard.

use std::path::{Path, PathBuf};

/// Largest a committed fixture may be. Every legitimate hand-authored fixture is
/// a few kilobytes; the deleted production topology was ~3.2 MB, so this cap sits
/// far above honest fixtures and far below any bulk capture.
const MAX_FIXTURE_BYTES: u64 = 64 * 1024;

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Every file committed under `tests/fixtures/`, recursively.
fn fixture_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read fixture dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    walk(&fixtures_root(), &mut files);
    files.sort();
    files
}

/// Markers that a fixture (or its provenance README) points at a source
/// repository rather than being hand-authored here.
///
/// The needles are assembled at runtime from fragments so this guard does not
/// itself commit the private repository names it is written to keep out.
fn private_source_markers() -> Vec<String> {
    vec![
        format!("{}{}", "stSoftware", "AU/"),
        format!("{}{}", "GRQ", "-cluster"),
        format!("{}{}", "GRQ", "-Discovery"),
    ]
}

/// No committed fixture — data file or provenance README — may name a source
/// repository. A fixture that cites another repository is either copied from it
/// or unverifiable by the public, and both are defects here.
#[test]
fn no_fixture_names_a_private_source_repository() {
    let markers = private_source_markers();
    let mut offenders = Vec::new();

    for path in fixture_files() {
        // Fixtures are text (JSON / Markdown); a non-UTF-8 file is itself a
        // finding, so surface it rather than skipping it silently.
        let contents = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
        let text = String::from_utf8(contents).unwrap_or_else(|_| {
            panic!(
                "fixture {} is not UTF-8 text; committed fixtures must be readable JSON/Markdown",
                path.display()
            )
        });

        for marker in &markers {
            if text.contains(marker.as_str()) {
                offenders.push(format!("{} names `{marker}`", path.display()));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "committed fixtures must be hand-authored and self-contained, but {} reference a \
         source repository:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// Every committed fixture stays small. A multi-megabyte fixture is a bulk
/// capture of production data, not a hand-authored shape.
#[test]
fn every_fixture_is_small_enough_to_be_hand_authored() {
    let mut offenders = Vec::new();

    for path in fixture_files() {
        let size = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("failed to stat fixture {}: {e}", path.display()))
            .len();
        if size > MAX_FIXTURE_BYTES {
            offenders.push(format!("{} is {size} bytes", path.display()));
        }
    }

    assert!(
        offenders.is_empty(),
        "committed fixtures must stay under {MAX_FIXTURE_BYTES} bytes (hand-authored shapes, \
         not bulk captures), but {} exceed it:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// Harness-integrity guard: the walk must actually find fixtures. An empty set
/// would make both guards above pass vacuously.
#[test]
fn fixture_walk_finds_the_committed_fixtures() {
    let files = fixture_files();
    assert!(
        files.len() >= 8,
        "expected the committed fixture set to be discovered, found {} files",
        files.len()
    );
    assert!(
        files
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "json")),
        "fixture walk found no JSON fixtures"
    );
    assert!(
        files
            .iter()
            .any(|p| p.file_name().is_some_and(|n| n == "README.md")),
        "fixture walk found no provenance README"
    );
}
