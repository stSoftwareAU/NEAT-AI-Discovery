//! Shipped source, benches, examples, and tests must not name private
//! repositories (Issues #1724, #1725).
//!
//! Comments across `src/`, `benches/`, and `examples/` used to cite private
//! `stSoftwareAU` repositories by name — commit hashes, creature ids, and local
//! checkout paths in a sibling private repo — as the derivation evidence for
//! scoring constants and bench workloads. Nobody outside the private
//! organisation can inspect that evidence, so the citations carried zero
//! verification value for public readers while continuously naming private
//! infrastructure. The example went further and documented an invocation
//! against private checkout paths no public user can obtain.
//!
//! Issue #1725 extended the walk to `tests/`. Test doc comments, helper names,
//! and test *file* names are part of the public reading surface — `cargo test`
//! prints the file and test names on every run — so the same rule applies
//! there.
//!
//! Evidence is now cited at concept level ("production discovery-cache
//! analysis"), with the internal issue numbers already carried in most comments
//! preserving traceability for maintainers. This suite is the regression gate
//! that keeps it that way: it fails loudly (Issue #3234) the moment a private
//! repository name reappears.

use std::path::{Path, PathBuf};

/// Directories whose Rust sources ship to, or are read by, the public.
const SCANNED_DIRS: [&str; 4] = ["src", "benches", "examples", "tests"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under the scanned directories, recursively.
fn scanned_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let root = repo_root();
    let mut files = Vec::new();
    for dir in SCANNED_DIRS {
        walk(&root.join(dir), &mut files);
    }
    files.sort();
    files
}

/// Names of the private repositories this gate keeps out of shipped source.
///
/// The needles are assembled at runtime from fragments so this guard does not
/// itself commit the private repository names it is written to exclude. The
/// short prefix matches every variant (`-sampler`, `-cluster`, `-Discovery`,
/// and the numbered deployment ids) in one pass.
fn private_repo_markers() -> Vec<String> {
    vec![format!("{}{}", "GR", "Q")]
}

/// Report every offending line so one run lists the full remaining set rather
/// than failing one file at a time.
///
/// Matching is case-insensitive: the lower-case form is exactly how the name
/// reaches Rust identifiers (test-function and helper names), which
/// `cargo test` prints on every run.
fn offending_lines(text: &str, marker: &str) -> Vec<usize> {
    let needle = marker.to_lowercase();
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.to_lowercase().contains(&needle))
        .map(|(index, _)| index + 1)
        .collect()
}

/// No shipped Rust source may name a private repository.
#[test]
fn no_shipped_source_names_a_private_repository() {
    let markers = private_repo_markers();
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in scanned_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read source file {}: {e}", path.display()));
        let display = path.strip_prefix(&root).unwrap_or(&path).display();

        for marker in &markers {
            for line in offending_lines(&text, marker) {
                offenders.push(format!("{display}:{line} names a private repository"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "shipped source must cite evidence at concept level, not by private repository name, \
         but {} line(s) still do:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// No Rust *file name* may embed a private repository or deployment name.
///
/// A file name is louder than a comment: `cargo test` prints the test binary
/// path on every run, so a private name baked into the name leaks on output a
/// public contributor sees before they read a single line.
#[test]
fn no_source_file_name_embeds_a_private_repository() {
    let markers: Vec<String> = private_repo_markers()
        .iter()
        .map(|m| m.to_lowercase())
        .collect();
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in scanned_files() {
        let name = path
            .file_name()
            .unwrap_or_else(|| panic!("scanned path {} has no file name", path.display()))
            .to_string_lossy()
            .to_lowercase();
        let display = path.strip_prefix(&root).unwrap_or(&path).display();

        if markers.iter().any(|marker| name.contains(marker.as_str())) {
            offenders.push(format!("{display} names a private repository"));
        }
    }

    assert!(
        offenders.is_empty(),
        "file names must describe behaviour, not private deployments, but {} do not:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// The example's usage text must describe its inputs generically, not as
/// checkout paths inside a private sibling repository.
#[test]
fn snapshot_example_documents_public_reproducible_inputs() {
    let path = repo_root().join("examples/generate_snapshot.rs");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let private_checkout = format!("~/src/{}", private_repo_markers()[0]);
    assert!(
        !text.contains(&private_checkout),
        "examples/generate_snapshot.rs documents a private checkout path a public user cannot \
         obtain"
    );
    assert!(
        text.contains("discovery_data.parquet"),
        "examples/generate_snapshot.rs must still document the parquet input it expects"
    );
}

/// Harness-integrity guard: the walk must actually find sources. An empty set
/// would make the guard above pass vacuously.
#[test]
fn source_walk_finds_the_scanned_directories() {
    let files = scanned_files();
    assert!(
        files.len() >= 100,
        "expected the shipped Rust sources to be discovered, found {} files",
        files.len()
    );
    for dir in SCANNED_DIRS {
        assert!(
            files
                .iter()
                .any(|p| p.components().any(|c| c.as_os_str() == dir)),
            "source walk found no files under `{dir}/`"
        );
    }
}
