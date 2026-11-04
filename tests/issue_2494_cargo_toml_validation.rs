//! Issue #2494: Rust port of `tests/test_cargo_toml_validation.sh` (Issue #691).
//!
//! The old script re-implemented the CI grep pattern and then tested its own
//! copy, so a change to the real `Validate Cargo.toml` step in
//! `.github/workflows/ci.yml` could never fail it. These tests read the real
//! step, extract its `if ! <pipeline>; then` line, and execute that exact
//! pipeline (under the same `set -euo pipefail` the step runs with) against
//! fixture manifests.
//!
//! Verdicts: exit 0 means the field was detected, exit 1 means it was not.
//! Any other status is a failure to run the pipeline, and panics rather than
//! being read as a negative result.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

const CI_YML: &str = ".github/workflows/ci.yml";
const STEP_NAME: &str = "Validate Cargo.toml";

fn read_ci_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(CI_YML);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// Returns the YAML text of the `Validate Cargo.toml` step, from its
/// `- name:` header up to (not including) the next sibling step.
fn extract_step(ci: &str) -> Result<String, String> {
    let lines: Vec<&str> = ci.lines().collect();
    let header = format!("- name: {STEP_NAME}");
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim() == header)
        .map(|(i, _)| i)
        .collect();
    let start = match starts.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "{CI_YML} has no step named '{STEP_NAME}'; the Cargo.toml field check cannot be verified"
            ));
        }
        many => {
            return Err(format!(
                "{CI_YML} has {} steps named '{STEP_NAME}'; expected exactly one",
                many.len()
            ));
        }
    };
    let indent = leading_spaces(lines[start]);
    let end = lines[start + 1..]
        .iter()
        .position(|l| leading_spaces(l) == indent && l.trim_start().starts_with("- name:"))
        .map_or(lines.len(), |p| start + 1 + p);
    Ok(lines[start..end].join("\n"))
}

/// Returns the pipeline text between `if ! ` and `; then` in the step, which
/// must contain exactly one such line and must use the `${field}` placeholder.
fn extract_pipeline(step: &str) -> Result<String, String> {
    let candidates: Vec<&str> = step
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("if ! ") && l.ends_with("; then"))
        .collect();
    let line = match candidates.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "step '{STEP_NAME}' in {CI_YML} has no `if ! <pipeline>; then` line"
            ));
        }
        many => {
            return Err(format!(
                "step '{STEP_NAME}' in {CI_YML} has {} `if ! <pipeline>; then` lines; expected exactly one",
                many.len()
            ));
        }
    };
    let pipeline = line
        .strip_prefix("if ! ")
        .and_then(|s| s.strip_suffix("; then"))
        .ok_or_else(|| format!("cannot split `if ! ... ; then` line in step '{STEP_NAME}'"))?;
    if !pipeline.contains("${field}") {
        return Err(format!(
            "pipeline in step '{STEP_NAME}' does not use the ${{field}} placeholder: {pipeline}"
        ));
    }
    Ok(pipeline.to_string())
}

/// Returns the words of the step's `for field in ...; do` loop.
fn step_field_list(step: &str) -> Result<Vec<String>, String> {
    let lines: Vec<&str> = step
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("for field in "))
        .collect();
    let line = match lines.as_slice() {
        [one] => *one,
        _ => {
            return Err(format!(
                "step '{STEP_NAME}' in {CI_YML} must have exactly one `for field in ...; do` line"
            ));
        }
    };
    let body = line
        .strip_prefix("for field in ")
        .and_then(|s| s.strip_suffix("; do"))
        .ok_or_else(|| format!("cannot parse field loop in step '{STEP_NAME}': {line}"))?;
    Ok(body
        .split_whitespace()
        .map(|w| w.trim_matches('"').to_string())
        .collect())
}

fn load_real_pipeline() -> String {
    let step = extract_step(&read_ci_yml()).unwrap_or_else(|e| panic!("{e}"));
    extract_pipeline(&step).unwrap_or_else(|e| panic!("{e}"))
}

/// Runs the pipeline as the CI step does (`set -euo pipefail`, field name in
/// the environment, stdin closed) in `dir`. Returns true for exit 0 (detected)
/// and false for exit 1 (not detected). Any other status panics.
fn pipeline_detects(pipeline: &str, field: &str, dir: &Path) -> bool {
    let script = format!("set -euo pipefail\n{pipeline}\n");
    let out = Command::new("bash")
        .arg("-c")
        .arg(&script)
        .env("field", field)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn bash: {e}"));
    match out.status.code() {
        Some(0) => true,
        Some(1) => false,
        other => panic!(
            "pipeline {pipeline:?} for field {field:?} exited with {other:?}, which is not a detection verdict; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}

/// Writes `manifest` as `Cargo.toml` in a fresh temporary directory and runs
/// the pipeline there.
fn detects_in_manifest(pipeline: &str, field: &str, manifest: &str) -> bool {
    let dir = tempfile::tempdir().expect("create tempdir");
    fs::write(dir.path().join("Cargo.toml"), manifest).expect("write fixture Cargo.toml");
    pipeline_detects(pipeline, field, dir.path())
}

fn assert_detected(pipeline: &str, description: &str, field: &str, manifest: &str) {
    assert!(
        detects_in_manifest(pipeline, field, manifest),
        "{description}: expected field '{field}' to be detected in {manifest:?}"
    );
}

fn assert_not_detected(pipeline: &str, description: &str, field: &str, manifest: &str) {
    assert!(
        !detects_in_manifest(pipeline, field, manifest),
        "{description}: expected field '{field}' NOT to be detected in {manifest:?}"
    );
}

// --- Uncommented fields are detected ---

#[test]
fn uncommented_name_version_and_edition_are_detected() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Uncommented name field",
        "name",
        "name = \"my-crate\"\n",
    );
    assert_detected(
        &p,
        "Uncommented version field",
        "version",
        "version = \"1.0.0\"\n",
    );
    assert_detected(
        &p,
        "Uncommented edition field",
        "edition",
        "edition = \"2021\"\n",
    );
}

// --- Commented-out fields are NOT detected, each paired with an uncommented control ---

#[test]
fn commented_out_name_is_not_detected() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Control: uncommented name",
        "name",
        "name = \"my-crate\"\n",
    );
    assert_not_detected(
        &p,
        "Commented-out name field",
        "name",
        "# name = \"my-crate\"\n",
    );
    assert_not_detected(
        &p,
        "Old pattern regression: commented name",
        "name",
        "# name = \"commented-out\"\n",
    );
}

#[test]
fn commented_out_version_with_leading_whitespace_is_not_detected() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Control: uncommented version",
        "version",
        "version = \"1.0.0\"\n",
    );
    assert_detected(
        &p,
        "Control: indented uncommented version",
        "version",
        "  version = \"1.0.0\"\n",
    );
    assert_not_detected(
        &p,
        "Commented-out version field with leading whitespace",
        "version",
        "  # version = \"1.0.0\"\n",
    );
}

#[test]
fn commented_out_edition_is_not_detected() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Control: uncommented edition",
        "edition",
        "edition = \"2021\"\n",
    );
    assert_not_detected(
        &p,
        "Commented-out edition field",
        "edition",
        "# edition = \"2021\"\n",
    );
}

// --- Substring fields must NOT match ---

#[test]
fn rust_version_does_not_match_version() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Control: exact version field",
        "version",
        "version = \"1.70\"\n",
    );
    assert_not_detected(
        &p,
        "rust-version must not match version",
        "version",
        "rust-version = \"1.70\"\n",
    );
    assert_not_detected(
        &p,
        "Old pattern regression: rust-version matching version",
        "version",
        "rust-version = \"1.70\"\n",
    );
}

#[test]
fn build_edition_does_not_match_edition() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Control: exact edition field",
        "edition",
        "edition = \"2021\"\n",
    );
    assert_not_detected(
        &p,
        "build-edition must not match edition",
        "edition",
        "build-edition = \"2021\"\n",
    );
}

// --- Varying whitespace is still detected ---

#[test]
fn fields_with_varying_whitespace_are_detected() {
    let p = load_real_pipeline();
    assert_detected(
        &p,
        "Field with spaces around equals",
        "name",
        "name   =   \"my-crate\"\n",
    );
    assert_detected(
        &p,
        "Field with leading whitespace",
        "version",
        "  version = \"1.0.0\"\n",
    );
    assert_detected(
        &p,
        "Field with tab before equals",
        "edition",
        "edition\t= \"2021\"\n",
    );
}

// --- The real manifest carries every required field ---

#[test]
fn real_cargo_toml_contains_all_required_fields() {
    let p = load_real_pipeline();
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for field in ["name", "version", "edition"] {
        assert!(
            pipeline_detects(&p, field, manifest_dir),
            "real Cargo.toml at {} must contain an uncommented '{field}' field",
            manifest_dir.display()
        );
    }
}

// --- The step's field list still covers the required fields ---

#[test]
fn real_step_field_list_covers_name_version_and_edition() {
    let step = extract_step(&read_ci_yml()).unwrap_or_else(|e| panic!("{e}"));
    let fields = step_field_list(&step).unwrap_or_else(|e| panic!("{e}"));
    for required in ["name", "version", "edition"] {
        assert!(
            fields.iter().any(|f| f == required),
            "step '{STEP_NAME}' in {CI_YML} no longer checks '{required}'; loop fields: {fields:?}"
        );
    }
}

// --- Fail-closed extraction ---

#[test]
fn real_ci_step_and_pipeline_are_extractable() {
    let step = extract_step(&read_ci_yml()).unwrap_or_else(|e| panic!("{e}"));
    let pipeline = extract_pipeline(&step).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        pipeline.starts_with("grep -vE ") && pipeline.contains("| grep -E "),
        "unexpected pipeline shape: {pipeline}"
    );
    // `grep -q` exits on the first match, so under `pipefail` the left-hand grep can
    // die of SIGPIPE (141) and the step wrongly reports a present field as missing.
    assert!(
        !pipeline.contains("grep -qE") && !pipeline.contains("grep -q "),
        "right-hand grep must read all input, not exit early with -q: {pipeline}"
    );
}

#[test]
fn extractor_fails_closed_when_step_is_renamed() {
    let ci = read_ci_yml();
    let header = format!("- name: {STEP_NAME}");
    assert_eq!(
        ci.matches(&header).count(),
        1,
        "fixture precondition: the real ci.yml must contain the step header exactly once"
    );
    let renamed = ci.replace(&header, "- name: Validate Cargo manifest");
    assert_ne!(
        renamed, ci,
        "fixture precondition: rename must change the text"
    );
    let err = extract_step(&renamed).expect_err("renamed step must not extract");
    assert!(
        err.contains(STEP_NAME),
        "error must name the missing step '{STEP_NAME}', got: {err}"
    );
}

#[test]
fn extractor_fails_closed_when_pipeline_line_is_missing() {
    let step = "    - name: Validate Cargo.toml\n      run: |\n        echo hi\n";
    let err = extract_pipeline(step).expect_err("step without an `if !` line must not extract");
    assert!(
        err.contains(STEP_NAME),
        "error must name the step '{STEP_NAME}', got: {err}"
    );
}
