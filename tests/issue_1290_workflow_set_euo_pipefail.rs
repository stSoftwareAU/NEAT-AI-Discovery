//! Issue #1290: every multi-line `run: |` block in `.github/workflows/ci.yml`
//! that was previously running unguarded bash must now start with
//! `set -euo pipefail` (and declare `shell: bash` to be explicit).
//!
//! Without `set -e` a failed intermediate command (e.g. a `sed` rewrite
//! of `Cargo.toml`) can fall through and the step still exits 0. Without
//! `-u` a typo in `${VAR}` silently expands to empty. Without
//! `-o pipefail` failures in piped commands are hidden by the exit code
//! of the last stage.
//!
//! This test parses `ci.yml` as plain text (no YAML parser is in the
//! dependency tree), locates each named step listed in the issue, and
//! asserts that the step's `run: |` block contains `set -euo pipefail`
//! on a line of its own and that the step declares `shell: bash`.

use std::fs;
use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn read_workflow(file_name: &str) -> String {
    let path = workflows_dir().join(file_name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Locate a step by its `- name: <step_name>` marker and return the
/// block of the step up to the next sibling step or end of file. Steps
/// are identified by a line of the form `    - name: <step_name>`
/// (four-space indent typical of a job's `steps:` list).
fn step_block<'a>(body: &'a str, step_name: &str) -> Option<&'a str> {
    let header = format!("    - name: {step_name}\n");
    let start = body.find(&header)?;
    let after = &body[start..];
    // Find next sibling step ("    - name:") or next job (two-space indent)
    // by scanning subsequent lines.
    let mut idx = header.len();
    let bytes = after.as_bytes();
    while idx < bytes.len() {
        if let Some(nl) = after[idx..].find('\n') {
            let line_start = idx + nl + 1;
            if line_start >= bytes.len() {
                return Some(after);
            }
            let line_end = after[line_start..]
                .find('\n')
                .map_or(bytes.len(), |n| line_start + n);
            let line = &after[line_start..line_end];
            // Sibling step at same indent, or a new job header.
            if line.starts_with("    - name:")
                || (line.starts_with("  ")
                    && line.as_bytes().get(2).is_some_and(|b| *b != b' ')
                    && line.trim_end().ends_with(':'))
            {
                return Some(&after[..line_start]);
            }
            idx = line_start;
        } else {
            return Some(after);
        }
    }
    Some(after)
}

fn has_set_euo_pipefail(block: &str) -> bool {
    block.lines().any(|line| line.trim() == "set -euo pipefail")
}

fn declares_shell_bash(block: &str) -> bool {
    block.lines().any(|line| line.trim() == "shell: bash")
}

/// The exact step names called out in Issue #1290 — every multi-line
/// bash block in ci.yml that was running unguarded.
const GUARDED_STEPS: &[&str] = &[
    "Check for source changes and increment version",
    "Check if there are changes",
    "Free up runner disk space",
    "Clean intermediate artefacts before tests",
    "Check for required files",
    "Validate Cargo.toml",
    "Check documentation",
    "Ensure bash scripts are executable",
];

#[test]
fn ci_yml_guarded_steps_contain_set_euo_pipefail() {
    let body = read_workflow("ci.yml");
    for step_name in GUARDED_STEPS {
        let block = step_block(&body, step_name)
            .unwrap_or_else(|| panic!("step `{step_name}` not found in ci.yml"));
        assert!(
            has_set_euo_pipefail(block),
            "step `{step_name}` in ci.yml must start its `run:` block with \
             `set -euo pipefail` so unset variables and pipe failures are not \
             silently swallowed (Issue #1290)",
        );
    }
}

#[test]
fn ci_yml_guarded_steps_declare_shell_bash() {
    let body = read_workflow("ci.yml");
    for step_name in GUARDED_STEPS {
        let block = step_block(&body, step_name)
            .unwrap_or_else(|| panic!("step `{step_name}` not found in ci.yml"));
        assert!(
            declares_shell_bash(block),
            "step `{step_name}` in ci.yml must declare `shell: bash` so the \
             `set -euo pipefail` guard is honoured by the explicit interpreter \
             (Issue #1290)",
        );
    }
}
