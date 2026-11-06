//! Issue #2558: the `semgrep` scan in `.github/workflows/semgrep.yml` must not
//! have `SEMGREP_APP_TOKEN` in scope. `semgrep ci` rejects an app token combined
//! with an explicit `--config`, so the scan step is a plain scan with no app
//! token at step, job or workflow level.
//!
//! This test reads the workflow as plain text (no YAML parser is in the
//! dependency tree). A line counts as carrying the token when its non-comment
//! content names it, so an `env:` entry at step, job or top level is caught, as
//! is a token on a continuation line of a multi-line `run: |` block. Full-line
//! and trailing `#` comments are ignored, so the explanatory comment in the
//! workflow does not trip the check.

use std::fs;
use std::path::Path;

const TOKEN: &str = "SEMGREP_APP_TOKEN";

fn load_semgrep_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/semgrep.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The non-comment part of one YAML line: empty for a full-line `#` comment,
/// cut before a trailing ` #...` comment otherwise.
fn code_part(line: &str) -> &str {
    if line.trim_start().starts_with('#') {
        return "";
    }
    let bytes = line.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i] == b'#' && matches!(bytes[i - 1], b' ' | b'\t') {
            return &line[..i];
        }
    }
    line
}

/// 1-based numbers of the lines whose non-comment content names the token.
fn token_in_scope_lines(yaml: &str) -> Vec<usize> {
    yaml.lines()
        .enumerate()
        .filter(|(_, line)| code_part(line).contains(TOKEN))
        .map(|(idx, _)| idx + 1)
        .collect()
}

/// True when a non-comment line carries the explicit scan invocation.
fn has_semgrep_ci_config_invocation(yaml: &str) -> bool {
    yaml.lines()
        .any(|line| code_part(line).contains("semgrep ci --config"))
}

#[test]
fn real_semgrep_yml_puts_no_app_token_in_scope() {
    let body = load_semgrep_yml();
    let hits = token_in_scope_lines(&body);
    assert!(
        hits.is_empty(),
        "semgrep.yml must not name SEMGREP_APP_TOKEN outside comments: \
         `semgrep ci` rejects an app token with an explicit --config \
         (Issue #2558); offending lines: {hits:?}",
    );
    // Positive control: the scan must still be present, so an emptied file
    // cannot pass the check above.
    assert!(
        has_semgrep_ci_config_invocation(&body),
        "semgrep.yml must still contain the non-comment `semgrep ci --config` \
         invocation (Issue #2558 positive control)",
    );
}

#[test]
fn flags_step_level_env_entry() {
    let yaml = r#"name: Semgrep
jobs:
  semgrep:
    steps:
      - run: semgrep ci --config p/default
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
"#;
    assert_eq!(token_in_scope_lines(yaml), vec![7]);
}

#[test]
fn flags_job_level_env_entry() {
    let yaml = r#"name: Semgrep
jobs:
  semgrep:
    env:
      SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
    steps:
      - run: semgrep ci --config p/default
"#;
    assert_eq!(token_in_scope_lines(yaml), vec![5]);
}

#[test]
fn flags_top_level_env_entry() {
    let yaml = r#"env:
  SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
on: pull_request
"#;
    assert_eq!(token_in_scope_lines(yaml), vec![2]);
}

#[test]
fn flags_token_on_continuation_line_of_multiline_run_block() {
    let yaml = r#"name: Semgrep
jobs:
  semgrep:
    steps:
      - run: |
          set -euo pipefail
          SEMGREP_APP_TOKEN="${{ secrets.SEMGREP_APP_TOKEN }}" \
            semgrep ci --config p/default
"#;
    assert_eq!(token_in_scope_lines(yaml), vec![7]);
}

#[test]
fn ignores_full_line_comment_naming_the_token() {
    let yaml = r#"name: Semgrep
jobs:
  semgrep:
    # SEMGREP_APP_TOKEN is deliberately absent (Issue #2558).
    steps:
      - run: semgrep ci --config p/default
"#;
    assert!(token_in_scope_lines(yaml).is_empty());
}

#[test]
fn ignores_trailing_comment_naming_the_token() {
    let yaml = r#"name: Semgrep
jobs:
  semgrep:
    steps:
      - run: semgrep ci --config p/default # no SEMGREP_APP_TOKEN here (Issue #2558)
"#;
    assert!(token_in_scope_lines(yaml).is_empty());
    // A trailing comment must not hide a token that is live on the same line.
    let live = "      SEMGREP_APP_TOKEN: x # trailing note\n";
    assert_eq!(token_in_scope_lines(live), vec![1]);
}
