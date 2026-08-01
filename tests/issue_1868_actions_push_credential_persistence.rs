//! Issue #1868: the write-capable `ACTIONS_PUSH` PAT must never be persisted
//! in `.git/config`.
//!
//! `actions/checkout` defaults `persist-credentials` to `true`, which writes the
//! supplied token into `.git/config` as an auth extraheader that every later
//! step in the job can read. The `version-increment` job then runs
//! `cargo install cargo-outdated`, executing third-party `build.rs` code with
//! filesystem access — turning a poisoned crate into persistent repository write
//! access via the PAT.
//!
//! The fix keeps the credential off disk: every checkout in `ci.yml` sets
//! `persist-credentials: false`, and the PAT reaches only the individual steps
//! that talk to the remote, through an `env:` block.
//!
//! This test reads `ci.yml` as plain text (no YAML parser is in the dependency
//! tree).

use std::fs;
use std::path::Path;

fn load_ci_yml() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Split a workflow body into step blocks, keyed on the `- name:` line that
/// starts each step. Returns `(step name, step body)` pairs.
fn steps(body: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("- name:") && line.len() > trimmed.len() {
            let name = trimmed.trim_start_matches("- name:").trim().to_string();
            out.push((name, String::new()));
        } else if let Some(last) = out.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    out
}

#[test]
fn every_checkout_disables_credential_persistence() {
    let body = load_ci_yml();
    let checkouts: Vec<(String, String)> = steps(&body)
        .into_iter()
        .filter(|(_, block)| block.contains("uses: actions/checkout@"))
        .collect();

    assert!(
        checkouts.len() >= 5,
        "expected ci.yml to still contain its checkout steps, found {} (Issue #1868)",
        checkouts.len(),
    );

    for (name, block) in checkouts {
        assert!(
            block.contains("persist-credentials: false"),
            "checkout step `{name}` in ci.yml must set `persist-credentials: false` \
             so no token is written to `.git/config` (Issue #1868)",
        );
    }
}

#[test]
fn actions_push_pat_is_never_handed_to_checkout() {
    let body = load_ci_yml();
    for (lineno, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        assert!(
            !(trimmed.starts_with("token:") && trimmed.contains("secrets.ACTIONS_PUSH")),
            "ci.yml:{} passes the write-capable ACTIONS_PUSH PAT to a checkout step; \
             supply it via an `env:` block on the step that pushes instead (Issue #1868)",
            lineno + 1,
        );
    }
}

#[test]
fn actions_push_pat_is_only_exposed_as_a_step_env_var() {
    let body = load_ci_yml();
    let mut references = 0_usize;
    for line in body.lines() {
        if !line.contains("secrets.ACTIONS_PUSH") {
            continue;
        }
        references += 1;
        let trimmed = line.trim();
        assert!(
            trimmed.starts_with("ACTIONS_PUSH_TOKEN:"),
            "ACTIONS_PUSH must only be bound to the `ACTIONS_PUSH_TOKEN` env var of a \
             remote-facing step, found: `{trimmed}` (Issue #1868)",
        );
    }
    assert!(
        references > 0,
        "ci.yml must still use the ACTIONS_PUSH PAT for its pushes so they re-trigger \
         workflows (AGENTS.md, Issue #1868)",
    );
}

#[test]
fn remote_facing_steps_authenticate_with_an_explicit_url() {
    let body = load_ci_yml();
    let authenticated: Vec<(String, String)> = steps(&body)
        .into_iter()
        .filter(|(_, block)| block.contains("ACTIONS_PUSH_TOKEN:"))
        .collect();

    assert!(
        !authenticated.is_empty(),
        "no step binds ACTIONS_PUSH_TOKEN — the pushes would fail without the PAT \
         (Issue #1868)",
    );

    for (name, block) in authenticated {
        assert!(
            block.contains("https://x-access-token:${ACTIONS_PUSH_TOKEN}@github.com/"),
            "step `{name}` binds ACTIONS_PUSH_TOKEN but does not use it in an \
             authenticated remote URL (Issue #1868)",
        );
        assert!(
            !block.contains("git remote set-url"),
            "step `{name}` must not write the credential back into `.git/config` \
             (Issue #1868)",
        );
    }
}
