//! Issue #1486 (BP-8be698853743): the repository must ship a `CODEOWNERS`
//! file that forces mandatory owner review on the high-blast-radius paths the
//! best-practices audit flagged.
//!
//! Without `CODEOWNERS` coverage over `.github/workflows/`, a pull request that
//! edits a workflow can be merged without review from a trusted maintainer. The
//! CI workflows here run with privileged secrets beyond the default
//! `GITHUB_TOKEN` — a write-capable PAT (`ACTIONS_PUSH`) plus
//! `SEMGREP_APP_TOKEN` and `CODECOV_TOKEN` — so a careless or malicious
//! workflow edit is a direct path to secret exfiltration or unauthorised pushes
//! to protected branches. `CODEOWNERS` is the mechanism that pins owner review
//! on those paths and underpins the "Require review from Code Owners"
//! branch-protection rule.
//!
//! These tests assert on the real committed artefact — its location, that every
//! rule assigns an owner, and that the security-sensitive paths are covered —
//! not on source-code patterns.

use std::fs;
use std::path::{Path, PathBuf};

/// Locate the committed `CODEOWNERS` file. GitHub recognises it at the repo
/// root, in `.github/`, or in `docs/`; we accept any of those.
fn codeowners_path() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for candidate in ["CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"] {
        let path = manifest_dir.join(candidate);
        if path.is_file() {
            return path;
        }
    }
    panic!(
        "CODEOWNERS must exist at the repo root, in .github/, or in docs/ \
         (Issue #1486); none found under {}",
        manifest_dir.display()
    );
}

fn read_codeowners() -> String {
    let path = codeowners_path();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// A single owner rule: the pattern plus the owners it assigns.
struct Rule {
    pattern: String,
    owners: Vec<String>,
}

/// Parse the file into rules, skipping blank lines and `#` comments. The
/// CODEOWNERS grammar is `<pattern> <owner> [<owner> ...]`.
fn parse_rules(body: &str) -> Vec<Rule> {
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut parts = line.split_whitespace();
            let pattern = parts.next().unwrap().to_string();
            let owners = parts.map(str::to_string).collect();
            Rule { pattern, owners }
        })
        .collect()
}

#[test]
fn codeowners_exists_and_is_non_empty() {
    let contents = read_codeowners();
    assert!(
        !contents.trim().is_empty(),
        "CODEOWNERS must not be empty (Issue #1486)"
    );
}

#[test]
fn every_rule_assigns_at_least_one_owner() {
    let rules = parse_rules(&read_codeowners());
    assert!(
        !rules.is_empty(),
        "CODEOWNERS must declare at least one owner rule (Issue #1486)"
    );
    for rule in &rules {
        assert!(
            !rule.owners.is_empty(),
            "CODEOWNERS rule for `{}` must assign at least one owner \
             (Issue #1486)",
            rule.pattern
        );
        for owner in &rule.owners {
            assert!(
                owner.starts_with('@') || owner.contains('@'),
                "CODEOWNERS owner `{}` for pattern `{}` must be a @username, \
                 @org/team, or email (Issue #1486)",
                owner,
                rule.pattern
            );
        }
    }
}

/// Does any rule whose pattern matches `path_prefix` exist? We match on the
/// normalised pattern (leading `/` and trailing `/` stripped) so `/.github/`,
/// `.github/`, and `.github` all count as covering `.github`.
fn covers(rules: &[Rule], path_prefix: &str) -> bool {
    let want = path_prefix.trim_matches('/');
    rules.iter().any(|rule| {
        let pat = rule.pattern.trim_matches('/');
        pat == want || pat.starts_with(&format!("{want}/"))
    })
}

#[test]
fn covers_github_workflows() {
    let rules = parse_rules(&read_codeowners());
    assert!(
        covers(&rules, ".github") || covers(&rules, ".github/workflows"),
        "CODEOWNERS must cover the privileged CI paths under .github/ \
         (workflows hold ACTIONS_PUSH / SEMGREP_APP_TOKEN / CODECOV_TOKEN) \
         (Issue #1486)"
    );
}

#[test]
fn covers_dependency_manifests() {
    let rules = parse_rules(&read_codeowners());
    assert!(
        covers(&rules, "Cargo.toml"),
        "CODEOWNERS must cover Cargo.toml (supply-chain surface) (Issue #1486)"
    );
    assert!(
        covers(&rules, "Cargo.lock"),
        "CODEOWNERS must cover Cargo.lock (supply-chain surface) (Issue #1486)"
    );
}

#[test]
fn declares_a_default_fallback_owner() {
    let rules = parse_rules(&read_codeowners());
    assert!(
        rules.iter().any(|rule| rule.pattern == "*"),
        "CODEOWNERS should declare a default `*` fallback owner so no path is \
         left unowned (Issue #1486)"
    );
}
