//! Issue #1914 (SEC-72ff2657a2fd): the `CODEOWNERS` security block must cover
//! the scripts that *enforce* the supply-chain controls, not just the policy
//! files that *declare* them.
//!
//! `renovate.json` and `deny.toml` are owned, but `bump-deps.sh` — which
//! implements the `VIBE_BUMP_QUARANTINE_HOURS` gate (Issue #1234) — and
//! `quality.sh` — the mandated pre-commit gate that runs `cargo deny check`
//! (Issue #1865) — were not. Editing the enforcement script disables the
//! control just as effectively as editing the declaration, so an attacker
//! targets the script first. `scripts/runlib.sh` and `scripts/fuzz-ci.sh`
//! install toolchains from the network and belong in the same set.
//!
//! These tests assert on the real committed `CODEOWNERS` artefact — that each
//! enforcement path is matched and assigned the same owners as the policy
//! files it guards.

use std::fs;
use std::path::{Path, PathBuf};

/// The supply-chain enforcement paths that must carry owner review.
const ENFORCEMENT_PATHS: &[&str] = &[
    "bump-deps.sh",
    "quality.sh",
    "scripts/runlib.sh",
    "scripts/fuzz-ci.sh",
];

/// The policy file whose owners define the expected owner set for the block.
const REFERENCE_POLICY_PATH: &str = "renovate.json";

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
         (Issue #1914); none found under {}",
        manifest_dir.display()
    );
}

fn read_codeowners() -> String {
    let path = codeowners_path();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

struct Rule {
    pattern: String,
    owners: Vec<String>,
}

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

/// The owners of the last rule matching `path`, mirroring GitHub's
/// last-match-wins precedence. Only the rule shapes this repo actually uses
/// are considered: `*`, an exact path, and a directory prefix.
fn owners_for(rules: &[Rule], path: &str) -> Option<Vec<String>> {
    let want = path.trim_matches('/');
    rules
        .iter()
        .rfind(|rule| {
            let pat = rule.pattern.trim_matches('/');
            pat == "*" || pat == want || want.starts_with(&format!("{pat}/"))
        })
        .map(|rule| rule.owners.clone())
}

/// The owners assigned by a rule whose pattern is exactly `path` — i.e. the
/// path is named explicitly rather than merely swept up by the `*` fallback.
fn explicit_owners(rules: &[Rule], path: &str) -> Option<Vec<String>> {
    let want = path.trim_matches('/');
    rules
        .iter()
        .find(|rule| rule.pattern.trim_matches('/') == want)
        .map(|rule| rule.owners.clone())
}

#[test]
fn enforcement_scripts_are_named_explicitly() {
    let rules = parse_rules(&read_codeowners());
    for path in ENFORCEMENT_PATHS {
        assert!(
            explicit_owners(&rules, path).is_some(),
            "CODEOWNERS must name `{path}` explicitly in the security-sensitive \
             block — it enforces a supply-chain control, so the `*` fallback is \
             not enough (Issue #1914)"
        );
    }
}

#[test]
fn enforcement_scripts_share_the_policy_owner_set() {
    let rules = parse_rules(&read_codeowners());
    let expected = explicit_owners(&rules, REFERENCE_POLICY_PATH).unwrap_or_else(|| {
        panic!("CODEOWNERS must own `{REFERENCE_POLICY_PATH}` (Issue #1486/#1914)")
    });
    assert!(
        !expected.is_empty(),
        "the `{REFERENCE_POLICY_PATH}` rule must assign at least one owner (Issue #1914)"
    );
    for path in ENFORCEMENT_PATHS {
        let owners = owners_for(&rules, path)
            .unwrap_or_else(|| panic!("no CODEOWNERS rule matches `{path}` (Issue #1914)"));
        assert_eq!(
            owners, expected,
            "`{path}` must carry the same owners as `{REFERENCE_POLICY_PATH}`; \
             the enforcement script and the policy it enforces are equally \
             sensitive (Issue #1914)"
        );
    }
}

#[test]
fn enforcement_paths_exist_on_disk() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for path in ENFORCEMENT_PATHS {
        assert!(
            manifest_dir.join(path).is_file(),
            "CODEOWNERS names `{path}`, so the file must exist — a stale rule \
             silently enforces nothing (Issue #1914)"
        );
    }
}

#[test]
fn security_block_states_its_inclusion_criterion() {
    let body = read_codeowners().to_lowercase();
    assert!(
        body.contains("enforce or bypass a supply-chain control"),
        "the security-sensitive CODEOWNERS block must state its inclusion \
         criterion — \"files that enforce or bypass a supply-chain control\" — \
         so future additions are obvious (Issue #1914)"
    );
}
