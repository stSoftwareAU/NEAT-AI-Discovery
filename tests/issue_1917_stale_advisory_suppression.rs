//! Issue #1917: a vulnerability suppression must not outlive the dependency
//! it was written for.
//!
//! `GHSA-2f9f-gq7v-9h6m` (Apache Thrift, memory allocation with excessive
//! size) was suppressed twice — an `[advisories] ignore` entry in `deny.toml`
//! and an `allow-ghsas` line on the `dependency-review-action` step in
//! `.github/workflows/security.yml`. Both rested on one premise: `thrift` is a
//! transitive dependency via `parquet`. The parquet stack moved off it and
//! `thrift` left `Cargo.lock`, so both entries became dead config.
//!
//! Dead config here is not merely noise. If `thrift` ever re-enters the graph
//! — a `parquet` major bump, a new crate pulling it in — the advisory would be
//! *silently* re-suppressed on the strength of a risk assessment written for a
//! different dependency shape, with nobody re-reading it.
//!
//! These tests hold the two halves of the fix:
//!   1. the dead suppression is gone from both config surfaces, and
//!   2. the remaining suppressions are live — each names a crate that is still
//!      in the resolved graph — and `deny.toml` is configured to fail loud
//!      (`unused-ignored-advisory = "deny"`) the next time one goes stale.

use std::fs;
use std::path::{Path, PathBuf};

/// The advisory retired by this issue. Neither config surface may name it.
const RETIRED_ADVISORY: &str = "GHSA-2f9f-gq7v-9h6m";

/// Every advisory `deny.toml` still ignores, mapped to the crate whose
/// presence in the graph justifies the ignore. A new ignore must be added
/// here, which forces its author to name the crate carrying the risk — and
/// makes the suppression fail loudly once that crate leaves `Cargo.lock`.
const JUSTIFIED_IGNORES: &[(&str, &str)] = &[
    // Empty: `deny.toml` suppresses no advisory. RUSTSEC-2024-0436 (paste)
    // was retired when parquet 59.3.0 finished migrating to `pastey` and
    // `paste` left the graph (Issue #2054).
];

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read_repo_file(relative: &str) -> String {
    let path: PathBuf = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Lines of `deny.toml`'s `[advisories]` table, comments and blanks stripped.
fn advisories_table() -> Vec<String> {
    let body = read_repo_file("deny.toml");
    let mut lines = Vec::new();
    let mut inside = false;
    for raw in body.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            inside = line == "[advisories]";
            continue;
        }
        if !inside || line.is_empty() || line.starts_with('#') {
            continue;
        }
        lines.push(line.to_string());
    }
    assert!(
        !lines.is_empty(),
        "deny.toml must carry an [advisories] table (Issue #1917)"
    );
    lines
}

/// Advisory ids ignored by `deny.toml`, read from the `ignore` list entries.
fn ignored_advisory_ids() -> Vec<String> {
    advisories_table()
        .iter()
        .filter_map(|line| {
            let rest = line.split_once("id = \"")?.1;
            let id = rest.split_once('"')?.0;
            Some(id.to_string())
        })
        .collect()
}

/// Crate names in the resolved `Cargo.lock` graph.
fn locked_crate_names() -> Vec<String> {
    read_repo_file("Cargo.lock")
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("name = \"")?;
            Some(rest.trim_end_matches('"').to_string())
        })
        .collect()
}

// ── The dead suppression is gone from both surfaces ───────────────────

#[test]
fn deny_toml_does_not_suppress_the_retired_thrift_advisory() {
    let body = read_repo_file("deny.toml");
    assert!(
        !body.contains(RETIRED_ADVISORY),
        "deny.toml must not mention {RETIRED_ADVISORY} — thrift left the \
         dependency graph, so the ignore is dead config that would silently \
         re-suppress the advisory if thrift ever returned (Issue #1917)"
    );
}

#[test]
fn security_workflow_does_not_allow_the_retired_thrift_advisory() {
    let body = read_repo_file(".github/workflows/security.yml");
    assert!(
        !body.contains(RETIRED_ADVISORY),
        "the dependency-review step in .github/workflows/security.yml must \
         not allow {RETIRED_ADVISORY} — it is the twin of the dead deny.toml \
         ignore and carries the identical silent-re-suppression hazard \
         (Issue #1917)"
    );
}

// ── A future stale suppression must fail loud ─────────────────────────

#[test]
fn deny_toml_fails_loud_on_an_unused_advisory_ignore() {
    let setting = advisories_table()
        .into_iter()
        .find(|line| line.starts_with("unused-ignored-advisory"))
        .unwrap_or_else(|| {
            panic!(
                "deny.toml [advisories] must set `unused-ignored-advisory` so \
                 cargo-deny surfaces a suppression that no longer matches any \
                 crate, instead of an audit finding it years later (Issue #1917)"
            )
        });
    assert!(
        setting.contains("\"deny\"") || setting.contains("\"warn\""),
        "`unused-ignored-advisory` must be \"deny\" (preferred) or \"warn\" — \
         anything else lets a stale suppression pass silently (Issue \
         #1917): {setting}"
    );
}

// ── The surviving suppressions are still live ─────────────────────────

#[test]
fn every_advisory_ignore_names_a_crate_still_in_the_graph() {
    let locked = locked_crate_names();
    for id in ignored_advisory_ids() {
        let (_, crate_name) = JUSTIFIED_IGNORES
            .iter()
            .find(|(advisory, _)| *advisory == id)
            .unwrap_or_else(|| {
                panic!(
                    "advisory ignore {id} in deny.toml has no entry in \
                     JUSTIFIED_IGNORES — record the crate whose presence in \
                     the graph justifies the suppression so it cannot outlive \
                     that crate (Issue #1917)"
                )
            });
        assert!(
            locked.iter().any(|name| name == crate_name),
            "deny.toml ignores {id} because of the `{crate_name}` crate, but \
             `{crate_name}` is no longer in Cargo.lock — remove the ignore \
             rather than leaving it to re-suppress the advisory silently \
             (Issue #1917)"
        );
    }
}

/// Replaces `the_paste_suppression_is_still_required`, whose own message said
/// to "drop this test with the ignore, not before". `paste` left `Cargo.lock`
/// when parquet 59.3.0 completed the `pastey` migration, so the ignore was
/// removed and the guard is inverted: neither the crate nor its suppression
/// may quietly return (Issue #2054).
#[test]
fn the_retired_paste_suppression_does_not_come_back() {
    if locked_crate_names().iter().any(|name| name == "paste") {
        return;
    }
    assert!(
        !ignored_advisory_ids()
            .iter()
            .any(|id| id == "RUSTSEC-2024-0436"),
        "`paste` is not in Cargo.lock, so ignoring RUSTSEC-2024-0436 matches no \
         crate — `unused-ignored-advisory = \"deny\"` fails every `cargo deny \
         check`, which disables ./bump-deps.sh for the whole repo (Issue #2054)"
    );
}
