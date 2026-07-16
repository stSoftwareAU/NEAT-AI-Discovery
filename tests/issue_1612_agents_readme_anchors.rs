//! Issue #1612 — `AGENTS.md` linked to a `README.md` section anchor that does
//! not exist (`README.md#development-guidelines`), promising a "full list of
//! allowed licences" the README does not contain. The licence allow-list is
//! actually enforced by `cargo deny` via `deny.toml`.
//!
//! These tests guard link integrity: every `README.md#anchor` cross-reference
//! in `AGENTS.md` must resolve to a real heading anchor in `README.md`, the
//! broken `#development-guidelines` anchor must be gone, and the licence
//! reference must point at the real source of truth (`deny.toml`).

const README: &str = include_str!("../README.md");
const AGENTS: &str = include_str!("../AGENTS.md");
const DENY: &str = include_str!("../deny.toml");

/// Derive the GitHub heading anchor slug for a heading's text.
///
/// Mirrors GitHub's slugger closely enough for this repo's headings: lowercase,
/// drop emoji/punctuation, map spaces and hyphens to `-`, collapse repeats, and
/// trim leading/trailing `-`. E.g. `💻 Development` → `development`,
/// `🖥️ GPU Requirement` → `gpu-requirement`.
fn heading_anchor(text: &str) -> String {
    let mut slug = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if c == ' ' || c == '-' || c == '_' {
            slug.push('-');
        }
        // Everything else (emoji, punctuation, variation selectors) is dropped.
    }
    // Collapse consecutive hyphens and trim the ends.
    let mut collapsed = String::new();
    let mut prev_hyphen = false;
    for c in slug.chars() {
        if c == '-' {
            if !prev_hyphen {
                collapsed.push(c);
            }
            prev_hyphen = true;
        } else {
            collapsed.push(c);
            prev_hyphen = false;
        }
    }
    collapsed.trim_matches('-').to_string()
}

/// Every anchor resolvable from a `## `/`### ` heading in `README.md`.
fn readme_anchors() -> Vec<String> {
    README
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let hashes = trimmed.chars().take_while(|&c| c == '#').count();
            if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
                Some(heading_anchor(trimmed[hashes..].trim()))
            } else {
                None
            }
        })
        .filter(|a| !a.is_empty())
        .collect()
}

/// Every `README.md#anchor` link target referenced from `AGENTS.md`.
fn agents_readme_anchor_links() -> Vec<String> {
    const MARKER: &str = "README.md#";
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = AGENTS[i..].find(MARKER) {
        let start = i + rel + MARKER.len();
        let anchor: String = AGENTS[start..]
            .chars()
            .take_while(|&c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            .collect();
        if !anchor.is_empty() {
            out.push(anchor);
        }
        i = start.max(i + rel + 1);
    }
    out
}

#[test]
fn every_agents_readme_anchor_resolves() {
    let anchors = readme_anchors();
    for link in agents_readme_anchor_links() {
        assert!(
            anchors.contains(&link),
            "AGENTS.md links to README.md#{link}, but no such heading anchor exists in README.md \
             (Issue #1612 broken cross-reference). Known anchors: {anchors:?}"
        );
    }
}

#[test]
fn broken_development_guidelines_anchor_is_gone() {
    assert!(
        !AGENTS.contains("README.md#development-guidelines"),
        "AGENTS.md must not link to the non-existent README.md#development-guidelines anchor \
         (Issue #1612)"
    );
}

#[test]
fn licence_reference_points_at_deny_toml() {
    // The licence allow-list is the `[licenses]` table in deny.toml, enforced by
    // `cargo deny`. AGENTS.md must point contributors there, not at a README
    // section that never carried the list.
    assert!(
        AGENTS.contains("deny.toml"),
        "AGENTS.md must reference deny.toml as the source of the allowed-licence list (Issue #1612)"
    );
    assert!(
        DENY.contains("[licenses]"),
        "deny.toml must contain the [licenses] allow-list that AGENTS.md points at"
    );
}
