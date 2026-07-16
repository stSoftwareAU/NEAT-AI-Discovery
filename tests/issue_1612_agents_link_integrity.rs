//! Issue #1612 — `AGENTS.md` linked to two `README.md` section anchors that do
//! not exist (check 8, broken internal links). One of them
//! (`README.md#development-guidelines`) also promised a "full list of allowed
//! licences" the README does not contain; the licence allow-list is actually
//! enforced by `cargo deny` via `deny.toml`.
//!
//! These tests enforce cross-reference integrity: every `README.md#anchor`
//! link in `AGENTS.md` must resolve to a real README heading, and the licence
//! guidance must point at the authoritative source (`deny.toml`).

const README: &str = include_str!("../README.md");
const AGENTS: &str = include_str!("../AGENTS.md");

/// Slugify a Markdown heading the way GitHub does, closely enough to resolve
/// the anchors used in this repo: lowercase, drop everything that is not an
/// ASCII letter, digit, space or hyphen (emoji, variation selectors and
/// punctuation), then turn runs of whitespace into single hyphens.
fn slug(heading: &str) -> String {
    let lowered = heading.trim().to_lowercase();
    let cleaned: String = lowered
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else if c.is_whitespace() {
                ' '
            } else {
                '\0' // marker for "drop"
            }
        })
        .filter(|&c| c != '\0')
        .collect();
    let mut out = String::new();
    let mut prev_dash = false;
    for c in cleaned.trim().chars() {
        if c == ' ' {
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        } else {
            out.push(c);
            prev_dash = false;
        }
    }
    out
}

/// Every anchor slug produced by a `#`-prefixed heading in the README.
fn readme_anchor_set() -> Vec<String> {
    README
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                let text = trimmed.trim_start_matches('#').trim();
                Some(slug(text))
            } else {
                None
            }
        })
        .collect()
}

/// Extract the anchor portion of every `README.md#anchor` link in `AGENTS.md`.
fn agents_readme_anchor_links() -> Vec<String> {
    const NEEDLE: &str = "README.md#";
    let mut anchors = Vec::new();
    let bytes = AGENTS.as_bytes();
    let mut i = 0;
    while let Some(rel) = AGENTS[i..].find(NEEDLE) {
        let start = i + rel + NEEDLE.len();
        let mut end = start;
        while end < bytes.len() {
            let c = bytes[end] as char;
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                end += 1;
            } else {
                break;
            }
        }
        anchors.push(AGENTS[start..end].to_string());
        i = end.max(start + 1);
    }
    anchors
}

#[test]
fn sanity_check_known_readme_anchors_resolve() {
    let anchors = readme_anchor_set();
    for expected in [
        "development",
        "gpu-requirement",
        "troubleshooting",
        "additional-documentation",
    ] {
        assert!(
            anchors.iter().any(|a| a == expected),
            "README.md is expected to expose the `#{expected}` anchor; got {anchors:?}"
        );
    }
}

#[test]
fn every_agents_readme_anchor_link_resolves() {
    let anchors = readme_anchor_set();
    for link in agents_readme_anchor_links() {
        assert!(
            anchors.contains(&link),
            "AGENTS.md links to README.md#{link}, but no README heading produces that anchor \
             (Issue #1612 broken cross-reference). Available anchors: {anchors:?}"
        );
    }
}

#[test]
fn agents_does_not_link_the_non_existent_development_guidelines_anchor() {
    assert!(
        !AGENTS.contains("README.md#development-guidelines"),
        "AGENTS.md must not link to the non-existent README.md#development-guidelines anchor \
         (Issue #1612)"
    );
}

#[test]
fn agents_points_licence_allow_list_at_deny_toml() {
    // The licence allow-list is enforced by `cargo deny` via `deny.toml`, not by
    // any README section. The licence guidance must name the authoritative source.
    assert!(
        AGENTS.contains("deny.toml"),
        "AGENTS.md must point the dependency-licence guidance at deny.toml (Issue #1612)"
    );
}
