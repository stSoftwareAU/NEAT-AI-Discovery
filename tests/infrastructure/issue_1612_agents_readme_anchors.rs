//! Issue #1612 — AGENTS.md cross-references to README.md anchors must resolve.
//!
//! `AGENTS.md` declares itself the single source of truth for AI coding agents,
//! so its internal links have to point at anchors README.md actually exposes.
//! Two links were broken: a licence link to a non-existent
//! `README.md#development-guidelines` section (the licence allow-list really
//! lives in `deny.toml`), and a GPU link to `README.md#gpu-performance-tuning`.
//! These tests parse both files and assert every `README.md#anchor` reference in
//! AGENTS.md resolves to a heading GitHub would generate from README.md.

use std::collections::HashSet;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_doc(name: &str) -> String {
    let path = repo_root().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Convert a Markdown heading's text into the anchor slug GitHub generates:
/// lowercase, drop anything that is not ASCII alphanumeric/space/hyphen (this
/// removes emoji and punctuation), trim, then map runs of spaces to a hyphen.
fn slug(heading_text: &str) -> String {
    let mut out = String::new();
    for ch in heading_text.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() || ch == ' ' || ch == '-' {
            out.push(ch);
        }
    }
    let mut slug = String::new();
    let mut prev_hyphen = false;
    for ch in out.trim().chars() {
        if ch == ' ' || ch == '-' {
            if !prev_hyphen {
                slug.push('-');
                prev_hyphen = true;
            }
        } else {
            slug.push(ch);
            prev_hyphen = false;
        }
    }
    slug.trim_matches('-').to_string()
}

/// All heading anchors README.md exposes.
fn readme_anchors() -> HashSet<String> {
    let mut anchors = HashSet::new();
    let mut in_fence = false;
    for line in read_doc("README.md").lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            let text = rest.trim_start_matches('#').trim();
            if !text.is_empty() {
                anchors.insert(slug(text));
            }
        }
    }
    anchors
}

/// Every `README.md#anchor` reference found in AGENTS.md.
fn agents_readme_anchor_refs() -> Vec<String> {
    let agents = read_doc("AGENTS.md");
    let needle = "README.md#";
    let mut refs = Vec::new();
    let mut idx = 0;
    while let Some(pos) = agents[idx..].find(needle) {
        let start = idx + pos + needle.len();
        let anchor: String = agents[start..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if !anchor.is_empty() {
            refs.push(anchor);
        }
        idx = start;
    }
    refs
}

#[test]
fn readme_slug_matches_known_github_anchors() {
    // Guard the slugger against the exact headings the issue verified.
    assert_eq!(slug("💻 Development"), "development");
    assert_eq!(slug("🖥️ GPU Requirement"), "gpu-requirement");
    assert_eq!(slug("🛠️ Troubleshooting"), "troubleshooting");
    assert_eq!(
        slug("📚 Additional Documentation"),
        "additional-documentation"
    );
}

#[test]
fn readme_exposes_expected_anchors() {
    let anchors = readme_anchors();
    for expected in [
        "development",
        "gpu-requirement",
        "troubleshooting",
        "additional-documentation",
    ] {
        assert!(
            anchors.contains(expected),
            "README.md should expose #{expected}; found anchors: {anchors:?}"
        );
    }
}

#[test]
fn all_agents_readme_links_resolve() {
    let anchors = readme_anchors();
    let refs = agents_readme_anchor_refs();
    assert!(
        !refs.is_empty(),
        "expected AGENTS.md to contain README.md#... links"
    );
    for anchor in &refs {
        assert!(
            anchors.contains(anchor),
            "AGENTS.md links to README.md#{anchor} but README.md has no such heading anchor. \
             Available anchors: {anchors:?}"
        );
    }
}

#[test]
fn agents_md_has_no_development_guidelines_anchor() {
    // The removed link pointed at a section README.md never had.
    let refs = agents_readme_anchor_refs();
    assert!(
        !refs.iter().any(|a| a == "development-guidelines"),
        "AGENTS.md still references the non-existent README.md#development-guidelines anchor"
    );
}
