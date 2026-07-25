//! Issue #1685 — stale pre-refactor paths and broken internal links across docs.
//!
//! A batch of internal links and file references still pointed at pre-refactor
//! flat paths (`src/analysis/saturation.rs`, `src/focus.rs`, …) after the source
//! tree was reorganised under `src/analysis/detection/`,
//! `src/analysis/recommendation/`, `src/analysis/neuron/`, `src/analysis/synapse/`
//! and `src/focus/`. These tests tie the docs back to the on-disk tree and to
//! `Cargo.toml` so the references cannot silently rot again:
//!
//!   1. Every `**Source:**` link in `docs/discoveries/*.md` resolves on disk.
//!   2. The `weight-polarity-flip` test link resolves.
//!   3. Every `**Source**:` path in `docs/DISCOVERY_TYPES.md` resolves.
//!   4. The `IMPACT_CALCULATION` "Related Code" paths resolve (no flat files).
//!   5. The `Squash + Weight Rescale` anchor uses the double-hyphen slug.
//!   6. The `docs/discoveries/README.md` scenario index lists every scenario file.
//!   7. `docs/BENCHMARKS.md` documents exactly the `[[bench]]` targets in
//!      `Cargo.toml`, with a matching suite count.
//!   8. `docs/ci-doc-build-step.md` stays consistent with reality (pointer,
//!      flags, and pending status).

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// Resolve a relative link found inside the doc at repo-relative `doc_rel` and
/// return the on-disk path it points at (anchors stripped).
fn resolve(doc_rel: &str, link: &str) -> PathBuf {
    let mut p = repo_root().join(doc_rel);
    p.pop(); // drop the file name, keep the doc's directory
    let link = link.split('#').next().unwrap();
    for seg in link.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                p.pop();
            }
            s => p.push(s),
        }
    }
    p
}

/// Extract every markdown link target `](target)` from a line.
fn link_targets(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(i) = rest.find("](") {
        let after = &rest[i + 2..];
        if let Some(j) = after.find(')') {
            out.push(after[..j].to_string());
            rest = &after[j + 1..];
        } else {
            break;
        }
    }
    out
}

// ============================================================================
// 1. docs/discoveries/*.md — every **Source:** link resolves.
// ============================================================================

#[test]
fn discovery_source_links_resolve() {
    let dir = repo_root().join("docs/discoveries");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("read docs/discoveries") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let doc_rel = format!("docs/discoveries/{name}");
        let content = read(&doc_rel);
        for line in content.lines().filter(|l| l.contains("**Source:**")) {
            for target in link_targets(line) {
                if !target.starts_with("..") {
                    continue; // skip README.md / external links
                }
                let resolved = resolve(&doc_rel, &target);
                assert!(
                    resolved.exists(),
                    "{name}: Source link `{target}` does not resolve ({}) (Issue #1685)",
                    resolved.display()
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 13,
        "expected to check the discovery source links, saw {checked}"
    );
}

// ============================================================================
// 2. weight-polarity-flip test link resolves.
// ============================================================================

#[test]
fn weight_polarity_flip_test_link_resolves() {
    let doc = "docs/discoveries/weight-polarity-flip.md";
    let content = read(doc);
    let line = content
        .lines()
        .find(|l| l.contains("issue_644_weight_polarity_flip"))
        .expect("weight-polarity-flip.md links its test file");
    let target = link_targets(line)
        .into_iter()
        .find(|t| t.contains("issue_644_weight_polarity_flip"))
        .expect("test link present");
    assert!(
        resolve(doc, &target).exists(),
        "weight-polarity-flip test link `{target}` must resolve (Issue #1685)"
    );
    // The old flat path must be gone.
    assert!(
        !content.contains("../../tests/issue_644_weight_polarity_flip.rs"),
        "stale flat tests/ link must be repointed to tests/detection/ (Issue #1685)"
    );
}

// ============================================================================
// 3. DISCOVERY_TYPES.md — every **Source**: backticked path resolves.
// ============================================================================

#[test]
fn discovery_types_source_paths_resolve() {
    let content = read("docs/DISCOVERY_TYPES.md");
    let mut checked = 0;
    for line in content.lines().filter(|l| l.contains("**Source**:")) {
        // Path is the first backtick-quoted span on the line.
        let start = line.find('`').expect("source path in backticks") + 1;
        let end = start + line[start..].find('`').expect("closing backtick");
        let rel = &line[start..end];
        assert!(
            repo_root().join(rel).exists(),
            "DISCOVERY_TYPES **Source** path `{rel}` does not resolve (Issue #1685)"
        );
        checked += 1;
    }
    assert!(checked >= 40, "expected many Source lines, saw {checked}");
    // The specific stale flat files must be gone.
    for stale in [
        "`src/analysis/saturation.rs`",
        "`src/analysis/neuron.rs`",
        "`src/analysis/synapse.rs`",
        "`src/analysis/implementation.rs`",
        "`src/focus.rs`",
    ] {
        assert!(
            !content.contains(stale),
            "DISCOVERY_TYPES must not cite the stale flat path {stale} (Issue #1685)"
        );
    }
}

// ============================================================================
// 4. IMPACT_CALCULATION "Related Code" paths resolve; flat files gone.
// ============================================================================

#[test]
fn impact_calculation_related_code_resolves() {
    let content = read("docs/IMPACT_CALCULATION.md");
    for good in [
        "src/focus/impact.rs",
        "src/focus/ranking/mod.rs",
        "tests/focus/",
    ] {
        assert!(
            content.contains(good),
            "IMPACT_CALCULATION Related Code must cite `{good}` (Issue #1685)"
        );
        assert!(
            repo_root().join(good).exists(),
            "cited path `{good}` must exist on disk (Issue #1685)"
        );
    }
    for stale in ["`src/focus.rs`", "`src/analysis.rs`", "`tests/focus.rs`"] {
        assert!(
            !content.contains(stale),
            "IMPACT_CALCULATION must not cite the stale directory-as-file path {stale} (Issue #1685)"
        );
    }
}

// ============================================================================
// 5. Squash + Weight Rescale anchor uses the double-hyphen slug.
// ============================================================================

#[test]
fn squash_weight_rescale_anchor_uses_double_hyphen() {
    // GitHub slugs "Squash + Weight Rescale Detection" -> squash--weight-rescale-detection.
    let heading = "### Squash + Weight Rescale Detection";
    let dt = read("docs/DISCOVERY_TYPES.md");
    assert!(dt.contains(heading), "heading must exist to slug against");
    assert!(
        dt.contains("(#squash--weight-rescale-detection)"),
        "DISCOVERY_TYPES anchors must use the double-hyphen slug (Issue #1685)"
    );
    assert!(
        !dt.contains("(#squash-weight-rescale-detection)"),
        "DISCOVERY_TYPES must not keep the single-hyphen (non-resolving) anchor (Issue #1685)"
    );
}

// ============================================================================
// 6. discoveries/README index lists every scenario file.
// ============================================================================

#[test]
fn discoveries_readme_index_lists_every_scenario() {
    let index = read("docs/discoveries/README.md");
    let dir = repo_root().join("docs/discoveries");
    for entry in std::fs::read_dir(&dir).expect("read docs/discoveries") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        if name == "README.md" {
            continue;
        }
        assert!(
            index.contains(&format!("({name})")),
            "scenario index is missing a row linking `{name}` (Issue #1685)"
        );
    }
}

// ============================================================================
// 7. BENCHMARKS.md documents exactly the Cargo.toml [[bench]] targets.
// ============================================================================

fn cargo_bench_names() -> Vec<String> {
    let cargo = read("Cargo.toml");
    let mut names = Vec::new();
    let mut in_bench = false;
    for line in cargo.lines() {
        let t = line.trim();
        if t == "[[bench]]" {
            in_bench = true;
            continue;
        }
        if t.starts_with('[') && t != "[[bench]]" {
            in_bench = false;
        }
        if in_bench && t.starts_with("name") {
            if let Some(v) = t.split('"').nth(1) {
                names.push(v.to_string());
            }
            in_bench = false;
        }
    }
    names
}

#[test]
fn benchmarks_doc_matches_cargo_bench_targets() {
    let benches = cargo_bench_names();
    assert!(
        benches.len() >= 40,
        "expected many [[bench]] targets, saw {}",
        benches.len()
    );
    let doc = read("docs/BENCHMARKS.md");
    for name in &benches {
        assert!(
            doc.contains(&format!("`{name}`")),
            "BENCHMARKS.md is missing suite `{name}` registered in Cargo.toml (Issue #1685)"
        );
    }
    // The documented count must match the registered target count.
    assert!(
        doc.contains(&format!("{} Criterion benchmark suites", benches.len())),
        "BENCHMARKS.md count must equal the {} registered [[bench]] targets (Issue #1685)",
        benches.len()
    );
}

// ============================================================================
// 8. ci-doc-build-step.md stays consistent with reality.
// ============================================================================

#[test]
fn ci_doc_build_step_pointers_are_current() {
    let doc = read("docs/ci-doc-build-step.md");
    // Stale pointer must be gone; corrected one present.
    assert!(
        !doc.contains("`quality.sh` (line 41)"),
        "ci-doc-build-step must not cite the stale quality.sh line 41 (Issue #1685)"
    );
    assert!(
        doc.contains("quality.sh:75-76"),
        "ci-doc-build-step must cite the real quality.sh doc-build location (Issue #1685)"
    );
    // Verify the pointer against quality.sh itself.
    let quality = read("quality.sh");
    let doc_line = quality
        .lines()
        .nth(75) // line 76, zero-indexed
        .expect("quality.sh has a line 76");
    assert!(
        doc_line.contains("cargo doc"),
        "quality.sh:76 should be the `cargo doc` build step; pointer would be stale otherwise"
    );

    // The proposal is only marked "pending" while no workflow builds the docs.
    let wf_dir = repo_root().join(".github/workflows");
    let mut has_cargo_doc = false;
    if let Ok(rd) = std::fs::read_dir(&wf_dir) {
        for e in rd.flatten() {
            let is_yml = e.path().extension().and_then(|x| x.to_str()) == Some("yml");
            if is_yml && std::fs::read_to_string(e.path()).is_ok_and(|c| c.contains("cargo doc")) {
                has_cargo_doc = true;
            }
        }
    }
    if !has_cargo_doc {
        assert!(
            doc.to_lowercase().contains("pending")
                || doc.to_lowercase().contains("not yet implemented"),
            "no workflow builds docs, so ci-doc-build-step must be marked pending (Issue #1685)"
        );
    }
}
