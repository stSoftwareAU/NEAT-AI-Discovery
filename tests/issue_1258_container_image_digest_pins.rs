//! Issue #1258: Every GitHub Actions `container.image` reference across this
//! repository's workflows must be pinned to an immutable `@sha256:<digest>`,
//! not a mutable tag (e.g. `:latest`, `:1.163.0`, or no tag at all).
//!
//! The supply-chain risk is identical to Issue #1216 for `uses:` lines —
//! a registry-account compromise or malicious release lets an attacker
//! re-point a mutable tag and execute arbitrary code in CI with access
//! to repository secrets (`SEMGREP_APP_TOKEN`, `GITHUB_TOKEN`, etc.) on
//! the next PR run.
//!
//! Renovate's `github-actions` manager does not track `jobs.<job>.container.image`,
//! so the 24h `minimumReleaseAge` quarantine configured in `renovate.json`
//! does not protect this surface. Pinning to a content digest closes the
//! gap; `bump-deps.sh` rotates the digest under the same 24h window.
//!
//! This test reads every `*.yml` file under `.github/workflows/` and
//! asserts that any `image:` value sitting inside a `container:` block
//! includes an `@sha256:<64-hex-chars>` suffix.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn all_workflow_container_images_are_pinned_to_sha256_digest() {
    let workflows_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    assert!(
        workflows_dir.is_dir(),
        "expected .github/workflows directory at {workflows_dir:?}"
    );

    let mut yml_files: Vec<PathBuf> = fs::read_dir(&workflows_dir)
        .expect("read .github/workflows")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("yml"))
        .collect();
    yml_files.sort();
    assert!(!yml_files.is_empty(), "no workflow files found");

    let mut violations: Vec<String> = Vec::new();
    let mut images_checked = 0usize;

    for path in &yml_files {
        let contents = fs::read_to_string(path).expect("read workflow file");
        for (image_line_idx, image_value) in find_container_images(&contents) {
            images_checked += 1;
            let lineno = image_line_idx + 1;
            if !has_sha256_digest(&image_value) {
                violations.push(format!(
                    "{}:{}: container image `{}` is not pinned to an immutable \
                     `@sha256:<digest>` (Issue #1258)",
                    path.display(),
                    lineno,
                    image_value
                ));
            }
        }
    }

    assert!(
        images_checked > 0,
        "no container.image references found — test fixture missing?"
    );
    assert!(
        violations.is_empty(),
        "GitHub Actions container images must be pinned to immutable sha256 \
         digests (Issue #1258):\n{}",
        violations.join("\n")
    );
}

/// Find every `image:` line that sits inside a `container:` mapping.
/// Returns the zero-based line index and the bare image value with any
/// surrounding whitespace, quoting, and trailing comment removed.
fn find_container_images(contents: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    let lines: Vec<&str> = contents.lines().collect();

    for (idx, raw_line) in lines.iter().enumerate() {
        let trimmed = raw_line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        let container_line = trimmed.trim_start_matches("- ").trim_start();
        if container_line != "container:" {
            continue;
        }
        let container_indent = leading_spaces(raw_line);

        // Walk forward through deeper-indented lines and pick up the
        // first `image:` key inside this container block.
        for (j, follow) in lines.iter().enumerate().skip(idx + 1) {
            let f_trimmed = follow.trim_start();
            if f_trimmed.is_empty() || f_trimmed.starts_with('#') {
                continue;
            }
            if leading_spaces(follow) <= container_indent {
                break;
            }
            if let Some(value) = f_trimmed.strip_prefix("image:") {
                hits.push((j, strip_yaml_value(value)));
                break;
            }
        }
    }

    hits
}

fn leading_spaces(line: &str) -> usize {
    line.bytes().take_while(|b| *b == b' ').count()
}

fn strip_yaml_value(raw: &str) -> String {
    let mut v = raw.trim().to_string();
    // Drop trailing comments.
    if let Some(hash) = v.find('#') {
        v.truncate(hash);
        v = v.trim().to_string();
    }
    v.trim_matches('"').trim_matches('\'').to_string()
}

fn has_sha256_digest(image: &str) -> bool {
    let Some((_repo, digest)) = image.rsplit_once("@sha256:") else {
        return false;
    };
    digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
}

// --- Inline unit tests for the parser helpers (no external workflow needed) ---

#[test]
fn has_sha256_digest_accepts_full_64_hex_suffix() {
    let img = "semgrep/semgrep@sha256:\
        7cad2bc2d1e44f87f0bf4be6d1fa23aa90fb72015bebc89fb91385d813987a03";
    assert!(has_sha256_digest(img));
}

#[test]
fn has_sha256_digest_rejects_tag_only() {
    assert!(!has_sha256_digest("semgrep/semgrep"));
    assert!(!has_sha256_digest("semgrep/semgrep:latest"));
    assert!(!has_sha256_digest("semgrep/semgrep:1.163.0"));
}

#[test]
fn has_sha256_digest_rejects_truncated_digest() {
    // 32 hex chars — too short.
    assert!(!has_sha256_digest(
        "foo/bar@sha256:7cad2bc2d1e44f87f0bf4be6d1fa23aa"
    ));
}

#[test]
fn find_container_images_picks_up_block_form() {
    let yaml = "\
jobs:
  semgrep:
    container:
      image: semgrep/semgrep@sha256:abc
    steps:
      - run: echo hi
";
    let hits = find_container_images(yaml);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].1, "semgrep/semgrep@sha256:abc");
}

#[test]
fn find_container_images_strips_trailing_comment() {
    let yaml = "\
jobs:
  j:
    container:
      image: foo/bar@sha256:def  # pinned per Issue #1258
";
    let hits = find_container_images(yaml);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].1, "foo/bar@sha256:def");
}
