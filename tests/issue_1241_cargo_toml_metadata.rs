//! Issue #1241: `Cargo.toml [package]` must declare `description`,
//! `repository`, and `readme` so the crate is discoverable, the source
//! of truth is unambiguous, and `cargo publish`/`cargo metadata`
//! consumers (tooling, SBOM, doc generators) have the expected metadata
//! per the Rust API Guidelines (C-METADATA).
//!
//! This test reads `Cargo.toml` directly (without parsing TOML) so the
//! library does not need to pull in a new dependency just for the test.

use std::fs;
use std::path::Path;

#[test]
fn cargo_toml_declares_required_package_metadata() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let contents = fs::read_to_string(&manifest_path).expect("read Cargo.toml");

    let package_section = extract_section(&contents, "[package]")
        .expect("Cargo.toml must contain a [package] section");

    let description = field_value(package_section, "description")
        .expect("Cargo.toml [package] must declare `description` (Issue #1241)");
    assert!(
        description.len() >= 20,
        "Cargo.toml [package].description should be a meaningful sentence, got: {description:?}"
    );

    let repository = field_value(package_section, "repository")
        .expect("Cargo.toml [package] must declare `repository` (Issue #1241)");
    assert!(
        repository.starts_with("https://github.com/stSoftwareAU/NEAT-AI-Discovery"),
        "Cargo.toml [package].repository must point at the canonical GitHub repo, got: \
         {repository:?}"
    );

    let readme = field_value(package_section, "readme")
        .expect("Cargo.toml [package] must declare `readme` (Issue #1241)");
    assert_eq!(
        readme, "README.md",
        "Cargo.toml [package].readme should be README.md"
    );

    // The referenced README must exist.
    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(&readme);
    assert!(
        readme_path.is_file(),
        "Cargo.toml [package].readme = {readme:?} but {readme_path:?} does not exist"
    );
}

/// Return the body of the named section (everything up to the next
/// `[section]` header).
fn extract_section<'a>(contents: &'a str, header: &str) -> Option<&'a str> {
    let start = contents.find(header)?;
    let after_header = &contents[start + header.len()..];
    // Find the next section header at the start of a line.
    let mut end = after_header.len();
    for (idx, line) in after_header.lines().enumerate() {
        if idx == 0 {
            continue;
        }
        if line.trim_start().starts_with('[') {
            // Recompute byte offset of this line within after_header.
            let mut offset = 0usize;
            for (i, l) in after_header.lines().enumerate() {
                if i == idx {
                    break;
                }
                offset += l.len() + 1; // +1 for the newline
            }
            end = offset;
            break;
        }
    }
    Some(&after_header[..end])
}

/// Find a `key = "value"` declaration inside a TOML section body.
fn field_value(section: &str, key: &str) -> Option<String> {
    for line in section.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix(key) else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let value = rest.trim();
        // Strip trailing comment.
        let value = value.split('#').next().unwrap_or("").trim();
        // Strip surrounding quotes.
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))?;
        return Some(value.to_string());
    }
    None
}
