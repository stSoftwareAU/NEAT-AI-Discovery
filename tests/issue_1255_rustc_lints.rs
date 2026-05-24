//! Issue #1255: `Cargo.toml` must declare a `[lints.rust]` section that
//! denies `unsafe_op_in_unsafe_fn`, making the project-wide convention
//! ("every `unsafe` op inside an `unsafe fn` is wrapped in its own
//! `unsafe { ... }` block with a `// SAFETY:` comment") enforceable at
//! the crate root.
//!
//! The test reads `Cargo.toml` as plain text so the library does not
//! need to pull in a new TOML parser dependency just for this check.

use std::fs;
use std::path::Path;

#[test]
fn cargo_toml_denies_unsafe_op_in_unsafe_fn() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let contents = fs::read_to_string(&manifest_path).expect("read Cargo.toml");

    let section = extract_section(&contents, "[lints.rust]").expect(
        "Cargo.toml must contain a [lints.rust] section that configures \
         rustc lints at the crate root (Issue #1255)",
    );

    let value = field_value(section, "unsafe_op_in_unsafe_fn").expect(
        "Cargo.toml [lints.rust] must configure `unsafe_op_in_unsafe_fn` \
         (Issue #1255)",
    );

    assert_eq!(
        value, "deny",
        "Cargo.toml [lints.rust].unsafe_op_in_unsafe_fn must be \"deny\" so \
         CI fails fast when a bare `unsafe` op inside an `unsafe fn` is \
         introduced (Issue #1255), got: {value:?}"
    );
}

/// Return the body of the named section (everything up to the next
/// `[section]` header).
fn extract_section<'a>(contents: &'a str, header: &str) -> Option<&'a str> {
    let start = contents.find(header)?;
    let after_header = &contents[start + header.len()..];
    let mut end = after_header.len();
    for (idx, line) in after_header.lines().enumerate() {
        if idx == 0 {
            continue;
        }
        if line.trim_start().starts_with('[') {
            let mut offset = 0usize;
            for (i, l) in after_header.lines().enumerate() {
                if i == idx {
                    break;
                }
                offset += l.len() + 1;
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
        let value = value.split('#').next().unwrap_or("").trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))?;
        return Some(value.to_string());
    }
    None
}
