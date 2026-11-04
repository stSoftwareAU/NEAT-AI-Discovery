//! Issue #2494: `tests/test_gitleaks_workflow_verification.sh` was never run by
//! any gate (no CI job, `quality.sh` step or `cargo test` target invoked it), so
//! its assertions are ported here as a Rust integration test that runs under
//! `cargo test`, and the script is removed.
//!
//! The assertions guard the Gitleaks workflow (`gitleaks.yml`) and carry forward
//! the intent of Issue #1217: the gitleaks release tarball must be verified
//! against a committed SHA-256 before it is extracted and executed in CI, and
//! the version must stay pinned alongside that hash.
//!
//! Each check is a pure function over the workflow text (`&str`), so it can be
//! exercised against fixtures. The `real_workflow_*` tests run every check
//! against `.github/workflows/gitleaks.yml`. The workflow is parsed as plain
//! text (no YAML crate is in the dependency tree), matching the sibling workflow
//! tests such as `issue_2390_coverage_workflow_guards.rs`.
//!
//! Every check fails closed: a missing key, step or command is an error, not a
//! pass. Comment lines are ignored, so a commented-out verification does not
//! satisfy a check.

use std::fs;
use std::path::Path;

const WORKFLOW_RELATIVE_PATH: &str = ".github/workflows/gitleaks.yml";

/// Name of the step that downloads and extracts the gitleaks binary.
const INSTALL_STEP: &str = "Install gitleaks";

/// Indentation of a step-list item (`      - `) inside `steps:`.
const STEP_ITEM_PREFIX: &str = "      - ";

/// The `linux_x64` SHA-256 of the gitleaks 8.24.3 release, taken from the
/// published `gitleaks_8.24.3_checksums.txt`.
const LINUX_X64_SHA256_8_24_3: &str =
    "9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c";

/// Published `linux_x64` checksums, keyed by gitleaks version. Every version the
/// workflow pins must have an entry here, and the workflow's `EXPECTED_SHA256`
/// must equal the entry's hash.
const PUBLISHED_CHECKSUMS: &[(&str, &str)] = &[("8.24.3", LINUX_X64_SHA256_8_24_3)];

const NO_SHA_CHECK: &str =
    "Install step has no non-comment line that runs `sha256sum -c` (Issue #1217)";

type Check = fn(&str) -> Result<(), String>;

fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

/// Collapses runs of whitespace so checks do not depend on spacing.
fn normalise(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalised non-comment lines. Comment-only lines are dropped so that
/// documentation mentioning a removed command does not trip a check.
fn code_lines(body: &str) -> Vec<String> {
    body.lines()
        .filter(|line| !is_comment_line(line))
        .map(normalise)
        .collect()
}

/// Splits the `steps:` list into step blocks. A step starts at a line with
/// exactly six spaces followed by `- ` and runs until the next such line or
/// until the list ends (a non-blank line indented less than six spaces).
/// Steps may open with `uses:` rather than `name:`.
fn step_blocks(body: &str) -> Result<Vec<Vec<&str>>, String> {
    let lines: Vec<&str> = body.lines().collect();
    let steps_at = lines
        .iter()
        .position(|line| line.trim() == "steps:")
        .ok_or_else(|| "workflow has no `steps:` key".to_string())?;

    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for &line in &lines[steps_at + 1..] {
        if line.trim().is_empty() || is_comment_line(line) {
            if let Some(block) = blocks.last_mut() {
                block.push(line);
            }
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent < 6 {
            break;
        }
        if line.starts_with(STEP_ITEM_PREFIX) {
            blocks.push(vec![line]);
        } else if let Some(block) = blocks.last_mut() {
            block.push(line);
        }
    }
    Ok(blocks)
}

/// Returns the step whose header line is exactly `- name: <name>`. A missing
/// name is an error, so a renamed step cannot silently skip its check.
fn named_step<'a>(blocks: &[Vec<&'a str>], name: &str) -> Result<Vec<&'a str>, String> {
    let header = format!("- name: {name}");
    blocks
        .iter()
        .find(|block| {
            block
                .first()
                .is_some_and(|first| normalise(first) == header)
        })
        .cloned()
        .ok_or_else(|| format!("workflow has no step named `{name}`"))
}

/// Joins backslash-continued physical lines into logical lines, dropping
/// comment lines first. A blank line ends a logical line, as it does in a
/// shell script. Each logical line is whitespace-normalised.
fn logical_lines(body: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut pending = String::new();
    for line in body.lines() {
        if is_comment_line(line) {
            continue;
        }
        if line.trim().is_empty() {
            push_logical(&mut lines, &mut pending);
            continue;
        }
        let trimmed = line.trim_end();
        if let Some(head) = trimmed.strip_suffix('\\') {
            pending.push_str(head);
            pending.push(' ');
        } else {
            pending.push_str(trimmed);
            push_logical(&mut lines, &mut pending);
        }
    }
    push_logical(&mut lines, &mut pending);
    lines
}

fn push_logical(lines: &mut Vec<String>, pending: &mut String) {
    let joined = normalise(pending);
    if !joined.is_empty() {
        lines.push(joined);
    }
    pending.clear();
}

/// Splits a logical line into shell-like words. Pipes, separators and
/// parentheses end a word, and surrounding double or single quotes are removed.
fn tokens(line: &str) -> Vec<&str> {
    line.split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&' | '(' | ')'))
        .map(|tok| tok.trim_matches(|c: char| c == '"' || c == '\''))
        .filter(|tok| !tok.is_empty())
        .collect()
}

/// The logical lines of the `Install gitleaks` step's run block.
fn install_step_lines(body: &str) -> Result<Vec<String>, String> {
    let block = named_step(&step_blocks(body)?, INSTALL_STEP)?;
    Ok(logical_lines(&block.join("\n")))
}

/// True when the words contain `sha256sum -c`.
fn runs_sha256_check(toks: &[&str]) -> bool {
    toks.windows(2)
        .any(|pair| pair[0] == "sha256sum" && pair[1] == "-c")
}

fn sha_check_index(lines: &[String]) -> Result<usize, String> {
    lines
        .iter()
        .position(|line| runs_sha256_check(&tokens(line)))
        .ok_or_else(|| NO_SHA_CHECK.to_string())
}

/// Index of the `tar -xzf` word pair, if any.
fn tar_extract_index(toks: &[&str]) -> Option<usize> {
    toks.windows(2)
        .position(|pair| pair[0] == "tar" && pair[1] == "-xzf")
}

/// The archive named by a `tar -xzf <archive>` command.
fn tar_archive<'a>(toks: &[&'a str]) -> Option<&'a str> {
    let at = tar_extract_index(toks)?;
    toks.get(at + 2).copied()
}

/// True for a short-flag cluster that ends in `o`, such as `-fsSLo`. In curl
/// the `o` takes the next word as the output path.
fn is_short_output_flag(tok: &str) -> bool {
    tok.len() > 1 && tok.starts_with('-') && !tok.starts_with("--") && tok.ends_with('o')
}

/// The path the `curl` download writes to (`-o <path>`, `--output <path>` or a
/// cluster such as `-fsSLo <path>`).
fn download_path(lines: &[String]) -> Result<String, String> {
    let curl = lines
        .iter()
        .map(|line| tokens(line))
        .find(|toks| toks.first() == Some(&"curl"))
        .ok_or_else(|| "Install step has no `curl` download line".to_string())?;
    let at = curl
        .iter()
        .position(|tok| *tok == "--output" || is_short_output_flag(tok))
        .ok_or_else(|| {
            "the `curl` download line names no output path (`-o` or `-fsSLo`)".to_string()
        })?;
    curl.get(at + 1)
        .map(ToString::to_string)
        .ok_or_else(|| "the `curl` output flag is not followed by a path".to_string())
}

/// Values of every non-comment line of the form `NAME="value"`, in file order.
fn declared_values(body: &str, name: &str) -> Vec<String> {
    let prefix = format!("{name}=\"");
    code_lines(body)
        .iter()
        .filter_map(|line| line.strip_prefix(prefix.as_str())?.strip_suffix('"'))
        .map(str::to_string)
        .collect()
}

/// The single declared value of `name`. Zero or several declarations fail
/// closed, because the shell would use the last one.
fn sole_value(values: &[String], name: &str) -> Result<String, String> {
    match values {
        [only] => Ok(only.clone()),
        [] => Err(format!("no non-comment line declares `{name}`")),
        _ => Err(format!(
            "`{name}` must be declared exactly once, found {}",
            values.len()
        )),
    }
}

/// `x.y.z` with each part a non-empty run of ASCII digits.
fn is_semver(value: &str) -> bool {
    let parts: Vec<&str> = value.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Exactly 64 lowercase hexadecimal characters.
fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

// --- Checks ----------------------------------------------------------------

/// `GITLEAKS_VERSION` is pinned to a semver on a non-comment line.
fn check_gitleaks_version_pinned(body: &str) -> Result<(), String> {
    let versions = declared_values(body, "GITLEAKS_VERSION");
    if versions.is_empty() {
        return Err("no non-comment line declares `GITLEAKS_VERSION=\"x.y.z\"`".to_string());
    }
    match versions.iter().find(|version| !is_semver(version)) {
        Some(bad) => Err(format!(
            "`GITLEAKS_VERSION` must be pinned to a semver `x.y.z`, found `{bad}`"
        )),
        None => Ok(()),
    }
}

/// `EXPECTED_SHA256` is declared on a non-comment line as 64 lowercase hex
/// characters.
fn check_expected_sha256_declared(body: &str) -> Result<(), String> {
    let values = declared_values(body, "EXPECTED_SHA256");
    if values.is_empty() {
        return Err("no non-comment line declares `EXPECTED_SHA256=\"<sha256>\"`".to_string());
    }
    match values.iter().find(|value| !is_lower_hex_sha256(value)) {
        Some(bad) => Err(format!(
            "`EXPECTED_SHA256` must be 64 lowercase hex characters, found `{bad}`"
        )),
        None => Ok(()),
    }
}

/// The workflow's version is listed in `PUBLISHED_CHECKSUMS`, and its
/// `EXPECTED_SHA256` equals the published hash for that version.
fn check_published_checksum_pairing(body: &str) -> Result<(), String> {
    let version = sole_value(
        &declared_values(body, "GITLEAKS_VERSION"),
        "GITLEAKS_VERSION",
    )?;
    let sha = sole_value(&declared_values(body, "EXPECTED_SHA256"), "EXPECTED_SHA256")?;
    let Some((_, published)) = PUBLISHED_CHECKSUMS
        .iter()
        .find(|(listed, _)| *listed == version)
    else {
        return Err(format!(
            "gitleaks {version} is not in PUBLISHED_CHECKSUMS: add the published checksum \
             (linux_x64 SHA-256 from gitleaks_{version}_checksums.txt) before pinning it \
             (Issue #1217)"
        ));
    };
    if sha == *published {
        Ok(())
    } else {
        Err(format!(
            "EXPECTED_SHA256 `{sha}` does not match the published linux_x64 checksum \
             `{published}` for gitleaks {version} (Issue #1217)"
        ))
    }
}

/// A logical line in the Install step runs `sha256sum -c` against
/// `${EXPECTED_SHA256}` and the tarball that `curl` wrote.
fn check_sha256sum_verifies_download(body: &str) -> Result<(), String> {
    let lines = install_step_lines(body)?;
    let path = download_path(&lines)?;
    sha_check_index(&lines)?;
    let verified = lines.iter().map(|line| tokens(line)).any(|toks| {
        runs_sha256_check(&toks)
            && toks.contains(&"${EXPECTED_SHA256}")
            && toks.contains(&path.as_str())
    });
    if verified {
        Ok(())
    } else {
        Err(format!(
            "the `sha256sum -c` line must verify `${{EXPECTED_SHA256}}` against the downloaded \
             tarball `{path}` (Issue #1217)"
        ))
    }
}

/// The `sha256sum -c` logical line comes before the `tar -xzf` logical line,
/// and the tar line extracts the tarball that was verified.
fn check_verify_before_extract(body: &str) -> Result<(), String> {
    let lines = install_step_lines(body)?;
    let sha_at = sha_check_index(&lines)?;
    let tar_at = lines
        .iter()
        .position(|line| tar_extract_index(&tokens(line)).is_some())
        .ok_or_else(|| "Install step has no `tar -xzf` extraction line".to_string())?;
    if sha_at >= tar_at {
        return Err(
            "the `sha256sum -c` line must come before the `tar -xzf` line (Issue #1217)"
                .to_string(),
        );
    }
    let path = download_path(&lines)?;
    let archive = tar_archive(&tokens(&lines[tar_at]))
        .ok_or_else(|| "the `tar -xzf` line names no archive".to_string())?;
    if archive == path.as_str() {
        Ok(())
    } else {
        Err(format!(
            "`tar -xzf` must extract the verified tarball `{path}`, found `{archive}` \
             (Issue #1217)"
        ))
    }
}

/// The Install step's run block sets `set -euo pipefail` before the
/// `sha256sum -c` line, so a failed check aborts the step.
fn check_set_euo_pipefail_before_sha(body: &str) -> Result<(), String> {
    let lines = install_step_lines(body)?;
    let sha_at = sha_check_index(&lines)?;
    let set_at = lines
        .iter()
        .position(|line| line.as_str() == "set -euo pipefail")
        .ok_or_else(|| {
            "the Install step's run block must open with `set -euo pipefail` before the \
             sha256sum line (Issue #1217)"
                .to_string()
        })?;
    if set_at < sha_at {
        Ok(())
    } else {
        Err(
            "`set -euo pipefail` must come before the `sha256sum -c` line (Issue #1217)"
                .to_string(),
        )
    }
}

fn all_checks() -> Vec<(&'static str, Check)> {
    vec![
        (
            "GITLEAKS_VERSION pinned to semver",
            check_gitleaks_version_pinned as Check,
        ),
        (
            "EXPECTED_SHA256 declared as lowercase hex",
            check_expected_sha256_declared as Check,
        ),
        (
            "published checksum pairing",
            check_published_checksum_pairing as Check,
        ),
        (
            "sha256sum -c verifies the download",
            check_sha256sum_verifies_download as Check,
        ),
        (
            "verification before extraction",
            check_verify_before_extract as Check,
        ),
        (
            "set -euo pipefail before verification",
            check_set_euo_pipefail_before_sha as Check,
        ),
    ]
}

fn read_real_workflow() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(WORKFLOW_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn assert_ok(result: Result<(), String>) {
    if let Err(message) = result {
        panic!("{message}");
    }
}

// --- Real-file tests -------------------------------------------------------

#[test]
fn real_workflow_pins_gitleaks_version() {
    assert_ok(check_gitleaks_version_pinned(&read_real_workflow()));
}

#[test]
fn real_workflow_declares_expected_sha256() {
    assert_ok(check_expected_sha256_declared(&read_real_workflow()));
}

#[test]
fn real_workflow_matches_published_checksum() {
    assert_ok(check_published_checksum_pairing(&read_real_workflow()));
}

#[test]
fn real_workflow_verifies_download_with_sha256sum() {
    assert_ok(check_sha256sum_verifies_download(&read_real_workflow()));
}

#[test]
fn real_workflow_verifies_before_extraction() {
    assert_ok(check_verify_before_extract(&read_real_workflow()));
}

#[test]
fn real_workflow_sets_pipefail_before_verification() {
    assert_ok(check_set_euo_pipefail_before_sha(&read_real_workflow()));
}

// --- Fixture tests ---------------------------------------------------------

/// A minimal workflow that satisfies every check. Negative tests mutate it one
/// property at a time, so each failure is attributable to a single edit.
const COMPLIANT: &str = r#"name: Gitleaks

on:
  pull_request:
    branches: ["*", "milestone/*"]

permissions:
  contents: read

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          fetch-depth: 0
          persist-credentials: false
      - name: Install gitleaks
        run: |
          set -euo pipefail
          GITLEAKS_VERSION="8.24.3"
          EXPECTED_SHA256="9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c"
          # Comment mentioning sha256sum -c and tar -xzf must be ignored.
          curl -fsSLo /tmp/gitleaks.tar.gz \
            --retry 5 --retry-delay 2 --retry-all-errors --connect-timeout 30 \
            "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
          echo "${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz" | sha256sum -c -
          tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks
          sudo install -m 0755 /tmp/gitleaks /usr/local/bin/gitleaks
          gitleaks version
      - name: Scan for secrets
        run: |
          set -euo pipefail
          gitleaks detect \
            --source . \
            --config .github/gitleaks.toml \
            --redact \
            --verbose
"#;

const VERSION_LINE: &str = "          GITLEAKS_VERSION=\"8.24.3\"\n";
const CURL_HEAD: &str = "          curl -fsSLo /tmp/gitleaks.tar.gz \\\n";
const SHA_LINE: &str =
    "          echo \"${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz\" | sha256sum -c -\n";
const SHA_CONTINUED: &str = "          echo \"${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz\" | sha256sum \\\n            -c -\n";
const TAR_LINE: &str = "          tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks\n";
const TAR_CONTINUED: &str =
    "          tar \\\n            -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks\n";
const SET_LINE: &str = "          set -euo pipefail\n";

/// The `EXPECTED_SHA256="<value>"` declaration line, indented as in the
/// Install step's run block.
fn expected_sha_line(value: &str) -> String {
    format!("          EXPECTED_SHA256=\"{value}\"\n")
}

/// Replaces exactly one occurrence, failing loudly if the fixture drifted so a
/// no-op mutation can never make a negative test pass vacuously.
fn replace_once(body: &str, from: &str, to: &str) -> String {
    assert!(body.contains(from), "fixture no longer contains {from:?}");
    body.replacen(from, to, 1)
}

#[test]
fn compliant_fixture_passes_every_check() {
    for (name, check) in all_checks() {
        assert!(
            check(COMPLIANT).is_ok(),
            "baseline fixture must pass `{name}`: {:?}",
            check(COMPLIANT)
        );
    }
}

// Check 1: GITLEAKS_VERSION pinned to a semver.

#[test]
fn version_not_semver_fails() {
    let body = replace_once(
        COMPLIANT,
        VERSION_LINE,
        "          GITLEAKS_VERSION=\"latest\"\n",
    );
    let err = check_gitleaks_version_pinned(&body).expect_err("`latest` is not a semver");
    assert!(err.contains("semver"), "unexpected message: {err}");
}

#[test]
fn version_semver_passes_for_the_same_fixture() {
    let bad = replace_once(
        COMPLIANT,
        VERSION_LINE,
        "          GITLEAKS_VERSION=\"latest\"\n",
    );
    let fixed = replace_once(
        &bad,
        "GITLEAKS_VERSION=\"latest\"",
        "GITLEAKS_VERSION=\"8.24.3\"",
    );
    assert_ok(check_gitleaks_version_pinned(&fixed));
}

#[test]
fn version_missing_fails_closed() {
    let body = replace_once(COMPLIANT, VERSION_LINE, "");
    let err = check_gitleaks_version_pinned(&body).expect_err("a missing version must fail");
    assert!(
        err.contains("GITLEAKS_VERSION"),
        "unexpected message: {err}"
    );
}

#[test]
fn version_only_in_comment_fails() {
    let body = replace_once(
        COMPLIANT,
        VERSION_LINE,
        "          # GITLEAKS_VERSION=\"8.24.3\"\n",
    );
    assert!(check_gitleaks_version_pinned(&body).is_err());
    let fixed = replace_once(
        &body,
        "          # GITLEAKS_VERSION=\"8.24.3\"\n",
        VERSION_LINE,
    );
    assert_ok(check_gitleaks_version_pinned(&fixed));
}

// Check 2: EXPECTED_SHA256 declared as 64 lowercase hex characters.

#[test]
fn expected_sha256_missing_fails() {
    let body = replace_once(COMPLIANT, &expected_sha_line(LINUX_X64_SHA256_8_24_3), "");
    let err = check_expected_sha256_declared(&body).expect_err("a missing hash must fail");
    assert!(err.contains("EXPECTED_SHA256"), "unexpected message: {err}");
}

#[test]
fn expected_sha256_63_chars_fails() {
    let short = &LINUX_X64_SHA256_8_24_3[..63];
    let body = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(short),
    );
    let err = check_expected_sha256_declared(&body).expect_err("63 characters must fail");
    assert!(
        err.contains("64 lowercase hex"),
        "unexpected message: {err}"
    );
}

#[test]
fn expected_sha256_63_chars_passes_once_restored_to_64() {
    let short = &LINUX_X64_SHA256_8_24_3[..63];
    let bad = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(short),
    );
    let fixed = replace_once(
        &bad,
        &expected_sha_line(short),
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
    );
    assert_ok(check_expected_sha256_declared(&fixed));
}

#[test]
fn expected_sha256_uppercase_fails() {
    let upper = LINUX_X64_SHA256_8_24_3.to_uppercase();
    let body = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(&upper),
    );
    let err = check_expected_sha256_declared(&body).expect_err("uppercase hex must fail");
    assert!(
        err.contains("64 lowercase hex"),
        "unexpected message: {err}"
    );
}

#[test]
fn expected_sha256_lowercase_passes_for_the_same_fixture() {
    let upper = LINUX_X64_SHA256_8_24_3.to_uppercase();
    let bad = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(&upper),
    );
    let fixed = replace_once(
        &bad,
        &expected_sha_line(&upper),
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
    );
    assert_ok(check_expected_sha256_declared(&fixed));
}

// Check 3: published-checksum pairing.

#[test]
fn wrong_hash_for_listed_version_fails() {
    // Change only the last digit: the hash stays well-formed but is not the
    // published one for 8.24.3.
    let tampered = format!("{}d", &LINUX_X64_SHA256_8_24_3[..63]);
    let body = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(&tampered),
    );
    assert!(check_expected_sha256_declared(&body).is_ok());
    let err = check_published_checksum_pairing(&body).expect_err("a wrong hash must fail");
    assert!(err.contains("does not match"), "unexpected message: {err}");
}

#[test]
fn wrong_hash_passes_once_hash_is_the_published_one() {
    let tampered = format!("{}d", &LINUX_X64_SHA256_8_24_3[..63]);
    let bad = replace_once(
        COMPLIANT,
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
        &expected_sha_line(&tampered),
    );
    let fixed = replace_once(
        &bad,
        &expected_sha_line(&tampered),
        &expected_sha_line(LINUX_X64_SHA256_8_24_3),
    );
    assert_ok(check_published_checksum_pairing(&fixed));
}

#[test]
fn unlisted_version_fails_and_asks_for_published_checksum() {
    let body = replace_once(
        COMPLIANT,
        VERSION_LINE,
        "          GITLEAKS_VERSION=\"8.24.4\"\n",
    );
    let err = check_published_checksum_pairing(&body).expect_err("an unlisted version must fail");
    assert!(
        err.contains("add the published checksum"),
        "unexpected message: {err}"
    );
    assert!(err.contains("8.24.4"), "unexpected message: {err}");
}

#[test]
fn unlisted_version_passes_once_version_is_listed() {
    let bad = replace_once(
        COMPLIANT,
        VERSION_LINE,
        "          GITLEAKS_VERSION=\"8.24.4\"\n",
    );
    let fixed = replace_once(
        &bad,
        "GITLEAKS_VERSION=\"8.24.4\"",
        "GITLEAKS_VERSION=\"8.24.3\"",
    );
    assert_ok(check_published_checksum_pairing(&fixed));
}

#[test]
fn pairing_with_missing_version_fails_closed() {
    let body = replace_once(COMPLIANT, VERSION_LINE, "");
    let err = check_published_checksum_pairing(&body).expect_err("a missing version must fail");
    assert!(
        err.contains("GITLEAKS_VERSION"),
        "unexpected message: {err}"
    );
}

// Check 4: a logical line runs `sha256sum -c` on ${EXPECTED_SHA256} and the
// downloaded tarball.

#[test]
fn sha_line_dropped_fails() {
    let body = replace_once(COMPLIANT, SHA_LINE, "");
    let err = check_sha256sum_verifies_download(&body).expect_err("a dropped line must fail");
    assert!(err.contains("sha256sum -c"), "unexpected message: {err}");
}

#[test]
fn sha_line_restored_passes_for_the_same_fixture() {
    let bad = replace_once(COMPLIANT, SHA_LINE, "");
    let fixed = replace_once(&bad, TAR_LINE, &format!("{SHA_LINE}{TAR_LINE}"));
    assert_ok(check_sha256sum_verifies_download(&fixed));
}

#[test]
fn sha_line_only_in_comment_fails() {
    let body = replace_once(
        COMPLIANT,
        SHA_LINE,
        "          # echo \"${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz\" | sha256sum -c -\n",
    );
    let err = check_sha256sum_verifies_download(&body).expect_err("a commented line must fail");
    assert!(err.contains("sha256sum -c"), "unexpected message: {err}");
}

#[test]
fn sha_line_uncommented_passes_for_the_same_fixture() {
    let bad = replace_once(
        COMPLIANT,
        SHA_LINE,
        "          # echo \"${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz\" | sha256sum -c -\n",
    );
    let fixed = replace_once(
        &bad,
        "          # echo \"${EXPECTED_SHA256}  /tmp/gitleaks.tar.gz\" | sha256sum -c -\n",
        SHA_LINE,
    );
    assert_ok(check_sha256sum_verifies_download(&fixed));
}

#[test]
fn hash_variable_not_used_fails() {
    let literal = SHA_LINE.replace("${EXPECTED_SHA256}", LINUX_X64_SHA256_8_24_3);
    let body = replace_once(COMPLIANT, SHA_LINE, &literal);
    let err = check_sha256sum_verifies_download(&body).expect_err("a literal hash must fail");
    assert!(
        err.contains("${EXPECTED_SHA256}"),
        "unexpected message: {err}"
    );
}

#[test]
fn hash_variable_used_passes_for_the_same_fixture() {
    let literal = SHA_LINE.replace("${EXPECTED_SHA256}", LINUX_X64_SHA256_8_24_3);
    let bad = replace_once(COMPLIANT, SHA_LINE, &literal);
    let fixed = replace_once(&bad, &literal, SHA_LINE);
    assert_ok(check_sha256sum_verifies_download(&fixed));
}

#[test]
fn download_path_not_verified_fails() {
    let other = SHA_LINE.replace("/tmp/gitleaks.tar.gz", "/tmp/other.tar.gz");
    let body = replace_once(COMPLIANT, SHA_LINE, &other);
    let err = check_sha256sum_verifies_download(&body).expect_err("the wrong path must fail");
    assert!(err.contains("downloaded"), "unexpected message: {err}");
}

#[test]
fn download_path_verified_passes_for_the_same_fixture() {
    let other = SHA_LINE.replace("/tmp/gitleaks.tar.gz", "/tmp/other.tar.gz");
    let bad = replace_once(COMPLIANT, SHA_LINE, &other);
    let fixed = replace_once(&bad, &other, SHA_LINE);
    assert_ok(check_sha256sum_verifies_download(&fixed));
}

#[test]
fn curl_without_output_path_fails_closed() {
    let body = replace_once(
        COMPLIANT,
        CURL_HEAD,
        "          curl -fsSL /tmp/gitleaks.tar.gz \\\n",
    );
    let err = check_sha256sum_verifies_download(&body).expect_err("no output flag must fail");
    assert!(err.contains("output path"), "unexpected message: {err}");
}

#[test]
fn curl_with_output_flag_passes_for_the_same_fixture() {
    let bad = replace_once(
        COMPLIANT,
        CURL_HEAD,
        "          curl -fsSL /tmp/gitleaks.tar.gz \\\n",
    );
    let fixed = replace_once(
        &bad,
        "curl -fsSL /tmp/gitleaks.tar.gz",
        "curl -fsSLo /tmp/gitleaks.tar.gz",
    );
    assert_ok(check_sha256sum_verifies_download(&fixed));
}

#[test]
fn install_step_renamed_fails_closed() {
    let body = replace_once(
        COMPLIANT,
        "      - name: Install gitleaks\n",
        "      - name: Install tools\n",
    );
    let err = check_sha256sum_verifies_download(&body).expect_err("a renamed step must fail");
    assert!(
        err.contains("Install gitleaks"),
        "unexpected message: {err}"
    );
}

// Check 5: ordering. The sha256sum -c logical line comes before the tar -xzf
// logical line, and extracts the verified tarball.

#[test]
fn verification_after_extraction_fails() {
    let swapped = replace_once(
        COMPLIANT,
        &format!("{SHA_LINE}{TAR_LINE}"),
        &format!("{TAR_LINE}{SHA_LINE}"),
    );
    let err = check_verify_before_extract(&swapped).expect_err("late verification must fail");
    assert!(
        err.contains("must come before"),
        "unexpected message: {err}"
    );
}

#[test]
fn verification_before_extraction_passes_for_the_same_fixture() {
    let swapped = replace_once(
        COMPLIANT,
        &format!("{SHA_LINE}{TAR_LINE}"),
        &format!("{TAR_LINE}{SHA_LINE}"),
    );
    let fixed = replace_once(
        &swapped,
        &format!("{TAR_LINE}{SHA_LINE}"),
        &format!("{SHA_LINE}{TAR_LINE}"),
    );
    assert_ok(check_verify_before_extract(&fixed));
}

#[test]
fn no_tar_extraction_fails_closed() {
    let body = replace_once(COMPLIANT, TAR_LINE, "");
    let err = check_verify_before_extract(&body).expect_err("no tar line must fail");
    assert!(err.contains("tar -xzf"), "unexpected message: {err}");
}

#[test]
fn no_tar_extraction_passes_once_restored() {
    let bad = replace_once(COMPLIANT, TAR_LINE, "");
    let fixed = replace_once(&bad, SHA_LINE, &format!("{SHA_LINE}{TAR_LINE}"));
    assert_ok(check_verify_before_extract(&fixed));
}

#[test]
fn no_sha256sum_fails_closed() {
    let body = replace_once(COMPLIANT, SHA_LINE, "");
    let err = check_verify_before_extract(&body).expect_err("no sha256sum line must fail");
    assert!(err.contains("sha256sum -c"), "unexpected message: {err}");
}

#[test]
fn no_sha256sum_passes_once_restored() {
    let bad = replace_once(COMPLIANT, SHA_LINE, "");
    let fixed = replace_once(&bad, TAR_LINE, &format!("{SHA_LINE}{TAR_LINE}"));
    assert_ok(check_verify_before_extract(&fixed));
}

#[test]
fn tar_extracts_different_tarball_fails() {
    let body = replace_once(
        COMPLIANT,
        TAR_LINE,
        "          tar -xzf /tmp/other.tar.gz -C /tmp gitleaks\n",
    );
    let err = check_verify_before_extract(&body).expect_err("a different tarball must fail");
    assert!(
        err.contains("verified tarball"),
        "unexpected message: {err}"
    );
}

#[test]
fn tar_extracts_verified_tarball_passes_for_the_same_fixture() {
    let bad = replace_once(
        COMPLIANT,
        TAR_LINE,
        "          tar -xzf /tmp/other.tar.gz -C /tmp gitleaks\n",
    );
    let fixed = replace_once(
        &bad,
        "tar -xzf /tmp/other.tar.gz",
        "tar -xzf /tmp/gitleaks.tar.gz",
    );
    assert_ok(check_verify_before_extract(&fixed));
}

#[test]
fn verification_after_backslash_continued_tar_fails() {
    // The tar command is re-wrapped across two physical lines. Joining the
    // continuation must still place it before the sha256sum line.
    let swapped = replace_once(
        COMPLIANT,
        &format!("{SHA_LINE}{TAR_LINE}"),
        &format!("{TAR_CONTINUED}{SHA_LINE}"),
    );
    let err = check_verify_before_extract(&swapped)
        .expect_err("verification after a continued tar line must fail");
    assert!(
        err.contains("must come before"),
        "unexpected message: {err}"
    );
}

#[test]
fn verification_before_backslash_continued_tar_passes_for_the_same_fixture() {
    let fixed = replace_once(
        COMPLIANT,
        &format!("{SHA_LINE}{TAR_LINE}"),
        &format!("{SHA_LINE}{TAR_CONTINUED}"),
    );
    assert_ok(check_verify_before_extract(&fixed));
}

#[test]
fn backslash_continued_sha_line_is_found() {
    let body = replace_once(COMPLIANT, SHA_LINE, SHA_CONTINUED);
    assert_ok(check_sha256sum_verifies_download(&body));
    assert_ok(check_verify_before_extract(&body));
    assert_ok(check_set_euo_pipefail_before_sha(&body));
}

// Check 6: `set -euo pipefail` precedes the sha256sum line in the Install step.

#[test]
fn pipefail_missing_fails() {
    // The first `set -euo pipefail` in the fixture belongs to the Install step.
    let body = replace_once(COMPLIANT, SET_LINE, "");
    let err = check_set_euo_pipefail_before_sha(&body).expect_err("missing pipefail must fail");
    assert!(
        err.contains("set -euo pipefail"),
        "unexpected message: {err}"
    );
}

#[test]
fn pipefail_missing_passes_once_restored() {
    let bad = replace_once(COMPLIANT, SET_LINE, "");
    let fixed = replace_once(&bad, CURL_HEAD, &format!("{SET_LINE}{CURL_HEAD}"));
    assert_ok(check_set_euo_pipefail_before_sha(&fixed));
}

#[test]
fn pipefail_after_sha_line_fails() {
    let moved = replace_once(COMPLIANT, SET_LINE, "");
    let moved = replace_once(&moved, SHA_LINE, &format!("{SHA_LINE}{SET_LINE}"));
    let err = check_set_euo_pipefail_before_sha(&moved).expect_err("late pipefail must fail");
    assert!(
        err.contains("must come before"),
        "unexpected message: {err}"
    );
}

#[test]
fn pipefail_before_sha_line_passes_for_the_same_fixture() {
    let moved = replace_once(COMPLIANT, SET_LINE, "");
    let moved = replace_once(&moved, SHA_LINE, &format!("{SHA_LINE}{SET_LINE}"));
    let fixed = replace_once(&moved, CURL_HEAD, &format!("{SET_LINE}{CURL_HEAD}"));
    assert_ok(check_set_euo_pipefail_before_sha(&fixed));
}
