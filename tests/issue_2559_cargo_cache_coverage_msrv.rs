//! Issue #2559: the `coverage` job (`cargo-quality.yml`) and the `msrv` job
//! (`msrv.yml`) each compile the crate from a cold cache unless a dependency
//! cache is restored first. Both jobs now restore the Cargo registry and git
//! caches before their first cargo invocation.
//!
//! The cache is limited to downloaded crates. `target/` must never be cached:
//! coverage instrumentation differs from the other jobs, and a restored
//! `target/` can exhaust the runner's disk (Issue #1489; see `ci.yml`). The
//! coverage key hashes `rust-toolchain.toml`, and the msrv key names the
//! declared toolchain, so no cache is shared across a toolchain bump.
//!
//! Each check is a pure function over workflow text (`&str`), so it can be
//! exercised against inline fixtures. The `real_*` tests run every check
//! against the two real workflow files. The workflows are parsed as plain text
//! (no YAML crate is in the dependency tree), matching
//! `issue_2390_coverage_workflow_guards.rs` and `issue_2395_msrv_declared.rs`.
//!
//! Every check fails closed: a missing job, step, path, key or restore key is
//! an error, not a pass.

use std::fs;
use std::path::PathBuf;

const COVERAGE_WORKFLOW: &str = ".github/workflows/cargo-quality.yml";
const MSRV_WORKFLOW: &str = ".github/workflows/msrv.yml";

/// The composite action every job uses to install Rust (Issues #1891, #2036).
const SETUP_RUST_ACTION: &str = "./.github/actions/setup-rust";
const CACHE_ACTION_PREFIX: &str = "actions/cache@";
const CARGO_COMMAND: &str = "cargo ";
const REGISTRY_PATH: &str = "~/.cargo/registry";
const GIT_PATH: &str = "~/.cargo/git";

/// How a job's cache key pins its toolchain.
enum KeyToolchain {
    /// Coverage: the key hashes `rust-toolchain.toml`.
    RustToolchainFile,
    /// MSRV: the key carries the toolchain passed to `setup-rust`.
    SetupRustInput,
}

struct JobExpectation {
    workflow: &'static str,
    job: &'static str,
    key_toolchain: KeyToolchain,
}

const COVERAGE_JOB: JobExpectation = JobExpectation {
    workflow: COVERAGE_WORKFLOW,
    job: "coverage",
    key_toolchain: KeyToolchain::RustToolchainFile,
};

const MSRV_JOB: JobExpectation = JobExpectation {
    workflow: MSRV_WORKFLOW,
    job: "msrv",
    key_toolchain: KeyToolchain::SetupRustInput,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &str) -> String {
    let full = repo_root().join(path);
    fs::read_to_string(&full).unwrap_or_else(|e| panic!("read {}: {e}", full.display()))
}

fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Collapses runs of whitespace so checks do not depend on spacing.
fn normalise(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn unquote(value: &str) -> &str {
    value.trim_matches(|c: char| c == '"' || c == '\'')
}

/// Splits the `steps:` list of `job` into step blocks. A step starts at a line
/// indented six spaces followed by `- `. Its block runs until the next step or
/// until the list ends (a non-blank line indented less than six spaces).
fn job_steps<'a>(workflow: &'a str, job: &str) -> Result<Vec<Vec<&'a str>>, String> {
    let lines: Vec<&str> = workflow.lines().collect();
    let job_header = format!("  {job}:");
    let job_at = lines
        .iter()
        .position(|line| line.trim_end() == job_header)
        .ok_or_else(|| format!("workflow has no job `{job}`"))?;
    let steps_at = lines[job_at + 1..]
        .iter()
        .position(|line| line.trim() == "steps:" && indent_of(line) == 4)
        .map(|offset| job_at + 1 + offset)
        .ok_or_else(|| format!("job `{job}` has no `steps:` key"))?;

    let mut blocks: Vec<Vec<&'a str>> = Vec::new();
    for &line in &lines[steps_at + 1..] {
        if line.trim().is_empty() || is_comment_line(line) {
            if let Some(block) = blocks.last_mut() {
                block.push(line);
            }
            continue;
        }
        if indent_of(line) < 6 {
            break;
        }
        if line.starts_with("      - ") {
            blocks.push(vec![line]);
        } else if let Some(block) = blocks.last_mut() {
            block.push(line);
        }
    }
    if blocks.is_empty() {
        return Err(format!("job `{job}` has no steps"));
    }
    Ok(blocks)
}

/// The action reference of a non-comment `uses:` line (any trailing comment
/// dropped), with or without a leading `- ` list marker.
fn uses_value(line: &str) -> Option<String> {
    if is_comment_line(line) {
        return None;
    }
    let norm = normalise(line);
    let rest = norm.strip_prefix("- ").unwrap_or(norm.as_str());
    let value = rest.strip_prefix("uses: ")?;
    value.split_whitespace().next().map(str::to_string)
}

fn step_uses(step: &[&str]) -> Vec<String> {
    step.iter().filter_map(|line| uses_value(line)).collect()
}

/// The text of every `run:` value in `step`, including the deeper-indented
/// lines of a `|` block. Comment lines are ignored.
fn run_text(step: &[&str]) -> String {
    let mut out = String::new();
    let mut block_indent: Option<usize> = None;
    for line in step.iter().copied().filter(|line| !is_comment_line(line)) {
        let indent = indent_of(line);
        if let Some(run_indent) = block_indent {
            if line.trim().is_empty() || indent > run_indent {
                out.push_str(line.trim());
                out.push('\n');
                continue;
            }
            block_indent = None;
        }
        let trimmed = line.trim();
        let body = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        if let Some(rest) = body.strip_prefix("run:") {
            out.push_str(rest.trim());
            out.push('\n');
            block_indent = Some(indent);
        }
    }
    out
}

/// The value(s) of the `name:` key inside `step`. An inline value is one
/// entry; a `|` or `>` block yields each of its deeper-indented lines. Comment
/// lines are ignored, so a commented-out entry is never read.
fn with_entries(step: &[&str], name: &str) -> Option<Vec<String>> {
    let prefix = format!("{name}:");
    let lines: Vec<&str> = step
        .iter()
        .copied()
        .filter(|line| !is_comment_line(line))
        .collect();
    let at = lines
        .iter()
        .position(|line| line.trim().starts_with(prefix.as_str()))?;
    let line = lines[at];
    let key_indent = indent_of(line);
    let rest = line.trim()[prefix.len()..].trim();
    if !rest.is_empty() && rest != "|" && rest != ">" {
        return Some(vec![unquote(rest).to_string()]);
    }
    Some(
        lines[at + 1..]
            .iter()
            .copied()
            .take_while(|line| line.trim().is_empty() || indent_of(line) > key_indent)
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect(),
    )
}

/// Index of the first step that uses an `actions/cache@<40-hex SHA>` action.
fn cache_step_index(steps: &[Vec<&str>], job: &str) -> Result<usize, String> {
    for (index, step) in steps.iter().enumerate() {
        if let Some(action) = step_uses(step)
            .into_iter()
            .find(|uses| uses.starts_with(CACHE_ACTION_PREFIX))
        {
            let sha = &action[CACHE_ACTION_PREFIX.len()..];
            if sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(index);
            }
            return Err(format!(
                "job `{job}`: `{action}` is not pinned to a 40-character hex commit SHA"
            ));
        }
    }
    Err(format!(
        "job `{job}` has no step using `{CACHE_ACTION_PREFIX}<40-hex SHA>`"
    ))
}

fn setup_rust_index(steps: &[Vec<&str>], job: &str) -> Result<usize, String> {
    steps
        .iter()
        .position(|step| step_uses(step).iter().any(|uses| uses == SETUP_RUST_ACTION))
        .ok_or_else(|| format!("job `{job}` has no `uses: {SETUP_RUST_ACTION}` step"))
}

fn first_cargo_index(steps: &[Vec<&str>], job: &str) -> Result<usize, String> {
    steps
        .iter()
        .position(|step| run_text(step).contains(CARGO_COMMAND))
        .ok_or_else(|| format!("job `{job}` has no step whose `run:` contains `{CARGO_COMMAND}`"))
}

/// The cache step must sit after the setup-rust step and before the first
/// step that runs cargo. Both anchors must exist, so a missing one fails.
fn check_ordering(steps: &[Vec<&str>], job: &str, cache: usize) -> Result<(), String> {
    let setup = setup_rust_index(steps, job)?;
    let cargo = first_cargo_index(steps, job)?;
    if cache <= setup {
        return Err(format!(
            "cache step (step {}) must come after the setup-rust step (step {})",
            cache + 1,
            setup + 1
        ));
    }
    if cache >= cargo {
        return Err(format!(
            "cache step (step {}) must come before the first cargo step (step {})",
            cache + 1,
            cargo + 1
        ));
    }
    Ok(())
}

/// The cache may restore only the registry and git caches. Any path entry that
/// mentions `target` is rejected, including one inside a `|` block.
fn check_path(cache: &[&str]) -> Result<(), String> {
    let entries =
        with_entries(cache, "path").ok_or_else(|| "cache step has no `path:`".to_string())?;
    if let Some(entry) = entries.iter().find(|entry| entry.contains("target")) {
        return Err(format!(
            "cache step must not cache `target` (path entry `{entry}`): only the Cargo registry and git caches are restored"
        ));
    }
    for required in [REGISTRY_PATH, GIT_PATH] {
        if !entries.iter().any(|entry| entry == required) {
            return Err(format!("cache path must include `{required}`"));
        }
    }
    Ok(())
}

fn cache_key(cache: &[&str]) -> Result<String, String> {
    with_entries(cache, "key")
        .map(|entries| entries.join(" "))
        .filter(|key| !key.is_empty())
        .ok_or_else(|| "cache step has no `key:`".to_string())
}

/// The argument text of every `hashFiles(...)` call in `key`.
fn hash_files_args(key: &str) -> Vec<&str> {
    key.split("hashFiles(")
        .skip(1)
        .map(|rest| rest.split(')').next().unwrap_or_default())
        .collect()
}

fn check_key_common(key: &str) -> Result<(), String> {
    if !key.contains("runner.os") {
        return Err(format!("cache key must contain `runner.os` (key `{key}`)"));
    }
    if !hash_files_args(key)
        .iter()
        .any(|arg| arg.contains("Cargo.lock"))
    {
        return Err(format!(
            "cache key must hash Cargo.lock with hashFiles(...) (key `{key}`)"
        ));
    }
    Ok(())
}

fn check_key_rust_toolchain_file(key: &str) -> Result<(), String> {
    if key.contains("hashFiles('rust-toolchain.toml')") {
        Ok(())
    } else {
        Err(format!(
            "coverage cache key must contain `hashFiles('rust-toolchain.toml')` (key `{key}`)"
        ))
    }
}

fn check_key_declared_toolchain(key: &str, declared: &str) -> Result<(), String> {
    let token = format!("-{declared}-");
    if key.contains(&token) {
        Ok(())
    } else {
        Err(format!(
            "msrv cache key must contain the declared toolchain `{declared}` between dashes, \
             so the cache is not shared across MSRV bumps (key `{key}`)"
        ))
    }
}

/// The `toolchain:` input passed to the setup-rust step.
fn declared_toolchain(steps: &[Vec<&str>], job: &str) -> Result<String, String> {
    let setup = &steps[setup_rust_index(steps, job)?];
    with_entries(setup, "toolchain")
        .and_then(|entries| entries.into_iter().next())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("job `{job}`: the setup-rust step has no `toolchain:` input"))
}

/// Drops a trailing `${{ hashFiles(...) }}` from a restore key, so the rest
/// can be compared as a prefix of the full cache key.
fn strip_trailing_hash_files(entry: &str) -> &str {
    if !entry.ends_with("}}") {
        return entry;
    }
    match entry.rfind("hashFiles(") {
        Some(at) => match entry[..at].rfind("${{") {
            Some(start) => &entry[..start],
            None => entry,
        },
        None => entry,
    }
}

fn check_restore_keys(cache: &[&str], key: &str) -> Result<(), String> {
    let entries = with_entries(cache, "restore-keys")
        .filter(|entries| !entries.is_empty())
        .ok_or_else(|| "cache step has no `restore-keys:` entry".to_string())?;
    let matched = entries.iter().any(|entry| {
        let prefix = strip_trailing_hash_files(entry);
        !prefix.trim().is_empty() && key.starts_with(prefix)
    });
    if matched {
        Ok(())
    } else {
        Err(format!(
            "no restore-keys entry is a prefix of the cache key `{key}`"
        ))
    }
}

/// Runs every Issue #2559 check against the cache step of `expect.job`.
fn check_cargo_cache(workflow: &str, expect: &JobExpectation) -> Result<(), String> {
    let job = expect.job;
    let steps = job_steps(workflow, job)?;
    let cache_at = cache_step_index(&steps, job)?;
    check_ordering(&steps, job, cache_at)?;
    let cache = &steps[cache_at];

    check_path(cache)?;

    let key = cache_key(cache)?;
    check_key_common(&key)?;
    match expect.key_toolchain {
        KeyToolchain::RustToolchainFile => check_key_rust_toolchain_file(&key)?,
        KeyToolchain::SetupRustInput => {
            let declared = declared_toolchain(&steps, job)?;
            check_key_declared_toolchain(&key, &declared)?;
        }
    }

    check_restore_keys(cache, &key)
}

fn check_real_workflow(expect: &JobExpectation) {
    let body = read(expect.workflow);
    if let Err(message) = check_cargo_cache(&body, expect) {
        panic!("{}: {message}", expect.workflow);
    }
}

// --- Fixtures ----------------------------------------------------------------

/// Wraps `steps` (already indented to the `steps:` list) in a one-job workflow.
fn workflow_fixture(job: &str, steps: &[&str]) -> String {
    format!(
        "name: fixture\non:\n  pull_request:\njobs:\n  {job}:\n    runs-on: ubuntu-latest\n    steps:\n{}",
        steps.concat()
    )
}

const CHECKOUT_STEP: &str = r##"      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          persist-credentials: false
"##;

const COVERAGE_SETUP_STEP: &str = r##"      - name: Install Rust toolchain
        uses: ./.github/actions/setup-rust
"##;

const MSRV_SETUP_STEP: &str = r##"      - name: Install Rust toolchain (declared MSRV)
        uses: ./.github/actions/setup-rust
        with:
          toolchain: "1.99"
"##;

const COVERAGE_CACHE_STEP: &str = r##"      - name: Cache Cargo dependencies
        uses: actions/cache@27d5ce7f107fe9357f9df03efb73ab90386fccae # v5.0.5
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
          key: ${{ runner.os }}-cargo-coverage-${{ hashFiles('rust-toolchain.toml') }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-coverage-${{ hashFiles('rust-toolchain.toml') }}-
"##;

const MSRV_CACHE_STEP: &str = r##"      - name: Cache Cargo dependencies
        uses: actions/cache@27d5ce7f107fe9357f9df03efb73ab90386fccae # v5.0.5
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
          key: ${{ runner.os }}-cargo-msrv-1.99-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-msrv-1.99-
"##;

const COVERAGE_CARGO_STEP: &str = r##"      - name: Generate coverage (lcov)
        run: cargo llvm-cov --all-features --lcov --output-path lcov.info -- --test-threads=2
"##;

const MSRV_CARGO_STEP: &str = r##"      - name: Build every target with the declared MSRV
        run: cargo check --locked --all-targets --all-features
"##;

/// Asserts `result` is an error whose message contains `needle`.
fn expect_err(result: Result<(), String>, needle: &str) {
    match result {
        Ok(()) => panic!("expected an error containing `{needle}`, but the check passed"),
        Err(message) => assert!(
            message.contains(needle),
            "error `{message}` does not contain `{needle}`"
        ),
    }
}

// --- Real workflow files -----------------------------------------------------

#[test]
fn real_coverage_job_restores_cargo_cache_before_its_first_cargo_command() {
    check_real_workflow(&COVERAGE_JOB);
}

#[test]
fn real_msrv_job_restores_cargo_cache_before_its_first_cargo_command() {
    check_real_workflow(&MSRV_JOB);
}

// --- Positive fixture --------------------------------------------------------

#[test]
fn fixture_positive_cache_between_setup_and_cargo_passes() {
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            COVERAGE_CACHE_STEP,
            COVERAGE_CARGO_STEP,
        ],
    );
    assert_eq!(check_cargo_cache(&body, &COVERAGE_JOB), Ok(()));
}

// --- Negative fixtures -------------------------------------------------------

#[test]
fn fixture_negative_missing_cache_step_is_rejected() {
    let body = workflow_fixture(
        "coverage",
        &[CHECKOUT_STEP, COVERAGE_SETUP_STEP, COVERAGE_CARGO_STEP],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "has no step using `actions/cache@",
    );
}

#[test]
fn fixture_negative_cache_after_cargo_step_is_rejected() {
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            COVERAGE_CARGO_STEP,
            COVERAGE_CACHE_STEP,
        ],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "must come before the first cargo step",
    );
}

#[test]
fn fixture_negative_inline_target_path_is_rejected() {
    let cache = COVERAGE_CACHE_STEP.replace(
        "          path: |\n            ~/.cargo/registry\n            ~/.cargo/git\n",
        "          path: target/\n",
    );
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            &cache,
            COVERAGE_CARGO_STEP,
        ],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "must not cache `target`",
    );
}

#[test]
fn fixture_negative_target_rewrapped_inside_path_block_is_rejected() {
    // The registry and git entries are followed by `target/` on its own line
    // inside the `|` block, so only a block-aware parser sees it.
    let cache = COVERAGE_CACHE_STEP.replace(
        "            ~/.cargo/registry\n            ~/.cargo/git\n",
        "            ~/.cargo/registry\n            ~/.cargo/git\n            target/\n",
    );
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            &cache,
            COVERAGE_CARGO_STEP,
        ],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "must not cache `target`",
    );
}

#[test]
fn fixture_negative_key_lacking_cargo_lock_is_rejected() {
    let cache = COVERAGE_CACHE_STEP.replace("-${{ hashFiles('**/Cargo.lock') }}\n", "\n");
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            &cache,
            COVERAGE_CARGO_STEP,
        ],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "must hash Cargo.lock",
    );
}

#[test]
fn fixture_negative_coverage_key_missing_toolchain_file_is_rejected() {
    let cache = COVERAGE_CACHE_STEP.replace("-${{ hashFiles('rust-toolchain.toml') }}", "");
    let body = workflow_fixture(
        "coverage",
        &[
            CHECKOUT_STEP,
            COVERAGE_SETUP_STEP,
            &cache,
            COVERAGE_CARGO_STEP,
        ],
    );
    expect_err(
        check_cargo_cache(&body, &COVERAGE_JOB),
        "must contain `hashFiles('rust-toolchain.toml')`",
    );
}

#[test]
fn fixture_negative_msrv_key_missing_declared_toolchain_is_rejected() {
    let cache = MSRV_CACHE_STEP.replace("-msrv-1.99-${{ hashFiles", "-msrv-${{ hashFiles");
    let body = workflow_fixture(
        "msrv",
        &[CHECKOUT_STEP, MSRV_SETUP_STEP, &cache, MSRV_CARGO_STEP],
    );
    expect_err(
        check_cargo_cache(&body, &MSRV_JOB),
        "must contain the declared toolchain `1.99`",
    );
}
