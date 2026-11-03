//! Issue #2478: the fuzzing CI template `docs/ci-fuzzing-workflow.yml` must
//! carry the hardening the live workflows already have:
//!   1. a column-0 `permissions:` block granting `contents: read` and no
//!      write scope (Issue #1286);
//!   2. a column-0 `concurrency:` block with a `group:` key that cancels
//!      superseded runs (Issue #1288);
//!   3. `timeout-minutes:` on the `fuzzing` job (Issue #1287);
//!   4. `persist-credentials: false` inside the block of every
//!      `actions/checkout` step (Issues #1648, #1868).
//!
//! The template is read as plain text because no YAML parser is in the
//! dependency tree. Blank and comment lines are ignored, so a commented-out
//! directive cannot satisfy an assertion. Each helper is exercised against
//! inline positive and negative YAML so that the assertions can fail.

use std::fs;
use std::path::Path;

fn load_template() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/ci-fuzzing-workflow.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// True for lines that carry YAML content: not blank and not a comment.
fn is_content_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with('#')
}

/// Number of leading spaces on `line`.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The YAML content of `line`: trimmed, with any trailing ` # comment` cut.
fn code_of(line: &str) -> &str {
    line.trim().split(" #").next().unwrap_or("").trim_end()
}

/// Child lines of the column-0 `key:` block, comments and blanks dropped.
/// Returns `None` when there is no column-0 `key:` line.
fn top_level_block<'a>(body: &'a str, key: &str) -> Option<Vec<&'a str>> {
    let header = format!("{key}:");
    let mut lines = body.lines().filter(|line| is_content_line(line));
    lines.find(|line| line.trim_end() == header)?;
    let mut children = Vec::new();
    for line in lines {
        if !line.starts_with(' ') {
            break;
        }
        children.push(code_of(line));
    }
    Some(children)
}

/// True when the column-0 `permissions:` block grants `contents: read` and no
/// write scope. False when the block is absent.
fn permissions_is_read_only(body: &str) -> bool {
    top_level_block(body, "permissions").is_some_and(|children| {
        let reads_contents = children.contains(&"contents: read");
        let grants_write = children.iter().any(|child| child.ends_with(": write"));
        reads_contents && !grants_write
    })
}

/// True when the column-0 `concurrency:` block declares a `group:` key.
fn concurrency_declares_group(body: &str) -> bool {
    top_level_block(body, "concurrency")
        .is_some_and(|children| children.iter().any(|child| child.starts_with("group:")))
}

/// Blocks of the YAML list items in `body`. Each `- ` item starts a block that
/// holds the item's own line (with the dash removed) and every later content
/// line indented deeper than the dash.
fn list_item_blocks(body: &str) -> Vec<Vec<&str>> {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut dash_indent: Option<usize> = None;
    for line in body.lines().filter(|line| is_content_line(line)) {
        let indent = indent_of(line);
        let code = code_of(line);
        if let Some(item) = code.strip_prefix("- ") {
            dash_indent = Some(indent);
            blocks.push(vec![item]);
        } else if let (Some(dash), Some(block)) = (dash_indent, blocks.last_mut()) {
            if indent > dash {
                block.push(code);
            } else {
                dash_indent = None;
            }
        }
    }
    blocks
}

/// Counts the `actions/checkout` steps in `body` and how many of them do not
/// set `persist-credentials: false` in their own block. Returns
/// `(checkouts, unhardened)`.
fn checkout_audit(body: &str) -> (usize, usize) {
    let mut checkouts = 0;
    let mut unhardened = 0;
    for block in list_item_blocks(body) {
        if block
            .iter()
            .any(|line| line.starts_with("uses: actions/checkout@"))
        {
            checkouts += 1;
            if !block.contains(&"persist-credentials: false") {
                unhardened += 1;
            }
        }
    }
    (checkouts, unhardened)
}

/// Child lines of the `  <job>:` block under `jobs:` (comments and blanks
/// dropped), or `None` when the job is absent.
fn job_block<'a>(body: &'a str, job: &str) -> Option<Vec<&'a str>> {
    let header = format!("  {job}:");
    let mut lines = body.lines().filter(|line| is_content_line(line));
    lines.find(|line| line.trim_end() == header)?;
    let mut block = Vec::new();
    for line in lines {
        if indent_of(line) <= 2 {
            break;
        }
        block.push(line);
    }
    Some(block)
}

/// True when `block` (from `job_block`) has a four-space, job-level
/// `timeout-minutes:` key with a value.
fn job_declares_timeout_minutes(block: &[&str]) -> bool {
    block.iter().any(|line| {
        indent_of(line) == 4
            && code_of(line)
                .strip_prefix("timeout-minutes:")
                .is_some_and(|value| !value.trim().is_empty())
    })
}

#[test]
fn permissions_helper_requires_read_only_top_level_block() {
    assert!(permissions_is_read_only(
        "name: Fuzzing\n\npermissions:\n  contents: read\n\njobs:\n  fuzzing:\n    runs-on: x\n"
    ));
    assert!(!permissions_is_read_only(
        "permissions:\n  contents: write\n"
    ));
    assert!(!permissions_is_read_only(
        "permissions:\n  contents: read\n  issues: write\n"
    ));
    assert!(!permissions_is_read_only("name: Fuzzing\n"));
    // A job-level permissions block is indented, so it is not top-level.
    assert!(!permissions_is_read_only(
        "jobs:\n  fuzzing:\n    permissions:\n      contents: read\n"
    ));
    // Commented-out directives do not count.
    assert!(!permissions_is_read_only(
        "# permissions:\n#   contents: read\n"
    ));
    // A commented-out write grant is ignored; the live read grant stands.
    assert!(permissions_is_read_only(
        "permissions:\n  # contents: write\n  contents: read\n"
    ));
}

#[test]
fn template_declares_read_only_top_level_permissions() {
    let body = load_template();
    assert!(
        permissions_is_read_only(&body),
        "docs/ci-fuzzing-workflow.yml must declare a column-0 `permissions:` block \
         granting `contents: read` and no write scope (Issues #1286, #2478)",
    );
}

#[test]
fn concurrency_helper_requires_column_zero_group_key() {
    assert!(concurrency_declares_group(
        "concurrency:\n  group: ${{ github.ref }}\n  cancel-in-progress: true\n"
    ));
    // An indented concurrency key belongs to a job, not the workflow.
    assert!(!concurrency_declares_group(
        "jobs:\n  fuzzing:\n    concurrency:\n      group: x\n"
    ));
    // A block without a group key does not satisfy the check.
    assert!(!concurrency_declares_group(
        "concurrency:\n  cancel-in-progress: true\n"
    ));
    // Commented-out directives do not count.
    assert!(!concurrency_declares_group(
        "# concurrency:\n#   group: x\n"
    ));
}

#[test]
fn template_declares_concurrency_group() {
    let body = load_template();
    assert!(
        concurrency_declares_group(&body),
        "docs/ci-fuzzing-workflow.yml must declare a column-0 `concurrency:` block \
         with a `group:` key (Issues #1288, #2478)",
    );
}

#[test]
fn checkout_helper_requires_persist_false_in_own_step() {
    // Positive: the checkout's own block carries the setting; a trailing
    // comment on that line does not stop it matching.
    let hardened = r#"    steps:
      - name: Checkout code
        uses: actions/checkout@abc # v6.0.2
        with:
          persist-credentials: false # Issue #1648
"#;
    assert_eq!(checkout_audit(hardened), (1, 0));

    // Negative: no persist setting at all.
    let bare = r#"    steps:
      - name: Checkout code
        uses: actions/checkout@abc # v6.0.2
"#;
    assert_eq!(checkout_audit(bare), (1, 1));

    // Negative: the setting sits in a different step's block.
    let other_step = r#"    steps:
      - name: Checkout code
        uses: actions/checkout@abc # v6.0.2
      - name: Other
        with:
          persist-credentials: false
"#;
    assert_eq!(checkout_audit(other_step), (1, 1));

    // Negative: a commented-out setting does not count.
    let commented = r#"    steps:
      - name: Checkout code
        uses: actions/checkout@abc # v6.0.2
        with:
          # persist-credentials: false
"#;
    assert_eq!(checkout_audit(commented), (1, 1));

    // Negative: a job with no checkout step reports zero checkouts.
    let no_checkout = r#"    steps:
      - name: Build
        run: cargo build
"#;
    assert_eq!(checkout_audit(no_checkout).0, 0);
}

#[test]
fn every_checkout_step_sets_persist_credentials_false() {
    let body = load_template();
    let (checkouts, unhardened) = checkout_audit(&body);
    assert!(
        checkouts > 0,
        "docs/ci-fuzzing-workflow.yml has no `actions/checkout` step to audit",
    );
    assert_eq!(
        unhardened, 0,
        "{unhardened} of {checkouts} `actions/checkout` step(s) in \
         docs/ci-fuzzing-workflow.yml lack `persist-credentials: false` in their \
         own block (Issues #1648, #1868, #2478)",
    );
}

#[test]
fn timeout_helper_requires_job_level_key_with_value() {
    let ok = r#"jobs:
  fuzzing:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: true
"#;
    let block = job_block(ok, "fuzzing").expect("fuzzing job");
    assert!(job_declares_timeout_minutes(&block));

    // Indented inside a step, so not a job-level key.
    let nested = r#"jobs:
  fuzzing:
    steps:
      - run: true
        timeout-minutes: 30
"#;
    let block = job_block(nested, "fuzzing").expect("fuzzing job");
    assert!(!job_declares_timeout_minutes(&block));

    // Commented-out directive does not count.
    let commented = r#"jobs:
  fuzzing:
    runs-on: ubuntu-latest
    # timeout-minutes: 30
    steps:
      - run: true
"#;
    let block = job_block(commented, "fuzzing").expect("fuzzing job");
    assert!(!job_declares_timeout_minutes(&block));

    // Key present without a value.
    let empty = r#"jobs:
  fuzzing:
    timeout-minutes:
    steps:
      - run: true
"#;
    let block = job_block(empty, "fuzzing").expect("fuzzing job");
    assert!(!job_declares_timeout_minutes(&block));

    // Another job's timeout does not count towards `fuzzing`.
    let other_job = r#"jobs:
  fuzzing:
    runs-on: ubuntu-latest
    steps:
      - run: true
  other:
    timeout-minutes: 5
"#;
    let block = job_block(other_job, "fuzzing").expect("fuzzing job");
    assert!(!job_declares_timeout_minutes(&block));

    assert!(job_block(ok, "missing").is_none());
}

#[test]
fn fuzzing_job_declares_timeout_minutes() {
    let body = load_template();
    let block = job_block(&body, "fuzzing")
        .expect("job `fuzzing` not found in docs/ci-fuzzing-workflow.yml");
    assert!(
        job_declares_timeout_minutes(&block),
        "the `fuzzing` job in docs/ci-fuzzing-workflow.yml must declare \
         `timeout-minutes:` (Issues #1287, #2478)",
    );
}
