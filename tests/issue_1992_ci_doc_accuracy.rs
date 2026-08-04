//! Issue #1992 — CONTRIBUTING's CI description must match the installed
//! workflows and the committed `quality.sh`.
//!
//! Every assertion below derives the expected text from the artefact it
//! describes (`.github/workflows/*.yml`, `quality.sh`) rather than hard-coding
//! it, so the guard survives unrelated edits to those files and fails loudly
//! the next time the prose drifts.

const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
const CI: &str = include_str!("../.github/workflows/ci.yml");
const QUALITY_SH: &str = include_str!("../quality.sh");

// ============================================================================
// Helpers — parse the artefacts the prose claims to describe.
// ============================================================================

/// The body of a top-level job in `ci.yml` (two-space indented key).
fn job_block(job: &str) -> String {
    let marker = format!("  {job}:");
    let mut body = Vec::new();
    let mut inside = false;
    for line in CI.lines() {
        if line.trim_end() == marker {
            inside = true;
            continue;
        }
        if inside {
            // A new top-level job key: exactly two spaces of indentation.
            let starts_new_job = line.starts_with("  ")
                && !line.starts_with("   ")
                && line.trim_end().ends_with(':');
            if starts_new_job {
                break;
            }
            body.push(line);
        }
    }
    assert!(!body.is_empty(), "ci.yml must define a `{job}` job");
    body.join("\n")
}

/// The ordered shell commands `quality.sh` actually runs.
fn quality_sh_commands() -> Vec<String> {
    QUALITY_SH
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let command = line
                .strip_prefix("RUSTDOCFLAGS=\"-D warnings\" ")
                .unwrap_or(line);
            if command.starts_with("./quality/")
                || command.starts_with("./scripts/")
                || command.starts_with("cargo ")
            {
                // Drop the trailing `.` path argument the gate scripts take —
                // the prose cites the script, not its argument.
                Some(command.trim_end_matches(" .").to_string())
            } else {
                None
            }
        })
        .collect()
}

/// The numbered items of CONTRIBUTING's ordered quality-gate list.
fn contributing_gate_items() -> Vec<String> {
    const MARKER: &str = "performs these checks in order:";
    let start = CONTRIBUTING
        .lines()
        .position(|line| line.contains(MARKER))
        .expect("CONTRIBUTING must introduce the ordered quality-gate list");

    let mut items: Vec<String> = Vec::new();
    for line in CONTRIBUTING.lines().skip(start + 1) {
        if line.trim().is_empty() {
            if items.is_empty() {
                continue; // blank line between the intro and the list
            }
            break; // blank line after the list ends it
        }
        let numbered = line
            .split_once(". ")
            .filter(|(number, _)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()));
        match numbered {
            Some((_, rest)) => items.push(rest.to_string()),
            None => match items.last_mut() {
                // A wrapped continuation line of the current item.
                Some(current) => {
                    current.push(' ');
                    current.push_str(line.trim());
                }
                None => break,
            },
        }
    }
    assert!(!items.is_empty(), "the ordered gate list must not be empty");
    items
}

/// A CONTRIBUTING bullet (including its wrapped continuation lines).
fn contributing_bullet(prefix: &str) -> String {
    let lines: Vec<&str> = CONTRIBUTING.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("CONTRIBUTING must carry a bullet starting `{prefix}`"));

    let mut bullet = lines[start].to_string();
    for line in lines.iter().skip(start + 1) {
        if line.trim().is_empty() || line.starts_with("- ") {
            break;
        }
        bullet.push(' ');
        bullet.push_str(line.trim());
    }
    bullet
}

// ============================================================================
// 1. The `quality` job bullet must describe the job's real steps.
// ============================================================================

#[test]
fn quality_job_bullet_claims_only_steps_ci_runs() {
    let quality = job_block("quality");
    let bullet = contributing_bullet("- `quality` —");
    // Only the bullet's opening sentence enumerates the job's steps; the prose
    // that follows may legitimately name what CI does *not* run.
    let step_list = bullet
        .split_once(". ")
        .map_or(bullet.as_str(), |(head, _)| head);

    if !quality.contains("cargo check") {
        assert!(
            !step_list.contains("cargo check"),
            "the `quality` job runs no `cargo check`, so CONTRIBUTING must not list one \
             (Issue #1992): {step_list}"
        );
    }
    if !quality.contains("cargo doc") {
        assert!(
            !step_list.to_lowercase().contains("doc build"),
            "the `quality` job builds no docs, so CONTRIBUTING must not list a doc build \
             (Issue #1992): {step_list}"
        );
    }

    assert!(
        quality.contains("cargo build --lib"),
        "sanity: the `quality` job builds the library"
    );
    assert!(
        bullet.contains("library build"),
        "CONTRIBUTING must describe the `cargo build --lib` step as a library build \
         (Issue #1992): {bullet}"
    );
    assert!(
        bullet.contains("tests"),
        "CONTRIBUTING must keep describing the test step (Issue #1992): {bullet}"
    );
}

// ============================================================================
// 2. The ordered gate list must match `quality.sh` step for step.
// ============================================================================

#[test]
fn ordered_gate_list_matches_quality_sh() {
    let commands = quality_sh_commands();
    let items = contributing_gate_items();

    assert_eq!(
        items.len(),
        commands.len(),
        "CONTRIBUTING lists {} gate steps but quality.sh runs {} (Issue #1992):\n\
         documented: {items:#?}\nactual: {commands:#?}",
        items.len(),
        commands.len()
    );

    for (index, (item, command)) in items.iter().zip(commands.iter()).enumerate() {
        assert!(
            item.contains(command),
            "gate list item {} must describe `{command}` (Issue #1992), got: {item}",
            index + 1
        );
    }
}

#[test]
fn ordered_gate_list_is_numbered_sequentially() {
    const MARKER: &str = "performs these checks in order:";
    let start = CONTRIBUTING
        .lines()
        .position(|line| line.contains(MARKER))
        .expect("CONTRIBUTING must introduce the ordered quality-gate list");

    let mut expected = 1usize;
    for line in CONTRIBUTING.lines().skip(start + 1) {
        if line.trim().is_empty() && expected > 1 {
            break;
        }
        let numbered = line
            .split_once(". ")
            .filter(|(number, _)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()));
        if let Some((number, _)) = numbered {
            assert_eq!(
                number.parse::<usize>().expect("digits parse"),
                expected,
                "the ordered gate list must be numbered sequentially (Issue #1992)"
            );
            expected += 1;
        }
    }
    assert!(expected > 1, "the ordered gate list must not be empty");
}

// ============================================================================
// 3. Every PR-triggered workflow must appear in the CI pipeline section.
// ============================================================================

#[test]
fn every_pull_request_workflow_is_documented() {
    let workflows = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut checked = 0usize;

    for entry in std::fs::read_dir(&workflows).expect("workflow directory must exist") {
        let path = entry.expect("readable directory entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("yml") {
            continue;
        }
        let body = std::fs::read_to_string(&path).expect("readable workflow");
        if !body.contains("pull_request:") {
            continue; // reusable workflows are documented through their caller
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("workflow file name")
            .to_string();
        assert!(
            CONTRIBUTING.contains(&name),
            "CONTRIBUTING's CI pipeline section must document the PR-triggered workflow \
             `{name}` (Issue #1992)"
        );
        checked += 1;
    }

    assert!(
        checked >= 5,
        "expected several PR-triggered workflows, saw {checked}"
    );
}

// ============================================================================
// 4. The `validation` job bullet must cover its documentation check.
// ============================================================================

#[test]
fn validation_bullet_covers_the_documentation_step() {
    let validation = job_block("validation");
    assert!(
        validation.contains("- name: Check documentation"),
        "sanity: the `validation` job checks documentation"
    );

    let bullet = contributing_bullet("- `validation` —");
    assert!(
        bullet.to_lowercase().contains("documentation"),
        "CONTRIBUTING must mention the `Check documentation` step of the `validation` job \
         (Issue #1992): {bullet}"
    );
}

// ============================================================================
// 5. The documented test command must account for CI's extra flags.
// ============================================================================

#[test]
fn documented_test_command_records_the_ci_delta() {
    let ci_command = job_block("quality")
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("run: cargo test"))
        .map(|line| line.trim_start_matches("run: ").to_string())
        .expect("the `quality` job must run cargo test");

    let local_command = quality_sh_commands()
        .into_iter()
        .find(|command| command.starts_with("cargo test"))
        .expect("quality.sh must run cargo test");

    if ci_command != local_command {
        assert!(
            CONTRIBUTING.contains(&ci_command),
            "CI runs `{ci_command}` but quality.sh runs `{local_command}`; CONTRIBUTING must \
             record the delta verbatim (Issue #1992)"
        );
    }
    assert!(
        CONTRIBUTING.contains(&local_command),
        "CONTRIBUTING must document the local command `{local_command}` (Issue #1992)"
    );
}
