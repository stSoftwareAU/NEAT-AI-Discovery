//! Issue #2560: the `actionlint`, `markdownlint` and `shellcheck` jobs must
//! always run, so their required check names always report, but they must
//! gate their tool setup and lint on `quality/lint_scope.sh`.
//!
//! For each (workflow, job, scope) this test asserts:
//!
//! - the job still exists under its name;
//! - the `on:` block has no `paths:` or `paths-ignore:` key (a workflow-level
//!   filter would leave the required check pending);
//! - exactly one step has `id: scope`, it runs `./quality/lint_scope.sh <scope>`
//!   and it comes after the `actions/checkout` step;
//! - the checkout step sets `fetch-depth: 2`, so HEAD^1 exists for the diff;
//! - every step after the scope step carries
//!   `if: steps.scope.outputs.run == 'true'`.
//!
//! The workflows are read as plain text (no YAML parser is in the dependency
//! tree, matching the sibling workflow tests). The checks take the workflow
//! text, so the negative tests feed them mutated copies and assert the exact
//! problem each mutation produces.

use std::fs;
use std::path::{Path, PathBuf};

const GATE_IF: &str = "if: steps.scope.outputs.run == 'true'";

struct GatedJob {
    file: &'static str,
    job: &'static str,
    scope: &'static str,
    /// Number of steps after the scope step, all of which must be gated.
    gated_steps: usize,
    /// The first line of the final step, used to name it in messages.
    last_step: &'static str,
}

const GATED_JOBS: [GatedJob; 3] = [
    GatedJob {
        file: "actionlint.yml",
        job: "actionlint",
        scope: "actionlint",
        gated_steps: 1,
        last_step: "- name: Run actionlint",
    },
    GatedJob {
        file: "markdown-lint.yml",
        job: "markdownlint",
        scope: "markdown",
        gated_steps: 3,
        last_step: "- name: Run markdownlint-cli2",
    },
    GatedJob {
        file: "shellcheck.yml",
        job: "shellcheck",
        scope: "shellcheck",
        gated_steps: 3,
        last_step: "- name: Run ShellCheck (koalaman/shellcheck)",
    },
];

fn workflow_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".github/workflows")
        .join(file)
}

fn read_workflow(file: &str) -> String {
    let path = workflow_path(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// True for a two-space-indented job key such as `  shellcheck:`.
fn is_job_key(line: &str) -> bool {
    line.starts_with("  ")
        && !line.starts_with("   ")
        && !line.trim_start().starts_with('#')
        && line.trim_end().ends_with(':')
}

/// Lines of the `<job>:` block under the top-level `jobs:` key, up to the next
/// job key or top-level key. `None` when the job is absent.
fn job_lines<'a>(body: &'a str, job: &str) -> Option<Vec<&'a str>> {
    let mut lines = body.lines();
    lines.find(|line| line.trim_end() == "jobs:")?;
    let header = format!("  {job}:");
    lines.find(|line| line.trim_end() == header)?;
    let mut block = Vec::new();
    for line in lines {
        let top_level = !line.is_empty() && !line.starts_with(' ') && !line.starts_with('#');
        if top_level || is_job_key(line) {
            break;
        }
        block.push(line);
    }
    Some(block)
}

/// Splits a job's `steps:` list into one chunk per step. A step starts at a
/// `      - ` line and runs to the next one, so comments between steps belong to
/// the step above them.
fn step_chunks<'a>(job: &[&'a str]) -> Vec<Vec<&'a str>> {
    let Some(steps_at) = job.iter().position(|line| line.trim_end() == "    steps:") else {
        return Vec::new();
    };
    let mut chunks: Vec<Vec<&str>> = Vec::new();
    for line in job[steps_at + 1..].iter().copied() {
        if line.starts_with("      - ") {
            chunks.push(vec![line]);
        } else if let Some(chunk) = chunks.last_mut() {
            chunk.push(line);
        }
    }
    chunks
}

fn has_line(chunk: &[&str], wanted: &str) -> bool {
    chunk.iter().any(|line| line.trim() == wanted)
}

fn is_checkout(chunk: &[&str]) -> bool {
    chunk.iter().any(|line| {
        line.trim()
            .trim_start_matches("- ")
            .starts_with("uses: actions/checkout@")
    })
}

fn run_command(chunk: &[&str]) -> Option<String> {
    chunk.iter().find_map(|line| {
        line.trim()
            .strip_prefix("run:")
            .map(|command| command.trim().to_string())
    })
}

fn step_label(chunk: &[&str]) -> String {
    chunk[0].trim().to_string()
}

/// Path filter keys declared inside the top-level `on:` block.
fn on_block_path_filters(body: &str) -> Vec<&'static str> {
    let mut filters = Vec::new();
    let mut in_on = false;
    for line in body.lines() {
        if line.trim_end() == "on:" {
            in_on = true;
            continue;
        }
        if in_on && !line.is_empty() && !line.starts_with(' ') && !line.starts_with('#') {
            break;
        }
        if in_on {
            let trimmed = line.trim();
            if trimmed.starts_with("paths-ignore:") {
                filters.push("paths-ignore");
            } else if trimmed.starts_with("paths:") {
                filters.push("paths");
            }
        }
    }
    filters
}

/// Every way the gate is wrong for one job, as human-readable problems. An
/// empty result means the workflow text satisfies the gate.
fn gate_problems(file: &str, body: &str, job: &str, scope: &str) -> Vec<String> {
    let mut problems: Vec<String> = on_block_path_filters(body)
        .into_iter()
        .map(|key| {
            format!(
                "{file}: the `on:` block declares a `{key}:` filter; a workflow-level path filter leaves the required `{job}` check pending (Issue #2560)"
            )
        })
        .collect();

    let Some(lines) = job_lines(body, job) else {
        problems.push(format!(
            "{file}: job `{job}` not found under `jobs:` (Issue #2560)"
        ));
        return problems;
    };
    let chunks = step_chunks(&lines);

    let scope_steps: Vec<usize> = (0..chunks.len())
        .filter(|&i| has_line(&chunks[i], "id: scope"))
        .collect();
    if scope_steps.is_empty() {
        problems.push(format!(
            "{file}: job `{job}` has no step with `id: scope` (Issue #2560)"
        ));
        return problems;
    }
    if scope_steps.len() > 1 {
        problems.push(format!(
            "{file}: job `{job}` has {} steps with `id: scope`, expected exactly one (Issue #2560)",
            scope_steps.len()
        ));
        return problems;
    }
    let scope_at = scope_steps[0];

    let expected_run = format!("./quality/lint_scope.sh {scope}");
    let actual_run = run_command(&chunks[scope_at]).unwrap_or_else(|| "<no run: line>".to_string());
    if actual_run != expected_run {
        problems.push(format!(
            "{file}: job `{job}` scope step runs `{actual_run}`, expected `{expected_run}` (Issue #2560)"
        ));
    }

    let Some(checkout_at) = chunks.iter().position(|chunk| is_checkout(chunk)) else {
        problems.push(format!(
            "{file}: job `{job}` has no `actions/checkout` step (Issue #2560)"
        ));
        return problems;
    };
    if scope_at <= checkout_at {
        problems.push(format!(
            "{file}: job `{job}` scope step must come after the checkout (Issue #2560)"
        ));
    }
    if !has_line(&chunks[checkout_at], "fetch-depth: 2") {
        problems.push(format!(
            "{file}: job `{job}` checkout step must set `fetch-depth: 2` so lint_scope.sh can diff HEAD^1 (Issue #2560)"
        ));
    }

    let after = &chunks[scope_at + 1..];
    let ungated: Vec<String> = after
        .iter()
        .filter(|chunk| !has_line(chunk, GATE_IF))
        .map(|chunk| format!("`{}`", step_label(chunk)))
        .collect();
    if !ungated.is_empty() {
        problems.push(format!(
            "{file}: job `{job}` has {} step(s) after the scope step but {} carry `{GATE_IF}`; ungated: {} (Issue #2560)",
            after.len(),
            after.len() - ungated.len(),
            ungated.join(", ")
        ));
    }
    problems
}

/// (steps after the scope step, of which carry the gate).
fn gate_counts(body: &str, job: &str) -> (usize, usize) {
    let lines = job_lines(body, job).expect("job must exist");
    let chunks = step_chunks(&lines);
    let scope_at = chunks
        .iter()
        .position(|chunk| has_line(chunk, "id: scope"))
        .expect("scope step must exist");
    let after = &chunks[scope_at + 1..];
    let gated = after
        .iter()
        .filter(|chunk| has_line(chunk, GATE_IF))
        .count();
    (after.len(), gated)
}

fn scope_run_command(body: &str, job: &str) -> Option<String> {
    let lines = job_lines(body, job)?;
    let chunks = step_chunks(&lines);
    let scope = chunks.iter().find(|chunk| has_line(chunk, "id: scope"))?;
    run_command(scope)
}

/// Adds a `paths:` (or `paths-ignore:`) key under `on.pull_request`, just
/// before its `branches:` key.
fn with_path_filter(body: &str, key: &str) -> String {
    let mutated = body.replacen(
        "    branches:",
        &format!("    {key}:\n      - \"**\"\n    branches:"),
        1,
    );
    assert_ne!(mutated, body, "the `branches:` anchor must exist");
    mutated
}

/// Drops the scope step: from its `- name:` line through its `run:` line.
fn without_scope_step(body: &str) -> String {
    let mut kept = Vec::new();
    let mut in_scope_step = false;
    for line in body.lines() {
        if line.trim() == "- name: Detect relevant changes" {
            in_scope_step = true;
        }
        if !in_scope_step {
            kept.push(line);
        }
        if in_scope_step && line.trim().starts_with("run: ./quality/lint_scope.sh") {
            in_scope_step = false;
        }
    }
    format!("{}\n", kept.join("\n"))
}

/// Removes the last `if: steps.scope.outputs.run == 'true'` line in the file.
fn without_last_gate(body: &str) -> String {
    let needle = format!("        {GATE_IF}\n");
    let at = body.rfind(&needle).expect("a gate line must be present");
    format!("{}{}", &body[..at], &body[at + needle.len()..])
}

fn assert_only_problem(problems: Vec<String>, expected: String) {
    assert_eq!(
        problems,
        vec![expected.clone()],
        "expected exactly one problem: {expected}"
    );
}

#[test]
fn gated_jobs_keep_their_names_and_pass_the_gate() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            job_lines(&body, g.job).is_some(),
            "job `{}` must still exist in {} (Issue #2560)",
            g.job,
            g.file
        );
        let problems = gate_problems(g.file, &body, g.job, g.scope);
        assert!(
            problems.is_empty(),
            "{} gate problems: {problems:#?}",
            g.file
        );
        assert_eq!(
            gate_counts(&body, g.job),
            (g.gated_steps, g.gated_steps),
            "{}: every step after the scope step must be gated",
            g.file
        );
    }
}

#[test]
fn paths_filters_under_on_block_are_rejected() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            gate_problems(g.file, &body, g.job, g.scope).is_empty(),
            "positive control: {} must pass before mutation",
            g.file
        );
        for key in ["paths", "paths-ignore"] {
            let mutated = with_path_filter(&body, key);
            assert_eq!(
                on_block_path_filters(&mutated),
                vec![key],
                "mutation must add `{key}:` under `on:`"
            );
            assert_only_problem(
                gate_problems(g.file, &mutated, g.job, g.scope),
                format!(
                    "{}: the `on:` block declares a `{key}:` filter; a workflow-level path filter leaves the required `{}` check pending (Issue #2560)",
                    g.file, g.job
                ),
            );
        }
    }
}

#[test]
fn missing_scope_step_is_rejected() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            gate_problems(g.file, &body, g.job, g.scope).is_empty(),
            "positive control: {} must pass before mutation",
            g.file
        );
        let mutated = without_scope_step(&body);
        assert!(
            !mutated.contains("id: scope"),
            "mutation must remove the scope step"
        );
        assert_only_problem(
            gate_problems(g.file, &mutated, g.job, g.scope),
            format!(
                "{}: job `{}` has no step with `id: scope` (Issue #2560)",
                g.file, g.job
            ),
        );
    }
}

#[test]
fn wrong_scope_name_is_rejected() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            gate_problems(g.file, &body, g.job, g.scope).is_empty(),
            "positive control: {} must pass before mutation",
            g.file
        );
        let right = format!("run: ./quality/lint_scope.sh {}", g.scope);
        let mutated = body.replace(&right, "run: ./quality/lint_scope.sh bogus");
        assert_ne!(mutated, body, "the scope run line must exist");
        assert_only_problem(
            gate_problems(g.file, &mutated, g.job, g.scope),
            format!(
                "{}: job `{}` scope step runs `./quality/lint_scope.sh bogus`, expected `./quality/lint_scope.sh {}` (Issue #2560)",
                g.file, g.job, g.scope
            ),
        );
    }
}

#[test]
fn gate_removed_from_a_later_step_is_rejected() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            gate_problems(g.file, &body, g.job, g.scope).is_empty(),
            "positive control: {} must pass before mutation",
            g.file
        );
        let mutated = without_last_gate(&body);
        assert_only_problem(
            gate_problems(g.file, &mutated, g.job, g.scope),
            format!(
                "{}: job `{}` has {} step(s) after the scope step but {} carry `{GATE_IF}`; ungated: `{}` (Issue #2560)",
                g.file,
                g.job,
                g.gated_steps,
                g.gated_steps - 1,
                g.last_step
            ),
        );
    }
}

#[test]
fn fetch_depth_removed_from_checkout_is_rejected() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        assert!(
            gate_problems(g.file, &body, g.job, g.scope).is_empty(),
            "positive control: {} must pass before mutation",
            g.file
        );
        let mutated = body.replace("          fetch-depth: 2\n", "");
        assert_ne!(mutated, body, "the `fetch-depth: 2` line must exist");
        assert_only_problem(
            gate_problems(g.file, &mutated, g.job, g.scope),
            format!(
                "{}: job `{}` checkout step must set `fetch-depth: 2` so lint_scope.sh can diff HEAD^1 (Issue #2560)",
                g.file, g.job
            ),
        );
    }
}

#[test]
fn lint_scope_script_exists_at_quality_path() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("quality/lint_scope.sh");
    assert!(
        script.exists(),
        "{} must exist: the gated lint workflows run it (Issue #2560)",
        script.display()
    );
}

#[test]
fn scope_script_referenced_by_each_gate_exists() {
    for g in &GATED_JOBS {
        let body = read_workflow(g.file);
        let command = scope_run_command(&body, g.job)
            .unwrap_or_else(|| panic!("{}: scope step has no run command", g.file));
        let script = command
            .split_whitespace()
            .next()
            .expect("run command must name a script");
        assert_eq!(
            script, "./quality/lint_scope.sh",
            "{}: scope step must run the committed gate script",
            g.file
        );
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(script.trim_start_matches("./"));
        assert!(
            path.exists(),
            "{}: referenced script {} does not exist (Issue #2560)",
            g.file,
            path.display()
        );
    }
}
