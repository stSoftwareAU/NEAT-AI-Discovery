//! Issue #2527: the fuzzing CI template `docs/ci-fuzzing-workflow.yml` ran only
//! the two FFI fuzz targets, so an operator who copied it never fuzzed the
//! parquet footer-statistics and pre-decode budget paths.
//!
//! Three lists name the fuzz targets and must agree: the `[[bin]]` entries in
//! `fuzz/Cargo.toml`, the `FUZZ_TARGETS` array in `scripts/fuzz-ci.sh`, and the
//! `cargo +nightly fuzz run` steps in the template. The parquet footer target
//! also needs its seed corpus and `-max_len`, as `fuzz-ci.sh` gives it.
//!
//! The files are read as text (no YAML or TOML parser is in the dependency
//! tree). Comment lines are ignored so a commented-out step cannot satisfy a
//! check, and `\` continuations are joined so a re-wrapped command still counts.
//! Each extractor is exercised against inline positive and negative text.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn read_repo_file(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Lines with `\` continuations joined and comment-only lines dropped.
fn logical_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut pending = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if pending.is_empty() && line.starts_with('#') {
            continue;
        }
        if let Some(stripped) = line.strip_suffix('\\') {
            pending.push_str(stripped.trim_end());
            pending.push(' ');
        } else {
            pending.push_str(line);
            lines.push(std::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        lines.push(pending);
    }
    lines
}

/// The `name = "…"` values of every `[[bin]]` table in a Cargo manifest.
fn bin_names(manifest: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_bin = false;
    for line in logical_lines(manifest) {
        if line.starts_with('[') {
            in_bin = line == "[[bin]]";
        } else if in_bin
            && let Some(value) = line.strip_prefix("name")
            && let Some(value) = value.trim_start().strip_prefix('=')
        {
            names.insert(value.trim().trim_matches('"').to_string());
        }
    }
    names
}

/// The target name of a `cargo … fuzz run [flags] <target> …` command line, or
/// `None` when the line is not one.
fn fuzz_run_target(line: &str) -> Option<String> {
    let (_, after) = line.split_once("fuzz run ")?;
    after
        .split_whitespace()
        .take_while(|token| *token != "--")
        .find(|token| !token.starts_with('-'))
        .map(str::to_string)
}

/// Every target a `cargo … fuzz run` command in `text` names, paired with its
/// full command line.
fn fuzz_run_commands(text: &str) -> Vec<(String, String)> {
    logical_lines(text)
        .into_iter()
        .filter_map(|line| fuzz_run_target(&line).map(|target| (target, line)))
        .collect()
}

/// The quoted entries of the `FUZZ_TARGETS=( … )` array in a shell script.
fn script_targets(script: &str) -> BTreeSet<String> {
    let mut targets = BTreeSet::new();
    let mut inside = false;
    for line in logical_lines(script) {
        if line.starts_with("FUZZ_TARGETS=(") {
            inside = true;
        } else if inside && line == ")" {
            break;
        } else if inside {
            targets.insert(line.trim_matches('"').to_string());
        }
    }
    targets
}

fn set_of(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn bin_names_reads_only_bin_tables() {
    let manifest = "[package]\nname = \"pkg\"\n\n[[bin]]\nname = \"one\"\npath = \"a.rs\"\n\n\
                    [[bin]]\nname = \"two\"\n# name = \"commented\"\n\n[dependencies]\nname = \"x\"\n";
    assert_eq!(bin_names(manifest), set_of(&["one", "two"]));
    assert!(bin_names("[package]\nname = \"pkg\"\n").is_empty());
}

#[test]
fn fuzz_run_commands_find_targets_across_variants() {
    let text = "steps:\n\
        - run: cargo +nightly fuzz run --locked alpha -- -max_total_time=30\n\
        - run: |\n\
        \x20   cargo +nightly fuzz run \\\n\
        \x20     --locked beta -- -max_total_time=30\n\
        # cargo +nightly fuzz run --locked commented -- -max_total_time=30\n\
        - run: cargo +nightly fuzz build --locked not_a_run\n";
    let targets: Vec<String> = fuzz_run_commands(text)
        .into_iter()
        .map(|(target, _)| target)
        .collect();
    assert_eq!(targets, vec!["alpha".to_string(), "beta".to_string()]);
}

#[test]
fn script_targets_reads_the_array_only() {
    let script = "X=1\nFUZZ_TARGETS=(\n    \"one\"\n    \"two\"\n)\nOTHER=(\n    \"three\"\n)\n";
    assert_eq!(script_targets(script), set_of(&["one", "two"]));
    assert!(script_targets("echo hi\n").is_empty());
}

#[test]
fn fuzz_ci_script_runs_every_declared_fuzz_target() {
    let declared = bin_names(&read_repo_file("fuzz/Cargo.toml"));
    assert!(!declared.is_empty(), "fuzz/Cargo.toml declares no [[bin]]");
    assert_eq!(
        script_targets(&read_repo_file("scripts/fuzz-ci.sh")),
        declared,
        "scripts/fuzz-ci.sh FUZZ_TARGETS must list exactly the [[bin]] targets in fuzz/Cargo.toml",
    );
}

#[test]
fn ci_template_runs_every_declared_fuzz_target() {
    let declared = bin_names(&read_repo_file("fuzz/Cargo.toml"));
    let template = read_repo_file("docs/ci-fuzzing-workflow.yml");
    let run: BTreeSet<String> = fuzz_run_commands(&template)
        .into_iter()
        .map(|(target, _)| target)
        .collect();
    assert_eq!(
        run, declared,
        "docs/ci-fuzzing-workflow.yml must run exactly the [[bin]] targets in \
         fuzz/Cargo.toml (Issue #2527)",
    );
}

#[test]
fn ci_template_gives_the_footer_target_its_seeds_and_max_len() {
    let script = read_repo_file("scripts/fuzz-ci.sh");
    let template = read_repo_file("docs/ci-fuzzing-workflow.yml");
    let (_, command) = fuzz_run_commands(&template)
        .into_iter()
        .find(|(target, _)| target == "fuzz_parquet_footer_stats")
        .expect("template must run fuzz_parquet_footer_stats (Issue #2527)");
    // Derive the seed directory and length cap from the script that owns them.
    for needle in ["fuzz/seeds/parquet_footer_stats", "-max_len=65536"] {
        assert!(
            script.contains(needle),
            "sanity: scripts/fuzz-ci.sh no longer carries `{needle}`"
        );
        assert!(
            command.contains(needle),
            "the template's footer-stats step must pass `{needle}` as fuzz-ci.sh does \
             (Issue #2527): {command}"
        );
    }
}
