//! Behavioural tests for Issue #2560: gate the specialised lint workflows
//! (actionlint, markdown lint, `ShellCheck`) on relevant changes.
//!
//! `quality/lint_scope.sh` decides, from a change set, whether a lint workflow
//! has anything to check. Each test builds a throwaway git repository with a
//! base commit and a change commit, runs the real script over that range, and
//! asserts on its exit status and stdout. Nothing greps the script's source.
//!
//! The markdown rule mirrors the `ignores` list in `.markdownlint-cli2.jsonc`;
//! the drift pin at the end keeps the two in step.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SCOPES: [&str; 3] = ["actionlint", "markdown", "shellcheck"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn gate_script() -> PathBuf {
    repo_root().join("quality/lint_scope.sh")
}

/// A `bash quality/lint_scope.sh` invocation run from `cwd`. `GITHUB_OUTPUT` is
/// cleared so a CI environment cannot redirect the output into a real file.
fn script(cwd: &Path) -> Command {
    let mut cmd = Command::new("bash");
    cmd.arg(gate_script())
        .current_dir(cwd)
        .env_remove("GITHUB_OUTPUT");
    cmd
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("failed to run git");
    assert!(
        output.status.success(),
        "git {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn write_file(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create fixture directory");
    }
    fs::write(&path, body).expect("write fixture file");
}

/// Initialises a repository with a local identity, so commits never depend on
/// the host's git configuration.
fn init_repo(root: &Path) {
    git(root, &["init", "-q"]);
    git(root, &["config", "user.name", "Lint Scope Test"]);
    git(
        root,
        &["config", "user.email", "lint-scope@example.invalid"],
    );
    git(root, &["config", "commit.gpgsign", "false"]);
}

fn commit_all(root: &Path, message: &str) {
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", message]);
}

/// A throwaway repository with two commits. `base` and `head` are the commit ids
/// the script is given, so it never depends on the host checkout's history.
struct Fixture {
    dir: tempfile::TempDir,
    base: String,
    head: String,
}

impl Fixture {
    fn root(&self) -> &Path {
        self.dir.path()
    }
}

/// Builds a base commit holding `base_files`, then a change commit made by
/// `change`, which may add, edit, rename or delete files.
fn two_commit_repo(base_files: &[&str], change: impl FnOnce(&Path)) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    init_repo(&root);
    for rel in base_files {
        write_file(&root, rel, "base\n");
    }
    commit_all(&root, "base");
    let base = git(&root, &["rev-parse", "HEAD"]);
    change(&root);
    commit_all(&root, "change");
    let head = git(&root, &["rev-parse", "HEAD"]);
    Fixture { dir, base, head }
}

/// The shape every path probe shares: a base commit holding `src/lib.rs`, and a
/// change commit that adds or edits each of `changed`.
fn repo_changing(changed: &[&str]) -> Fixture {
    two_commit_repo(&["src/lib.rs"], |root| {
        for rel in changed {
            write_file(root, rel, "changed\n");
        }
    })
}

/// Runs the script for `scope` over the fixture's range and returns the decision.
/// Asserts that the script exited 0 and printed exactly one `run=` line.
fn decide(fixture: &Fixture, scope: &str) -> bool {
    let output = script(fixture.root())
        .args([scope, fixture.base.as_str(), fixture.head.as_str()])
        .output()
        .expect("failed to run quality/lint_scope.sh");
    assert!(
        output.status.success(),
        "{scope}: quality/lint_scope.sh must exit 0 (Issue #2560):\n{}",
        combined(&output)
    );
    match String::from_utf8_lossy(&output.stdout).as_ref() {
        "run=true\n" => true,
        "run=false\n" => false,
        other => panic!("{scope}: expected exactly one `run=` line, got {other:?}"),
    }
}

/// Asserts that a change to `path` makes `scope` run.
fn assert_runs(scope: &str, path: &str) {
    assert!(
        decide(&repo_changing(&[path]), scope),
        "{scope} must run for a change to `{path}` (Issue #2560)"
    );
}

/// Asserts that a change to `path` does not make `scope` run.
fn assert_skips(scope: &str, path: &str) {
    assert!(
        !decide(&repo_changing(&[path]), scope),
        "{scope} must skip a change to `{path}` (Issue #2560)"
    );
}

#[test]
fn rust_only_change_skips_all_three_lint_scopes() {
    let fixture = repo_changing(&["src/lib.rs"]);
    for scope in SCOPES {
        assert!(
            !decide(&fixture, scope),
            "{scope} must skip a Rust-only change (Issue #2560)"
        );
    }
    // Positive controls: the same shape with a relevant path runs each scope,
    // so the skips above cannot pass by the script erroring.
    assert_runs("actionlint", ".github/workflows/ci.yml");
    assert_runs("markdown", "docs/guide.md");
    assert_runs("shellcheck", "scripts/x.sh");
}

#[test]
fn actionlint_runs_for_workflow_files_but_not_for_dependabot_config() {
    assert_runs("actionlint", ".github/workflows/ci.yml");
    assert_skips("actionlint", ".github/dependabot.yml");
}

#[test]
fn markdown_runs_for_markdown_and_config_but_not_for_ignored_or_look_alike_paths() {
    assert_runs("markdown", "docs/guide.md");
    assert_runs("markdown", ".markdownlint-cli2.jsonc");
    assert_runs("markdown", ".markdownlintignore");
    assert_runs("markdown", "sub/.markdownlint-cli2.jsonc");
    assert_skips("markdown", "docs/archive/pr-summaries/pr-summary-1.md");
    assert_skips("markdown", "CHANGELOG.md");
    assert_skips("markdown", "sub/CHANGELOG.md");
    assert_skips("markdown", "notes.markdownish");
}

#[test]
fn shellcheck_runs_for_shell_scripts_and_config_but_not_for_look_alikes() {
    assert_runs("shellcheck", "scripts/x.sh");
    assert_runs("shellcheck", ".shellcheckrc");
    assert_skips("shellcheck", "scripts/x.txt");
}

#[test]
fn actionlint_workflow_enables_only_its_own_gate() {
    assert_runs("actionlint", ".github/workflows/actionlint.yml");
    assert_skips("markdown", ".github/workflows/actionlint.yml");
    assert_skips("shellcheck", ".github/workflows/actionlint.yml");
}

#[test]
fn markdown_lint_workflow_enables_its_own_gate() {
    assert_runs("markdown", ".github/workflows/markdown-lint.yml");
    assert_skips("shellcheck", ".github/workflows/markdown-lint.yml");
    // The actionlint rule covers every file under .github/workflows/, so this
    // change also runs actionlint by design.
    assert_runs("actionlint", ".github/workflows/markdown-lint.yml");
}

#[test]
fn shellcheck_workflow_enables_its_own_gate() {
    assert_runs("shellcheck", ".github/workflows/shellcheck.yml");
    assert_skips("markdown", ".github/workflows/shellcheck.yml");
    // As above, the actionlint rule covers every file under .github/workflows/.
    assert_runs("actionlint", ".github/workflows/shellcheck.yml");
}

#[test]
fn changing_the_gate_script_enables_every_lint_scope() {
    let fixture = repo_changing(&["quality/lint_scope.sh"]);
    for scope in SCOPES {
        assert!(
            decide(&fixture, scope),
            "{scope} must run when quality/lint_scope.sh changes (Issue #2560)"
        );
    }
}

#[test]
fn deleting_a_shell_script_still_enables_shellcheck() {
    let fixture = two_commit_repo(&["src/lib.rs", "scripts/old.sh"], |root| {
        fs::remove_file(root.join("scripts/old.sh")).expect("delete fixture script");
    });
    assert!(
        decide(&fixture, "shellcheck"),
        "a deleted shell script must still enable shellcheck (Issue #2560)"
    );
}

#[test]
fn renaming_a_markdown_file_enables_markdown_through_its_old_path() {
    let fixture = two_commit_repo(&["src/lib.rs", "a.md"], |root| {
        fs::rename(root.join("a.md"), root.join("a.txt")).expect("rename fixture file");
    });
    assert!(
        decide(&fixture, "markdown"),
        "renaming a.md away must enable markdown through its old path (Issue #2560)"
    );
    // The new path alone is not Markdown, so the old path is what matched.
    assert_skips("markdown", "a.txt");
}

#[test]
fn a_shell_script_with_spaces_in_its_path_enables_shellcheck() {
    assert_runs("shellcheck", "a dir/my script.sh");
}

#[test]
fn unreadable_base_ref_fails_closed_to_run() {
    // scripts/x.txt is irrelevant to every scope, so each `run=true` below can
    // only come from the fail-closed path.
    let fixture = repo_changing(&["scripts/x.txt"]);
    for scope in SCOPES {
        let output = script(fixture.root())
            .args([scope, "no-such-ref", fixture.head.as_str()])
            .output()
            .expect("failed to run quality/lint_scope.sh");
        assert!(
            output.status.success(),
            "{scope}: an unreadable base must exit 0 (Issue #2560):\n{}",
            combined(&output)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "run=true\n",
            "{scope}: an unreadable base must report run=true (Issue #2560)"
        );
        assert!(
            !output.stderr.is_empty(),
            "{scope}: fail-closed path must warn on stderr (Issue #2560)"
        );
    }
}

#[test]
fn default_range_in_a_single_commit_repo_fails_closed_to_run() {
    // A single commit has no HEAD^1, so the default range cannot be read.
    let dir = tempfile::tempdir().expect("tempdir");
    init_repo(dir.path());
    write_file(dir.path(), "src/lib.rs", "base\n");
    commit_all(dir.path(), "base");

    let output = script(dir.path())
        .arg("shellcheck")
        .output()
        .expect("failed to run quality/lint_scope.sh");
    assert!(
        output.status.success(),
        "default range without HEAD^1 must exit 0 (Issue #2560):\n{}",
        combined(&output)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "run=true\n",
        "default range without HEAD^1 must report run=true (Issue #2560)"
    );
    assert!(
        !output.stderr.is_empty(),
        "fail-closed path must warn on stderr (Issue #2560)"
    );
}

#[test]
fn default_range_on_merge_commit_selects_only_pr_changes() {
    for (pr_file, base_file, expected) in [
        ("scripts/pr.sh", "base.txt", "run=true\n"),
        ("src/lib.rs", "scripts/base.sh", "run=false\n"),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        init_repo(root);
        write_file(root, "initial.txt", "base\n");
        commit_all(root, "common base");
        git(root, &["checkout", "-q", "-b", "pr"]);
        write_file(root, pr_file, "PR change\n");
        commit_all(root, "PR change");
        git(root, &["checkout", "-q", "-b", "base", "HEAD^1"]);
        write_file(root, base_file, "base change\n");
        commit_all(root, "base change");
        git(root, &["merge", "-q", "--no-ff", "pr", "-m", "merge PR"]);

        let output = script(root)
            .arg("shellcheck")
            .output()
            .expect("failed to run quality/lint_scope.sh");
        assert!(output.status.success(), "{}", combined(&output));
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            expected,
            "default range must select PR change {pr_file}, excluding base change {base_file}"
        );
        assert!(
            !String::from_utf8_lossy(&output.stderr).contains("fail closed"),
            "{}",
            combined(&output)
        );
    }
}

#[test]
fn unknown_scope_and_bad_argument_counts_are_usage_errors() {
    let fixture = repo_changing(&["scripts/x.sh"]);
    let (base, head) = (fixture.base.as_str(), fixture.head.as_str());

    let unknown = script(fixture.root())
        .args(["yamllint", base, head])
        .output()
        .expect("failed to run quality/lint_scope.sh");
    assert_eq!(
        unknown.status.code(),
        Some(2),
        "an unknown scope must exit 2 (Issue #2560):\n{}",
        combined(&unknown)
    );
    assert!(
        unknown.stdout.is_empty(),
        "an unknown scope must print no run= line (Issue #2560)"
    );

    let none = script(fixture.root())
        .output()
        .expect("failed to run quality/lint_scope.sh");
    assert_eq!(
        none.status.code(),
        Some(2),
        "zero arguments must exit 2 (Issue #2560):\n{}",
        combined(&none)
    );
    assert!(
        none.stdout.is_empty(),
        "zero arguments must print no run= line (Issue #2560)"
    );

    let too_many = script(fixture.root())
        .args(["shellcheck", base, head, "extra"])
        .output()
        .expect("failed to run quality/lint_scope.sh");
    assert_eq!(
        too_many.status.code(),
        Some(2),
        "four arguments must exit 2 (Issue #2560):\n{}",
        combined(&too_many)
    );
    assert!(
        too_many.stdout.is_empty(),
        "four arguments must print no run= line (Issue #2560)"
    );
}

#[test]
fn github_output_file_gets_exactly_the_decision_line_appended() {
    let cases = [
        ("scripts/x.sh", "run=true\n"),
        ("src/lib.rs", "run=false\n"),
    ];
    for (changed, expected) in cases {
        let fixture = repo_changing(&[changed]);
        let out_file = fixture.root().join("github_output.txt");
        // Pre-seed the file: GitHub reads step outputs from an appended file,
        // so the script must append rather than truncate.
        fs::write(&out_file, "existing=1\n").expect("seed GITHUB_OUTPUT file");

        let output = script(fixture.root())
            .env("GITHUB_OUTPUT", &out_file)
            .args(["shellcheck", fixture.base.as_str(), fixture.head.as_str()])
            .output()
            .expect("failed to run quality/lint_scope.sh");
        assert!(
            output.status.success(),
            "shellcheck over `{changed}` must exit 0 (Issue #2560):\n{}",
            combined(&output)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            expected,
            "stdout for `{changed}` must be exactly the decision line (Issue #2560)"
        );
        let written = fs::read_to_string(&out_file).expect("read GITHUB_OUTPUT file");
        assert_eq!(
            written,
            format!("existing=1\n{expected}"),
            "GITHUB_OUTPUT must receive the same decision line appended (Issue #2560)"
        );
    }
}

/// `quality/lint_scope.sh` mirrors the markdownlint ignores for docs/archive and
/// CHANGELOG files. If the config stops ignoring them, the markdown rule must be
/// updated to match.
#[test]
fn markdown_rule_still_mirrors_the_markdownlint_ignores() {
    let config = repo_root().join(".markdownlint-cli2.jsonc");
    let body =
        fs::read_to_string(&config).unwrap_or_else(|e| panic!("read {}: {e}", config.display()));
    for ignore in ["docs/archive/**", "**/CHANGELOG.md"] {
        assert!(
            body.contains(ignore),
            "{} no longer ignores `{ignore}`, which quality/lint_scope.sh mirrors: \
             update its markdown rule to match (Issue #2560)",
            config.display()
        );
    }
}
