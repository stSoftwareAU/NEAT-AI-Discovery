//! Issue #2139: `scripts/check-pr-summary-location.sh` must not report ✅ for a
//! scan that never ran.
//!
//! The guard used to collect strays through
//! `done < <(find docs … 2>/dev/null | sort)`. `set -euo pipefail` does not
//! check a process substitution's exit status, and `sort` succeeds on empty
//! input, so a `find` that could not scan — a missing `docs/`, an unreadable
//! subtree — was indistinguishable from a `find` that found nothing: the script
//! printed the ✅ line and exited `0`. Since `quality.sh` runs this as a
//! pre-commit gate, the masked failure was reported to the contributor as a
//! clean check.
//!
//! These tests drive the real script in a `tempfile` sandbox holding nothing but
//! a copy of the script, so the tree under it is entirely controlled here.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const TICK: &str = "✅";

/// A sandbox containing only `scripts/check-pr-summary-location.sh`; the script
/// resolves its repo root as the script's parent directory, so the sandbox root
/// is the tree it scans.
struct Sandbox {
    _tmp: tempfile::TempDir,
    root: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("temp dir");
        let root = tmp.path().to_path_buf();
        fs::create_dir_all(root.join("scripts")).expect("create scripts dir");
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/check-pr-summary-location.sh"),
            root.join("scripts/check-pr-summary-location.sh"),
        )
        .expect("copy the guard into the sandbox");
        Self { _tmp: tmp, root }
    }

    /// Create `relative_path` (parents included) with placeholder content.
    fn write_file(&self, relative_path: &str) -> &Self {
        let path = self.root.join(relative_path);
        fs::create_dir_all(path.parent().expect("a parent directory")).expect("create parent dir");
        fs::write(&path, "# placeholder\n").expect("write sandbox file");
        self
    }

    fn run(&self) -> Output {
        Command::new("bash")
            .arg(self.root.join("scripts/check-pr-summary-location.sh"))
            // Run from elsewhere so only the script's own root resolution decides
            // which tree is scanned.
            .current_dir("/")
            .stdin(Stdio::null())
            .output()
            .expect("run the PR summary location guard")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn a_scan_that_cannot_run_fails_loud_instead_of_reporting_a_clean_tree() {
    // No `docs/` at all: `find docs …` cannot scan, so the guard has checked
    // nothing and must say so.
    let output = Sandbox::new().run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "the guard must exit non-zero when its scan cannot run — a check that \
         never happened is not a pass (Issue #2139)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("find"),
        "the failure must name the failed `find` scan on stderr (Issue #2139) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(TICK),
        "no ✅ line may be printed for a scan that never ran (Issue #2139) — stdout:\n{stdout}"
    );
}

#[test]
fn a_canonical_tree_still_passes() {
    let sandbox = Sandbox::new();
    sandbox
        .write_file("docs/archive/pr-summaries/README.md")
        .write_file("docs/archive/pr-summaries/pr-summary-1613.md")
        .write_file("docs/archive/pr-summaries/pr-summary-2139.md")
        .write_file("docs/CONFIGURATION.md");
    let output = sandbox.run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "a tree whose summaries all live in the canonical directory must pass \
         (Issue #2139)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains(TICK) && stdout.contains("docs/archive/pr-summaries"),
        "the passing run must still print the ✅ line naming the canonical \
         directory (Issue #2139) — stdout:\n{stdout}"
    );
}

#[test]
fn a_stray_summary_is_still_listed_and_fails_the_gate() {
    let sandbox = Sandbox::new();
    sandbox
        .write_file("docs/archive/pr-summaries/pr-summary-1613.md")
        .write_file("docs/pr-summary-99.md");
    let output = sandbox.run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "a stray summary must still exit 1 (Issue #2139)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("docs/pr-summary-99.md"),
        "the stray file must be listed by path (Issue #2139) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains(TICK),
        "no ✅ line may accompany a failed check (Issue #2139) — stdout:\n{stdout}"
    );
}

#[test]
fn a_stray_summary_under_a_path_containing_spaces_is_listed_intact() {
    let sandbox = Sandbox::new();
    sandbox.write_file("docs/old notes/pr-summary-100.md");
    let output = sandbox.run();
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(1),
        "a stray summary under a path with spaces must fail the gate \
         (Issue #2139)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("docs/old notes/pr-summary-100.md"),
        "the stray path must be reported whole, not split on its spaces \
         (Issue #2139) — stdout:\n{stdout}"
    );
}
