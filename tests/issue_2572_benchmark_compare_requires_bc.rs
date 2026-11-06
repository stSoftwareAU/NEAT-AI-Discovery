//! Issue #2572: `benchmark_compare.sh` must not give a regression verdict when
//! `bc` is missing.
//!
//! The comparison arithmetic is `$(echo ... | bc -l)`. On a host without `bc` that
//! command substitution fails, the `(( ))` test sees an empty operand, and every
//! measured change fell through to the "within threshold" branch. The script then
//! printed "No regressions detected." and exited 0, so a missing tool looked like a
//! clean benchmark run. The script must check for `bc` before it runs any suite
//! and fail with a clear message instead.
//!
//! These tests run the real script in a sandbox whose `PATH` holds only a stub
//! `cargo`, optionally a stub `bc`, and symlinks to the handful of host tools the
//! script needs. `bc` is never linked from the host, so the result does not depend
//! on whether the machine running the tests has `bc` installed.
//!
//! The stub `cargo` reports a change within the threshold and the stub `bc`
//! answers `0`. The positive control therefore reaches a verdict once `bc` is
//! added, which shows the missing-tool failure comes from `bc` alone.

use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Host tools the script uses. `bc` is deliberately absent from this list. `echo`
/// is needed because `xargs` runs it as an external command when the script
/// trims a benchmark name in a pipeline.
const HOST_TOOLS: &[&str] = &[
    "dirname", "grep", "sed", "xargs", "tail", "mktemp", "tee", "rm", "echo",
];

/// Write an executable stub script into `dir`.
fn write_stub(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, body).expect("write stub");
    let mut perms = fs::metadata(&path).expect("stat stub").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).expect("chmod stub");
}

/// Stub `bc`: drains its input with a bash builtin, so it does not depend on any
/// host tool, then prints `0`. Every comparison therefore answers "not above the
/// threshold" and "not below minus the threshold", so a stubbed measurement is
/// counted as within threshold.
const BC_STUB: &str = "#!/bin/bash\nwhile IFS= read -r _line; do :; done\necho '0'\n";

/// Stub `cargo`. It prints one Criterion-style comparison showing a change of
/// about 0.1%, which is inside the default 5% threshold. The positive control
/// relies on this line reaching `bc` and being counted as one verdict.
const CARGO_STUB: &str = r#"#!/bin/bash
echo "suite_a  time:   [1 µs 1 µs 1 µs]"
echo "                    change: [-0.1000% +0.0000% +0.1000%] (p = 0.50 > 0.05)"
echo "                    No change in performance detected."
exit 0
"#;

/// `Cargo.toml` with one suite. The script discovers suites by reading
/// `./Cargo.toml` from the working directory.
const CARGO_TOML: &str = "\
[package]
name = \"sandbox\"
version = \"0.0.0\"
edition = \"2021\"

[[bench]]
name = \"suite_a\"
";

/// Symlink each host tool in `HOST_TOOLS` that exists under `/usr/bin` or `/bin`
/// into `bin`. `bc` is never linked.
fn link_host_tools(bin: &Path) {
    for tool in HOST_TOOLS {
        let found = ["/usr/bin", "/bin"]
            .iter()
            .map(|dir| Path::new(dir).join(tool))
            .find(|candidate| candidate.exists());
        if let Some(source) = found {
            symlink(&source, bin.join(tool)).expect("symlink host tool into sandbox bin");
        }
    }
}

/// A prepared sandbox holding a copy of the real `benchmark_compare.sh`, its
/// threshold helper, a one-suite `Cargo.toml`, and a `target/criterion/` directory
/// so comparison mode finds a baseline.
struct Sandbox {
    _tmp: tempfile::TempDir,
    bin: PathBuf,
    work: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("temp dir");
        let bin = tmp.path().join("bin");
        let work = tmp.path().join("work");
        fs::create_dir_all(&bin).expect("create stub bin dir");
        fs::create_dir_all(work.join("scripts")).expect("create work/scripts");
        fs::create_dir_all(work.join("target").join("criterion")).expect("create baseline dir");

        fs::copy(
            repo_root().join("benchmark_compare.sh"),
            work.join("benchmark_compare.sh"),
        )
        .expect("copy benchmark_compare.sh into the sandbox");
        fs::copy(
            repo_root().join("scripts").join("benchmark_threshold.sh"),
            work.join("scripts").join("benchmark_threshold.sh"),
        )
        .expect("copy benchmark_threshold.sh into the sandbox");
        fs::write(work.join("Cargo.toml"), CARGO_TOML).expect("write sandbox Cargo.toml");

        write_stub(&bin, "cargo", CARGO_STUB);
        link_host_tools(&bin);

        Self {
            _tmp: tmp,
            bin,
            work,
        }
    }

    /// Add a stub `bc` to the sandbox `PATH`.
    fn with_bc(self) -> Self {
        write_stub(&self.bin, "bc", BC_STUB);
        self
    }

    fn has_bc(&self) -> bool {
        self.bin.join("bc").exists()
    }

    /// Run the script with `args`. `PATH` holds only the sandbox bin directory,
    /// so the host `bc` is never found.
    fn run(&self, args: &[&str]) -> Output {
        Command::new("/bin/bash")
            .arg("benchmark_compare.sh")
            .args(args)
            .current_dir(&self.work)
            .env("PATH", &self.bin)
            // HOME points at the sandbox so the run cannot read or write the
            // developer's own environment.
            .env("HOME", &self.work)
            .stdin(Stdio::null())
            .output()
            .expect("run benchmark_compare.sh")
    }
}

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn compare_without_bc_fails_loudly_and_gives_no_clean_verdict() {
    let sandbox = Sandbox::new();
    assert!(
        !sandbox.has_bc(),
        "the sandbox must not contain bc for this test (Issue #2572)"
    );

    let output = sandbox.run(&[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !output.status.success(),
        "benchmark_compare.sh must exit non-zero when bc is not installed — a comparison \
         without bc gives no regression verdict (Issue #2572)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("bc is not installed"),
        "the missing tool must be named in a clear error (Issue #2572) — stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("No regressions detected"),
        "no clean verdict may be printed when bc is missing (Issue #2572) — stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("Within threshold"),
        "no threshold summary may be printed when bc is missing (Issue #2572) — stdout:\n{stdout}"
    );
}

#[test]
fn compare_without_bc_stops_before_running_any_suite() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        !stdout.contains("Comparing:"),
        "bc must be checked before any suite runs, so no suite is compared (Issue #2572) \
         — stdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

#[test]
fn compare_with_bc_reaches_a_verdict() {
    // Positive control: once bc is added, the same stubbed measurement reaches a
    // verdict (one comparison, within threshold). The failures above therefore
    // come from the missing tool alone.
    let output = Sandbox::new().with_bc().run(&[]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);

    assert!(
        output.status.success(),
        "a within-threshold comparison must exit successfully when bc is present (Issue #2572)\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Total comparisons: 1"),
        "the one stubbed comparison must be counted when bc is present (Issue #2572) \
         — stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Within threshold:  1"),
        "the stubbed comparison must be reported as within threshold when bc is present \
         (Issue #2572) — stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("bc is not installed"),
        "a present bc must not trigger the missing-tool error (Issue #2572) — stderr:\n{stderr}"
    );
}

#[test]
fn save_baseline_and_list_do_not_need_bc() {
    // Save mode and list mode never compare, so they must keep working without
    // bc. Each run gets a fresh sandbox.
    let save = Sandbox::new().run(&["--save-baseline"]);
    let save_stdout = text(&save.stdout);
    let save_stderr = text(&save.stderr);
    assert!(
        save.status.success(),
        "--save-baseline must not need bc (Issue #2572)\nstdout:\n{save_stdout}\nstderr:\n{save_stderr}"
    );
    assert!(
        save_stdout.contains("Baseline 'saved' saved for 1 suites"),
        "--save-baseline must still report the saved baseline without bc (Issue #2572) \
         — stdout:\n{save_stdout}\nstderr:\n{save_stderr}"
    );

    let list = Sandbox::new().run(&["--list"]);
    let list_stdout = text(&list.stdout);
    let list_stderr = text(&list.stderr);
    assert!(
        list.status.success(),
        "--list must not need bc (Issue #2572)\nstdout:\n{list_stdout}\nstderr:\n{list_stderr}"
    );
    assert!(
        list_stdout.contains("suite_a"),
        "--list must still name the suite without bc (Issue #2572) — stdout:\n{list_stdout}"
    );
}
