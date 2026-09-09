//! Issue #2055 — `scripts/runlib.sh` must honour `CARGO_HOME` when it extends
//! `PATH`.
//!
//! `_require_tools` only ever prepended `$HOME/.cargo/bin`. On a host whose
//! toolchain lives elsewhere (`CARGO_HOME=/…/.container-state/cargo`, no
//! `$HOME/.cargo`), the `rustup` that is installed — or that was already
//! present — is never resolvable, so the function dies with
//! "rustup installation appears incomplete" long before the caller's real work,
//! and `./quality.sh` cannot run at all.
//!
//! Both tests drive the real script with a sandboxed toolchain: a stub `rustup`
//! / `cargo` / `rustc` under the toolchain root, and a shim directory earlier on
//! `PATH` whose `rustup` always fails. If the script resolves its toolchain
//! correctly it reaches the missing-manifest abort; if it resolves the wrong
//! directory it finds the failing shim instead. No network, no real toolchain.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/runlib.sh"))
}

fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod");
    }
}

fn write_stub(dir: &Path, name: &str, body: &str) {
    fs::create_dir_all(dir).expect("create stub dir");
    let path = dir.join(name);
    fs::write(&path, format!("#!/usr/bin/env bash\n{body}\n")).expect("write stub");
    make_executable(&path);
}

/// A sandbox with a working toolchain at `toolchain_root/bin` and a shim
/// directory on `PATH` whose `rustup` always fails.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    /// `toolchain_root` is relative to the sandbox (e.g. `home/.cargo`).
    fn new(toolchain_root: &str) -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::create_dir_all(dir.path().join("home")).expect("create fake HOME");
        fs::create_dir_all(dir.path().join("elsewhere")).expect("create cwd without Cargo.toml");

        // The toolchain the script is expected to find.
        let bin = dir.path().join(toolchain_root).join("bin");
        write_stub(&bin, "rustup", "exit 0");
        write_stub(&bin, "cargo", r#"echo "cargo 1.99.0 (stub)""#);
        write_stub(&bin, "rustc", r#"echo "rustc 1.99.0 (stub 2026-01-01)""#);

        // Earlier on PATH, but lower priority than a correctly prepended
        // toolchain: a rustup that cannot work. Reaching this one is the bug.
        let shim = dir.path().join("shim");
        write_stub(
            &shim,
            "rustup",
            r#"echo "shim rustup: wrong toolchain" >&2; exit 1"#,
        );
        write_stub(
            &shim,
            "cargo",
            r#"echo "shim cargo: wrong toolchain" >&2; exit 1"#,
        );
        // Satisfy the script's system-tool checks without depending on the host.
        write_stub(&shim, "jq", "exit 0");
        write_stub(&shim, "cc", "exit 0");

        Self { dir }
    }

    /// Run `runlib.sh` from a directory holding no `Cargo.toml`.
    fn run(&self, cargo_home: Option<&str>) -> Output {
        let path = format!("{}:/usr/bin:/bin", self.dir.path().join("shim").display());
        let mut cmd = Command::new("bash");
        cmd.arg(script_path())
            .current_dir(self.dir.path().join("elsewhere"))
            .env_clear()
            .env("PATH", path)
            .env("HOME", self.dir.path().join("home"))
            .stdin(std::process::Stdio::null());
        if let Some(relative) = cargo_home {
            cmd.env("CARGO_HOME", self.dir.path().join(relative));
        }
        cmd.output().expect("run runlib.sh")
    }
}

/// Asserts the script got far enough to reach its own missing-manifest abort,
/// which it can only do once the sandboxed toolchain is on `PATH`.
fn assert_reached_manifest_abort(out: &Output) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("rustup installation appears incomplete"),
        "runlib.sh failed to resolve the sandboxed toolchain; stderr: {stderr}"
    );
    assert!(
        !stderr.contains("Installing Rust"),
        "runlib.sh must not reinstall a toolchain it can already resolve; stderr: {stderr}"
    );
    assert!(
        stderr.contains("Cargo.toml not found"),
        "expected the missing-manifest abort, got stderr: {stderr}"
    );
    assert!(
        !out.status.success(),
        "runlib.sh must fail when the caller's cwd has no Cargo.toml"
    );
}

/// The regression: a toolchain under a non-default `CARGO_HOME` must be found.
#[test]
fn runlib_resolves_a_toolchain_under_a_non_default_cargo_home() {
    let sandbox = Sandbox::new("container-state/cargo");
    assert_reached_manifest_abort(&sandbox.run(Some("container-state/cargo")));
}

/// The default layout keeps working: unset `CARGO_HOME` still means
/// `$HOME/.cargo/bin`.
#[test]
fn runlib_falls_back_to_home_cargo_bin_when_cargo_home_is_unset() {
    let sandbox = Sandbox::new("home/.cargo");
    assert_reached_manifest_abort(&sandbox.run(None));
}
