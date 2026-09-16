//! Issue #2055 — `scripts/runlib.sh` must honour `CARGO_HOME`.
//!
//! The drifted copy only ever prepended `$HOME/.cargo/bin`. On a host whose
//! toolchain lives elsewhere (`CARGO_HOME=/…/.container-state/cargo`, no
//! `$HOME/.cargo`), the toolchain that was already present was never
//! resolvable, so the script died long before the caller's real work and
//! `./quality.sh` could not run at all.
//!
//! **Updated by Issue #2072.** `scripts/runlib.sh` is now the canonical
//! NEAT-AI-core helper (core #680), which resolves `CARGO_HOME` in one place
//! (`_runlib_cargo_home`) and uses it for both the toolchain lookup *and* the
//! install destination. The regression is therefore asserted against both: the
//! toolchain under a non-default `CARGO_HOME` is the one that runs, and the
//! library lands under that same root. The previous tests drove the old
//! script's `_require_tools`/rustup-shim contract, which no longer exists.
//!
//! Both tests drive the real script with a sandboxed toolchain: a stub `cargo`
//! under the toolchain root, and a shim directory on `PATH` whose `cargo`
//! always fails. If the script resolves its toolchain correctly it installs the
//! library; if it resolves the wrong directory it finds the failing shim
//! instead. No network, no real toolchain, no real compile.

mod common;

use common::runlib_support::{
    lib_file as lib_file_for, link_system_tools, resolve_tool, write_rustc_stub, write_stub,
};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const CRATE: &str = "neat_ai_discovery";
const VERSION: &str = "7.7.7";

fn script_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/runlib.sh"))
}

fn lib_file() -> String {
    lib_file_for(CRATE)
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A sandbox whose only working toolchain lives at `toolchain_root/bin`, behind
/// a shim directory on `PATH` whose `cargo` always fails.
struct Sandbox {
    dir: tempfile::TempDir,
    toolchain_root: PathBuf,
}

impl Sandbox {
    /// `toolchain_root` is relative to the sandbox (e.g. `home/.cargo`).
    fn new(toolchain_root: &str) -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        let checkout = root.join("checkout");
        fs::create_dir_all(checkout.join("src")).expect("create checkout");
        fs::create_dir_all(root.join("home")).expect("create fake HOME");

        fs::write(
            checkout.join("Cargo.toml"),
            format!(
                "[package]\n\
                 name = \"{CRATE}\"\n\
                 version = \"{VERSION}\"\n\
                 edition = \"2024\"\n\
                 \n\
                 [lib]\n\
                 name = \"{CRATE}\"\n\
                 crate-type = [\"cdylib\", \"rlib\"]\n"
            ),
        )
        .expect("write Cargo.toml");
        fs::write(checkout.join("src/lib.rs"), "// fake crate\n").expect("write lib.rs");

        // The toolchain the script is expected to find.
        let bin = root.join(toolchain_root).join("bin");
        let target = checkout.join("target");
        write_stub(
            &bin,
            "cargo",
            &format!(
                r#"case "${{1:-}}" in
  metadata)
    cat <<'JSON'
{{"packages":[{{"name":"{CRATE}","version":"{VERSION}","manifest_path":"{manifest}","targets":[{{"kind":["cdylib","rlib"],"name":"{CRATE}"}}]}}],"target_directory":"{target}"}}
JSON
    ;;
  build)
    mkdir -p "{target}/release"
    printf 'fake cdylib\n' > "{target}/release/{lib}"
    ;;
  *) echo "stub cargo: unexpected $*" >&2; exit 1 ;;
esac
"#,
                manifest = checkout.join("Cargo.toml").display(),
                target = target.display(),
                lib = lib_file(),
            ),
        );
        write_rustc_stub(&bin, "1.99.0");
        write_stub(&bin, "install_name_tool", "exit 0");
        write_stub(&bin, "codesign", "exit 0");

        // Earlier on `PATH`, but lower priority than a correctly prepended
        // toolchain: a cargo that cannot work. Reaching this one is the bug.
        let shim = root.join("shim");
        write_stub(
            &shim,
            "cargo",
            r#"echo "shim cargo: wrong toolchain" >&2; exit 1"#,
        );
        write_stub(
            &shim,
            "rustc",
            r#"echo "shim rustc: wrong toolchain" >&2; exit 1"#,
        );
        link_system_tools(&shim);

        let toolchain_root = root.join(toolchain_root);
        Self {
            dir,
            toolchain_root,
        }
    }

    fn lib_path(&self) -> PathBuf {
        self.toolchain_root.join("lib").join(lib_file())
    }

    fn stamp_path(&self) -> PathBuf {
        self.toolchain_root
            .join("lib")
            .join(format!(".{CRATE}.version"))
    }

    /// Run `runlib.sh` from the sandbox checkout, optionally setting
    /// `CARGO_HOME` to a path relative to the sandbox.
    fn run(&self, cargo_home: Option<&str>) -> Output {
        let mut cmd = Command::new(resolve_tool("bash"));
        cmd.arg(script_path())
            .current_dir(self.dir.path().join("checkout"))
            .env_clear()
            .env("PATH", self.dir.path().join("shim"))
            .env("HOME", self.dir.path().join("home"))
            .stdin(std::process::Stdio::null());
        if let Some(relative) = cargo_home {
            cmd.env("CARGO_HOME", self.dir.path().join(relative));
        }
        cmd.output().expect("run runlib.sh")
    }
}

/// Asserts the sandboxed toolchain — not the failing shim — did the work, and
/// that the artefact landed under the same root the toolchain came from.
fn assert_installed_under_the_toolchain_root(sandbox: &Sandbox, out: &Output) {
    let stderr = stderr_of(out);
    assert!(
        !stderr.contains("wrong toolchain"),
        "runlib.sh resolved the shim instead of the sandboxed toolchain; stderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "runlib.sh must succeed with a resolvable toolchain; stderr: {stderr}"
    );
    assert!(
        sandbox.lib_path().is_file(),
        "the library must be installed at {}",
        sandbox.lib_path().display()
    );
    assert_eq!(
        fs::read_to_string(sandbox.stamp_path())
            .expect("read stamp")
            .trim(),
        VERSION,
        "the version stamp must sit beside the installed artefact"
    );
}

/// The regression: a toolchain under a non-default `CARGO_HOME` must be found,
/// and the library installed under that same root.
#[test]
fn runlib_resolves_a_toolchain_under_a_non_default_cargo_home() {
    let sandbox = Sandbox::new("container-state/cargo");
    let out = sandbox.run(Some("container-state/cargo"));
    assert_installed_under_the_toolchain_root(&sandbox, &out);
}

/// The default layout keeps working: unset `CARGO_HOME` still means
/// `$HOME/.cargo`.
#[test]
fn runlib_falls_back_to_home_cargo_bin_when_cargo_home_is_unset() {
    let sandbox = Sandbox::new("home/.cargo");
    let out = sandbox.run(None);
    assert_installed_under_the_toolchain_root(&sandbox, &out);
}
