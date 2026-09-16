//! Shared sandbox primitives for the `scripts/runlib.sh` tests (Issue #2072).
//!
//! `tests/issue_2072_canonical_runlib.rs` and
//! `tests/issue_2055_runlib_cargo_home_path.rs` each drive the real script in a
//! sandbox of its own, but both need the same four things: the platform's
//! library basename, an executable stub, an absolute path for a system utility,
//! and the set of utilities the script shells out to. Only those primitives are
//! shared — the two `Sandbox` types stay separate, because what they are proving
//! differs and one parameterised harness would obscure both.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The POSIX utilities the script shells out to, plus the `bash` its stubs'
/// shebangs name. Symlinked into a directory of their own so a sandbox `PATH`
/// can offer them *without* offering the host's real `cargo` — "no toolchain
/// installed" has to be a state a test can actually create.
#[allow(dead_code)]
pub const REQUIRED_TOOLS: &[&str] = &[
    "bash",
    "jq",
    "awk",
    "grep",
    "sed",
    "cat",
    "dirname",
    "uname",
    "du",
    "rm",
    "mv",
    "cp",
    "mkdir",
    "chmod",
    "head",
    "mktemp",
    "sha256sum",
    "ldd",
];

/// The shared-library basename `runlib.sh` installs for `crate` on this
/// platform.
#[allow(dead_code)]
pub fn lib_file(crate_name: &str) -> String {
    let extension = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    format!("lib{crate_name}.{extension}")
}

#[allow(dead_code)]
pub fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod");
    }
}

/// Write an executable bash stub named `name` into `dir`.
#[allow(dead_code)]
pub fn write_stub(dir: &Path, name: &str, body: &str) {
    fs::create_dir_all(dir).expect("create stub dir");
    let path = dir.join(name);
    fs::write(&path, format!("#!/usr/bin/env bash\n{body}\n")).expect("write stub");
    make_executable(&path);
}

/// The host target triple the stub `rustc` reports, derived from the platform
/// these tests are compiled for so the sandbox names the host it runs on.
#[allow(dead_code)]
pub fn host_triple() -> String {
    let suffix = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_env = "musl") {
        "unknown-linux-musl"
    } else {
        "unknown-linux-gnu"
    };
    format!("{}-{suffix}", std::env::consts::ARCH)
}

/// Write the sandbox's stub `rustc` into `dir`.
///
/// It answers both surfaces the script reads: `rustc --version` for the
/// toolchain gate's active version, and `rustc -vV` for the `host:` line that
/// gate passes to `cargo metadata --filter-platform`. A stub that answered only
/// the first made the script die on every run (`rustc -vV named no host
/// target`), so the whole contract lives here rather than in each sandbox.
#[allow(dead_code)]
pub fn write_rustc_stub(dir: &Path, version: &str) {
    write_stub(
        dir,
        "rustc",
        &format!(
            r#"if [[ "${{1:-}}" == "-vV" || "${{1:-}}" == "--version" && "${{2:-}}" == "--verbose" ]]; then
  cat <<'EOF'
rustc {version} (stub 2026-01-01)
binary: rustc
commit-hash: unknown
commit-date: unknown
host: {host}
release: {version}
EOF
else
  echo "rustc {version} (stub 2026-01-01)"
fi"#,
            host = host_triple(),
        ),
    );
}

/// Absolute path of a system utility, resolved through the host's own `PATH`.
/// A utility the sandbox needs but the host does not have fails the test rather
/// than quietly leaving a gap.
#[allow(dead_code)]
pub fn resolve_tool(name: &str) -> PathBuf {
    let out = Command::new("bash")
        .arg("-c")
        .arg(format!("command -v {name}"))
        .output()
        .expect("resolve tool");
    assert!(
        out.status.success(),
        "`{name}` must be on PATH to run this test"
    );
    PathBuf::from(String::from_utf8_lossy(&out.stdout).trim())
}

/// Symlink every [`REQUIRED_TOOLS`] entry into `dir`, plus the macOS-only pair
/// when the host has them.
#[allow(dead_code)]
pub fn link_system_tools(dir: &Path) {
    fs::create_dir_all(dir).expect("create tools dir");
    for tool in REQUIRED_TOOLS {
        std::os::unix::fs::symlink(resolve_tool(tool), dir.join(tool)).expect("link system tool");
    }
    // Reached only on macOS, and absent on Linux — linked when present.
    for tool in ["codesign", "install_name_tool"] {
        if let Ok(out) = Command::new("bash")
            .arg("-c")
            .arg(format!("command -v {tool}"))
            .output()
            && out.status.success()
        {
            let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            std::os::unix::fs::symlink(path, dir.join(tool)).expect("link system tool");
        }
    }
}
