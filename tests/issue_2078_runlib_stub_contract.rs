//! Issue #2078 — the runlib sandbox stub must answer the whole `rustc` surface
//! the synced script reads (check added by the Issue #2085 retro).
//!
//! `scripts/runlib.sh` (re-copied from NEAT-AI-core on every pull request) reads
//! two surfaces from `rustc`: `--version` for the toolchain gate, and `-vV` for
//! the `host:` line it passes to `cargo metadata --filter-platform`. When core
//! added the `-vV` call, the sandbox stub in `tests/common/runlib_support.rs`
//! still only answered `--version`, and the next sync died on
//! `rustc -vV named no host target` — on whatever unrelated PR happened to be
//! open at the time. These tests drive the stub directly and pin it to the
//! script's contract, so the failure names `write_rustc_stub` instead of
//! surfacing deep inside a full script run.
//!
//! The full-script tests (`issue_2072_canonical_runlib.rs`,
//! `issue_2055_runlib_cargo_home_path.rs`) remain the backstop for any *other*
//! command surface a future sync adds: they run the real script through a build
//! path that exercises every `rustc` call.

mod common;

use common::runlib_support::{host_triple, write_rustc_stub};
use std::path::Path;
use std::process::Command;

/// Run the stub `rustc` in `dir` with `arg` and return its stdout, failing the
/// test when the stub exits non-zero.
fn stub_rustc(dir: &Path, arg: &str) -> String {
    let out = Command::new(dir.join("rustc"))
        .arg(arg)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run the stub rustc");
    assert!(
        out.status.success(),
        "the stub rustc must exit 0 for `{arg}`; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// `_runlib_read_rust_version` parses `rustc --version` with
/// `sed -n 's/^rustc \([0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)\{0,1\}\).*/\1/p'`,
/// so the stub's output must carry a parseable `rustc X.Y.Z` on the first line.
#[test]
fn stub_rustc_version_is_parseable_by_the_script() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_rustc_stub(dir.path(), "1.99.0");

    let out = stub_rustc(dir.path(), "--version");
    let first = out.lines().next().unwrap_or_default();
    assert!(
        first.starts_with("rustc 1.99.0"),
        "the stub rustc --version must start with `rustc 1.99.0` so the script's \
         version sed extracts a version; got `{first}`. Update \
         tests/common/runlib_support.rs::write_rustc_stub (Issue #2078)"
    );
}

/// `_runlib_required_rust_version` extracts the host from `rustc -vV` with
/// `sed -n 's/^host: //p'`, so the stub must emit a `host:` line naming this
/// platform's triple.
#[test]
fn stub_rustc_vv_names_the_host_for_the_scripts_platform_filter() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_rustc_stub(dir.path(), "1.99.0");

    let out = stub_rustc(dir.path(), "-vV");
    let host_line = format!("host: {}", host_triple());
    assert!(
        out.lines().any(|l| l == host_line),
        "the stub rustc -vV must print `{host_line}` so the script's \
         `sed -n 's/^host: //p'` extracts a host target; got:\n{out}\nUpdate \
         tests/common/runlib_support.rs::write_rustc_stub (Issue #2078)"
    );
}
