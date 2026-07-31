//! Issue #1865: the mandated pre-commit gate must not mutate the dependency graph.
//!
//! `./quality.sh` is the documented pre-commit step (CONTRIBUTING.md). It used
//! to run `cargo upgrade --incompatible` followed by `cargo update`, which
//! re-resolved every direct and transitive dependency to the newest published
//! version with **no** age check — bypassing both the Renovate
//! `minimumReleaseAge` window and the `VIBE_BUMP_QUARANTINE_HOURS` gate in
//! `bump-deps.sh`. A crate published minutes earlier was accepted, and its
//! `build.rs` then executed on the contributor's machine.
//!
//! This test runs the real `quality.sh` with a stubbed `cargo` on `PATH` that
//! records every invocation, and asserts the gate verifies the tree without
//! ever upgrading or re-resolving it.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Stdio};

/// Write an executable stub script into `dir`.
fn write_stub(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, body).expect("write stub");
    let mut perms = fs::metadata(&path).expect("stat stub").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).expect("chmod stub");
}

#[test]
fn quality_gate_never_mutates_the_dependency_graph() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let tmp = tempfile::tempdir().expect("temp dir");
    let bin = tmp.path().join("bin");
    fs::create_dir_all(&bin).expect("create stub bin dir");
    let log = tmp.path().join("cargo-invocations.log");

    // Stub `cargo`: record the sub-command line, succeed, mutate nothing.
    write_stub(
        &bin,
        "cargo",
        &format!(
            "#!/bin/bash\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
            log.display()
        ),
    );
    // `cargo-upgrade` present on PATH: the gate must still not upgrade.
    write_stub(&bin, "cargo-upgrade", "#!/bin/bash\nexit 0\n");
    // `shellcheck` is a hard requirement of the gate; stub it so this test
    // does not depend on the host having it installed.
    write_stub(&bin, "shellcheck", "#!/bin/bash\nexit 0\n");

    // HOME points at the temp dir so quality.sh does not source the real
    // ~/.cargo/env and prepend the real cargo ahead of our stub.
    let output = Command::new("bash")
        .arg("quality.sh")
        .current_dir(root)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("HOME", tmp.path())
        .stdin(Stdio::null())
        .output()
        .expect("run quality.sh");

    assert!(
        output.status.success(),
        "quality.sh failed under stubbed cargo (Issue #1865)\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let invocations = fs::read_to_string(&log).expect("cargo stub was never invoked");

    // Sanity: the gate really did drive cargo (otherwise the negative
    // assertions below would pass vacuously).
    assert!(
        invocations.lines().any(|line| line.starts_with("build")),
        "quality.sh must still build the tree — recorded cargo calls:\n{invocations}"
    );

    for line in invocations.lines() {
        assert!(
            !line.starts_with("upgrade"),
            "quality.sh must not run `cargo upgrade` — it bypasses the \
             {}h quarantine window (Issue #1865). Recorded: `cargo {line}`",
            24
        );
        assert!(
            !line.starts_with("update"),
            "quality.sh must not run `cargo update` — re-resolving the \
             lockfile pulls unquarantined transitive versions (Issue #1865). \
             Recorded: `cargo {line}`"
        );
    }
}
