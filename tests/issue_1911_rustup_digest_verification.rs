//! Issue #1911: the rustup bootstrap must verify a pinned digest before it
//! executes anything.
//!
//! `scripts/runlib.sh` used to pipe `https://sh.rustup.rs` straight into `sh`.
//! Transport pinning proves the bytes came from that host; it does not prove
//! they are the bytes anyone reviewed. A hijacked distribution point — or a
//! proxy holding a CA the machine trusts — would have executed arbitrary code
//! as the invoking user, installing the compiler that then runs `build.rs` for
//! every dependency.
//!
//! `scripts/install-rustup.sh` now downloads the pinned `rustup-init` for the
//! host target and executes it only when its SHA-256 matches the digest
//! committed in `scripts/rustup-init.sha256`.
//!
//! These tests run the real script against a stub `curl`, so the assertions are
//! on observable behaviour — exit codes, stderr, and whether the downloaded
//! file was executed at all — rather than on source text.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_path() -> PathBuf {
    repo_root().join("scripts/install-rustup.sh")
}

fn manifest_path() -> PathBuf {
    repo_root().join("scripts/rustup-init.sha256")
}

/// Every target triple the committed manifest is expected to pin.
const PINNED_TARGETS: [&str; 6] = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
];

fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod");
    }
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// SHA-256 of a file, computed independently of the script under test.
fn sha256_of(path: &Path) -> String {
    for (tool, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        let Ok(out) = Command::new(tool).args(&args).arg(path).output() else {
            continue;
        };
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .next()
                .expect("digest field")
                .to_string();
        }
    }
    panic!("no SHA-256 tool available (need sha256sum or shasum)");
}

/// A sandbox holding a copy of the script, a generated digest manifest, and a
/// stub `curl` that serves a chosen payload.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    /// `served` is the payload the stub `curl` writes; `pinned` is the payload
    /// whose digest is recorded in the manifest. Passing different bodies
    /// simulates a tampered download. `download_ok` false makes `curl` fail.
    fn new(served: &str, pinned: &str, download_ok: bool) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bin = dir.path().join("bin");
        let scripts = dir.path().join("scripts");
        fs::create_dir_all(&bin).expect("create bin dir");
        fs::create_dir_all(&scripts).expect("create scripts dir");
        fs::create_dir_all(dir.path().join("home")).expect("create fake home");

        let sentinel = dir.path().join("executed");
        let served_path = dir.path().join("served-payload");
        let pinned_path = dir.path().join("pinned-payload");
        let curl_log = dir.path().join("curl.log");

        for (path, body) in [(&served_path, served), (&pinned_path, pinned)] {
            fs::write(
                path,
                body.replace("@SENTINEL@", &sentinel.display().to_string()),
            )
            .expect("write payload");
        }

        fs::copy(script_path(), scripts.join("install-rustup.sh")).expect("copy script");

        // Same digest on every target, so the host's detected triple resolves
        // regardless of which machine runs the test.
        let digest = sha256_of(&pinned_path);
        let manifest: String = PINNED_TARGETS
            .iter()
            .map(|t| format!("{digest}  {t}\n"))
            .collect();
        fs::write(scripts.join("rustup-init.sha256"), manifest).expect("write manifest");

        let stub = format!(
            r#"#!/bin/bash
echo "$*" >> "{log}"
out=""
prev=""
for a in "$@"; do
    if [[ "$prev" == "-o" ]]; then out="$a"; fi
    prev="$a"
done
if [[ "{ok}" != "true" ]]; then
    echo "stub curl: simulated download failure" >&2
    exit 22
fi
cp "{served}" "$out"
"#,
            log = curl_log.display(),
            ok = download_ok,
            served = served_path.display(),
        );
        let stub_path = bin.join("curl");
        fs::write(&stub_path, stub).expect("write stub curl");
        make_executable(&stub_path);

        Self { dir }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new("bash")
            .arg(self.dir.path().join("scripts/install-rustup.sh"))
            .args(args)
            .env_clear()
            .env(
                "PATH",
                format!(
                    "{}:/usr/bin:/bin:/usr/sbin:/sbin",
                    self.dir.path().join("bin").display()
                ),
            )
            .env("HOME", self.dir.path().join("home"))
            .output()
            .expect("run install-rustup.sh")
    }

    /// Contents of the sentinel file, present only if the download was executed.
    fn executed(&self) -> Option<String> {
        fs::read_to_string(self.dir.path().join("executed")).ok()
    }

    fn curl_invocations(&self) -> usize {
        fs::read_to_string(self.dir.path().join("curl.log")).map_or(0, |log| log.lines().count())
    }
}

/// A payload that records the arguments it was invoked with.
const INSTALLER: &str = "#!/bin/bash\necho \"$*\" > \"@SENTINEL@\"\n";
const TAMPERED_INSTALLER: &str = "#!/bin/bash\necho \"pwned $*\" > \"@SENTINEL@\"\n";

#[test]
fn script_is_committed_and_executable() {
    let path = script_path();
    assert!(path.is_file(), "{} must exist", path.display());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&path)
            .expect("stat script")
            .permissions()
            .mode();
        assert!(
            mode & 0o111 != 0,
            "{} must be executable (mode {mode:o})",
            path.display()
        );
    }
}

#[test]
fn executes_the_installer_when_the_digest_matches() {
    let sandbox = Sandbox::new(INSTALLER, INSTALLER, true);
    let out = sandbox.run(&["-y"]);
    assert!(
        out.status.success(),
        "a digest-matching installer must run; stderr: {}",
        stderr_of(&out)
    );
    assert_eq!(
        sandbox.executed().as_deref().map(str::trim),
        Some("-y"),
        "the verified installer must be executed with the forwarded arguments"
    );
}

#[test]
fn defaults_to_the_unattended_flag_when_no_arguments_are_given() {
    let sandbox = Sandbox::new(INSTALLER, INSTALLER, true);
    let out = sandbox.run(&[]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    assert_eq!(
        sandbox.executed().as_deref().map(str::trim),
        Some("-y"),
        "the bootstrap must stay unattended by default"
    );
}

#[test]
fn rejects_a_tampered_download_without_executing_it() {
    // The acceptance criterion: served bytes differ from the pinned digest.
    let sandbox = Sandbox::new(TAMPERED_INSTALLER, INSTALLER, true);
    let out = sandbox.run(&["-y"]);

    assert!(
        !out.status.success(),
        "a digest mismatch must exit non-zero, stderr: {}",
        stderr_of(&out)
    );
    assert!(
        sandbox.executed().is_none(),
        "the tampered installer must never be executed, got: {:?}",
        sandbox.executed()
    );

    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("expected:") && stderr.contains("actual:"),
        "the failure must name the expected and actual digests, got: {stderr}"
    );
    assert!(
        stderr.contains(&sha256_of(&sandbox.dir.path().join("pinned-payload"))),
        "the expected digest must appear in the message, got: {stderr}"
    );
}

#[test]
fn fails_loud_when_the_download_fails() {
    let sandbox = Sandbox::new(INSTALLER, INSTALLER, false);
    let out = sandbox.run(&["-y"]);
    assert!(
        !out.status.success(),
        "a failed download must not be reported as success"
    );
    assert!(
        sandbox.executed().is_none(),
        "nothing may be executed when the download failed"
    );
    assert!(
        stderr_of(&out).contains("failed to download"),
        "stderr must name the download failure, got: {}",
        stderr_of(&out)
    );
}

#[test]
fn fails_closed_when_no_digest_is_pinned_for_the_host_target() {
    let sandbox = Sandbox::new(INSTALLER, INSTALLER, true);
    // Drop every pin: an unpinned host must abort before any download.
    fs::write(
        sandbox.dir.path().join("scripts/rustup-init.sha256"),
        "# no pins\n",
    )
    .expect("truncate manifest");

    let out = sandbox.run(&["-y"]);
    assert!(
        !out.status.success(),
        "an unpinned target must exit non-zero"
    );
    assert_eq!(
        sandbox.curl_invocations(),
        0,
        "nothing may be downloaded when no digest is pinned"
    );
    assert!(
        stderr_of(&out).contains("no pinned rustup-init digest"),
        "stderr must explain the missing pin, got: {}",
        stderr_of(&out)
    );
}

#[test]
fn fails_closed_when_the_digest_manifest_is_missing() {
    let sandbox = Sandbox::new(INSTALLER, INSTALLER, true);
    fs::remove_file(sandbox.dir.path().join("scripts/rustup-init.sha256"))
        .expect("remove manifest");

    let out = sandbox.run(&["-y"]);
    assert!(
        !out.status.success(),
        "a missing digest manifest must exit non-zero"
    );
    assert_eq!(
        sandbox.curl_invocations(),
        0,
        "nothing may be downloaded without a digest manifest"
    );
}

// --- Committed manifest and call-site guards -------------------------------

/// Parsed `<digest> <target>` pairs from the committed manifest.
fn committed_pins() -> Vec<(String, String)> {
    fs::read_to_string(manifest_path())
        .expect("read committed digest manifest")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut parts = l.split_whitespace();
            let digest = parts.next().expect("digest field").to_string();
            let target = parts.next().expect("target field").to_string();
            (digest, target)
        })
        .collect()
}

#[test]
fn committed_manifest_pins_a_valid_digest_for_every_supported_target() {
    let pins = committed_pins();
    for target in PINNED_TARGETS {
        let pin = pins.iter().find(|(_, t)| t == target).unwrap_or_else(|| {
            panic!("{target} has no pinned digest in scripts/rustup-init.sha256")
        });
        assert_eq!(
            pin.0.len(),
            64,
            "{target} digest must be a 64-character SHA-256, got {:?}",
            pin.0
        );
        assert!(
            pin.0.chars().all(|c| c.is_ascii_hexdigit()),
            "{target} digest must be hexadecimal, got {:?}",
            pin.0
        );
    }
}

#[test]
fn the_host_target_resolves_to_a_pinned_digest() {
    // Sources the real script and asks it, on this machine, for the digest it
    // would verify against — so a host the repository cannot bootstrap fails
    // here rather than during an install.
    let out = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "source '{}'; _pinned_digest \"$(_host_target)\"",
            script_path().display()
        ))
        .output()
        .expect("source install-rustup.sh");
    assert!(
        out.status.success(),
        "this host has no pinned rustup-init digest: {}",
        stderr_of(&out)
    );
    let digest = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(digest.len(), 64, "expected a SHA-256, got {digest:?}");
}

#[test]
fn runlib_no_longer_pipes_a_network_download_into_a_shell() {
    let body = fs::read_to_string(repo_root().join("scripts/runlib.sh")).expect("read runlib.sh");
    for line in body.lines() {
        assert!(
            !(line.contains("curl") && line.contains("| sh")),
            "runlib.sh must not pipe a download into a shell (Issue #1911): {line}"
        );
    }
    assert!(
        body.contains("install-rustup.sh"),
        "runlib.sh must bootstrap rustup through the digest-verifying script"
    );
}

#[test]
fn runlib_keeps_its_path_persistence_and_sanity_check() {
    // Acceptance criterion: the surrounding behaviour is untouched.
    let body = fs::read_to_string(repo_root().join("scripts/runlib.sh")).expect("read runlib.sh");
    for rc in [".bashrc", ".zshrc", ".bash_profile"] {
        assert!(
            body.contains(rc),
            "runlib.sh must still persist PATH into {rc}"
        );
    }
    assert!(
        body.contains("rustup show"),
        "runlib.sh must keep the `rustup show` sanity check"
    );
}
