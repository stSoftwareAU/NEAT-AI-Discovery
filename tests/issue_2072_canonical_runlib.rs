//! Issue #2072 — `scripts/runlib.sh` is the canonical NEAT-AI-core helper.
//!
//! The copy contract (NEAT-AI-core #680) gives the script one home —
//! `scripts/runlib.sh` on NEAT-AI-core `Develop` — from which every Rust
//! sibling takes a byte-identical copy. Discovery's previous copy had drifted:
//! it forced a full release rebuild whenever `target/` lacked the artefact,
//! even when `~/.cargo/lib/libneat_ai_discovery.so` and
//! `.neat_ai_discovery.version` already matched the crate version. On a fleet
//! build host that left an 8 GB checkout, almost all of it `target/`.
//!
//! The canonical helper instead skips the build entirely when the installed
//! artefact and its stamp match, and removes the checkout's `target/` after a
//! successful install.
//!
//! Every test drives the real committed script in a sandbox: a fake crate, a
//! stubbed toolchain on a controlled `PATH`, and a `CARGO_HOME` of its own. No
//! network, no real compile — the assertions are on observable behaviour
//! (exit status, stdout, stderr, and what is left on disk).

mod common;

use common::runlib_support::{
    lib_file as lib_file_for, link_system_tools, resolve_tool, write_rustc_stub, write_stub,
};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const CRATE: &str = "neat_ai_discovery";
const VERSION: &str = "9.9.9";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script_path() -> PathBuf {
    repo_root().join("scripts/runlib.sh")
}

/// The shared-library basename the script installs on this platform.
fn lib_file() -> String {
    lib_file_for(CRATE)
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A fake Discovery checkout with its own `CARGO_HOME` and a stubbed toolchain.
///
/// The stub `cargo` answers `metadata` with this crate's real shape (a
/// `cdylib`/`rlib` named `neat_ai_discovery`) and, for `build`, writes a
/// placeholder artefact into `target/release/`. Every invocation is appended to
/// `cargo.log`, so "ran no cargo command" is an assertion rather than a guess.
struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
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
                 rust-version = \"1.92\"\n\
                 \n\
                 [lib]\n\
                 name = \"{CRATE}\"\n\
                 crate-type = [\"cdylib\", \"rlib\"]\n"
            ),
        )
        .expect("write Cargo.toml");
        fs::write(checkout.join("src/lib.rs"), "// fake crate\n").expect("write lib.rs");

        let bin = root.join("cargo-home/bin");
        let target_dir = checkout.join("target");
        let stub_cargo = format!(
            r#"echo "$*" >> "{log}"
case "${{1:-}}" in
  metadata)
    cat <<'JSON'
{{"packages":[{{"name":"{CRATE}","version":"{VERSION}","manifest_path":"{manifest}","targets":[{{"kind":["cdylib","rlib"],"name":"{CRATE}"}}]}}],"target_directory":"{target}"}}
JSON
    ;;
  build)
    mkdir -p "{target}/release"
    printf 'fake cdylib\n' > "{target}/release/{lib}"
    # A real release build leaves a large tree behind; this stands in for it so
    # the target/ removal has something to measure and free.
    mkdir -p "{target}/release/deps"
    printf 'fake object\n' > "{target}/release/deps/placeholder.o"
    ;;
  *)
    echo "stub cargo: unexpected subcommand $*" >&2
    exit 1
    ;;
esac
"#,
            log = root.join("cargo.log").display(),
            manifest = checkout.join("Cargo.toml").display(),
            target = target_dir.display(),
            lib = lib_file(),
        );
        write_stub(&bin, "cargo", &stub_cargo);
        write_rustc_stub(&bin, "1.99.0");

        link_system_tools(&root.join("tools"));

        Self { dir }
    }

    fn checkout(&self) -> PathBuf {
        self.dir.path().join("checkout")
    }

    fn cargo_home(&self) -> PathBuf {
        self.dir.path().join("cargo-home")
    }

    fn lib_path(&self) -> PathBuf {
        self.cargo_home().join("lib").join(lib_file())
    }

    fn stamp_path(&self) -> PathBuf {
        self.cargo_home()
            .join("lib")
            .join(format!(".{CRATE}.version"))
    }

    fn target_dir(&self) -> PathBuf {
        self.checkout().join("target")
    }

    /// Pretend a previous run already installed the library at `version`.
    fn install_stamped(&self, version: &str) {
        fs::create_dir_all(self.cargo_home().join("lib")).expect("create lib dir");
        fs::write(self.lib_path(), "already installed\n").expect("write lib");
        fs::write(self.stamp_path(), format!("{version}\n")).expect("write stamp");
    }

    /// Strip the toolchain and serve a tampered `rustup-init` from a stub
    /// `curl`, so the digest gate is exercised with no network at all. Returns
    /// the sentinel path the tampered installer would write if it ever ran.
    fn tamper_with_the_rustup_bootstrap(&self) -> PathBuf {
        let bin = self.cargo_home().join("bin");
        for tool in ["cargo", "rustc"] {
            fs::remove_file(bin.join(tool)).expect("remove stub toolchain");
        }
        let sentinel = self.dir.path().join("rustup-init-executed");
        let stub_curl = format!(
            r#"out=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "-o" ]]; then out="$arg"; fi
  prev="$arg"
done
printf '#!/usr/bin/env bash
echo ran > "{sentinel}"
' > "$out"
"#,
            sentinel = sentinel.display(),
        );
        write_stub(&bin, "curl", &stub_curl);
        sentinel
    }

    /// Number of `cargo` invocations the script has made so far.
    fn cargo_invocations(&self) -> usize {
        fs::read_to_string(self.dir.path().join("cargo.log")).map_or(0, |l| l.lines().count())
    }

    fn forget_cargo_invocations(&self) {
        let _ = fs::remove_file(self.dir.path().join("cargo.log"));
    }

    /// Run the real `scripts/runlib.sh` from the fake checkout.
    fn run(&self) -> Output {
        // Deliberately not the host `PATH`: the stubbed toolchain plus the
        // symlinked utilities are all the script may see, so removing the stub
        // `cargo` really does leave the sandbox without one.
        let path = format!(
            "{}:{}",
            self.cargo_home().join("bin").display(),
            self.dir.path().join("tools").display()
        );
        // `env_clear` drops PATH, so bash itself is named absolutely.
        Command::new(resolve_tool("bash"))
            .arg(script_path())
            .current_dir(self.checkout())
            .env_clear()
            .env("PATH", path)
            .env("HOME", self.dir.path().join("home"))
            .env("CARGO_HOME", self.cargo_home())
            .stdin(std::process::Stdio::null())
            .output()
            .expect("run runlib.sh")
    }
}

/// The regression this issue exists for: an installed artefact whose stamp
/// matches must skip the build outright, even though `target/` is absent.
///
/// The drifted copy treated a missing `target/` artefact as a rebuild trigger,
/// so a host that had already installed the library rebuilt it — and regrew an
/// 8 GB `target/` — on every single invocation.
#[test]
fn an_installed_matching_version_skips_the_build_without_running_cargo() {
    let sandbox = Sandbox::new();
    sandbox.install_stamped(VERSION);
    assert!(
        !sandbox.target_dir().exists(),
        "the sandbox starts with no target/ — that is the state under test"
    );

    let out = sandbox.run();

    assert!(
        out.status.success(),
        "an up-to-date install must succeed; stderr: {}",
        stderr_of(&out)
    );
    assert_eq!(
        sandbox.cargo_invocations(),
        0,
        "no cargo command may run when the artefact and stamp already match"
    );
    assert!(
        stderr_of(&out).contains(&format!("[{CRATE}] already installed v{VERSION}")),
        "the skip must announce itself; stderr: {}",
        stderr_of(&out)
    );
    assert_eq!(
        stdout_of(&out),
        sandbox.lib_path().display().to_string(),
        "stdout must carry the installed library path and nothing else"
    );
    assert!(
        !sandbox.target_dir().exists(),
        "the skip path must not recreate target/"
    );
}

/// A stamp naming a different version is not an up-to-date install.
#[test]
fn a_stale_stamp_rebuilds_and_restamps() {
    let sandbox = Sandbox::new();
    sandbox.install_stamped("1.0.0");

    let out = sandbox.run();

    assert!(
        out.status.success(),
        "a stale install must rebuild and succeed; stderr: {}",
        stderr_of(&out)
    );
    assert!(
        sandbox.cargo_invocations() > 0,
        "a version mismatch must drive a real build"
    );
    assert_eq!(
        fs::read_to_string(sandbox.stamp_path())
            .expect("read stamp")
            .trim(),
        VERSION,
        "the stamp must be rewritten to the crate version"
    );
}

/// The whole cycle the acceptance criteria describe: build → install → remove
/// `target/`, then a second run that skips, runs no cargo command, and leaves
/// no `target/` behind.
#[test]
fn a_successful_install_removes_target_and_the_next_run_skips() {
    let sandbox = Sandbox::new();

    let first = sandbox.run();
    assert!(
        first.status.success(),
        "the first run must build and install; stderr: {}",
        stderr_of(&first)
    );
    assert!(
        sandbox.cargo_invocations() > 0,
        "the first run has nothing installed, so it must build"
    );
    assert!(
        sandbox.lib_path().is_file(),
        "the built library must be installed under CARGO_HOME/lib"
    );
    assert_eq!(
        fs::read_to_string(sandbox.stamp_path())
            .expect("read stamp")
            .trim(),
        VERSION,
        "the install must stamp the crate version beside the artefact"
    );
    assert!(
        !sandbox.target_dir().exists(),
        "the checkout's target/ must be removed after a successful install"
    );
    let first_err = stderr_of(&first);
    assert!(
        first_err.contains("removed") && first_err.contains("freed"),
        "the removal must name the path and the bytes freed; stderr: {first_err}"
    );
    assert_eq!(
        stdout_of(&first),
        sandbox.lib_path().display().to_string(),
        "stdout must carry the installed library path and nothing else"
    );

    sandbox.forget_cargo_invocations();
    let second = sandbox.run();

    assert!(
        second.status.success(),
        "the second run must succeed; stderr: {}",
        stderr_of(&second)
    );
    assert!(
        stderr_of(&second).contains(&format!("[{CRATE}] already installed v{VERSION}")),
        "the second run must report the existing install; stderr: {}",
        stderr_of(&second)
    );
    assert_eq!(
        sandbox.cargo_invocations(),
        0,
        "the second run must run no cargo command at all"
    );
    assert!(
        !sandbox.target_dir().exists(),
        "the second run must not recreate target/"
    );
}

/// Deleting the stamp is the documented way to force a rebuild — there is no
/// force flag.
#[test]
fn removing_the_stamp_forces_a_rebuild() {
    let sandbox = Sandbox::new();
    sandbox.install_stamped(VERSION);
    fs::remove_file(sandbox.stamp_path()).expect("remove stamp");

    let out = sandbox.run();

    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    assert!(
        sandbox.cargo_invocations() > 0,
        "a missing stamp must drive a rebuild rather than read as installed"
    );
}

/// The toolchain is a precondition, not something the helper installs: a host
/// with no `cargo` must be told what to install and must fail non-zero.
#[test]
fn a_missing_toolchain_fails_loud_and_installs_nothing() {
    let sandbox = Sandbox::new();
    fs::remove_file(sandbox.cargo_home().join("bin/cargo")).expect("remove stub cargo");

    let out = sandbox.run();

    assert!(
        !out.status.success(),
        "a missing toolchain must exit non-zero"
    );
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("cargo not found") && stderr.contains("rustup.rs"),
        "the failure must name the missing tool and where to get it; stderr: {stderr}"
    );
    assert!(
        !sandbox.lib_path().exists(),
        "nothing may be installed when the toolchain is missing"
    );
}

/// The helper reads its crate from the working directory, so a caller who has
/// not changed into the checkout is told so rather than silently doing nothing.
#[test]
fn it_aborts_when_the_working_directory_holds_no_manifest() {
    let elsewhere = tempfile::tempdir().expect("temp dir");
    let out = Command::new("bash")
        .arg(script_path())
        .current_dir(elsewhere.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run runlib.sh");

    assert!(
        !out.status.success(),
        "a manifest-less cwd must exit non-zero"
    );
    assert!(
        stderr_of(&out).contains("no Cargo.toml in"),
        "the abort must name the missing manifest; stderr: {}",
        stderr_of(&out)
    );
}

// --- The CI job that keeps the copy canonical -------------------------------

/// The committed workflow is the only thing that keeps the copy byte-identical
/// after this PR merges, so its load-bearing properties are asserted here. A
/// workflow cannot be executed from a unit test, so these read the artefact —
/// the same way the `CODEOWNERS` guards do (Issue #1914).
const FAMILY_SYNC: &str = include_str!("../.github/workflows/family-sync.yml");

/// It must take the file from core's `Develop`, and from nowhere else.
#[test]
fn family_sync_fetches_the_canonical_copy_from_neat_ai_core_develop() {
    assert!(
        FAMILY_SYNC.contains(
            "https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI-core/Develop/scripts/runlib.sh"
        ),
        "family-sync must fetch scripts/runlib.sh from NEAT-AI-core Develop"
    );
    assert!(
        FAMILY_SYNC.contains("cmp -s \"$fetched\" scripts/runlib.sh"),
        "family-sync must compare the fetched copy byte-for-byte"
    );
}

/// A fetch that fails must fail the job. Passing a pull request on a stale copy
/// because the canonical one was unreachable is the silent failure this guards.
#[test]
fn family_sync_fails_the_job_when_the_canonical_copy_cannot_be_fetched() {
    assert!(
        FAMILY_SYNC.contains("curl --fail"),
        "family-sync must pass curl --fail so an HTTP error exits non-zero"
    );
    for guard in [
        "fetched an empty scripts/runlib.sh",
        "did not answer with a shell script",
    ] {
        assert!(
            FAMILY_SYNC.contains(guard),
            "family-sync must reject a bad payload with `{guard}`"
        );
    }
    // Counted, not merely present: one `set -euo pipefail` would otherwise
    // satisfy a check meant to cover every multi-line `run:` block.
    let run_blocks = FAMILY_SYNC
        .lines()
        .filter(|line| line.trim_end().ends_with("run: |"))
        .count();
    assert!(run_blocks > 0, "family-sync must have run steps to check");
    assert_eq!(
        FAMILY_SYNC
            .lines()
            .filter(|line| line.trim() == "set -euo pipefail")
            .count(),
        run_blocks,
        "every multi-line run step must abort on the first failure"
    );
}

/// A drifted copy must be refreshed on the PR branch, rebasing first so a
/// branch another job moved is not rejected as a non-fast-forward.
#[test]
fn family_sync_commits_and_pushes_the_refreshed_copy() {
    assert!(
        FAMILY_SYNC
            .contains("git commit -m \"chore: sync scripts/runlib.sh from NEAT-AI-core Develop\""),
        "family-sync must commit the refreshed copy"
    );
    assert!(
        FAMILY_SYNC.contains("git rebase FETCH_HEAD"),
        "family-sync must rebase onto the branch head before pushing"
    );
    assert!(
        FAMILY_SYNC.contains("git push \"$REMOTE_URL\" \"HEAD:$HEAD_REF\""),
        "family-sync must push the refreshed copy back to the PR branch"
    );
}

/// The push credential posture must match `version-increment` in ci.yml: the
/// PAT never lands in `.git/config`, and the job is skipped on fork PRs, which
/// carry no push credential at all (Issues #1868, #2072).
#[test]
fn family_sync_keeps_the_push_token_off_disk_and_skips_forks() {
    assert!(
        FAMILY_SYNC.contains("persist-credentials: false"),
        "family-sync's checkout must not persist the push credential"
    );
    assert!(
        !FAMILY_SYNC.contains("token: ${{ secrets.ACTIONS_PUSH }}"),
        "the ACTIONS_PUSH PAT must never be handed to actions/checkout"
    );
    assert!(
        FAMILY_SYNC.contains("x-access-token:${ACTIONS_PUSH_TOKEN}@github.com"),
        "the PAT must reach git only through an explicit authenticated remote URL"
    );
    assert!(
        FAMILY_SYNC.contains("github.event.pull_request.head.repo.full_name == github.repository"),
        "family-sync must be guarded to same-repo pull requests"
    );
}

/// The gate has to run on the branches ci.yml gates, milestone PRs included.
#[test]
fn family_sync_runs_on_every_gated_pull_request() {
    assert!(
        FAMILY_SYNC.contains("pull_request:"),
        "family-sync must be a pull-request check"
    );
    for branch in ["- \"*\"", "- milestone/*"] {
        assert!(
            FAMILY_SYNC.contains(branch),
            "family-sync's branch filter must include `{branch}`"
        );
    }
    // The trigger keys sit at two-space indent under `on:`; a `push:` anywhere
    // else in the file (a comment, a step name) is not a trigger.
    assert!(
        !FAMILY_SYNC.lines().any(|line| line.trim_end() == "  push:"),
        "family-sync must not also run on pushes to the default branch"
    );
}

/// Issue #1911's property, now inside the canonical helper (core #705): a
/// `rustup-init` whose SHA-256 does not match the committed pin is never
/// executed, and the run fails loud rather than continuing without a toolchain.
///
/// The stub `curl` serves a payload of the test's own making, so this exercises
/// the digest comparison with no network access whatsoever.
#[test]
fn a_tampered_rustup_init_is_refused_without_being_executed() {
    let sandbox = Sandbox::new();
    let sentinel = sandbox.tamper_with_the_rustup_bootstrap();

    let out = sandbox.run();

    assert!(
        !out.status.success(),
        "a digest mismatch must exit non-zero; stderr: {}",
        stderr_of(&out)
    );
    assert!(
        !sentinel.exists(),
        "the unverified rustup-init must never be executed"
    );
    assert!(
        stderr_of(&out).contains("digest mismatch"),
        "the failure must name the digest mismatch; stderr: {}",
        stderr_of(&out)
    );
    assert!(
        !sandbox.lib_path().exists(),
        "nothing may be installed when the toolchain bootstrap was refused"
    );
}
