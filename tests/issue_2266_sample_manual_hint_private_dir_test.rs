//! Issue #2266 — the manual `sample` hint must not re-teach `/tmp/sample.txt`.
//!
//! Issue #1905 moved the automated capture into an owner-only per-invocation
//! directory because a fixed path in shared `/tmp` can be pre-planted as a
//! symlink. The hint printed on every sampler failure still told the operator to
//! write to `/tmp/sample.txt` — the very path #1905 removed, at the moment an
//! incident makes a human most likely to copy it. The hint must point at a fresh
//! private directory instead.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use neat_ai_discovery::debug::render_thread_dump;

/// `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` is process-global, so cases must not overlap.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Write an executable shell script to a unique temporary path.
fn write_script(name: &str, body: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "neat_ai_discovery_2266_{name}_{}.sh",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).expect("create script");
    file.write_all(body.as_bytes()).expect("write script");
    drop(file);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path).expect("metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod");
    }

    path
}

/// Render a dump with the sampler pointed at `script`.
fn dump_with_sampler(script: &Path) -> String {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // SAFETY: serialised by ENV_LOCK, and no other thread in this test binary
    // reads the environment outside that lock.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM", script);
    }
    let dump = render_thread_dump();
    // SAFETY: still holding ENV_LOCK.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM");
    }
    dump
}

fn assert_hint_uses_private_dir(dump: &str, case: &str) {
    assert!(
        dump.contains("Try manually: sample"),
        "{case}: the manual hint must still be offered\n--- dump ---\n{dump}"
    );
    assert!(
        !dump.contains("/tmp/sample.txt"),
        "{case}: the hint must not name the predictable /tmp/sample.txt path\n--- dump ---\n{dump}"
    );
    assert!(
        dump.contains("-file \"$(mktemp -d)/sample.txt\""),
        "{case}: the hint must write into a fresh mktemp -d directory\n--- dump ---\n{dump}"
    );
}

/// Non-zero exit: the most common failure path.
#[test]
#[cfg(unix)]
fn failing_sampler_hint_points_at_a_fresh_private_dir() {
    let script = write_script("fail", "#!/bin/sh\nexit 3\n");
    let dump = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert_hint_uses_private_dir(&dump, "failing sampler");
}

/// Clean exit with no capture: the empty-output path emits the same hint.
#[test]
#[cfg(unix)]
fn empty_capture_hint_points_at_a_fresh_private_dir() {
    let script = write_script("empty", "#!/bin/sh\nexit 0\n");
    let dump = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert_hint_uses_private_dir(&dump, "sampler wrote no output");
}

/// Spawn failure: a sampler that cannot be executed at all.
#[test]
#[cfg(unix)]
fn unspawnable_sampler_hint_points_at_a_fresh_private_dir() {
    let missing = std::env::temp_dir().join(format!(
        "neat_ai_discovery_2266_missing_{}.sh",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&missing);
    let dump = dump_with_sampler(&missing);

    assert_hint_uses_private_dir(&dump, "sampler could not be spawned");
}
