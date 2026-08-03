//! Issue #1905 — the sampler's output must not live at a predictable `/tmp` path.
//!
//! The macOS `sample` output file used to be `$TMPDIR/neat_ai_discovery.sample.<pid>.<ms>.txt`:
//! both components observable, created by an external process with no `O_EXCL`,
//! read back, and echoed to stderr. A local user who won the race controlled
//! what the operator's log was fed.
//!
//! These tests drive the real capture path through
//! `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` and assert the three properties the fix
//! owes: the containing directory is owner-only, a symlink at the output path is
//! refused rather than followed, and the directory survives no exit path —
//! neither a clean exit nor a kill on timeout.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use neat_ai_discovery::debug::{SAMPLE_KILL_GRACE_MS, SAMPLE_TIMEOUT_SECS, render_thread_dump};

/// `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` is process-global, so the cases below must
/// not overlap.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Slack over the sampler's own bound, for a loaded CI runner.
const OUTER_MARGIN: Duration = Duration::from_secs(20);

/// The longest a dump may take: the sampler bound plus its kill grace.
fn sampler_bound() -> Duration {
    Duration::from_secs(SAMPLE_TIMEOUT_SECS) + Duration::from_millis(SAMPLE_KILL_GRACE_MS)
}

/// Write an executable shell script to a unique temporary path.
fn write_script(name: &str, body: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "neat_ai_discovery_1905_{name}_{}.sh",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).expect("create script");
    file.write_all(body.as_bytes()).expect("write script");
    drop(file);

    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&path).expect("metadata").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).expect("chmod");

    path
}

/// Render a dump with `sample_program` pointed at `script`, under an outer
/// timeout so a blocking regression surfaces as a failure rather than a hang.
///
/// Returns the dump alongside any sampler scratch space still present — read
/// while `ENV_LOCK` is held, so a concurrent case's in-flight capture directory
/// cannot be mistaken for a leak.
fn dump_with_sampler(script: &Path) -> (String, Vec<PathBuf>) {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // SAFETY: serialised by ENV_LOCK, and no other thread in this test binary
    // reads the environment outside that lock.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM", script);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(render_thread_dump());
    });
    let outcome = rx.recv_timeout(sampler_bound() + OUTER_MARGIN);
    let leftovers = sampler_leftovers();

    // SAFETY: still holding ENV_LOCK.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM");
    }

    let dump = outcome.unwrap_or_else(|_| {
        panic!(
            "the thread dump blocked for more than {:?} — the handler must never block the process",
            sampler_bound() + OUTER_MARGIN
        )
    });
    (dump, leftovers)
}

/// Any sampler scratch space this process left behind under the temp directory.
fn sampler_leftovers() -> Vec<PathBuf> {
    let prefix = format!("neat_ai_discovery.sample.{}.", std::process::id());
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .collect()
}

/// The capture directory is owner-only, so the inner filename is irrelevant to
/// an attacker who can write to `/tmp`.
#[test]
#[cfg(unix)]
fn the_capture_directory_is_created_owner_only() {
    // `$5` is the `-file` argument: `<pid> 1 -mayDie -file <path>`. The script
    // records the mode of the *containing* directory beside itself.
    let script = write_script(
        "mode",
        "#!/bin/sh\n\
         dir=$(dirname \"$5\")\n\
         echo \"$dir\" > \"$0.dirmode\"\n\
         { stat -f '%Lp' \"$dir\" || stat -c '%a' \"$dir\"; } >> \"$0.dirmode\" 2>/dev/null\n\
         printf 'Call graph:\\n    Thread_11: worker\\nBinary Images:\\n' > \"$5\"\n\
         exit 0\n",
    );
    let (dump, _) = dump_with_sampler(&script);

    let mode_path = format!("{}.dirmode", script.display());
    let recorded =
        std::fs::read_to_string(&mode_path).expect("sampler recorded the directory mode");
    let _ = std::fs::remove_file(&mode_path);
    let _ = std::fs::remove_file(&script);

    let mut lines = recorded.lines();
    let dir = lines.next().unwrap_or_default().trim();
    let mode = lines.next().unwrap_or_default().trim();

    // A shared temp directory is the bug — the capture needs its own.
    assert_ne!(
        Path::new(dir),
        std::env::temp_dir(),
        "the capture must live in a per-invocation directory, not the shared temp dir: {dump}"
    );
    assert_eq!(
        mode, "700",
        "the sampler's output must live in an owner-only directory: {dump}"
    );
}

/// A symlink planted at the output path must be refused, not followed — and the
/// refusal must be visible in the dump rather than swallowed.
#[test]
#[cfg(unix)]
fn a_symlink_at_the_output_path_is_refused() {
    let bait = std::env::temp_dir().join(format!(
        "neat_ai_discovery_1905_bait_{}.txt",
        std::process::id()
    ));
    std::fs::write(
        &bait,
        "Call graph:\n    Thread_666: secret\nBinary Images:\n",
    )
    .expect("write bait file");

    let script = write_script(
        "symlink",
        &format!(
            "#!/bin/sh\n\
             rm -f \"$5\"\n\
             ln -s '{}' \"$5\"\n\
             exit 0\n",
            bait.display()
        ),
    );
    let (dump, _) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&bait);

    assert!(
        !dump.contains("Thread_666"),
        "a symlinked capture must never be followed: {dump}"
    );
    assert!(
        dump.contains("refusing to read"),
        "the refusal must be reported, not swallowed: {dump}"
    );
    assert!(
        dump.contains("END THREAD DUMP - no backtraces captured"),
        "a refused capture is not a capture: {dump}"
    );
}

/// A sampler that exits cleanly leaves no scratch space behind.
#[test]
#[cfg(unix)]
fn the_capture_directory_is_removed_when_the_sampler_exits() {
    let script = write_script(
        "clean",
        "#!/bin/sh\n\
         printf 'Call graph:\\n    Thread_12: worker\\nBinary Images:\\n' > \"$5\"\n\
         exit 0\n",
    );
    let (dump, leftovers) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("END THREAD DUMP - full dump"),
        "the capture must still be read before cleanup: {dump}"
    );
    assert_eq!(
        leftovers,
        Vec::<PathBuf>::new(),
        "a clean sampler exit must leave no temp directory behind"
    );
}

/// The kill-on-timeout path is the one that used to leak: partial output is
/// still reported, and the directory is still removed.
#[test]
#[cfg(unix)]
fn the_capture_directory_is_removed_when_the_sampler_is_killed() {
    let script = write_script(
        "hang",
        "#!/bin/sh\n\
         printf 'Call graph:\\n    Thread_13: stuck\\nBinary Images:\\n' > \"$5\"\n\
         sleep 600\n",
    );
    let started = Instant::now();
    let (dump, leftovers) = dump_with_sampler(&script);
    let elapsed = started.elapsed();
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("END THREAD DUMP - partial dump"),
        "partial output must survive the kill: {dump}"
    );
    assert_eq!(
        leftovers,
        Vec::<PathBuf>::new(),
        "a killed sampler must not leak its temp directory"
    );
    assert!(
        elapsed < sampler_bound() + OUTER_MARGIN,
        "cleanup must not introduce a blocking operation, took {elapsed:?}"
    );
}
