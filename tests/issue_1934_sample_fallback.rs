//! Issue #1934 — a SIGUSR1 thread dump must degrade, never disappear.
//!
//! The M2 Ultra wedge produced a dump whose whole body was a warning that
//! `sample` had timed out, followed immediately by `END THREAD DUMP`. An empty
//! dump is worse than no dump: it looks clean.
//!
//! These tests point `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` at scripts that
//! reproduce the two ways the sampler fails a wedged process, and assert that
//! the dump still carries the in-process fallback and an honest banner.
//!
//! Every case is bounded by an outer timeout so a regression that reintroduces
//! blocking fails the test rather than wedging CI.

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
        "neat_ai_discovery_1934_{name}_{}.sh",
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

/// Render a dump with `sample_program` pointed at `script`, under an outer
/// timeout so a blocking regression surfaces as a failure rather than a hang.
fn dump_with_sampler(script: &Path) -> (String, Duration) {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // SAFETY: serialised by ENV_LOCK, and no other thread in this test binary
    // reads the environment outside that lock.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM", script);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let started = Instant::now();
    std::thread::spawn(move || {
        let _ = tx.send(render_thread_dump());
    });
    let outcome = rx.recv_timeout(sampler_bound() + OUTER_MARGIN);
    let elapsed = started.elapsed();

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
    (dump, elapsed)
}

/// The fallback block is what makes an otherwise-empty dump actionable.
fn assert_fallback_present(dump: &str, case: &str) {
    for expected in [
        "In-process state",
        "Process ID:",
        "Elapsed run time:",
        "GPU circuit breaker:",
        "Abandoned GPU threads:",
        "Last heartbeat:",
        "Outstanding GPU requests",
        "Dumping thread:",
    ] {
        assert!(
            dump.contains(expected),
            "{case}: the fallback must report {expected:?}\n--- dump ---\n{dump}"
        );
    }

    assert!(
        dump.contains("END THREAD DUMP - no backtraces captured"),
        "{case}: the banner must say no backtraces were captured\n--- dump ---\n{dump}"
    );
    assert!(
        !dump.contains("END THREAD DUMP\n"),
        "{case}: a bare END THREAD DUMP banner is the regression\n--- dump ---\n{dump}"
    );
    assert!(
        dump.contains("Try manually: sample"),
        "{case}: the manual hint must survive the fallback\n--- dump ---\n{dump}"
    );
}

/// Case 1: the sampler hangs past its timeout, exactly as on the wedged M2 Ultra.
#[test]
#[cfg(unix)]
fn a_hanging_sampler_still_produces_the_in_process_fallback() {
    let script = write_script("hang", "#!/bin/sh\nsleep 600\n");
    let (dump, elapsed) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("did not exit within"),
        "the timeout must be reported: {dump}"
    );
    assert_fallback_present(&dump, "hanging sampler");

    // Case 3: the timing bound. A regression that reintroduces blocking fails
    // here rather than hanging CI.
    assert!(
        elapsed < sampler_bound() + OUTER_MARGIN,
        "the handler must return within {:?}, took {elapsed:?}",
        sampler_bound() + OUTER_MARGIN
    );
}

/// Case 2: the sampler exits cleanly but writes no output file.
#[test]
#[cfg(unix)]
fn a_sampler_that_writes_no_output_still_produces_the_in_process_fallback() {
    let script = write_script("noout", "#!/bin/sh\nexit 0\n");
    let (dump, _elapsed) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("wrote no readable output"),
        "an empty capture must be called out, not treated as success: {dump}"
    );
    assert_fallback_present(&dump, "sampler wrote no output");
}

/// A sampler that fails outright is the third way to get nothing — the dump must
/// still degrade rather than disappear.
#[test]
#[cfg(unix)]
fn a_failing_sampler_still_produces_the_in_process_fallback() {
    let script = write_script("fail", "#!/bin/sh\nexit 3\n");
    let (dump, _elapsed) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("command failed"),
        "a non-zero exit must be reported: {dump}"
    );
    assert_fallback_present(&dump, "failing sampler");
}

/// A sampler that writes a real capture is a "full dump" — the banner must
/// distinguish it from the fallback cases above.
#[test]
#[cfg(unix)]
fn a_successful_sampler_is_reported_as_a_full_dump() {
    // `$5` is the `-file` argument: `<pid> 1 -mayDie -file <path>`.
    let script = write_script(
        "full",
        "#!/bin/sh\n\
         printf 'Call graph:\\n    Thread_99: worker\\n    2000 neat_ai_discovery::analysis\\nBinary Images:\\n' > \"$5\"\n\
         exit 0\n",
    );
    let (dump, _elapsed) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("END THREAD DUMP - full dump"),
        "a complete capture must be banner-ed as a full dump: {dump}"
    );
    assert!(
        dump.contains("Thread_99"),
        "the captured thread must be printed: {dump}"
    );
    // The state block is unconditional — a full dump carries it too.
    assert!(dump.contains("In-process state"), "{dump}");
}

/// A sampler killed mid-capture leaves partial output, which must be reported as
/// a partial dump — neither a full one nor an empty one.
#[test]
#[cfg(unix)]
fn a_killed_sampler_with_partial_output_is_reported_as_a_partial_dump() {
    let script = write_script(
        "partial",
        "#!/bin/sh\n\
         printf 'Call graph:\\n    Thread_7: stuck in Metal\\nBinary Images:\\n' > \"$5\"\n\
         sleep 600\n",
    );
    let (dump, elapsed) = dump_with_sampler(&script);
    let _ = std::fs::remove_file(&script);

    assert!(
        dump.contains("END THREAD DUMP - partial dump"),
        "recovered partial output must be banner-ed as a partial dump: {dump}"
    );
    assert!(dump.contains("Thread_7"), "partial content printed: {dump}");
    assert!(dump.contains("In-process state"), "{dump}");
    assert!(
        elapsed < sampler_bound() + OUTER_MARGIN,
        "the handler must return within {:?}, took {elapsed:?}",
        sampler_bound() + OUTER_MARGIN
    );
}
