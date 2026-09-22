//! Issue #2141: `benchmark.sh` read the clock with `date +%s.%N`.
//!
//! `%N` is a GNU extension. BSD `date` (macOS) emits the two characters
//! verbatim, so `start` came back as `1769040000.N`, `bc` rejected the operand,
//! `duration` was empty, and `printf "%11.2fs"` then failed on an empty numeric
//! argument. The repo's shared standard requires macOS (bash 3.2), Ubuntu and
//! AWS Linux to all work.
//!
//! These tests drive the real `now_seconds` helper (sourceable via
//! `BENCHMARK_SOURCE_ONLY=1`) and assert on its actual output and exit codes —
//! including under a stubbed BSD `date` that ignores `%N`.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Source `benchmark.sh` in helper-only mode and run `snippet`, returning the
/// raw `Output` so tests can assert on the exit status as well as the streams.
/// `path` replaces `PATH` when `Some`, which is how the macOS/BSD and
/// no-clock-source scenarios are simulated.
fn run_helper(snippet: &str, path: Option<&str>) -> Output {
    let script = repo_root().join("benchmark.sh");
    let mut cmd = Command::new("/bin/bash");
    cmd.arg("-c")
        .arg(format!("source '{}' && {snippet}", script.display()))
        .current_dir(repo_root())
        .env("BENCHMARK_SOURCE_ONLY", "1")
        .stdin(Stdio::null());
    if let Some(path) = path {
        cmd.env("PATH", path);
    }
    cmd.output().expect("run benchmark.sh helper")
}

/// Write a BSD-flavoured `date` stub into `dir`: every `%N` in a format
/// argument is emitted as a literal `N`, exactly as macOS's `date` does.
fn write_bsd_date_stub(dir: &Path) {
    let stub = dir.join("date");
    fs::write(
        &stub,
        "#!/bin/bash\n\
         # BSD `date`: %N is not a conversion spec, so it survives verbatim.\n\
         args=()\n\
         for a in \"$@\"; do\n\
         \x20   case \"$a\" in\n\
         \x20       +*) args+=( \"${a//%N/N}\" ) ;;\n\
         \x20       *) args+=( \"$a\" ) ;;\n\
         \x20   esac\n\
         done\n\
         exec /bin/date \"${args[@]}\"\n",
    )
    .expect("write date stub");
    let mut perms = fs::metadata(&stub).expect("stub metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub, perms).expect("chmod date stub");
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn describe(out: &Output) -> String {
    format!(
        "exit: {:?}\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// The downstream arithmetic `run_benchmark` performs, minus the benchmarked
/// command: two readings, a subtraction, and the `printf` that formats the
/// summary table. `bc` is not installed everywhere, so the subtraction uses
/// `awk` when `bc` is absent — either way the readings must be accepted.
const TIMING_CHAIN: &str = r#"
start=$(now_seconds)
end=$(now_seconds)
if command -v bc > /dev/null 2>&1; then
    duration=$(echo "$end - $start" | bc)
else
    duration=$(awk -v e="$end" -v s="$start" 'BEGIN { printf "%.6f", e - s }')
fi
printf '%11.2fs\n' "$duration"
"#;

// ── The helper's output shape ─────────────────────────────────────────

#[test]
fn now_seconds_emits_a_number_printf_accepts() {
    let out = run_helper("printf '%.2f\\n' \"$(now_seconds)\"", None);

    assert!(
        out.status.success(),
        "now_seconds must emit a value printf accepts (Issue #2141).\n{}",
        describe(&out)
    );
    let reading = stdout_of(&out);
    reading
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("printf produced a non-numeric reading {reading:?}: {e}"));
}

#[test]
fn now_seconds_readings_are_monotonic_epoch_seconds() {
    let out = run_helper("now_seconds; now_seconds", None);
    assert!(out.status.success(), "{}", describe(&out));

    let readings: Vec<f64> = stdout_of(&out)
        .lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("non-numeric reading {line:?}: {e}"))
        })
        .collect();

    assert_eq!(readings.len(), 2, "expected two readings\n{}", describe(&out));
    assert!(
        readings[1] >= readings[0],
        "the clock must not run backwards: {readings:?}"
    );
    assert!(
        readings[0] > 1_600_000_000.0,
        "readings must be seconds since the epoch, got {:?}",
        readings[0]
    );
}

// ── macOS: BSD `date` and a bash with no EPOCHREALTIME ────────────────

#[test]
fn bsd_date_stub_reproduces_the_literal_n() {
    let tmp = tempfile::tempdir().expect("temp dir");
    write_bsd_date_stub(tmp.path());
    let path = format!("{}:{}", tmp.path().display(), std::env::var("PATH").unwrap_or_default());

    // Guards the tests below from going vacuous: the stub must actually
    // reproduce the BSD behaviour this issue is about.
    let out = run_helper("date +%s.%N", Some(&path));
    let reading = stdout_of(&out);
    assert!(
        reading.ends_with(".N"),
        "the BSD date stub must emit a literal N, got {reading:?}\n{}",
        describe(&out)
    );
}

#[test]
fn macos_bsd_date_and_bash_3_2_still_time_the_run() {
    let tmp = tempfile::tempdir().expect("temp dir");
    write_bsd_date_stub(tmp.path());
    let path = format!("{}:{}", tmp.path().display(), std::env::var("PATH").unwrap_or_default());

    // bash 3.2 (the macOS system bash) has no EPOCHREALTIME.
    let out = run_helper(&format!("unset EPOCHREALTIME\n{TIMING_CHAIN}"), Some(&path));

    assert!(
        out.status.success(),
        "the timing chain must work on a BSD-date host (Issue #2141).\n{}",
        describe(&out)
    );
    let formatted = stdout_of(&out);
    assert!(
        !formatted.contains('N'),
        "a literal N leaked into the duration: {formatted:?}\n{}",
        describe(&out)
    );
    let seconds = formatted
        .trim_end_matches('s')
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("duration {formatted:?} is not numeric: {e}"));
    assert!(
        (0.0..60.0).contains(&seconds),
        "two back-to-back readings should be seconds apart at most, got {seconds}"
    );
}

#[test]
fn whole_second_date_fallback_is_used_when_nothing_finer_exists() {
    let tmp = tempfile::tempdir().expect("temp dir");
    write_bsd_date_stub(tmp.path());

    // Only the stubbed `date` is reachable: no perl, no EPOCHREALTIME.
    let out = run_helper("unset EPOCHREALTIME; now_seconds", Some(&tmp.path().display().to_string()));

    assert!(
        out.status.success(),
        "a POSIX `date +%s` is enough to time the run (Issue #2141).\n{}",
        describe(&out)
    );
    let reading = stdout_of(&out);
    assert!(
        reading.chars().all(|c| c.is_ascii_digit()) && !reading.is_empty(),
        "the whole-second fallback must emit bare digits, got {reading:?}\n{}",
        describe(&out)
    );
}

// ── No clock source at all must fail loud ─────────────────────────────

#[test]
fn missing_clock_source_fails_loud_instead_of_returning_empty() {
    let out = run_helper("unset EPOCHREALTIME; now_seconds", Some(""));

    assert!(
        !out.status.success(),
        "with no clock source available now_seconds must fail loud, not emit \
         an empty duration (Issue #2141).\n{}",
        describe(&out)
    );
    assert!(
        stdout_of(&out).is_empty(),
        "nothing may reach stdout when the clock cannot be read\n{}",
        describe(&out)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(
        stderr.contains("date") && stderr.contains("perl"),
        "the failure must name the tools that would satisfy it\n{}",
        describe(&out)
    );
}

// ── Sourcing the script must not run the benchmark ────────────────────

#[test]
fn sourcing_in_helper_mode_does_not_run_the_benchmark() {
    let out = run_helper("true", None);

    assert!(out.status.success(), "{}", describe(&out));
    assert!(
        stdout_of(&out).is_empty(),
        "BENCHMARK_SOURCE_ONLY=1 must stop before the benchmark runs — it \
         checks out git refs and stashes work\n{}",
        describe(&out)
    );
}
