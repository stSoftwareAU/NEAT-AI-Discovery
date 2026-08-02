//! External sampler capture for the SIGUSR1 thread dump (Issue #1934).
//!
//! On macOS the `sample` tool produces full thread backtraces, which is the
//! most useful thing a wedged process can emit. But `sample` itself can hang on
//! a wedged driver — and when it did on the M2 Ultra the dump contained nothing
//! at all, which is exactly when it was needed.
//!
//! This module owns the bounded attempt to run the sampler and classifies the
//! result so the caller can fall back and label the dump honestly:
//!
//! ```text
//! sample exits 0, file readable ──▶ Full           ("full dump")
//! sample times out, file readable ─▶ Partial       ("partial dump")
//! anything else ──────────────────▶ NoBacktraces  ("no backtraces captured")
//! ```
//!
//! The sampler is only attempted when one exists: on macOS by default, and
//! elsewhere only when `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` names one (see
//! [`crate::config::sample_program_override`]). That same
//! indirection is what lets the fallback paths be tested off macOS.

use std::fmt::Write as _;
use std::thread;
use std::time::Duration;

use super::DumpCompleteness;

/// Hard cap on how long the sampler may run before it is killed.
///
/// Deliberately short: the fallback state block below is now always emitted, so
/// there is nothing to gain from making a wedged process wait longer for a tool
/// that is itself wedged.
pub(super) const SAMPLE_TIMEOUT_SECS: u64 = 5;

/// How long we wait for a killed sampler to actually die before giving up on it.
pub(super) const SAMPLE_KILL_GRACE_MS: u64 = 500;

/// The sampler to run, or `None` when this platform has no sampler configured.
fn sampler_program() -> Option<String> {
    if let Some(program) = crate::config::sample_program_override() {
        return Some(program);
    }
    if cfg!(target_os = "macos") {
        return Some("sample".to_string());
    }
    None
}

/// Attempt a full thread-backtrace capture, appending the narrative to `out`.
///
/// Never blocks for longer than [`SAMPLE_TIMEOUT_SECS`] plus
/// [`SAMPLE_KILL_GRACE_MS`], whatever the sampler does.
pub(super) fn capture(pid: u32, out: &mut String) -> DumpCompleteness {
    let Some(program) = sampler_program() else {
        let _ = writeln!(
            out,
            "--- No thread sampler on this platform \
             (set NEAT_AI_DISCOVERY_SAMPLE_PROGRAM to name one) ---"
        );
        #[cfg(not(target_os = "macos"))]
        let _ = writeln!(
            out,
            "Try manually: gdb -p {pid} -ex 'thread apply all bt' -ex 'quit'"
        );
        return DumpCompleteness::NoBacktraces;
    };

    let _ = writeln!(
        out,
        "--- Running '{program}' for full thread analysis (1 second) ---\n"
    );

    // IMPORTANT: Do NOT pipe stdout here. `sample` can emit a lot of output; if we
    // pipe it and don't continuously drain the pipe, the child can block forever
    // once the buffer fills. That manifests exactly as "sample did not exit".
    //
    // Issue #1905 owns the predictability of this path; keep the construction in
    // one place so that fix has a single site to change.
    let out_path = std::env::temp_dir().join(format!(
        "neat_ai_discovery.sample.{pid}.{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis())
    ));
    let out_path_str = out_path.to_string_lossy().to_string();

    let args = vec![
        pid.to_string(),
        "1".to_string(),
        "-mayDie".to_string(),
        "-file".to_string(),
        out_path_str.clone(),
    ];

    let run = match run_external_command_with_timeout(
        &program,
        &args,
        Duration::from_secs(SAMPLE_TIMEOUT_SECS),
        Duration::from_millis(SAMPLE_KILL_GRACE_MS),
    ) {
        Ok(run) => run,
        Err(e) => {
            let _ = writeln!(out, "Failed to run '{program}': {e}");
            write_manual_hint(out, pid);
            return DumpCompleteness::NoBacktraces;
        }
    };

    // A killed child that never reported a status is the same situation as a
    // timeout: whatever landed in the file is all we will ever get.
    if run.timed_out || run.status.is_none() {
        let _ = writeln!(
            out,
            "[NEAT-AI-Discovery][debug] WARNING: '{program}' did not exit within \
             {SAMPLE_TIMEOUT_SECS}s. Attempting to print any partial output captured so far."
        );
        return match read_non_empty(&out_path) {
            Some(contents) => {
                let _ = writeln!(
                    out,
                    "[NEAT-AI-Discovery][debug] Partial '{program}' output saved to: {out_path_str}\n"
                );
                write_filtered_sample_output(out, &contents, &out_path_str);
                DumpCompleteness::Partial
            }
            None => {
                let _ = writeln!(
                    out,
                    "[NEAT-AI-Discovery][debug] WARNING: no readable '{program}' output at {out_path_str}."
                );
                write_manual_hint(out, pid);
                DumpCompleteness::NoBacktraces
            }
        };
    }

    // Unreachable: the `None` status case returned above. Handled explicitly
    // rather than unwrapped so a future edit cannot panic in a signal handler.
    let Some(status) = run.status else {
        write_manual_hint(out, pid);
        return DumpCompleteness::NoBacktraces;
    };
    if !status.success() {
        let _ = writeln!(out, "'{program}' command failed (exit code: {status}).");
        write_manual_hint(out, pid);
        return DumpCompleteness::NoBacktraces;
    }

    match read_non_empty(&out_path) {
        Some(contents) => {
            let _ = writeln!(
                out,
                "[NEAT-AI-Discovery][debug] Full '{program}' output saved to: {out_path_str}\n"
            );
            write_filtered_sample_output(out, &contents, &out_path_str);
            DumpCompleteness::Full
        }
        None => {
            let _ = writeln!(
                out,
                "[NEAT-AI-Discovery][debug] '{program}' exited successfully but wrote no readable \
                 output to {out_path_str}."
            );
            write_manual_hint(out, pid);
            DumpCompleteness::NoBacktraces
        }
    }
}

/// Read the sampler's output file, treating missing/empty/unreadable alike.
///
/// An empty file is not evidence of anything, so it must not be reported as a
/// successful capture (Issue #1934).
fn read_non_empty(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .filter(|contents| !contents.trim().is_empty())
}

fn write_manual_hint(out: &mut String, pid: u32) {
    let _ = writeln!(
        out,
        "Try manually: sample {pid} 1 -mayDie -file /tmp/sample.txt"
    );
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExternalCommandRun {
    pub(super) status: Option<std::process::ExitStatus>,
    pub(super) timed_out: bool,
}

/// Run an external command with a hard timeout, avoiding pipe backpressure.
///
/// IMPORTANT: This helper must never hang. It intentionally discards stdout/stderr
/// to avoid deadlocks when a child writes more than a pipe buffer and the parent
/// isn't continuously draining it.
pub(super) fn run_external_command_with_timeout(
    program: &str,
    args: &[String],
    timeout: Duration,
    kill_grace: Duration,
) -> std::io::Result<ExternalCommandRun> {
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Ok(ExternalCommandRun {
                    status: Some(status),
                    timed_out: false,
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();

                    // Never block indefinitely waiting for the child to die.
                    let kill_start = Instant::now();
                    while kill_start.elapsed() < kill_grace {
                        match child.try_wait() {
                            Ok(Some(status)) => {
                                return Ok(ExternalCommandRun {
                                    status: Some(status),
                                    timed_out: true,
                                });
                            }
                            Ok(None) => thread::sleep(Duration::from_millis(25)),
                            Err(_) => break,
                        }
                    }

                    return Ok(ExternalCommandRun {
                        status: None,
                        timed_out: true,
                    });
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Filter sample output down to the most relevant thread information.
fn write_filtered_sample_output(out: &mut String, output: &str, out_path: &str) {
    let mut in_call_graph = false;
    let mut thread_count = 0;
    let mut printed = 0_usize;

    for line in output.lines() {
        // Start of call graph section
        if line.starts_with("Call graph:") {
            in_call_graph = true;
            let _ = writeln!(out, "{line}");
            printed += 1;
            continue;
        }

        // End markers
        if line.starts_with("Total number in stack") || line.starts_with("Binary Images:") {
            if in_call_graph {
                let _ = writeln!(
                    out,
                    "\n--- End of call graph ({thread_count} threads) ---\n"
                );
            }
            in_call_graph = false;
            continue;
        }

        if in_call_graph {
            // Thread headers
            if line.contains("Thread_") {
                thread_count += 1;
                let _ = writeln!(out, "\n{line}");
                printed += 1;
            }
            // Show lines containing our library or interesting keywords
            else if line.contains("neat_ai_discovery")
                || line.contains("wgpu")
                || line.contains("metal")
                || line.contains("Metal")
                || line.contains("crossbeam")
                || line.contains("rayon")
                || line.contains("recv")
                || line.contains("poll")
                || line.contains("wait")
                || line.contains("park")
                || line.contains("sleep")
                || line.contains("pthread_cond")
                || line.contains("kevent")
            {
                let _ = writeln!(out, "{line}");
                printed += 1;
            }
        }
    }

    if printed == 0 {
        let _ = writeln!(
            out,
            "(no call-graph lines matched the filter — read the full capture at {out_path})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[cfg(unix)]
    fn run_external_command_with_timeout_does_not_hang_and_preserves_partial_output() {
        use std::os::unix::fs::PermissionsExt;

        // Create a small shell script that writes to a file immediately, then hangs.
        // This simulates the "child doesn't exit promptly" scenario while ensuring we
        // can still read partial output from the file after we kill it.
        let tmp = std::env::temp_dir();
        let script_path = tmp.join(format!(
            "neat_ai_discovery_test_hang_{}_{}.sh",
            std::process::id(),
            super::super::chrono_lite_timestamp().replace(' ', "_")
        ));
        let out_path = tmp.join(format!(
            "neat_ai_discovery_test_output_{}_{}.txt",
            std::process::id(),
            super::super::chrono_lite_timestamp().replace(' ', "_")
        ));
        // Pre-create the output file so the test can't fail with "not found" if the
        // child is killed before it gets scheduled.
        std::fs::write(&out_path, "").expect("precreate output file");

        // The script writes a sentinel line, then hangs on `sleep`.
        // Using a single `echo` keeps I/O minimal so even slow runners flush before
        // the timeout fires.
        let script = "#!/bin/sh\n\
             echo \"Call graph:\" > \"$1\"\n\
             echo \"Thread_0\" >> \"$1\"\n\
             # Signal that output is ready via a separate sentinel file.\n\
             touch \"$1.ready\"\n\
             sleep 60\n"
            .to_string();
        std::fs::write(&script_path, script).expect("write test script");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("chmod");

        // Spawn the child ourselves first so we can wait for it to finish writing
        // before we exercise the timeout/kill logic in `run_external_command_with_timeout`.
        let args = vec![out_path.to_string_lossy().to_string()];
        {
            use std::process::{Command, Stdio};
            let mut child = Command::new(script_path.to_string_lossy().as_ref())
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn pre-run child");

            // Wait for the sentinel file that proves the script flushed its output.
            let sentinel = format!("{}.ready", out_path.display());
            let poll_start = Instant::now();
            while poll_start.elapsed() < Duration::from_secs(10) {
                if std::path::Path::new(&sentinel).exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&sentinel);
        }

        // Verify that the pre-run child wrote partial output.
        let contents = std::fs::read_to_string(&out_path).expect("read partial output");
        assert!(
            contents.contains("Call graph:") && contents.contains("Thread_0"),
            "expected partial output, got: {contents:?}"
        );

        // Now exercise the function under test – a fresh invocation that will time out.
        // Reset output file so the second child writes fresh.
        std::fs::write(&out_path, "").expect("reset output file");
        let start = Instant::now();
        let run = run_external_command_with_timeout(
            script_path.to_string_lossy().as_ref(),
            &args,
            Duration::from_secs(3),
            Duration::from_millis(500),
        )
        .expect("run");

        assert!(run.timed_out, "expected timeout");
        // Touch `status` so it doesn't get optimised into "dead code" on non-macOS test builds.
        // (It is used by the macOS `sample` path.)
        let _ = run.status;
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "expected quick return, got {:#?}",
            start.elapsed()
        );

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_file(&out_path);
    }

    /// An empty capture file is not evidence — it must not read as success.
    #[test]
    fn an_empty_output_file_is_not_readable_content() {
        let path = std::env::temp_dir().join(format!(
            "neat_ai_discovery_empty_{}_{}.txt",
            std::process::id(),
            super::super::chrono_lite_timestamp().replace(' ', "_")
        ));
        std::fs::write(&path, "   \n\n").expect("write empty-ish file");
        assert!(read_non_empty(&path).is_none(), "whitespace is not output");

        std::fs::write(&path, "Call graph:\n").expect("write content");
        assert!(read_non_empty(&path).is_some(), "real content is readable");

        let _ = std::fs::remove_file(&path);
        assert!(
            read_non_empty(&path).is_none(),
            "a missing file yields no output"
        );
    }

    /// Output that matches nothing must say so rather than print an empty block.
    #[test]
    fn unmatched_call_graph_lines_are_reported_not_silently_dropped() {
        let mut out = String::new();
        write_filtered_sample_output(&mut out, "nothing interesting here\n", "/tmp/x.txt");
        assert!(
            out.contains("no call-graph lines matched"),
            "unmatched output must be reported: {out}"
        );
    }

    /// Thread headers and library frames survive the filter.
    #[test]
    fn call_graph_thread_lines_are_kept() {
        let sample = "Call graph:\n\
             Thread_1234\n\
             +  1000 neat_ai_discovery::analysis::gpu\n\
             +  1000 unrelated_symbol\n\
             Binary Images:\n";
        let mut out = String::new();
        write_filtered_sample_output(&mut out, sample, "/tmp/x.txt");
        assert!(out.contains("Thread_1234"), "thread header kept: {out}");
        assert!(
            out.contains("neat_ai_discovery"),
            "library frame kept: {out}"
        );
        assert!(
            out.contains("End of call graph (1 threads)"),
            "thread count reported: {out}"
        );
    }
}
