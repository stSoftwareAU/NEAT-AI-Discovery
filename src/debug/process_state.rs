//! In-process state block for the SIGUSR1 thread dump (Issue #1934).
//!
//! Everything here is read from this process's own memory: no external tool, no
//! GPU driver call, nothing that a wedged device can stall. On a wedged GPU this
//! block is often more actionable than a raw backtrace — it names the breaker
//! state, the abandoned-thread count, the last heartbeat and the GPU requests
//! whose callers are still waiting, with their ages.
//!
//! It is printed on **every** dump, not only on the fallback path, so a "full
//! dump" and a "no backtraces captured" dump carry the same state.

use std::fmt::Write as _;
use std::sync::LazyLock;
use std::time::Instant;

use crate::analysis::gpu::breaker;
use crate::analysis::gpu::inflight;
use crate::watchdog::HeartbeatSnapshot;

/// When this library was first used, for the elapsed-run-time line.
static PROCESS_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Record the process start reference point.
///
/// Called from `init_debug_handlers()` so the elapsed time in a dump is measured
/// from library initialisation rather than from the first dump.
pub(super) fn mark_start() {
    let _ = *PROCESS_START;
}

/// Append the always-available state block to `out`.
pub(super) fn render(out: &mut String, pid: u32) {
    let _ = writeln!(out, "--- In-process state (no external tools) ---");
    let _ = writeln!(out, "Process ID: {pid}");
    let _ = writeln!(
        out,
        "Elapsed run time: {:.1}s (since library initialisation)",
        PROCESS_START.elapsed().as_secs_f64()
    );

    render_breaker(out);
    render_heartbeat(out);
    render_outstanding_requests(out);
    render_local_backtrace(out);
}

/// GPU circuit-breaker state (Issue #1930) — why GPU work is being refused.
fn render_breaker(out: &mut String) {
    let state = match breaker::gpu_breaker_trip_reason() {
        Some(reason) => format!("TRIPPED (reason: {})", reason.as_str()),
        None => "closed".to_string(),
    };
    let _ = writeln!(out, "GPU circuit breaker: {state}");
    let _ = writeln!(
        out,
        "Abandoned GPU threads: {}",
        breaker::abandoned_gpu_thread_count()
    );
}

/// Watchdog heartbeat — the last stage that reported progress, and how long ago.
fn render_heartbeat(out: &mut String) {
    let line = match crate::watchdog::heartbeat_snapshot() {
        HeartbeatSnapshot::Active { stage, age } => {
            format!("stage \"{stage}\", last beat {:.1}s ago", age.as_secs_f64())
        }
        HeartbeatSnapshot::Inactive => {
            "not active (NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS unset)".to_string()
        }
        HeartbeatSnapshot::Contended => {
            "UNREADABLE — heartbeat lock contended, a thread may be stuck holding it".to_string()
        }
    };
    let _ = writeln!(out, "Last heartbeat: {line}");
}

/// Outstanding GPU requests (Issue #1934) — what the GPU was asked to do.
fn render_outstanding_requests(out: &mut String) {
    let Some(requests) = inflight::outstanding_requests() else {
        let _ = writeln!(
            out,
            "Outstanding GPU requests: UNREADABLE — registry lock contended"
        );
        return;
    };

    if requests.is_empty() {
        let _ = writeln!(out, "Outstanding GPU requests: none");
        return;
    }

    let _ = writeln!(out, "Outstanding GPU requests: {}", requests.len());
    for request in &requests {
        let _ = writeln!(
            out,
            "  - {} (waiting {:.1}s)",
            request.label,
            request.age.as_secs_f64()
        );
    }
}

/// The dumping thread's own backtrace.
///
/// This is the in-process capture that needs no external tool. It covers only
/// the thread running the handler — the banner says "no backtraces captured"
/// precisely because this is not a full per-thread dump.
fn render_local_backtrace(out: &mut String) {
    let current = std::thread::current();
    let _ = writeln!(
        out,
        "\n--- Dumping thread: {} ({:?}) ---",
        current.name().unwrap_or("<unnamed>"),
        current.id()
    );
    let _ = writeln!(out, "{}", std::backtrace::Backtrace::force_capture());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state block must never be empty — that is the failure this fixes.
    #[test]
    fn the_state_block_always_names_the_process_and_gpu_state() {
        let mut out = String::new();
        render(&mut out, 4242);

        assert!(out.contains("Process ID: 4242"), "{out}");
        assert!(out.contains("Elapsed run time:"), "{out}");
        assert!(out.contains("GPU circuit breaker:"), "{out}");
        assert!(out.contains("Abandoned GPU threads:"), "{out}");
        assert!(out.contains("Last heartbeat:"), "{out}");
        assert!(out.contains("Outstanding GPU requests"), "{out}");
        assert!(out.contains("Dumping thread:"), "{out}");
    }

    /// A GPU request that is still waiting must appear with its label.
    #[test]
    fn an_outstanding_request_is_named_in_the_state_block() {
        let _guard = inflight::register("state_block_probe");
        let mut out = String::new();
        render_outstanding_requests(&mut out);
        assert!(
            out.contains("state_block_probe"),
            "outstanding label must be reported: {out}"
        );
        assert!(out.contains("waiting"), "age must be reported: {out}");
    }
}
