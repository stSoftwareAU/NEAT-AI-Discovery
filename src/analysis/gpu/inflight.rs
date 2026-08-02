//! Outstanding GPU request registry (Issue #1934).
//!
//! When the GPU wedges, the single most actionable datum is *what the GPU was
//! asked to do and how long ago*. The queue itself cannot answer that: a
//! submitted request lives inside a crossbeam channel or inside the GPU
//! thread, neither of which is inspectable from a SIGUSR1 handler.
//!
//! Every submission registers a short label here for exactly as long as its
//! caller is waiting, so the thread dump can list the outstanding requests even
//! when no backtrace can be captured at all.
//!
//! ```text
//! submit ──▶ register("helpful_batch") ──▶ InflightGuard held while waiting
//!                                              │
//! SIGUSR1 ──▶ outstanding_requests() ──────────┘  (label + age)
//! ```
//!
//! The registry must never block a wedged process: readers use a bounded
//! `try_lock_for()` and report a contended lock rather than waiting on it.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Longest a dump reader will wait for the registry lock.
///
/// Writers hold it only for a push/remove, so contention means something is
/// badly wrong — and the dump must degrade rather than block.
const READ_LOCK_TIMEOUT: Duration = Duration::from_millis(50);

/// Monotonic id source so a guard can remove exactly its own entry.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The requests whose callers are still waiting.
static INFLIGHT: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

struct Entry {
    id: u64,
    label: &'static str,
    started: Instant,
}

/// A snapshot of one outstanding request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutstandingRequest {
    /// Short operation label, e.g. `helpful_batch`.
    pub label: &'static str,
    /// How long the caller has been waiting.
    pub age: Duration,
}

/// Keeps a request listed as outstanding until it is dropped.
///
/// Held beside the caller's response receiver, so the registry empties itself
/// whether the wait succeeded, timed out or was cancelled.
#[derive(Debug)]
pub(crate) struct InflightGuard {
    id: u64,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let mut guard = INFLIGHT.lock();
        guard.retain(|entry| entry.id != self.id);
    }
}

/// Register a request as outstanding for the lifetime of the returned guard.
pub(crate) fn register(label: &'static str) -> InflightGuard {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    INFLIGHT.lock().push(Entry {
        id,
        label,
        started: Instant::now(),
    });
    InflightGuard { id }
}

/// Snapshot the outstanding requests, oldest first.
///
/// Returns `None` — never blocks — when the registry lock cannot be taken
/// within a short bound, so a caller can report the contention rather
/// than silently showing an empty list.
#[must_use]
pub fn outstanding_requests() -> Option<Vec<OutstandingRequest>> {
    let guard = INFLIGHT.try_lock_for(READ_LOCK_TIMEOUT)?;
    let mut requests: Vec<OutstandingRequest> = guard
        .iter()
        .map(|entry| OutstandingRequest {
            label: entry.label,
            age: entry.started.elapsed(),
        })
        .collect();
    drop(guard);
    requests.sort_by_key(|request| std::cmp::Reverse(request.age));
    Some(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Labels are only visible while the caller is waiting.
    #[test]
    fn a_registered_request_is_listed_until_its_guard_drops() {
        let guard = register("test_op_listed");
        let listed = outstanding_requests().expect("registry readable");
        assert!(
            listed.iter().any(|r| r.label == "test_op_listed"),
            "an outstanding request must be listed: {listed:?}"
        );

        drop(guard);
        let listed = outstanding_requests().expect("registry readable");
        assert!(
            !listed.iter().any(|r| r.label == "test_op_listed"),
            "a completed request must not linger: {listed:?}"
        );
    }

    /// Each guard removes only its own entry, so concurrent submissions of the
    /// same operation do not erase one another.
    #[test]
    fn dropping_one_guard_leaves_its_sibling_registered() {
        let first = register("test_op_sibling");
        let second = register("test_op_sibling");

        drop(first);
        let listed = outstanding_requests().expect("registry readable");
        assert_eq!(
            listed
                .iter()
                .filter(|r| r.label == "test_op_sibling")
                .count(),
            1,
            "exactly one sibling remains: {listed:?}"
        );

        drop(second);
        let listed = outstanding_requests().expect("registry readable");
        assert_eq!(
            listed
                .iter()
                .filter(|r| r.label == "test_op_sibling")
                .count(),
            0
        );
    }

    /// The age is what makes the label actionable on a wedge.
    #[test]
    fn an_outstanding_request_reports_its_age() {
        let _guard = register("test_op_age");
        std::thread::sleep(Duration::from_millis(20));
        let listed = outstanding_requests().expect("registry readable");
        let entry = listed
            .iter()
            .find(|r| r.label == "test_op_age")
            .expect("registered request present");
        assert!(
            entry.age >= Duration::from_millis(15),
            "age must reflect the wait, got {:?}",
            entry.age
        );
    }
}
