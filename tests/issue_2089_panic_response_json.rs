//! Issue #2089 (security sweep chunk 2) — a panic caught at the FFI boundary
//! must still reach the host as parsable JSON.
//!
//! `catch_unwind` hands the caught payload to the boundary's panic formatter,
//! whose output is the response the Deno host receives. That formatter
//! hand-rolled its JSON and escaped only `\` and `"`, so any payload carrying a
//! control character produced a response the host cannot parse — and a newline
//! is exactly what every `assert!` / `assert_eq!` message carries. The
//! documented `{"success":false,"error":…}` contract that `AGENTS.md` requires
//! every controller to branch on therefore degraded into unparsable text at the
//! one moment it matters.
//!
//! The panic is driven through the shipped `extern "C"` entry point
//! (`rank_focus_neurons`), not through the crate-private formatter, per the
//! #1806 convention in `CONTRIBUTING.md`: the host installs a `tracing`
//! subscriber, and a subscriber that panics unwinds out of the analysis path
//! into the entry point's own `catch_unwind` — the production route by which a
//! real panic payload reaches the formatter.
//!
//! This file is its own test binary so the process-wide `tracing` default and
//! panic hook it installs cannot reach any other suite.

use std::ffi::{CStr, CString};
use std::panic;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::subscriber::with_default;
use tracing::{Event, Metadata, Subscriber, span};

/// What the host subscriber panics with, mirroring the payload shapes
/// `std::panic` actually produces.
#[derive(Clone)]
enum Payload {
    /// `panic!("literal")` — payload is `&'static str`.
    StaticStr(&'static str),
    /// `panic!("{formatted}")` / `assert_eq!` — payload is `String`.
    Owned(String),
    /// `panic_any(non_string)` — neither downcast arm matches.
    Other(i32),
}

/// A subscriber standing in for a host that installed a faulty layer: it
/// panics the moment the library emits an event.
struct PanicOnEvent {
    payload: Payload,
}

impl Subscriber for PanicOnEvent {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    fn event(&self, _event: &Event<'_>) {
        match &self.payload {
            Payload::StaticStr(s) => panic::panic_any(*s),
            Payload::Owned(s) => panic::panic_any(s.clone()),
            Payload::Other(v) => panic::panic_any(*v),
        }
    }

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

/// Counts the events a quiet subscriber sees, so the fixture's own
/// precondition — that this input makes the library emit at all — is asserted
/// rather than assumed.
struct CountEvents(Arc<AtomicUsize>);

impl Subscriber for CountEvents {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    fn event(&self, _event: &Event<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

/// A `rank_focus_neurons` request that always reaches an emitting code path:
/// the sub-minimum analysis deadline makes the removal-triage skip log a WARN
/// on every call (Issue #4139), so the fixture cannot go quiet behind our back.
const REQUEST: &str = r#"{
    "parquetFile": "/tmp/issue-2089-does-not-need-to-exist.parquet",
    "creature": {
        "neurons": [
            { "uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY", "bias": 0.0 },
            { "uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 0.0 }
        ],
        "synapses": [
            { "from_uuid": "hidden-0", "to_uuid": "output-0", "weight": 1.0 }
        ],
        "input": 1,
        "output": 1
    },
    "analysisDeadlineMs": 1
}"#;

/// Call the shipped entry point under `subscriber` and read its response back,
/// freeing the pointer through the documented `free_discovery_result` route.
fn rank_focus_under<S>(subscriber: S) -> String
where
    S: Subscriber + Send + Sync,
{
    let request = CString::new(REQUEST).expect("fixture has no interior NUL");
    let ptr = with_default(subscriber, || {
        // SAFETY: `request` is a valid null-terminated C string that outlives
        // the call, which is the entry point's documented input contract.
        unsafe { neat_ai_discovery::ffi::rank_focus_neurons(request.as_ptr()) }
    });
    assert!(!ptr.is_null(), "the entry point must never return null");
    // SAFETY: the entry point returns a null-terminated C string it allocated.
    let response = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("FFI response must be valid UTF-8")
        .to_string();
    // SAFETY: `ptr` came from this call and is freed exactly once.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(ptr) };
    response
}

fn parse(response: &str) -> serde_json::Value {
    serde_json::from_str(response).unwrap_or_else(|e| {
        panic!("FFI response must be valid JSON ({e}): {response:?}");
    })
}

/// Assert the response is the *panic* response, not an ordinary error — the
/// positive precondition that stops this test passing vacuously if the panic
/// path ever stops firing.
fn panic_error(response: &str) -> String {
    let json = parse(response);
    assert_eq!(
        json["success"], false,
        "a caught panic must report success:false, got: {response}"
    );
    let error = json["error"]
        .as_str()
        .unwrap_or_else(|| panic!("a caught panic must carry an error string: {response}"))
        .to_string();
    assert!(
        error.starts_with("Internal panic caught:"),
        "the panic path did not fire — this fixture no longer reaches it: {error:?}"
    );
    error
}

/// Silence the default hook for the deliberate panics below, restoring it
/// afterwards. The whole suite is one test in one binary, so nothing else can
/// observe the swap.
fn with_quiet_panic_hook<T>(body: impl FnOnce() -> T) -> T {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = body();
    panic::set_hook(previous);
    outcome
}

#[test]
fn a_panic_caught_at_the_ffi_boundary_reaches_the_host_as_parsable_json() {
    // Warm-up under the real subscriber: `log_version_once` is a one-shot, so
    // spending it here leaves the per-call WARN as the only event the payload
    // cases below can trip. Also pins the happy path — a well-formed request
    // answers `success: true`.
    let baseline = rank_focus_under(CountEvents(Arc::new(AtomicUsize::new(0))));
    assert_eq!(
        parse(&baseline)["success"],
        true,
        "the fixture must be a valid request, so a failure below is the panic: {baseline}"
    );

    // Precondition: this request really does make the library emit an event,
    // which is the hook the panic cases ride.
    let events = Arc::new(AtomicUsize::new(0));
    let _ = rank_focus_under(CountEvents(Arc::clone(&events)));
    assert!(
        events.load(Ordering::Relaxed) > 0,
        "the fixture must reach an emitting code path, or the panic cases prove nothing"
    );

    with_quiet_panic_hook(|| {
        // The realistic trigger: a standard `assert_eq!` failure message. Its
        // embedded newlines made the hand-rolled response unparsable.
        let assertion = "assertion `left == right` failed\n  left: 1\n right: 2";
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::Owned(assertion.to_string()),
        }));
        assert!(
            error.ends_with(assertion),
            "the panic message must survive escaping intact: {error:?}"
        );

        // Every control character, not just the newline: tab, carriage return,
        // form feed and a bare C0 byte alike.
        let control = "tab\there\rreturn\u{000c}form\u{0001}c0-byte\nnewline";
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::Owned(control.to_string()),
        }));
        assert!(
            error.ends_with(control),
            "the payload must round-trip byte-for-byte through the escape: {error:?}"
        );

        // A `&'static str` payload takes the other downcast arm.
        let literal = "first line\nsecond line";
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::StaticStr(literal),
        }));
        assert!(error.ends_with(literal), "got: {error:?}");

        // Quotes and backslashes kept working — the fix must not regress the
        // escaping the hand-rolled formatter did get right.
        let quoted = r#"path "C:\temp\x" not found"#;
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::Owned(quoted.to_string()),
        }));
        assert!(error.ends_with(quoted), "got: {error:?}");

        // An oversized payload stays truncated (Issue #1365) and parsable,
        // including when the cut lands beside a control character.
        let oversized = format!("{}\ntail", "line\n".repeat(4096));
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::Owned(oversized),
        }));
        assert!(
            error.len() < 16_384,
            "the embedded payload must stay bounded, got {} bytes",
            error.len()
        );

        // A payload of neither string type still yields a well-formed response.
        let error = panic_error(&rank_focus_under(PanicOnEvent {
            payload: Payload::Other(42),
        }));
        assert!(error.contains("Unknown panic"), "got: {error:?}");
    });
}
