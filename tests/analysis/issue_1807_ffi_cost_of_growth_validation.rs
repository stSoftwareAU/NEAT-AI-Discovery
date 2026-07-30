//! Issue #1807 — `costOfGrowth` validation must hold on the **shipped** FFI
//! path, not only on the in-crate adapter.
//!
//! The guard lives in `effective_cost_of_growth`, reached from
//! `identify_structural_removal_candidates`, which `rank_focus_neurons_internal`
//! calls. A zero or negative cost makes `boosted_savings <= contribution` true
//! for every neuron, so the removal candidates vanish with no diagnostic (the
//! silent-empty-result class of #1782); a `NaN` cost inverts the comparison and
//! makes *every* hidden neuron a candidate. Both are caller bugs, so the value
//! is replaced with [`DEFAULT_COST_OF_GROWTH`] and the substitution is logged at
//! WARN — never applied silently.
//!
//! These tests drive the FFI entry point with the JSON a caller actually sends.
//! `NaN` and `Infinity` are not JSON literals, so:
//!
//! * **infinity** is reached the way a real caller reaches it — a JSON number
//!   that overflows `f32` (`1e39`), which deserialises to `f32::INFINITY`;
//! * **underflow** (`1e-60` → `0.0`) is the same trap in the other direction and
//!   is FFI-specific: the request looks positive but the criterion sees zero;
//! * a literal `NaN` token is rejected loudly at the JSON boundary, and the
//!   `NaN` half of the guard is pinned on the shipped criterion through the
//!   public `triage_removal_candidates` adapter, which delegates to it.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use neat_ai_discovery::focus::{DEFAULT_COST_OF_GROWTH, triage_removal_candidates};
use neat_ai_discovery::rank_focus_neurons_internal;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serde_json::{Value, json};
use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// The noise-floor env var, unset so the tests depend on the compiled default.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";

/// A discovery parquet that cannot be opened — the focus path is structure-only
/// (Issue #1766), so a successful response also proves no decode was attempted.
const MISSING_PARQUET: &str = "/nonexistent/issue-1807/discovery.parquet";

/// Cost-of-growth large enough to clear the noise floor, used to prove the
/// fallback list differs from a *valid* run rather than being empty either way.
const VALID_COST_OF_GROWTH: f32 = 1e-4;

/// RAII guard that unsets an env var for a single test and restores it after.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(v) = &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            unsafe { std::env::set_var(self.key, v) };
        }
    }
}

// ---------------------------------------------------------------------------
// Tracing capture — the substitution must be observable, not silent.
// ---------------------------------------------------------------------------

/// One captured WARN event: its message plus its structured fields.
#[derive(Debug, Clone)]
struct CapturedWarn {
    message: String,
    fields: HashMap<String, String>,
}

#[derive(Default)]
struct FieldVisitor {
    message: String,
    fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.store(field.name(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.store(field.name(), value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.store(field.name(), value.to_string());
    }
}

impl FieldVisitor {
    fn store(&mut self, name: &str, value: String) {
        if name == "message" {
            self.message = value;
        } else {
            self.fields.insert(name.to_string(), value);
        }
    }
}

struct WarnCaptureLayer {
    events: Arc<Mutex<Vec<CapturedWarn>>>,
}

impl<S: tracing::Subscriber> Layer<S> for WarnCaptureLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture buffer poisoned")
            .push(CapturedWarn {
                message: visitor.message,
                fields: visitor.fields,
            });
    }
}

/// Run `body` with a capturing subscriber installed on this thread, returning
/// its result alongside every WARN event emitted.
fn capture_warnings<T, F: FnOnce() -> T>(body: F) -> (T, Vec<CapturedWarn>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(WarnCaptureLayer {
        events: Arc::clone(&events),
    });
    let result = tracing::subscriber::with_default(subscriber, body);
    let captured = events.lock().expect("capture buffer poisoned").clone();
    (result, captured)
}

/// The rejection WARN, identified by its structured field rather than wording.
fn cost_of_growth_warning(events: &[CapturedWarn]) -> Option<&CapturedWarn> {
    events
        .iter()
        .find(|e| e.fields.contains_key("invalid_cost_of_growth"))
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn neuron(uuid: &str, ntype: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: ntype.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// One input feeding a dominant hidden neuron and three negligible ones, ordered
/// inputs → hidden → output so the forward-only FFI gate (Issue #1184) passes.
fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("h-high", "hidden"),
            neuron("h-low-a", "hidden"),
            neuron("h-low-b", "hidden"),
            neuron("h-low-c", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-high", 0.5),
            synapse("in-0", "h-low-a", 0.5),
            synapse("in-0", "h-low-b", 0.5),
            synapse("in-0", "h-low-c", 0.5),
            synapse("h-high", "out", 1.0),
            synapse("h-low-a", "out", 1e-8),
            synapse("h-low-b", "out", 1e-8),
            synapse("h-low-c", "out", 1e-8),
        ],
        input: 1,
        output: 1,
    }
}

/// Drive `rank_focus_neurons_internal` with a raw `costOfGrowth` JSON value.
fn ffi_focus_response(creature: &CreatureJson, cost_of_growth: Value) -> Value {
    let input = json!({
        "parquetFile": MISSING_PARQUET,
        "creature": creature,
        "maxResults": 64,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": cost_of_growth,
    })
    .to_string();
    serde_json::from_str(&rank_focus_neurons_internal(&input).expect("FFI focus path"))
        .expect("FFI response JSON")
}

/// A successful response's removal candidates and noise-floor rejections — the
/// two observable outputs the cost-of-growth drives.
fn removal_outcome(response: &Value) -> (Vec<Value>, Value) {
    assert_eq!(
        response["success"], true,
        "the structure-only FFI focus path must succeed: {response:?}"
    );
    let candidates = response["removalCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    (candidates, response["rejectionBreakdown"].clone())
}

/// Every `costOfGrowth` the FFI can carry that the criterion must reject, with
/// the `f32` value it deserialises to.
///
/// `1e39` overflows `f32` to `+∞` and `1e-60` underflows to `0.0`: both look
/// like ordinary JSON numbers to a caller, which is why the guard has to sit
/// behind the deserialiser rather than in front of it.
fn invalid_ffi_costs() -> Vec<(Value, f32)> {
    vec![
        (json!(0.0), 0.0),
        (json!(-1.0), -1.0),
        (json!(-1e-4), -1e-4),
        (json!(1e39), f32::INFINITY),
        (json!(-1e39), f32::NEG_INFINITY),
        (json!(1e-60), 0.0),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Acceptance: a non-finite or non-positive `costOfGrowth` arriving over FFI is
/// replaced with [`DEFAULT_COST_OF_GROWTH`], producing exactly the outcome an
/// omitted `costOfGrowth` produces — never nonsense savings.
#[test]
#[serial]
fn invalid_ffi_cost_of_growth_falls_back_to_the_default() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let (default_candidates, default_rejections) =
        removal_outcome(&ffi_focus_response(&creature, Value::Null));
    let (explicit_candidates, explicit_rejections) = removal_outcome(&ffi_focus_response(
        &creature,
        json!(DEFAULT_COST_OF_GROWTH),
    ));
    assert_eq!(
        explicit_candidates, default_candidates,
        "passing the default explicitly must match omitting it, or the comparison below proves nothing"
    );
    assert_eq!(explicit_rejections, default_rejections);

    for (payload, deserialised) in invalid_ffi_costs() {
        let (candidates, rejections) =
            removal_outcome(&ffi_focus_response(&creature, payload.clone()));
        assert_eq!(
            candidates, default_candidates,
            "costOfGrowth {payload} (f32 {deserialised}) must fall back to the default candidate list"
        );
        assert_eq!(
            rejections, default_rejections,
            "costOfGrowth {payload} (f32 {deserialised}) must fall back to the default rejection breakdown"
        );
    }
}

/// Acceptance: the substitution is **observable**. Each rejected value is logged
/// at WARN carrying both the offending value and the substituted default, so an
/// operator can root-cause an unexpected candidate list from the log alone.
#[test]
#[serial]
fn invalid_ffi_cost_of_growth_is_logged_at_warn() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    for (payload, deserialised) in invalid_ffi_costs() {
        let (_, events) = capture_warnings(|| ffi_focus_response(&creature, payload.clone()));
        let warning = cost_of_growth_warning(&events).unwrap_or_else(|| {
            panic!("costOfGrowth {payload} must be reported at WARN, not substituted silently")
        });

        let reported: f32 = warning.fields["invalid_cost_of_growth"]
            .parse()
            .unwrap_or_else(|e| panic!("invalid_cost_of_growth must be numeric: {e}"));
        assert_eq!(
            reported.is_nan(),
            deserialised.is_nan(),
            "the WARN must report the value the criterion saw for {payload}"
        );
        if !deserialised.is_nan() {
            assert_eq!(
                reported, deserialised,
                "the WARN must report the value the criterion saw for {payload}"
            );
        }

        let substituted: f32 = warning.fields["default_cost_of_growth"]
            .parse()
            .unwrap_or_else(|e| panic!("default_cost_of_growth must be numeric: {e}"));
        assert_eq!(
            substituted, DEFAULT_COST_OF_GROWTH,
            "the WARN must name the substituted default for {payload}"
        );
        assert!(
            warning.message.contains("costOfGrowth"),
            "the WARN message must name the offending knob; got {:?}",
            warning.message
        );
    }
}

/// A **valid** `costOfGrowth` is used as given: it is not swept into the
/// fallback, and it emits no rejection WARN.
#[test]
#[serial]
fn a_valid_ffi_cost_of_growth_is_neither_replaced_nor_warned_about() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let (default_candidates, _) = removal_outcome(&ffi_focus_response(&creature, Value::Null));
    let (response, events) =
        capture_warnings(|| ffi_focus_response(&creature, json!(VALID_COST_OF_GROWTH)));
    let (candidates, _) = removal_outcome(&response);

    assert!(
        cost_of_growth_warning(&events).is_none(),
        "a valid costOfGrowth must not be reported as invalid: {events:?}"
    );
    assert!(
        !candidates.is_empty(),
        "a cost-of-growth clearing the noise floor must yield candidates: {response:?}"
    );
    assert_ne!(
        candidates, default_candidates,
        "the fallback assertions are only meaningful because a valid value gives a different list"
    );
}

/// `NaN` and `Infinity` are not JSON literals: a caller writing one gets a loud
/// parse failure, not a silently ignored field that leaves the criterion running
/// on a default it never asked for.
#[test]
#[serial]
fn non_json_cost_of_growth_tokens_are_rejected_loudly() {
    let creature = make_creature();

    for token in ["NaN", "Infinity", "-Infinity"] {
        let input = json!({
            "parquetFile": MISSING_PARQUET,
            "creature": creature,
            "costOfGrowth": null,
        })
        .to_string()
        .replace(
            "\"costOfGrowth\":null",
            &format!("\"costOfGrowth\":{token}"),
        );
        let response: Value =
            serde_json::from_str(&rank_focus_neurons_internal(&input).expect("FFI focus path"))
                .expect("FFI response JSON");

        assert_eq!(
            response["success"], false,
            "costOfGrowth {token} is not JSON and must fail loudly: {response:?}"
        );
        assert_eq!(
            response["errorKind"], "data_validation",
            "a malformed request must be classified as invalid input: {response:?}"
        );
    }
}

/// The `NaN` half of the guard, pinned on the shipped criterion through the
/// public `triage_removal_candidates` adapter (it delegates to
/// `identify_structural_removal_candidates`, the function the FFI calls).
///
/// Without the guard, `NaN` savings make `savings <= contribution` false for
/// every hidden neuron, so all of them would be offered for removal.
#[test]
#[serial]
fn nan_and_infinite_cost_of_growth_fall_back_on_the_shipped_criterion() {
    let _floor = EnvVarGuard::unset(NOISE_FLOOR_ENV);
    let creature = make_creature();

    let expected = triage_removal_candidates(&creature, None);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
        let (triage, events) =
            capture_warnings(|| triage_removal_candidates(&creature, Some(invalid)));
        assert_eq!(
            triage, expected,
            "cost-of-growth {invalid} must fall back to the default triage"
        );
        assert!(
            cost_of_growth_warning(&events).is_some(),
            "cost-of-growth {invalid} must be reported at WARN: {events:?}"
        );
    }

    // The guard is what stops NaN from nominating every hidden neuron.
    let (nan_triage, _) = capture_warnings(|| triage_removal_candidates(&creature, Some(f32::NAN)));
    assert!(
        nan_triage.candidates.len() < 3,
        "a NaN cost must not nominate the whole hidden layer: {:?}",
        nan_triage.candidates
    );
}

/// Drift guard for the second half of Issue #1807: the default must exist in
/// exactly one place. A re-introduced literal (`unwrap_or(1e-7)`) agrees with
/// [`DEFAULT_COST_OF_GROWTH`] today and silently disagrees the moment the
/// constant is retuned — which is how the FFI path drifted in the first place.
#[test]
fn no_cost_of_growth_default_literal_survives_outside_the_constant() {
    let mut offenders = Vec::new();
    for file in rust_sources("src".as_ref()) {
        let text = std::fs::read_to_string(&file).expect("source file must be readable");
        for (index, line) in text.lines().enumerate() {
            let mentions_knob = line.contains("cost_of_growth")
                || line.contains("costOfGrowth")
                || line.contains("cost of growth");
            if mentions_knob && line.contains("1e-7") && !line.contains("DEFAULT_COST_OF_GROWTH") {
                offenders.push(format!("{}:{}: {}", file.display(), index + 1, line.trim()));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "cost-of-growth defaults must reference DEFAULT_COST_OF_GROWTH, not the literal 1e-7:\n{}",
        offenders.join("\n")
    );
}

/// Every `.rs` file beneath `root`.
fn rust_sources(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("source directory must be readable") {
            let path = entry.expect("directory entry must be readable").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    assert!(!files.is_empty(), "no Rust sources found under {root:?}");
    files
}
