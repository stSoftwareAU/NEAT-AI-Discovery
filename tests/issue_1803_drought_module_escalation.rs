//! Issue #1803 — the drought-time revert to Normal mode must not drop the
//! expensive discovery modules when the drought is worst.
//!
//! Background (recorded on the issue): `max_conservative_epochs` was introduced
//! by Issue #1132 purely as a cooldown on the **risk bias** — "exit conservative
//! mode as soon as one successful discovery occurs, or after a max cooldown of
//! `CONSERVATIVE_MODE_MAX_EPOCHS`". It was protecting against paying the
//! Conservative penalties (high-risk modules down-weighted, coordinated gain
//! floor tightened 10×) forever when they were demonstrably not helping. Issue
//! #1547 later reused `discovery_mode == Conservative` as the *module tiering*
//! escalation signal, which silently coupled module **breadth** to that bias
//! cooldown — so crossing the cooldown also tiered out the seven expensive
//! modules on creatures over 1000 hidden neurons.
//!
//! The chosen fix is option 1 from the issue: keep the escalated module set
//! while the low-success-rate condition holds, and revert only the risk-biasing
//! part of Conservative mode once the streak exceeds the cooldown.
//!
//! Option 2 (gate the revert on `candidate_starvation::classify`) was rejected:
//! the classification is derived from a pass's `RejectionBreakdown`, which only
//! exists *after* dispatch, whereas the tiering decision must be made *before*
//! the module set is built.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DEFAULT_LOW_SUCCESS_RATE_THRESHOLD, DiscoveryMode,
    DiscoveryOutcomeLog, decide_mode, decide_mode_with_escalation,
};
use neat_ai_discovery::analysis::module_tiering::{
    DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD, EXPENSIVE_MODULES, log_tiering_decision,
    should_skip_module,
};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// A creature well above the tiering threshold (the production profile).
const LARGE_CREATURE_HIDDEN: usize = 1662;

fn failures(n: usize) -> Vec<bool> {
    vec![false; n]
}

/// A deep drought: the streak is longer than the Conservative cooldown and the
/// rolling success rate has collapsed to zero.
fn deep_drought_log() -> DiscoveryOutcomeLog {
    DiscoveryOutcomeLog::from_outcomes(failures(DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS as usize + 5))
}

fn decide(log: &DiscoveryOutcomeLog) -> neat_ai_discovery::analysis::discovery_mode::ModeDecision {
    decide_mode_with_escalation(
        log,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
    )
}

// =============================================================================
// AC 1 — pin the existing risk-bias revert
// =============================================================================

/// `decide_mode` still reverts to Normal on a long streak despite a collapsed
/// success rate. This pins the pre-existing behaviour so any future change to
/// the *risk bias* cooldown shows up in the diff.
#[test]
fn ac1_long_streak_still_reverts_risk_bias_to_normal() {
    let log = deep_drought_log();
    assert_eq!(log.rolling_success_rate(), 0.0);
    assert!(log.consecutive_trailing_failures() > DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);

    assert_eq!(
        decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
        ),
        DiscoveryMode::Normal,
        "the Issue #1132 risk-bias cooldown is retained"
    );
    assert_eq!(
        decide(&log).mode,
        DiscoveryMode::Normal,
        "the escalation-aware decision must agree with decide_mode on the mode"
    );
}

// =============================================================================
// AC 2 — the expensive modules survive a deep drought on a large creature
// =============================================================================

/// The fix: on a >1000-hidden-neuron creature in a deep drought, every
/// expensive-tier module is retained even though the mode has reverted to
/// Normal.
#[test]
fn ac2_deep_drought_retains_expensive_modules_on_large_creature() {
    let decision = decide(&deep_drought_log());
    assert_eq!(decision.mode, DiscoveryMode::Normal);
    assert!(
        decision.module_escalation_active,
        "module escalation must survive the risk-bias revert"
    );
    assert!(decision.is_extended_drought());

    for name in EXPENSIVE_MODULES {
        assert!(
            !should_skip_module(
                name,
                LARGE_CREATURE_HIDDEN,
                DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
                decision.module_escalation_active,
            ),
            "{name} must be retained during a deep drought"
        );
    }
}

/// The bug this issue reports, expressed as a contrast: deriving the escalation
/// flag from the mode alone (the pre-fix `discovery_mode == Conservative`) drops
/// every expensive module on the same deep-drought log.
#[test]
fn ac2_deriving_escalation_from_mode_alone_would_drop_them() {
    let decision = decide(&deep_drought_log());
    let pre_fix_escalation = decision.mode == DiscoveryMode::Conservative;
    assert!(!pre_fix_escalation);

    for name in EXPENSIVE_MODULES {
        assert!(
            should_skip_module(
                name,
                LARGE_CREATURE_HIDDEN,
                DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
                pre_fix_escalation,
            ),
            "pre-fix behaviour dropped {name} — this is the regression being fixed"
        );
    }
}

/// Inside the cooldown the module set was already escalated; that must not
/// change.
#[test]
fn ac2_conservative_regime_still_escalates() {
    let log = DiscoveryOutcomeLog::from_outcomes(failures(10));
    let decision = decide(&log);
    assert_eq!(decision.mode, DiscoveryMode::Conservative);
    assert!(decision.module_escalation_active);
    assert!(!decision.is_extended_drought());
}

/// A healthy creature must still be tiered — the fix must not escalate every
/// large creature permanently.
#[test]
fn ac2_healthy_creature_is_still_tiered() {
    let mut outcomes = vec![true; 8];
    outcomes.extend(failures(2));
    let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
    let decision = decide(&log);

    assert_eq!(decision.mode, DiscoveryMode::Normal);
    assert!(
        !decision.module_escalation_active,
        "a 0.8 rolling success rate is not a drought"
    );
    for name in EXPENSIVE_MODULES {
        assert!(should_skip_module(
            name,
            LARGE_CREATURE_HIDDEN,
            DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
            decision.module_escalation_active,
        ));
    }
}

/// An empty log knows nothing, so it must not escalate.
#[test]
fn ac2_empty_log_does_not_escalate() {
    let decision = decide(&DiscoveryOutcomeLog::default());
    assert_eq!(decision.mode, DiscoveryMode::Normal);
    assert!(!decision.module_escalation_active);
    assert!(!decision.is_extended_drought());
}

/// A single success in an otherwise-dead window keeps the streak at zero but
/// leaves the rolling rate collapsed — the module set stays escalated because
/// the drought condition is the rolling rate, not the streak.
#[test]
fn ac2_streak_reset_by_one_success_keeps_escalation_while_rate_is_collapsed() {
    let mut outcomes = failures(9);
    outcomes.push(true);
    let log = DiscoveryOutcomeLog::from_outcomes(outcomes);
    let decision = decide(&log);

    assert_eq!(decision.trailing_failure_streak, 0);
    assert!((decision.rolling_success_rate - 0.1).abs() < 1e-6);
    assert_eq!(decision.mode, DiscoveryMode::Conservative);
    assert!(decision.module_escalation_active);
}

/// The escalation-aware decision must never disagree with `decide_mode` about
/// the mode, across the whole streak range around the cooldown boundary.
#[test]
fn ac2_mode_agrees_with_decide_mode_across_streak_range() {
    for streak in 0..=(DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS as usize + 3) {
        let log = DiscoveryOutcomeLog::from_outcomes(failures(streak));
        let expected = decide_mode(
            &log,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
        );
        let decision = decide(&log);
        assert_eq!(decision.mode, expected, "streak {streak}");
        assert_eq!(
            decision.trailing_failure_streak,
            u32::try_from(streak).expect("small streak")
        );
    }
}

// =============================================================================
// AC 3 — the transition is observable
// =============================================================================

/// One captured tracing event.
#[derive(Debug, Clone)]
struct CapturedEvent {
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

struct CaptureLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture buffer poisoned")
            .push(CapturedEvent {
                message: visitor.message,
                fields: visitor.fields,
            });
    }
}

fn capture_events<F: FnOnce()>(body: F) -> Vec<CapturedEvent> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Registry::default().with(CaptureLayer {
        events: Arc::clone(&events),
    });
    tracing::subscriber::with_default(subscriber, body);
    events.lock().expect("capture buffer poisoned").clone()
}

/// The deep-drought escalation must log the streak length, the rolling success
/// rate and the resulting module count.
#[test]
fn ac3_extended_drought_escalation_logs_streak_rate_and_module_count() {
    let decision = decide(&deep_drought_log());
    let events = capture_events(|| {
        log_tiering_decision(
            &decision,
            LARGE_CREATURE_HIDDEN,
            DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
            48,
            &[],
        );
    });

    let event = events
        .iter()
        .find(|e| e.fields.contains_key("module_count"))
        .expect("escalation must log a module_count line");

    assert_eq!(
        event
            .fields
            .get("trailing_failure_streak")
            .map(String::as_str),
        Some("25")
    );
    assert_eq!(
        event.fields.get("rolling_success_rate").map(String::as_str),
        Some("0.0")
    );
    assert_eq!(
        event.fields.get("module_count").map(String::as_str),
        Some("48")
    );
    assert_eq!(
        event.fields.get("discovery_mode").map(String::as_str),
        Some("normal"),
        "the risk bias has reverted while the module set stays escalated"
    );
    assert_eq!(
        event.fields.get("extended_drought").map(String::as_str),
        Some("true")
    );
    assert!(
        event.message.contains("module set re-enabled"),
        "message must name the re-enable: {}",
        event.message
    );
}

/// The tiered-out (non-drought) path must carry the same three fields so an
/// operator can see why the set narrowed.
#[test]
fn ac3_tiered_out_path_logs_streak_rate_and_module_count() {
    let mut outcomes = vec![true; 8];
    outcomes.extend(failures(2));
    let decision = decide(&DiscoveryOutcomeLog::from_outcomes(outcomes));
    let skipped: Vec<String> = EXPENSIVE_MODULES.iter().map(|s| (*s).to_string()).collect();

    let events = capture_events(|| {
        log_tiering_decision(
            &decision,
            LARGE_CREATURE_HIDDEN,
            DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
            41,
            &skipped,
        );
    });

    let event = events
        .iter()
        .find(|e| e.fields.contains_key("module_count"))
        .expect("tiering must log a module_count line");

    assert_eq!(
        event
            .fields
            .get("trailing_failure_streak")
            .map(String::as_str),
        Some("2")
    );
    let logged_rate: f32 = event
        .fields
        .get("rolling_success_rate")
        .expect("rate field")
        .parse()
        .expect("rate parses as f32");
    assert!(
        (logged_rate - 0.8).abs() < 1e-6,
        "logged rate {logged_rate}"
    );
    assert_eq!(
        event.fields.get("module_count").map(String::as_str),
        Some("41")
    );
    assert_eq!(
        event.fields.get("skipped_count").map(String::as_str),
        Some("7")
    );
}

/// A small creature is not a tiering decision at all — nothing is logged, so the
/// drought lines stay meaningful.
#[test]
fn ac3_small_creature_logs_nothing() {
    let decision = decide(&deep_drought_log());
    let events = capture_events(|| {
        log_tiering_decision(
            &decision,
            500,
            DEFAULT_MODULE_TIERING_HIDDEN_THRESHOLD,
            48,
            &[],
        );
    });
    assert!(
        events
            .iter()
            .all(|e| !e.fields.contains_key("module_count")),
        "below the threshold tiering is a no-op and must not log a decision"
    );
}
