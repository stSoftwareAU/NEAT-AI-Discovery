//! Issue #2468 (CWE-778): a config reader that silently disables a guard on `0`
//! or on an unparsable value must log one `WARN` at the read site, naming the
//! variable, the raw value and the guard. Unset and valid values stay silent.
//!
//! The opt-out warning is deduplicated per `(env, raw value)` per process, so
//! every `(env, value)` pair that is expected to warn is used by exactly one
//! case in this binary. Reusing a pair would make the second expectation pass
//! or fail for the wrong reason.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use neat_ai_discovery::analysis::constants::{
    min_bypass_weight_for_collapse, min_expected_creature_score_gain, remove_low_impact_noise_floor,
};
use neat_ai_discovery::analysis::gpu::heartbeat::GPU_STALL_WINDOW_ENV;
use neat_ai_discovery::config::{
    DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, analysis_reserve_ms,
    constant_source_effect_threshold, constant_source_threshold_with_dynamic,
    cpu_pre_reject_enabled, drought_alarm_epochs, drought_reset_after_epochs,
    effective_focus_ranking_budget_ms, gpu_stall_window, hidden_squash_prune_enabled,
    insufficient_recording_fraction, max_activation_configs_per_target, max_sources_per_target,
    min_available_memory_gb, strict_candidate_reconciliation, watchdog_stall_timeout,
};
use serial_test::serial;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

const FOCUS_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS";
const ANALYSIS_RESERVE_ENV: &str = "NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS";
const INSUFFICIENT_ENV: &str = "NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION";
const MIN_AVAILABLE_ENV: &str = "NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB";
const DROUGHT_RESET_ENV: &str = "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS";
const DROUGHT_ALARM_ENV: &str = "NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS";
const CPU_PRE_REJECT_ENV: &str = "NEAT_AI_DISCOVERY_CPU_PRE_REJECT";
const HIDDEN_SQUASH_ENV: &str = "NEAT_AI_DISCOVERY_HIDDEN_SQUASH_PRUNE";
const WATCHDOG_ENV: &str = "NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS";
const MAX_SOURCES_ENV: &str = "NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET";
const MAX_ACTIVATION_ENV: &str = "NEAT_AI_DISCOVERY_MAX_ACTIVATION_CONFIGS_PER_TARGET";
const CONSTANT_SOURCE_ENV: &str = "NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD";
const STRICT_RECONCILIATION_ENV: &str = "NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION";
const MIN_EXPECTED_GAIN_ENV: &str = "NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN";
const MIN_BYPASS_ENV: &str = "NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE";
const REMOVE_LOW_IMPACT_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
const REMOVE_LOW_IMPACT_UNITS_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS";

/// Restores the previous value of an environment variable on drop.
struct EnvGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Every test using these process-wide environment variables is serialised.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }

    fn unset(name: &'static str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Every test using these process-wide environment variables is serialised.
        unsafe { std::env::remove_var(name) };
        Self { name, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            // SAFETY: Every test using these process-wide environment variables is serialised.
            Some(value) => unsafe { std::env::set_var(self.name, value) },
            // SAFETY: Every test using these process-wide environment variables is serialised.
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

#[derive(Debug, Clone)]
struct CapturedEvent {
    level: String,
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
                level: event.metadata().level().to_string(),
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

fn warnings(events: &[CapturedEvent]) -> Vec<&CapturedEvent> {
    events.iter().filter(|e| e.level == "WARN").collect()
}

/// Whether an opt-out disables its guard, or an unparsable value leaves it
/// off or uncapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Disabled,
    Invalid,
}

/// One opt-out: the variable and value that must warn exactly once, the guard
/// text the message must name, and the reader with its expected return value.
struct Case {
    site: &'static str,
    env: &'static str,
    value: &'static str,
    kind: Kind,
    guard: &'static str,
    read: fn() -> String,
    expected: String,
}

fn opt_out_cases() -> Vec<Case> {
    vec![
        Case {
            site: "focus-ranking budget",
            env: FOCUS_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the focus-ranking time budget",
            read: || format!("{:?}", effective_focus_ranking_budget_ms(false, 0)),
            expected: "None".to_string(),
        },
        Case {
            site: "analysis reserve",
            env: ANALYSIS_RESERVE_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the analysis wall-clock reserve",
            read: || format!("{:?}", analysis_reserve_ms()),
            expected: "0".to_string(),
        },
        Case {
            site: "GPU stall window",
            env: GPU_STALL_WINDOW_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the GPU stall detector",
            read: || format!("{:?}", gpu_stall_window()),
            expected: format!("{:?}", Duration::ZERO),
        },
        Case {
            site: "insufficient-recording gate",
            env: INSUFFICIENT_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the insufficient-recording fail-fast gate",
            read: || format!("{:?}", insufficient_recording_fraction()),
            expected: "None".to_string(),
        },
        Case {
            site: "available-memory gate",
            env: MIN_AVAILABLE_ENV,
            value: "0.0",
            kind: Kind::Disabled,
            guard: "the available-memory gate",
            read: || format!("{:?}", min_available_memory_gb()),
            expected: format!("{:?}", 0.0_f64),
        },
        Case {
            site: "drought-reset escape hatch",
            env: DROUGHT_RESET_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the drought-reset escape hatch",
            read: || format!("{:?}", drought_reset_after_epochs()),
            expected: "None".to_string(),
        },
        Case {
            site: "creature drought alarm",
            env: DROUGHT_ALARM_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the creature drought alarm",
            read: || format!("{:?}", drought_alarm_epochs()),
            expected: "None".to_string(),
        },
        Case {
            site: "CPU pre-reject screen",
            env: CPU_PRE_REJECT_ENV,
            value: "no",
            kind: Kind::Disabled,
            guard: "the CPU pre-reject screen",
            read: || format!("{:?}", cpu_pre_reject_enabled()),
            expected: "false".to_string(),
        },
        Case {
            site: "hidden squash-set pruning",
            env: HIDDEN_SQUASH_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "hidden squash-set pruning",
            read: || format!("{:?}", hidden_squash_prune_enabled()),
            expected: "false".to_string(),
        },
        Case {
            site: "watchdog stall timeout (invalid)",
            env: WATCHDOG_ENV,
            value: "3O",
            kind: Kind::Invalid,
            guard: "the watchdog stays disabled",
            read: || format!("{:?}", watchdog_stall_timeout()),
            expected: "None".to_string(),
        },
        Case {
            site: "per-target source cap (disabled)",
            env: MAX_SOURCES_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the per-target source cap",
            read: || format!("{:?}", max_sources_per_target()),
            expected: "None".to_string(),
        },
        Case {
            site: "per-target source cap (invalid)",
            env: MAX_SOURCES_ENV,
            value: "lots",
            kind: Kind::Invalid,
            guard: "the per-target source cap is disabled",
            read: || format!("{:?}", max_sources_per_target()),
            expected: "None".to_string(),
        },
        Case {
            site: "activation-config cap (disabled)",
            env: MAX_ACTIVATION_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the activation-config cap",
            read: || format!("{:?}", max_activation_configs_per_target()),
            expected: "0".to_string(),
        },
        Case {
            site: "activation-config cap (invalid)",
            env: MAX_ACTIVATION_ENV,
            value: "x1",
            kind: Kind::Invalid,
            guard: "the activation-config cap is disabled",
            read: || format!("{:?}", max_activation_configs_per_target()),
            expected: "0".to_string(),
        },
        Case {
            site: "constant-source effect threshold (disabled)",
            env: CONSTANT_SOURCE_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "constant-source folding",
            read: || format!("{:?}", constant_source_effect_threshold()),
            expected: "None".to_string(),
        },
        Case {
            site: "constant-source effect threshold (invalid)",
            env: CONSTANT_SOURCE_ENV,
            value: "lots",
            kind: Kind::Invalid,
            guard: "constant-source folding uses its default threshold",
            read: || format!("{:?}", constant_source_effect_threshold()),
            expected: format!("{:?}", Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)),
        },
        Case {
            site: "constant-source threshold with dynamic (disabled)",
            env: CONSTANT_SOURCE_ENV,
            value: "0.0",
            kind: Kind::Disabled,
            guard: "constant-source folding",
            read: || format!("{:?}", constant_source_threshold_with_dynamic(None)),
            expected: "None".to_string(),
        },
        Case {
            site: "constant-source threshold with dynamic (invalid)",
            env: CONSTANT_SOURCE_ENV,
            value: "nan",
            kind: Kind::Invalid,
            guard: "constant-source folding uses the dynamic threshold",
            read: || format!("{:?}", constant_source_threshold_with_dynamic(None)),
            expected: format!("{:?}", Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)),
        },
        Case {
            site: "strict candidate reconciliation",
            env: STRICT_RECONCILIATION_ENV,
            value: "false",
            kind: Kind::Disabled,
            guard: "the strict candidate-reconciliation debug assertion",
            read: || format!("{:?}", strict_candidate_reconciliation()),
            expected: "false".to_string(),
        },
        Case {
            site: "minimum expected-gain screen",
            env: MIN_EXPECTED_GAIN_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the minimum expected-gain screen",
            read: || format!("{:?}", min_expected_creature_score_gain()),
            expected: format!("{:?}", 0.0_f32),
        },
        Case {
            site: "collapse bypass-weight floor",
            env: MIN_BYPASS_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the collapse bypass-weight floor",
            read: || format!("{:?}", min_bypass_weight_for_collapse()),
            expected: format!("{:?}", 0.0_f32),
        },
        Case {
            site: "remove-low-impact noise floor (absolute)",
            env: REMOVE_LOW_IMPACT_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the remove-low-impact noise floor",
            read: || format!("{:?}", remove_low_impact_noise_floor(1e-6)),
            expected: format!("{:?}", 0.0_f32),
        },
        Case {
            site: "remove-low-impact noise floor (units)",
            env: REMOVE_LOW_IMPACT_UNITS_ENV,
            value: "0",
            kind: Kind::Disabled,
            guard: "the remove-low-impact noise floor",
            read: || format!("{:?}", remove_low_impact_noise_floor(1e-6)),
            expected: format!("{:?}", 0.0_f32),
        },
    ]
}

#[test]
#[serial]
fn every_opt_out_logs_one_warn_naming_variable_value_and_guard() {
    let mut violations = Vec::new();
    for case in opt_out_cases() {
        let site = case.site;
        let mut actual = String::new();
        let events = capture_events(|| {
            let _env = EnvGuard::set(case.env, case.value);
            actual = (case.read)();
        });
        if actual != case.expected {
            violations.push(format!(
                "{site}: returned {actual}, expected {} (the return value must not change)",
                case.expected
            ));
        }
        let warns = warnings(&events);
        if warns.len() != 1 {
            violations.push(format!(
                "{site}: expected exactly one WARN for {}={:?}, got {}: {events:?}",
                case.env,
                case.value,
                warns.len()
            ));
            continue;
        }
        let warn = warns[0];
        let head = format!("{}={:?}", case.env, case.value);
        if !warn.message.contains(&head) {
            violations.push(format!(
                "{site}: WARN message does not contain {head:?}: {}",
                warn.message
            ));
        }
        if !warn.message.contains(case.guard) {
            violations.push(format!(
                "{site}: WARN message does not name the guard {:?}: {}",
                case.guard, warn.message
            ));
        }
        let kind_clause = match case.kind {
            Kind::Disabled => "disables",
            Kind::Invalid => "is not a valid value",
        };
        if !warn.message.contains(kind_clause) {
            violations.push(format!(
                "{site}: WARN message should say {kind_clause:?}: {}",
                warn.message
            ));
        }
        for (field, expected) in [
            ("env", case.env),
            ("value", case.value),
            ("guard", case.guard),
        ] {
            if warn.fields.get(field).map(String::as_str) != Some(expected) {
                violations.push(format!(
                    "{site}: WARN field {field:?} is {:?}, expected {expected:?}",
                    warn.fields.get(field)
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "opt-out warnings wrong:\n{}",
        violations.join("\n")
    );
}

#[test]
#[serial]
fn unset_variables_log_nothing() {
    let mut violations = Vec::new();
    for case in opt_out_cases() {
        let events = capture_events(|| {
            let _env = EnvGuard::unset(case.env);
            let _ = (case.read)();
        });
        if !warnings(&events).is_empty() {
            violations.push(format!(
                "{}: unset {} logged {:?}",
                case.site,
                case.env,
                warnings(&events)
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "unset variables must stay silent:\n{}",
        violations.join("\n")
    );
}

/// A valid, non-opt-out, or empty value: the reader returns `expected` and logs
/// no `WARN`.
struct Control {
    site: &'static str,
    env: &'static str,
    value: &'static str,
    read: fn() -> String,
    expected: String,
}

fn control_cases() -> Vec<Control> {
    vec![
        Control {
            site: "analysis reserve (valid)",
            env: ANALYSIS_RESERVE_ENV,
            value: "5000",
            read: || format!("{:?}", analysis_reserve_ms()),
            expected: "5000".to_string(),
        },
        Control {
            site: "CPU pre-reject (truthy)",
            env: CPU_PRE_REJECT_ENV,
            value: "1",
            read: || format!("{:?}", cpu_pre_reject_enabled()),
            expected: "true".to_string(),
        },
        Control {
            site: "hidden squash pruning (truthy)",
            env: HIDDEN_SQUASH_ENV,
            value: "1",
            read: || format!("{:?}", hidden_squash_prune_enabled()),
            expected: "true".to_string(),
        },
        Control {
            site: "watchdog stall timeout (valid)",
            env: WATCHDOG_ENV,
            value: "300",
            read: || format!("{:?}", watchdog_stall_timeout()),
            expected: format!("{:?}", Some(Duration::from_secs(300))),
        },
        Control {
            site: "watchdog stall timeout (empty)",
            env: WATCHDOG_ENV,
            value: "   ",
            read: || format!("{:?}", watchdog_stall_timeout()),
            expected: "None".to_string(),
        },
        Control {
            site: "per-target source cap (valid)",
            env: MAX_SOURCES_ENV,
            value: "8",
            read: || format!("{:?}", max_sources_per_target()),
            expected: "Some(8)".to_string(),
        },
        Control {
            site: "per-target source cap (empty)",
            env: MAX_SOURCES_ENV,
            value: "",
            read: || format!("{:?}", max_sources_per_target()),
            expected: "None".to_string(),
        },
        Control {
            site: "activation-config cap (valid)",
            env: MAX_ACTIVATION_ENV,
            value: "4",
            read: || format!("{:?}", max_activation_configs_per_target()),
            expected: "4".to_string(),
        },
        Control {
            site: "activation-config cap (empty)",
            env: MAX_ACTIVATION_ENV,
            value: "",
            read: || format!("{:?}", max_activation_configs_per_target()),
            expected: "0".to_string(),
        },
        Control {
            site: "constant-source threshold (valid)",
            env: CONSTANT_SOURCE_ENV,
            value: "0.5",
            read: || format!("{:?}", constant_source_effect_threshold()),
            expected: format!("{:?}", Some(0.5_f32)),
        },
        Control {
            site: "strict candidate reconciliation (truthy)",
            env: STRICT_RECONCILIATION_ENV,
            value: "1",
            read: || format!("{:?}", strict_candidate_reconciliation()),
            expected: "true".to_string(),
        },
        Control {
            site: "insufficient-recording fraction (valid)",
            env: INSUFFICIENT_ENV,
            value: "0.5",
            read: || format!("{:?}", insufficient_recording_fraction()),
            expected: format!("{:?}", Some(0.5_f64)),
        },
        Control {
            site: "drought-reset (valid)",
            env: DROUGHT_RESET_ENV,
            value: "7",
            read: || format!("{:?}", drought_reset_after_epochs()),
            expected: "Some(7)".to_string(),
        },
        Control {
            site: "drought alarm (valid)",
            env: DROUGHT_ALARM_ENV,
            value: "12",
            read: || format!("{:?}", drought_alarm_epochs()),
            expected: "Some(12)".to_string(),
        },
        Control {
            site: "GPU stall window (valid)",
            env: GPU_STALL_WINDOW_ENV,
            value: "30",
            read: || format!("{:?}", gpu_stall_window()),
            expected: format!("{:?}", Duration::from_secs(30)),
        },
        Control {
            site: "minimum expected-gain screen (valid)",
            env: MIN_EXPECTED_GAIN_ENV,
            value: "0.001",
            read: || format!("{:?}", min_expected_creature_score_gain()),
            expected: format!("{:?}", 0.001_f32),
        },
    ]
}

#[test]
#[serial]
fn valid_and_empty_values_log_no_warn_and_keep_their_value() {
    let mut violations = Vec::new();
    for control in control_cases() {
        let site = control.site;
        let mut actual = String::new();
        let events = capture_events(|| {
            let _env = EnvGuard::set(control.env, control.value);
            actual = (control.read)();
        });
        if actual != control.expected {
            violations.push(format!(
                "{site}: returned {actual}, expected {}",
                control.expected
            ));
        }
        if !warnings(&events).is_empty() {
            violations.push(format!(
                "{site}: {}={:?} must not warn, logged {:?}",
                control.env,
                control.value,
                warnings(&events)
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "controls wrong:\n{}",
        violations.join("\n")
    );
}

#[test]
#[serial]
fn repeated_opt_out_warns_once_and_a_new_value_warns_again() {
    // `NEAT_AI_DISCOVERY_CPU_PRE_REJECT` is used with `no` by the table above
    // and with `0`/`false` here, so the pairs do not collide with it.
    let events = capture_events(|| {
        for value in ["0", "0", "0", "false"] {
            let _env = EnvGuard::set(CPU_PRE_REJECT_ENV, value);
            assert!(!cpu_pre_reject_enabled(), "{value} must disable the screen");
        }
    });
    let warns = warnings(&events);
    let values: Vec<&str> = warns
        .iter()
        .map(|w| w.fields.get("value").map_or("", String::as_str))
        .collect();
    assert_eq!(
        values,
        vec!["0", "false"],
        "repeated opt-outs must warn once per distinct value: {events:?}"
    );
}
