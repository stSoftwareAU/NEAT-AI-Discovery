//! Issue #1424 — creature-level drought alarm when weeks pass with no accepted
//! candidate.
//!
//! Verifies the public contract of
//! [`neat_ai_discovery::analysis::creature_drought_alarm`]:
//! - The alarm fires exactly once, on the pass where epochs-since-acceptance
//!   crosses the configured threshold.
//! - The payload carries the creature uuid, the epochs since the last
//!   acceptance, and the environmental-vs-search-exhaustion classification.
//! - The threshold is configurable and disablable via env var.

use neat_ai_discovery::analysis::creature_drought_alarm::{
    CreatureDroughtAlarmInputs, DEFAULT_DROUGHT_ALARM_EPOCHS, DroughtClassification,
    classify_drought, emit_creature_drought_alarm,
};
use neat_ai_discovery::config::{drought_alarm_epochs, resolve_drought_alarm_epochs};

fn inputs<'a>(
    uuid: &'a str,
    epochs: u32,
    genuine: u32,
    disabled: u32,
) -> CreatureDroughtAlarmInputs<'a> {
    CreatureDroughtAlarmInputs {
        creature_uuid: uuid,
        epochs_since_last_accepted: epochs,
        genuinely_empty_passes: genuine,
        environmentally_disabled_passes: disabled,
    }
}

#[test]
fn alarm_fires_exactly_once_at_the_crossing_pass() {
    let threshold = DEFAULT_DROUGHT_ALARM_EPOCHS;

    // Walk the counter up to and beyond the threshold; only the crossing pass
    // must produce an alarm.
    let mut fired = 0;
    for epochs in (threshold - 2)..=(threshold + 3) {
        if emit_creature_drought_alarm(&inputs("creature-a", epochs, epochs, 0), threshold)
            .is_some()
        {
            fired += 1;
        }
    }
    assert_eq!(fired, 1, "exactly one alarm across the drought");
}

#[test]
fn alarm_payload_reports_search_exhaustion() {
    // Every drought pass evaluated the creature and found nothing.
    let alarm = emit_creature_drought_alarm(&inputs("creature-exhausted", 30, 30, 0), 30)
        .expect("crossing pass fires");

    assert_eq!(alarm.creature_uuid, "creature-exhausted");
    assert_eq!(alarm.epochs_since_last_accepted, 30);
    assert_eq!(alarm.genuinely_empty_passes, 30);
    assert_eq!(alarm.environmentally_disabled_passes, 0);
    assert_eq!(
        alarm.classification,
        DroughtClassification::SearchExhaustion
    );
}

#[test]
fn alarm_payload_reports_environmental() {
    // Most drought passes were host-gated (memory / GPU), so the creature was
    // never actually evaluated — environmental, not exhausted.
    let alarm = emit_creature_drought_alarm(&inputs("creature-gated", 30, 4, 26), 30)
        .expect("crossing pass fires");

    assert_eq!(alarm.genuinely_empty_passes, 4);
    assert_eq!(alarm.environmentally_disabled_passes, 26);
    assert_eq!(alarm.classification, DroughtClassification::Environmental);
}

#[test]
fn classification_helper_matches_both_cases() {
    assert_eq!(
        classify_drought(20, 0),
        DroughtClassification::SearchExhaustion
    );
    assert_eq!(
        classify_drought(0, 20),
        DroughtClassification::Environmental
    );
    // Ties favour search exhaustion — an evaluated pass is firmer evidence.
    assert_eq!(
        classify_drought(10, 10),
        DroughtClassification::SearchExhaustion
    );
}

#[test]
fn alarm_payload_serialises_camel_case_for_automation() {
    let alarm = emit_creature_drought_alarm(&inputs("c-1", 30, 10, 20), 30).expect("fires");
    let json = serde_json::to_string(&alarm).expect("serialise");
    assert!(json.contains("\"creatureUuid\":\"c-1\""));
    assert!(json.contains("\"epochsSinceLastAccepted\":30"));
    assert!(json.contains("\"classification\":\"environmental\""));
}

#[test]
fn threshold_is_configurable_and_disablable() {
    // Default when unset/invalid; a positive override is honoured; 0 disables.
    assert_eq!(
        resolve_drought_alarm_epochs(None, DEFAULT_DROUGHT_ALARM_EPOCHS),
        Some(DEFAULT_DROUGHT_ALARM_EPOCHS)
    );
    assert_eq!(
        resolve_drought_alarm_epochs(Some("42"), DEFAULT_DROUGHT_ALARM_EPOCHS),
        Some(42)
    );
    assert_eq!(
        resolve_drought_alarm_epochs(Some("0"), DEFAULT_DROUGHT_ALARM_EPOCHS),
        None
    );

    // The accessor always yields a sane value regardless of the ambient env
    // (Some(n>=1) when armed, None when disabled via env — both valid).
    if let Some(n) = drought_alarm_epochs() {
        assert!(n >= 1);
    }
}
