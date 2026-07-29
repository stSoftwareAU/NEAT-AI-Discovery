//! Creature-level discovery-drought alarm (Issue #1424).
//!
//! The per-pass drought diagnostic (Issue #1202) logs at WARN, but it emits no
//! durable, creature-level alarm when the *time since the last accepted
//! candidate* crosses a "weeks" threshold. Droughts are therefore invisible
//! until a human notices — which is exactly how Issue #1418 was found.
//!
//! This module adds that missing signal: when a creature's
//! `epochs_since_last_accepted_candidate` crosses a configurable threshold
//! (`NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS`), the orchestrator emits **exactly
//! one** structured alarm carrying the creature uuid, the epochs since the last
//! acceptance, and an environmental-vs-search-exhaustion classification.
//!
//! # Classification (Issue #1421)
//!
//! The alarm reuses the disambiguation introduced for the discovery-outcome
//! log: passes that the host could not actually evaluate (memory / GPU gated)
//! are *environmental*, while passes that evaluated the creature and found no
//! improving move are *search exhaustion*. A drought dominated by
//! environmentally-disabled passes is classified [`DroughtClassification::Environmental`];
//! otherwise it is [`DroughtClassification::SearchExhaustion`]. This tells the
//! surrounding automation whether to fix the host or to escalate the creature.
//!
//! # Exactly-once emission
//!
//! The orchestrator advances `epochs_since_last_accepted_candidate` by exactly
//! one per discovery pass and resets it to zero when a candidate is accepted.
//! [`emit_creature_drought_alarm`] fires only on the pass where the counter is
//! *equal to* the threshold (the crossing pass), so a single drought produces a
//! single alarm rather than one per pass.

use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// Derive a stable creature identifier from its output-neuron uuids
/// (Issue #1424).
///
/// The FFI `CreatureJson` carries no creature-level uuid, but a creature's
/// **output** neurons are created at genesis and persist for its entire
/// lifetime even as hidden neurons are added and removed. Their uuids are
/// therefore a stable identity that survives the structural churn of evolution
/// — and, crucially, is constant throughout a drought (when the topology is
/// not changing at all).
///
/// The id is order-independent (uuids are sorted before hashing) and
/// deterministic across runs (`DefaultHasher` uses fixed keys). Returns
/// `"creature-unknown"` when no uuids are supplied.
#[must_use]
pub fn derive_creature_id(output_neuron_uuids: &[&str]) -> String {
    if output_neuron_uuids.is_empty() {
        return "creature-unknown".to_string();
    }
    let mut sorted: Vec<&str> = output_neuron_uuids.to_vec();
    sorted.sort_unstable();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for uuid in sorted {
        uuid.hash(&mut hasher);
        // Length-delimit so ["ab","c"] and ["a","bc"] cannot collide.
        0xff_u8.hash(&mut hasher);
    }
    format!("creature-{:016x}", hasher.finish())
}

/// Default epochs-since-last-acceptance at which the creature-level drought
/// alarm fires (Issue #1424).
///
/// A discovery "epoch" is one analysis pass for the creature. `100` sits well
/// above the per-pass drought diagnostic (default 5, Issue #1202) and the
/// one-shot reset escape hatch (default 50, Issue #1205/#1422), so the alarm
/// only fires once a creature has been genuinely stuck for an extended period —
/// the "weeks with no accepted candidate" case from Issue #1418. Tune via
/// `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS`.
pub const DEFAULT_DROUGHT_ALARM_EPOCHS: u32 = 100;

/// Whether a drought is driven by the host environment or by genuine search
/// exhaustion (Issue #1424).
///
/// Serialised in camelCase so the surrounding automation can branch on the
/// cause without parsing the human-readable message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DroughtClassification {
    /// The drought is dominated by passes the host could not evaluate (memory
    /// budget, memory pressure, or a missing GPU adapter). Discovery was gated
    /// off, not exhausted — fix the host.
    Environmental,
    /// The drought is dominated by passes that evaluated the creature and found
    /// no improving move. The search itself is stuck — escalate the creature.
    SearchExhaustion,
}

impl DroughtClassification {
    /// Stable, greppable identifier for logs and metrics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Environmental => "environmental",
            Self::SearchExhaustion => "search_exhaustion",
        }
    }
}

/// Classify a drought from its trailing pass composition (Issue #1424).
///
/// Environmental when environmentally-disabled passes strictly outnumber
/// genuinely-empty (search-exhausted) passes; otherwise search exhaustion.
/// Ties favour [`DroughtClassification::SearchExhaustion`] — a pass that
/// actually evaluated the creature is firmer evidence than one that never ran.
#[must_use]
pub fn classify_drought(
    genuinely_empty_passes: u32,
    environmentally_disabled_passes: u32,
) -> DroughtClassification {
    if environmentally_disabled_passes > genuinely_empty_passes {
        DroughtClassification::Environmental
    } else {
        DroughtClassification::SearchExhaustion
    }
}

/// Structured creature-level drought alarm payload (Issue #1424).
///
/// Surfaced as a structured `tracing::warn!` event and attached to the
/// discovery summary metadata (`creatureDroughtAlarm`). All fields use
/// camelCase in JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatureDroughtAlarm {
    /// UUID of the creature that has gone without an accepted candidate.
    pub creature_uuid: String,
    /// Discovery passes since the creature last accepted a candidate. Counts
    /// both genuinely-empty and environmentally-disabled passes.
    pub epochs_since_last_accepted: u32,
    /// Of [`Self::epochs_since_last_accepted`], how many evaluated the creature
    /// and found no improving move (true search exhaustion).
    pub genuinely_empty_passes: u32,
    /// Of [`Self::epochs_since_last_accepted`], how many the host could not
    /// evaluate (memory / GPU gated).
    pub environmentally_disabled_passes: u32,
    /// Whether the drought is environmental or genuine search exhaustion.
    pub classification: DroughtClassification,
}

/// Inputs the orchestrator gathers before evaluating the creature-level alarm.
#[derive(Debug, Clone, Copy)]
pub struct CreatureDroughtAlarmInputs<'a> {
    /// UUID of the creature under analysis.
    pub creature_uuid: &'a str,
    /// Discovery passes since the creature last accepted a candidate.
    pub epochs_since_last_accepted: u32,
    /// Genuinely-empty (search-exhausted) passes within the drought.
    pub genuinely_empty_passes: u32,
    /// Environmentally-disabled passes within the drought.
    pub environmentally_disabled_passes: u32,
}

/// Emit the creature-level drought alarm when the epochs-since-acceptance count
/// crosses `alarm_threshold` (Issue #1424).
///
/// Returns `Some(payload)` and emits a single structured `tracing::warn!` event
/// **only** on the crossing pass — the pass where
/// `epochs_since_last_accepted == alarm_threshold`. Because the orchestrator
/// advances the counter by one per pass, this guarantees exactly one alarm per
/// drought. Returns `None` on every other pass (below the threshold, or already
/// alarmed).
#[must_use]
pub fn emit_creature_drought_alarm(
    inputs: &CreatureDroughtAlarmInputs<'_>,
    alarm_threshold: u32,
) -> Option<CreatureDroughtAlarm> {
    // Fire on the crossing pass only. Equality (not `>=`) is what makes the
    // alarm fire exactly once: the counter passes through the threshold value
    // on a single pass.
    if inputs.epochs_since_last_accepted != alarm_threshold {
        return None;
    }

    let classification = classify_drought(
        inputs.genuinely_empty_passes,
        inputs.environmentally_disabled_passes,
    );

    let alarm = CreatureDroughtAlarm {
        creature_uuid: inputs.creature_uuid.to_string(),
        epochs_since_last_accepted: inputs.epochs_since_last_accepted,
        genuinely_empty_passes: inputs.genuinely_empty_passes,
        environmentally_disabled_passes: inputs.environmentally_disabled_passes,
        classification,
    };

    tracing::warn!(
        creature_uuid = alarm.creature_uuid.as_str(),
        epochs_since_last_accepted = alarm.epochs_since_last_accepted,
        genuinely_empty_passes = alarm.genuinely_empty_passes,
        environmentally_disabled_passes = alarm.environmentally_disabled_passes,
        classification = classification.as_str(),
        alarm_threshold,
        "Issue #1424: creature-level discovery drought — no accepted candidate \
         for {} epochs ({})",
        alarm.epochs_since_last_accepted,
        classification.as_str()
    );

    Some(alarm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(epochs: u32, genuine: u32, disabled: u32) -> CreatureDroughtAlarmInputs<'static> {
        CreatureDroughtAlarmInputs {
            creature_uuid: "creature-1424",
            epochs_since_last_accepted: epochs,
            genuinely_empty_passes: genuine,
            environmentally_disabled_passes: disabled,
        }
    }

    #[test]
    fn derive_creature_id_is_deterministic_and_order_independent() {
        let a = derive_creature_id(&["out-1", "out-2", "out-3"]);
        let b = derive_creature_id(&["out-3", "out-1", "out-2"]);
        assert_eq!(a, b, "output order must not change the id");
        assert!(a.starts_with("creature-"));
    }

    #[test]
    fn derive_creature_id_distinguishes_different_creatures() {
        let a = derive_creature_id(&["out-1", "out-2"]);
        let b = derive_creature_id(&["out-1", "out-9"]);
        assert_ne!(a, b);
    }

    #[test]
    fn derive_creature_id_handles_empty() {
        assert_eq!(derive_creature_id(&[]), "creature-unknown");
    }

    #[test]
    fn no_alarm_below_threshold() {
        // One short of the threshold — no alarm yet.
        assert!(emit_creature_drought_alarm(&inputs(99, 99, 0), 100).is_none());
    }

    #[test]
    fn alarm_fires_exactly_once_at_crossing() {
        // Below, at, and above the threshold: only the crossing pass alarms.
        assert!(emit_creature_drought_alarm(&inputs(19, 19, 0), 20).is_none());
        assert!(emit_creature_drought_alarm(&inputs(20, 20, 0), 20).is_some());
        assert!(
            emit_creature_drought_alarm(&inputs(21, 21, 0), 20).is_none(),
            "already alarmed on the previous pass — must not re-fire"
        );
    }

    #[test]
    fn alarm_classifies_search_exhaustion() {
        // All passes evaluated the creature and found nothing -> exhaustion.
        let alarm =
            emit_creature_drought_alarm(&inputs(20, 20, 0), 20).expect("crossing pass fires");
        assert_eq!(alarm.creature_uuid, "creature-1424");
        assert_eq!(alarm.epochs_since_last_accepted, 20);
        assert_eq!(alarm.genuinely_empty_passes, 20);
        assert_eq!(alarm.environmentally_disabled_passes, 0);
        assert_eq!(
            alarm.classification,
            DroughtClassification::SearchExhaustion
        );
    }

    #[test]
    fn alarm_classifies_environmental() {
        // Host could not evaluate the creature on most passes -> environmental.
        let alarm =
            emit_creature_drought_alarm(&inputs(20, 5, 15), 20).expect("crossing pass fires");
        assert_eq!(alarm.genuinely_empty_passes, 5);
        assert_eq!(alarm.environmentally_disabled_passes, 15);
        assert_eq!(alarm.classification, DroughtClassification::Environmental);
    }

    #[test]
    fn classification_tie_favours_search_exhaustion() {
        assert_eq!(
            classify_drought(10, 10),
            DroughtClassification::SearchExhaustion
        );
        assert_eq!(
            classify_drought(0, 0),
            DroughtClassification::SearchExhaustion
        );
        assert_eq!(classify_drought(0, 1), DroughtClassification::Environmental);
        assert_eq!(
            classify_drought(1, 0),
            DroughtClassification::SearchExhaustion
        );
    }

    #[test]
    fn classification_strings_are_stable() {
        assert_eq!(
            DroughtClassification::Environmental.as_str(),
            "environmental"
        );
        assert_eq!(
            DroughtClassification::SearchExhaustion.as_str(),
            "search_exhaustion"
        );
    }

    #[test]
    fn alarm_round_trips_through_serde() {
        let alarm = emit_creature_drought_alarm(&inputs(20, 8, 12), 20).expect("fires");
        let json = serde_json::to_string(&alarm).expect("serialise");
        // camelCase keys for the surrounding automation.
        assert!(json.contains("\"creatureUuid\""));
        assert!(json.contains("\"epochsSinceLastAccepted\""));
        assert!(json.contains("\"classification\":\"environmental\""));
        let restored: CreatureDroughtAlarm = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored, alarm);
    }

    #[test]
    fn disabled_threshold_zero_never_fires_with_active_drought() {
        // A threshold of 0 is the orchestrator's "disabled" sentinel; an active
        // drought (epochs >= 1) must never match it.
        assert!(emit_creature_drought_alarm(&inputs(50, 50, 0), 0).is_none());
    }
}
