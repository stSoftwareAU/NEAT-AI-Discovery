//! Structured `SampleCreatureDisconnect` event (Issue #1195).
//!
//! Emitted whenever the
//! [`crate::analysis::scoring::sample_creature_disconnect`] detector fires
//! for a failure-cache entry — that is, when a candidate's
//! `improved_count / total_count` ratio is at or above
//! [`crate::analysis::scoring::sample_creature_disconnect::SAMPLE_DISCONNECT_RATIO_THRESHOLD`]
//! while its aggregated `actual_error_reduction` is non-positive.
//!
//! The event is logged via `tracing::warn!` with
//! `target = "neat_ai_discovery::observability"` and an
//! `event = "sample_creature_disconnect"` tag so downstream tooling can
//! filter on it without scanning by message text.

#![allow(clippy::cast_precision_loss)] // ratio computed from u32 sample counts.

use serde::Serialize;

use crate::analysis::scoring::calibration_correction::FailureCacheEntry;
use crate::analysis::scoring::sample_creature_disconnect::detect_disconnect_entry;

/// One emitted [`SampleCreatureDisconnect`] event.
///
/// The struct is `Serialize` so callers (and downstream tests) can capture
/// the emission for assertion. All numeric fields are taken verbatim from
/// the failure-cache entry; the `improved_ratio` is computed eagerly so
/// downstream tooling does not have to redo the division.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SampleCreatureDisconnect {
    /// Opaque identifier for the parent batch — matches the `batch_id`
    /// field on [`super::ZeroSuccessBatchSummary`] when both events fire
    /// for the same batch.
    pub batch_id: u64,
    /// Candidate change-type (e.g. `add-neurons`).
    pub change_type: String,
    /// Target neuron's activation function, if known.
    pub target_squash: Option<String>,
    /// Variant key produced by `variant_generation`, if known.
    pub variant_key: Option<String>,
    /// Stable UUID of the target neuron, if known.
    pub target_uuid: Option<String>,
    /// Per-sample improvement counter from the failure cache.
    pub improved_count: u32,
    /// Per-sample total counter from the failure cache.
    pub total_count: u32,
    /// `improved_count / total_count`, computed eagerly.
    pub improved_ratio: f32,
    /// What the prediction model promised (`expectedErrorReduction`).
    pub expected_error_reduction: f32,
    /// What the post-apply evaluation actually delivered
    /// (`actualErrorReduction`).
    pub actual_error_reduction: f32,
}

impl SampleCreatureDisconnect {
    /// Build an event from a failure-cache entry. Returns `None` when the
    /// detector does not fire for the supplied entry — the caller should
    /// only ever observe `Some(_)` for genuine disconnects.
    #[must_use]
    pub fn from_entry(batch_id: u64, entry: &FailureCacheEntry) -> Option<Self> {
        if !detect_disconnect_entry(entry) {
            return None;
        }
        // detect_disconnect_entry guaranteed both counts are populated.
        let improved = entry.improved_count?;
        let total = entry.total_count?;
        // total > 0 guaranteed by detect_disconnect.
        let ratio = improved as f32 / total as f32;
        Some(Self {
            batch_id,
            change_type: entry.change_type.clone(),
            target_squash: entry.target_squash.clone(),
            variant_key: entry.variant_key.clone(),
            target_uuid: entry.target_uuid.clone(),
            improved_count: improved,
            total_count: total,
            improved_ratio: ratio,
            expected_error_reduction: entry.expected_error_reduction,
            actual_error_reduction: entry.actual_error_reduction,
        })
    }

    /// Emit the event via `tracing::warn!`.
    pub fn emit(&self) {
        tracing::warn!(
            target: "neat_ai_discovery::observability",
            event = "sample_creature_disconnect",
            batch_id = self.batch_id,
            change_type = %self.change_type,
            target_squash = ?self.target_squash,
            variant_key = ?self.variant_key,
            target_uuid = ?self.target_uuid,
            improved_count = self.improved_count,
            total_count = self.total_count,
            improved_ratio = self.improved_ratio,
            expected_error_reduction = self.expected_error_reduction,
            actual_error_reduction = self.actual_error_reduction,
            "candidate showed sample-vs-creature disconnect — calibration penalty applied (Issue #1195)"
        );
    }
}

/// Build and emit a [`SampleCreatureDisconnect`] event for the supplied
/// failure-cache entry, returning `Some(event)` when the detector fired
/// and `None` otherwise.
pub fn maybe_emit_sample_creature_disconnect(
    batch_id: u64,
    entry: &FailureCacheEntry,
) -> Option<SampleCreatureDisconnect> {
    let event = SampleCreatureDisconnect::from_entry(batch_id, entry)?;
    event.emit();
    Some(event)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        change_type: &str,
        squash: Option<&str>,
        variant: Option<&str>,
        improved: Option<u32>,
        total: Option<u32>,
        actual: f32,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: 0.001,
            actual_error_reduction: actual,
            target_squash: squash.map(str::to_string),
            variant_key: variant.map(str::to_string),
            target_uuid: None,
            improved_count: improved,
            total_count: total,
            age_epochs: None,
        }
    }

    #[test]
    fn from_entry_returns_none_when_detector_silent() {
        // Per-sample ratio below threshold — no event.
        let e = entry(
            "add-neurons",
            Some("SINE"),
            Some("v2"),
            Some(800),
            Some(1036),
            -0.5,
        );
        assert!(SampleCreatureDisconnect::from_entry(1, &e).is_none());
    }

    #[test]
    fn from_entry_returns_event_on_pattern() {
        let e = entry(
            "add-neurons",
            Some("SINE"),
            Some("v2_add-neurons_demo"),
            Some(1032),
            Some(1036),
            -0.01,
        );
        let event = SampleCreatureDisconnect::from_entry(7, &e).expect("should fire");
        assert_eq!(event.batch_id, 7);
        assert_eq!(event.change_type, "add-neurons");
        assert_eq!(event.target_squash.as_deref(), Some("SINE"));
        assert_eq!(event.variant_key.as_deref(), Some("v2_add-neurons_demo"));
        assert_eq!(event.improved_count, 1032);
        assert_eq!(event.total_count, 1036);
        assert!((event.improved_ratio - (1032.0_f32 / 1036.0_f32)).abs() < 1e-6);
        assert!((event.actual_error_reduction - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn maybe_emit_returns_event_when_detector_fires() {
        let e = entry(
            "add-neurons",
            Some("SINE"),
            Some("v2"),
            Some(1032),
            Some(1036),
            -0.01,
        );
        assert!(maybe_emit_sample_creature_disconnect(0, &e).is_some());
    }

    #[test]
    fn maybe_emit_returns_none_when_detector_silent() {
        let e = entry(
            "add-neurons",
            Some("SINE"),
            Some("v2"),
            Some(800),
            Some(1036),
            -0.01,
        );
        assert!(maybe_emit_sample_creature_disconnect(0, &e).is_none());
    }

    #[test]
    fn event_serialises_with_camel_case_fields() {
        let e = entry(
            "add-neurons",
            Some("SINE"),
            Some("v2"),
            Some(1032),
            Some(1036),
            -0.01,
        );
        let event = SampleCreatureDisconnect::from_entry(99, &e).expect("fires");
        let json = serde_json::to_value(&event).expect("serialise");
        assert_eq!(json["batchId"], 99);
        assert_eq!(json["changeType"], "add-neurons");
        assert_eq!(json["targetSquash"], "SINE");
        assert!(json["improvedRatio"].is_number());
    }
}
