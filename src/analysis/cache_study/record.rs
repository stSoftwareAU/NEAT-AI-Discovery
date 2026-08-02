//! The per-candidate JSON record as persisted in the production discovery candidates cache.
//!
//! One file per evaluated candidate, laid out as
//! `success|failures/<model-hash>/<strategy>/<key>.json` (Issue #1920).

use serde::Deserialize;
use std::collections::BTreeMap;

/// A single evaluated candidate, as written by the NEAT-AI controller.
///
/// Only the fields common to every wire-schema-v2 record are mandatory; the
/// per-strategy extras (`expectedErrorReduction`, `sampleSize`, …) are optional
/// because `remove-low-impact` records omit them.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateRecord {
    /// Cache key, unique per candidate within a model hash.
    pub key: String,
    /// Strategy that proposed the candidate, e.g. `remove-low-impact`.
    pub change_type: String,
    /// Human-readable summary written by the controller.
    #[serde(default)]
    pub description: String,
    /// Creature score before the change was applied.
    pub original_score: f64,
    /// Creature score after the change was applied.
    pub candidate_score: f64,
    /// `candidate_score - original_score`; positive means the change helped.
    pub score_delta: f64,
    /// Creature error before the change was applied.
    pub original_error: f64,
    /// Creature error after the change was applied.
    pub error: f64,
    /// RFC-3339 evaluation timestamp.
    pub timestamp: String,
    /// `neat_ai_discovery` version that proposed the candidate.
    #[serde(default)]
    pub discovery_version: String,
    /// The proposal this library sent to the controller, keyed by candidate kind.
    #[serde(default)]
    pub rust_request: BTreeMap<String, serde_json::Value>,
    /// Error reduction this library predicted, when the strategy predicts one.
    pub expected_error_reduction: Option<f64>,
    /// Error reduction actually observed by the controller.
    pub actual_error_reduction: Option<f64>,
    /// Training samples behind the prediction, when recorded.
    pub sample_size: Option<f64>,
}

impl CandidateRecord {
    /// The `rustRequest` payload kinds, joined by `+` (e.g. `neuronCandidate+neuronDetails`).
    ///
    /// Returns `"none"` for a record with an empty request, which is itself a
    /// finding worth surfacing rather than hiding.
    #[must_use]
    pub fn request_kind(&self) -> String {
        if self.rust_request.is_empty() {
            return "none".to_string();
        }
        self.rust_request
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("+")
    }

    /// The `YYYY-MM-DD` day of the evaluation, or `"unknown"` for a short timestamp.
    #[must_use]
    pub fn day(&self) -> &str {
        self.timestamp.get(..10).unwrap_or("unknown")
    }

    /// Reads a nested numeric field out of `rustRequest`, e.g.
    /// `["removalCandidate", "impact"]`.
    ///
    /// Returns `None` when any path element is missing or the leaf is not a
    /// number — a missing predictor must not be silently coerced to zero.
    #[must_use]
    pub fn request_number(&self, path: &[&str]) -> Option<f64> {
        let (head, rest) = path.split_first()?;
        let mut node = self.rust_request.get(*head)?;
        for step in rest {
            node = node.get(step)?;
        }
        node.as_f64()
    }
}
