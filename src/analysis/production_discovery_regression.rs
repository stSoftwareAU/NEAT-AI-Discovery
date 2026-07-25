//! Production-network discovery regression harness (Issue #1741).
//!
//! Milestone #1736 asked whether discovery on the large converged production
//! cluster network can find *any* accepted improvement in *most* runs. The
//! sibling work settled the diagnosis:
//!
//! - #1737 — the network is **proposal-rich but over-rejected**: generators emit
//!   candidates in quantity; nearly all are discarded at the expected-gain gate.
//! - #1738 — the focus/impact contribution math through MAX/MIN/IF aggregation
//!   squashes is **sound** (no fault found).
//! - #1740 — the acceptance gain floors are **correctly scaled** and must not be
//!   lowered.
//!
//! Those audits concluded the low accepted rate is a **well-evidenced plateau**:
//! the creature is genuinely saturated, not mis-calculated. Over the 40-run
//! production discovery window (2026-06-16 → 2026-07-23) only 2 runs synced an
//! accepted candidate — a ~5% accepted-run rate (see
//! `docs/analysis/rejection-diagnosis-1737.md`).
//!
//! This module provides the repeatable harness the milestone's close-criterion
//! needs: it runs the **real** acceptance logic
//! ([`validate_coordinated_candidate_gain`] plus the post-discount noise floor,
//! and the evaluate-before-accept rule for error-ranked removals) over a
//! committed production-representative run batch, and reports the
//! accepted-improvement rate. A future regression in discovery yield on the
//! production topology drops the computed rate below the recorded plateau
//! baseline and fails the harness test
//! (`tests/production_discovery_regression.rs`).
//!
//! The harness deliberately drives the *shipped* acceptance functions rather
//! than re-implementing the gate, so a change that weakens or breaks acceptance
//! on the production profile changes the reported rate here.
//!
//! Naming note: milestone #1736 refers to the deployed network by a private
//! deployment name. The shipped-source private-name guards (Issues #1724/#1725)
//! keep that token out of `.rs` sources and file names, so this harness uses
//! the concept-level term "production network" throughout.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::analysis::candidate_aggregation::validate_coordinated_candidate_gain;
use crate::analysis::constants::coordinated_post_discount_noise_floor;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Parent #1736 success criterion: at least one accepted improvement in *most*
/// discovery runs — i.e. a strict majority (> 50%) of runs accept a candidate.
pub const SUCCESS_ACCEPTED_RUN_RATE: f64 = 0.5;

/// Recorded plateau baseline for the production topology (Issue #1741,
/// diagnosis #1737): 2 accepted runs across the 40-run production discovery
/// window, a 0.05 accepted-run rate, recorded **after** the milestone audits
/// (#1738 focus/impact, #1740 thresholds) concluded the network is genuinely
/// saturated. The harness test guards against the rate regressing below this.
pub const PLATEAU_ACCEPTED_RUN_RATE: f64 = 0.05;

/// Absolute tolerance for accepted-run-rate comparisons.
const RATE_EPSILON: f64 = 1e-9;

/// One recorded discovery candidate from a production-representative run,
/// carrying the two numbers acceptance actually turns on: the generator's
/// estimated `expected_gain` and the realised `realised_delta` measured by
/// evaluate-before-accept (#1623).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryCandidateRecord {
    /// The discovery change type (e.g. `"change-squash"`, `"remove-neuron"`).
    /// Descriptive only — acceptance turns on the numeric fields below.
    pub change_type: String,
    /// Number of atomic operations in the candidate (1 for a single-op change).
    #[serde(default = "default_operation_count")]
    pub operation_count: usize,
    /// The generator's estimated expected creature-score gain.
    pub expected_gain: f32,
    /// The realised score delta measured by evaluate-before-accept (#1623).
    pub realised_delta: f32,
    /// Error-ranked acceptance path (harmful-neuron / remove-neuron). On the
    /// production network the expected-gain estimate for these collapses to `0`,
    /// so they are ranked by error magnitude and accepted only on a positive
    /// realised delta rather than through the expected-gain gate (#1737).
    #[serde(default)]
    pub error_ranked: bool,
}

const fn default_operation_count() -> usize {
    1
}

impl DiscoveryCandidateRecord {
    /// Decide whether this candidate would be **accepted** by the shipped
    /// discovery acceptance logic.
    ///
    /// - Error-ranked removals (#1737): accepted only when the realised delta is
    ///   a genuine improvement (`> 0`), per evaluate-before-accept (#1623).
    /// - Gain-gated coordinated-structural candidates: must clear **both** the
    ///   multi-op validation gate ([`validate_coordinated_candidate_gain`],
    ///   #732) **and** the per-op-count post-discount noise floor
    ///   ([`coordinated_post_discount_noise_floor`], #1128/#1272).
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        if self.error_ranked {
            return self.realised_delta > 0.0;
        }
        let candidate = self.as_coordinated_candidate();
        validate_coordinated_candidate_gain(&candidate)
            && self.expected_gain >= coordinated_post_discount_noise_floor(self.operation_count)
    }

    /// Build a shipped [`CoordinatedStructuralCandidateJson`] with the recorded
    /// expected gain and `operation_count` placeholder operations, so the real
    /// gate functions can be applied. Only the operation count and expected gain
    /// affect the gate, so the placeholder ops are inert.
    fn as_coordinated_candidate(&self) -> CoordinatedStructuralCandidateJson {
        let operations = (0..self.operation_count.max(1))
            .map(|i| CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: format!("regress-src-{i}"),
                to_neuron_uuid: format!("regress-dst-{i}"),
            })
            .collect();
        CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: self.expected_gain,
            comment: Some("discovery-regression-harness".to_string()),
            ..Default::default()
        }
    }
}

/// A single production-representative discovery run: the candidates it produced.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryRunOutcome {
    /// Opaque identifier for the run (e.g. a discovery commit short-sha).
    pub run_id: String,
    /// The candidates the run generated and evaluated.
    pub candidates: Vec<DiscoveryCandidateRecord>,
}

impl DiscoveryRunOutcome {
    /// Count the candidates the shipped acceptance logic would accept.
    #[must_use]
    pub fn accepted_candidate_count(&self) -> usize {
        self.candidates.iter().filter(|c| c.is_accepted()).count()
    }

    /// Whether the run produced at least one accepted improvement.
    #[must_use]
    pub fn has_accepted_improvement(&self) -> bool {
        self.candidates
            .iter()
            .any(DiscoveryCandidateRecord::is_accepted)
    }
}

/// A committed batch of production-representative discovery runs.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryRunBatch {
    /// Free-text provenance of the batch (kept in the fixture for traceability).
    #[serde(default)]
    pub description: String,
    /// The runs in the batch.
    pub runs: Vec<DiscoveryRunOutcome>,
}

impl DiscoveryRunBatch {
    /// Parse a batch from a JSON string, failing loud on malformed input or an
    /// empty run list (a silently-empty batch would report a meaningless 0/0
    /// rate — Issue #3234).
    pub fn from_json_str(json: &str) -> Result<Self> {
        let batch: Self =
            serde_json::from_str(json).context("failed to parse discovery run batch JSON")?;
        if batch.runs.is_empty() {
            anyhow::bail!("discovery run batch contains no runs — refusing to report a 0/0 rate");
        }
        Ok(batch)
    }

    /// Load and parse a batch from a JSON file on disk.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| {
            format!(
                "failed to read discovery run batch fixture: {}",
                path.display()
            )
        })?;
        Self::from_json_str(&text)
            .with_context(|| format!("invalid discovery run batch fixture: {}", path.display()))
    }
}

/// The computed accepted-improvement metrics for a batch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryBatchReport {
    /// Total number of runs in the batch.
    pub total_runs: usize,
    /// Runs that produced at least one accepted improvement.
    pub accepted_runs: usize,
    /// Runs that produced zero accepted improvements.
    pub empty_runs: usize,
    /// `accepted_runs / total_runs` — the parent's accepted-improvement rate.
    pub accepted_run_rate: f64,
    /// Total candidates evaluated across the batch.
    pub total_candidates: usize,
    /// Total candidates accepted across the batch.
    pub total_accepted_candidates: usize,
}

impl DiscoveryBatchReport {
    /// Whether the batch meets parent #1736's success criterion — an accepted
    /// improvement in *most* (a strict majority of) runs.
    #[must_use]
    pub fn meets_success_threshold(&self) -> bool {
        self.accepted_run_rate > SUCCESS_ACCEPTED_RUN_RATE
    }

    /// Whether the accepted-improvement rate has regressed **below** the given
    /// baseline (beyond floating-point tolerance).
    #[must_use]
    pub fn regressed_below_baseline(&self, baseline: f64) -> bool {
        self.accepted_run_rate < baseline - RATE_EPSILON
    }

    /// A one-line human-readable summary for test output / PR evidence.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "accepted-improvement rate: {}/{} runs = {:.4} ({} accepted / {} total candidates)",
            self.accepted_runs,
            self.total_runs,
            self.accepted_run_rate,
            self.total_accepted_candidates,
            self.total_candidates,
        )
    }
}

/// Compute the accepted-improvement metrics for a batch by running the shipped
/// acceptance logic over every candidate in every run.
#[must_use]
pub fn compute_batch_report(batch: &DiscoveryRunBatch) -> DiscoveryBatchReport {
    let total_runs = batch.runs.len();
    let mut accepted_runs = 0;
    let mut total_candidates = 0;
    let mut total_accepted_candidates = 0;

    for run in &batch.runs {
        let accepted = run.accepted_candidate_count();
        total_candidates += run.candidates.len();
        total_accepted_candidates += accepted;
        if accepted > 0 {
            accepted_runs += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)] // run counts are far below f64 precision limits
    let accepted_run_rate = if total_runs == 0 {
        0.0
    } else {
        accepted_runs as f64 / total_runs as f64
    };

    DiscoveryBatchReport {
        total_runs,
        accepted_runs,
        empty_runs: total_runs - accepted_runs,
        accepted_run_rate,
        total_candidates,
        total_accepted_candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_gated(
        change_type: &str,
        op_count: usize,
        expected_gain: f32,
    ) -> DiscoveryCandidateRecord {
        DiscoveryCandidateRecord {
            change_type: change_type.to_string(),
            operation_count: op_count,
            expected_gain,
            realised_delta: 0.0,
            error_ranked: false,
        }
    }

    fn error_ranked(realised_delta: f32) -> DiscoveryCandidateRecord {
        DiscoveryCandidateRecord {
            change_type: "remove-neuron".to_string(),
            operation_count: 1,
            expected_gain: 0.0,
            realised_delta,
            error_ranked: true,
        }
    }

    #[test]
    fn error_ranked_removal_accepted_only_on_positive_realised_delta() {
        // The accepted candidates: harmful-neuron removals with expected gain 0
        // but a positive realised delta (#1737).
        assert!(error_ranked(1.95e-7).is_accepted());
        // A harmful removal that turned out to hurt is rejected.
        assert!(!error_ranked(-0.00218).is_accepted());
        // Exactly zero is not an improvement.
        assert!(!error_ranked(0.0).is_accepted());
    }

    #[test]
    fn gain_gated_single_op_below_noise_floor_is_rejected() {
        // The production achievable band (~1e-7) sits below the 5e-7 single-op floor.
        assert!(!gain_gated("change-squash", 1, 1.0e-7).is_accepted());
        // A positive gain above the floor is accepted.
        assert!(gain_gated("change-squash", 1, 1.0e-6).is_accepted());
        // Non-positive gain fails the validation gate outright.
        assert!(!gain_gated("change-squash", 1, 0.0).is_accepted());
    }

    #[test]
    fn gain_gated_multi_op_below_floor_is_rejected() {
        // The production coordinated change-squash: ~4e-10 expected gain — far
        // below both the multi-op gate and the noise floor.
        assert!(!gain_gated("change-squash", 2, 4.17e-10).is_accepted());
    }

    #[test]
    fn accepted_candidate_count_and_flag_agree() {
        let run = DiscoveryRunOutcome {
            run_id: "r1".to_string(),
            candidates: vec![
                gain_gated("change-squash", 2, 4.17e-10),
                error_ranked(1.95e-7),
                error_ranked(-0.001),
            ],
        };
        assert_eq!(run.accepted_candidate_count(), 1);
        assert!(run.has_accepted_improvement());
    }

    fn empty_run(id: &str) -> DiscoveryRunOutcome {
        DiscoveryRunOutcome {
            run_id: id.to_string(),
            candidates: vec![
                gain_gated("change-squash", 2, 4.17e-10),
                error_ranked(-0.00218),
            ],
        }
    }

    fn accepted_run(id: &str) -> DiscoveryRunOutcome {
        DiscoveryRunOutcome {
            run_id: id.to_string(),
            candidates: vec![
                gain_gated("change-squash", 1, 1.0e-7),
                error_ranked(1.95e-7),
            ],
        }
    }

    #[test]
    fn compute_batch_report_counts_accepted_runs() {
        let batch = DiscoveryRunBatch {
            description: "unit".to_string(),
            runs: vec![empty_run("a"), empty_run("b"), accepted_run("c")],
        };
        let report = compute_batch_report(&batch);
        assert_eq!(report.total_runs, 3);
        assert_eq!(report.accepted_runs, 1);
        assert_eq!(report.empty_runs, 2);
        assert_eq!(report.total_accepted_candidates, 1);
        assert!((report.accepted_run_rate - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn plateau_batch_does_not_regress_but_misses_success() {
        // 38 empty + 2 accepted = the recorded 0.05 plateau.
        let mut runs: Vec<DiscoveryRunOutcome> =
            (0..38).map(|i| empty_run(&format!("e{i}"))).collect();
        runs.push(accepted_run("acc-0"));
        runs.push(accepted_run("acc-1"));
        let report = compute_batch_report(&DiscoveryRunBatch {
            description: "plateau".to_string(),
            runs,
        });
        assert!((report.accepted_run_rate - PLATEAU_ACCEPTED_RUN_RATE).abs() < 1e-12);
        assert!(!report.regressed_below_baseline(PLATEAU_ACCEPTED_RUN_RATE));
        assert!(!report.meets_success_threshold());
    }

    #[test]
    fn total_yield_collapse_regresses_below_baseline() {
        // A regression that stops discovery accepting anything drops below the
        // plateau baseline and is flagged.
        let runs: Vec<DiscoveryRunOutcome> = (0..40).map(|i| empty_run(&format!("e{i}"))).collect();
        let report = compute_batch_report(&DiscoveryRunBatch {
            description: "collapsed".to_string(),
            runs,
        });
        assert_eq!(report.accepted_runs, 0);
        assert!(report.regressed_below_baseline(PLATEAU_ACCEPTED_RUN_RATE));
    }

    #[test]
    fn majority_accepted_batch_meets_success_threshold() {
        // If a future sibling fix lifts yield, the same harness flips to the
        // parent's success criterion.
        let mut runs: Vec<DiscoveryRunOutcome> =
            (0..3).map(|i| accepted_run(&format!("a{i}"))).collect();
        runs.push(empty_run("e0"));
        let report = compute_batch_report(&DiscoveryRunBatch {
            description: "success".to_string(),
            runs,
        });
        assert!(report.meets_success_threshold());
        assert!(!report.regressed_below_baseline(PLATEAU_ACCEPTED_RUN_RATE));
    }

    #[test]
    fn from_json_str_rejects_empty_batch() {
        let err = DiscoveryRunBatch::from_json_str(r#"{"runs":[]}"#).unwrap_err();
        assert!(err.to_string().contains("no runs"));
    }

    #[test]
    fn from_json_str_round_trips_a_run() {
        let json = r#"{
            "description": "t",
            "runs": [
                {"runId": "r1", "candidates": [
                    {"changeType": "remove-neuron", "expectedGain": 0.0, "realisedDelta": 1.95e-7, "errorRanked": true}
                ]}
            ]
        }"#;
        let batch = DiscoveryRunBatch::from_json_str(json).expect("valid batch");
        assert_eq!(batch.runs.len(), 1);
        assert_eq!(batch.runs[0].candidates[0].operation_count, 1);
        assert!(batch.runs[0].has_accepted_improvement());
    }
}
