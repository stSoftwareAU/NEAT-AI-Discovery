//! Partial-dominance safety analysis for MAX / MIN / IF selection aggregates
//! (Issue #1712, gap **G2** from `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md`).
//!
//! The clean, fully-dominated fixtures characterised by #1706 are the easy
//! case: a branch that can *never* win its aggregate. This module handles the
//! "not so clean" partially-dominated shapes the parent (#1704) asked to
//! catalogue — and decides, per branch, whether a collapse is **provably safe**
//! over the recorded observation window:
//!
//! - **IF conditional dominance (F1).** IF selection is driven by the condition
//!   synapse — the positive branch runs when the summed condition contribution
//!   is `> 0`, the negative branch when `≤ 0` — **not** by branch magnitude.
//!   A branch that looks dominated on one window becomes the *only* selected
//!   branch when the condition sign flips. A safe IF collapse must therefore
//!   first prove the condition is **degenerate** (always selects the same
//!   branch) over the window; see [`IfConditionRegime`].
//! - **Multi-branch aggregates.** More than two branches feeding one MAX/MIN/IF.
//!   A branch may be dominated by the *combination* of the others without being
//!   pairwise-dominated by any single one. The empirical win fraction
//!   ([`compute_selection_stats`]) captures this
//!   naturally: it evaluates the true multi-branch winner per observation, so a
//!   combination-dominated branch scores a win fraction of `0`.
//! - **Small-but-non-zero win fraction.** A branch that wins occasionally
//!   (empirical win fraction small but `> 0`) is **not** safe to remove. It is
//!   classified [`BranchDominance::Partial`] and surfaced only as a *gated
//!   candidate*: never removed on the empirical signal alone, only through the
//!   #1623 evaluate-before-accept gate.
//!
//! ## What this module does — and does not — do
//!
//! It **characterises** partial dominance and produces a **safety verdict** per
//! branch. It does **not** mutate the creature: the actual collapse transform
//! (remove the branch, fold the single-branch aggregate to a pass-through) is
//! the analytical-dominance detector of #1711/G1, which consumes this verdict
//! behind the evaluate-before-accept gate. This module is the *gate input* —
//! the proof-of-safety step the issue asks for — not the mutation.
//!
//! Failure is loud (Issue #3234): a branch with no win-fraction evidence is
//! treated conservatively as [`BranchDominance::Contributing`] and is never
//! reported safe, so absence of evidence never masquerades as proven dominance.

use super::compute_selection_stats;
use super::ranking::RecordProvider;
use crate::{CreatureJson, SynapseJson};
use anyhow::Result;
use std::collections::HashMap;

/// Default upper bound (exclusive) of the "partially dominated" band. A branch
/// whose empirical win fraction is in `(0, DEFAULT_PARTIAL_WIN_FRACTION)` wins
/// rarely but non-trivially — a gated candidate, never an automatic removal.
pub const DEFAULT_PARTIAL_WIN_FRACTION: f32 = 0.05;

/// Numerical slack for treating a win fraction as exactly zero. Win fractions
/// are exact `wins / total` ratios, so this only absorbs float round-trip noise.
const WIN_FRACTION_EPS: f32 = 1e-9;

/// Thresholds governing the partial-dominance classification.
#[derive(Debug, Clone, Copy)]
pub struct DominanceThresholds {
    /// A branch whose empirical win fraction is `<=` this is treated as fully
    /// dominated over the observation window. Default `0.0` — strict: only a
    /// branch that *never* wins qualifies.
    pub dominated_win_fraction: f32,
    /// Upper bound (exclusive) of the "partially dominated" band. A branch whose
    /// win fraction lies in `(dominated_win_fraction, partial_win_fraction)`
    /// wins occasionally and is **not** safe to remove without the
    /// evaluate-before-accept gate.
    pub partial_win_fraction: f32,
}

impl Default for DominanceThresholds {
    fn default() -> Self {
        Self {
            dominated_win_fraction: 0.0,
            partial_win_fraction: DEFAULT_PARTIAL_WIN_FRACTION,
        }
    }
}

/// How dominated a single branch is over the observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchDominance {
    /// Win fraction `<= dominated_win_fraction`: provably dominated over the
    /// window. A collapse *candidate*, still subject to the evaluate-before-accept
    /// gate before any removal.
    Dominated,
    /// `dominated_win_fraction < win fraction < partial_win_fraction`: wins
    /// occasionally. A gated candidate only — never removed on the empirical
    /// signal alone.
    Partial,
    /// Win fraction `>= partial_win_fraction` (or no evidence available): a
    /// genuine contributor. Must not be removed.
    Contributing,
}

/// Condition-degeneracy regime of an IF aggregate over the window (F1).
///
/// IF dominance is not a magnitude property — it is entirely condition-driven.
/// Only a *degenerate* condition (one that always selects the same branch)
/// makes the unselected branch safe to collapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfConditionRegime {
    /// Summed condition contribution `> 0` on every observation — only the
    /// positive branch is ever selected, so the negative branch is dominated
    /// *for this window*.
    AlwaysPositive,
    /// Summed condition contribution `<= 0` on every observation — only the
    /// negative branch runs; the positive branch is dominated.
    AlwaysNegative,
    /// Both branches are selected across the window. **Neither** branch is
    /// removable, regardless of branch magnitude — this is the F1 case.
    Mixed,
    /// No usable condition records — degeneracy cannot be proven, so nothing is
    /// safe. Distinct from `Mixed` because the reason is missing evidence, not a
    /// genuinely varying condition.
    Unknown,
}

impl IfConditionRegime {
    /// Is the condition provably degenerate (always selects one branch)?
    #[must_use]
    pub fn is_degenerate(self) -> bool {
        matches!(self, Self::AlwaysPositive | Self::AlwaysNegative)
    }
}

/// Safety verdict for one branch synapse feeding a selection aggregate.
#[derive(Debug, Clone)]
pub struct BranchVerdict {
    /// The selection aggregate (MIN/MAX/IF neuron) this branch feeds.
    pub aggregate_uuid: String,
    /// Source neuron of the branch synapse.
    pub from_uuid: String,
    /// IF branch role (`condition` / `positive` / `negative`); `None` for
    /// MIN/MAX branches, which carry no synapse type.
    pub synapse_type: Option<String>,
    /// Empirical win fraction over the window (`1.0` when no evidence, the
    /// conservative default).
    pub win_fraction: f32,
    /// Dominance classification under the supplied thresholds.
    pub dominance: BranchDominance,
    /// `true` only when the branch is provably safe to collapse over the window
    /// — still gated by the #1623 evaluate-before-accept check before removal.
    pub safe_to_collapse: bool,
}

/// Partial-dominance analysis for one selection aggregate.
#[derive(Debug, Clone)]
pub struct AggregateDominance {
    /// The selection neuron (MIN/MAX/IF).
    pub aggregate_uuid: String,
    /// Its squash: `MINIMUM`, `MAXIMUM`, or `IF`.
    pub squash: String,
    /// One verdict per incoming branch synapse.
    pub branches: Vec<BranchVerdict>,
    /// Condition regime — `Some` only for IF aggregates (F1).
    pub if_regime: Option<IfConditionRegime>,
}

impl AggregateDominance {
    /// Branches that are provably safe to collapse over the window (still gated
    /// by evaluate-before-accept).
    #[must_use]
    pub fn safe_branches(&self) -> Vec<&BranchVerdict> {
        self.branches
            .iter()
            .filter(|b| b.safe_to_collapse)
            .collect()
    }

    /// Branches in the partial band — collapse candidates only through the
    /// evaluate-before-accept gate, never on the empirical signal alone.
    #[must_use]
    pub fn gated_candidates(&self) -> Vec<&BranchVerdict> {
        self.branches
            .iter()
            .filter(|b| b.dominance == BranchDominance::Partial)
            .collect()
    }
}

/// Classify a win fraction into a [`BranchDominance`] band.
fn classify(win_fraction: f32, thresholds: &DominanceThresholds) -> BranchDominance {
    if win_fraction <= thresholds.dominated_win_fraction + WIN_FRACTION_EPS {
        BranchDominance::Dominated
    } else if win_fraction < thresholds.partial_win_fraction {
        BranchDominance::Partial
    } else {
        BranchDominance::Contributing
    }
}

/// Incoming branch synapses to a selection neuron, in creature order.
fn incoming_branches<'a>(creature: &'a CreatureJson, aggregate_uuid: &str) -> Vec<&'a SynapseJson> {
    creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == aggregate_uuid)
        .collect()
}

/// Compute the IF condition regime for one aggregate over the window (F1).
///
/// Sums the condition-synapse contributions (`weight × recorded activation`)
/// per observation, mirroring the IF semantics in
/// [`compute_selection_stats`], and reports
/// whether the condition is degenerate.
fn if_condition_regime(
    branches: &[&SynapseJson],
    records: &dyn RecordProvider,
) -> Result<IfConditionRegime> {
    let condition_synapses: Vec<&&SynapseJson> = branches
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some("condition"))
        .collect();

    if condition_synapses.is_empty() {
        return Ok(IfConditionRegime::Unknown);
    }

    let mut obs_sums: HashMap<u32, f32> = HashMap::new();
    for synapse in &condition_synapses {
        if let Some(recs) = records.get(&synapse.from_uuid)? {
            for record in recs.iter() {
                if record.activation.is_finite() {
                    *obs_sums.entry(record.obs_index).or_insert(0.0) +=
                        synapse.weight * record.activation;
                }
            }
        }
    }

    if obs_sums.is_empty() {
        return Ok(IfConditionRegime::Unknown);
    }

    let any_positive = obs_sums.values().any(|&s| s > 0.0);
    let any_non_positive = obs_sums.values().any(|&s| s <= 0.0);
    Ok(match (any_positive, any_non_positive) {
        (true, false) => IfConditionRegime::AlwaysPositive,
        (false, true) => IfConditionRegime::AlwaysNegative,
        _ => IfConditionRegime::Mixed,
    })
}

/// Whether an IF branch is provably safe to collapse, given the condition
/// regime. Only a branch the degenerate condition *never* selects is safe;
/// condition synapses are never removed here. Mixed/Unknown regimes make **no**
/// branch safe (F1).
fn if_branch_safe(
    synapse_type: Option<&str>,
    dominance: BranchDominance,
    regime: IfConditionRegime,
) -> bool {
    if dominance != BranchDominance::Dominated {
        return false;
    }
    match synapse_type {
        Some("positive") => regime == IfConditionRegime::AlwaysNegative,
        Some("negative") => regime == IfConditionRegime::AlwaysPositive,
        // Condition synapses (and untyped/unknown roles) are never collapsed.
        _ => false,
    }
}

/// Analyse partial dominance across every MAX / MIN / IF aggregate in the
/// creature, using the empirical win fractions from the recorded observation
/// window.
///
/// # Errors
/// Propagates record-provider access failures from
/// [`compute_selection_stats`].
pub fn analyse_partial_dominance(
    creature: &CreatureJson,
    records: &dyn RecordProvider,
    thresholds: &DominanceThresholds,
) -> Result<Vec<AggregateDominance>> {
    let squash_map = super::gradient::build_squash_map(creature);
    let stats = compute_selection_stats(creature, records)?;

    let mut result = Vec::new();
    for neuron in &creature.neurons {
        let squash = squash_map
            .get(&neuron.uuid)
            .map_or("", std::string::String::as_str);
        if !matches!(squash, "MINIMUM" | "MAXIMUM" | "IF") {
            continue;
        }

        let branches = incoming_branches(creature, &neuron.uuid);
        if branches.is_empty() {
            continue;
        }

        let if_regime = if squash == "IF" {
            Some(if_condition_regime(&branches, records)?)
        } else {
            None
        };

        let mut verdicts = Vec::with_capacity(branches.len());
        for synapse in &branches {
            let key = (synapse.from_uuid.clone(), synapse.to_uuid.clone());
            // No evidence ⇒ conservative win fraction of 1.0 (Contributing), so
            // absence of proof never reads as proven dominance (Issue #3234).
            let win_fraction = stats.get(&key).copied().unwrap_or(1.0);
            let dominance = classify(win_fraction, thresholds);

            let safe_to_collapse = match squash {
                "MINIMUM" | "MAXIMUM" => dominance == BranchDominance::Dominated,
                "IF" => if_branch_safe(
                    synapse.synapse_type.as_deref(),
                    dominance,
                    if_regime.unwrap_or(IfConditionRegime::Unknown),
                ),
                _ => false,
            };

            verdicts.push(BranchVerdict {
                aggregate_uuid: neuron.uuid.clone(),
                from_uuid: synapse.from_uuid.clone(),
                synapse_type: synapse.synapse_type.clone(),
                win_fraction,
                dominance,
                safe_to_collapse,
            });
        }

        result.push(AggregateDominance {
            aggregate_uuid: neuron.uuid.clone(),
            squash: squash.to_string(),
            branches: verdicts,
            if_regime,
        });
    }

    Ok(result)
}

/// Flatten the analysis to just the branches that are provably safe to collapse
/// (subject to the evaluate-before-accept gate) — the plan a collapse transform
/// (#1711) would consume.
///
/// # Errors
/// Propagates provider access failures from [`analyse_partial_dominance`].
pub fn safe_collapse_branches(
    creature: &CreatureJson,
    records: &dyn RecordProvider,
    thresholds: &DominanceThresholds,
) -> Result<Vec<BranchVerdict>> {
    Ok(analyse_partial_dominance(creature, records, thresholds)?
        .into_iter()
        .flat_map(|a| a.branches)
        .filter(|b| b.safe_to_collapse)
        .collect())
}
