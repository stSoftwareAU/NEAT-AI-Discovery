//! Analytical dominated-branch collapse for MAX/MIN selection aggregates
//! (Issue #1711, gap **G1** of `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md`).
//!
//! ## The gap this closes
//!
//! A branch feeding a MAXIMUM/MINIMUM aggregate can be *dominated* — provably
//! never the branch the aggregate selects — yet vary across the observation
//! window, so it is not *functionally constant* and the #1620/#1623 bias-fold
//! (`functionally_constant_neuron_uuids`) flags nothing. Before this module the
//! extent of automatic collapse for selection aggregates was **zero**.
//!
//! ## Analytical dominance proof
//!
//! Every branch feeds `weight × squash(pre-activation)` into the aggregate. Many
//! squashes are *one-signed for all real inputs*: `RELU`, `ABSOLUTE`, `SQUARE`,
//! `SQRT`, `GAUSSIAN`, `LOGISTIC`, `SOFTPLUS`, `EXPONENTIAL`, `RELU6`, `STEP`
//! are always `≥ 0`; `LOGSIGMOID` is always `≤ 0`. Multiplying by the branch
//! weight's sign gives the branch *contribution* sign — independent of the
//! upstream pre-activation, bias, or observation window, because the squash
//! range itself is one-signed.
//!
//! For a **MAXIMUM**, a branch whose contribution is always `≤ 0` can never be
//! selected while another branch is always `≥ 0`: the non-negative branch is
//! `≥ 0 ≥` the non-positive one on every input, so it is always at least as
//! large and removing the dominated branch leaves the maximum unchanged. For a
//! **MINIMUM** the mirror holds (sign flipped): an always-`≥ 0` branch can never
//! be the minimum while an always-`≤ 0` branch is present.
//!
//! This is a *sound* proof: it only ever flags a branch whose contribution sign
//! is provably opposite to, and dominated by, a surviving branch. Branches whose
//! squash spans both signs (`TANH`, `IDENTITY`, `SINE`, …) are never flagged.
//!
//! ## Collapse transform + evaluate-before-accept gate
//!
//! Consistent with the #1623 bias-fold pattern, the transform never deletes
//! blind. [`collapse_dominated_branch`] first runs an **evaluate-before-accept
//! gate**: for every recorded observation it compares the aggregate's selection
//! over *all* branches against its selection over the *surviving* branches and
//! measures the residual `|select_all − select_surviving|`. A genuinely
//! dominated branch leaves a zero residual and is accepted; a branch that only
//! *looks* dominated moves the selection, leaves a residual above tolerance, and
//! is **rejected** — the creature is left untouched. Missing records fail loud
//! (no blind delete), exactly as #1623.
//!
//! On acceptance the transform removes each dominated branch (and its now-dangling
//! branch neuron), and — if a single survivor remains — folds the pass-through
//! aggregate away: the survivor is rewired straight to the aggregate's targets
//! with `weight_in × weight_out`, and the aggregate's bias folds into each
//! target (`bias_agg × weight_out`). A single-input MAXIMUM/MINIMUM equals its
//! input, so this rewrite is exact.

use std::collections::HashSet;

use crate::CreatureJson;
use crate::activations::normalise_squash_name;
use crate::types::DiscoverRecord;

/// Default tolerance for the evaluate-before-accept gate.
///
/// Mirrors [`super::remove_neuron_bias_fold::BIAS_FOLD_GATE_TOLERANCE`]: `1e-6`
/// sits well above `f32`-recording round-off yet far below any selection change
/// a genuinely non-dominated branch would introduce.
pub const COLLAPSE_GATE_TOLERANCE: f64 = 1e-6;

/// Sign of a scalar squash's output range over **all** real inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquashSign {
    /// Output is always `≥ 0` (e.g. `RELU`, `ABSOLUTE`, `SQUARE`, `LOGISTIC`).
    NonNegative,
    /// Output is always `≤ 0` (e.g. `LOGSIGMOID`).
    NonPositive,
    /// Output spans both signs, or the squash is aggregate/unknown.
    Mixed,
}

impl SquashSign {
    /// The sign obtained by multiplying this one-signed range by `−1`.
    fn flipped(self) -> Self {
        match self {
            Self::NonNegative => Self::NonPositive,
            Self::NonPositive => Self::NonNegative,
            Self::Mixed => Self::Mixed,
        }
    }
}

/// Analytical sign of a scalar squash's output range over all real inputs.
///
/// Sound and conservative: only squashes whose **entire** output range is
/// one-signed for every real input are classified `NonNegative`/`NonPositive`.
/// Everything else — two-signed squashes, aggregate squashes, and unknown names
/// — is `Mixed`, so it can never seed a dominance claim.
#[must_use]
pub fn scalar_squash_sign(name: &str) -> SquashSign {
    let n = normalise_squash_name(name);
    match n.as_ref() {
        // Always ≥ 0 across all real inputs.
        "ABSOLUTE" | "RELU" | "RELU6" | "SQUARE" | "SQRT" | "GAUSSIAN" | "LOGISTIC"
        | "SOFTPLUS" | "EXPONENTIAL" | "STEP" => SquashSign::NonNegative,
        // Always ≤ 0 across all real inputs: LOGSIGMOID(x) = −ln(1+e^{−x}) ≤ 0.
        "LOGSIGMOID" => SquashSign::NonPositive,
        // Two-signed, aggregate, or unknown — never a dominance seed.
        _ => SquashSign::Mixed,
    }
}

/// The contribution sign a branch feeds into its aggregate: `weight × squash`.
///
/// A positive weight preserves the squash sign, a negative weight flips it. A
/// zero weight is treated as `Mixed` — a degenerate branch is never collapsed.
#[must_use]
fn branch_contribution_sign(squash: &str, weight: f32) -> SquashSign {
    let base = scalar_squash_sign(squash);
    if weight > 0.0 {
        base
    } else if weight < 0.0 {
        base.flipped()
    } else {
        SquashSign::Mixed
    }
}

/// Which selection aggregate a neuron is — the two this module collapses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateKind {
    /// `MAXIMUM` — selects the largest branch contribution.
    Maximum,
    /// `MINIMUM` — selects the smallest branch contribution.
    Minimum,
}

impl AggregateKind {
    fn from_squash(name: &str) -> Option<Self> {
        match normalise_squash_name(name).as_ref() {
            "MAXIMUM" => Some(Self::Maximum),
            "MINIMUM" => Some(Self::Minimum),
            _ => None,
        }
    }

    /// Select the winning value from a set of branch contributions.
    fn select(self, contributions: &[f64]) -> Option<f64> {
        contributions.iter().copied().reduce(|acc, c| match self {
            Self::Maximum => acc.max(c),
            Self::Minimum => acc.min(c),
        })
    }
}

/// One analytically-dominated branch feeding a MAX/MIN aggregate (Issue #1711).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DominatedBranch {
    /// The MAX/MIN aggregate neuron the branch feeds.
    pub aggregate_uuid: String,
    /// The dominated branch's source neuron — never selected by the aggregate.
    pub branch_uuid: String,
    /// A surviving branch that provably dominates it (opposite, winning sign).
    pub dominator_uuid: String,
    /// Which aggregate the dominance was proven against.
    pub aggregate_kind: AggregateKind,
}

/// A single branch feeding an aggregate: source neuron, weight, contribution sign.
struct Branch {
    from_uuid: String,
    sign: SquashSign,
}

/// Collect the branches feeding an aggregate neuron and classify each branch's
/// contribution sign from its source neuron's squash and the synapse weight.
fn aggregate_branches(creature: &CreatureJson, aggregate_uuid: &str) -> Vec<Branch> {
    creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == aggregate_uuid)
        .map(|s| {
            let squash = creature
                .neurons
                .iter()
                .find(|n| n.uuid == s.from_uuid)
                .map_or("", |n| n.squash.as_str());
            Branch {
                from_uuid: s.from_uuid.clone(),
                sign: branch_contribution_sign(squash, s.weight),
            }
        })
        .collect()
}

/// Detect every analytically-dominated branch across all MAX/MIN aggregates
/// (Issue #1711).
///
/// For a MAXIMUM, a branch whose contribution is always `≤ 0` is dominated when
/// at least one branch is always `≥ 0`. For a MINIMUM the signs are mirrored.
/// The proof is sign-based and sound, so a branch is flagged only when a
/// surviving branch provably wins against it on every input.
#[must_use]
pub fn detect_dominated_branches(creature: &CreatureJson) -> Vec<DominatedBranch> {
    let mut out = Vec::new();
    for neuron in &creature.neurons {
        let Some(kind) = AggregateKind::from_squash(&neuron.squash) else {
            continue;
        };
        let branches = aggregate_branches(creature, &neuron.uuid);
        // For MAX the dominator sign is NonNegative and the dominated sign is
        // NonPositive; for MIN they swap.
        let (dominator_sign, dominated_sign) = match kind {
            AggregateKind::Maximum => (SquashSign::NonNegative, SquashSign::NonPositive),
            AggregateKind::Minimum => (SquashSign::NonPositive, SquashSign::NonNegative),
        };
        let Some(dominator) = branches.iter().find(|b| b.sign == dominator_sign) else {
            continue;
        };
        for branch in branches.iter().filter(|b| b.sign == dominated_sign) {
            out.push(DominatedBranch {
                aggregate_uuid: neuron.uuid.clone(),
                branch_uuid: branch.from_uuid.clone(),
                dominator_uuid: dominator.from_uuid.clone(),
                aggregate_kind: kind,
            });
        }
    }
    out
}

/// UUIDs of every analytically-dominated branch neuron — the seam mirroring
/// [`super::functionally_constant_neuron_uuids`] (Issue #1711).
///
/// Unlike that seam, this one is *not* a stub: it returns the real dominated set
/// so the orchestrator (and characterisation suites) can consume analytical
/// dominance directly.
#[must_use]
pub fn analytically_dominated_branch_uuids(creature: &CreatureJson) -> HashSet<String> {
    detect_dominated_branches(creature)
        .into_iter()
        .map(|d| d.branch_uuid)
        .collect()
}

/// The outcome of evaluating (and possibly applying) a dominated-branch collapse
/// (Issue #1711).
#[derive(Debug, Clone, PartialEq)]
pub struct CollapseOutcome {
    /// `true` when the evaluate-before-accept gate passed and the collapse was
    /// applied; `false` when the gate rejected it (the creature is unchanged).
    pub accepted: bool,
    /// The MAX/MIN aggregate the collapse targeted.
    pub aggregate_uuid: String,
    /// The branch neurons removed by the collapse.
    pub removed_branch_uuids: Vec<String>,
    /// `true` when a single survivor remained and the aggregate was folded to a
    /// pass-through (aggregate neuron removed, survivor rewired to its targets).
    pub folded_to_passthrough: bool,
    /// The maximum per-observation selection residual `|select_all −
    /// select_surviving|` measured by the gate.
    pub max_residual: f64,
    /// When the collapse could not be evaluated or was rejected, a short reason;
    /// `None` on acceptance.
    pub rejection_reason: Option<String>,
}

fn reject(aggregate_uuid: &str, reason: String) -> CollapseOutcome {
    CollapseOutcome {
        accepted: false,
        aggregate_uuid: aggregate_uuid.to_string(),
        removed_branch_uuids: Vec::new(),
        folded_to_passthrough: false,
        max_residual: 0.0,
        rejection_reason: Some(reason),
    }
}

/// Per-observation activation of a single neuron, keyed by observation index.
fn activations_by_obs(records: &[DiscoverRecord], neuron_uuid: &str) -> Vec<(u32, f64)> {
    records
        .iter()
        .filter(|r| r.neuron_uuid == neuron_uuid)
        .map(|r| (r.obs_index, f64::from(r.activation)))
        .collect()
}

/// Evaluate a dominated-branch collapse **without** mutating the creature
/// (Issue #1711).
///
/// Runs the analytical detector for `aggregate_uuid`, then the empirical gate:
/// for every observation for which all branch activations are recorded, it
/// compares the aggregate's selection over all branches against its selection
/// over the surviving branches and tracks the maximum residual. Returns `None`
/// when there is nothing to evaluate (not a MAX/MIN aggregate, or no dominated
/// branch); a fail-loud rejection (missing records) is returned as an outcome so
/// no branch is ever deleted blind.
#[must_use]
pub fn evaluate_dominated_branch_collapse(
    creature: &CreatureJson,
    records: &[DiscoverRecord],
    aggregate_uuid: &str,
    tolerance: f64,
) -> Option<CollapseOutcome> {
    let kind = creature
        .neurons
        .iter()
        .find(|n| n.uuid == aggregate_uuid)
        .and_then(|n| AggregateKind::from_squash(&n.squash))?;

    let dominated: Vec<DominatedBranch> = detect_dominated_branches(creature)
        .into_iter()
        .filter(|d| d.aggregate_uuid == aggregate_uuid)
        .collect();
    if dominated.is_empty() {
        return None;
    }
    let dominated_set: HashSet<&str> = dominated.iter().map(|d| d.branch_uuid.as_str()).collect();

    // Every branch feeding the aggregate, with its weight, plus the per-obs
    // activations of its source neuron.
    let branch_synapses: Vec<(&str, f64)> = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == aggregate_uuid)
        .map(|s| (s.from_uuid.as_str(), f64::from(s.weight)))
        .collect();

    // Index each branch's activations by observation.
    let mut obs_indices: HashSet<u32> = HashSet::new();
    let mut per_branch: Vec<(&str, f64, std::collections::HashMap<u32, f64>)> = Vec::new();
    for &(uuid, weight) in &branch_synapses {
        let map: std::collections::HashMap<u32, f64> = activations_by_obs(records, uuid)
            .into_iter()
            .inspect(|(obs, _)| {
                obs_indices.insert(*obs);
            })
            .collect();
        per_branch.push((uuid, weight, map));
    }

    if obs_indices.is_empty() {
        // No recorded branch activations — constancy of the selection cannot be
        // verified, so fail loud rather than delete blind (Issue #3234).
        return Some(reject(
            aggregate_uuid,
            format!(
                "no recorded activations for the branches of aggregate {aggregate_uuid}; cannot verify dominance"
            ),
        ));
    }

    let mut max_residual = 0.0_f64;
    for obs in &obs_indices {
        let mut all_contribs = Vec::with_capacity(per_branch.len());
        let mut surviving_contribs = Vec::with_capacity(per_branch.len());
        let mut complete = true;
        for (uuid, weight, map) in &per_branch {
            let Some(&activation) = map.get(obs) else {
                complete = false;
                break;
            };
            let contribution = weight * activation;
            all_contribs.push(contribution);
            if !dominated_set.contains(uuid) {
                surviving_contribs.push(contribution);
            }
        }
        // Skip observations missing a branch activation — an incomplete record
        // cannot confirm or refute dominance for that observation.
        if !complete || surviving_contribs.is_empty() {
            continue;
        }
        let full = kind.select(&all_contribs).unwrap_or(0.0);
        let reduced = kind.select(&surviving_contribs).unwrap_or(0.0);
        max_residual = max_residual.max((full - reduced).abs());
    }

    let accepted = max_residual <= tolerance;
    let removed_branch_uuids: Vec<String> =
        dominated.iter().map(|d| d.branch_uuid.clone()).collect();
    let rejection_reason = if accepted {
        None
    } else {
        Some(format!(
            "per-observation selection residual {max_residual:.3e} exceeds gate tolerance {tolerance:.3e}"
        ))
    };

    // A single remaining survivor means the aggregate becomes a pass-through.
    let survivors = branch_synapses.len().saturating_sub(dominated.len());

    Some(CollapseOutcome {
        accepted,
        aggregate_uuid: aggregate_uuid.to_string(),
        removed_branch_uuids,
        folded_to_passthrough: accepted && survivors == 1,
        max_residual,
        rejection_reason,
    })
}

/// Delete a branch neuron together with every synapse touching it, but only when
/// removing its synapse into the aggregate leaves it with no other consumers.
///
/// A branch neuron that still feeds other neurons keeps its incoming edges; only
/// its edge into the collapsed aggregate is removed.
fn remove_branch(creature: &mut CreatureJson, branch_uuid: &str, aggregate_uuid: &str) {
    // Drop the branch → aggregate edge first.
    creature
        .synapses
        .retain(|s| !(s.from_uuid == branch_uuid && s.to_uuid == aggregate_uuid));
    // If the branch neuron now feeds nothing else, it is dangling — remove it and
    // its incoming edges too, matching the full-collapse target.
    let still_consumed = creature.synapses.iter().any(|s| s.from_uuid == branch_uuid);
    if !still_consumed {
        creature.neurons.retain(|n| n.uuid != branch_uuid);
        creature.synapses.retain(|s| s.to_uuid != branch_uuid);
    }
}

/// Fold a single-survivor aggregate to a pass-through: rewire the survivor to the
/// aggregate's targets and fold the aggregate bias, then delete the aggregate.
///
/// A single-input MAXIMUM/MINIMUM equals its input, so for each target `T` the
/// aggregate's contribution `w_out · (w_in · a_S + bias_agg)` is reproduced
/// exactly by a direct `S → T` edge of weight `w_in · w_out` plus a
/// `bias_agg · w_out` addition to `T`'s bias. Existing `S → T` edges are merged
/// by summing weights so no duplicate edge is created.
fn fold_passthrough(creature: &mut CreatureJson, aggregate_uuid: &str) {
    // The lone surviving incoming edge and its weight.
    let Some((survivor_uuid, weight_in)) = creature
        .synapses
        .iter()
        .find(|s| s.to_uuid == aggregate_uuid)
        .map(|s| (s.from_uuid.clone(), s.weight))
    else {
        return;
    };
    let bias_agg = creature
        .neurons
        .iter()
        .find(|n| n.uuid == aggregate_uuid)
        .map_or(0.0_f32, |n| n.bias);

    // Snapshot the aggregate's outgoing edges before mutating the synapse list.
    let outgoing: Vec<(String, f32)> = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == aggregate_uuid)
        .map(|s| (s.to_uuid.clone(), s.weight))
        .collect();

    for (target_uuid, weight_out) in &outgoing {
        // Fold the aggregate bias into the target: bias_agg × weight_out.
        if bias_agg != 0.0 {
            for neuron in creature
                .neurons
                .iter_mut()
                .filter(|n| n.uuid == *target_uuid)
            {
                neuron.bias += bias_agg * weight_out;
            }
        }
        let folded_weight = weight_in * weight_out;
        // Merge into an existing survivor → target edge if present, else add one.
        if let Some(existing) = creature
            .synapses
            .iter_mut()
            .find(|s| s.from_uuid == survivor_uuid && s.to_uuid == *target_uuid)
        {
            existing.weight += folded_weight;
        } else {
            creature.synapses.push(crate::ffi_types::SynapseJson {
                from_uuid: survivor_uuid.clone(),
                to_uuid: target_uuid.clone(),
                weight: folded_weight,
                synapse_type: None,
            });
        }
    }

    // Remove the aggregate neuron and every synapse touching it.
    creature.neurons.retain(|n| n.uuid != aggregate_uuid);
    creature
        .synapses
        .retain(|s| s.from_uuid != aggregate_uuid && s.to_uuid != aggregate_uuid);
}

/// Collapse the analytically-dominated branch(es) of a MAX/MIN aggregate behind
/// the evaluate-before-accept gate (Issue #1711).
///
/// Evaluates the collapse, and **only if the gate passes** applies it: each
/// dominated branch (and its now-dangling branch neuron) is removed, and if a
/// single survivor remains the pass-through aggregate is folded away. If the gate
/// rejects the collapse (a branch that only looks dominated moves the selection)
/// or it cannot be evaluated (no recorded activations), the creature is left
/// **exactly** as it was and the returned outcome reports why. Returns `None`
/// when there is nothing to collapse (not a MAX/MIN aggregate, or no dominated
/// branch).
#[must_use]
pub fn collapse_dominated_branch(
    creature: &mut CreatureJson,
    records: &[DiscoverRecord],
    aggregate_uuid: &str,
    tolerance: f64,
) -> Option<CollapseOutcome> {
    let outcome = evaluate_dominated_branch_collapse(creature, records, aggregate_uuid, tolerance)?;
    if !outcome.accepted {
        return Some(outcome);
    }
    for branch_uuid in &outcome.removed_branch_uuids {
        remove_branch(creature, branch_uuid, aggregate_uuid);
    }
    if outcome.folded_to_passthrough {
        fold_passthrough(creature, aggregate_uuid);
    }
    Some(outcome)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Intentional numeric casts in test helpers (Issue #873)
mod tests {
    use super::*;
    use crate::ffi_types::{NeuronJson, SynapseJson};

    fn neuron(uuid: &str, neuron_type: &str, squash: &str, bias: f32) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: squash.to_string(),
            bias,
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

    fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
        DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
    }

    /// The worked example: ABSOLUTE × (−1) vs RELU into a MAXIMUM.
    fn max_worked_example() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                neuron("neuron-abs", "hidden", "ABSOLUTE", 0.0),
                neuron("neuron-relu", "hidden", "RELU", 0.0),
                neuron("neuron-max", "hidden", "MAXIMUM", 0.0),
                neuron("output-0", "output", "IDENTITY", 0.0),
            ],
            synapses: vec![
                synapse("input-0", "neuron-abs", 1.0),
                synapse("input-1", "neuron-relu", 1.0),
                synapse("neuron-abs", "neuron-max", -1.0),
                synapse("neuron-relu", "neuron-max", 1.0),
                synapse("neuron-max", "output-0", 1.0),
            ],
            input: 2,
            output: 1,
        }
    }

    /// Records where the ABSOLUTE branch (≤ 0) never wins the MAXIMUM.
    fn max_records() -> Vec<DiscoverRecord> {
        let mut recs = Vec::new();
        for obs in 0..8u32 {
            let abs = (obs as f32) - 3.5; // pre-activation; ABSOLUTE ≥ 0
            recs.push(record(obs, "neuron-abs", abs.abs()));
            recs.push(record(obs, "neuron-relu", (obs as f32).max(0.0)));
            // Aggregate activation = max(−|abs|, relu) = relu here.
            recs.push(record(obs, "neuron-max", (obs as f32).max(0.0)));
        }
        recs
    }

    // ---- Analytical sign classifier ------------------------------------

    #[test]
    fn one_signed_squashes_classified() {
        assert_eq!(scalar_squash_sign("RELU"), SquashSign::NonNegative);
        assert_eq!(scalar_squash_sign("absolute"), SquashSign::NonNegative);
        assert_eq!(scalar_squash_sign("LOGISTIC"), SquashSign::NonNegative);
        assert_eq!(scalar_squash_sign("SOFTPLUS"), SquashSign::NonNegative);
        assert_eq!(scalar_squash_sign("LOGSIGMOID"), SquashSign::NonPositive);
    }

    #[test]
    fn two_signed_and_unknown_squashes_are_mixed() {
        for name in ["TANH", "IDENTITY", "SINE", "SELU", "MAXIMUM", "WAT"] {
            assert_eq!(
                scalar_squash_sign(name),
                SquashSign::Mixed,
                "{name} must be Mixed"
            );
        }
    }

    #[test]
    fn contribution_sign_flips_with_negative_weight() {
        // ABSOLUTE (≥0) × (−1) ⇒ ≤ 0.
        assert_eq!(
            branch_contribution_sign("ABSOLUTE", -1.0),
            SquashSign::NonPositive
        );
        // RELU (≥0) × (+1) ⇒ ≥ 0.
        assert_eq!(
            branch_contribution_sign("RELU", 1.0),
            SquashSign::NonNegative
        );
        // Zero weight is a degenerate branch — never a dominance seed.
        assert_eq!(branch_contribution_sign("RELU", 0.0), SquashSign::Mixed);
    }

    // ---- Detector ------------------------------------------------------

    #[test]
    fn detects_dominated_absolute_branch_in_maximum() {
        let creature = max_worked_example();
        let dominated = detect_dominated_branches(&creature);
        assert_eq!(dominated.len(), 1);
        assert_eq!(dominated[0].branch_uuid, "neuron-abs");
        assert_eq!(dominated[0].dominator_uuid, "neuron-relu");
        assert_eq!(dominated[0].aggregate_kind, AggregateKind::Maximum);

        let uuids = analytically_dominated_branch_uuids(&creature);
        assert!(uuids.contains("neuron-abs"));
        assert!(!uuids.contains("neuron-relu"));
    }

    #[test]
    fn detects_dominated_relu_branch_in_minimum() {
        let mut creature = max_worked_example();
        // Reshape into the MINIMUM mirror: RELU (≥0) is dominated.
        for n in &mut creature.neurons {
            if n.uuid == "neuron-max" {
                n.uuid = "neuron-min".to_string();
                n.squash = "MINIMUM".to_string();
            }
        }
        for s in &mut creature.synapses {
            if s.to_uuid == "neuron-max" {
                s.to_uuid = "neuron-min".to_string();
            }
            if s.from_uuid == "neuron-max" {
                s.from_uuid = "neuron-min".to_string();
            }
        }
        let dominated = detect_dominated_branches(&creature);
        assert_eq!(dominated.len(), 1);
        assert_eq!(dominated[0].branch_uuid, "neuron-relu");
        assert_eq!(dominated[0].dominator_uuid, "neuron-abs");
        assert_eq!(dominated[0].aggregate_kind, AggregateKind::Minimum);
    }

    #[test]
    fn no_dominance_when_all_branches_same_sign() {
        // Two RELU branches (both ≥ 0) into a MAXIMUM — neither is dominated.
        let creature = CreatureJson {
            neurons: vec![
                neuron("a", "hidden", "RELU", 0.0),
                neuron("b", "hidden", "RELU", 0.0),
                neuron("m", "hidden", "MAXIMUM", 0.0),
                neuron("out", "output", "IDENTITY", 0.0),
            ],
            synapses: vec![
                synapse("a", "m", 1.0),
                synapse("b", "m", 1.0),
                synapse("m", "out", 1.0),
            ],
            input: 2,
            output: 1,
        };
        assert!(detect_dominated_branches(&creature).is_empty());
    }

    // ---- Collapse transform + gate -------------------------------------

    #[test]
    fn collapse_removes_dominated_branch_and_folds_to_passthrough() {
        let mut creature = max_worked_example();
        let outcome = collapse_dominated_branch(
            &mut creature,
            &max_records(),
            "neuron-max",
            COLLAPSE_GATE_TOLERANCE,
        )
        .expect("dominated branch present");
        assert!(outcome.accepted, "{:?}", outcome.rejection_reason);
        assert!(outcome.folded_to_passthrough);
        assert!(outcome.max_residual <= COLLAPSE_GATE_TOLERANCE);

        // Target end state: input-1 → neuron-relu → output-0 (2 neurons, 2 synapses).
        assert_eq!(creature.neurons.len(), 2);
        assert_eq!(creature.synapses.len(), 2);
        assert!(!creature.neurons.iter().any(|n| n.uuid == "neuron-abs"));
        assert!(!creature.neurons.iter().any(|n| n.uuid == "neuron-max"));
        assert!(creature.neurons.iter().any(|n| n.uuid == "neuron-relu"));
        // The survivor is wired straight to the output with folded weight 1×1 = 1.
        let edge = creature
            .synapses
            .iter()
            .find(|s| s.from_uuid == "neuron-relu" && s.to_uuid == "output-0")
            .expect("survivor rewired to output");
        assert!((edge.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gate_rejects_a_branch_that_actually_wins() {
        // Records where the "dominated" branch actually takes the maximum on one
        // observation — the empirical gate must reject and leave the creature
        // untouched even though the analytical detector flagged it.
        let mut creature = max_worked_example();
        let before = creature.clone();
        let mut recs = max_records();
        // Force obs 0: abs branch contributes +5 (impossible for a true ABSOLUTE
        // ×(−1) branch, but the gate must not trust the label blindly).
        recs.retain(|r| !(r.obs_index == 0));
        recs.push(record(0, "neuron-abs", -5.0)); // contribution = (−1)×(−5) = +5
        recs.push(record(0, "neuron-relu", 1.0));
        recs.push(record(0, "neuron-max", 5.0));

        let outcome =
            collapse_dominated_branch(&mut creature, &recs, "neuron-max", COLLAPSE_GATE_TOLERANCE)
                .expect("dominated branch flagged");
        assert!(!outcome.accepted);
        assert!(outcome.rejection_reason.is_some());
        assert!(outcome.max_residual > COLLAPSE_GATE_TOLERANCE);
        assert_eq!(creature.neurons.len(), before.neurons.len());
        assert_eq!(creature.synapses.len(), before.synapses.len());
    }

    #[test]
    fn missing_records_fail_loud_without_deleting() {
        let mut creature = max_worked_example();
        let outcome =
            collapse_dominated_branch(&mut creature, &[], "neuron-max", COLLAPSE_GATE_TOLERANCE)
                .expect("dominated branch flagged");
        assert!(!outcome.accepted);
        assert!(outcome.rejection_reason.is_some());
        // No blind delete.
        assert!(creature.neurons.iter().any(|n| n.uuid == "neuron-abs"));
        assert!(creature.neurons.iter().any(|n| n.uuid == "neuron-max"));
    }

    #[test]
    fn non_aggregate_target_returns_none() {
        let creature = max_worked_example();
        // output-0 is IDENTITY, not a MAX/MIN aggregate.
        assert!(
            evaluate_dominated_branch_collapse(
                &creature,
                &max_records(),
                "output-0",
                COLLAPSE_GATE_TOLERANCE
            )
            .is_none()
        );
    }

    #[test]
    fn aggregate_bias_folds_into_target_on_passthrough() {
        let mut creature = max_worked_example();
        // Give the aggregate a bias and the output edge a non-unit weight.
        for n in &mut creature.neurons {
            if n.uuid == "neuron-max" {
                n.bias = 2.0;
            }
        }
        for s in &mut creature.synapses {
            if s.from_uuid == "neuron-max" && s.to_uuid == "output-0" {
                s.weight = 3.0;
            }
        }
        let outcome = collapse_dominated_branch(
            &mut creature,
            &max_records(),
            "neuron-max",
            COLLAPSE_GATE_TOLERANCE,
        )
        .expect("dominated branch present");
        assert!(outcome.accepted);
        // survivor → output weight = w_in(1) × w_out(3) = 3.
        let edge = creature
            .synapses
            .iter()
            .find(|s| s.from_uuid == "neuron-relu" && s.to_uuid == "output-0")
            .expect("survivor rewired");
        assert!((edge.weight - 3.0).abs() < 1e-5);
        // output bias gains bias_agg(2) × w_out(3) = 6.
        let out = creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!((out.bias - 6.0).abs() < 1e-5);
    }
}
