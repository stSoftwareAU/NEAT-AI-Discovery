//! Weight-redistribution compensation for remove-neuron candidates
//! (Issue #1559).
//!
//! ## Background
//!
//! The #1558 counterfactual study asked *"what would have made remove-neuron
//! `1359350630` succeed?"* Its finding: NEAT-AI's mean-preserving **bias**
//! compensation cancels only the *mean* of a removed neuron's downstream
//! contribution. What survives is the neuron's genuine **per-sample (variance)**
//! downstream signal, and no counterfactual evaluable against the persisted
//! scalar-aggregate artefact flips the removal's score delta to `≥ 0`.
//!
//! Of the strategies studied, only **(d) — redistributing the removed neuron's
//! per-sample signal into a downstream weight** has a ceiling that reaches a
//! neutral (non-regressive) removal. Its blocker was data: (d) needs the
//! per-sample distribution of the candidate's activation *and* its cross-neuron
//! correlation with surviving neurons, but the failure artefact held only
//! scalar aggregates (`averageActivation`, `sampleCount`).
//!
//! ## What this module persists
//!
//! Persisting full per-sample activation vectors for every candidate is
//! prohibitive. Instead we persist a compact **sufficient statistic**: the
//! per-pair mean, variance, and covariance of the candidate neuron against each
//! surviving neuron that shares a downstream target
//! ([`ActivationCovariance`]). That is `O(survivors)` scalars per candidate,
//! independent of the sample count, yet it is exactly the statistic
//! counterfactual (d) needs.
//!
//! ## What this module evaluates
//!
//! [`evaluate_weight_redistribution`] folds the candidate's per-sample
//! contribution into a survivor's downstream weight rather than only its bias.
//! The optimal weight bump is the least-squares regression coefficient
//! `Δw = w_c · cov(a_c, a_s) / var(a_s)`; the residual per-sample variance it
//! leaves is `w_c² · var(a_c) · (1 − ρ²)`, where `ρ` is the candidate/survivor
//! activation correlation. A perfectly correlated survivor (`ρ = 1`) drives the
//! residual to zero — the removal becomes non-regressive, which the mean-only
//! bias lever can never achieve.

use crate::CreatureJson;
use crate::types::DiscoverRecord;
use std::collections::HashMap;

/// Variances at or below this magnitude are treated as zero — a neuron whose
/// activation never varies carries no per-sample signal to redistribute.
const VARIANCE_EPSILON: f64 = 1e-12;

/// Compact per-pair sufficient statistic for counterfactual (d) (Issue #1559).
///
/// Holds the population mean, variance, and covariance of a **candidate**
/// neuron's per-sample activation against one **surviving** neuron's, plus the
/// sample count. This is the minimal data needed to evaluate whether the
/// candidate's per-sample downstream signal can be folded into the survivor's
/// weight, and it is `O(1)` in size regardless of how many samples were
/// observed — cheap enough to persist per candidate/survivor pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationCovariance {
    /// Number of aligned per-sample activation pairs observed.
    pub count: u64,
    /// Mean of the candidate neuron's per-sample activation.
    pub candidate_mean: f64,
    /// Mean of the surviving neuron's per-sample activation.
    pub survivor_mean: f64,
    /// Population variance of the candidate neuron's per-sample activation.
    pub candidate_variance: f64,
    /// Population variance of the surviving neuron's per-sample activation.
    pub survivor_variance: f64,
    /// Population covariance of the candidate and survivor activations.
    pub covariance: f64,
}

impl ActivationCovariance {
    /// Accumulate the sufficient statistic from aligned `(candidate, survivor)`
    /// per-sample activation pairs.
    ///
    /// Uses a single-pass, numerically stable co-moment update (the two-variable
    /// extension of Welford's algorithm) so long sample streams do not lose
    /// precision to catastrophic cancellation. Returns `None` for an empty
    /// stream — there is nothing to persist and (d) cannot be evaluated.
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (f64, f64)>) -> Option<Self> {
        let mut count: u64 = 0;
        // Track the sample count as an `f64` alongside the integer count to
        // avoid a lossy `u64 as f64` cast in the running-mean update.
        let mut n = 0.0_f64;
        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        let mut m2_x = 0.0;
        let mut m2_y = 0.0;
        let mut co_moment = 0.0;

        for (x, y) in pairs {
            count += 1;
            n += 1.0;
            let dx = x - mean_x;
            let dy = y - mean_y;
            mean_x += dx / n;
            mean_y += dy / n;
            // Use the *updated* means for the second factor (stable form).
            m2_x += dx * (x - mean_x);
            m2_y += dy * (y - mean_y);
            co_moment += dx * (y - mean_y);
        }

        if count == 0 {
            return None;
        }

        Some(Self {
            count,
            candidate_mean: mean_x,
            survivor_mean: mean_y,
            candidate_variance: m2_x / n,
            survivor_variance: m2_y / n,
            covariance: co_moment / n,
        })
    }

    /// Pearson correlation coefficient between the candidate and survivor
    /// activations, in `[-1, 1]`.
    ///
    /// Returns `0.0` when either neuron's activation has (near-)zero variance —
    /// a constant signal is uncorrelated with everything and offers no
    /// redistribution leverage.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        if self.candidate_variance <= VARIANCE_EPSILON || self.survivor_variance <= VARIANCE_EPSILON
        {
            return 0.0;
        }
        let denom = (self.candidate_variance * self.survivor_variance).sqrt();
        (self.covariance / denom).clamp(-1.0, 1.0)
    }
}

/// The outcome of evaluating counterfactual (d) for one candidate/survivor pair
/// (Issue #1559).
///
/// Compares NEAT-AI's current mean-only **bias** compensation against folding
/// the candidate's per-sample contribution into the survivor's downstream
/// **weight**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightRedistribution {
    /// Optimal least-squares weight bump `Δw` to add to the survivor's weight
    /// into the shared downstream target: `Δw = w_c · cov / var(a_s)`.
    pub delta_weight: f64,
    /// Per-sample residual variance under mean-preserving **bias-only**
    /// compensation — the full variance of the removed contribution,
    /// `w_c² · var(a_c)`. This is the residual cost that survives NEAT-AI's
    /// current lever (the `-5.58e-4` in #1558).
    pub bias_only_residual_variance: f64,
    /// Per-sample residual variance after **weight redistribution** (d):
    /// `w_c² · var(a_c) · (1 − ρ²)`. Zero when a perfectly correlated survivor
    /// absorbs the entire per-sample signal.
    pub redistributed_residual_variance: f64,
    /// Variance recovered by redistribution over bias-only compensation,
    /// `bias_only_residual_variance − redistributed_residual_variance` (`≥ 0`).
    pub variance_recovered: f64,
    /// `true` when redistribution drives the residual variance to ~0 — the
    /// removal becomes non-regressive (delta flips to `≥ 0`), which the
    /// mean-only bias lever can never achieve.
    pub fully_compensable: bool,
}

/// Evaluate counterfactual (d): fold the candidate's per-sample downstream
/// contribution into a correlated survivor's weight rather than only its bias
/// (Issue #1559).
///
/// # Arguments
/// * `candidate_downstream_weight` - the candidate neuron's synapse weight
///   `w_c` into the shared downstream target.
/// * `stats` - the persisted [`ActivationCovariance`] sufficient statistic for
///   the candidate against the chosen survivor.
///
/// # Returns
/// A [`WeightRedistribution`] describing the optimal weight bump, the residual
/// per-sample variance before/after redistribution, and whether the removal
/// becomes fully compensable.
#[must_use]
pub fn evaluate_weight_redistribution(
    candidate_downstream_weight: f64,
    stats: &ActivationCovariance,
) -> WeightRedistribution {
    let wc = candidate_downstream_weight;
    // Bias compensation cancels the mean, so the full per-sample variance of
    // the removed contribution survives as residual cost.
    let bias_only = wc * wc * stats.candidate_variance;

    // A constant candidate (no per-sample variance) is already fully handled by
    // the bias lever — nothing to redistribute.
    if bias_only <= VARIANCE_EPSILON {
        return WeightRedistribution {
            delta_weight: 0.0,
            bias_only_residual_variance: bias_only,
            redistributed_residual_variance: bias_only,
            variance_recovered: 0.0,
            fully_compensable: true,
        };
    }

    // A constant survivor carries no signal to absorb the candidate's variance.
    if stats.survivor_variance <= VARIANCE_EPSILON {
        return WeightRedistribution {
            delta_weight: 0.0,
            bias_only_residual_variance: bias_only,
            redistributed_residual_variance: bias_only,
            variance_recovered: 0.0,
            fully_compensable: false,
        };
    }

    // Least-squares regression coefficient: the weight bump that best explains
    // the candidate's contribution using the survivor's per-sample signal.
    let delta_weight = wc * stats.covariance / stats.survivor_variance;
    // Variance recovered = w_c² · cov² / var(a_s) = bias_only · ρ².
    let recovered = wc * wc * stats.covariance * stats.covariance / stats.survivor_variance;
    let recovered = recovered.min(bias_only).max(0.0);
    let residual = (bias_only - recovered).max(0.0);

    WeightRedistribution {
        delta_weight,
        bias_only_residual_variance: bias_only,
        redistributed_residual_variance: residual,
        variance_recovered: recovered,
        // Non-regressive once the residual per-sample variance is a negligible
        // fraction of the bias-only residual.
        fully_compensable: residual <= 1e-9 * bias_only,
    }
}

/// A surviving neuron that shares a downstream target with a remove-neuron
/// candidate (Issue #1559).
#[derive(Debug, Clone, PartialEq)]
pub struct SharedTarget {
    /// The downstream neuron both the candidate and survivor feed.
    pub target_uuid: String,
    /// The candidate's synapse weight into `target_uuid`.
    pub candidate_weight: f64,
    /// A surviving neuron that also feeds `target_uuid`.
    pub survivor_uuid: String,
}

/// Enumerate every `(target, survivor)` pair where a surviving neuron feeds the
/// same downstream target as the candidate (Issue #1559).
///
/// The candidate itself is excluded as a survivor, and only targets the
/// candidate actually feeds are considered. The returned `candidate_weight` is
/// the candidate's synapse weight into that shared target — the `w_c` that
/// [`evaluate_weight_redistribution`] redistributes.
#[must_use]
pub fn shared_downstream_targets(
    creature: &CreatureJson,
    candidate_uuid: &str,
) -> Vec<SharedTarget> {
    let mut shared = Vec::new();
    for out in creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == candidate_uuid)
    {
        let target = &out.to_uuid;
        for other in creature.synapses.iter().filter(|s| {
            &s.to_uuid == target && s.from_uuid != candidate_uuid && s.from_uuid != *target
        }) {
            shared.push(SharedTarget {
                target_uuid: target.clone(),
                candidate_weight: f64::from(out.weight),
                survivor_uuid: other.from_uuid.clone(),
            });
        }
    }
    shared
}

/// Align two neurons' per-sample activations on `obs_index` (Issue #1559).
///
/// Only observations present for **both** neurons yield a pair; samples present
/// for one neuron only are dropped rather than fabricated. The result feeds
/// [`ActivationCovariance::from_pairs`].
#[must_use]
pub fn aligned_activations(
    candidate: &[DiscoverRecord],
    survivor: &[DiscoverRecord],
) -> Vec<(f64, f64)> {
    let survivor_by_obs: HashMap<u32, f32> = survivor
        .iter()
        .map(|r| (r.obs_index, r.activation))
        .collect();
    candidate
        .iter()
        .filter_map(|r| {
            survivor_by_obs
                .get(&r.obs_index)
                .map(|&s| (f64::from(r.activation), f64::from(s)))
        })
        .collect()
}

/// Evaluate counterfactual (d) across every shared-target survivor of a
/// candidate and return the redistribution that recovers the most per-sample
/// variance (Issue #1559).
///
/// Ties the pieces together end to end: it finds each surviving neuron that
/// shares a downstream target ([`shared_downstream_targets`]), joins the
/// candidate's and survivor's per-sample activations from `records`
/// ([`aligned_activations`]), builds the compact [`ActivationCovariance`]
/// sufficient statistic, and evaluates redistribution
/// ([`evaluate_weight_redistribution`]).
///
/// # Arguments
/// * `creature` - the network topology.
/// * `candidate_uuid` - the remove-neuron candidate.
/// * `records` - per-sample discovery records for all neurons (the persisted
///   per-sample activations).
///
/// # Returns
/// `Some((target, redistribution))` for the best survivor, or `None` when the
/// candidate has no shared-target survivor with aligned samples — in which case
/// (d) cannot be evaluated and no compensation is fabricated.
#[must_use]
pub fn best_weight_redistribution(
    creature: &CreatureJson,
    candidate_uuid: &str,
    records: &[DiscoverRecord],
) -> Option<(SharedTarget, WeightRedistribution)> {
    let candidate_records: Vec<DiscoverRecord> = records
        .iter()
        .filter(|r| r.neuron_uuid == candidate_uuid)
        .cloned()
        .collect();
    if candidate_records.is_empty() {
        return None;
    }

    let mut best: Option<(SharedTarget, WeightRedistribution)> = None;
    for shared in shared_downstream_targets(creature, candidate_uuid) {
        let survivor_records: Vec<DiscoverRecord> = records
            .iter()
            .filter(|r| r.neuron_uuid == shared.survivor_uuid)
            .cloned()
            .collect();
        let pairs = aligned_activations(&candidate_records, &survivor_records);
        let Some(stats) = ActivationCovariance::from_pairs(pairs) else {
            continue;
        };
        let redist = evaluate_weight_redistribution(shared.candidate_weight, &stats);
        let better = match &best {
            Some((_, current)) => redist.variance_recovered > current.variance_recovered,
            None => true,
        };
        if better {
            best = Some((shared, redist));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_is_zero_for_constant_survivor() {
        let stats = ActivationCovariance::from_pairs(vec![(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)])
            .expect("pairs");
        assert!(stats.survivor_variance <= VARIANCE_EPSILON);
        assert_eq!(stats.correlation(), 0.0);
    }

    #[test]
    fn redistribution_never_reports_negative_recovery() {
        // Anti-correlated survivor: cov is negative but recovered variance
        // (∝ cov²) is still non-negative.
        let stats = ActivationCovariance::from_pairs(vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)])
            .expect("pairs");
        let redist = evaluate_weight_redistribution(1.0, &stats);
        assert!(redist.variance_recovered >= 0.0);
        assert!(
            redist.fully_compensable,
            "perfect anti-correlation still absorbs the signal"
        );
    }
}
