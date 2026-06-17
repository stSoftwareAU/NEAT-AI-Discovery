//! Novelty / diversification escalation for plateaued creatures (Issue #1423).
//!
//! Follow-up to the drought-diagnostic work (#1418). The drought escape hatch
//! (#1205) clears the *memory* of past rejections, but on a genuinely
//! search-exhausted creature the next pass simply re-proposes the same losers
//! once the cache suppression ages out. This module escalates **what** is
//! proposed rather than merely forgetting what failed.
//!
//! When the search is plateaued — the rolling success rate has fallen below the
//! conservative-mode threshold **and** most of the candidate pool is currently
//! cache-suppressed — three levers engage:
//!
//! 1. **Source-type novelty bias.** Candidates are reordered to favour
//!    under-tried source types ([`rank_source_types_by_novelty`]), complementing
//!    the success-rate-driven
//!    [`source_type_boost`](super::candidate_cache::CandidateOutcomeCache::source_type_boost).
//! 2. **Operator widening.** When every standard operation on a
//!    `(source, target)` pair is suppressed, an alternative operation from the
//!    widened operator set that has **never** been tried is proposed instead
//!    ([`seed_forced_novel_candidates`]).
//! 3. **Gain-floor relaxation.** The coordinated-structural expected-gain floor
//!    is loosened ([`gain_floor_multiplier`]) so structurally-novel candidates
//!    that conservative mode would otherwise gate are allowed through.
//!
//! All three levers are **inert** when the creature is not plateaued, so a
//! healthy, steadily-accepting creature sees no change in behaviour.
//!
//! This module is pure logic operating on the existing
//! [`CandidateOutcomeCache`] and [`DiscoveryMode`] types, mirroring the
//! self-contained design of [`drought_reset`](super::drought_reset).

use std::collections::HashSet;

use super::candidate_cache::CandidateOutcomeCache;
use super::discovery_mode::DiscoveryMode;

/// Default fraction of the considered candidate pool that must be
/// cache-suppressed before escalation engages.
///
/// At `0.8`, escalation only fires once at least 80 % of the candidates the
/// search would normally propose are currently suppressed — i.e. the search is
/// genuinely re-treading old ground rather than merely having a slow pass.
pub const DEFAULT_SUPPRESSION_RATIO_THRESHOLD: f64 = 0.8;

/// Default multiplier applied to the coordinated-structural expected-gain floor
/// when escalation engages.
///
/// Values below `1.0` *relax* the floor (let more structural candidates
/// through); `0.5` halves it. This deliberately opposes conservative mode,
/// which tightens the same floor — under sustained drought, widening the search
/// is the less-bad option.
pub const DEFAULT_GAIN_FLOOR_RELAXATION: f32 = 0.5;

/// Widened operator set considered when seeding forced-novel candidates.
///
/// Mirrors the candidate types documented in `AGENTS.md`. When every standard
/// operation on a `(source, target)` pair is suppressed, the seeder proposes
/// the first operation from this set that has never been tried for that pair.
pub const WIDENED_OPERATORS: &[&str] = &[
    "addSynapse",
    "addNeuron",
    "changeSquash",
    "setBias",
    "setWeight",
    "removeSynapse",
    "removeNeuron",
    "coordinatedStructural",
];

/// A candidate identity plus the source-type label used for novelty bias.
///
/// Mirrors the `(source_uuid, target_uuid, operation)` key used by
/// [`CandidateOutcomeCache`], carrying the `source_type` string the cache uses
/// for its per-source-type statistics (e.g. `"input"`, `"hidden"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateKey {
    /// Source neuron UUID.
    pub source_uuid: String,
    /// Target neuron UUID.
    pub target_uuid: String,
    /// Operation type (e.g. `"addSynapse"`, `"coordinatedStructural"`).
    pub operation: String,
    /// Source-type label (e.g. `"input"`, `"hidden"`).
    pub source_type: String,
}

impl CandidateKey {
    /// Convenience constructor.
    #[must_use]
    pub fn new(
        source_uuid: impl Into<String>,
        target_uuid: impl Into<String>,
        operation: impl Into<String>,
        source_type: impl Into<String>,
    ) -> Self {
        Self {
            source_uuid: source_uuid.into(),
            target_uuid: target_uuid.into(),
            operation: operation.into(),
            source_type: source_type.into(),
        }
    }
}

/// Outcome of the escalation decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EscalationDecision {
    /// Whether novelty/diversification escalation should engage this pass.
    pub engaged: bool,
    /// Fraction of the considered pool that was cache-suppressed (`0.0` when
    /// nothing was considered).
    pub suppression_ratio: f64,
    /// Rolling success rate that fed the decision (echoed for diagnostics).
    pub rolling_success_rate: f32,
}

/// Decide whether escalation should engage for this pass.
///
/// Escalation engages when **all** of the following hold:
///
/// 1. At least one candidate was considered (`considered_count > 0`).
/// 2. The rolling success rate is strictly below `low_threshold` (the same
///    threshold that drives conservative mode).
/// 3. The suppressed fraction (`suppressed_count / considered_count`) is at or
///    above `suppression_ratio_threshold`.
///
/// A non-plateaued creature fails condition 2, so escalation stays inert and
/// steady-state behaviour is unchanged.
#[must_use]
pub fn decide_escalation(
    rolling_success_rate: f32,
    low_threshold: f32,
    suppressed_count: usize,
    considered_count: usize,
    suppression_ratio_threshold: f64,
) -> EscalationDecision {
    let suppression_ratio = if considered_count == 0 {
        0.0
    } else {
        // Counts are bounded by the candidate pool; saturate to u32 before the
        // exact `f64::from` conversion so the ratio never loses precision.
        let suppressed = f64::from(u32::try_from(suppressed_count).unwrap_or(u32::MAX));
        let considered = f64::from(u32::try_from(considered_count).unwrap_or(u32::MAX));
        suppressed / considered
    };

    let engaged = considered_count > 0
        && rolling_success_rate < low_threshold
        && suppression_ratio >= suppression_ratio_threshold;

    EscalationDecision {
        engaged,
        suppression_ratio,
        rolling_success_rate,
    }
}

/// Multiplier to apply to the coordinated-structural expected-gain floor.
///
/// Returns `relaxation` (clamped to `(0.0, 1.0]`) when escalation is engaged so
/// the floor is loosened, otherwise `1.0` (no change). The clamp guarantees the
/// floor is never raised by this lever and never collapses to zero.
#[must_use]
pub fn gain_floor_multiplier(engaged: bool, relaxation: f32) -> f32 {
    if !engaged {
        return 1.0;
    }
    if !relaxation.is_finite() {
        return DEFAULT_GAIN_FLOOR_RELAXATION;
    }
    relaxation.clamp(f32::MIN_POSITIVE, 1.0)
}

/// Rank source types from least-tried to most-tried (novelty-first).
///
/// Under-tried source types — those the search has barely explored — sort
/// first. Ties on attempt count are broken by the higher Bayesian success rate
/// (prefer the more promising of two equally-tried types), then alphabetically
/// for deterministic ordering. Duplicate input labels are de-duplicated.
#[must_use]
pub fn rank_source_types_by_novelty(
    cache: &CandidateOutcomeCache,
    source_types: &[&str],
) -> Vec<String> {
    let mut unique: Vec<&str> = Vec::new();
    for &st in source_types {
        if !unique.contains(&st) {
            unique.push(st);
        }
    }

    unique.sort_by(|a, b| {
        let sa = cache.source_type_stats(a);
        let sb = cache.source_type_stats(b);
        // Fewer attempts first (more novel).
        sa.attempts
            .cmp(&sb.attempts)
            // Then higher success rate first.
            .then_with(|| {
                sb.success_rate()
                    .partial_cmp(&sa.success_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            // Then alphabetical for determinism.
            .then_with(|| a.cmp(b))
    });

    unique.into_iter().map(str::to_string).collect()
}

/// Seed structurally-novel candidates not present in the failure cache.
///
/// Two phases, both ordered by source-type novelty
/// ([`rank_source_types_by_novelty`]):
///
/// 1. **Natural novelty.** Pool members that are *not* currently suppressed are
///    returned directly (the search has eligible candidates it simply ranked
///    below the suppressed ones).
/// 2. **Operator widening.** When the entire pool is suppressed, each
///    `(source, target)` pair is escalated to the first operation in
///    [`WIDENED_OPERATORS`] that has **never** been recorded in the cache. Such
///    a candidate is structurally novel and guaranteed absent from the failure
///    cache.
///
/// Within each phase the selection greedily favours distinct source types and
/// operation types, so the emitted set is maximally diverse. At most `max`
/// candidates are returned (`max == 0` yields an empty vector).
#[must_use]
pub fn seed_forced_novel_candidates(
    cache: &CandidateOutcomeCache,
    pool: &[CandidateKey],
    current_epoch: u64,
    mode: DiscoveryMode,
    drought_failures: u32,
    max: usize,
) -> Vec<CandidateKey> {
    if max == 0 || pool.is_empty() {
        return Vec::new();
    }

    let source_types: Vec<&str> = pool.iter().map(|k| k.source_type.as_str()).collect();
    let novelty_order = rank_source_types_by_novelty(cache, &source_types);
    let novelty_rank = |st: &str| {
        novelty_order
            .iter()
            .position(|n| n == st)
            .unwrap_or(usize::MAX)
    };

    // Phase 1: pool members that are not currently suppressed.
    let mut eligible: Vec<CandidateKey> = pool
        .iter()
        .filter(|k| {
            !cache.is_suppressed(
                &k.source_uuid,
                &k.target_uuid,
                &k.operation,
                current_epoch,
                mode,
                drought_failures,
            )
        })
        .cloned()
        .collect();
    eligible.sort_by(|a, b| {
        novelty_rank(&a.source_type)
            .cmp(&novelty_rank(&b.source_type))
            .then_with(|| a.operation.cmp(&b.operation))
            .then_with(|| a.source_uuid.cmp(&b.source_uuid))
            .then_with(|| a.target_uuid.cmp(&b.target_uuid))
    });
    let selected = select_diverse(eligible, max);
    if !selected.is_empty() {
        return selected;
    }

    // Phase 2: operator widening for a fully-suppressed pool.
    let mut ordered_pool: Vec<&CandidateKey> = pool.iter().collect();
    ordered_pool.sort_by(|a, b| {
        novelty_rank(&a.source_type)
            .cmp(&novelty_rank(&b.source_type))
            .then_with(|| a.source_uuid.cmp(&b.source_uuid))
            .then_with(|| a.target_uuid.cmp(&b.target_uuid))
    });

    // Spread the widened operators across pool members so the emitted set is
    // diverse in operation type, not just source type. Each pair prefers an
    // untried operator that has not already been assigned to an earlier pair;
    // only when every untried operator is taken does it reuse one.
    let mut forced: Vec<CandidateKey> = Vec::new();
    let mut assigned_ops: HashSet<&str> = HashSet::new();
    for key in &ordered_pool {
        let untried: Vec<&str> = WIDENED_OPERATORS
            .iter()
            .copied()
            .filter(|&op| {
                op != key.operation
                    && cache
                        .get_outcome(&key.source_uuid, &key.target_uuid, op)
                        .is_none()
            })
            .collect();
        // Prefer an operator not yet used by another pair; else fall back to the
        // first untried operator for this pair.
        let chosen_op = untried
            .iter()
            .find(|&&op| !assigned_ops.contains(op))
            .or_else(|| untried.first())
            .copied();
        if let Some(op) = chosen_op {
            assigned_ops.insert(op);
            forced.push(CandidateKey::new(
                &key.source_uuid,
                &key.target_uuid,
                op,
                &key.source_type,
            ));
        }
    }

    select_diverse(forced, max)
}

/// Greedily select up to `max` candidates favouring distinct source types and
/// operation types, preserving the input order for ties.
fn select_diverse(candidates: Vec<CandidateKey>, max: usize) -> Vec<CandidateKey> {
    let mut chosen: Vec<CandidateKey> = Vec::new();
    let mut seen_types: HashSet<&str> = HashSet::new();
    let mut seen_ops: HashSet<&str> = HashSet::new();

    // First pass: take candidates that introduce a new source type or operation.
    for c in &candidates {
        if chosen.len() >= max {
            break;
        }
        let new_type = !seen_types.contains(c.source_type.as_str());
        let new_op = !seen_ops.contains(c.operation.as_str());
        if new_type || new_op {
            seen_types.insert(c.source_type.as_str());
            seen_ops.insert(c.operation.as_str());
            chosen.push(c.clone());
        }
    }

    // Second pass: fill remaining slots with leftovers, preserving order.
    if chosen.len() < max {
        for c in &candidates {
            if chosen.len() >= max {
                break;
            }
            if !chosen.contains(c) {
                chosen.push(c.clone());
            }
        }
    }

    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD;

    fn key(src: &str, tgt: &str, op: &str, st: &str) -> CandidateKey {
        CandidateKey::new(src, tgt, op, st)
    }

    #[test]
    fn decide_escalation_engages_when_plateaued_and_suppressed() {
        let d = decide_escalation(
            0.1,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            9,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(d.engaged);
        assert!((d.suppression_ratio - 0.9).abs() < 1e-9);
    }

    #[test]
    fn decide_escalation_inert_when_not_plateaued() {
        // Healthy success rate, fully suppressed — must NOT engage (steady state).
        let d = decide_escalation(
            0.6,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            10,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(!d.engaged);
    }

    #[test]
    fn decide_escalation_inert_when_few_suppressed() {
        // Plateaued but most candidates still fresh — escalation not yet needed.
        let d = decide_escalation(
            0.1,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            2,
            10,
            DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
        );
        assert!(!d.engaged);
    }

    #[test]
    fn decide_escalation_inert_with_empty_pool() {
        let d = decide_escalation(0.0, 0.2, 0, 0, 0.8);
        assert!(!d.engaged);
        assert_eq!(d.suppression_ratio, 0.0);
    }

    #[test]
    fn gain_floor_multiplier_relaxes_only_when_engaged() {
        assert!((gain_floor_multiplier(false, 0.5) - 1.0).abs() < 1e-9);
        assert!((gain_floor_multiplier(true, 0.5) - 0.5).abs() < 1e-9);
        // Never raised above 1.0.
        assert!((gain_floor_multiplier(true, 4.0) - 1.0).abs() < 1e-9);
        // Never collapses to zero or below.
        assert!(gain_floor_multiplier(true, 0.0) > 0.0);
        assert!(gain_floor_multiplier(true, -1.0) > 0.0);
    }

    #[test]
    fn rank_source_types_puts_under_tried_first() {
        let mut cache = CandidateOutcomeCache::new();
        // "hidden" tried 10×, "input" tried once, "bias" never tried.
        for i in 0..10 {
            cache.record_with_source_type("h", "t", &format!("op{i}"), false, 0, "hidden");
        }
        cache.record_with_source_type("i", "t", "addSynapse", true, 0, "input");

        let ranked = rank_source_types_by_novelty(&cache, &["hidden", "input", "bias"]);
        // "bias" (0 attempts) first, then "input" (1), then "hidden" (10).
        assert_eq!(ranked, vec!["bias", "input", "hidden"]);
    }

    #[test]
    fn seed_returns_eligible_candidates_when_some_are_fresh() {
        let mut cache = CandidateOutcomeCache::new();
        // Suppress one candidate; leave another untried.
        cache.record_with_source_type("s1", "t1", "addSynapse", false, 0, "hidden");

        let pool = vec![
            key("s1", "t1", "addSynapse", "hidden"),
            key("s2", "t2", "addSynapse", "input"),
        ];
        let seeded =
            seed_forced_novel_candidates(&cache, &pool, 1, DiscoveryMode::Conservative, 5, 4);
        // The fresh candidate is eligible; the suppressed one is excluded.
        assert!(seeded.iter().any(|k| k.source_uuid == "s2"));
        assert!(seeded.iter().all(|k| k.source_uuid != "s1"));
    }

    #[test]
    fn seed_widens_operator_when_pool_fully_suppressed() {
        let mut cache = CandidateOutcomeCache::new();
        // Suppress the standard op on the only pool member.
        cache.record_with_source_type("s1", "t1", "addSynapse", false, 0, "hidden");

        let pool = vec![key("s1", "t1", "addSynapse", "hidden")];
        let seeded =
            seed_forced_novel_candidates(&cache, &pool, 1, DiscoveryMode::Conservative, 5, 4);
        assert!(!seeded.is_empty(), "must emit a forced-novel candidate");
        // Every emitted candidate is absent from the failure cache.
        for k in &seeded {
            assert!(k.operation != "addSynapse");
            assert!(
                cache
                    .get_outcome(&k.source_uuid, &k.target_uuid, &k.operation)
                    .is_none(),
                "forced candidate must not be in the failure cache"
            );
        }
    }

    #[test]
    fn seed_respects_max_and_empty_inputs() {
        let cache = CandidateOutcomeCache::new();
        let pool = vec![key("s1", "t1", "addSynapse", "input")];
        assert!(
            seed_forced_novel_candidates(&cache, &pool, 0, DiscoveryMode::Normal, 0, 0).is_empty()
        );
        assert!(
            seed_forced_novel_candidates(&cache, &[], 0, DiscoveryMode::Normal, 0, 4).is_empty()
        );
        let one = seed_forced_novel_candidates(&cache, &pool, 0, DiscoveryMode::Normal, 0, 1);
        assert_eq!(one.len(), 1);
    }
}
