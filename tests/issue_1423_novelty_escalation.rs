//! Issue #1423 — novelty / diversification escalation for plateaued creatures.
//!
//! Acceptance criteria from the issue, expressed as behaviour tests against the
//! real [`CandidateOutcomeCache`] and the escalation logic:
//!
//! 1. On a synthetic plateau where every standard candidate is cache-suppressed,
//!    the search emits at least one structurally-novel candidate **not** present
//!    in the failure cache.
//! 2. Measurable increase in candidate diversity (distinct source types /
//!    operation types) under sustained drought.
//! 3. No regression in steady-state acceptance rate on a non-plateaued creature
//!    (escalation stays inert when the rolling success rate is healthy).

use std::collections::HashSet;

use neat_ai_discovery::analysis::candidate_cache::CandidateOutcomeCache;
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_LOW_SUCCESS_RATE_THRESHOLD, DiscoveryMode,
};
use neat_ai_discovery::analysis::novelty_escalation::{
    CandidateKey, DEFAULT_GAIN_FLOOR_RELAXATION, DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    decide_escalation, gain_floor_multiplier, seed_forced_novel_candidates,
};

/// Build a pool mirroring the plateaued-creature scenario: several
/// `(source, target)` pairs, all on the same standard operation, spanning two
/// source types.
fn plateau_pool() -> Vec<CandidateKey> {
    vec![
        CandidateKey::new("h1", "out", "changeSquash", "hidden"),
        CandidateKey::new("h2", "out", "changeSquash", "hidden"),
        CandidateKey::new("i1", "out", "changeSquash", "input"),
        CandidateKey::new("i2", "out", "changeSquash", "input"),
    ]
}

/// Suppress every standard candidate in the pool at `epoch`.
fn suppress_all(cache: &mut CandidateOutcomeCache, pool: &[CandidateKey], epoch: u64) {
    for k in pool {
        cache.record_with_source_type(
            &k.source_uuid,
            &k.target_uuid,
            &k.operation,
            false,
            epoch,
            &k.source_type,
        );
    }
}

/// AC1: a fully-suppressed pool still yields a structurally-novel candidate
/// absent from the failure cache.
#[test]
fn ac1_emits_novel_candidate_not_in_failure_cache() {
    let pool = plateau_pool();
    let mut cache = CandidateOutcomeCache::new();
    suppress_all(&mut cache, &pool, 0);

    // Sanity: with the staleness window, every standard candidate is suppressed.
    let suppressed = pool
        .iter()
        .filter(|k| {
            cache.is_suppressed(
                &k.source_uuid,
                &k.target_uuid,
                &k.operation,
                1,
                DiscoveryMode::Conservative,
                10,
            )
        })
        .count();
    assert_eq!(suppressed, pool.len(), "scenario must be fully suppressed");

    let decision = decide_escalation(
        0.0,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        suppressed,
        pool.len(),
        DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    );
    assert!(decision.engaged, "escalation must engage on a full plateau");

    let seeded = seed_forced_novel_candidates(&cache, &pool, 1, DiscoveryMode::Conservative, 10, 4);
    assert!(
        !seeded.is_empty(),
        "must emit at least one structurally-novel candidate"
    );
    for k in &seeded {
        assert!(
            cache
                .get_outcome(&k.source_uuid, &k.target_uuid, &k.operation)
                .is_none(),
            "novel candidate {k:?} must not be present in the failure cache"
        );
    }
}

/// AC2: the escalated candidate set is more diverse (distinct source types and
/// operation types) than the suppressed standard pool's repeated proposals.
#[test]
fn ac2_increases_candidate_diversity_under_drought() {
    let pool = plateau_pool();
    let mut cache = CandidateOutcomeCache::new();
    suppress_all(&mut cache, &pool, 0);

    // Baseline: every suppressed candidate uses the single standard operation.
    let baseline_ops: HashSet<&str> = pool.iter().map(|k| k.operation.as_str()).collect();
    assert_eq!(baseline_ops.len(), 1, "baseline collapses to one operation");

    let seeded = seed_forced_novel_candidates(&cache, &pool, 1, DiscoveryMode::Conservative, 10, 4);

    let novel_ops: HashSet<&str> = seeded.iter().map(|k| k.operation.as_str()).collect();
    let novel_types: HashSet<&str> = seeded.iter().map(|k| k.source_type.as_str()).collect();

    assert!(
        novel_ops.len() > baseline_ops.len(),
        "escalation must broaden operation diversity: {novel_ops:?}"
    );
    assert!(
        novel_types.len() >= 2,
        "escalation must span multiple source types: {novel_types:?}"
    );
}

/// AC3: on a non-plateaued creature (healthy rolling success rate) escalation
/// stays inert, even when candidates happen to be suppressed — steady-state
/// behaviour is unchanged.
#[test]
fn ac3_no_escalation_on_non_plateaued_creature() {
    let pool = plateau_pool();
    let mut cache = CandidateOutcomeCache::new();
    suppress_all(&mut cache, &pool, 0);

    // Healthy success rate above the threshold.
    let decision = decide_escalation(
        0.6,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        pool.len(),
        pool.len(),
        DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    );
    assert!(
        !decision.engaged,
        "escalation must stay inert on a non-plateaued creature"
    );

    // The gain floor is unchanged (multiplier 1.0) when escalation is inert.
    let multiplier = gain_floor_multiplier(decision.engaged, DEFAULT_GAIN_FLOOR_RELAXATION);
    assert!((multiplier - 1.0).abs() < 1e-9);
}

/// The gain floor is relaxed (multiplier < 1.0) only when escalation engages.
#[test]
fn gain_floor_relaxed_only_under_escalation() {
    let engaged = decide_escalation(0.0, 0.2, 10, 10, 0.8);
    assert!(engaged.engaged);
    let m = gain_floor_multiplier(engaged.engaged, DEFAULT_GAIN_FLOOR_RELAXATION);
    assert!(
        m < 1.0 && m > 0.0,
        "floor must be loosened, not zeroed: {m}"
    );
}
