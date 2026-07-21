//! Partial-dominance safety analysis: IF conditional dominance, multi-branch
//! aggregates, and small-but-non-zero win fractions (Issue #1712, gap **G2**
//! from `docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md`, parent #1704).
//!
//! These tests drive the real [`analyse_partial_dominance`] /
//! [`safe_collapse_branches`] API with crafted observation windows and assert on
//! the returned safety verdicts. They cover the three "not so clean" shapes the
//! parent asked to catalogue:
//!
//! 1. **IF conditional dominance (F1).** A branch dominated on a degenerate
//!    condition window becomes the *only* selected branch when the condition
//!    sign flips, so it is not globally dominated. A safe IF collapse is offered
//!    **only** when the condition is provably degenerate over the window; a mixed
//!    condition makes no branch safe regardless of magnitude.
//! 2. **Multi-branch combination dominance.** A branch that never wins a 3-way
//!    MAXIMUM (dominated by the *combination* of the others) is reported safe,
//!    even though it is not pairwise-dominated by any single other branch.
//! 3. **Small-but-non-zero win fraction.** A branch that wins a few observations
//!    is classified `Partial` — a gated candidate, never reported safe.
//!
//! The safety verdict is the input to the #1623 evaluate-before-accept gate; the
//! actual collapse transform is the #1711/G1 detector this issue depends on.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::focus::{
    BranchDominance, DominanceThresholds, IfConditionRegime, RecordProvider,
    analyse_partial_dominance, safe_collapse_branches,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Fixture loading (offline, never fetched at runtime — Issue #1705). Panics on a
// missing/malformed fixture so drift fails loud (Issue #3234).
// ---------------------------------------------------------------------------

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dominated_branch_collapse")
}

fn load_network(file: &str) -> CreatureJson {
    let path = fixture_root().join("networks").join(file);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing/unreadable fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// In-memory record provider (mirrors the #1706 / contribution-propagation tests).
// ---------------------------------------------------------------------------

struct InMemoryProvider {
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl RecordProvider for InMemoryProvider {
    fn get(&self, neuron_uuid: &str) -> anyhow::Result<Option<Arc<Vec<DiscoverRecord>>>> {
        Ok(self.records.get(neuron_uuid).cloned())
    }
    fn len(&self) -> usize {
        self.records.len()
    }
}

impl InMemoryProvider {
    fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Add one neuron's recorded activations as `obs_index`-ordered records.
    /// Selection stats read the `activation` field directly.
    fn with_activations(mut self, uuid: &str, activations: &[f32]) -> Self {
        let recs: Vec<DiscoverRecord> = activations
            .iter()
            .enumerate()
            .map(|(obs, &a)| {
                let obs = u32::try_from(obs).expect("obs index fits in u32");
                DiscoverRecord::new(obs, uuid.to_string(), Some(a), a, Vec::new())
            })
            .collect();
        self.records.insert(uuid.to_string(), Arc::new(recs));
        self
    }
}

const ABS_UUID: &str = "neuron-abs";
const RELU_UUID: &str = "neuron-relu";
const COND_UUID: &str = "neuron-cond";
const IF_UUID: &str = "neuron-if";

/// Look up one branch verdict by aggregate + source neuron.
fn verdict_for<'a>(
    analysis: &'a [neat_ai_discovery::focus::AggregateDominance],
    aggregate: &str,
    from: &str,
) -> &'a neat_ai_discovery::focus::BranchVerdict {
    analysis
        .iter()
        .find(|a| a.aggregate_uuid == aggregate)
        .unwrap_or_else(|| panic!("no analysis for aggregate {aggregate}"))
        .branches
        .iter()
        .find(|b| b.from_uuid == from)
        .unwrap_or_else(|| panic!("no branch verdict for {from} → {aggregate}"))
}

// ===========================================================================
// F1 — IF conditional dominance.
// ===========================================================================

#[test]
fn if_negative_branch_safe_only_when_condition_degenerate_positive() {
    // Condition strictly positive on every observation ⇒ the negative (ABSOLUTE)
    // branch is never selected and the condition is provably degenerate. Only
    // then is the negative branch reported safe to collapse.
    let creature = load_network("if_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(COND_UUID, &vec![0.8_f32; 64]) // condition > 0 ∀ obs
        .with_activations(RELU_UUID, &linspace(0.3, 0.07, 64))
        .with_activations(ABS_UUID, &linspace(0.5, 0.10, 64));

    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let agg = analysis
        .iter()
        .find(|a| a.aggregate_uuid == IF_UUID)
        .expect("IF aggregate analysed");
    assert_eq!(
        agg.if_regime,
        Some(IfConditionRegime::AlwaysPositive),
        "condition>0 on every obs must be an AlwaysPositive (degenerate) regime"
    );

    let negative = verdict_for(&analysis, IF_UUID, ABS_UUID);
    assert_eq!(negative.dominance, BranchDominance::Dominated);
    assert!(
        negative.safe_to_collapse,
        "negative branch is safe to collapse only because the condition is degenerate-positive"
    );

    // The condition synapse itself is always active and never collapsed.
    let condition = verdict_for(&analysis, IF_UUID, COND_UUID);
    assert!(
        !condition.safe_to_collapse,
        "the condition synapse must never be reported safe to collapse"
    );
}

#[test]
fn if_mixed_condition_makes_no_branch_safe() {
    // The F1 case: the condition crosses zero across the window, so BOTH branches
    // are exercised. No branch may be reported safe — IF dominance is condition-
    // driven, not a magnitude property (test `if_dominance_is_conditional...`).
    let creature = load_network("if_aggregate.json");
    // Half the observations positive, half negative → Mixed regime.
    let cond: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 0.9 } else { -0.9 })
        .collect();
    let provider = InMemoryProvider::new()
        .with_activations(COND_UUID, &cond)
        .with_activations(RELU_UUID, &linspace(0.3, 0.07, 64))
        .with_activations(ABS_UUID, &linspace(0.5, 0.10, 64));

    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let agg = analysis
        .iter()
        .find(|a| a.aggregate_uuid == IF_UUID)
        .unwrap();
    assert_eq!(
        agg.if_regime,
        Some(IfConditionRegime::Mixed),
        "a condition that crosses zero must be a Mixed regime"
    );
    assert!(
        !agg.if_regime.unwrap().is_degenerate(),
        "Mixed regime is not degenerate"
    );

    // No branch of a Mixed-condition IF is safe to collapse.
    assert!(
        agg.branches.iter().all(|b| !b.safe_to_collapse),
        "no branch of a Mixed-condition IF may be reported safe"
    );
    let safe =
        safe_collapse_branches(&creature, &provider, &DominanceThresholds::default()).unwrap();
    assert!(
        !safe.iter().any(|b| b.aggregate_uuid == IF_UUID),
        "safe_collapse_branches must offer nothing for a Mixed-condition IF"
    );
}

#[test]
fn if_condition_sign_flip_reverses_the_dominated_branch() {
    // Flip the condition sign (all negative) and the previously-dominated
    // negative branch becomes the ONLY selected branch; now the positive branch
    // is the safe one. Proves the dominance is not global (F1).
    let creature = load_network("if_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(COND_UUID, &vec![-0.8_f32; 64]) // condition <= 0 ∀ obs
        .with_activations(RELU_UUID, &linspace(0.3, 0.07, 64))
        .with_activations(ABS_UUID, &linspace(0.5, 0.10, 64));

    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let agg = analysis
        .iter()
        .find(|a| a.aggregate_uuid == IF_UUID)
        .unwrap();
    assert_eq!(agg.if_regime, Some(IfConditionRegime::AlwaysNegative));

    let positive = verdict_for(&analysis, IF_UUID, RELU_UUID);
    let negative = verdict_for(&analysis, IF_UUID, ABS_UUID);
    assert!(
        positive.safe_to_collapse,
        "on a condition<0 window the positive branch is the dominated, safe-to-collapse one"
    );
    assert!(
        !negative.safe_to_collapse,
        "the negative branch is now the ONLY selected branch — never safe here"
    );
}

// ===========================================================================
// Multi-branch combination dominance.
// ===========================================================================

#[test]
fn multi_branch_combination_dominated_is_safe_though_not_pairwise() {
    // Three RELU branches feed one MAXIMUM. Observation window:
    //   even obs: a=10, b=0,  c=5  → max is a; c beats b but loses to a.
    //   odd  obs: a=0,  b=10, c=5  → max is b; c beats a but loses to b.
    // So neuron-c NEVER wins the 3-way MAXIMUM (combination-dominated by a∪b),
    // yet it is NOT pairwise-dominated: it outscores a on odd obs and b on even.
    let creature = load_network("multi_branch_maximum.json");
    const N: usize = 64;
    let a: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 10.0 } else { 0.0 })
        .collect();
    let b: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 0.0 } else { 10.0 })
        .collect();
    let c: Vec<f32> = vec![5.0_f32; N];

    // Sanity: c is not pairwise-dominated by a or b.
    assert!(
        (0..N).any(|i| c[i] > a[i]) && (0..N).any(|i| c[i] > b[i]),
        "fixture window must leave neuron-c non-pairwise-dominated"
    );

    let provider = InMemoryProvider::new()
        .with_activations("neuron-a", &a)
        .with_activations("neuron-b", &b)
        .with_activations("neuron-c", &c);

    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();

    let vc = verdict_for(&analysis, "neuron-maxm", "neuron-c");
    assert!(
        (vc.win_fraction).abs() < 1e-6,
        "neuron-c must win 0 observations of the 3-way MAXIMUM, got {}",
        vc.win_fraction
    );
    assert_eq!(vc.dominance, BranchDominance::Dominated);
    assert!(
        vc.safe_to_collapse,
        "combination-dominated neuron-c is safe to collapse"
    );

    // a and b each win and must never be reported safe.
    let va = verdict_for(&analysis, "neuron-maxm", "neuron-a");
    let vb = verdict_for(&analysis, "neuron-maxm", "neuron-b");
    assert!(!va.safe_to_collapse && !vb.safe_to_collapse);
    assert_eq!(va.dominance, BranchDominance::Contributing);
    assert_eq!(vb.dominance, BranchDominance::Contributing);
}

// ===========================================================================
// Small-but-non-zero win fraction — threshold policy + gate.
// ===========================================================================

#[test]
fn small_win_fraction_is_partial_not_safe() {
    // The MAXIMUM fixture: neuron-abs feeds via weight −1. Give it activations
    // that make its weighted contribution beat RELU on exactly 2 of 64 obs, so
    // its empirical win fraction is 2/64 ≈ 0.031 — small but non-zero.
    let creature = load_network("maximum_aggregate.json");
    const N: usize = 64;
    // RELU weighted contribution is +1 × relu_act. ABS weighted is −1 × abs_act,
    // so a NEGATIVE abs_act makes the ABS branch positive and able to win.
    let relu: Vec<f32> = vec![1.0_f32; N];
    let mut abs = vec![1.0_f32; N]; // −1 × 1 = −1 ⇒ loses to RELU (+1)
    abs[10] = -5.0; // −1 × −5 = +5 ⇒ beats RELU
    abs[20] = -5.0; // second win
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &abs)
        .with_activations(RELU_UUID, &relu);

    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let vabs = verdict_for(&analysis, "neuron-max", ABS_UUID);
    assert!(
        (vabs.win_fraction - 2.0 / N as f32).abs() < 1e-6,
        "ABS branch must win exactly 2/64 observations, got {}",
        vabs.win_fraction
    );
    assert_eq!(
        vabs.dominance,
        BranchDominance::Partial,
        "a small non-zero win fraction is Partial, not Dominated"
    );
    assert!(
        !vabs.safe_to_collapse,
        "a partially-dominated branch must never be auto-collapsed"
    );

    // It IS surfaced as a gated candidate (needs evaluate-before-accept).
    let agg = analysis
        .iter()
        .find(|a| a.aggregate_uuid == "neuron-max")
        .unwrap();
    assert!(
        agg.gated_candidates()
            .iter()
            .any(|b| b.from_uuid == ABS_UUID),
        "the partial branch must appear as a gated collapse candidate"
    );
    assert!(
        safe_collapse_branches(&creature, &provider, &DominanceThresholds::default())
            .unwrap()
            .is_empty(),
        "nothing is auto-safe when the only dominated-looking branch is merely Partial"
    );
}

#[test]
fn threshold_policy_reclassifies_partial_band() {
    // Same 2/64 ≈ 0.031 win fraction. With a stricter partial band that ends
    // below 0.031, the branch drops OUT of the partial band into Contributing.
    let creature = load_network("maximum_aggregate.json");
    const N: usize = 64;
    let relu: Vec<f32> = vec![1.0_f32; N];
    let mut abs = vec![1.0_f32; N];
    abs[10] = -5.0;
    abs[20] = -5.0;
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &abs)
        .with_activations(RELU_UUID, &relu);

    let strict = DominanceThresholds {
        dominated_win_fraction: 0.0,
        partial_win_fraction: 0.02, // 0.031 > 0.02 ⇒ Contributing
    };
    let analysis = analyse_partial_dominance(&creature, &provider, &strict).unwrap();
    let vabs = verdict_for(&analysis, "neuron-max", ABS_UUID);
    assert_eq!(
        vabs.dominance,
        BranchDominance::Contributing,
        "with partial band capped at 0.02, a 0.031 win fraction is Contributing"
    );
    assert!(!vabs.safe_to_collapse);
}

#[test]
fn fully_dominated_branch_is_safe_to_collapse() {
    // Baseline: the clean fully-dominated MAX fixture. ABS×(−1) ≤ 0 never beats
    // RELU ≥ 0, so its win fraction is 0 and it is safe (subject to the gate).
    let creature = load_network("maximum_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &linspace(0.5, 0.10, 64))
        .with_activations(RELU_UUID, &linspace(0.3, 0.07, 64));
    let safe =
        safe_collapse_branches(&creature, &provider, &DominanceThresholds::default()).unwrap();
    assert!(
        safe.iter().any(|b| b.from_uuid == ABS_UUID),
        "the fully-dominated ABSOLUTE branch must be reported safe to collapse"
    );
    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let vabs = verdict_for(&analysis, "neuron-max", ABS_UUID);
    assert_eq!(vabs.dominance, BranchDominance::Dominated);
}

#[test]
fn missing_evidence_is_never_reported_safe() {
    // No records at all: absence of evidence must NOT read as proven dominance
    // (fail loud, Issue #3234). Every branch defaults to Contributing/unsafe.
    let creature = load_network("maximum_aggregate.json");
    let provider = InMemoryProvider::new();
    let analysis =
        analyse_partial_dominance(&creature, &provider, &DominanceThresholds::default()).unwrap();
    let agg = analysis
        .iter()
        .find(|a| a.aggregate_uuid == "neuron-max")
        .unwrap();
    assert!(
        agg.branches
            .iter()
            .all(|b| !b.safe_to_collapse && b.dominance == BranchDominance::Contributing),
        "with no records, no branch may be Dominated or safe"
    );
}

// ---------------------------------------------------------------------------
// Deterministic activation window: base + step·i across `n` observations.
// ---------------------------------------------------------------------------
fn linspace(base: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| base + step * i as f32).collect()
}
