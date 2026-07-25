//! production-shaped MAX / MIN / IF contribution-propagation audit (Issue #1738,
//! parent #1736 — "Discovery finds very few successful candidates for the production
//! network").
//!
//! Issue #1738 is the **lead-hypothesis audit**: verify (or refute) that
//! neuron-focus selection and impact/contribution calculations are correct on
//! the production network, especially where contribution propagates through the three
//! aggregation squashes (MAXIMUM / MINIMUM / IF). The earlier dominated-branch /
//! partial-dominance fixes (#1704, #1706, #1707, #1712) covered the simple
//! two-branch shapes; this audit extends the net to the production-shaped topologies
//! those fixtures did **not** exercise:
//!
//! 1. **Multi-branch aggregate** (`agg_multi_branch_minimum.json`) — four RELU
//!    branches feed one MINIMUM. Characterises multi-branch contribution
//!    attribution and, critically, **conservation**: the per-branch impacts must
//!    equal their empirical win fractions and sum to the aggregate's impact (no
//!    branch is starved, none is inflated).
//! 2. **IF condition sign-flip** (`agg_if_condition_signflip.json`) — the
//!    condition changes sign across the window so *both* branches are selected
//!    (the F1 "mixed" case). The always-active condition synapse must carry full
//!    impact; the two branch impacts must equal their selection fractions and
//!    sum to the aggregate's impact.
//! 3. **Chained aggregates** (`agg_chained_maximum.json`) — MAXIMUM feeding
//!    MAXIMUM. Characterises contribution propagation as a product of per-hop win
//!    fractions, with conservation asserted at every hop. The fixture is shaped
//!    so the inner winner is deterministic, making the product-of-marginals
//!    exactly equal to the true joint (so the expected values are unambiguous).
//! 4. **Focus ranking / selection** — a high-impact aggregate branch, embedded
//!    among a large pool of low-impact neurons (as on the real ~1,600-neuron production
//!    creature), must land in the exploitation head of the exploit/explore
//!    selection, never be starved into the exploration tail.
//!
//! Every expected value below is computed **independently by hand** from the
//! committed observation window (documented inline), not read back from the
//! engine. All assertions PASS against the current engine: the audit's
//! conclusion is that focus/impact through aggregation squashes is **sound** on
//! production-shaped topologies. Any future regression in this math fails here in the
//! `cargo test` CI gate — the primary detection surface named in the issue.
//!
//! Fixture drift is caught at load: the loader panics on a missing/malformed
//! file with an explicit path error, so renaming or deleting a committed fixture
//! fails loudly rather than passing silently (fail loud, Issue #3234).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::focus::{
    FocusCandidate, RecordProvider, compute_impacts_public, compute_impacts_with_activations,
    compute_selection_stats, select_focus_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Fixture loading (offline, never fetched at runtime — Issue #1705).
// ---------------------------------------------------------------------------

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dominated_branch_collapse")
}

/// Load a committed aggregate-network fixture into `CreatureJson`. Panics with a
/// clear message if the fixture is missing or malformed — the drift guard named
/// in the issue's failure-detection section (fail loud, Issue #3234).
fn load_network(file: &str) -> CreatureJson {
    let path = fixture_root().join("networks").join(file);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing/unreadable production fixture {}: {e}",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Minimal in-memory record provider (mirrors the #1707 characterisation test).
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

    /// Add one neuron's activations as `obs_index`-ordered records. `value`
    /// mirrors `activation` (pre = post here); selection stats read `activation`.
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

fn tol_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Impact of `uuid` in the map, defaulting to 0.0 when absent.
fn impact_of(map: &HashMap<String, f32>, uuid: &str) -> f32 {
    map.get(uuid).copied().unwrap_or(0.0)
}

/// Win fraction for a (`from`, `to`) synapse, or panic if the stat is absent.
fn win_of(stats: &neat_ai_discovery::focus::SelectionStats, from: &str, to: &str) -> f32 {
    stats
        .get(&(from.to_string(), to.to_string()))
        .copied()
        .unwrap_or_else(|| panic!("selection stat missing for {from} -> {to}"))
}

const TOL: f32 = 1e-6;

// ===========================================================================
// 1. Multi-branch MINIMUM — attribution + conservation.
//
// Window (4 obs), all synapse weights = 1 so weighted activation == activation.
// The MINIMUM winner each obs is the smallest branch activation:
//   obs0: b1=1  b2=2  b3=3  b4=4  -> b1 wins
//   obs1: b1=5  b2=1  b3=5  b4=5  -> b2 wins
//   obs2: b1=5  b2=5  b3=1  b4=5  -> b3 wins
//   obs3: b1=1  b2=5  b3=5  b4=5  -> b1 wins
// Hand-computed win fractions: b1=2/4=0.5, b2=0.25, b3=0.25, b4=0.0 (dominated).
// The aggregate feeds a single IDENTITY output, so its impact is 1.0; each
// branch's impact == its win fraction, and the four sum back to 1.0.
// ===========================================================================

#[test]
fn multi_branch_minimum_attribution_and_conservation() {
    let creature = load_network("agg_multi_branch_minimum.json");
    let provider = InMemoryProvider::new()
        .with_activations("neuron-b1", &[1.0, 5.0, 5.0, 1.0])
        .with_activations("neuron-b2", &[2.0, 1.0, 5.0, 5.0])
        .with_activations("neuron-b3", &[3.0, 5.0, 1.0, 5.0])
        .with_activations("neuron-b4", &[4.0, 5.0, 5.0, 5.0]);

    // --- Selection stats (canary: asserts CORRECT win fractions). ---
    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    assert!(tol_eq(win_of(&stats, "neuron-b1", "neuron-minm"), 0.5, TOL));
    assert!(tol_eq(
        win_of(&stats, "neuron-b2", "neuron-minm"),
        0.25,
        TOL
    ));
    assert!(tol_eq(
        win_of(&stats, "neuron-b3", "neuron-minm"),
        0.25,
        TOL
    ));
    assert!(
        tol_eq(win_of(&stats, "neuron-b4", "neuron-minm"), 0.0, TOL),
        "b4 is combination-dominated and must win 0 observations"
    );

    // --- Impact attribution matches the hand-computed win fractions. ---
    let impacts = compute_impacts_with_activations(&creature, &provider).expect("impacts");
    assert!(
        tol_eq(impact_of(&impacts, "neuron-minm"), 1.0, TOL),
        "aggregate feeding a single identity output has impact 1.0"
    );
    let b1 = impact_of(&impacts, "neuron-b1");
    let b2 = impact_of(&impacts, "neuron-b2");
    let b3 = impact_of(&impacts, "neuron-b3");
    let b4 = impact_of(&impacts, "neuron-b4");
    assert!(
        tol_eq(b1, 0.5, TOL),
        "b1 impact must equal win fraction 0.5, got {b1}"
    );
    assert!(
        tol_eq(b2, 0.25, TOL),
        "b2 impact must equal win fraction 0.25, got {b2}"
    );
    assert!(
        tol_eq(b3, 0.25, TOL),
        "b3 impact must equal win fraction 0.25, got {b3}"
    );
    assert!(
        tol_eq(b4, 0.0, TOL),
        "dominated b4 impact must be 0, got {b4}"
    );

    // --- Conservation: branch impacts sum to the aggregate's impact. No branch
    //     is starved and none is inflated through the MINIMUM squash. ---
    assert!(
        tol_eq(b1 + b2 + b3 + b4, 1.0, TOL),
        "branch impacts must sum to the aggregate impact (conservation), got {}",
        b1 + b2 + b3 + b4
    );
}

// ===========================================================================
// 2. IF condition sign-flip — condition carries full impact, branches split.
//
// Condition neuron activations = [1, 2, -1, -2], condition synapse weight = 1.
// Summed condition contribution per obs: 1, 2, -1, -2.
//   > 0  on obs0, obs1  -> positive branch selected (2/4)
//   <= 0 on obs2, obs3  -> negative branch selected (2/4)
// So this is the F1 "mixed" regime: BOTH branches are live, neither dominated.
// Hand-computed: condition stat = 1.0, positive = 0.5, negative = 0.5.
// Impacts: if = 1.0, cond = 1.0 (always active), pos = 0.5, neg = 0.5;
// pos + neg == aggregate impact, condition is additive on top.
// ===========================================================================

#[test]
fn if_condition_sign_flip_attribution() {
    let creature = load_network("agg_if_condition_signflip.json");
    let provider = InMemoryProvider::new()
        .with_activations("neuron-cond", &[1.0, 2.0, -1.0, -2.0])
        .with_activations("neuron-pos", &[0.3, 0.4, 0.5, 0.6])
        .with_activations("neuron-neg", &[0.7, 0.8, 0.9, 1.0]);

    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    assert!(
        tol_eq(win_of(&stats, "neuron-cond", "neuron-if"), 1.0, TOL),
        "condition synapse is always active (stat 1.0)"
    );
    assert!(
        tol_eq(win_of(&stats, "neuron-pos", "neuron-if"), 0.5, TOL),
        "positive branch is selected on the condition>0 half of the window"
    );
    assert!(
        tol_eq(win_of(&stats, "neuron-neg", "neuron-if"), 0.5, TOL),
        "negative branch is selected on the condition<=0 half of the window"
    );

    let impacts = compute_impacts_with_activations(&creature, &provider).expect("impacts");
    let if_impact = impact_of(&impacts, "neuron-if");
    let cond = impact_of(&impacts, "neuron-cond");
    let pos = impact_of(&impacts, "neuron-pos");
    let neg = impact_of(&impacts, "neuron-neg");

    assert!(
        tol_eq(if_impact, 1.0, TOL),
        "IF aggregate impact must be 1.0, got {if_impact}"
    );
    assert!(
        tol_eq(cond, 1.0, TOL),
        "always-active condition synapse must carry full impact, got {cond}"
    );
    assert!(
        tol_eq(pos, 0.5, TOL),
        "positive branch impact must be 0.5, got {pos}"
    );
    assert!(
        tol_eq(neg, 0.5, TOL),
        "negative branch impact must be 0.5, got {neg}"
    );
    assert!(
        tol_eq(pos + neg, if_impact, TOL),
        "branch impacts must sum to the aggregate impact (condition is additive on top), got {}",
        pos + neg
    );
}

// ===========================================================================
// 3. Chained MAXIMUM -> MAXIMUM — product-of-marginals with conservation.
//
// Window (4 obs), all weights = 1.
//   Inner MAXIMUM (m1) over {l1, l2}:
//     l1 = [10,10, 1, 1], l2 = [0,0,0,0]  ->  l1 wins ALL 4 obs (stat 1.0),
//     l2 wins 0. m1's recorded output activation is therefore [10,10,1,1].
//   Outer MAXIMUM (m2) over {m1, l3}:
//     m1 = [10,10,1,1], l3 = [5,5,5,5]  ->  m1 wins obs0,obs1 (stat 0.5),
//     l3 wins obs2,obs3 (stat 0.5).
// Impacts (output=1, m2=1):
//   m1 = 0.5,  l3 = 0.5.
//   l1 = win(l1@m1) * impact(m1) = 1.0 * 0.5 = 0.5;  l2 = 0.0 * 0.5 = 0.0.
// Because l1's inner win is deterministic, the product-of-marginals for l1
// (1.0 * 0.5) equals the TRUE joint P(l1 wins m1 AND m1 wins m2) = 0.5 exactly.
// Conservation holds at every hop: l1+l2 == m1, and m1+l3 == m2.
// ===========================================================================

#[test]
fn chained_maximum_product_propagation_and_conservation() {
    let creature = load_network("agg_chained_maximum.json");
    let provider = InMemoryProvider::new()
        .with_activations("neuron-l1", &[10.0, 10.0, 1.0, 1.0])
        .with_activations("neuron-l2", &[0.0, 0.0, 0.0, 0.0])
        .with_activations("neuron-l3", &[5.0, 5.0, 5.0, 5.0])
        // The inner MAXIMUM's recorded output drives the outer MAXIMUM's stats.
        .with_activations("neuron-m1", &[10.0, 10.0, 1.0, 1.0]);

    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    assert!(tol_eq(win_of(&stats, "neuron-l1", "neuron-m1"), 1.0, TOL));
    assert!(tol_eq(win_of(&stats, "neuron-l2", "neuron-m1"), 0.0, TOL));
    assert!(tol_eq(win_of(&stats, "neuron-m1", "neuron-m2"), 0.5, TOL));
    assert!(tol_eq(win_of(&stats, "neuron-l3", "neuron-m2"), 0.5, TOL));

    let impacts = compute_impacts_with_activations(&creature, &provider).expect("impacts");
    let m2 = impact_of(&impacts, "neuron-m2");
    let m1 = impact_of(&impacts, "neuron-m1");
    let l1 = impact_of(&impacts, "neuron-l1");
    let l2 = impact_of(&impacts, "neuron-l2");
    let l3 = impact_of(&impacts, "neuron-l3");

    assert!(
        tol_eq(m2, 1.0, TOL),
        "outer MAXIMUM impact must be 1.0, got {m2}"
    );
    assert!(
        tol_eq(m1, 0.5, TOL),
        "inner MAXIMUM impact must be 0.5, got {m1}"
    );
    assert!(
        tol_eq(l1, 0.5, TOL),
        "l1 impact must be the product 1.0*0.5 = 0.5 (== true joint), got {l1}"
    );
    assert!(
        tol_eq(l2, 0.0, TOL),
        "dominated l2 impact must be 0, got {l2}"
    );
    assert!(tol_eq(l3, 0.5, TOL), "l3 impact must be 0.5, got {l3}");

    // Conservation at every hop of the chain.
    assert!(
        tol_eq(l1 + l2, m1, TOL),
        "inner-branch impacts must sum to the inner aggregate impact, got {}",
        l1 + l2
    );
    assert!(
        tol_eq(m1 + l3, m2, TOL),
        "outer-branch impacts must sum to the outer aggregate impact, got {}",
        m1 + l3
    );
}

// ===========================================================================
// 4. No-records fallback divergence is still present on the production multi-branch
//    shape — the 1/N flattening this audit's empirical path corrects. Pins the
//    behavioural contrast so a regression that silently drops the empirical
//    selection stats (reverting to 1/N) is caught here too.
// ===========================================================================

#[test]
fn multi_branch_fallback_flattens_to_one_over_n() {
    let creature = load_network("agg_multi_branch_minimum.json");
    // No records -> conservative 1/N split across the four branches.
    let no_acts = compute_impacts_public(&creature);
    let quarter = 0.25_f32;
    for b in ["neuron-b1", "neuron-b2", "neuron-b3", "neuron-b4"] {
        assert!(
            tol_eq(impact_of(&no_acts, b), quarter, TOL),
            "fallback must split impact 1/N = 0.25 across {b}, got {}",
            impact_of(&no_acts, b)
        );
    }
}

// ===========================================================================
// 5. Focus ranking / selection — a high-impact aggregate branch must NOT be
//    starved out of the exploit/explore selection.
//
// On the real production creature (~1,600 eligible neurons) a genuinely high-impact
// aggregate branch is a rare needle in a large low-impact haystack. We take the
// hand-verified high-impact branch b1 (impact 0.5) from fixture 1, embed it
// among 99 near-zero-impact "noise" neurons, rank strongest-first (as
// rank_focus_neurons does), and assert b1 lands in the EXPLOITATION head — never
// flattened into the exploration tail. This is the ranking-starvation guard the
// issue's failure-detection section calls for.
// ===========================================================================

#[test]
fn high_impact_aggregate_branch_is_not_starved() {
    let creature = load_network("agg_multi_branch_minimum.json");
    let provider = InMemoryProvider::new()
        .with_activations("neuron-b1", &[1.0, 5.0, 5.0, 1.0])
        .with_activations("neuron-b2", &[2.0, 1.0, 5.0, 5.0])
        .with_activations("neuron-b3", &[3.0, 5.0, 1.0, 5.0])
        .with_activations("neuron-b4", &[4.0, 5.0, 5.0, 5.0]);
    let impacts = compute_impacts_with_activations(&creature, &provider).expect("impacts");
    let b1_weight = impact_of(&impacts, "neuron-b1");
    assert!(b1_weight > 0.4, "precondition: b1 is genuinely high-impact");

    // Ranked list: the high-impact branch first, then a large low-impact tail
    // (mimicking the production needle-in-haystack). rank_focus_neurons orders
    // strongest-first; b1's impact-derived weight puts it at rank 0.
    let mut ranked = vec![FocusCandidate {
        neuron_uuid: "neuron-b1".to_string(),
        weight: b1_weight,
    }];
    for i in 0..99 {
        ranked.push(FocusCandidate {
            neuron_uuid: format!("noise-{i}"),
            weight: 0.001,
        });
    }

    let sel = select_focus_neurons(&ranked, 10, false, 0);
    // b1 is selected...
    assert!(
        sel.selected.iter().any(|u| u == "neuron-b1"),
        "high-impact branch must be selected"
    );
    // ...and specifically in the exploitation head (the first `exploitation_count`
    // slots are the ranked prefix), not merely swept up by exploration.
    let head: Vec<&String> = sel.selected.iter().take(sel.exploitation_count).collect();
    assert!(
        head.iter().any(|u| *u == "neuron-b1"),
        "high-impact branch must be EXPLOITED, not starved into the exploration tail"
    );
    assert_eq!(
        sel.selected[0], "neuron-b1",
        "the strongest branch leads the head"
    );

    // Even under drought (widened exploration) the exploitation majority keeps
    // the high-impact branch in the head.
    let drought = select_focus_neurons(&ranked, 10, true, 3);
    let drought_head: Vec<&String> = drought
        .selected
        .iter()
        .take(drought.exploitation_count)
        .collect();
    assert!(
        drought_head.iter().any(|u| *u == "neuron-b1"),
        "drought must not starve the high-impact branch out of exploitation"
    );
}
