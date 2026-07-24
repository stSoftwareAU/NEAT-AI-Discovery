//! Characterisation tests: contribution propagation through MAX / MIN / IF
//! (Issue #1707, parent #1704).
//!
//! These are **characterisation** tests, not an implementation. They pin the
//! *current* engine behaviour for how a candidate's predicted contribution is
//! propagated through the three selection aggregates, over the committed
//! hand-authored fixtures under `tests/fixtures/dominated_branch_collapse/`
//! (Issue #1705, re-based by Issue #1722; offline — never fetched at runtime).
//! They cover the three propagation paths the parent investigation names:
//!
//! 1. **Error-walk attribution** — how impact/error is attributed back through a
//!    MAX / MIN / IF node to its branches (incl. the IF condition synapse).
//!    Characterised via [`compute_impacts_with_activations`] (empirical selection
//!    stats) versus [`compute_impacts_public`] (the no-records 1/N fallback). The
//!    **divergence** hunted here (per the issue) is that *without* activation
//!    records the walk splits impact 1/N across every branch — crediting a
//!    provably dominated branch a full 1/N share and under-crediting the
//!    always-active IF condition synapse.
//!
//! 2. **Win-fraction selection stats** — [`compute_selection_stats`], the
//!    empirical per-branch win probabilities. Per NEAT-AI-Explore#513 this path
//!    is **correct**, so [`win_fraction_selection_stats_match_fixtures`] asserts
//!    *correct* behaviour and acts as a canary: any future failure there is a
//!    true regression, not a characterised defect.
//!
//! 3. **Candidate scoring** — predicted `expectedErrorReduction` versus recorded
//!    `actualErrorReduction`, driven straight from the committed candidate-cache
//!    fixtures. Characterises the `change-squash` SELU→ABSOLUTE misprediction
//!    (predicted **+3.0e-10**, outcome **−6.0e-4**) and the `d1ac1f41`
//!    1-success / 5-failure aggregate divergence rate.
//!
//! Fixture drift is caught at load: the loaders panic on a missing or malformed
//! file with an explicit fixture-path error, so renaming or deleting a committed
//! fixture fails these tests loudly rather than passing silently (Issue #3234).
//!
//! Divergences are **characterised, not fixed** — each is captured here for the
//! extent-report sub-issue under #1704.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CalibrationCorrection, FailureCacheEntry, MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION,
};
use neat_ai_discovery::focus::{
    RecordProvider, compute_impacts_public, compute_impacts_with_activations,
    compute_selection_stats,
};
use neat_ai_discovery::types::DiscoverRecord;
use serde::Deserialize;
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
            "missing/unreadable dominated-branch fixture {}: {e}",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

/// Read a committed candidate-cache fixture as raw JSON text. Panics (fail loud)
/// on a missing/unreadable file so path-3 cannot silently pass on absent ground
/// truth.
fn read_candidate_cache(file: &str) -> String {
    let path = fixture_root().join("candidate_cache").join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing/unreadable candidate-cache fixture {}: {e}",
            path.display()
        )
    })
}

// ---------------------------------------------------------------------------
// Minimal in-memory record provider (mirrors tests/focus/issue_1370).
// ---------------------------------------------------------------------------

/// In-memory `RecordProvider` keyed by neuron UUID.
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

    /// Add one neuron's activations as `obs_index`-ordered records. The `value`
    /// field mirrors `activation` (pre = post here); selection stats read
    /// `activation`.
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
const OBS: usize = 64;

/// Strictly-positive branch activations across `OBS` observations. `ABSOLUTE`
/// and `RELU` both emit `>= 0`, so this faithfully stands in for their recorded
/// output activation. The two series differ so no accidental ties occur.
fn positive_series(base: f32, step: f32) -> Vec<f32> {
    (0..OBS).map(|i| base + step * i as f32).collect()
}

fn tol_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// ===========================================================================
// Path 2 (canary — asserts CORRECT behaviour, NEAT-AI-Explore#513).
// compute_selection_stats win fractions match the fixtures' analytical
// dominance. A failure here is a true regression, not a characterised defect.
// ===========================================================================

#[test]
fn win_fraction_selection_stats_match_fixtures() {
    // --- MAXIMUM: ABSOLUTE×(−1) branch (≤ 0) never wins; RELU (≥ 0) always wins.
    let creature = load_network("maximum_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10))
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07));
    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    let abs_win = stats
        .get(&(ABS_UUID.to_string(), "neuron-max".to_string()))
        .copied()
        .expect("abs→max win fraction present");
    let relu_win = stats
        .get(&(RELU_UUID.to_string(), "neuron-max".to_string()))
        .copied()
        .expect("relu→max win fraction present");
    assert!(
        tol_eq(abs_win, 0.0, 1e-6),
        "MAX: dominated ABSOLUTE branch must win 0.0 of observations, got {abs_win}"
    );
    assert!(
        tol_eq(relu_win, 1.0, 1e-6),
        "MAX: RELU branch must win 1.0 of observations, got {relu_win}"
    );

    // --- MINIMUM: mirror — RELU (≥ 0) never wins; ABSOLUTE×(−1) (≤ 0) always wins.
    let creature = load_network("minimum_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10))
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07));
    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    let abs_win = stats
        .get(&(ABS_UUID.to_string(), "neuron-min".to_string()))
        .copied()
        .expect("abs→min win fraction present");
    let relu_win = stats
        .get(&(RELU_UUID.to_string(), "neuron-min".to_string()))
        .copied()
        .expect("relu→min win fraction present");
    assert!(
        tol_eq(relu_win, 0.0, 1e-6),
        "MIN: dominated RELU branch must win 0.0 of observations, got {relu_win}"
    );
    assert!(
        tol_eq(abs_win, 1.0, 1e-6),
        "MIN: ABSOLUTE branch must win 1.0 of observations, got {abs_win}"
    );

    // --- IF: condition always active (prob 1.0); on a condition>0 window the
    // positive (RELU) branch takes every observation and the negative (ABSOLUTE)
    // branch is never selected.
    let creature = load_network("if_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(COND_UUID, &positive_series(0.4, 0.01)) // condition > 0 ∀ obs
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07))
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10));
    let stats = compute_selection_stats(&creature, &provider).expect("selection stats");
    let cond = stats
        .get(&(COND_UUID.to_string(), "neuron-if".to_string()))
        .copied()
        .expect("condition synapse stat present");
    let pos = stats
        .get(&(RELU_UUID.to_string(), "neuron-if".to_string()))
        .copied()
        .expect("positive branch stat present");
    let neg = stats
        .get(&(ABS_UUID.to_string(), "neuron-if".to_string()))
        .copied()
        .expect("negative branch stat present");
    assert!(
        tol_eq(cond, 1.0, 1e-6),
        "IF: condition synapse is always active (1.0), got {cond}"
    );
    assert!(
        tol_eq(pos, 1.0, 1e-6),
        "IF: positive branch selected every obs on a condition>0 window, got {pos}"
    );
    assert!(
        tol_eq(neg, 0.0, 1e-6),
        "IF: dominated negative branch never selected on a condition>0 window, got {neg}"
    );
}

// ===========================================================================
// Path 1 — error-walk attribution through the aggregate to its branches.
// Characterises the divergence between the empirical walk (with activation
// records) and the no-records 1/N fallback.
// ===========================================================================

/// Impact of `uuid` in the map, defaulting to 0.0 when absent.
fn impact_of(map: &HashMap<String, f32>, uuid: &str) -> f32 {
    map.get(uuid).copied().unwrap_or(0.0)
}

#[test]
fn error_walk_attribution_through_max_min_if() {
    // ---- MAXIMUM ----------------------------------------------------------
    let creature = load_network("maximum_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10))
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07));

    let with_acts =
        compute_impacts_with_activations(&creature, &provider).expect("impacts w/ acts");
    let no_acts = compute_impacts_public(&creature); // None records → 1/N fallback

    // The aggregate feeds a single IDENTITY output, so its own impact is 1.0 in
    // both modes (independent of selection stats).
    assert!(
        tol_eq(impact_of(&with_acts, "neuron-max"), 1.0, 1e-5),
        "MAX aggregate impact should be 1.0"
    );

    // Empirical walk: the dominated ABSOLUTE branch (wins 0) is attributed ~0
    // impact; the RELU branch (wins all) carries the full downstream impact.
    assert!(
        tol_eq(impact_of(&with_acts, ABS_UUID), 0.0, 1e-6),
        "MAX/empirical: dominated ABSOLUTE branch must get ~0 impact, got {}",
        impact_of(&with_acts, ABS_UUID)
    );
    assert!(
        tol_eq(impact_of(&with_acts, RELU_UUID), 1.0, 1e-5),
        "MAX/empirical: RELU branch must carry full impact, got {}",
        impact_of(&with_acts, RELU_UUID)
    );

    // DIVERGENCE (current vs correct): without activation records the walk
    // splits the aggregate's impact 1/N = 0.5 across BOTH branches, crediting the
    // provably-dominated ABSOLUTE branch a full half share it never earns.
    assert!(
        tol_eq(impact_of(&no_acts, ABS_UUID), 0.5, 1e-5),
        "MAX/fallback divergence: dominated ABSOLUTE branch is mis-credited 1/N=0.5, got {}",
        impact_of(&no_acts, ABS_UUID)
    );
    assert!(
        tol_eq(impact_of(&no_acts, RELU_UUID), 0.5, 1e-5),
        "MAX/fallback: RELU branch is under-credited to 1/N=0.5, got {}",
        impact_of(&no_acts, RELU_UUID)
    );

    // ---- MINIMUM (mirror: RELU is the dominated branch) -------------------
    let creature = load_network("minimum_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10))
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07));
    let with_acts =
        compute_impacts_with_activations(&creature, &provider).expect("impacts w/ acts");
    let no_acts = compute_impacts_public(&creature);

    assert!(
        tol_eq(impact_of(&with_acts, RELU_UUID), 0.0, 1e-6),
        "MIN/empirical: dominated RELU branch must get ~0 impact, got {}",
        impact_of(&with_acts, RELU_UUID)
    );
    assert!(
        tol_eq(impact_of(&with_acts, ABS_UUID), 1.0, 1e-5),
        "MIN/empirical: ABSOLUTE branch must carry full impact, got {}",
        impact_of(&with_acts, ABS_UUID)
    );
    assert!(
        tol_eq(impact_of(&no_acts, RELU_UUID), 0.5, 1e-5),
        "MIN/fallback divergence: dominated RELU branch is mis-credited 1/N=0.5, got {}",
        impact_of(&no_acts, RELU_UUID)
    );

    // ---- IF (condition synapse + positive/negative branches) --------------
    let creature = load_network("if_aggregate.json");
    let provider = InMemoryProvider::new()
        .with_activations(COND_UUID, &positive_series(0.4, 0.01))
        .with_activations(RELU_UUID, &positive_series(0.3, 0.07))
        .with_activations(ABS_UUID, &positive_series(0.5, 0.10));
    let with_acts =
        compute_impacts_with_activations(&creature, &provider).expect("impacts w/ acts");
    let no_acts = compute_impacts_public(&creature);

    // Empirical walk: the always-active condition synapse carries the full
    // downstream impact; the positive branch (selected every obs) also carries
    // it; the dominated negative branch gets ~0.
    assert!(
        tol_eq(impact_of(&with_acts, COND_UUID), 1.0, 1e-5),
        "IF/empirical: condition synapse must carry full impact (always active), got {}",
        impact_of(&with_acts, COND_UUID)
    );
    assert!(
        tol_eq(impact_of(&with_acts, RELU_UUID), 1.0, 1e-5),
        "IF/empirical: positive branch selected every obs must carry full impact, got {}",
        impact_of(&with_acts, RELU_UUID)
    );
    assert!(
        tol_eq(impact_of(&with_acts, ABS_UUID), 0.0, 1e-6),
        "IF/empirical: dominated negative branch must get ~0 impact, got {}",
        impact_of(&with_acts, ABS_UUID)
    );

    // DIVERGENCE: without records the IF walk splits impact 1/N = 1/3 uniformly
    // across the three inbound synapses — badly under-crediting the always-active
    // condition (1/3 vs 1.0) and over-crediting the never-selected negative
    // branch (1/3 vs 0.0). This is the aggregate-contribution breakdown the
    // parent investigation targets, captured for the extent report (#1704).
    let third = 1.0_f32 / 3.0;
    assert!(
        tol_eq(impact_of(&no_acts, COND_UUID), third, 1e-5),
        "IF/fallback divergence: condition synapse under-credited to 1/3, got {}",
        impact_of(&no_acts, COND_UUID)
    );
    assert!(
        tol_eq(impact_of(&no_acts, ABS_UUID), third, 1e-5),
        "IF/fallback divergence: dominated negative branch over-credited to 1/3, got {}",
        impact_of(&no_acts, ABS_UUID)
    );
    assert!(
        tol_eq(impact_of(&no_acts, RELU_UUID), third, 1e-5),
        "IF/fallback: positive branch flattened to 1/3, got {}",
        impact_of(&no_acts, RELU_UUID)
    );
}

// ===========================================================================
// Path 3 — candidate scoring: predicted expectedErrorReduction vs measured
// actualErrorReduction, straight from the committed candidate-cache fixtures.
// ===========================================================================

/// The `d1ac1f41` cache-directory shape: 1 success versus 5 failures.
#[derive(Deserialize)]
struct D1ac1f41Cache {
    successes: Vec<FailureCacheEntry>,
    failures: Vec<FailureCacheEntry>,
}

/// A predicted-vs-measured sign-flip misprediction: predicted an improvement
/// (`expected > 0`) yet measured harm (`actual < 0`).
fn is_sign_flip_misprediction(e: &FailureCacheEntry) -> bool {
    e.expected_error_reduction > 0.0 && e.actual_error_reduction < 0.0
}

#[test]
fn expected_vs_actual_error_reduction_divergence() {
    // ---- The concrete change-squash SELU→ABSOLUTE misprediction ----------
    let raw = read_candidate_cache("v2_change-squash_selu-to-absolute.json");
    let entry: FailureCacheEntry =
        serde_json::from_str(&raw).expect("change-squash fixture parses as a failure-cache entry");

    assert_eq!(entry.change_type, "change-squash");
    // Predicted +3.0e-10 (essentially "no change"), outcome −6.0e-4 (real harm).
    let expected = f64::from(entry.expected_error_reduction);
    let actual = f64::from(entry.actual_error_reduction);
    assert!(
        (expected / 3.0e-10 - 1.0).abs() < 1e-3,
        "predicted expectedErrorReduction must be ~+3.0e-10, got {expected:e}"
    );
    assert!(
        (actual / -6.0e-4 - 1.0).abs() < 1e-3,
        "recorded actualErrorReduction must be ~−6.0e-4, got {actual:e}"
    );

    // The misprediction is qualitative (sign flip) AND quantitative: the
    // recorded harm is > 1e5× the predicted-negligible magnitude.
    assert!(
        is_sign_flip_misprediction(&entry),
        "change-squash: predicted improvement but recorded harm (sign flip)"
    );
    assert!(
        actual.abs() / expected.abs() > 1e5,
        "change-squash: recorded/predicted magnitude gap must exceed 1e5, got {:e}",
        actual.abs() / expected.abs()
    );

    // Feeding this single wildly-optimistic outcome back through the calibration
    // collapses the change-squash correction to the floor: the enormous negative
    // ratio clamps to MIN_CALIBRATION_CORRECTION.
    let correction = CalibrationCorrection::from_failure_cache(std::slice::from_ref(&entry));
    assert!(
        tol_eq(
            correction.get_correction("change-squash"),
            MIN_CALIBRATION_CORRECTION,
            1e-9
        ),
        "change-squash correction must clamp to the floor {MIN_CALIBRATION_CORRECTION}, got {}",
        correction.get_correction("change-squash")
    );

    // ---- The d1ac1f41 1-success / 5-failure aggregate divergence rate -----
    let raw = read_candidate_cache("d1ac1f41.json");
    let cache: D1ac1f41Cache =
        serde_json::from_str(&raw).expect("d1ac1f41 fixture parses (successes + failures)");

    assert_eq!(cache.successes.len(), 1, "d1ac1f41: exactly 1 success");
    assert_eq!(cache.failures.len(), 5, "d1ac1f41: exactly 5 failures");

    // Change-type mix of the failures: 1 change-squash + 4 remove-neuron.
    let change_squash = cache
        .failures
        .iter()
        .filter(|e| e.change_type == "change-squash")
        .count();
    let remove_neuron_failures = cache
        .failures
        .iter()
        .filter(|e| e.change_type == "remove-neuron")
        .count();
    assert_eq!(change_squash, 1, "d1ac1f41: 1 change-squash failure");
    assert_eq!(
        remove_neuron_failures, 4,
        "d1ac1f41: 4 remove-neuron failures"
    );

    // Aggregate expected-vs-actual divergence rate: every one of the 5 failures
    // predicted a positive error reduction yet measured harm, while the single
    // success predicted and measured an improvement. So 5 of 6 entries are
    // sign-flip mispredictions — an 83.3% divergence rate straight from the
    // committed ground truth.
    let all: Vec<&FailureCacheEntry> = cache
        .successes
        .iter()
        .chain(cache.failures.iter())
        .collect();
    let sign_flips = all.iter().filter(|e| is_sign_flip_misprediction(e)).count();
    assert_eq!(
        sign_flips, 5,
        "d1ac1f41: 5 of 6 outcomes are sign-flip mispredictions"
    );
    assert!(
        cache
            .successes
            .iter()
            .all(|e| e.expected_error_reduction > 0.0 && e.actual_error_reduction > 0.0),
        "d1ac1f41: the single success is the only non-divergent outcome"
    );
    let divergence_rate = sign_flips as f32 / all.len() as f32;
    assert!(
        tol_eq(divergence_rate, 5.0 / 6.0, 1e-6),
        "d1ac1f41: divergence rate must be 5/6, got {divergence_rate}"
    );

    // Feeding the whole set back through the calibration discounts the
    // remove-neuron correction below neutral (recent outcomes are negative) yet
    // not fully to the floor — characterising the current, non-collapsing
    // calibration response to a mostly-failing group.
    let cache_vec: Vec<FailureCacheEntry> =
        cache.successes.into_iter().chain(cache.failures).collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache_vec);
    let rn = correction.get_correction("remove-neuron");
    assert!(
        rn > MIN_CALIBRATION_CORRECTION && rn < NEUTRAL_CORRECTION,
        "d1ac1f41: remove-neuron correction must be discounted below neutral but above the floor, got {rn}"
    );
}
