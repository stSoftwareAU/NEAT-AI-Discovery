//! Weight-redistribution compensation for remove-neuron candidates
//! (Issue #1559).
//!
//! Follow-up from the #1558 counterfactual study. NEAT-AI's mean-preserving
//! bias compensation only cancels the *mean* of a removed neuron's downstream
//! contribution; the residual cost is its genuine **per-sample (variance)**
//! signal. Counterfactual (d) — folding that per-sample contribution into a
//! correlated survivor's downstream weight — is the sole lever whose ceiling
//! reaches a neutral (non-regressive) removal.
//!
//! To evaluate (d) we need the per-sample distribution of the candidate's
//! activation and its cross-neuron correlation with survivors. Persisting full
//! per-sample vectors for every candidate is prohibitive, so we persist a
//! compact **sufficient statistic**: the per-pair mean/variance/covariance of
//! the candidate against each surviving neuron that shares a downstream target.
//! These tests exercise that statistic and the redistribution evaluator on
//! small synthetic samples where the outcome is easy to reason about.

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{
    ActivationCovariance, aligned_activations, best_weight_redistribution,
    evaluate_weight_redistribution, shared_downstream_targets,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Build a `CreatureJson` from a compact JSON description.
fn creature(json: &str) -> CreatureJson {
    serde_json::from_str(json).expect("valid creature JSON")
}

/// Build a per-sample record for `neuron` at observation `obs` with `activation`.
fn rec(obs: u32, neuron: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(
        obs,
        neuron.to_string(),
        Some(activation),
        activation,
        Vec::new(),
    )
}

/// The covariance sufficient statistic is accumulated correctly from aligned
/// per-sample activation pairs (population mean/variance/covariance).
#[test]
fn covariance_statistic_matches_hand_computed_values() {
    // candidate = [1, 2, 3, 4]; survivor = [2, 4, 6, 8] = 2 * candidate.
    let pairs = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];
    let stats = ActivationCovariance::from_pairs(pairs).expect("non-empty pairs");

    assert_eq!(stats.count, 4);
    assert!((stats.candidate_mean - 2.5).abs() < 1e-12);
    assert!((stats.survivor_mean - 5.0).abs() < 1e-12);
    // population variance of [1,2,3,4] = 1.25; survivor = 4 * 1.25 = 5.0.
    assert!((stats.candidate_variance - 1.25).abs() < 1e-12);
    assert!((stats.survivor_variance - 5.0).abs() < 1e-12);
    // covariance = 2 * var(candidate) = 2.5.
    assert!((stats.covariance - 2.5).abs() < 1e-12);
    // Perfectly (linearly) correlated → correlation = 1.
    assert!((stats.correlation() - 1.0).abs() < 1e-9);
}

/// An empty pair stream yields no statistic (nothing to persist).
#[test]
fn covariance_statistic_is_none_for_empty_input() {
    let empty: Vec<(f64, f64)> = Vec::new();
    assert!(ActivationCovariance::from_pairs(empty).is_none());
}

/// Records for two neurons are joined on `obs_index`; samples present for only
/// one neuron are dropped (no fabricated pairs).
#[test]
fn aligned_activations_joins_on_obs_index() {
    let candidate = vec![rec(0, "c", 1.0), rec(1, "c", 2.0), rec(2, "c", 3.0)];
    // Survivor missing obs 2, and has an extra obs 5 the candidate lacks.
    let survivor = vec![rec(0, "s", 10.0), rec(1, "s", 20.0), rec(5, "s", 99.0)];

    let mut pairs = aligned_activations(&candidate, &survivor);
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    assert_eq!(pairs, vec![(1.0, 10.0), (2.0, 20.0)]);
}

/// Counterfactual (d) with a perfectly correlated survivor drives the residual
/// per-sample variance to ~0 — the removal becomes fully compensable
/// (non-regressive), which the mean-only bias lever could never achieve.
#[test]
fn perfectly_correlated_survivor_is_fully_compensable() {
    // Candidate activation equals the survivor's each sample (correlation 1).
    let pairs = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (0.5, 0.5)];
    let stats = ActivationCovariance::from_pairs(pairs).expect("pairs");

    // Candidate feeds the shared target with weight 0.8.
    let redist = evaluate_weight_redistribution(0.8, &stats);

    // Bias-only leaves the full per-sample variance as residual cost.
    assert!(redist.bias_only_residual_variance > 0.0);
    // Redistribution recovers essentially all of it.
    assert!(redist.redistributed_residual_variance < 1e-9);
    assert!(redist.fully_compensable);
    // Optimal survivor weight bump equals the candidate's weight (identical
    // signal): Δw = w_c * cov/var = 0.8 * 1 = 0.8.
    assert!((redist.delta_weight - 0.8).abs() < 1e-9);
    assert!((redist.variance_recovered - redist.bias_only_residual_variance).abs() < 1e-9);
}

/// An uncorrelated survivor recovers nothing: redistribution cannot beat the
/// bias-only residual, so the removal stays regressive.
#[test]
fn uncorrelated_survivor_recovers_nothing() {
    // candidate = [1, -1, 1, -1]; survivor = [1, 1, -1, -1] → covariance 0.
    let pairs = vec![(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
    let stats = ActivationCovariance::from_pairs(pairs).expect("pairs");
    assert!(stats.covariance.abs() < 1e-12);

    let redist = evaluate_weight_redistribution(1.0, &stats);

    assert!((redist.delta_weight).abs() < 1e-12);
    assert!(redist.variance_recovered.abs() < 1e-12);
    assert!(
        (redist.redistributed_residual_variance - redist.bias_only_residual_variance).abs() < 1e-12
    );
    assert!(!redist.fully_compensable);
}

/// A partially correlated survivor recovers exactly the shared-variance
/// fraction (rho^2) and leaves the rest as residual.
#[test]
fn partial_correlation_recovers_rho_squared_fraction() {
    // candidate = [2, -2, 2, -2]; survivor = [2, -2, -2, 2].
    // var_c = 4, var_s = 4, cov = (4 + 4 - 4 - 4)/4 = 0 ... choose a genuine
    // partial correlation instead:
    // candidate = [1, 2, 3, 4]; survivor = [1, 2, 3, 0].
    let c = [1.0, 2.0, 3.0, 4.0];
    let s = [1.0, 2.0, 3.0, 0.0];
    let pairs: Vec<(f64, f64)> = c.iter().copied().zip(s.iter().copied()).collect();
    let stats = ActivationCovariance::from_pairs(pairs).expect("pairs");

    let wc = 1.0;
    let redist = evaluate_weight_redistribution(wc, &stats);
    let rho = stats.correlation();

    // 0 < rho^2 < 1 for genuine partial correlation.
    assert!(rho.abs() > 0.0 && rho.abs() < 1.0, "rho = {rho}");
    // Recovered fraction = rho^2 of the bias-only residual.
    let expected_recovered = rho * rho * redist.bias_only_residual_variance;
    assert!(
        (redist.variance_recovered - expected_recovered).abs() < 1e-9,
        "recovered {} vs expected {expected_recovered}",
        redist.variance_recovered
    );
    // Some residual remains → not fully compensable.
    assert!(redist.redistributed_residual_variance > 1e-9);
    assert!(!redist.fully_compensable);
}

/// `shared_downstream_targets` finds every surviving neuron that feeds a target
/// the candidate also feeds, tagged with the candidate's weight into that
/// target — and excludes the candidate itself and non-shared survivors.
#[test]
fn shared_downstream_targets_enumerates_survivors() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "cand", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "surv", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "lonely", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "cand", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "surv", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "lonely", "weight": 1.0},
                {"fromUUID": "cand", "toUUID": "out-0", "weight": 0.7},
                {"fromUUID": "surv", "toUUID": "out-0", "weight": 0.3},
                {"fromUUID": "lonely", "toUUID": "cand", "weight": 0.5}
            ]
        }"#,
    );

    let targets = shared_downstream_targets(&c, "cand");
    assert_eq!(targets.len(), 1, "only `surv` shares a downstream target");
    let t = &targets[0];
    assert_eq!(t.target_uuid, "out-0");
    assert_eq!(t.survivor_uuid, "surv");
    assert!((t.candidate_weight - 0.7).abs() < 1e-6);
}

/// End-to-end: given a topology and per-sample records, pick the survivor whose
/// weight redistribution recovers the most variance. A survivor whose signal
/// tracks the candidate makes the hygiene-forced removal non-regressive.
#[test]
fn best_weight_redistribution_selects_correlated_survivor() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "cand", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "good", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "bad", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "cand", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "good", "toUUID": "out-0", "weight": 0.1},
                {"fromUUID": "bad", "toUUID": "out-0", "weight": 0.1}
            ]
        }"#,
    );

    // `good` tracks `cand` exactly; `bad` is anti-phase / uncorrelated.
    let mut records = Vec::new();
    let cand = [1.0f32, 2.0, 3.0, 4.0];
    let good = [1.0f32, 2.0, 3.0, 4.0];
    let bad = [1.0f32, 1.0, 1.0, 1.0]; // constant → zero variance
    for (i, ((&cv, &gv), &bv)) in cand.iter().zip(good.iter()).zip(bad.iter()).enumerate() {
        let obs = u32::try_from(i).expect("small index");
        records.push(rec(obs, "cand", cv));
        records.push(rec(obs, "good", gv));
        records.push(rec(obs, "bad", bv));
    }

    let (target, redist) =
        best_weight_redistribution(&c, "cand", &records).expect("a shared target with samples");

    assert_eq!(target.survivor_uuid, "good");
    assert!(redist.fully_compensable);
    assert!(redist.redistributed_residual_variance < 1e-9);
}

/// No shared downstream survivor (or no samples) means (d) cannot be evaluated
/// — the function returns `None` rather than fabricating a compensation.
#[test]
fn best_weight_redistribution_none_without_shared_survivor() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "cand", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "cand", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );
    let records = vec![rec(0, "cand", 1.0), rec(1, "cand", 2.0)];
    assert!(best_weight_redistribution(&c, "cand", &records).is_none());
}
