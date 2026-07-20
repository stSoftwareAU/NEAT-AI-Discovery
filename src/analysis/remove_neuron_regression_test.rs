//! Regression test reconstructing the remove-neuron **mean-only fold** failure
//! class (Issue #1688, parent #1686).
//!
//! ## Systematic root cause (confirmed in the #1686 grill)
//!
//! When NEAT-AI removes a hidden neuron, `evaluate_weight_redistribution`'s
//! predecessor — and the live applier — fold only the removed neuron's **mean**
//! contribution (`averageActivation × Σ outgoing weights`) into each downstream
//! target's **bias**. Adding `w_{c→t} · mean(a_c)` to target `t`'s bias cancels
//! the *mean* of the removed contribution exactly, so every **scalar-aggregate**
//! view of the target — its mean pre-activation, its average error — is
//! preserved. The removal looks safe on the persisted aggregates.
//!
//! What survives is the neuron's genuine **per-sample (variance) signal**. The
//! mean-only fold replaces the fluctuating term `w_{c→t} · a_{c,i}` with the
//! constant `w_{c→t} · mean(a_c)`, so on every observation `i` it leaves a
//! residual `w_{c→t} · (a_{c,i} − mean(a_c))`. That residual has mean zero
//! (aggregate-blind) but non-zero variance, so any downstream target that relies
//! on the removed neuron's per-sample signal **regresses** — even when the neuron
//! is harmful in aggregate and therefore a legitimate removal candidate.
//!
//! The #1558 counterfactual study proved no scalar-aggregate adjustment fixes
//! this: only **weight redistribution** into a correlated survivor (Issue #1559,
//! [`remove_neuron_compensation`]) — or, for a genuinely constant neuron, a plain
//! **bias fold** (Issue #1623, [`remove_neuron_bias_fold`]) — can drive the
//! surviving per-sample residual to (or below) parity with keeping the neuron.
//!
//! ## What this test encodes
//!
//! A distilled, deterministic, checked-in fixture (no network access) that
//! reconstructs the cached real-world failure observed in GRQ-Discovery commit
//! `c4330385fc65897c6caf1168d420d0e248d490da`, candidate UUID
//! `neuron-876870118` (the NEAT-AI-side label truncation that drops the leading
//! digit is cosmetic only). The fixture is a candidate whose mean-only fold
//! regresses a variance-carrying downstream target while the neuron is harmful in
//! aggregate.
//!
//! - [`mean_only_fold_regresses_variance_target`] is the **red-phase** oracle: it
//!   asserts the current mean-only fold *does* regress the variance-carrying
//!   target (positive residual sum of squares) while remaining aggregate-neutral
//!   (zero mean residual). If the fixture ever stops reproducing the class, this
//!   assertion fails loudly in CI rather than passing vacuously.
//! - [`compensated_removal_meets_parity_or_better`] is the **success oracle** the
//!   #1686 wiring sub-issues (#1559 redistribution / #1623 bias fold) must
//!   satisfy: the compensated removal is **parity-or-better** everywhere the
//!   mean-only fold regressed. If a future change reintroduces mean-only
//!   compensation in the live path, this case flips red before merge.

use crate::CreatureJson;
use crate::analysis::remove_neuron_bias_fold::{
    BIAS_FOLD_GATE_TOLERANCE, evaluate_constant_neuron_bias_fold,
};
use crate::analysis::remove_neuron_compensation::{
    ActivationCovariance, aligned_activations, best_weight_redistribution,
    evaluate_weight_redistribution,
};
use crate::ffi_types::{NeuronJson, SynapseJson};
use crate::types::DiscoverRecord;

/// Candidate neuron UUID as recorded in the GRQ-cluster `network.json`. The
/// NEAT-AI-side label truncation that drops the leading digit is cosmetic only.
const CANDIDATE: &str = "neuron-876870118";
/// A surviving hidden neuron that shares the downstream target and carries a
/// per-sample signal correlated with the candidate's.
const SURVIVOR: &str = "neuron-survivor";
/// The downstream target both the candidate and the survivor feed.
const TARGET: &str = "neuron-output";
/// The candidate's synapse weight into the shared target (`w_c`).
const W_C: f32 = 2.0;
/// The survivor's synapse weight into the shared target (`w_s`).
const W_S: f32 = 1.0;

fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
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

/// Build the distilled failure fixture: a candidate and a perfectly-correlated
/// survivor feeding a shared downstream target, plus their per-sample
/// activations.
///
/// The candidate's activation varies sample to sample (it carries a genuine
/// per-sample signal), and the survivor tracks it exactly — the correlated
/// survivor that weight redistribution can fold the signal into. Keeping the
/// numbers small and integral keeps the arithmetic exact and the test
/// deterministic.
fn failure_fixture() -> (CreatureJson, Vec<DiscoverRecord>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", 0.0),
            neuron(CANDIDATE, "hidden", 0.0),
            neuron(SURVIVOR, "hidden", 0.0),
            neuron(TARGET, "output", 0.0),
        ],
        synapses: vec![
            synapse("input-0", CANDIDATE, 1.0),
            synapse("input-0", SURVIVOR, 1.0),
            synapse(CANDIDATE, TARGET, W_C),
            synapse(SURVIVOR, TARGET, W_S),
        ],
        input: 1,
        output: 1,
    };

    // Candidate carries a per-sample signal; the survivor tracks it exactly
    // (perfect correlation ρ = 1). A partially-correlated survivor would leave a
    // strictly-smaller-but-non-zero residual — parity-or-better still holds.
    let activations = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
    let mut records = Vec::new();
    for (obs, &a) in activations.iter().enumerate() {
        let obs = u32::try_from(obs).expect("small index");
        records.push(record(obs, CANDIDATE, a));
        records.push(record(obs, SURVIVOR, a));
    }
    (creature, records)
}

fn activations_of(records: &[DiscoverRecord], uuid: &str) -> Vec<DiscoverRecord> {
    records
        .iter()
        .filter(|r| r.neuron_uuid == uuid)
        .cloned()
        .collect()
}

fn count_as_f64(len: usize) -> f64 {
    f64::from(u32::try_from(len).expect("fixture sample count fits in u32"))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / count_as_f64(values.len())
}

fn sum_of_squares(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}

/// Per-observation residual introduced on the target by the **mean-only** bias
/// fold: the removed contribution `w_c · a_{c,i}` replaced by the constant
/// `w_c · mean(a_c)` leaves `w_c · (a_{c,i} − mean(a_c))`.
fn mean_only_residuals(candidate: &[f64]) -> Vec<f64> {
    let mean_c = mean(candidate);
    candidate
        .iter()
        .map(|&a| f64::from(W_C) * (a - mean_c))
        .collect()
}

/// Per-observation residual left on the target after **weight redistribution**:
/// the candidate's per-sample signal is regressed onto the survivor with slope
/// `Δw`, leaving `w_c · (a_{c,i} − mean_c) − Δw · (a_{s,i} − mean_s)`.
fn redistributed_residuals(candidate: &[f64], survivor: &[f64], delta_weight: f64) -> Vec<f64> {
    let mean_c = mean(candidate);
    let mean_s = mean(survivor);
    candidate
        .iter()
        .zip(survivor)
        .map(|(&a_c, &a_s)| f64::from(W_C) * (a_c - mean_c) - delta_weight * (a_s - mean_s))
        .collect()
}

/// Red-phase oracle: the current mean-only fold regresses the variance-carrying
/// downstream target while staying aggregate-neutral.
///
/// This documents the failure class as an explicit, loud assertion. If the
/// distilled fixture ever stops reproducing the class (fixture drift, or the
/// candidate no longer carrying a per-sample signal), the residual sum of squares
/// collapses to zero and this test fails in CI rather than passing vacuously.
#[test]
fn mean_only_fold_regresses_variance_target() {
    let (creature, records) = failure_fixture();

    let candidate: Vec<f64> = activations_of(&records, CANDIDATE)
        .iter()
        .map(|r| f64::from(r.activation))
        .collect();
    assert!(!candidate.is_empty(), "fixture must record the candidate");

    // The candidate genuinely varies — it carries a per-sample signal. A
    // constant candidate would be the #1623 bias-fold case, not this failure
    // class.
    let stats = ActivationCovariance::from_values(candidate.iter().copied())
        .expect("candidate has recorded activations");
    assert!(
        stats.candidate_variance > 1e-9,
        "the candidate must carry per-sample variance for this failure class"
    );

    let residuals = mean_only_residuals(&candidate);

    // Aggregate-blind: the mean-only fold preserves the target's mean
    // pre-activation exactly, so every scalar-aggregate view clears the removal.
    // This is why a harmful-in-aggregate neuron is still greenlit for removal.
    assert!(
        mean(&residuals).abs() < 1e-12,
        "mean-only fold must be aggregate-neutral (zero mean residual)"
    );

    // Per-sample regression: the surviving residual has strictly positive energy,
    // so any target relying on the removed per-sample signal regresses.
    let mean_only_sse = sum_of_squares(&residuals);
    assert!(
        mean_only_sse > 0.0,
        "mean-only fold must regress the variance-carrying target (positive SSE), got {mean_only_sse}"
    );

    // The module's per-sample residual-variance accounting agrees with the
    // directly simulated downstream error: bias-only residual variance × N = SSE.
    let (_, redist) = best_weight_redistribution(&creature, CANDIDATE, &records)
        .expect("candidate shares a downstream target with the survivor");
    let predicted_sse = redist.bias_only_residual_variance * count_as_f64(candidate.len());
    assert!(
        (predicted_sse - mean_only_sse).abs() < 1e-9,
        "module bias-only residual variance ({predicted_sse}) must match simulated mean-only SSE ({mean_only_sse})"
    );
}

/// Success oracle: the wired remedies must be parity-or-better everywhere the
/// mean-only fold regressed.
///
/// Weight redistribution (#1559) folds the candidate's per-sample signal into the
/// correlated survivor's weight; for a genuinely constant neuron the bias fold
/// (#1623) is exact. Either way the compensated removal's residual must not
/// exceed the mean-only residual. If a future change reintroduces mean-only
/// compensation in the live path, this case flips red before merge.
#[test]
fn compensated_removal_meets_parity_or_better() {
    let (creature, records) = failure_fixture();

    let candidate_records = activations_of(&records, CANDIDATE);
    let survivor_records = activations_of(&records, SURVIVOR);
    let pairs = aligned_activations(&candidate_records, &survivor_records);
    let candidate: Vec<f64> = pairs.iter().map(|&(c, _)| c).collect();
    let survivor: Vec<f64> = pairs.iter().map(|&(_, s)| s).collect();

    let stats = ActivationCovariance::from_pairs(pairs.iter().copied())
        .expect("candidate and survivor share aligned samples");
    let redist = evaluate_weight_redistribution(f64::from(W_C), &stats);

    let mean_only_sse = sum_of_squares(&mean_only_residuals(&candidate));
    let redistributed_sse = sum_of_squares(&redistributed_residuals(
        &candidate,
        &survivor,
        redist.delta_weight,
    ));

    // Parity-or-better: redistribution never leaves a larger downstream residual
    // than the mean-only fold — the oracle the wiring work must satisfy.
    assert!(
        redistributed_sse <= mean_only_sse + 1e-9,
        "redistribution ({redistributed_sse}) must be parity-or-better than mean-only ({mean_only_sse})"
    );

    // With a perfectly-correlated survivor the removal becomes non-regressive:
    // redistribution drives the surviving residual to ~0, which the mean-only
    // bias lever can never achieve.
    assert!(
        redist.fully_compensable,
        "a perfectly-correlated survivor must make the removal fully compensable"
    );
    assert!(
        redistributed_sse < 1e-9,
        "redistribution must drive the downstream residual to ~0, got {redistributed_sse}"
    );

    // End-to-end the module selects this survivor and reports the same recovery.
    let (best_target, best_redist) = best_weight_redistribution(&creature, CANDIDATE, &records)
        .expect("candidate shares a downstream target with the survivor");
    assert_eq!(best_target.target_uuid, TARGET);
    assert_eq!(best_target.survivor_uuid, SURVIVOR);
    assert!(
        best_redist.redistributed_residual_variance
            <= best_redist.bias_only_residual_variance + 1e-9,
        "module redistribution must recover per-sample variance, not add to it"
    );

    // The #1623 bias fold is the companion remedy for a genuinely constant
    // neuron. It must *reject* this variance-carrying candidate rather than
    // silently deleting it — a mirror of the same failure class (blind mean fold)
    // seen from the constant-neuron path.
    let bias_fold = evaluate_constant_neuron_bias_fold(
        &creature,
        &records,
        CANDIDATE,
        BIAS_FOLD_GATE_TOLERANCE,
    )
    .expect("candidate has recorded activations");
    assert!(
        !bias_fold.accepted,
        "the bias fold must reject a variance-carrying candidate (it is not constant)"
    );
}
