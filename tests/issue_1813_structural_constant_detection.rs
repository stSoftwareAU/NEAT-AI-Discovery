//! Issue #1813 — the structural functional-constancy detector is wired.
//!
//! `functionally_constant_neuron_uuids` used to ignore its argument and return
//! an empty `HashSet`, so the #1622 promotion escape hatch fired only for
//! neurons that already carried an accepted #1623 bias fold. A *structurally*
//! constant hidden neuron — one whose output cannot vary given the topology
//! alone — had no recorded activations to fold, so it was flagged by neither
//! source and deleted by the positive-gain floor with no diagnostic.
//!
//! These tests drive the detector directly and end to end:
//!
//! - **Positive**: a hidden neuron with no incoming synapses (and the zero-weight
//!   and chained-constant variants) is flagged, promoted to
//!   `CONSTANT_NEURON_PRIORITY_GAIN`, and survives the FFI-facing final gain
//!   floor.
//! - **Negative**: a variance-carrying neuron is never flagged, and neither are
//!   input or output neurons.
//! - **Boundary**: constancy that is only apparent from *activations* is not a
//!   structural judgement — it belongs to the #1623 bias fold, so the structural
//!   seam leaves it alone.
//!
//! If the seam ever regresses to returning an empty set unconditionally, the
//! positive cases below fail immediately in CI.

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::candidate_aggregation::apply_final_coordinated_gain_floor;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::remove_neuron_constant_promotion::{
    CONSTANT_NEURON_PRIORITY_GAIN, functionally_constant_neuron_uuids,
    promote_constant_remove_neuron_candidates,
};
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};
use serial_test::serial;

/// Env vars that would move the final gain floor. Both are unset for every test
/// that drives the floor, so the promoted candidate is measured against the
/// shipped defaults rather than a local override.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
const COORDINATED_MULTIPLIER_ENV: &str = "NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER";

/// RAII guard that unsets an env var for a single test and restores it after.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(v) = &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            unsafe { std::env::set_var(self.key, v) };
        }
    }
}

fn default_gates() -> (EnvVarGuard, EnvVarGuard) {
    (
        EnvVarGuard::unset(NOISE_FLOOR_ENV),
        EnvVarGuard::unset(COORDINATED_MULTIPLIER_ENV),
    )
}

fn creature(json: &str) -> CreatureJson {
    serde_json::from_str(json).expect("fixture creature JSON parses")
}

fn remove_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

/// The neuron a sole-op `RemoveNeuron` candidate targets.
fn removal_target(candidate: &CoordinatedStructuralCandidateJson) -> &str {
    match candidate.operations.as_slice() {
        [CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }] => neuron_uuid.as_str(),
        other => panic!("expected a sole-op RemoveNeuron candidate, got {other:?}"),
    }
}

/// Drive the FFI-facing final gain floor and return the surviving candidates.
fn final_floor_survivors(
    candidates: Vec<CoordinatedStructuralCandidateJson>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut synapse = AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: candidates,
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata::default(),
    };
    apply_final_coordinated_gain_floor(&mut synapse, DiscoveryMode::Normal, 1.0);
    synapse.coordinated_structural_candidates
}

/// `h-orphan` has no incoming synapse, so its output is `squash(bias)` on every
/// observation; `h-live` is driven by an input and carries real variance.
const ORPHAN_AND_LIVE: &str = r#"{
    "input": 1, "output": 1,
    "neurons": [
        {"uuid": "in-0",     "type": "input",  "squash": "IDENTITY"},
        {"uuid": "h-orphan", "type": "hidden", "squash": "TANH", "bias": 0.3},
        {"uuid": "h-live",   "type": "hidden", "squash": "RELU", "bias": 0.1},
        {"uuid": "out-0",    "type": "output", "squash": "IDENTITY"}
    ],
    "synapses": [
        {"fromUUID": "in-0",     "toUUID": "h-live", "weight": 0.8},
        {"fromUUID": "h-orphan", "toUUID": "out-0",  "weight": 0.5},
        {"fromUUID": "h-live",   "toUUID": "out-0",  "weight": 0.4}
    ]
}"#;

#[test]
fn structurally_constant_orphan_is_flagged_and_variance_carrier_is_not() {
    let creature = creature(ORPHAN_AND_LIVE);
    let flagged = functionally_constant_neuron_uuids(&creature);

    assert!(
        flagged.contains("h-orphan"),
        "a hidden neuron with no incoming synapses cannot vary — it must be flagged, got {flagged:?}"
    );
    assert!(
        !flagged.contains("h-live"),
        "a neuron driven by an input carries variance and must never be flagged, got {flagged:?}"
    );
    assert_eq!(flagged.len(), 1, "exactly one neuron is constant here");
}

#[test]
#[serial]
fn flagged_structural_constant_is_promoted_past_the_final_gain_floor() {
    let _gates = default_gates();
    let creature = creature(ORPHAN_AND_LIVE);

    // Both candidates start below the floor, exactly as the honest #1518 gain
    // ranking leaves a zero-influence neuron.
    let mut candidates = vec![
        remove_candidate("h-orphan", -0.000_1),
        remove_candidate("h-live", -0.000_1),
    ];

    let flagged = functionally_constant_neuron_uuids(&creature);
    let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged);
    assert_eq!(
        promoted, 1,
        "only the structurally-constant neuron is promoted"
    );
    assert!(
        (candidates[0].expected_creature_score_gain - CONSTANT_NEURON_PRIORITY_GAIN).abs()
            < f32::EPSILON,
        "the promoted candidate carries the priority gain"
    );

    let survivors = final_floor_survivors(candidates);
    let surviving_uuids: Vec<&str> = survivors.iter().map(removal_target).collect();
    assert_eq!(
        surviving_uuids,
        vec!["h-orphan"],
        "the promoted structural constant must clear the final gain floor while the \
         variance-carrying candidate is rejected"
    );
}

#[test]
fn all_zero_incoming_weights_is_structurally_constant() {
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "in-0",    "type": "input",  "squash": "IDENTITY"},
                {"uuid": "h-muted", "type": "hidden", "squash": "TANH", "bias": -0.2},
                {"uuid": "out-0",   "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "in-0",    "toUUID": "h-muted", "weight": 0.0},
                {"fromUUID": "h-muted", "toUUID": "out-0",   "weight": 0.7}
            ]
        }"#,
    );
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.contains("h-muted"),
        "every incoming weight is zero, so the pre-activation is the bias alone — got {flagged:?}"
    );
}

#[test]
fn constancy_propagates_through_a_chain_of_constant_sources() {
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "in-0",  "type": "input",  "squash": "IDENTITY"},
                {"uuid": "h-a",   "type": "hidden", "squash": "TANH", "bias": 0.4},
                {"uuid": "h-b",   "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "h-mix", "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "h-a",   "toUUID": "h-b",   "weight": 1.5},
                {"fromUUID": "in-0",  "toUUID": "h-mix", "weight": 0.9},
                {"fromUUID": "h-a",   "toUUID": "h-mix", "weight": 0.5},
                {"fromUUID": "h-b",   "toUUID": "out-0", "weight": 0.6},
                {"fromUUID": "h-mix", "toUUID": "out-0", "weight": 0.6}
            ]
        }"#,
    );
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(flagged.contains("h-a"), "the orphan source is constant");
    assert!(
        flagged.contains("h-b"),
        "a neuron whose only source is constant is itself constant — got {flagged:?}"
    );
    assert!(
        !flagged.contains("h-mix"),
        "one varying source is enough to carry variance — got {flagged:?}"
    );
}

#[test]
fn a_neuron_fed_by_a_declared_constant_neuron_is_flagged() {
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "bias-0", "type": "constant", "squash": "IDENTITY", "bias": 1.0},
                {"uuid": "h-c",    "type": "hidden",   "squash": "TANH", "bias": 0.0},
                {"uuid": "out-0",  "type": "output",   "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "bias-0", "toUUID": "h-c",   "weight": 1.0},
                {"fromUUID": "h-c",    "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.contains("h-c"),
        "the NEAT-AI input-side constant class cannot vary, so its consumer is constant too — \
         got {flagged:?}"
    );
    assert!(
        !flagged.contains("bias-0"),
        "only hidden neurons are removal candidates, so the constant input itself is never flagged"
    );
}

#[test]
fn input_and_output_neurons_are_never_flagged() {
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "in-0",  "type": "input",  "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.5}
            ],
            "synapses": []
        }"#,
    );
    let flagged = functionally_constant_neuron_uuids(&creature);
    assert!(
        flagged.is_empty(),
        "an unfed output is not a removal candidate and an input is never constant — got {flagged:?}"
    );
}

#[test]
fn measured_only_constancy_is_left_to_the_bias_fold() {
    // `h-flat` is structurally free to vary (a live input drives it with a
    // non-zero weight); only its *recorded activations* could reveal constancy.
    // That judgement belongs to the #1623 bias fold, not to this seam.
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "in-0",   "type": "input",  "squash": "IDENTITY"},
                {"uuid": "h-flat", "type": "hidden", "squash": "STEP", "bias": 5.0},
                {"uuid": "out-0",  "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "in-0",   "toUUID": "h-flat", "weight": 0.001},
                {"fromUUID": "h-flat", "toUUID": "out-0",  "weight": 1.0}
            ]
        }"#,
    );
    assert!(
        functionally_constant_neuron_uuids(&creature).is_empty(),
        "constancy visible only in activations must not be claimed structurally"
    );
}

#[test]
fn a_dangling_source_is_never_assumed_constant() {
    // `h-dangle`'s only incoming synapse names a neuron that is not in the
    // creature. Nothing can be proven about it, so it must not be flagged
    // (fail-safe rather than a silent assumption).
    let creature = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "in-0",     "type": "input",  "squash": "IDENTITY"},
                {"uuid": "h-dangle", "type": "hidden", "squash": "TANH", "bias": 0.0},
                {"uuid": "out-0",    "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "h-missing", "toUUID": "h-dangle", "weight": 1.0},
                {"fromUUID": "h-dangle",  "toUUID": "out-0",    "weight": 1.0}
            ]
        }"#,
    );
    assert!(
        functionally_constant_neuron_uuids(&creature).is_empty(),
        "an unknown source cannot be proven constant"
    );
}
