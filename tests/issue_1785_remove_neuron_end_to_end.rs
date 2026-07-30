//! Issue #1815 — end-to-end guard for the remove-neuron path.
//!
//! Milestone #1785 observed a removal path that yielded nothing and explained
//! nothing (`rejectionBreakdown: null`). Its test-gap section named the exact
//! hole: "No test drives `analyze_parallel` end-to-end and asserts a
//! `removeNeuron` candidate survives to the FFI response." Every existing
//! removal test drives one unit — `apply_honest_remove_neuron_gain`,
//! `apply_coordinated_gain_floor`, `identify_structural_removal_candidates` —
//! on a hand-built input. Nothing composed them, which is precisely how two
//! independent unreachability gates could both ship green.
//!
//! This suite is the missing composition. Unlike
//! `tests/issue_1785_remove_neuron_reachability.rs`, which is a
//! characterisation pin of today's numbers, these are **standing invariants**:
//! they must hold for every future calibration of the gates.
//!
//! 1. **Reachability** — on a fixture carrying a hidden neuron that is
//!    genuinely worth pruning (zero downstream influence, non-trivial synapse
//!    count), both shipped entry points must return a `removeNeuron` candidate:
//!    `analyze_parallel` (analysis path) and `rank_focus_neurons`, which is how
//!    production reaches `identify_structural_removal_candidates` (focus path).
//!    Asserted on the **FFI response shape**, so no intermediate-vector
//!    refactor can keep it green while the candidate is deleted downstream.
//! 2. **Non-silence** — on a fixture where nothing should be pruned, the
//!    response's rejection breakdown must be non-empty and must name a
//!    **removal-related reason key**. A zero-candidate removal pass carrying no
//!    reason is a fail-loud violation, not a quiet success.
//! 3. **Negative direction** — the high-influence neuron in the reachability
//!    fixture must **not** be returned, so the guard cannot be satisfied by
//!    weakening a gate into accepting everything.
//!
//! Both fixtures are built in code from the shipped constants — the prunable
//! neuron's synapse degree is derived from `remove_low_impact_noise_floor()` and
//! `calculate_removal_savings`, never hard-coded — so a recalibration of either
//! moves the fixture with it rather than breaking the invariant.
//!
//! No GPU-only or discovery-parquet fixture is required: the focus cases are
//! structure-only (Issue #1766) and are handed a deliberately unopenable
//! parquet path, and the analysis cases write their records to a `tempfile`
//! parquet and skip on the repo's standard GPU guard.

// Numeric casts below build synthetic activations from loop counters.
#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analysis::constants::{
    REMOVAL_CANDIDATE_BOOST, remove_low_impact_noise_floor,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_REMOVAL_ACTIVE_NEURON, REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
    REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING, REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
};
use neat_ai_discovery::focus::{DEFAULT_COST_OF_GROWTH, calculate_removal_savings};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal, rank_focus_neurons_internal,
};
use serde_json::{Value, json};
use serial_test::serial;

/// The production cost-of-growth (NEAT-AI's `Score.ts` value). Every assertion
/// below runs at the shipped default, never a test-friendly override.
const COST_OF_GROWTH: f32 = DEFAULT_COST_OF_GROWTH;

/// Env vars that would move the gates under test. Both are unset for every test
/// so a stray local override cannot make a red guard look green.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
const COORDINATED_MULTIPLIER_ENV: &str = "NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER";

/// A parquet path that cannot be opened. The focus path is structure-only
/// (Issue #1766), so a successful response also proves no record decode happened.
const MISSING_PARQUET: &str = "/nonexistent/issue-1815/discovery.parquet";

/// The hidden neuron that is genuinely worth pruning: no outgoing synapse (so
/// zero downstream influence) and a non-trivial incoming degree.
const PRUNABLE: &str = "h-prunable";

/// The hidden neuron carrying the creature's whole output path — the negative
/// direction. It must never be offered for removal.
const HIGH_INFLUENCE: &str = "h-hot";

/// The non-silence fixture's neuron: quiet enough for the low-impact detector to
/// propose removing it, yet structurally dominant, so the removal is *proposed*
/// and must then be *rejected with a named reason*.
const QUIET_BUT_DOMINANT: &str = "h-quiet-dominant";

const OUTPUT: &str = "out-0";

/// Observations recorded per neuron — comfortably above the 20-sample minimum
/// the dead-neuron detector needs (`MIN_DISCOVERY_SAMPLE_COUNT`).
const OBSERVATIONS: u32 = 64;

/// Every reason key that means "a removal was considered and dropped". The
/// non-silence invariant checks the **key**, so a mis-keyed or renamed reason is
/// caught rather than being masked by a non-zero total.
const REMOVAL_REJECTION_REASONS: [&str; 4] = [
    REJECTION_REMOVAL_SAVINGS_BELOW_IMPACT,
    REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
    REJECTION_REMOVAL_ACTIVE_NEURON,
    REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING,
];

// ---------------------------------------------------------------------------
// Env guard
// ---------------------------------------------------------------------------

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

/// Unset both gate-moving env vars for the duration of a test.
fn default_gates() -> (EnvVarGuard, EnvVarGuard) {
    (
        EnvVarGuard::unset(NOISE_FLOOR_ENV),
        EnvVarGuard::unset(COORDINATED_MULTIPLIER_ENV),
    )
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

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

/// The smallest synapse degree whose boosted removal savings clear the shipped
/// noise floor with margin, at the production cost-of-growth.
///
/// Derived rather than hard-coded: the guard is "a neuron worth pruning reaches
/// the response", so when a gate is recalibrated (Issue #1814 lowers this floor)
/// the fixture shrinks with it and the invariant still holds. Since #1814 the
/// floor is denominated in units of `costOfGrowth`, so the target is evaluated
/// at the same `COST_OF_GROWTH` the savings are.
fn prunable_degree() -> usize {
    let target = remove_low_impact_noise_floor(COST_OF_GROWTH) * 1.5;
    (1..=100_000)
        .find(|degree| {
            calculate_removal_savings(*degree, 0, COST_OF_GROWTH) * REMOVAL_CANDIDATE_BOOST
                >= target
        })
        .expect("a degree clearing the noise floor must exist below 100k synapses")
}

/// The reachability fixture.
///
/// - [`PRUNABLE`]: `ReLU` with a large negative bias, fed by every input and
///   feeding nothing. Zero downstream influence, [`prunable_degree`] synapses —
///   the removal every gate should agree on.
/// - [`HIGH_INFLUENCE`]: carries the sole path to the output. Removing it costs
///   the whole creature, so it must be rejected.
///
/// Neurons are ordered inputs → hidden → output, so every synapse points
/// forward (Issue #1184).
fn prunable_creature() -> CreatureJson {
    let degree = prunable_degree();
    let inputs: Vec<String> = (0..degree).map(|i| format!("in-{i}")).collect();

    let mut neurons: Vec<NeuronJson> = inputs
        .iter()
        .map(|uuid| neuron(uuid, "input", "IDENTITY", 0.0))
        .collect();
    neurons.push(neuron(PRUNABLE, "hidden", "ReLU", -10.0));
    neurons.push(neuron(HIGH_INFLUENCE, "hidden", "TANH", 0.0));
    neurons.push(neuron(OUTPUT, "output", "IDENTITY", 0.0));

    let mut synapses: Vec<SynapseJson> = inputs
        .iter()
        .map(|uuid| synapse(uuid, PRUNABLE, 0.1))
        .collect();
    synapses.push(synapse("in-0", HIGH_INFLUENCE, 1.0));
    synapses.push(synapse(HIGH_INFLUENCE, OUTPUT, 1.0));

    CreatureJson {
        input: degree,
        output: 1,
        neurons,
        synapses,
    }
}

/// The non-silence fixture: **no** neuron here is worth pruning.
///
/// [`QUIET_BUT_DOMINANT`] is recorded quietly enough for the low-impact detector
/// to *propose* removing it — but it carries a dominant weight into the output,
/// so removing it destroys the creature. Both paths must reject it, and both must
/// say why.
fn no_removal_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            neuron("in-0", "input", "IDENTITY", 0.0),
            neuron("in-1", "input", "IDENTITY", 0.0),
            neuron(QUIET_BUT_DOMINANT, "hidden", "TANH", 0.0),
            neuron(HIGH_INFLUENCE, "hidden", "TANH", 0.0),
            neuron(OUTPUT, "output", "IDENTITY", 0.0),
        ],
        synapses: vec![
            synapse("in-0", QUIET_BUT_DOMINANT, 1.0),
            synapse("in-1", HIGH_INFLUENCE, 1.0),
            synapse(QUIET_BUT_DOMINANT, OUTPUT, 100.0),
            synapse(HIGH_INFLUENCE, OUTPUT, 1.0),
        ],
    }
}

/// `ReLU(x − 10)` is zero across the whole input range: dead, not merely quiet.
fn dead_activation(_x: f32) -> f32 {
    0.0
}

/// Small but genuinely varying — inside the low-impact detector's band
/// (`1e-6 < mean|a| < 0.04`, `std < 0.02`) and well outside the constant-neuron
/// band, so neither the #1623 bias fold nor the #1622 promotion can rescue the
/// candidate from the gain gate.
fn quiet_but_varying_activation(x: f32) -> f32 {
    0.02 + 0.004 * x
}

/// Records for one creature: `quiet_uuid`'s activation comes from `quiet`,
/// `live_uuid` varies over `[-1, 1]`, and the output carries a residual error so
/// the analysis pass has signal to work with.
fn records_for(quiet_uuid: &str, quiet: fn(f32) -> f32, live_uuid: &str) -> Vec<DiscoverRecord> {
    let mut out = Vec::with_capacity(OBSERVATIONS as usize * 3);
    for obs in 0..OBSERVATIONS {
        let x = (obs as f32 / OBSERVATIONS as f32) * 2.0 - 1.0;
        let live = x.tanh();
        out.push(DiscoverRecord::new(
            obs,
            quiet_uuid.to_string(),
            Some(x),
            quiet(x),
            vec![0.0],
        ));
        out.push(DiscoverRecord::new(
            obs,
            live_uuid.to_string(),
            Some(x),
            live,
            vec![0.1 * x],
        ));
        out.push(DiscoverRecord::new(
            obs,
            OUTPUT.to_string(),
            Some(live),
            live,
            vec![0.3 * x],
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Drive the shipped analysis entry point, `analyze_parallel`, over a parquet
/// written for this test. Returns the parsed FFI response.
fn analyse(creature: &CreatureJson, records: &[DiscoverRecord], focus: &[&str]) -> Value {
    let dir = tempfile::tempdir().expect("create temp dir");
    let parquet = dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();
    write_records_to_parquet(&parquet, records).expect("write parquet");

    let input = json!({
        "parquetFile": parquet,
        "creature": creature,
        "focusNeurons": focus,
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 32,
        "randomSeed": 42,
    })
    .to_string();

    let response: Value =
        serde_json::from_str(&analyze_parallel_internal(&input).expect("analyze_parallel returns"))
            .expect("FFI response JSON");
    assert_eq!(
        response["success"], true,
        "analyze_parallel must succeed: {response:?}"
    );
    response
}

/// Drive the shipped focus entry point, `rank_focus_neurons`, which is how
/// production reaches `identify_structural_removal_candidates` (Issue #1806).
fn focus(creature: &CreatureJson) -> Value {
    let input = json!({
        "parquetFile": MISSING_PARQUET,
        "creature": creature,
        "maxResults": 256,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": COST_OF_GROWTH,
    })
    .to_string();
    let response: Value =
        serde_json::from_str(&rank_focus_neurons_internal(&input).expect("focus path returns"))
            .expect("FFI response JSON");
    assert_eq!(
        response["success"], true,
        "the structure-only focus path must succeed: {response:?}"
    );
    response
}

// ---------------------------------------------------------------------------
// Response readers — all assertions run on the FFI response shape
// ---------------------------------------------------------------------------

/// Every neuron targeted by a `removeNeuron` operation in an `analyze_parallel`
/// response's `coordinatedStructuralCandidates`, sole-op or grouped.
fn removal_targets_in_analysis_response(response: &Value) -> Vec<String> {
    response["coordinatedStructuralCandidates"]
        .as_array()
        .map(|candidates| {
            candidates
                .iter()
                .filter_map(|c| c["operations"].as_array())
                .flatten()
                .filter(|op| op["type"] == "removeNeuron")
                .filter_map(|op| op["neuronUuid"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Every neuron offered for removal in a `rank_focus_neurons` response.
fn removal_targets_in_focus_response(response: &Value) -> Vec<String> {
    response["removalCandidates"]
        .as_array()
        .map(|candidates| {
            candidates
                .iter()
                .filter_map(|c| c["neuronUuid"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The removal-related reason keys carrying a non-zero count in `breakdown`.
fn named_removal_reasons(breakdown: &Value) -> Vec<&'static str> {
    REMOVAL_REJECTION_REASONS
        .into_iter()
        .filter(|reason| breakdown[*reason].as_u64().unwrap_or(0) > 0)
        .collect()
}

/// Assert `breakdown` is populated and names at least one removal reason —
/// the fail-loud contract #1785 saw violated as `rejectionBreakdown: null`.
fn assert_names_a_removal_reason(breakdown: &Value, surface: &str) {
    assert!(
        breakdown.is_object(),
        "{surface} must carry a rejection breakdown after a zero-yield removal pass, got \
         {breakdown} — a pass that drops every removal without saying why is the exact \
         fail-loud violation Issue #1785 observed"
    );
    let counts = breakdown.as_object().expect("checked to be an object");
    assert!(
        !counts.is_empty(),
        "{surface} rejection breakdown must not be empty after a zero-yield removal pass"
    );
    let named = named_removal_reasons(breakdown);
    assert!(
        !named.is_empty(),
        "{surface} rejection breakdown must name a removal-related reason key (one of \
         {REMOVAL_REJECTION_REASONS:?}); got keys {:?} — a non-zero total under an unrelated \
         key does not explain why the removal pass yielded nothing",
        counts.keys().collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Invariant 1 — reachability
// ---------------------------------------------------------------------------

/// A `removeNeuron` candidate for a genuinely prunable neuron must survive to
/// the `analyze_parallel` FFI response, and the high-influence neuron must not.
#[test]
#[serial]
fn analyze_parallel_returns_a_remove_neuron_candidate_for_a_prunable_neuron() {
    let _gates = default_gates();
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = prunable_creature();
    let records = records_for(PRUNABLE, dead_activation, HIGH_INFLUENCE);
    let response = analyse(&creature, &records, &[OUTPUT, HIGH_INFLUENCE, PRUNABLE]);
    let targets = removal_targets_in_analysis_response(&response);

    println!("=== analyze_parallel removal targets: {targets:?}");

    assert!(
        targets.iter().any(|uuid| uuid == PRUNABLE),
        "a removeNeuron candidate for {PRUNABLE} (zero downstream influence, {} synapses) must \
         reach the analyze_parallel response; got {targets:?}. Any gate, floor or triage change \
         that strips the candidate before the FFI boundary fails here — see \
         docs/analysis/remove-neuron-reachability-1785.md",
        prunable_degree(),
    );
    assert!(
        !targets.iter().any(|uuid| uuid == HIGH_INFLUENCE),
        "{HIGH_INFLUENCE} carries the creature's whole output path and must never be offered for \
         removal; got {targets:?}. This guard must not be satisfiable by weakening a gate into \
         accepting everything"
    );
}

/// The same invariant on the focus path: `rank_focus_neurons` — the shipped
/// route into `identify_structural_removal_candidates` — must return the
/// prunable neuron and must not return the high-influence one.
#[test]
#[serial]
fn focus_ffi_path_returns_a_removal_candidate_for_a_prunable_neuron() {
    let _gates = default_gates();
    let creature = prunable_creature();
    let response = focus(&creature);
    let targets = removal_targets_in_focus_response(&response);

    println!("=== focus removal targets: {targets:?}");
    println!(
        "=== focus rejectionBreakdown: {}",
        response["rejectionBreakdown"]
    );

    assert!(
        targets.iter().any(|uuid| uuid == PRUNABLE),
        "a removal candidate for {PRUNABLE} (zero structural contribution, {} synapses, \
         costOfGrowth={COST_OF_GROWTH:e}) must reach the rank_focus_neurons response; got \
         {targets:?}",
        prunable_degree(),
    );
    assert!(
        !targets.iter().any(|uuid| uuid == HIGH_INFLUENCE),
        "{HIGH_INFLUENCE} carries the creature's whole output path and must never be offered for \
         removal; got {targets:?}"
    );
}

// ---------------------------------------------------------------------------
// Invariant 2 — non-silence
// ---------------------------------------------------------------------------

/// A zero-yield removal pass on the focus path must name a removal reason.
#[test]
#[serial]
fn focus_ffi_path_names_a_removal_reason_when_nothing_is_prunable() {
    let _gates = default_gates();
    let creature = no_removal_creature();
    let response = focus(&creature);
    let targets = removal_targets_in_focus_response(&response);
    let breakdown = &response["rejectionBreakdown"];

    println!("=== focus zero-yield rejectionBreakdown: {breakdown}");

    assert!(
        targets.is_empty(),
        "no neuron in this fixture is worth pruning, so the focus response must offer none; got \
         {targets:?}"
    );
    assert_names_a_removal_reason(breakdown, "rank_focus_neurons rejectionBreakdown");
}

/// A zero-yield removal pass on the analysis path must name a removal reason.
///
/// [`QUIET_BUT_DOMINANT`] is quiet enough for the low-impact detector to propose
/// its removal, so the candidate *is* created and then dropped by the gain gate —
/// exactly the case where a silent drop would leave the FFI caller with no
/// explanation.
#[test]
#[serial]
fn analyze_parallel_names_a_removal_reason_when_nothing_is_prunable() {
    let _gates = default_gates();
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = no_removal_creature();
    let records = records_for(
        QUIET_BUT_DOMINANT,
        quiet_but_varying_activation,
        HIGH_INFLUENCE,
    );
    let response = analyse(
        &creature,
        &records,
        &[OUTPUT, HIGH_INFLUENCE, QUIET_BUT_DOMINANT],
    );
    let targets = removal_targets_in_analysis_response(&response);
    let breakdown = &response["synapseMetadata"]["rejectionBreakdown"];

    println!("=== analyze_parallel zero-yield removal targets: {targets:?}");
    println!("=== synapseMetadata.rejectionBreakdown: {breakdown}");

    assert!(
        !targets.iter().any(|uuid| uuid == QUIET_BUT_DOMINANT),
        "{QUIET_BUT_DOMINANT} dominates the output path — removing it destroys the creature, so it \
         must not reach the response; got {targets:?}"
    );
    assert_names_a_removal_reason(
        breakdown,
        "analyze_parallel synapseMetadata.rejectionBreakdown",
    );
}
