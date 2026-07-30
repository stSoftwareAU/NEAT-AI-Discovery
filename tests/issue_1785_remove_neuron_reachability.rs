//! Issue #1810 — characterise the remove-neuron zero yield end to end.
//!
//! Milestone #1785 cites a reproduction (`tests/issue_1777_discovery_diagnostic.rs`
//! and a "Fresh-run evidence" section in
//! `docs/analysis/candidate-rate-diagnosis-1777.md`) that does not exist on
//! `Develop`, so none of its quoted measurements could be re-derived and no test
//! pinned the end-to-end zero yield the milestone is trying to change. This suite
//! is that missing evidence base: it drives **both** shipped gates over one
//! committed production-shaped fixture and pins what they emit today.
//!
//! The two gates, from one creature:
//!
//! - **Gate 1 — analysis path.** A sole-op `RemoveNeuron` candidate per hidden
//!   neuron, run through `apply_honest_remove_neuron_gain` (#1530) and then the
//!   FFI-facing `apply_final_coordinated_gain_floor`.
//!
//!   **Updated by Issue #1812.** The original pin recorded a zero yield: the
//!   emitted gain was `−impact`, non-positive by construction, screened against
//!   the strictly-positive `coordinated_post_discount_noise_floor(1)`, so no
//!   candidate could clear the floor. #1812 gave the gain its missing benefit
//!   term and unit conversion (`saving − calibrated influence loss`, decided by
//!   #1811) and routes sole-op removals to `removal_net_gain_floor`. Blocks 1
//!   and 1b now pin the post-fix behaviour: the three zero-influence orphans
//!   reach the FFI-facing survivor set, and every neuron carrying real influence
//!   — including a synthetic high-influence one — is still rejected and counted
//!   under `REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING`.
//! - **Gate 2 — focus / FFI path.** `identify_structural_removal_candidates` is
//!   `pub(crate)`, so it is reached the way production reaches it — through
//!   `rank_focus_neurons_internal` (Issue #1806) — and the boosted savings from
//!   pruning a neuron are compared against its structural contribution.
//!
//! Plus the promotion escape hatch (#1622/#1779) and the break-even degree
//! #1785 quotes as 657 synapses — checked by construction rather than quoted,
//! and now `0` since #1814 re-denominated Gate 2's floor in units of
//! `costOfGrowth`.
//!
//! # This is a characterisation pin, not an invariant
//!
//! Every measurement below records **today's** behaviour. A fix to either gate —
//! a change to `coordinated_post_discount_noise_floor`,
//! `REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS`, `REMOVAL_CANDIDATE_BOOST`,
//! `calculate_removal_savings`, or the silent `boosted_savings <= contribution`
//! drop — is *expected* to break these assertions. That is the point: the fix
//! shows up as a test diff. When it does, update the pin **and**
//! `docs/analysis/remove-neuron-reachability-1785.md` together.
//!
//! Run with:
//! `cargo test --test issue_1785_remove_neuron_reachability -- --nocapture --test-threads=1`

use std::path::{Path, PathBuf};

use neat_ai_discovery::analysis::candidate_aggregation::apply_final_coordinated_gain_floor;
use neat_ai_discovery::analysis::constants::{
    REMOVAL_CANDIDATE_BOOST, REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS,
    coordinated_post_discount_noise_floor, removal_net_gain_floor, remove_low_impact_noise_floor,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_BELOW_EXPECTED_GAIN_FLOOR, REJECTION_REMOVAL_BELOW_NOISE_FLOOR,
    REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING,
};
use neat_ai_discovery::analysis::discovery_dispatch::apply_honest_remove_neuron_gain;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::estimate_remove_neuron_gain;
use neat_ai_discovery::analysis::remove_neuron_constant_promotion::{
    bias_folded_constant_neuron_uuids, functionally_constant_neuron_uuids,
    promote_constant_remove_neuron_candidates,
};
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::focus::{DEFAULT_COST_OF_GROWTH, SynapseCounts, calculate_removal_savings};
use neat_ai_discovery::rank_focus_neurons_internal;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson,
};
use serde_json::{Value, json};
use serial_test::serial;

/// Appended to every characterisation assertion so a gate fix reads as an
/// expected pin break rather than a mystery regression.
const PIN_HINT: &str = "this pin is expected to break when a gate is fixed — update the pin and \
                        docs/analysis/remove-neuron-reachability-1785.md";

/// The production cost-of-growth (NEAT-AI's `Score.ts` value), the setting every
/// number in #1785 was measured under.
const COST_OF_GROWTH: f32 = 1e-7;

/// Env vars that would move the gates under test. Both are unset for the whole
/// suite so the pinned numbers are the shipped defaults, never a local override.
const NOISE_FLOOR_ENV: &str = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
const COORDINATED_MULTIPLIER_ENV: &str = "NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER";

/// A discovery parquet that cannot be opened — the focus path is structure-only
/// (Issue #1766), so a successful response also proves no record decode happened.
const MISSING_PARQUET: &str = "/nonexistent/issue-1810/discovery.parquet";

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

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/remove_neuron_reachability/network.json")
}

/// The committed production-shaped creature: 36 hidden neurons across four
/// feed-forward layers plus three orphans, maximum synapse degree 12.
fn fixture_creature() -> CreatureJson {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()))
}

fn hidden_uuids(creature: &CreatureJson) -> Vec<String> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect()
}

/// One sole-op `RemoveNeuron` candidate per hidden neuron — the maximal removal
/// candidate set the analysis path can offer for this creature.
fn sole_op_remove_neuron_candidates(
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    hidden_uuids(creature)
        .into_iter()
        .map(|uuid| CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid: uuid }],
            // The fabricated NEAT-AI #2483 placeholder the #1530 override replaces.
            expected_creature_score_gain: 0.17879,
            comment: None,
        })
        .collect()
}

/// The synthetic neuron Block 1b adds to the fixture: one hidden neuron wired
/// straight into `out-1` with a dominant weight, so removing it costs almost the
/// whole output.
const SYNTHETIC_HOT_NEURON: &str = "h-synthetic-hot";

/// The fixture plus [`SYNTHETIC_HOT_NEURON`], inserted before the outputs so the
/// creature stays forward-only.
fn creature_with_synthetic_high_influence_neuron() -> CreatureJson {
    let mut creature = fixture_creature();
    let first_output = creature
        .neurons
        .iter()
        .position(|n| n.neuron_type == "output")
        .expect("the fixture has output neurons");
    let mut hot = creature.neurons[first_output - 1].clone();
    hot.uuid = SYNTHETIC_HOT_NEURON.to_string();
    hot.neuron_type = "hidden".to_string();
    creature.neurons.insert(first_output, hot);

    let template = creature.synapses[0].clone();
    let mut inbound = template.clone();
    inbound.from_uuid = "in-0".to_string();
    inbound.to_uuid = SYNTHETIC_HOT_NEURON.to_string();
    inbound.weight = 1.0;
    let mut outbound = template;
    outbound.from_uuid = SYNTHETIC_HOT_NEURON.to_string();
    outbound.to_uuid = "out-1".to_string();
    outbound.weight = 1000.0;
    creature.synapses.push(inbound);
    creature.synapses.push(outbound);
    creature
}

/// The neuron a sole-op `RemoveNeuron` candidate targets.
fn removal_target(candidate: &CoordinatedStructuralCandidateJson) -> &str {
    match candidate.operations.as_slice() {
        [CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }] => neuron_uuid.as_str(),
        other => panic!("expected a sole-op RemoveNeuron candidate, got {other:?}"),
    }
}

/// Drive the FFI-facing final gain floor over `candidates` and return the
/// survivors plus the rejection breakdown the FFI response would carry.
fn final_floor_survivors(
    candidates: Vec<CoordinatedStructuralCandidateJson>,
) -> (Vec<CoordinatedStructuralCandidateJson>, Value) {
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
    let breakdown = serde_json::to_value(synapse.metadata.rejection_breakdown.counts())
        .expect("rejection breakdown serialises");
    (synapse.coordinated_structural_candidates, breakdown)
}

/// One reason's count from a serialised rejection breakdown (absent → 0).
fn breakdown_count(breakdown: &Value, reason: &str) -> usize {
    usize::try_from(breakdown[reason].as_u64().unwrap_or(0)).expect("count fits a usize")
}

/// Assert the fixture still exercises what this suite measures. Without these
/// the measurement blocks below could pass vacuously on a hollowed-out fixture.
fn assert_fixture_preconditions(creature: &CreatureJson) {
    let hidden = hidden_uuids(creature);
    assert!(
        hidden.len() >= 20,
        "the fixture must carry at least 20 hidden neurons to be production-shaped, found {}",
        hidden.len()
    );
    assert!(
        !sole_op_remove_neuron_candidates(creature).is_empty(),
        "the sole-op RemoveNeuron candidate set must be non-empty, or every measurement below is vacuous"
    );
    assert!(
        functionally_constant_neuron_uuids(creature).is_empty(),
        "the structural constant-neuron detector (#1813) flags nothing on this fixture — every \
         hidden neuron here has a non-zero-weight path from a live input; a non-empty set means \
         the promotion escape hatch changed — {PIN_HINT}"
    );
}

/// `min`, `median`, `max` of a non-empty sample, median by the lower midpoint.
fn distribution(values: &mut [f64]) -> (f64, f64, f64) {
    assert!(!values.is_empty(), "distribution of an empty sample");
    values.sort_by(f64::total_cmp);
    (
        values[0],
        values[(values.len() - 1) / 2],
        values[values.len() - 1],
    )
}

/// Drive the shipped focus path — `rank_focus_neurons_internal`, which calls
/// `focus::identify_structural_removal_candidates` (`src/ffi_internal/analysis.rs`).
fn ffi_focus_response(creature: &CreatureJson, cost_of_growth: f32) -> Value {
    let input = json!({
        "parquetFile": MISSING_PARQUET,
        "creature": creature,
        "maxResults": 256,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": cost_of_growth,
    })
    .to_string();
    let response: Value =
        serde_json::from_str(&rank_focus_neurons_internal(&input).expect("FFI focus path"))
            .expect("FFI response JSON");
    assert_eq!(
        response["success"], true,
        "the structure-only FFI focus path must succeed: {response:?}"
    );
    response
}

/// **Block 1 — Gate 1, the analysis path (post-#1812).**
///
/// Build the sole-op `RemoveNeuron` candidate set, apply the #1530/#1812
/// net-gain override, and drive the FFI-facing `apply_final_coordinated_gain_floor`
/// — the last stage before the FFI response, and the one that populates
/// `metadata.rejection_breakdown` and `metadata.candidates_returned`.
///
/// Pins the flip this milestone exists to make: the three zero-influence orphans
/// survive to the response **without** the #1622 promotion (nothing is flagged
/// here — Block 2 proves that), and the other 33 are rejected and counted under
/// `REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING`.
#[test]
#[serial]
fn gate_1_analysis_path_yields_surviving_remove_neuron_candidates() {
    let _gates = default_gates();
    let creature = fixture_creature();
    assert_fixture_preconditions(&creature);

    let mut candidates = sole_op_remove_neuron_candidates(&creature);
    let hidden_count = candidates.len();
    let removable = apply_honest_remove_neuron_gain(&creature, &mut candidates);

    let mut gains: Vec<f64> = candidates
        .iter()
        .map(|c| f64::from(c.expected_creature_score_gain))
        .collect();
    let (min, median, max) = distribution(&mut gains);
    let removal_floor = removal_net_gain_floor(COST_OF_GROWTH);
    let shared_floor = coordinated_post_discount_noise_floor(1);

    let (survivors, breakdown) = final_floor_survivors(candidates);
    let counted_rejections = breakdown_count(&breakdown, REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING);
    let shared_floor_rejections = breakdown_count(&breakdown, REJECTION_BELOW_EXPECTED_GAIN_FLOOR);

    println!("=== Block 1 — Gate 1 (analysis path, net remove-neuron gain) ===");
    println!("hidden neurons (candidate set): {hidden_count}");
    println!("removable neurons (gain overridden): {removable}");
    println!("net gain min:    {min:e}");
    println!("net gain median: {median:e}");
    println!("net gain max:    {max:e}");
    println!("removal_net_gain_floor({COST_OF_GROWTH:e}): {removal_floor:e}");
    println!("coordinated_post_discount_noise_floor(1): {shared_floor:e}");
    println!(
        "candidates surviving to the FFI response: {}",
        survivors.len()
    );
    println!(
        "rejections counted under {REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING}: {counted_rejections}"
    );

    assert_eq!(
        removable, hidden_count,
        "every hidden neuron must receive a net gain, or the distribution is measured over a \
         partly-placeholder set"
    );
    assert!(
        max > 0.0,
        "at least one removal must now net a positive benefit; best was {max:e} — {PIN_HINT}"
    );
    assert!(
        !survivors.is_empty(),
        "at least one non-promoted sole-op RemoveNeuron candidate must reach the FFI response — \
         {PIN_HINT}"
    );
    assert_eq!(
        survivors.len(),
        3,
        "the fixture's three zero-influence orphans are exactly the removals worth making — \
         {PIN_HINT}"
    );
    let mut surviving_uuids: Vec<&str> = survivors.iter().map(removal_target).collect();
    surviving_uuids.sort_unstable();
    assert_eq!(
        surviving_uuids,
        vec!["h-x-0", "h-x-1", "h-x-2"],
        "the survivors must be the zero-influence orphans, not a connected neuron — {PIN_HINT}"
    );
    assert_eq!(
        counted_rejections + survivors.len(),
        hidden_count,
        "every sole-op removal must leave the pass as a survivor or a counted rejection — a \
         silent drop would reproduce the exact fail-loud violation #1785 raised"
    );
    assert_eq!(
        shared_floor_rejections, 0,
        "a sole-op removal must be counted under its own reason, never under the shared \
         add-path floor reason — {PIN_HINT}"
    );
}

/// **Block 1b — Gate 1 still rejects a harmful removal.**
///
/// The same fixture plus one synthetic hidden neuron wired straight into an
/// output with a dominant weight, so it carries near-total downstream influence.
/// The rule must reject it — and count the rejection under a named reason —
/// while the zero-influence orphans still survive. Without this the fix could
/// have been "accept everything".
#[test]
#[serial]
fn gate_1_rejects_a_synthetic_high_influence_neuron() {
    let _gates = default_gates();
    let creature = creature_with_synthetic_high_influence_neuron();

    let influence = estimate_remove_neuron_gain(&creature, SYNTHETIC_HOT_NEURON)
        .expect("the synthetic neuron is a removal candidate");
    let mut candidates = sole_op_remove_neuron_candidates(&creature);
    apply_honest_remove_neuron_gain(&creature, &mut candidates);
    let hot_gain = candidates
        .iter()
        .find(|c| removal_target(c) == SYNTHETIC_HOT_NEURON)
        .expect("the synthetic neuron has a candidate")
        .expected_creature_score_gain;

    let (survivors, breakdown) = final_floor_survivors(candidates);
    let counted_rejections = breakdown_count(&breakdown, REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING);
    let surviving_uuids: Vec<&str> = survivors.iter().map(removal_target).collect();

    println!("=== Block 1b — Gate 1 rejects a synthetic high-influence neuron ===");
    println!("synthetic neuron: {SYNTHETIC_HOT_NEURON}");
    println!("estimate_remove_neuron_gain: {influence:e}");
    println!("net expectedCreatureScoreGain: {hot_gain:e}");
    println!(
        "removal_net_gain_floor: {:e}",
        removal_net_gain_floor(COST_OF_GROWTH)
    );
    println!("survivors: {surviving_uuids:?}");
    println!(
        "rejections counted under {REJECTION_REMOVAL_LOSS_EXCEEDS_SAVING}: {counted_rejections}"
    );

    assert!(
        influence <= -0.5,
        "the synthetic neuron must carry near-total downstream influence, got {influence:e} — \
         the fixture no longer exercises the harmful-removal case"
    );
    assert!(
        hot_gain < 0.0,
        "a high-influence removal must net a negative benefit, got {hot_gain:e} — {PIN_HINT}"
    );
    assert!(
        !surviving_uuids.contains(&SYNTHETIC_HOT_NEURON),
        "the synthetic high-influence neuron must never reach the FFI response — {PIN_HINT}"
    );
    assert!(
        counted_rejections > 0,
        "its rejection must be counted under a named reason, not dropped silently — {PIN_HINT}"
    );
    assert!(
        surviving_uuids.contains(&"h-x-0"),
        "the zero-influence orphans must still survive alongside the rejection — {PIN_HINT}"
    );
}

/// **Block 2 — the promotion escape hatch.**
///
/// The only route past Gate 1 is #1622 promotion. Its structural flag source
/// (`functionally_constant_neuron_uuids`, wired by #1813) flags nothing on this
/// fixture — every hidden neuron carries variance from a live input — and its
/// measured source (#1779 `bias_folded_constant_neuron_uuids`) needs a candidate
/// carrying an accepted bias fold, which the structure-only path never attaches.
/// So nothing is promoted for *this* creature, and the escape hatch yields zero.
#[test]
#[serial]
fn gate_1_promotion_escape_hatch_promotes_nothing() {
    let _gates = default_gates();
    let creature = fixture_creature();
    assert_fixture_preconditions(&creature);

    let mut candidates = sole_op_remove_neuron_candidates(&creature);
    apply_honest_remove_neuron_gain(&creature, &mut candidates);

    let structural_flags = functionally_constant_neuron_uuids(&creature);
    let fold_carrying = candidates
        .iter()
        .filter(|c| c.constant_neuron_bias_fold.is_some())
        .count();
    let fold_flags = bias_folded_constant_neuron_uuids(&candidates);
    let mut flags = structural_flags.clone();
    flags.extend(fold_flags.iter().cloned());
    let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flags);

    println!("=== Block 2 — promotion escape hatch (#1622 / #1779) ===");
    println!(
        "functionally_constant_neuron_uuids: {}",
        structural_flags.len()
    );
    println!("candidates carrying a constant_neuron_bias_fold: {fold_carrying}");
    println!("bias_folded_constant_neuron_uuids: {}", fold_flags.len());
    println!("candidates promoted: {promoted}");

    assert!(
        structural_flags.is_empty(),
        "no hidden neuron in this fixture is structurally constant (#1813) — {PIN_HINT}"
    );
    assert_eq!(
        fold_carrying, 0,
        "no candidate on this path carries a bias fold, so the live #1779 flag source is empty — \
         {PIN_HINT}"
    );
    assert_eq!(
        promoted, 0,
        "the escape hatch promotes nothing today, so Gate 1's zero yield stands — {PIN_HINT}"
    );
}

/// **Block 3 — Gate 2, the focus / FFI path (post-#1814).**
///
/// `identify_structural_removal_candidates(&creature, 1e-7)` reached through the
/// shipped FFI entry point. Measured: surviving candidates,
/// `noise_floor_rejections`, the hidden neurons dropped at the
/// `boosted_savings <= contribution` return, and the best boosted savings
/// available anywhere in the fixture.
///
/// **Updated by Issue #1814.** The original pin recorded a zero yield: the noise
/// floor was an absolute `1e-5` screening `boostedSavings − contribution`, a
/// term linear in `costOfGrowth`, so at `1e-7` the best boosted savings in the
/// whole fixture (`3.3e-7`) sat 30× below it and every neuron was rejected
/// regardless of contribution. The floor is now
/// `REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS × costOfGrowth`, so the fixture's
/// zero-contribution orphans reach the response and the noise floor stops being
/// the universal rejector.
#[test]
#[serial]
fn gate_2_focus_path_emits_the_zero_contribution_orphans() {
    let _gates = default_gates();
    let creature = fixture_creature();
    assert_fixture_preconditions(&creature);

    let response = ffi_focus_response(&creature, COST_OF_GROWTH);
    let surviving = response["removalCandidates"].as_array().map_or(0, Vec::len);
    let reported = response["rejectionBreakdown"][REJECTION_REMOVAL_BELOW_NOISE_FLOOR]
        .as_u64()
        .unwrap_or(0);
    let rejections = usize::try_from(reported).expect("rejection count fits a usize");

    // The `boosted_savings <= contribution` drop is a bare `return None` — it is
    // counted nowhere, so it is measured as the residue: every hidden neuron that
    // neither survived nor was reported as a noise-floor rejection.
    let hidden = hidden_uuids(&creature);
    let silent_drops = hidden.len() - surviving - rejections;

    // Best boosted savings anywhere in the fixture, from the shipped formula.
    let counts = SynapseCounts::new(&creature);
    let best_boosted = hidden
        .iter()
        .map(|uuid| {
            let (incoming, outgoing) = counts.get(uuid);
            calculate_removal_savings(incoming, outgoing, COST_OF_GROWTH) * REMOVAL_CANDIDATE_BOOST
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let floor = remove_low_impact_noise_floor(COST_OF_GROWTH);

    println!("=== Block 3 — Gate 2 (focus / FFI structural removal triage) ===");
    println!("hidden neurons considered: {}", hidden.len());
    println!("surviving candidates: {surviving}");
    println!("noise_floor_rejections (reported): {rejections}");
    println!("`boosted_savings <= contribution` drops: {silent_drops}");
    println!("best boosted savings across the fixture: {best_boosted:e}");
    println!("remove_low_impact_noise_floor({COST_OF_GROWTH:e}): {floor:e}");

    assert_eq!(
        surviving, 3,
        "the three zero-contribution orphans must survive the structural triage at \
         costOfGrowth={COST_OF_GROWTH:e} (Issue #1814) — {PIN_HINT}"
    );
    assert!(
        best_boosted >= floor,
        "the best boosted savings in the fixture ({best_boosted:e}) must now clear the \
         {floor:e} noise floor — the floor scales with costOfGrowth (Issue #1814) — {PIN_HINT}"
    );
    assert_eq!(
        rejections, 0,
        "the noise floor must no longer be the universal rejector: every remaining drop is a \
         genuine `boosted_savings <= contribution` verdict (Issue #1814) — {PIN_HINT}"
    );
    assert_eq!(
        silent_drops + rejections,
        hidden.len() - surviving,
        "every hidden neuron must be accounted for as a survivor, a reported rejection, or a \
         savings-below-impact drop — otherwise this block is measuring the wrong residue"
    );
    assert!(
        silent_drops > 0,
        "the `boosted_savings <= contribution` drop must still be exercised, or the fixture no \
         longer reaches it — {PIN_HINT}"
    );
}

/// **Block 4 — the 657-synapse break-even is gone, by construction.**
///
/// #1785 quoted 657 synapses as the point where a zero-contribution neuron's
/// boosted savings finally reached the old absolute `REMOVE_LOW_IMPACT_NOISE_FLOOR`.
/// Issue #1814 re-denominated the floor in units of `costOfGrowth`, so both
/// sides of the comparison are now linear in `costOfGrowth` and the degree
/// requirement disappears: `REMOVAL_CANDIDATE_BOOST` (1.5) already exceeds
/// `REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS` (1.0) at degree 0.
///
/// This searches the shipped `calculate_removal_savings` for the smallest degree
/// that clears the floor rather than quoting it, so the number stays checked.
#[test]
#[serial]
fn gate_2_break_even_needs_no_synapses_on_a_zero_contribution_neuron() {
    let _gates = default_gates();
    let floor = remove_low_impact_noise_floor(COST_OF_GROWTH);

    let boosted = |synapses: usize| {
        calculate_removal_savings(synapses, 0, COST_OF_GROWTH) * REMOVAL_CANDIDATE_BOOST
    };
    let break_even = (0..2_000)
        .find(|n| boosted(*n) >= floor)
        .expect("a break-even degree must exist below 2000 synapses");

    println!("=== Block 4 — noise-floor break-even (zero-contribution neuron) ===");
    println!("costOfGrowth: {COST_OF_GROWTH:e}");
    println!("REMOVAL_CANDIDATE_BOOST: {REMOVAL_CANDIDATE_BOOST}");
    println!("REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS: {REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS}");
    println!("remove_low_impact_noise_floor({COST_OF_GROWTH:e}): {floor:e}");
    println!("break-even synapse count: {break_even}");
    println!("boosted savings at {break_even}: {:e}", boosted(break_even));

    assert_eq!(
        break_even, 0,
        "a zero-contribution neuron must clear the floor at any degree once the floor is \
         denominated in costOfGrowth (Issue #1814) — {PIN_HINT}"
    );
    assert!(
        boosted(0) >= floor,
        "an orphan's boosted savings ({:e}) must clear the {floor:e} floor — {PIN_HINT}",
        boosted(0)
    );

    // The break-even is now invariant to costOfGrowth, which is the whole point:
    // the old absolute floor made it move by the same factor the host changed.
    for cost_of_growth in [1e-8_f32, 1e-7, 1e-6, 1e-4] {
        let floor = remove_low_impact_noise_floor(cost_of_growth);
        let boosted_at =
            |n: usize| calculate_removal_savings(n, 0, cost_of_growth) * REMOVAL_CANDIDATE_BOOST;
        let break_even_at = (0..2_000)
            .find(|n| boosted_at(*n) >= floor)
            .expect("a break-even degree must exist below 2000 synapses");
        assert_eq!(
            break_even_at, 0,
            "the break-even degree must not move with costOfGrowth {cost_of_growth:e} — {PIN_HINT}"
        );
    }

    assert_eq!(
        COST_OF_GROWTH, DEFAULT_COST_OF_GROWTH,
        "the break-even is quoted at the production default cost-of-growth"
    );
}
