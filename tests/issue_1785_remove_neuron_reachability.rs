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
//!   neuron, run through `apply_honest_remove_neuron_gain` (#1530). The honest
//!   gain is `−impact`, so it is non-positive by construction while
//!   `coordinated_post_discount_noise_floor(1)` is strictly positive: **no**
//!   candidate can clear the floor.
//! - **Gate 2 — focus / FFI path.** `identify_structural_removal_candidates` is
//!   `pub(crate)`, so it is reached the way production reaches it — through
//!   `rank_focus_neurons_internal` (Issue #1806) — and the boosted savings from
//!   pruning a neuron are compared against its structural contribution.
//!
//! Plus the promotion escape hatch (#1622/#1779) and the 657-synapse break-even
//! that #1785 quotes, checked by construction rather than quoted.
//!
//! # This is a characterisation pin, not an invariant
//!
//! Every measurement below records **today's** behaviour. A fix to either gate —
//! a change to `coordinated_post_discount_noise_floor`,
//! `REMOVE_LOW_IMPACT_NOISE_FLOOR`, `REMOVAL_CANDIDATE_BOOST`,
//! `calculate_removal_savings`, or the silent `boosted_savings <= contribution`
//! drop — is *expected* to break these assertions. That is the point: the fix
//! shows up as a test diff. When it does, update the pin **and**
//! `docs/analysis/remove-neuron-reachability-1785.md` together.
//!
//! Run with:
//! `cargo test --test issue_1785_remove_neuron_reachability -- --nocapture --test-threads=1`

use std::path::{Path, PathBuf};

use neat_ai_discovery::analysis::constants::{
    REMOVAL_CANDIDATE_BOOST, coordinated_post_discount_noise_floor, remove_low_impact_noise_floor,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR;
use neat_ai_discovery::analysis::discovery_dispatch::apply_honest_remove_neuron_gain;
use neat_ai_discovery::analysis::remove_neuron_constant_promotion::{
    bias_folded_constant_neuron_uuids, functionally_constant_neuron_uuids,
    promote_constant_remove_neuron_candidates,
};
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
        "the structural constant-neuron detector is unwired on Develop; a non-empty set means \
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

/// **Block 1 — Gate 1, the analysis path.**
///
/// Build the sole-op `RemoveNeuron` candidate set, apply the #1530 honest-gain
/// override, and measure how many honest gains clear
/// `coordinated_post_discount_noise_floor(1)`. Today: none, and none can — the
/// honest gain is `−impact` (non-positive) against a strictly positive floor.
#[test]
#[serial]
fn gate_1_analysis_path_yields_no_floor_clearing_remove_neuron_candidate() {
    let _gates = default_gates();
    let creature = fixture_creature();
    assert_fixture_preconditions(&creature);

    let mut candidates = sole_op_remove_neuron_candidates(&creature);
    let removable = apply_honest_remove_neuron_gain(&creature, &mut candidates);

    let mut gains: Vec<f64> = candidates
        .iter()
        .map(|c| f64::from(c.expected_creature_score_gain))
        .collect();
    let (min, median, max) = distribution(&mut gains);
    let floor = coordinated_post_discount_noise_floor(1);
    let clearing = gains.iter().filter(|g| **g >= f64::from(floor)).count();

    println!("=== Block 1 — Gate 1 (analysis path, honest remove-neuron gain) ===");
    println!("hidden neurons (candidate set): {}", candidates.len());
    println!("removable neurons (gain overridden): {removable}");
    println!("honest gain min:    {min:e}");
    println!("honest gain median: {median:e}");
    println!("honest gain max:    {max:e}");
    println!("coordinated_post_discount_noise_floor(1): {floor:e}");
    println!("neurons clearing the floor: {clearing}");

    assert_eq!(
        removable,
        candidates.len(),
        "every hidden neuron must receive an honest gain, or the distribution is measured over a \
         partly-placeholder set"
    );
    assert!(
        max <= 0.0,
        "the honest gain is −impact and must stay non-positive; best was {max:e} — {PIN_HINT}"
    );
    assert_eq!(
        clearing, 0,
        "no remove-neuron candidate clears the {floor:e} floor today (best honest gain {max:e}) — \
         {PIN_HINT}"
    );
}

/// **Block 2 — the promotion escape hatch.**
///
/// The only route past Gate 1 is #1622 promotion. Its structural flag source
/// (`functionally_constant_neuron_uuids`) is an unwired seam returning an empty
/// set, and its live source (#1779 `bias_folded_constant_neuron_uuids`) needs a
/// candidate carrying an accepted bias fold — which the structure-only path
/// never attaches. So nothing is promoted, and the escape hatch yields zero.
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
        "the structural detector seam is unwired on Develop — {PIN_HINT}"
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

/// **Block 3 — Gate 2, the focus / FFI path.**
///
/// `identify_structural_removal_candidates(&creature, 1e-7)` reached through the
/// shipped FFI entry point. Measured: surviving candidates,
/// `noise_floor_rejections`, the hidden neurons that hit the **uncounted**
/// `boosted_savings <= contribution` return, and the best boosted savings
/// available anywhere in the fixture.
#[test]
#[serial]
fn gate_2_focus_path_yields_no_surviving_removal_candidate() {
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
    let floor = remove_low_impact_noise_floor();

    println!("=== Block 3 — Gate 2 (focus / FFI structural removal triage) ===");
    println!("hidden neurons considered: {}", hidden.len());
    println!("surviving candidates: {surviving}");
    println!("noise_floor_rejections (reported): {rejections}");
    println!("silent `boosted_savings <= contribution` drops (uncounted): {silent_drops}");
    println!("best boosted savings across the fixture: {best_boosted:e}");
    println!("REMOVE_LOW_IMPACT_NOISE_FLOOR: {floor:e}");

    assert_eq!(
        surviving, 0,
        "no hidden neuron survives the structural triage at costOfGrowth={COST_OF_GROWTH:e} — \
         {PIN_HINT}"
    );
    assert!(
        best_boosted < floor,
        "the best boosted savings in the fixture ({best_boosted:e}) still sit below the \
         {floor:e} noise floor, so even a zero-contribution neuron is rejected — {PIN_HINT}"
    );
    assert_eq!(
        silent_drops + rejections,
        hidden.len(),
        "every hidden neuron must be accounted for as a survivor, a reported rejection, or a \
         silent drop — otherwise this block is measuring the wrong residue"
    );
    assert!(
        silent_drops > 0,
        "the uncounted drop at the `boosted_savings <= contribution` return must still be \
         exercised, or the fixture no longer reaches it — {PIN_HINT}"
    );
}

/// **Block 4 — the 657-synapse break-even, by construction.**
///
/// #1785 quotes 657 synapses as the point where a zero-contribution neuron's
/// boosted savings finally reach `REMOVE_LOW_IMPACT_NOISE_FLOOR`. Rather than
/// quote it, this searches the shipped `calculate_removal_savings` for the
/// smallest degree that clears the floor, so the number is checked.
#[test]
#[serial]
fn gate_2_break_even_needs_657_synapses_on_a_zero_contribution_neuron() {
    let _gates = default_gates();
    let floor = remove_low_impact_noise_floor();

    let boosted = |synapses: usize| {
        calculate_removal_savings(synapses, 0, COST_OF_GROWTH) * REMOVAL_CANDIDATE_BOOST
    };
    let break_even = (0..2_000)
        .find(|n| boosted(*n) >= floor)
        .expect("a break-even degree must exist below 2000 synapses");

    println!("=== Block 4 — noise-floor break-even (zero-contribution neuron) ===");
    println!("costOfGrowth: {COST_OF_GROWTH:e}");
    println!("REMOVAL_CANDIDATE_BOOST: {REMOVAL_CANDIDATE_BOOST}");
    println!("REMOVE_LOW_IMPACT_NOISE_FLOOR: {floor:e}");
    println!("break-even synapse count: {break_even}");
    println!("boosted savings at {break_even}: {:e}", boosted(break_even));
    println!(
        "boosted savings at {}: {:e}",
        break_even - 1,
        boosted(break_even - 1)
    );

    assert_eq!(
        break_even, 657,
        "the shipped constants put the break-even at 657 synapses — {PIN_HINT}"
    );
    assert!(
        boosted(656) < floor,
        "656 synapses must still fall short ({:e} < {floor:e}) — {PIN_HINT}",
        boosted(656)
    );
    assert_eq!(
        COST_OF_GROWTH, DEFAULT_COST_OF_GROWTH,
        "the break-even is quoted at the production default cost-of-growth"
    );
}
