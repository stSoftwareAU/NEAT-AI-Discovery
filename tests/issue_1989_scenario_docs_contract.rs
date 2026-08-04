//! Issue #1989 — the per-scenario pages under `docs/discoveries/` must quote the
//! weight caps, score maps and formulas the shipped code actually applies.
//!
//! Issue #1938 swept `docs/DISCOVERY_TYPES.md` and `docs/ANALYSIS_DEEP_DIVE.md`
//! but left the scenario pages beneath them quoting pre-#888 caps, activations
//! that appear in no score map, and worked examples the shipped clamps make
//! impossible. Each test below proves one claim against the real code, then
//! asserts the page agrees with what that behaviour just proved.

mod common;

use common::{make_creature, neuron, record, synapse};
use neat_ai_discovery::CandidateNeuronJson;
use neat_ai_discovery::analysis::detection::input_sensitivity::{
    InputSensitivityConfig, detect_dominant_inputs,
};
use neat_ai_discovery::analysis::detection::low_impact_neuron::detect_low_impact_neurons;
use neat_ai_discovery::analysis::recommendation::activation_recommendation::{
    InputDistribution, InputDistributionClass, classify_activation_suitability,
};
use neat_ai_discovery::analysis::recommendation::sample_weighted::{
    SampleWeightedConfig, detect_high_error_neurons,
};
use neat_ai_discovery::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_outgoing_weight,
};
use neat_ai_discovery::analysis::utils::filter_candidates_to_sensible_ranges;
use neat_ai_discovery::types::DiscoverRecord;

const ADD_NEURON: &str = include_str!("../docs/discoveries/add-neuron.md");
const ADD_SYNAPSE: &str = include_str!("../docs/discoveries/add-synapse.md");
const ACTIVATION_RECOMMENDATION: &str =
    include_str!("../docs/discoveries/activation-recommendation.md");
const SAMPLE_WEIGHTED: &str = include_str!("../docs/discoveries/sample-weighted.md");
const INPUT_SENSITIVITY: &str = include_str!("../docs/discoveries/input-sensitivity.md");
const REMOVE_LOW_IMPACT: &str = include_str!("../docs/discoveries/remove-low-impact.md");
const DISCOVERY_TYPES: &str = include_str!("../docs/DISCOVERY_TYPES.md");

// ---------------------------------------------------------------------------
// Markdown helpers
// ---------------------------------------------------------------------------

/// Text of the markdown section introduced by `heading`, up to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("doc must contain the heading {heading:?}"));
    let level = heading.chars().filter(|c| *c == '#').count();
    let body = &doc[start + heading.len()..];
    body.match_indices("\n#")
        .find(|(idx, _)| body[idx + 1..].chars().take_while(|c| *c == '#').count() <= level)
        .map_or(body, |(idx, _)| &body[..idx])
}

/// The single line containing `needle`.
fn line_with<'a>(doc: &'a str, needle: &str) -> &'a str {
    doc.lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("doc must contain a line mentioning {needle:?}"))
}

/// Is `name` an activation-function token (`RELU`, `RELU6`, `HARD_TANH`)?
fn is_activation_token(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Every `NAME (0.85)` pair on `text`, as (activation, quoted score).
fn quoted_score_pairs(text: &str) -> Vec<(String, f32)> {
    let mut pairs = Vec::new();
    for (idx, _) in text.match_indices('(') {
        let rest = &text[idx + 1..];
        let Some(end) = rest.find(')') else { continue };
        let Ok(score) = rest[..end].parse::<f32>() else {
            continue;
        };
        let before = text[..idx].trim_end();
        let name: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
            .collect::<Vec<char>>()
            .into_iter()
            .rev()
            .collect();
        if is_activation_token(&name) {
            pairs.push((name, score));
        }
    }
    pairs
}

/// Every `| NAME | 0.90 |` table row in `text`, as (activation, quoted score).
fn quoted_table_scores(text: &str) -> Vec<(String, f32)> {
    text.lines()
        .filter_map(|line| {
            let mut cells = line.split('|').map(str::trim);
            cells.next()?;
            let name = cells.next()?.trim_matches('*').trim().to_string();
            let score = cells.next()?.trim_matches('*').trim().parse::<f32>().ok()?;
            is_activation_token(&name).then_some((name, score))
        })
        .collect()
}

/// Every synapse weight the page quotes as `w = ±X` or `weight ±X`.
fn quoted_synapse_weights(doc: &str) -> Vec<f32> {
    let mut weights = Vec::new();
    for marker in ["w = ", "weight +", "weight −", "weight -"] {
        for (idx, _) in doc.match_indices(marker) {
            let token: String = doc[idx + marker.len()..]
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '+' || *c == '-')
                .collect();
            if let Ok(value) = token.trim_start_matches('+').parse::<f32>() {
                // Reconstruct the sign the marker itself carried.
                let signed = if marker.ends_with('−') || marker.ends_with('-') {
                    -value.abs()
                } else {
                    value
                };
                weights.push(signed);
            }
        }
    }
    weights
}

/// The first signed decimal number in `text`.
fn first_number(text: &str) -> Option<f32> {
    let chars: Vec<char> = text.replace('−', "-").chars().collect();
    let mut idx = 0;
    while idx < chars.len() {
        let starts_number = chars[idx].is_ascii_digit()
            || ((chars[idx] == '-' || chars[idx] == '+')
                && chars.get(idx + 1).is_some_and(char::is_ascii_digit));
        if starts_number {
            let start = idx;
            idx += 1;
            while idx < chars.len() && (chars[idx].is_ascii_digit() || chars[idx] == '.') {
                idx += 1;
            }
            let token: String = chars[start..idx].iter().collect();
            if let Ok(value) = token
                .trim_end_matches('.')
                .trim_start_matches('+')
                .parse::<f32>()
            {
                return Some(value);
            }
            continue;
        }
        idx += 1;
    }
    None
}

/// The value a `**Label:** 1.23` bullet or a `| Label | 1.23 |` table row quotes.
fn labelled_value(text: &str, label: &str) -> f32 {
    let bullet = format!("**{label}:**");
    let cell = format!("| {label} |");
    let (line, marker) = text
        .lines()
        .find_map(|line| {
            if line.contains(&bullet) {
                Some((line, bullet.as_str()))
            } else if line.contains(&cell) {
                Some((line, cell.as_str()))
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("doc must quote a value for {label:?}"));
    let rest = line
        .split(marker)
        .nth(1)
        .expect("the label must be followed by a value");
    first_number(rest).unwrap_or_else(|| panic!("{label} must quote a number, got {rest:?}"))
}

/// An add-neuron candidate with the three range-filtered parameters supplied.
fn candidate(incoming: f32, outgoing: f32, bias: f32, squash: &str) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: incoming,
        outgoing_weight: outgoing,
        squash: squash.to_string(),
        bias,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 1.0e-5,
        expected_creature_score_gain: 1.0e-5,
        improved_count: 10,
        total_count: 20,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        target_saturation_factor: None,
        variant_key: None,
    }
}

/// A record carrying an explicit error value.
fn record_with_error(uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

// ---------------------------------------------------------------------------
// add-neuron.md — post-#888 sensible ranges
// ---------------------------------------------------------------------------

/// The parameter-constraint table must quote the post-#888 caps, not the
/// 20 / 0.1 / 10 bounds the shipped filter rejects outright.
#[test]
fn add_neuron_page_quotes_the_post_888_sensible_ranges() {
    assert_eq!(
        filter_candidates_to_sensible_ranges(vec![candidate(5.0, 0.01, 2.0, "RELU")]).len(),
        1,
        "the post-#888 bounds (5.0 / 0.01 / 2.0) must pass the shipped filter"
    );
    for stale in [
        candidate(20.0, 0.01, 2.0, "RELU"),
        candidate(5.0, 0.1, 2.0, "RELU"),
        candidate(5.0, 0.01, 10.0, "RELU"),
    ] {
        assert!(
            filter_candidates_to_sensible_ranges(vec![stale]).is_empty(),
            "the pre-#888 bounds (20 / 0.1 / 10) must be rejected by the shipped filter"
        );
    }

    let constraints = section(ADD_NEURON, "\n### 🔒 Parameter Constraints");
    for shipped in ["5.0", "0.01", "2.0"] {
        assert!(
            constraints.contains(shipped),
            "the constraint table must quote the shipped cap {shipped}"
        );
    }
    for stale in ["≤ 20", "≤ 10"] {
        assert!(
            !constraints.contains(stale),
            "the constraint table must not quote the pre-#888 bound {stale:?}"
        );
    }
    assert!(
        !ADD_NEURON.contains("±0.1"),
        "add-neuron.md must not quote the pre-#888 ±0.1 outgoing cap"
    );
    assert!(
        constraints.contains("MIN_WEIGHT_RATIO_NON_LINEAR"),
        "the constraint table must name the relaxed non-linear ratio (Issue #905)"
    );
}

/// The worked example must survive the sensible-range filter, otherwise the page
/// illustrates a candidate that can never be emitted.
#[test]
fn add_neuron_worked_example_survives_the_sensible_range_filter() {
    let example = section(ADD_NEURON, "\n## 📝 Example");
    let activation = line_with(example, "**Activation:**")
        .split("**Activation:**")
        .nth(1)
        .expect("the example must name an activation")
        .trim()
        .to_string();

    let documented = candidate(
        labelled_value(example, "Incoming weight"),
        labelled_value(example, "Outgoing weight"),
        labelled_value(example, "Bias"),
        &activation,
    );
    assert_eq!(
        filter_candidates_to_sensible_ranges(vec![documented]).len(),
        1,
        "the documented add-neuron example must pass filter_candidates_to_sensible_ranges"
    );
}

// ---------------------------------------------------------------------------
// add-synapse.md — every quoted weight must survive the shipped clamp
// ---------------------------------------------------------------------------

/// Least-squares synapse weights are clamped to ±`MAX_OUTGOING_WEIGHT`, so a
/// page quoting 0.04–0.08 illustrates candidates the code cannot emit.
#[test]
fn add_synapse_page_weights_survive_the_shipped_clamp() {
    let clamped = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0)
        .expect("a large least-squares weight must still yield a candidate");
    assert!(
        (clamped - MAX_OUTGOING_WEIGHT).abs() < f32::EPSILON,
        "the shipped clamp must pin an over-large weight to {MAX_OUTGOING_WEIGHT}"
    );

    let quoted = quoted_synapse_weights(ADD_SYNAPSE);
    assert!(
        !quoted.is_empty(),
        "add-synapse.md must quote at least one worked weight"
    );
    for weight in quoted {
        assert!(
            weight.abs() <= MAX_OUTGOING_WEIGHT,
            "quoted weight {weight} exceeds the shipped clamp ±{MAX_OUTGOING_WEIGHT}"
        );
    }
    assert!(
        !ADD_SYNAPSE.contains("±0.1"),
        "add-synapse.md must not quote the pre-#888 ±0.1 cap"
    );
    assert!(
        ADD_SYNAPSE.contains("MAX_OUTGOING_WEIGHT"),
        "add-synapse.md must cite the constant that sets the clamp"
    );
}

// ---------------------------------------------------------------------------
// activation-recommendation.md — score maps and worked arithmetic
// ---------------------------------------------------------------------------

/// A neutral distribution of `class`: no gradient-flow penalty applies, so the
/// scores are the raw per-class map.
fn neutral_distribution(class: InputDistributionClass) -> InputDistribution {
    InputDistribution {
        class,
        mean: 0.0,
        std_dev: 1.0,
        min: -1.0,
        max: 1.0,
        sparsity: 0.0,
        kurtosis: 3.0,
    }
}

/// Every `NAME (score)` pair on the detection flowchart must match the score
/// `classify_activation_suitability` actually assigns for that class.
#[test]
fn activation_page_score_pairs_match_the_shipped_score_maps() {
    let branches = [
        ("Sparse:", InputDistributionClass::Sparse),
        ("Bounded:", InputDistributionClass::Bounded),
        ("Bimodal:", InputDistributionClass::Bimodal),
        ("Gaussian:", InputDistributionClass::Gaussian),
        ("Uniform\"", InputDistributionClass::Uniform),
    ];

    for (label, class) in branches {
        let scores = classify_activation_suitability(&neutral_distribution(class));
        let line = line_with(ACTIVATION_RECOMMENDATION, label);
        let pairs = quoted_score_pairs(line);
        assert!(
            !pairs.is_empty(),
            "the {label} branch must quote at least one scored activation: {line}"
        );
        for (name, quoted) in pairs {
            let actual = scores.get(&name).copied().unwrap_or_else(|| {
                panic!("{name} is quoted for {label} but appears in no score map")
            });
            assert!(
                (actual - quoted).abs() < 1.0e-6,
                "{label} quotes {name} ({quoted}) but the shipped map scores it {actual}"
            );
        }
    }
}

/// The worked example must be reproducible: every quoted score, the improvement
/// delta, and the expected improvement all come from the shipped code.
#[test]
fn activation_page_worked_example_matches_the_shipped_scores() {
    let example = section(ACTIVATION_RECOMMENDATION, "\n## 📝 Example");
    let distribution = InputDistribution {
        class: InputDistributionClass::Gaussian,
        mean: labelled_value(example, "Mean"),
        std_dev: labelled_value(example, "Std dev"),
        min: labelled_value(example, "Min"),
        max: labelled_value(example, "Max"),
        sparsity: 0.0,
        kurtosis: labelled_value(example, "Kurtosis"),
    };
    let scores = classify_activation_suitability(&distribution);

    let quoted = quoted_table_scores(example);
    assert!(
        quoted.len() >= 3,
        "the example must tabulate the suitability scores it reasons from"
    );
    for (name, score) in &quoted {
        let actual = scores
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("{name} is tabulated but appears in no Gaussian score map"));
        assert!(
            (actual - score).abs() < 1.0e-6,
            "the example quotes {name} = {score} but the shipped map scores it {actual}"
        );
    }

    // The candidate is TANH replacing the penalised RELU; both the delta and the
    // ×0.02 expected improvement must appear as the code computes them.
    let best = scores["TANH"];
    let current = scores["RELU"];
    let delta = best - current;
    assert!(
        example.contains(&format!("{delta:.3}")),
        "the example must quote the improvement delta {delta:.3}"
    );
    let expected_improvement = delta * 0.02;
    assert!(
        example.contains(&format!("{expected_improvement:.4}")),
        "the example must quote the expected improvement {expected_improvement:.4}"
    );
}

// ---------------------------------------------------------------------------
// sample-weighted.md — the hard-to-easy ratio is clamped at 10
// ---------------------------------------------------------------------------

/// `estimated_improvement` clamps the hard-to-easy ratio at 10 before scaling,
/// so the page must state the clamp and its example must respect it.
#[test]
fn sample_weighted_page_states_the_ratio_clamp() {
    // 100 easy samples at 0.04 and 100 hard at 0.72 → hard-to-easy ratio 18.
    let mut records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record_with_error("hidden-1", i, 0.5, 0.04))
        .collect();
    records.extend((100..200).map(|i| record_with_error("hidden-1", i, 0.5, 0.72)));

    let found = detect_high_error_neurons(
        &[("hidden-1".to_string(), records)],
        &SampleWeightedConfig::default(),
    );
    let candidate = found
        .first()
        .expect("a neuron with a 0.68 weighted mean error must be detected");
    assert!(
        candidate.hard_to_easy_ratio > 10.0,
        "the fixture must exercise a ratio above the clamp, got {}",
        candidate.hard_to_easy_ratio
    );

    let clamped = (candidate.weighted_mean_error * 10.0 * 0.01).min(0.1);
    let unclamped = (candidate.weighted_mean_error * candidate.hard_to_easy_ratio * 0.01).min(0.1);
    assert!(
        (candidate.estimated_improvement - clamped).abs() < 1.0e-6,
        "estimated_improvement must clamp the ratio at 10: expected {clamped}, got {}",
        candidate.estimated_improvement
    );
    assert!(
        (clamped - unclamped).abs() > 1.0e-6,
        "the fixture must distinguish the clamped from the unclamped formula"
    );

    let fix = section(SAMPLE_WEIGHTED, "\n## 🛠️ How We Fix It");
    assert!(
        fix.contains("min(hard_to_easy_ratio, 10)"),
        "the improvement formula must show the ratio clamp"
    );

    // The example's own inputs must be run through the clamped formula.
    let example = section(SAMPLE_WEIGHTED, "\n## 📝 Example");
    let ratio = labelled_value(example, "Hard-to-easy ratio");
    let weighted_mean = labelled_value(example, "Weighted mean error");
    let documented = (weighted_mean * ratio.min(10.0) * 0.01).min(0.1);
    assert!(
        example.contains(&format!("{documented:.3}")),
        "the example must quote the clamped estimated improvement {documented:.3}"
    );
}

// ---------------------------------------------------------------------------
// input-sensitivity.md — the setWeight recommendation scales the weight
// ---------------------------------------------------------------------------

/// The dominant-input fix scales the *existing* weight by
/// `min(dominance_threshold × 0.8 / sensitivity, WEIGHT_REDUCTION_FACTOR)`; it
/// does not set the weight to `dominance_threshold × 0.8`.
#[test]
fn input_sensitivity_page_states_the_real_weight_scaling() {
    let creature = make_creature(
        vec![
            neuron("input-7", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-7", "output-1", 3.2)],
    );
    // Activation and error move together, so correlation is 1.0 and the
    // variance ratio is 1.0 — sensitivity is then weight² = 10.24.
    let ramp: Vec<f32> = std::iter::successors(Some(0.0f32), |v| Some(v + 0.1))
        .take(40)
        .collect();
    let inputs: Vec<DiscoverRecord> = (0..40u32)
        .zip(&ramp)
        .map(|(i, &v)| record_with_error("input-7", i, v, v))
        .collect();
    let outputs: Vec<DiscoverRecord> = (0..40u32)
        .zip(&ramp)
        .map(|(i, &v)| record_with_error("output-1", i, 0.5, v))
        .collect();

    let config = InputSensitivityConfig::default();
    let found = detect_dominant_inputs(
        &creature,
        &[
            ("input-7".to_string(), inputs),
            ("output-1".to_string(), outputs),
        ],
        &config,
    );
    let candidate = found
        .first()
        .expect("a weight-3.2 input perfectly correlated with the error must be flagged");

    let target = config.dominance_threshold * 0.8;
    let expected = candidate.current_weight * (target / candidate.sensitivity_score).min(0.3);
    assert!(
        (candidate.recommended_weight - expected).abs() < 1.0e-5,
        "recommended_weight must scale the existing weight: expected {expected}, got {}",
        candidate.recommended_weight
    );
    assert!(
        (candidate.recommended_weight - target).abs() > 1.0e-3,
        "the code must not set the weight to dominance_threshold × 0.8 ({target})"
    );

    let fix = section(INPUT_SENSITIVITY, "\n## 🛠️ How We Fix It");
    assert!(
        fix.contains("WEIGHT_REDUCTION_FACTOR"),
        "the fix table must name the constant that floors the scaling"
    );
    assert!(
        !fix.contains("Scale to dominance_threshold × 0.8"),
        "the fix table must not claim the weight is set to dominance_threshold × 0.8"
    );

    // The page's own worked example must be recomputed with the real expression.
    let example = section(INPUT_SENSITIVITY, "\n## 📝 Example");
    let weight = labelled_value(example, "Current weight");
    let sensitivity = labelled_value(example, "Sensitivity score");
    let documented = weight * (config.dominance_threshold * 0.8 / sensitivity).min(0.3);
    assert!(
        example.contains(&format!("{documented:.2}")),
        "the example must quote the recommended weight {documented:.2}"
    );
}

// ---------------------------------------------------------------------------
// remove-low-impact.md — the criterion lives in FOCUS_SELECTION.md §4.1
// ---------------------------------------------------------------------------

/// Low-impact detection does not compare impact against `costOfGrowth`: a
/// neuron whose activation dwarfs the 1e-7 default is still detected.
#[test]
fn remove_low_impact_page_delegates_the_criterion_to_focus_selection() {
    let creature = make_creature(
        vec![
            neuron("hidden-quiet", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-quiet", "output-1", 0.5)],
    );
    // 0.02 is five orders of magnitude above the 1e-7 costOfGrowth default, yet
    // the neuron is still a low-impact candidate.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-quiet", i, 0.02, Some(0.02)))
        .collect();
    assert_eq!(
        detect_low_impact_neurons(&creature, &[("hidden-quiet".to_string(), records)], None).len(),
        1,
        "detection must not gate on impact < costOfGrowth"
    );

    assert!(
        !REMOVE_LOW_IMPACT.contains("impact < costOfGrowth"),
        "the page must not restate the superseded impact < costOfGrowth criterion"
    );
    assert!(
        REMOVE_LOW_IMPACT.contains("FOCUS_SELECTION.md"),
        "the page must delegate the removal criterion to FOCUS_SELECTION.md §4.1"
    );
    assert!(
        REMOVE_LOW_IMPACT.contains("REMOVAL_CANDIDATE_BOOST"),
        "the page must name the boost the §4.1 criterion applies"
    );
}

/// `remove-low-impact` is not the highest success-rate type — `change-squash`
/// beats it in the same production table — so the page must not claim it is.
#[test]
fn remove_low_impact_page_drops_the_highest_success_rate_claim() {
    let rate = |row_label: &str| -> f32 {
        let row = line_with(DISCOVERY_TYPES, row_label);
        row.split('|')
            .map(str::trim)
            .find_map(|cell| cell.strip_suffix('%')?.parse::<f32>().ok())
            .unwrap_or_else(|| panic!("the {row_label} row must quote a success rate: {row}"))
    };
    assert!(
        rate("**change-squash**") > rate("**remove-low-impact**"),
        "the production table must still rank change-squash above remove-low-impact"
    );

    assert!(
        !REMOVE_LOW_IMPACT.contains("highest success-rate"),
        "the page must not claim remove-low-impact has the highest success rate"
    );
}
