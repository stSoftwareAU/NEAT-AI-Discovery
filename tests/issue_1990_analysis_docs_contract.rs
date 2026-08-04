//! Issue #1990 — point-in-time studies and audits must not state superseded
//! facts as current.
//!
//! `docs/archive/README.md` mandates a dated "as at" header plus inline
//! supersession notes for every `docs/analysis/*.md` study, but that convention
//! was enforced for only two files. These tests enforce it for **every** study
//! and for the two `docs/`-level audits, so the convention is checked by CI
//! rather than re-audited by hand.
//!
//! Each supersession test first proves the *current* behaviour with the real
//! code, then asserts the study's prose names the issue that closed the claim.

use std::path::{Path, PathBuf};

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::constants::{
    SYNAPSE_PESSIMISM_CURVE_EXPONENT, SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
    SYNAPSE_PREDICTION_CALIBRATION,
};
use neat_ai_discovery::analysis::detection::dormant_synapse::detect_dormant_synapses;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::dominated_branch_collapse::detect_dominated_branches;
use neat_ai_discovery::analysis::evaluation_drops::EvaluationDropCounters;
use neat_ai_discovery::analysis::fingerprint_skip_escape::{
    DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS, should_bypass_fingerprint_cache,
};
use neat_ai_discovery::types::DiscoverRecord;

/// The two `docs/`-level audits that are point-in-time reports, not live
/// reference pages, and so carry the same obligation as `docs/analysis/*.md`.
const AUDIT_DOCS: [&str; 2] = [
    "docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md",
    "docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("must read {}: {e}", path.display()))
}

/// Every study under `docs/analysis/`, excluding the index, sorted by name.
fn analysis_studies() -> Vec<String> {
    let dir = repo_root().join("docs/analysis");
    let mut studies: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("must read {}: {e}", dir.display()))
        .map(|entry| entry.expect("directory entry must be readable").path())
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .filter(|p| p.file_name().is_some_and(|n| n != "README.md"))
        .map(|p| format!("docs/analysis/{}", file_name(&p)))
        .collect();
    studies.sort();
    assert!(
        studies.len() >= 13,
        "the analysis directory must still hold every committed study, found {}",
        studies.len()
    );
    studies
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .expect("path must have a file name")
        .to_string_lossy()
        .into_owned()
}

/// Whether `text` carries a dated "as at YYYY-MM-DD" marker.
fn has_dated_as_at(text: &str) -> bool {
    text.match_indices("as at ").any(|(idx, _)| {
        let rest = &text[idx + "as at ".len()..];
        let date: Vec<char> = rest.chars().take(10).collect();
        date.len() == 10
            && date[..4].iter().all(char::is_ascii_digit)
            && date[4] == '-'
            && date[5..7].iter().all(char::is_ascii_digit)
            && date[7] == '-'
            && date[8..].iter().all(char::is_ascii_digit)
    })
}

/// The document's header block — everything before the first `##` heading.
fn header(doc: &str) -> &str {
    doc.find("\n## ").map_or(doc, |idx| &doc[..idx])
}

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

/// The convention itself: every point-in-time document carries a dated "as at"
/// header, in its header block where a reader meets it before any finding.
#[test]
fn every_point_in_time_document_carries_a_dated_as_at_header() {
    let docs: Vec<String> = analysis_studies()
        .into_iter()
        .chain(AUDIT_DOCS.iter().map(ToString::to_string))
        .collect();

    for rel in docs {
        let doc = read(&rel);
        assert!(
            has_dated_as_at(header(&doc)),
            "{rel} must carry a dated 'as at YYYY-MM-DD' header before its first '##' section \
             — see docs/archive/README.md § What goes where"
        );
    }
}

/// The index must reach every study, and the root README must reach the index —
/// eleven of thirteen studies were previously unreachable from the README.
#[test]
fn the_analysis_index_lists_every_study_and_the_readme_links_it() {
    let index = read("docs/analysis/README.md");
    for rel in analysis_studies() {
        let name = file_name(Path::new(&rel));
        assert!(
            index.contains(&name),
            "docs/analysis/README.md must index {name}"
        );
    }
    for audit in AUDIT_DOCS {
        assert!(
            index.contains(audit) || index.contains(&file_name(Path::new(audit))),
            "docs/analysis/README.md must cross-reference the {audit} audit"
        );
    }

    assert!(
        read("README.md").contains("docs/analysis/README.md"),
        "the root README documentation table must link the analysis index"
    );
}

/// #1923 resolved the activation-weighted removal gate, so the #1920 study's
/// "hard-coded to 0.0 … the gate never runs" finding is history. Proved by the
/// resolver module the study's finding A asked for.
#[test]
fn the_1920_study_annotates_the_resolved_activation_gate() {
    // The resolver exists and is reachable — the deferral the study found is
    // no longer permanent.
    let resolver = read("src/focus/ranking/activation_weighting.rs");
    assert!(
        resolver.contains("candidate.mean_activation = mean_activation"),
        "the #1923 resolver must write the measured mean activation back"
    );

    let doc = read("docs/analysis/candidates-cache-study-1920.md");
    assert!(
        has_dated_as_at(header(&doc)) && doc.contains("#1923"),
        "the study must be dated and name #1923 as the issue that closed finding A"
    );
    let finding_a = section(&doc, "**A. The dominant strategy has no gain ranking.**");
    assert!(
        finding_a.contains("Superseded") && finding_a.contains("#1923"),
        "finding A must carry an inline Superseded annotation naming #1923"
    );
    let follow_ups = section(&doc, "## Follow-ups");
    for issue in ["#1923", "#1924", "#1925"] {
        assert!(
            follow_ups.contains(issue),
            "the follow-up table must still name {issue}"
        );
    }
    assert!(
        follow_ups.contains("closed") || follow_ups.contains("shipped"),
        "the follow-up table must record that its three follow-ups have shipped"
    );
}

/// #1802 wired `below_improved_ratio` into the breakdown, so the #1737
/// "never incremented" claim is closed — but the `interference_filtered` half
/// still holds and must be kept explicitly.
#[test]
fn the_1737_diagnosis_annotates_the_wired_below_improved_ratio_counter() {
    let counters = EvaluationDropCounters::default();
    assert_eq!(counters.below_improved_ratio(), 0, "fixture starts at zero");
    counters.drop_below_improved_ratio();
    assert_eq!(
        counters.below_improved_ratio(),
        1,
        "the counter increments today — the #1737 'never incremented' claim is false"
    );

    let doc = read("docs/analysis/rejection-diagnosis-1737.md");
    assert!(has_dated_as_at(header(&doc)), "the study must be dated");
    assert!(
        doc.contains("Superseded") && doc.contains("#1802"),
        "the below_improved_ratio claim must be annotated with #1802"
    );
    assert!(
        doc.contains("interference_filtered"),
        "the still-true interference_filtered half must be kept explicitly"
    );
    assert!(
        doc.contains("threshold-review-1740.md"),
        "the study must cross-link the opposite floor verdict in #1740"
    );
}

/// #1778 rescaled the gain-floor screen and #1812 carved sole-op `RemoveNeuron`
/// out of the 1-op floor, so #1740's "no floor change" verdict is history.
#[test]
fn the_1740_threshold_review_annotates_the_rescaled_floor() {
    let constants = read("src/analysis/constants/candidate_scoring.rs");
    assert!(
        constants.contains("GAIN_FLOOR_NOISE_BACKSTOP"),
        "the #1778 rescale must still be in the shipped constants"
    );

    let doc = read("docs/analysis/threshold-review-1740.md");
    assert!(has_dated_as_at(header(&doc)), "the review must be dated");
    assert!(
        doc.contains("Superseded"),
        "the 'floors are correctly scaled' verdict must be annotated as superseded"
    );
    for issue in ["#1778", "#1812"] {
        assert!(
            doc.contains(issue),
            "the review must name {issue} as an issue that changed the floors"
        );
    }
    assert!(
        doc.contains("rejection-diagnosis-1737.md"),
        "the review must cross-link the opposite verdict in #1737"
    );
}

/// #1632 moved dormancy onto contribution, so #1631's `|weight| > 1e-4` skip
/// describes the pre-#1632 detector. Proved by a large-weight synapse whose
/// source never activates: weight-magnitude gating would have skipped it.
#[test]
fn the_1631_snapshot_study_annotates_the_contribution_first_detector() {
    // Two branches feed the output, so the detector's fan-in guard passes.
    // `gated` carries a large weight (5.0, far above the retired 1e-4 skip) but
    // never activates; `live` carries a small weight and does activate.
    let creature: CreatureJson = serde_json::from_str(
        r#"{
            "input": 1,
            "output": 1,
            "neurons": [
                {"uuid": "gated", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "live", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "gated", "toUUID": "out-0", "weight": 5.0},
                {"fromUUID": "live", "toUUID": "out-0", "weight": 0.01}
            ]
        }"#,
    )
    .expect("fixture creature must parse");

    let record = |uuid: &str, obs: u32, activation: f32| DiscoverRecord {
        obs_index: obs,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![0.1],
    };
    let gated: Vec<DiscoverRecord> = (0..40).map(|i| record("gated", i, 0.0)).collect();
    let live: Vec<DiscoverRecord> = (0..40).map(|i| record("live", i, 1.0)).collect();
    let neuron_records = vec![("gated".to_string(), gated), ("live".to_string(), live)];

    let dormant = detect_dormant_synapses(&creature, &neuron_records);
    assert!(
        dormant.iter().any(|c| c.from_neuron_uuid == "gated"),
        "a |weight| = 5.0 synapse whose source never activates is dormant today \
         — the retired weight-magnitude skip would have missed it"
    );

    let doc = read("docs/analysis/snapshot-mining-1631.md");
    assert!(has_dated_as_at(header(&doc)), "the study must be dated");
    assert!(
        doc.contains("Superseded") && doc.contains("#1632"),
        "the weight-magnitude skip claim must be annotated with #1632"
    );
}

/// #1781 shipped the fingerprint-skip escape hatch, so the 1777 diagnosis'
/// "the `previous_neuron_fingerprints` half above stands as written" is false.
#[test]
fn the_1777_diagnosis_annotates_the_shipped_fingerprint_escape() {
    let log = DiscoveryOutcomeLog::from_outcomes(vec![false; 4]);
    assert!(
        should_bypass_fingerprint_cache(Some(&log), DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS),
        "the #1781 escape hatch releases the fingerprint cache during a drought"
    );

    let doc = read("docs/analysis/candidate-rate-diagnosis-1777.md");
    assert!(
        !doc.contains("half above stands as written"),
        "the fingerprint half no longer stands — #1781 shipped the escape hatch"
    );
    assert!(
        doc.contains("fingerprint_skip_escape"),
        "the diagnosis must point at the module that closed the claim"
    );
    // The two other rotted claims on the same path.
    assert!(
        doc.contains("#1813") && doc.contains("#1781"),
        "the empty-HashSet stub claim must be annotated with the #1813 fixpoint"
    );
    assert!(
        doc.contains("accepted_constant_bias_fold"),
        "the renamed gate must be named — `is_constant_neuron` no longer exists"
    );
}

/// The MCMC audit quoted retired constants as current. The values it states must
/// agree with the shipped constants, or be annotated with today's value.
#[test]
fn the_mcmc_audit_agrees_with_the_shipped_calibration_constants() {
    let doc = read("docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md");
    assert!(has_dated_as_at(header(&doc)), "the audit must be dated");

    for (label, value) in [
        (
            "synapse prediction calibration",
            SYNAPSE_PREDICTION_CALIBRATION,
        ),
        ("synapse pessimism floor", SYNAPSE_PESSIMISM_DISCOUNT_FLOOR),
        (
            "synapse pessimism exponent",
            SYNAPSE_PESSIMISM_CURVE_EXPONENT,
        ),
    ] {
        let rendered = format!("{value}");
        assert!(
            doc.contains(&rendered),
            "the audit must state today's {label} ({rendered}) beside the retired value"
        );
    }

    // `order_focus_targets` moved out of `orchestration.rs` (Issue #1428 era
    // deadline work); the audit must cite where it actually lives.
    assert!(
        doc.contains("utils/deadline.rs"),
        "the audit must attribute order_focus_targets to utils/deadline.rs"
    );
    assert!(
        read("src/analysis/utils/deadline.rs").contains("pub fn order_focus_targets"),
        "order_focus_targets must still live in utils/deadline.rs"
    );

    // §6.2 recommended an adaptive proposal distribution as future work; #1019
    // landed it, so the Postscript must say so.
    assert!(
        section(&doc, "## 8. Postscript").contains("#1019"),
        "the Postscript must record that #1019 landed the adaptive proposal distribution"
    );
}

/// #1711 delivered dominance detection, so the extent report's "no dominance
/// detection at all … extent of automatic collapse is zero" must be annotated.
#[test]
fn the_dominated_branch_report_annotates_the_shipped_detector() {
    let creature: CreatureJson = serde_json::from_str(&read(
        "tests/fixtures/dominated_branch_collapse/networks/maximum_aggregate.json",
    ))
    .expect("the #1705 MAXIMUM fixture must parse");
    assert_eq!(
        detect_dominated_branches(&creature).len(),
        1,
        "the engine detects the dominated ABSOLUTE branch today — the report's \
         'no dominance detection at all' is history"
    );

    let doc = read("docs/DOMINATED_BRANCH_COLLAPSE_EXTENT.md");
    assert!(has_dated_as_at(header(&doc)), "the report must be dated");
    assert!(
        doc.contains("Superseded") && doc.contains("#1711"),
        "the 'no dominance detection at all' claim must be annotated with #1711"
    );
    assert!(
        doc.contains("dominated_branch_collapse.rs"),
        "the report must name the module that delivered the detector"
    );
}
