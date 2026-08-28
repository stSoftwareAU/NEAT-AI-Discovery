//! Issue #2025 — the docs must map our vocabulary onto the published work it
//! implements.
//!
//! The detectors, the impact model and the candidate pipeline were documented
//! entirely in house vocabulary, so a reader could not tell a well-grounded
//! design from fifty heuristics. `docs/PRIOR_ART.md` is now the single home for
//! the bibliography, and the other docs cite into it:
//!
//!   1. Every `DISCOVERY_TYPES.md` summary row carries a `Prior art` cell.
//!   2. Every citation in those cells resolves to a bibliography entry (or is
//!      an explicit "No close precedent found").
//!   3. No detector was renamed while the column was added.
//!   4. Every bibliography entry is actually cited somewhere (no padding), and
//!      carries a title and a link.
//!   5. README + `ANALYSIS_DEEP_DIVE.md` name the surrogate-assisted-EA and
//!      attribution framings with citations.
//!   6. `IMPACT_CALCULATION.md` cites the attribution literature and states the
//!      Shapley-versus-discounting trade-off.
//!   7. `COST_FUNCTION_NOTES.md` documents the multiple-comparisons exposure
//!      alongside the existing SSE caveat.

const README: &str = include_str!("../README.md");
const DEEP_DIVE: &str = include_str!("../docs/ANALYSIS_DEEP_DIVE.md");
const DISCOVERY_TYPES: &str = include_str!("../docs/DISCOVERY_TYPES.md");
const IMPACT: &str = include_str!("../docs/IMPACT_CALCULATION.md");
const COST_NOTES: &str = include_str!("../docs/COST_FUNCTION_NOTES.md");
const PRIOR_ART: &str = include_str!("../docs/PRIOR_ART.md");

/// The literal used when a detector genuinely has no published precedent.
const NO_PRECEDENT: &str = "No close precedent found";

/// Every detector name that appears in the `DISCOVERY_TYPES.md` summary tables.
/// Acceptance criterion: "No detector is renamed" — this list is the gate.
const DETECTOR_NAMES: &[&str] = &[
    "Saturated Neuron",
    "Dead Neuron",
    "Oscillating Neuron",
    "Bimodal Neuron",
    "Restricted Range",
    "Operating Point",
    "Unbounded Capping",
    "Activation Mismatch",
    "Monotonicity",
    "Error Plateau",
    "Output Range Compression",
    "Output Squash Mismatch",
    "Activation Recommendation",
    "Bias Perturbation",
    "Squash + Weight Rescale",
    "High Error Squash Exploration",
    "Low-Impact Neuron",
    "Dormant Synapse",
    "Opposing Synapse",
    "Weight Coherence",
    "Weight Magnitude Reset",
    "Weight Polarity Flip",
    "Noise-to-Signal",
    "Fan-in Polarity Conflict",
    "Gradient Discovery",
    "Compound Degradation",
    "Bottleneck Neuron",
    "Correlated Error",
    "Redundant Path",
    "Topology Structure",
    "Topology Diversification",
    "Skip Connection",
    "Symmetry Breaking",
    "Co-Adaptation",
    "Merge Redundant Neuron",
    "Output Conflict",
    "Hard Sample Cluster",
    "Multi-Hop",
    "Combo Successful",
    "Fan-in Candidates",
    "Cross-Detection Synthesis",
    "Bounded Range",
    "Sentinel Gating",
    "Observation Utilisation",
    "Input Sensitivity",
    "Output Bias Drift",
    "Sample-Weighted",
    "Add Neurons",
    "Add Synapses",
    "Remove Low-Impact",
    "Remove Harmful Synapse",
    "Remove Neuron (Error)",
    "Batch-Successful Grouping",
];

/// Split a markdown table row into its trimmed cells.
fn cells(row: &str) -> Vec<&str> {
    let trimmed = row.trim().trim_start_matches('|').trim_end_matches('|');
    trimmed.split('|').map(str::trim).collect()
}

/// The `## 📋 Discovery Type Summary` block, up to the status legend.
fn summary_block() -> &'static str {
    let start = DISCOVERY_TYPES
        .find("## 📋 Discovery Type Summary")
        .expect("DISCOVERY_TYPES.md must carry the Discovery Type Summary section");
    let rest = &DISCOVERY_TYPES[start..];
    let end = rest
        .find("### 🏷️ Status Legend")
        .expect("the summary section must end at the status legend");
    &rest[..end]
}

/// Header rows of the summary tables (the lines naming `Discovery Type`).
fn summary_headers() -> Vec<&'static str> {
    summary_block()
        .lines()
        .filter(|l| l.starts_with("| Discovery Type"))
        .collect()
}

/// Data rows of the summary tables — one per detector.
fn summary_rows() -> Vec<&'static str> {
    summary_block()
        .lines()
        .filter(|l| l.starts_with("| ["))
        .collect()
}

/// The `## 📖 Bibliography` section, up to the next top-level heading.
fn bibliography_section() -> &'static str {
    let start = PRIOR_ART
        .find("## 📖 Bibliography")
        .expect("PRIOR_ART.md must carry a Bibliography section");
    let section = &PRIOR_ART[start..];
    let end = section[1..]
        .find("\n## ")
        .map_or(section.len(), |idx| idx + 1);
    &section[..end]
}

/// Citation keys from the `docs/PRIOR_ART.md` bibliography table.
fn bibliography() -> Vec<(&'static str, &'static str)> {
    bibliography_section()
        .lines()
        .filter(|l| l.starts_with("| ") && !l.starts_with("| Citation") && !l.starts_with("|---"))
        .map(|row| {
            let c = cells(row);
            assert!(
                c.len() >= 3,
                "bibliography rows are `| Citation | Work | Where it shows up |`: {row}"
            );
            (c[0], c[1])
        })
        .collect()
}

/// The prior-art citations in one summary row, with any trailing parenthetical
/// note stripped (`Wu et al. 2020 (Firefly)` → `Wu et al. 2020`).
fn citations(cell: &str) -> Vec<String> {
    cell.split(';')
        .map(|part| {
            let part = part.trim();
            match part.find(" (") {
                Some(i) => part[..i].trim().to_string(),
                None => part.to_string(),
            }
        })
        .filter(|part| !part.is_empty())
        .collect()
}

// ============================================================================
// 1–2. Every detector row carries a prior-art citation that resolves.
// ============================================================================

#[test]
fn every_summary_table_has_a_prior_art_column() {
    let headers = summary_headers();
    assert_eq!(
        headers.len(),
        5,
        "expected the five summary tables (activation, weight, structural, range, scoring)"
    );
    for header in headers {
        let c = cells(header);
        assert_eq!(
            c.last().copied(),
            Some("Prior art"),
            "each summary table must end with a `Prior art` column: {header}"
        );
    }
}

#[test]
fn every_detector_row_cites_prior_art_or_says_none_was_found() {
    let bib: Vec<&str> = bibliography().into_iter().map(|(key, _)| key).collect();
    assert!(
        bib.len() >= 20,
        "the bibliography must be populated, found {} entries",
        bib.len()
    );

    let rows = summary_rows();
    assert_eq!(
        rows.len(),
        DETECTOR_NAMES.len(),
        "every detector must appear exactly once in the summary tables"
    );

    for row in rows {
        let c = cells(row);
        assert_eq!(
            c.len(),
            6,
            "summary rows are `| Type | Module | Issue | Operations | Status | Prior art |`: {row}"
        );
        let prior_art = c[5];
        assert!(
            !prior_art.is_empty(),
            "the prior-art cell must not be blank: {row}"
        );

        for citation in citations(prior_art) {
            if citation.starts_with(NO_PRECEDENT) {
                continue;
            }
            assert!(
                bib.contains(&citation.as_str()),
                "`{citation}` is cited in DISCOVERY_TYPES.md but is not in the \
                 PRIOR_ART.md bibliography: {row}"
            );
        }
    }
}

#[test]
fn discovery_types_points_at_the_bibliography() {
    assert!(
        DISCOVERY_TYPES.contains("PRIOR_ART.md"),
        "DISCOVERY_TYPES.md must link to the prior-art bibliography"
    );
}

// ============================================================================
// 3. No detector was renamed while the column was added.
// ============================================================================

#[test]
fn no_detector_was_renamed() {
    let rows = summary_rows();
    for name in DETECTOR_NAMES {
        let needle = format!("| [{name}](");
        assert!(
            rows.iter().any(|row| row.starts_with(&needle)),
            "detector `{name}` must keep its name in the summary tables"
        );
    }
}

// ============================================================================
// 4. The bibliography is complete and free of padding.
// ============================================================================

#[test]
fn every_bibliography_entry_has_a_title_and_a_link() {
    for (citation, work) in bibliography() {
        assert!(
            citation.chars().any(|ch| ch.is_ascii_digit()),
            "bibliography citation `{citation}` must carry a year"
        );
        assert!(
            work.contains("](http"),
            "bibliography entry `{citation}` must link to the work: {work}"
        );
    }
}

#[test]
fn every_bibliography_entry_is_actually_cited() {
    let start = PRIOR_ART
        .find("## 📖 Bibliography")
        .expect("PRIOR_ART.md must carry a Bibliography section");
    let prose = &PRIOR_ART[..start];

    for (citation, _) in bibliography() {
        let cited_in_prose = prose.contains(citation);
        let cited_in_types = DISCOVERY_TYPES.contains(citation);
        let cited_elsewhere = IMPACT.contains(citation) || COST_NOTES.contains(citation);
        assert!(
            cited_in_prose || cited_in_types || cited_elsewhere,
            "bibliography entry `{citation}` is never cited — the bibliography \
             is a map, not a reading list"
        );
    }
}

// ============================================================================
// 5. The pipeline framings — README and the deep dive.
// ============================================================================

#[test]
fn readme_frames_the_pipeline_as_a_surrogate_assisted_ea() {
    for needle in [
        "surrogate-assisted evolutionary",
        "Jin 2011",
        "Jones et al. 1998",
        "expected improvement",
        "docs/PRIOR_ART.md",
    ] {
        assert!(
            README.contains(needle),
            "README.md must frame the pipeline with `{needle}`"
        );
    }
}

#[test]
fn deep_dive_frames_the_pipeline_and_the_attribution_model() {
    for needle in [
        "surrogate-assisted evolutionary",
        "Jin 2011",
        "Jones et al. 1998",
        "Bach et al. 2015",
        "PRIOR_ART.md",
    ] {
        assert!(
            DEEP_DIVE.contains(needle),
            "ANALYSIS_DEEP_DIVE.md must frame the pipeline with `{needle}`"
        );
    }
}

#[test]
fn prior_art_doc_is_listed_in_the_readme_documentation_index() {
    assert!(
        README.contains("[docs/PRIOR_ART.md](docs/PRIOR_ART.md)"),
        "README.md's documentation index must list docs/PRIOR_ART.md"
    );
}

// ============================================================================
// 6. Impact is an attribution measure — with the Shapley trade-off stated.
// ============================================================================

#[test]
fn impact_doc_cites_the_attribution_and_pruning_literature() {
    for needle in [
        "Bach et al. 2015",
        "Shrikumar et al. 2017",
        "LeCun et al. 1989",
        "Molchanov et al. 2017",
        "Lundberg & Lee 2017",
        "PRIOR_ART.md",
    ] {
        assert!(
            IMPACT.contains(needle),
            "IMPACT_CALCULATION.md must cite `{needle}`"
        );
    }
}

#[test]
fn impact_doc_states_the_shapley_versus_discounting_trade_off() {
    let start = IMPACT
        .find("## 🔗 Prior Art — Impact as an Attribution Measure")
        .expect("IMPACT_CALCULATION.md must carry the prior-art section");
    let section = &IMPACT[start..];
    let section = &section[..section[1..].find("\n## ").map_or(section.len(), |i| i + 1)];

    assert!(
        section.contains("Shapley"),
        "the prior-art section must name Shapley allocation as the exact treatment"
    );
    assert!(
        section.contains("discount"),
        "the prior-art section must explain why impact discounting exists"
    );
    assert!(
        section.contains("2^"),
        "the prior-art section must state what the exact (Shapley) version would \
         cost — the exponential coalition count"
    );
}

// ============================================================================
// 7. The multiple-comparisons exposure sits beside the SSE caveat.
// ============================================================================

#[test]
fn cost_notes_document_the_multiple_comparisons_exposure() {
    for needle in [
        "multiple-comparisons",
        "Dwork et al. 2015",
        "Blum & Hardt 2015",
        "ablation",
        "PRIOR_ART.md",
    ] {
        assert!(
            COST_NOTES.contains(needle),
            "COST_FUNCTION_NOTES.md must document the repeated-selection exposure: `{needle}`"
        );
    }
}

#[test]
fn cost_notes_state_what_is_done_about_repeated_selection() {
    let start = COST_NOTES
        .find("## 9. Repeated Selection on One Corpus")
        .expect("COST_FUNCTION_NOTES.md must carry the repeated-selection section");
    let section = &COST_NOTES[start..];
    let section = &section[..section[1..].find("\n## ").map_or(section.len(), |i| i + 1)];

    assert!(
        section.contains("What we do about it"),
        "the section must state the current mitigation, even if it is 'nothing yet'"
    );
    assert!(
        section.contains("same corpus"),
        "the section must name the reused-corpus exposure explicitly"
    );
}
