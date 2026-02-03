//! Issue #410: Verify that each discovery scenario has its own documentation
//! file with diagrams and references, and that the index and README link to them.
//!
//! These tests ensure the discovery scenario documentation is complete and
//! consistently cross-referenced.

use std::fs;
use std::path::Path;

/// Each discovery scenario document and the heading it must contain.
const DISCOVERY_SCENARIO_DOCS: &[(&str, &str)] = &[
    (
        "docs/discoveries/saturated-neuron.md",
        "Saturated Neuron Detection",
    ),
    (
        "docs/discoveries/bottleneck-neuron.md",
        "Bottleneck Neuron Detection",
    ),
    ("docs/discoveries/dead-neuron.md", "Dead Neuron Detection"),
    (
        "docs/discoveries/dormant-synapse.md",
        "Dormant Synapse Detection",
    ),
    (
        "docs/discoveries/opposing-synapse.md",
        "Opposing Synapse Detection",
    ),
    (
        "docs/discoveries/output-bias-drift.md",
        "Output Bias Drift Detection",
    ),
    (
        "docs/discoveries/oscillating-neuron.md",
        "Oscillating Neuron Detection",
    ),
    (
        "docs/discoveries/correlated-error.md",
        "Correlated Error Pattern Detection",
    ),
    (
        "docs/discoveries/multi-hop.md",
        "Multi-Hop Candidate Analysis",
    ),
    (
        "docs/discoveries/redundant-path.md",
        "Redundant Path Pruning",
    ),
    ("docs/discoveries/add-neuron.md", "Add Neuron Discovery"),
    ("docs/discoveries/add-synapse.md", "Add Synapse Discovery"),
    (
        "docs/discoveries/remove-low-impact.md",
        "Remove Low-Impact Neurons",
    ),
];

#[test]
fn each_discovery_scenario_doc_exists_with_heading() {
    for (doc_path, heading) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains(heading),
            "{doc_path} must contain heading \"{heading}\""
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_problem_section() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("## The Problem"),
            "{doc_path} must contain a '## The Problem' section explaining the scenario"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_detection_section() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("## How We Detect It"),
            "{doc_path} must contain a '## How We Detect It' section"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_fix_section() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("## How We Fix It"),
            "{doc_path} must contain a '## How We Fix It' section"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_example() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("## Example"),
            "{doc_path} must contain a '## Example' section"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_references() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("## References"),
            "{doc_path} must contain a '## References' section"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_has_diagrams() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        // Each doc should contain at least one ASCII diagram (code block with
        // box-drawing or arrow characters)
        assert!(
            content.contains("```") && (content.contains("──") || content.contains("→")),
            "{doc_path} must contain at least one ASCII diagram"
        );
    }
}

#[test]
fn each_discovery_scenario_doc_links_back_to_index() {
    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));

        assert!(
            content.contains("Back to Discovery Index"),
            "{doc_path} must link back to the discovery index"
        );
    }
}

#[test]
fn discovery_index_exists_and_links_to_all_scenarios() {
    let index_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/discoveries/README.md");
    let content =
        fs::read_to_string(&index_path).expect("Failed to read docs/discoveries/README.md");

    for (doc_path, _) in DISCOVERY_SCENARIO_DOCS {
        let file_name = Path::new(doc_path).file_name().unwrap().to_str().unwrap();
        assert!(
            content.contains(file_name),
            "docs/discoveries/README.md must link to {file_name}"
        );
    }
}

#[test]
fn discovery_index_contains_pipeline_overview() {
    let index_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/discoveries/README.md");
    let content =
        fs::read_to_string(&index_path).expect("Failed to read docs/discoveries/README.md");

    assert!(
        content.contains("## What Is Discovery?"),
        "Discovery index must contain a 'What Is Discovery?' section"
    );
    assert!(
        content.contains("## How It Works"),
        "Discovery index must contain a 'How It Works' section"
    );
    assert!(
        content.contains("## Discovery Scenario Index"),
        "Discovery index must contain a 'Discovery Scenario Index' section"
    );
}

#[test]
fn readme_links_to_discovery_scenarios_guide() {
    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let content = fs::read_to_string(&readme_path).expect("Failed to read README.md");

    assert!(
        content.contains("docs/discoveries/README.md"),
        "README.md must link to docs/discoveries/README.md"
    );
    assert!(
        content.contains("## Discovery Scenarios"),
        "README.md must contain a '## Discovery Scenarios' section"
    );
}
