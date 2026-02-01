//! Verifies that discovery type documentation is consistent (Issue #370).
//!
//! Ensures each discovery analysis module references `docs/DISCOVERY_TYPES.md`
//! and that `DISCOVERY_TYPES.md` documents each module.

use std::fs;
use std::path::Path;

/// Discovery analysis modules and the section heading they should reference
/// in `docs/DISCOVERY_TYPES.md`.
const DISCOVERY_MODULES: &[(&str, &str)] = &[
    ("src/analysis/saturation.rs", "Saturated Neuron Detection"),
    ("src/analysis/bottleneck.rs", "Bottleneck Neuron Detection"),
    ("src/analysis/dead_neuron.rs", "Dead Neuron Detection"),
    (
        "src/analysis/dormant_synapse.rs",
        "Dormant Synapse Detection",
    ),
    (
        "src/analysis/opposing_synapse.rs",
        "Opposing Synapse Detection",
    ),
    (
        "src/analysis/output_bias_drift.rs",
        "Output Bias Drift Detection",
    ),
    (
        "src/analysis/oscillating_neuron.rs",
        "Oscillating Neuron Detection",
    ),
    (
        "src/analysis/correlated_error.rs",
        "Correlated Error Pattern Detection",
    ),
    ("src/analysis/multi_hop.rs", "Multi-Hop Candidate Analysis"),
    ("src/analysis/redundant_path.rs", "Redundant Path Pruning"),
];

#[test]
fn each_discovery_module_references_discovery_types_md() {
    for (module_path, section_name) in DISCOVERY_MODULES {
        let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(module_path);
        let content = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {module_path}: {e}"));

        assert!(
            content.contains("docs/DISCOVERY_TYPES.md"),
            "{module_path} must reference docs/DISCOVERY_TYPES.md"
        );
        assert!(
            content.contains(section_name),
            "{module_path} must reference section \"{section_name}\""
        );
    }
}

#[test]
fn discovery_types_md_documents_each_module() {
    let doc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/DISCOVERY_TYPES.md");
    let content = fs::read_to_string(&doc_path).expect("Failed to read docs/DISCOVERY_TYPES.md");

    for (module_path, section_name) in DISCOVERY_MODULES {
        // Check that the section heading exists
        let heading = format!("### {section_name}");
        assert!(
            content.contains(&heading),
            "docs/DISCOVERY_TYPES.md must contain heading \"{heading}\" for {module_path}"
        );

        // Check that the source module file name is referenced
        let file_name = Path::new(module_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            content.contains(file_name),
            "docs/DISCOVERY_TYPES.md must reference source file \"{file_name}\""
        );
    }
}

#[test]
fn readme_links_to_discovery_types_md() {
    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let content = fs::read_to_string(&readme_path).expect("Failed to read README.md");

    // README must link to DISCOVERY_TYPES.md
    assert!(
        content.contains("docs/DISCOVERY_TYPES.md"),
        "README.md must link to docs/DISCOVERY_TYPES.md"
    );

    // README must not contain detailed detection criteria (DRY)
    assert!(
        !content.contains("## Detection Criteria") && !content.contains("## Detection criteria"),
        "README.md must not contain detailed detection criteria sections — \
         these belong in docs/DISCOVERY_TYPES.md"
    );
}

#[test]
fn discovery_types_md_is_single_source_of_truth() {
    let doc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/DISCOVERY_TYPES.md");
    let content = fs::read_to_string(&doc_path).expect("Failed to read docs/DISCOVERY_TYPES.md");

    // Must contain detection criteria for the documented types
    assert!(
        content.contains("**Detection criteria**") || content.contains("**Detection method**"),
        "docs/DISCOVERY_TYPES.md must contain detection criteria or methods"
    );

    // Must contain recommended actions
    assert!(
        content.contains("**Recommended actions**"),
        "docs/DISCOVERY_TYPES.md must contain recommended actions"
    );

    // Must contain output format information
    assert!(
        content.contains("**Output**"),
        "docs/DISCOVERY_TYPES.md must contain output format information"
    );
}
