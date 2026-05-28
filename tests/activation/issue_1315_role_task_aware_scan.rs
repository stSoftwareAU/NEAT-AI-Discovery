//! Issue #1315 — the activation/squash scan candidate set is role- and
//! task-aware.
//!
//! Acceptance criteria:
//! - Under a OneHot/Simplex descriptor, the scan offers bounded squashes
//!   for **output** neurons and the existing set for **hidden** neurons.
//! - `OTHER` / `Unknown` / absent ⇒ current scan set (regression guard).

use neat_ai_discovery::analysis::activation::{
    ACTIVATION_SPECS, BOUNDED_OUTPUT_SCAN_NAMES, NeuronRole, scan_specs_for_role,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;

fn spec_names(
    specs: &[&'static neat_ai_discovery::analysis::activation::ActivationCandidateSpec],
) -> Vec<&'static str> {
    specs.iter().map(|s| s.name).collect()
}

#[test]
fn onehot_output_scan_is_bounded_only() {
    let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 5);
    let names = spec_names(&scan_specs_for_role(NeuronRole::Output, &d));

    // Bounded set members that exist in ACTIVATION_SPECS must be present.
    assert!(names.contains(&"LOGISTIC"));
    assert!(names.contains(&"BIPOLAR"));

    // Unbounded hidden-layer favourites must NOT appear.
    for unbounded in ["GELU", "ELU", "Mish", "Softplus", "IDENTITY", "ReLU6"] {
        assert!(
            !names.contains(&unbounded),
            "{unbounded} must be excluded from the OneHot output scan",
        );
    }

    // Every reported name belongs to the documented bounded-subset.
    for name in &names {
        assert!(BOUNDED_OUTPUT_SCAN_NAMES.contains(name));
    }
}

#[test]
fn simplex_output_scan_is_bounded_only() {
    let d = TaskDescriptor::from_name("CROSS_ENTROPY", 4);
    let names = spec_names(&scan_specs_for_role(NeuronRole::Output, &d));

    assert!(names.contains(&"LOGISTIC"));
    assert!(names.contains(&"BIPOLAR"));
    for unbounded in ["GELU", "ELU", "Mish"] {
        assert!(!names.contains(&unbounded));
    }
}

#[test]
fn onehot_hidden_scan_is_full_set() {
    let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 5);
    let specs = scan_specs_for_role(NeuronRole::Hidden, &d);
    assert_eq!(specs.len(), ACTIVATION_SPECS.len());
}

#[test]
fn simplex_hidden_scan_is_full_set() {
    let d = TaskDescriptor::from_name("CROSS_ENTROPY", 4);
    let specs = scan_specs_for_role(NeuronRole::Hidden, &d);
    assert_eq!(specs.len(), ACTIVATION_SPECS.len());
}

#[test]
fn other_cost_is_regression_safe_for_both_roles() {
    let d = TaskDescriptor::from_name("OTHER", 3);
    for role in [NeuronRole::Output, NeuronRole::Hidden] {
        let specs = scan_specs_for_role(role, &d);
        assert_eq!(
            specs.len(),
            ACTIVATION_SPECS.len(),
            "OTHER must preserve the full scan set for {role:?}",
        );
    }
}

#[test]
fn absent_descriptor_is_regression_safe_for_both_roles() {
    // Absent ≡ neutral().
    let d = TaskDescriptor::neutral();
    for role in [NeuronRole::Output, NeuronRole::Hidden] {
        let specs = scan_specs_for_role(role, &d);
        assert_eq!(specs.len(), ACTIVATION_SPECS.len());
    }
}

#[test]
fn independent_output_keeps_full_scan() {
    // MSE / MAE / MAPE / MSLE / BCE / HINGE all have target_topology that
    // is *not* OneHot or Simplex, so the scan must keep its full set
    // (Issue #1315 only narrows OneHot / Simplex output scans).
    for cost in [
        "MSE",
        "MAE",
        "MAPE",
        "MSLE",
        "BINARY_CROSS_ENTROPY",
        "HINGE",
    ] {
        let d = TaskDescriptor::from_name(cost, 3);
        let specs = scan_specs_for_role(NeuronRole::Output, &d);
        assert_eq!(
            specs.len(),
            ACTIVATION_SPECS.len(),
            "Cost {cost} must keep the full output scan",
        );
    }
}
