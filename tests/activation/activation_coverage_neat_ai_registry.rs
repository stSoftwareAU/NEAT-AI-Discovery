//! Behavioural tests for activation function coverage (Issue #813).
//!
//! Verifies that every known scalar activation function produces finite,
//! sensible outputs and that aggregate squash names are handled correctly.
//! Converted from a file-reading cross-repo check to a behavioural test
//! that exercises the public API directly.

/// All known scalar squash names (including aliases).
const ALL_SCALAR_NAMES: &[&str] = &[
    "ABSOLUTE",
    "ARCTAN",
    "BENT_IDENTITY",
    "BIPOLAR",
    "BIPOLAR_SIGMOID",
    "COMPLEMENT",
    "INVERSE", // alias for COMPLEMENT
    "COSINE",
    "CUBE",
    "ELU",
    "EXPONENTIAL",
    "GAUSSIAN",
    "GELU",
    "HARD_TANH",
    "CLIPPED", // alias for HARD_TANH
    "IDENTITY",
    "ISRU",
    "LEAKYRELU",
    "LOGISTIC",
    "LOGSIGMOID",
    "MISH",
    "RELU",
    "RELU6",
    "SELU",
    "SINE",
    "SINUSOID", // alias for SINE
    "SOFTPLUS",
    "SOFTSIGN",
    "SQRT",
    "SQUARE",
    "STDINVERSE",
    "STEP",
    "SWISH",
    "TAN",
    "TANH",
];

/// All known aggregate squash names.
const ALL_AGGREGATE_NAMES: &[&str] = &["IF", "MAXIMUM", "MINIMUM", "MEAN", "HYPOT", "HYPOTV2"];

#[test]
fn all_scalar_squashes_are_recognised_and_computable() {
    for name in ALL_SCALAR_NAMES {
        assert!(
            neat_ai_discovery::activations::is_known_squash_name(name),
            "Scalar squash '{name}' must be recognised"
        );
        assert!(
            !neat_ai_discovery::activations::is_aggregate_squash(name),
            "Scalar squash '{name}' must not be flagged as aggregate"
        );

        let result = neat_ai_discovery::activations::apply_scalar_squash(name, 0.5);
        assert!(
            result.is_some(),
            "Scalar squash '{name}' must return Some from apply_scalar_squash"
        );
    }
}

#[test]
fn all_aggregate_squashes_are_recognised_but_not_scalar() {
    for name in ALL_AGGREGATE_NAMES {
        assert!(
            neat_ai_discovery::activations::is_known_squash_name(name),
            "Aggregate squash '{name}' must be recognised"
        );
        assert!(
            neat_ai_discovery::activations::is_aggregate_squash(name),
            "Aggregate squash '{name}' must be flagged as aggregate"
        );

        let result = neat_ai_discovery::activations::apply_scalar_squash(name, 0.5);
        assert!(
            result.is_none(),
            "Aggregate squash '{name}' must return None from apply_scalar_squash"
        );
    }
}

#[test]
fn scalar_squashes_produce_finite_results_for_typical_inputs() {
    let test_inputs: &[f32] = &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for name in ALL_SCALAR_NAMES {
        for &x in test_inputs {
            if let Some(y) = neat_ai_discovery::activations::apply_scalar_squash(name, x) {
                assert!(
                    y.is_finite(),
                    "apply_scalar_squash('{name}', {x}) returned non-finite {y}"
                );
            }
        }
    }
}

#[test]
fn identity_squash_returns_input_unchanged() {
    let inputs: &[f32] = &[-10.0, -1.0, 0.0, 0.5, 1.0, 42.0];
    for &x in inputs {
        let y = neat_ai_discovery::activations::apply_scalar_squash("IDENTITY", x).unwrap();
        assert!(
            (y - x).abs() < f32::EPSILON,
            "IDENTITY({x}) should be {x}, got {y}"
        );
    }
}

#[test]
fn relu_squash_clamps_negative_to_zero() {
    assert_eq!(
        neat_ai_discovery::activations::apply_scalar_squash("RELU", -5.0),
        Some(0.0)
    );
    assert_eq!(
        neat_ai_discovery::activations::apply_scalar_squash("RELU", 0.0),
        Some(0.0)
    );
    assert_eq!(
        neat_ai_discovery::activations::apply_scalar_squash("RELU", 3.0),
        Some(3.0)
    );
}

#[test]
fn alias_pairs_produce_identical_results() {
    let test_inputs: &[f32] = &[-1.0, 0.0, 0.5, 1.0];
    let alias_pairs: &[(&str, &str)] = &[
        ("INVERSE", "COMPLEMENT"),
        ("CLIPPED", "HARD_TANH"),
        ("SINUSOID", "SINE"),
    ];

    for (alias, canonical) in alias_pairs {
        for &x in test_inputs {
            let a = neat_ai_discovery::activations::apply_scalar_squash(alias, x);
            let b = neat_ai_discovery::activations::apply_scalar_squash(canonical, x);
            assert_eq!(a, b, "{alias}({x}) != {canonical}({x}): {a:?} vs {b:?}");
        }
    }
}

#[test]
fn case_insensitive_name_recognition() {
    // Names should be recognised regardless of case
    assert!(neat_ai_discovery::activations::is_known_squash_name("relu"));
    assert!(neat_ai_discovery::activations::is_known_squash_name("Relu"));
    assert!(neat_ai_discovery::activations::is_known_squash_name("RELU"));
    assert!(neat_ai_discovery::activations::is_known_squash_name(
        "Softplus"
    ));
    assert!(neat_ai_discovery::activations::is_known_squash_name("tanh"));
}

#[test]
fn unknown_name_is_not_recognised() {
    assert!(!neat_ai_discovery::activations::is_known_squash_name(
        "NOT_A_REAL_SQUASH"
    ));
    assert!(
        neat_ai_discovery::activations::apply_scalar_squash("NOT_A_REAL_SQUASH", 1.0).is_none()
    );
}
