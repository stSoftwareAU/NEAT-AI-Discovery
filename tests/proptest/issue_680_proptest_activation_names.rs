//! Property-based tests for activation name functions (Issue #680).
//!
//! Uses `proptest` to verify normalisation idempotency, case-insensitivity,
//! consistency between `apply_scalar_squash` and `target_simulation_fn`, and
//! aggregate squash handling.

use neat_ai_discovery::activations::{
    apply_scalar_squash, is_aggregate_squash, is_known_squash_name, normalise_squash_name,
    target_simulation_fn,
};
use proptest::prelude::*;

/// All known scalar (non-aggregate) squash names.
const SCALAR_SQUASH_NAMES: &[&str] = &[
    "ABSOLUTE",
    "ARCTAN",
    "BENT_IDENTITY",
    "BIPOLAR",
    "BIPOLAR_SIGMOID",
    "COMPLEMENT",
    "COSINE",
    "CUBE",
    "ELU",
    "EXPONENTIAL",
    "GAUSSIAN",
    "GELU",
    "HARD_TANH",
    "IDENTITY",
    "INVERSE",
    "ISRU",
    "LEAKYRELU",
    "LOGISTIC",
    "LOGSIGMOID",
    "MISH",
    "RELU",
    "RELU6",
    "SELU",
    "SINE",
    "SINUSOID",
    "SOFTPLUS",
    "SOFTSIGN",
    "SQRT",
    "SQUARE",
    "STDINVERSE",
    "STEP",
    "SWISH",
    "TAN",
    "TANH",
    "CLIPPED",
];

/// All known aggregate squash names.
const AGGREGATE_SQUASH_NAMES: &[&str] = &["IF", "MAXIMUM", "MINIMUM", "MEAN", "HYPOT", "HYPOTV2"];

// =============================================================================
// 1. normalise_squash_name Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Normalisation must be idempotent: normalise(normalise(s)) == normalise(s).
    #[test]
    fn normalise_idempotent(name in "[a-zA-Z_]{1,20}") {
        let first = normalise_squash_name(&name).to_string();
        let second = normalise_squash_name(&first).to_string();
        prop_assert_eq!(
            &first, &second,
            "Normalisation not idempotent: '{}' -> '{}' -> '{}'",
            name, first, second
        );
    }

    /// Normalised name must be all uppercase ASCII.
    #[test]
    fn normalise_produces_uppercase(name in "[a-zA-Z_]{1,20}") {
        let normalised = normalise_squash_name(&name);
        for ch in normalised.chars() {
            prop_assert!(
                !ch.is_ascii_lowercase(),
                "Normalised name '{}' contains lowercase char '{}'",
                normalised, ch
            );
        }
    }

    /// Normalisation must trim leading and trailing whitespace.
    #[test]
    fn normalise_trims_whitespace(
        name in "[a-zA-Z_]{1,10}",
        leading in " {0,5}",
        trailing in " {0,5}",
    ) {
        let padded = format!("{leading}{name}{trailing}");
        let normalised_padded = normalise_squash_name(&padded).to_string();
        let normalised_bare = normalise_squash_name(&name).to_string();
        prop_assert_eq!(
            &normalised_padded, &normalised_bare,
            "Whitespace should not affect normalisation"
        );
    }

    /// Empty string must normalise to empty string.
    #[test]
    fn normalise_empty(_dummy in 0u8..1) {
        let result = normalise_squash_name("");
        prop_assert_eq!(result.as_ref(), "");
    }
}

// =============================================================================
// 2. is_known_squash_name Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// is_known_squash_name must be case-insensitive.
    #[test]
    fn known_squash_case_insensitive(idx in 0..SCALAR_SQUASH_NAMES.len()) {
        let name = SCALAR_SQUASH_NAMES[idx];
        let lower = name.to_ascii_lowercase();
        let mixed = name
            .chars()
            .enumerate()
            .map(|(i, c)| if i % 2 == 0 { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
            .collect::<String>();

        prop_assert!(
            is_known_squash_name(name),
            "'{name}' should be known"
        );
        prop_assert!(
            is_known_squash_name(&lower),
            "'{lower}' (lowercase) should be known"
        );
        prop_assert!(
            is_known_squash_name(&mixed),
            "'{mixed}' (mixed case) should be known"
        );
    }

    /// Random garbage strings should not be recognised as known squash names.
    #[test]
    fn unknown_squash_not_recognised(name in "[xyz]{5,15}") {
        prop_assert!(
            !is_known_squash_name(&name),
            "'{name}' should not be a known squash name"
        );
    }

    /// All aggregate squash names must be recognised as known.
    #[test]
    fn aggregate_squash_names_known(idx in 0..AGGREGATE_SQUASH_NAMES.len()) {
        let name = AGGREGATE_SQUASH_NAMES[idx];
        prop_assert!(
            is_known_squash_name(name),
            "Aggregate '{name}' should be known"
        );
    }
}

// =============================================================================
// 3. is_aggregate_squash Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Scalar squash names must not be flagged as aggregate.
    #[test]
    fn scalar_not_aggregate(idx in 0..SCALAR_SQUASH_NAMES.len()) {
        let name = SCALAR_SQUASH_NAMES[idx];
        // INVERSE and CLIPPED are aliases; they are scalar, not aggregate.
        prop_assert!(
            !is_aggregate_squash(name),
            "Scalar '{name}' should not be aggregate"
        );
    }

    /// Aggregate squash names must be flagged as aggregate.
    #[test]
    fn aggregate_is_aggregate(idx in 0..AGGREGATE_SQUASH_NAMES.len()) {
        let name = AGGREGATE_SQUASH_NAMES[idx];
        prop_assert!(
            is_aggregate_squash(name),
            "Aggregate '{name}' should be flagged as aggregate"
        );
    }
}

// =============================================================================
// 4. target_simulation_fn Consistency with apply_scalar_squash
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// For all known scalar squash names, target_simulation_fn and apply_scalar_squash
    /// must produce identical results for finite inputs.
    #[test]
    fn simulation_fn_consistent_with_apply(
        idx in 0..SCALAR_SQUASH_NAMES.len(),
        x in -10.0f32..10.0,
    ) {
        let name = SCALAR_SQUASH_NAMES[idx];
        let scalar_result = apply_scalar_squash(name, x);
        let sim_fn = target_simulation_fn(name);

        match (scalar_result, sim_fn) {
            (Some(expected), Some(f)) => {
                let actual = f(x);
                let diff = (expected - actual).abs();
                let tolerance = expected.abs().max(1.0) * 1e-5;
                prop_assert!(
                    diff < tolerance,
                    "{name}({x}): apply_scalar_squash={expected}, target_simulation_fn={actual}, diff={diff}"
                );
            }
            (None, None) => {
                // Both return None — consistent
            }
            (Some(_), None) | (None, Some(_)) => {
                // Both should agree on whether a function exists
                prop_assert!(
                    false,
                    "{name}: apply_scalar_squash returns {:?}, target_simulation_fn returns {:?}",
                    scalar_result.is_some(),
                    sim_fn.is_some()
                );
            }
        }
    }

    /// Aggregate squash names must return None from both functions.
    #[test]
    fn aggregate_returns_none_from_both(
        idx in 0..AGGREGATE_SQUASH_NAMES.len(),
        x in -10.0f32..10.0,
    ) {
        let name = AGGREGATE_SQUASH_NAMES[idx];
        prop_assert!(
            apply_scalar_squash(name, x).is_none(),
            "apply_scalar_squash('{name}', {x}) should be None for aggregate"
        );
        prop_assert!(
            target_simulation_fn(name).is_none(),
            "target_simulation_fn('{name}') should be None for aggregate"
        );
    }

    /// Unknown names must return None from both functions.
    #[test]
    fn unknown_returns_none_from_both(
        name in "[xyz]{5,10}",
        x in -10.0f32..10.0,
    ) {
        prop_assert!(
            apply_scalar_squash(&name, x).is_none(),
            "apply_scalar_squash('{name}', {x}) should be None for unknown"
        );
        prop_assert!(
            target_simulation_fn(&name).is_none(),
            "target_simulation_fn('{name}') should be None for unknown"
        );
    }
}
