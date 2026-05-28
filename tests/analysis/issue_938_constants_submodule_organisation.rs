//! Tests for Issue #938: Split constants.rs into thematic sub-modules.
//!
//! The stated intent of these tests is to verify that all constants remain
//! accessible via the same `crate::analysis::constants::CONSTANT_NAME` paths
//! after the sub-module split — i.e. that the re-export contract is intact.
//!
//! Issue #1295: this file previously asserted `assert_eq!(constants::FOO, <literal>)`
//! for every constant. Those assertions were circular magic-value tests: they
//! pass if and only if the literal in the test matches the literal in the
//! source, and they obstruct legitimate retuning of any constant without
//! catching any real defect. They have been replaced with:
//!
//! - **Compile-only re-export and invariant checks** (`const _: T = constants::FOO;`
//!   and `const _: () = assert!(...)`) for the symbol-accessibility intent and
//!   pure-constant invariants. If the re-export breaks or a value tuned past
//!   its declared envelope, the file fails to compile — a stronger guarantee
//!   than the runtime assertion ever provided, at zero runtime cost.
//! - **Behavioural tests** for the small number of helper functions exposed by
//!   the constants module (NaN-safe comparators, activation boost lookup,
//!   empirical discount lookup).

use neat_ai_discovery::analysis::constants;

// =============================================================================
// Compile-time re-export checks — verifies every constant tested by the
// original file is still reachable via `constants::*`. If any re-export is
// removed the file will fail to compile.
// =============================================================================

// Sample thresholds
const _: usize = constants::MIN_NEURON_SAMPLE_COUNT;
const _: usize = constants::MIN_DISCOVERY_SAMPLE_COUNT;
const _: usize = constants::HOLDOUT_MIN_SAMPLE_COUNT;
const _: f32 = constants::HOLDOUT_VALIDATION_FRACTION;

// Sentinel detection
const _: [f32; 3] = constants::CANDIDATE_SENTINELS;
const _: f32 = constants::MIN_SENTINEL_FRACTION;
const _: f32 = constants::SENTINEL_TOLERANCE;
const _: f32 = constants::MIN_SENTINEL_GAP;

// Source variance
const _: f32 = constants::MIN_SOURCE_STD_DEV;

// Candidate scoring — diversification
const _: usize = constants::DIVERSIFY_TOP_K;

// Candidate scoring — source / target type boosts
const _: f64 = constants::INPUT_SOURCE_BOOST;
const _: f64 = constants::HIDDEN_SOURCE_BOOST;
const _: usize = constants::HIDDEN_SOURCE_INTERLEAVE_INTERVAL;
const _: usize = constants::MIN_BOOST_SAMPLES;
const _: f64 = constants::EXISTING_HIDDEN_TARGET_BOOST;

// Candidate scoring — activation boosts
const _: f64 = constants::ACTIVATION_BOOST_GELU;
const _: f64 = constants::ACTIVATION_BOOST_IDENTITY;
const _: f64 = constants::ACTIVATION_BOOST_TANH;
const _: f64 = constants::ACTIVATION_BOOST_HARD_TANH;
const _: f64 = constants::ACTIVATION_BOOST_MIN;
const _: f64 = constants::ACTIVATION_BOOST_MAX;

// Candidate scoring — pessimism discounting
const _: f32 = constants::PESSIMISM_DISCOUNT_FLOOR;
const _: f32 = constants::PESSIMISM_CURVE_EXPONENT;
const _: f32 = constants::NEURON_PESSIMISM_DISCOUNT_FLOOR;
const _: f32 = constants::NEURON_PESSIMISM_CURVE_EXPONENT;
const _: f32 = constants::SYNAPSE_PESSIMISM_DISCOUNT_FLOOR;
const _: f32 = constants::SYNAPSE_PESSIMISM_CURVE_EXPONENT;

// Candidate scoring — coordinated-structural
const _: f32 = constants::COORDINATED_EMPIRICAL_DISCOUNT_2OPS;
const _: f32 = constants::COORDINATED_EMPIRICAL_DISCOUNT_3OPS;
const _: f32 = constants::COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS;
const _: f32 = constants::MIN_COORDINATED_MULTI_OP_GAIN;
const _: f32 = constants::COORDINATED_ESTIMATION_WEIGHT_SCALE;

// Candidate scoring — prediction calibration
const _: f32 = constants::SYNAPSE_PREDICTION_CALIBRATION;
const _: f32 = constants::NEURON_PREDICTION_CALIBRATION;
const _: f32 = constants::COORDINATED_PREDICTION_CALIBRATION;
const _: f32 = constants::LOGISTIC_CALIBRATION_FLOOR;
const _: f32 = constants::LOGISTIC_CALIBRATION_STEEPNESS;
const _: f32 = constants::LOGISTIC_CALIBRATION_MIDPOINT;

// Candidate scoring — scoring boost multipliers
const _: f32 = constants::MICRO_NUDGE_VARIANT_BOOST;
const _: f32 = constants::REMOVAL_CANDIDATE_BOOST;

// Detection thresholds
const _: f32 = constants::MIN_IMPROVED_RATIO;
const _: f32 = constants::NEURON_MIN_IMPROVED_RATIO;
const _: f32 = constants::REMOVAL_MEAN_ACTIVATION_THRESHOLD;
const _: f32 = constants::REMOVAL_IMPACT_THRESHOLD;
const _: f32 = constants::MAX_INCOMING_WEIGHT;
const _: f32 = constants::MAX_BIAS_MAGNITUDE;
const _: f32 = constants::MAX_INDIVIDUAL_HARM_FOR_PAIRING;

// Compression
const _: usize = constants::MIN_COMPRESSED_SOURCES;
const _: usize = constants::MAX_COMPRESSION_INPUTS;
const _: f32 = constants::COMPRESSION_SATURATION_THRESHOLD;
const _: f32 = constants::COMPRESSION_MIN_BENEFIT_RATIO;

// =============================================================================
// Compile-time invariant checks — pure-constant relationships that would
// be `#[test]` assertions if the values were not const-eval-able. Using
// `const _: () = assert!(...)` evaluates them at compile time and avoids
// the `clippy::assertions_on_constants` lint.
// =============================================================================

// Logistic calibration parameters describe a usable curve.
const _: () = assert!(
    constants::LOGISTIC_CALIBRATION_FLOOR > 0.0,
    "logistic floor must be strictly positive — a zero floor would collapse \
     the calibration onto the asymptote"
);
const _: () = assert!(
    constants::LOGISTIC_CALIBRATION_STEEPNESS > 0.0,
    "logistic steepness must be strictly positive — a non-positive value \
     would flip the curve"
);
const _: () = assert!(
    constants::LOGISTIC_CALIBRATION_MIDPOINT > 0.0
        && constants::LOGISTIC_CALIBRATION_MIDPOINT < 1.0,
    "logistic midpoint must lie inside (0, 1)"
);

// Hold-out validation parameters describe a usable train/validate split.
const _: () = assert!(
    constants::HOLDOUT_VALIDATION_FRACTION > 0.0 && constants::HOLDOUT_VALIDATION_FRACTION < 1.0,
    "validation fraction must lie strictly inside (0, 1) — otherwise one \
     partition would be empty"
);
const _: () = assert!(
    constants::HOLDOUT_MIN_SAMPLE_COUNT >= constants::MIN_DISCOVERY_SAMPLE_COUNT,
    "hold-out minimum must be at least as large as the generic discovery minimum"
);

// Pessimism discount floors stay inside the unit interval.
const _: () = assert!(
    constants::PESSIMISM_DISCOUNT_FLOOR >= 0.0 && constants::PESSIMISM_DISCOUNT_FLOOR <= 1.0,
    "PESSIMISM_DISCOUNT_FLOOR must lie within [0, 1]"
);
const _: () = assert!(
    constants::NEURON_PESSIMISM_DISCOUNT_FLOOR >= 0.0
        && constants::NEURON_PESSIMISM_DISCOUNT_FLOOR <= 1.0,
    "NEURON_PESSIMISM_DISCOUNT_FLOOR must lie within [0, 1]"
);
const _: () = assert!(
    constants::SYNAPSE_PESSIMISM_DISCOUNT_FLOOR >= 0.0
        && constants::SYNAPSE_PESSIMISM_DISCOUNT_FLOOR <= 1.0,
    "SYNAPSE_PESSIMISM_DISCOUNT_FLOOR must lie within [0, 1]"
);
const _: () = assert!(
    constants::PESSIMISM_CURVE_EXPONENT > 0.0,
    "PESSIMISM_CURVE_EXPONENT must be strictly positive"
);
const _: () = assert!(
    constants::NEURON_PESSIMISM_CURVE_EXPONENT > 0.0,
    "NEURON_PESSIMISM_CURVE_EXPONENT must be strictly positive"
);
const _: () = assert!(
    constants::SYNAPSE_PESSIMISM_CURVE_EXPONENT > 0.0,
    "SYNAPSE_PESSIMISM_CURVE_EXPONENT must be strictly positive"
);

// Coordinated-structural empirical discounts shrink monotonically with op count
// (more ops = more pessimism), and the 4-plus bucket is the smallest.
const _: () = assert!(
    constants::COORDINATED_EMPIRICAL_DISCOUNT_2OPS > constants::COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
    "two-op discount must exceed three-op discount"
);
const _: () = assert!(
    constants::COORDINATED_EMPIRICAL_DISCOUNT_3OPS
        > constants::COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS,
    "three-op discount must exceed four-plus discount"
);

// Compression bounds form a valid range.
const _: () = assert!(
    constants::MIN_COMPRESSED_SOURCES >= 2,
    "compressing fewer than 2 sources is a no-op"
);
const _: () = assert!(
    constants::MIN_COMPRESSED_SOURCES <= constants::MAX_COMPRESSION_INPUTS,
    "minimum compressed sources must not exceed the maximum"
);
const _: () = assert!(
    constants::COMPRESSION_MIN_BENEFIT_RATIO > 1.0,
    "compression must require at least a 1.0x benefit to be worthwhile"
);
const _: () = assert!(
    constants::COMPRESSION_SATURATION_THRESHOLD > 0.0
        && constants::COMPRESSION_SATURATION_THRESHOLD < 1.0,
    "compression saturation threshold must lie strictly inside (0, 1)"
);

// =============================================================================
// Behavioural tests — these exercise function behaviour, not magic values.
// They survive value tuning because they assert on relationships and contracts.
// =============================================================================

/// `activation_neuron_boost` returns the per-squash boost for known squashes
/// and a neutral 1.0 for unknown squashes.
#[test]
fn test_activation_neuron_boost_contract() {
    // Known squash returns its registered boost.
    assert!(
        (constants::activation_neuron_boost("GELU") - constants::ACTIVATION_BOOST_GELU).abs()
            < f64::EPSILON,
        "GELU boost must match the registered constant"
    );
    assert!(
        (constants::activation_neuron_boost("IDENTITY") - constants::ACTIVATION_BOOST_IDENTITY)
            .abs()
            < f64::EPSILON,
        "IDENTITY boost must match the registered constant"
    );
    // Unknown squash falls back to the neutral 1.0 multiplier (no boost, no penalty).
    assert!(
        (constants::activation_neuron_boost("unknown") - 1.0).abs() < f64::EPSILON,
        "unknown squash must fall back to a neutral 1.0 multiplier"
    );
    // Every returned boost stays inside the declared [MIN, MAX] envelope.
    for squash in ["GELU", "IDENTITY", "TANH", "HARD_TANH"] {
        let boost = constants::activation_neuron_boost(squash);
        assert!(
            (constants::ACTIVATION_BOOST_MIN..=constants::ACTIVATION_BOOST_MAX).contains(&boost),
            "{squash} boost {boost} must lie within \
             [ACTIVATION_BOOST_MIN, ACTIVATION_BOOST_MAX]"
        );
    }
}

/// NaN-safe comparators sort in the documented direction.
#[test]
fn test_nan_safe_comparators_ordering() {
    use std::cmp::Ordering;

    // Descending: a larger value sorts before a smaller one.
    assert_eq!(constants::cmp_f32_desc(&2.0, &1.0), Ordering::Less);
    assert_eq!(constants::cmp_f32_desc(&1.0, &2.0), Ordering::Greater);
    assert_eq!(constants::cmp_f32_desc(&1.0, &1.0), Ordering::Equal);

    // Ascending: a smaller value sorts first.
    assert_eq!(constants::cmp_f32_asc(&1.0, &2.0), Ordering::Less);
    assert_eq!(constants::cmp_f32_asc(&2.0, &1.0), Ordering::Greater);

    // f64 descending mirrors the f32 contract.
    assert_eq!(constants::cmp_f64_desc(&2.0, &1.0), Ordering::Less);
    assert_eq!(constants::cmp_f64_desc(&1.0, &2.0), Ordering::Greater);
}

/// `coordinated_empirical_discount` returns the per-op-count discount factor
/// and saturates at the 4-plus-ops bucket for higher op counts.
#[test]
fn test_coordinated_empirical_discount_contract() {
    let d2 = constants::coordinated_empirical_discount(2);
    let d3 = constants::coordinated_empirical_discount(3);
    let d4 = constants::coordinated_empirical_discount(4);
    let d5 = constants::coordinated_empirical_discount(5);

    // Each op-count bucket returns its registered discount.
    assert!((d2 - constants::COORDINATED_EMPIRICAL_DISCOUNT_2OPS).abs() < f32::EPSILON);
    assert!((d3 - constants::COORDINATED_EMPIRICAL_DISCOUNT_3OPS).abs() < f32::EPSILON);
    assert!((d4 - constants::COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS).abs() < f32::EPSILON);

    // The 4-plus bucket saturates — five-op candidates use the same factor
    // as four-op candidates.
    assert!(
        (d5 - d4).abs() < f32::EPSILON,
        "4-plus bucket must saturate"
    );
}

/// `CANDIDATE_SENTINELS` covers the canonical low / zero / high sentinel
/// values used for sentinel-gating detection.
#[test]
fn test_candidate_sentinels_cover_low_zero_high() {
    let sentinels = constants::CANDIDATE_SENTINELS;
    assert!(
        sentinels.iter().any(|s| *s < 0.0),
        "expected a negative sentinel"
    );
    assert!(sentinels.contains(&0.0), "expected a zero sentinel");
    assert!(
        sentinels.iter().any(|s| *s > 0.0),
        "expected a positive sentinel"
    );
}
