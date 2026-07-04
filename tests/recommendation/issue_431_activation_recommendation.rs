//! Tests for Issue #431: Activation function recommendation engine.
//!
//! This module implements proactive activation function analysis that recommends
//! optimal activation functions based on input distribution characteristics,
//! output range requirements, and gradient flow analysis.
//!
//! ## TDD Plan
//! 1. Test input distribution classification (Gaussian, sparse, bounded)
//! 2. Test activation function matching to input distributions
//! 3. Test output range requirement detection
//! 4. Test gradient flow analysis
//! 5. Test proactive recommendation generation
//! 6. Test that recommendations differ from current reactive approach

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::recommendation::activation_recommendation::{
    ActivationRecommendation, InputDistribution, InputDistributionClass, OutputRangeRequirement,
    analyse_input_distribution, classify_activation_suitability, detect_output_range_requirements,
    recommend_activation_function,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a `DiscoverRecord` with given activation value.
fn record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation,
        errors: vec![0.01],
    }
}

// =============================================================================
// Test 1: Input Distribution Classification - Gaussian
// =============================================================================

/// Test that a Gaussian-like distribution is correctly classified.
/// Gaussian inputs should be recommended TANH or SOFTPLUS.
#[test]
fn test_classifies_gaussian_input_distribution() {
    // Generate Gaussian-like distribution using Box-Muller approximation
    // More values near the mean (0), fewer at extremes
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Simulate Gaussian with more density near centre
            // Use a simple approximation: values cluster around 0
            let t = (i as f32 - 50.0) / 50.0; // -1 to 1
            // Apply a transformation that clusters values near 0
            let x = t * t.abs().sqrt() * 2.0; // More values near 0
            record("hidden-1", i, x, Some(x))
        })
        .collect();

    let distribution = analyse_input_distribution(&records);

    // Gaussian-like distributions should be classified as Gaussian or Uniform
    // (both are acceptable for smooth, centred distributions)
    assert!(
        distribution.class == InputDistributionClass::Gaussian
            || distribution.class == InputDistributionClass::Uniform,
        "Should classify as Gaussian or Uniform, got {:?}",
        distribution.class
    );
    assert!(
        distribution.mean.abs() < 0.5,
        "Mean should be near zero for Gaussian-like, got {}",
        distribution.mean
    );
}

// =============================================================================
// Test 2: Input Distribution Classification - Sparse
// =============================================================================

/// Test that a sparse distribution (many zeros, few non-zeros) is correctly classified.
/// Sparse inputs should be recommended RELU variants.
#[test]
fn test_classifies_sparse_input_distribution() {
    // Generate sparse distribution: 80% zeros, 20% positive values
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 5 == 0 {
                (i as f32 / 20.0) + 0.5 // Non-zero values
            } else {
                0.0 // Zeros
            };
            record("hidden-2", i, activation, Some(activation))
        })
        .collect();

    let distribution = analyse_input_distribution(&records);

    assert_eq!(
        distribution.class,
        InputDistributionClass::Sparse,
        "Should classify as Sparse distribution"
    );
    assert!(
        distribution.sparsity > 0.5,
        "Sparsity should be > 0.5, got {}",
        distribution.sparsity
    );
}

// =============================================================================
// Test 3: Input Distribution Classification - Bounded
// =============================================================================

/// Test that a bounded distribution (values in tight range) is correctly classified.
/// Bounded inputs should be recommended LOGISTIC or `HARD_TANH`.
#[test]
fn test_classifies_bounded_input_distribution() {
    // Generate bounded distribution: values strictly in [0, 1]
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 / 100.0).clamp(0.0, 1.0);
            record("hidden-3", i, activation, Some(activation))
        })
        .collect();

    let distribution = analyse_input_distribution(&records);

    assert_eq!(
        distribution.class,
        InputDistributionClass::Bounded,
        "Should classify as Bounded distribution"
    );
    assert!(
        distribution.min >= 0.0 && distribution.max <= 1.0,
        "Range should be [0, 1], got [{}, {}]",
        distribution.min,
        distribution.max
    );
}

// =============================================================================
// Test 4: Input Distribution Classification - Uniform
// =============================================================================

/// Test that a uniform distribution (evenly spread values) is correctly classified.
#[test]
fn test_classifies_uniform_input_distribution() {
    // Generate uniform distribution: values evenly spread from -5 to 5
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = -5.0 + (i as f32 / 10.0);
            record("hidden-4", i, activation, Some(activation))
        })
        .collect();

    let distribution = analyse_input_distribution(&records);

    assert_eq!(
        distribution.class,
        InputDistributionClass::Uniform,
        "Should classify as Uniform distribution"
    );
}

// =============================================================================
// Test 5: Activation Suitability for Gaussian Inputs
// =============================================================================

/// Test that TANH and SOFTPLUS are recommended for Gaussian inputs.
#[test]
fn test_recommends_tanh_for_gaussian_inputs() {
    let distribution = InputDistribution {
        class: InputDistributionClass::Gaussian,
        mean: 0.0,
        std_dev: 1.0,
        min: -3.0,
        max: 3.0,
        sparsity: 0.0,
        kurtosis: 3.0, // Normal kurtosis
    };

    let suitability = classify_activation_suitability(&distribution);

    assert!(
        suitability.contains_key("TANH"),
        "Should include TANH for Gaussian"
    );
    assert!(
        suitability.contains_key("SOFTPLUS"),
        "Should include SOFTPLUS for Gaussian"
    );

    let tanh_score = suitability.get("TANH").unwrap();
    let relu_score = suitability.get("RELU").unwrap_or(&0.0);

    assert!(
        tanh_score > relu_score,
        "TANH should score higher than RELU for Gaussian inputs"
    );
}

// =============================================================================
// Test 6: Activation Suitability for Sparse Inputs
// =============================================================================

/// Test that RELU variants are recommended for sparse inputs.
#[test]
fn test_recommends_relu_for_sparse_inputs() {
    let distribution = InputDistribution {
        class: InputDistributionClass::Sparse,
        mean: 0.5,
        std_dev: 1.5,
        min: 0.0,
        max: 5.0,
        sparsity: 0.8, // 80% zeros
        kurtosis: 5.0, // High kurtosis (peaky)
    };

    let suitability = classify_activation_suitability(&distribution);

    assert!(
        suitability.contains_key("RELU") || suitability.contains_key("RELU6"),
        "Should include RELU variants for sparse inputs"
    );

    // For sparse inputs, RELU should score higher than TANH
    let relu_score = suitability.get("RELU").copied().unwrap_or(0.0);
    let tanh_score = suitability.get("TANH").copied().unwrap_or(0.0);

    assert!(
        relu_score > tanh_score,
        "RELU should score higher than TANH for sparse inputs"
    );
}

// =============================================================================
// Test 7: Activation Suitability for Bounded Inputs
// =============================================================================

/// Test that LOGISTIC and `HARD_TANH` are recommended for bounded inputs.
#[test]
fn test_recommends_logistic_for_bounded_inputs() {
    let distribution = InputDistribution {
        class: InputDistributionClass::Bounded,
        mean: 0.5,
        std_dev: 0.2,
        min: 0.0,
        max: 1.0,
        sparsity: 0.0,
        kurtosis: 2.0,
    };

    let suitability = classify_activation_suitability(&distribution);

    assert!(
        suitability.contains_key("LOGISTIC") || suitability.contains_key("HARD_TANH"),
        "Should include LOGISTIC or HARD_TANH for bounded inputs"
    );
}

// =============================================================================
// Test 8: Output Range Requirements Detection
// =============================================================================

/// Test detection of output range requirements from target neuron properties.
#[test]
fn test_detects_binary_output_requirement() {
    // A neuron that only outputs 0 or 1 requires a bounded activation
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.0 } else { 1.0 };
            record("output-0", i, activation, Some(activation))
        })
        .collect();

    let requirement = detect_output_range_requirements(&records);

    assert_eq!(
        requirement,
        OutputRangeRequirement::Binary,
        "Should detect binary output requirement"
    );
}

/// Test detection of unbounded output requirements.
#[test]
fn test_detects_unbounded_output_requirement() {
    // A neuron with large range needs unbounded activation
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = -100.0 + (i as f32 * 2.0);
            record("output-1", i, activation, Some(activation))
        })
        .collect();

    let requirement = detect_output_range_requirements(&records);

    assert_eq!(
        requirement,
        OutputRangeRequirement::Unbounded,
        "Should detect unbounded output requirement"
    );
}

// =============================================================================
// Test 9: Proactive Recommendation Generation
// =============================================================================

/// Test that recommendations are generated proactively for a neuron.
#[test]
fn test_generates_proactive_recommendation() {
    // Create a neuron with Gaussian-like inputs but currently using RELU
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let x = (i as f32 - 50.0) / 25.0;
            record("hidden-test", i, x, Some(x))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "RELU");

    assert!(recommendation.is_some(), "Should generate a recommendation");

    let rec = recommendation.unwrap();
    assert_eq!(rec.current_squash, "RELU");
    assert!(
        rec.recommended_squash == "TANH" || rec.recommended_squash == "SOFTPLUS",
        "Should recommend TANH or SOFTPLUS for Gaussian inputs, got {}",
        rec.recommended_squash
    );
    assert!(
        rec.confidence > 0.5,
        "Confidence should be > 0.5, got {}",
        rec.confidence
    );
    assert!(
        rec.expected_improvement > 0.0,
        "Expected improvement should be positive"
    );
}

// =============================================================================
// Test 10: No Recommendation When Current is Optimal
// =============================================================================

/// Test that no recommendation is made when current activation is already optimal.
#[test]
fn test_no_recommendation_when_optimal() {
    // Gaussian inputs with TANH (already optimal match)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let x = (i as f32 - 50.0) / 25.0;
            record("hidden-opt", i, x, Some(x))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "TANH");

    // Should either return None or return a recommendation with very low improvement
    if let Some(rec) = recommendation {
        assert!(
            rec.expected_improvement < 0.001,
            "Improvement should be negligible when already optimal"
        );
    }
}

// =============================================================================
// Test 11: Recommendation Includes Rationale
// =============================================================================

/// Test that recommendations include a human-readable rationale.
#[test]
fn test_recommendation_includes_rationale() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Sparse input pattern
            let activation = if i % 5 == 0 { 1.0 } else { 0.0 };
            record("hidden-sparse", i, activation, Some(activation))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "TANH");

    assert!(recommendation.is_some(), "Should generate recommendation");
    let rec = recommendation.unwrap();

    assert!(!rec.rationale.is_empty(), "Rationale should not be empty");
    assert!(
        rec.rationale.contains("sparse") || rec.rationale.contains("distribution"),
        "Rationale should mention the input distribution type"
    );
}

// =============================================================================
// Test 12: Insufficient Samples Returns None
// =============================================================================

/// Test that insufficient samples don't produce recommendations.
#[test]
fn test_insufficient_samples_no_recommendation() {
    // Only 5 samples - not enough for reliable analysis
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("hidden-few", i, i as f32 * 0.1, Some(i as f32 * 0.1)))
        .collect();

    let recommendation = recommend_activation_function(&records, "RELU");

    assert!(
        recommendation.is_none(),
        "Should not recommend with insufficient samples"
    );
}

// =============================================================================
// Test 13: Recommendation to Coordinated Structural Candidate
// =============================================================================

/// Test conversion of recommendation to coordinated structural candidate.
#[test]
fn test_recommendation_to_coordinated_candidate() {
    use neat_ai_discovery::analysis::recommendation::activation_recommendation::recommendation_to_coordinated_candidate;

    let recommendation = ActivationRecommendation {
        neuron_uuid: "hidden-42".to_string(),
        current_squash: "RELU".to_string(),
        recommended_squash: "TANH".to_string(),
        confidence: 0.85,
        expected_improvement: 0.015,
        rationale: "Gaussian input distribution matches TANH characteristics".to_string(),
        input_distribution: InputDistributionClass::Gaussian,
    };

    let candidate = recommendation_to_coordinated_candidate(&recommendation);

    assert!(
        candidate.expected_creature_score_gain > 0.0,
        "Should have positive expected gain"
    );
    assert!(candidate.comment.is_some(), "Should have a comment");

    // Check that the candidate has a changeSquash operation
    let ops_json = serde_json::to_string(&candidate.operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "Should include changeSquash operation"
    );
    assert!(ops_json.contains("TANH"), "Should recommend TANH");
}

// =============================================================================
// Test 14: Bimodal Distribution Detection
// =============================================================================

/// Test that a bimodal distribution is correctly classified.
#[test]
fn test_classifies_bimodal_input_distribution() {
    // Generate bimodal distribution: values clustered around -2 and +2
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 50 {
                -2.0 + (i as f32 * 0.02) // Cluster around -2
            } else {
                2.0 + ((i - 50) as f32 * 0.02) // Cluster around +2
            };
            record("hidden-bimodal", i, activation, Some(activation))
        })
        .collect();

    let distribution = analyse_input_distribution(&records);

    // Bimodal distributions may be classified as Uniform or a special Bimodal class
    assert!(
        distribution.class == InputDistributionClass::Bimodal
            || distribution.class == InputDistributionClass::Uniform,
        "Should classify as Bimodal or Uniform for bimodal data"
    );
}

// =============================================================================
// Test 15: Gradient Flow Analysis
// =============================================================================

/// Test that gradient flow concerns are factored into recommendations.
#[test]
fn test_considers_gradient_flow() {
    use neat_ai_discovery::analysis::recommendation::activation_recommendation::analyse_gradient_flow_risk;

    // Activations near saturation for TANH
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-sat", i, 0.98, Some(5.0)))
        .collect();

    let gradient_risk = analyse_gradient_flow_risk(&records, "TANH");

    assert!(
        gradient_risk > 0.5,
        "Should detect high gradient flow risk for near-saturated TANH"
    );
}

// =============================================================================
// Test 16: Documented Behaviour — Activation Recommendation emits changeSquash
// =============================================================================

/// Assert the behaviour `DISCOVERY_TYPES.md` promises for activation
/// recommendation (Issue #1503): the public recommender, run on a neuron whose
/// input distribution favours a different activation, produces a recommendation
/// that converts to the documented `changeSquash` candidate operation naming a
/// valid squash op.
///
/// This replaces a former doc-prose grep (which only checked that a heading
/// string appeared in the Markdown file) with a WHAT-test that exercises the
/// real recommender output — the behaviour the documentation describes.
#[test]
fn test_activation_recommendation_produces_change_squash_candidate() {
    use neat_ai_discovery::analysis::recommendation::activation_recommendation::recommendation_to_coordinated_candidate;

    // Gaussian-like inputs on a neuron currently using RELU: the documented
    // scenario where a better-matched activation should be recommended.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let x = (i as f32 - 50.0) / 25.0;
            record("hidden-doc-behaviour", i, x, Some(x))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "RELU")
        .expect("activation recommendation should be produced for a Gaussian neuron");

    // The recommendation must name a non-empty squash op distinct from current.
    assert!(
        !recommendation.recommended_squash.is_empty(),
        "recommended squash op should be named"
    );
    assert_ne!(
        recommendation.recommended_squash, recommendation.current_squash,
        "a recommendation should change the activation function"
    );

    // The documented candidate operation is `changeSquash` (DISCOVERY_TYPES.md).
    let candidate = recommendation_to_coordinated_candidate(&recommendation);
    let ops_json = serde_json::to_string(&candidate.operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "activation recommendation should emit the documented changeSquash operation, got {ops_json}"
    );
    assert!(
        ops_json.contains(&recommendation.recommended_squash),
        "changeSquash operation should carry the recommended squash op"
    );
}
