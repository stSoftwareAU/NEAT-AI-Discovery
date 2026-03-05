//! Activation function recommendation engine (Issue #431).
//!
//! Provides **proactive** activation function analysis that recommends optimal
//! activation functions based on input distribution characteristics, output range
//! requirements, and gradient flow analysis.
//!
//! Unlike the reactive `changeSquash` recommendations triggered by saturation or
//! oscillation detection, this module analyses input patterns BEFORE problems occur
//! and suggests activations that match the data characteristics.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Activation Function Recommendation" for full documentation.
//!
//! ## Recommendation Logic
//!
//! 1. **Input distribution matching**:
//!    - Gaussian inputs → TANH or SOFTPLUS
//!    - Sparse inputs → RELU variants
//!    - Bounded inputs → LOGISTIC or HARD_TANH
//!
//! 2. **Output range requirements**:
//!    - Binary outputs → LOGISTIC, STEP, BIPOLAR
//!    - Bounded \[0,1\] → LOGISTIC
//!    - Bounded [-1,1] → TANH, HARD_TANH
//!    - Unbounded → IDENTITY, RELU
//!
//! 3. **Gradient flow analysis**:
//!    - Prefer activations with healthy gradient flow for the observed input range
//!    - Penalise activations that would saturate on the observed inputs

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};
use std::collections::HashMap;

// MIN_SAMPLES_FOR_ANALYSIS moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_ANALYSIS;

/// Threshold for sparsity classification (fraction of zeros).
const SPARSITY_THRESHOLD: f32 = 0.5;

/// Threshold for bounded classification (range ratio).
const BOUNDED_RANGE_THRESHOLD: f32 = 2.0;

/// Minimum improvement required to generate a recommendation.
const MIN_IMPROVEMENT_THRESHOLD: f32 = 0.001;

// =============================================================================
// Input Distribution Types
// =============================================================================

/// Classification of input distribution patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputDistributionClass {
    /// Gaussian-like: centred around mean with bell-curve spread.
    Gaussian,
    /// Sparse: many zeros with occasional non-zero values.
    Sparse,
    /// Bounded: values constrained to a tight range (e.g., \[0,1\]).
    Bounded,
    /// Uniform: evenly distributed across the range.
    Uniform,
    /// Bimodal: two distinct clusters of values.
    Bimodal,
    /// Unknown: insufficient data or unclassifiable.
    Unknown,
}

/// Statistics about the input distribution of a neuron.
#[derive(Debug, Clone)]
pub struct InputDistribution {
    /// Classification of the distribution type.
    pub class: InputDistributionClass,
    /// Mean of the activation values.
    pub mean: f32,
    /// Standard deviation of the activation values.
    pub std_dev: f32,
    /// Minimum observed activation.
    pub min: f32,
    /// Maximum observed activation.
    pub max: f32,
    /// Fraction of samples with activation ≈ 0.
    pub sparsity: f32,
    /// Kurtosis (peakedness) of the distribution.
    pub kurtosis: f32,
}

// =============================================================================
// Output Range Requirements
// =============================================================================

/// Detected output range requirement for a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputRangeRequirement {
    /// Binary output (only 0 or 1).
    Binary,
    /// Bounded to [0, 1].
    UnitInterval,
    /// Bounded to [-1, 1].
    SymmetricUnit,
    /// Unbounded (any value).
    Unbounded,
    /// Unknown or insufficient data.
    Unknown,
}

// =============================================================================
// Recommendation Result
// =============================================================================

/// A proactive activation function recommendation.
#[derive(Debug, Clone)]
pub struct ActivationRecommendation {
    /// UUID of the neuron being analysed.
    pub neuron_uuid: String,
    /// Current activation function.
    pub current_squash: String,
    /// Recommended activation function.
    pub recommended_squash: String,
    /// Confidence in the recommendation (0.0 to 1.0).
    pub confidence: f32,
    /// Expected improvement from the change.
    pub expected_improvement: f32,
    /// Human-readable rationale for the recommendation.
    pub rationale: String,
    /// The detected input distribution class.
    pub input_distribution: InputDistributionClass,
}

// =============================================================================
// Distribution Analysis
// =============================================================================

/// Analyse the input distribution of a neuron from its recorded activations.
///
/// # Arguments
/// * `records` - Discovery records for the neuron.
///
/// # Returns
/// An `InputDistribution` struct with statistics and classification.
pub fn analyse_input_distribution(records: &[DiscoverRecord]) -> InputDistribution {
    if records.len() < MIN_SAMPLES_FOR_ANALYSIS {
        return InputDistribution {
            class: InputDistributionClass::Unknown,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            sparsity: 0.0,
            kurtosis: 0.0,
        };
    }

    let n = records.len() as f32;

    // Compute basic statistics
    let sum: f32 = records.iter().map(|r| r.activation).sum();
    let mean = sum / n;

    let variance: f32 = records
        .iter()
        .map(|r| {
            let diff = r.activation - mean;
            diff * diff
        })
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();

    let min = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::INFINITY, f32::min);
    let max = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute sparsity (fraction of near-zero values)
    let zero_threshold = 1e-6;
    let zero_count = records
        .iter()
        .filter(|r| r.activation.abs() < zero_threshold)
        .count();
    let sparsity = zero_count as f32 / n;

    // Compute kurtosis (measure of peakedness)
    let kurtosis = if std_dev > 1e-6 {
        let fourth_moment: f32 = records
            .iter()
            .map(|r| {
                let diff = (r.activation - mean) / std_dev;
                diff * diff * diff * diff
            })
            .sum::<f32>()
            / n;
        fourth_moment
    } else {
        0.0
    };

    // Classify the distribution
    let class = classify_distribution(mean, std_dev, min, max, sparsity, kurtosis, n as usize);

    InputDistribution {
        class,
        mean,
        std_dev,
        min,
        max,
        sparsity,
        kurtosis,
    }
}

/// Classify the distribution type based on statistics.
fn classify_distribution(
    mean: f32,
    std_dev: f32,
    min: f32,
    max: f32,
    sparsity: f32,
    kurtosis: f32,
    _sample_count: usize,
) -> InputDistributionClass {
    let range = max - min;

    // Check for sparse distribution (many zeros)
    if sparsity > SPARSITY_THRESHOLD {
        return InputDistributionClass::Sparse;
    }

    // Check for bounded distribution (tight range)
    if range < BOUNDED_RANGE_THRESHOLD && range > 0.0 {
        // If range is approximately [0,1] or [-1,1], it's bounded
        // Simplified: max must be <= 1.1, and min must be >= -1.1
        if max <= 1.1 && min >= -1.1 {
            return InputDistributionClass::Bounded;
        }
    }

    // Check for bimodal distribution (two clusters)
    // Bimodal distributions typically have low kurtosis < 3 and high variance relative to range
    if kurtosis < 2.5 && std_dev > range * 0.3 {
        // Additional check: mean should be roughly in the middle of the range
        let range_midpoint = (min + max) / 2.0;
        if (mean - range_midpoint).abs() < std_dev {
            return InputDistributionClass::Bimodal;
        }
    }

    // Check for Gaussian distribution
    // Gaussian has kurtosis ≈ 3.0 and symmetric distribution around mean
    if kurtosis > 2.0 && kurtosis < 5.0 {
        let skewness_proxy = (mean - (min + max) / 2.0).abs() / (range / 2.0).max(1e-6);
        if skewness_proxy < 0.3 {
            return InputDistributionClass::Gaussian;
        }
    }

    // Default to uniform for evenly spread distributions
    InputDistributionClass::Uniform
}

// =============================================================================
// Activation Suitability Classification
// =============================================================================

/// Compute suitability scores for different activation functions given an input distribution.
///
/// # Arguments
/// * `distribution` - The analysed input distribution.
///
/// # Returns
/// A HashMap mapping activation function names to suitability scores (0.0 to 1.0).
pub fn classify_activation_suitability(distribution: &InputDistribution) -> HashMap<String, f32> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    match distribution.class {
        InputDistributionClass::Gaussian => {
            // Gaussian inputs work well with smooth, symmetric activations
            scores.insert("TANH".to_string(), 0.9);
            scores.insert("SOFTPLUS".to_string(), 0.85);
            scores.insert("GELU".to_string(), 0.8);
            scores.insert("IDENTITY".to_string(), 0.7);
            scores.insert("ELU".to_string(), 0.75);
            scores.insert("RELU".to_string(), 0.5); // Loses negative information
            scores.insert("LOGISTIC".to_string(), 0.6);
        }
        InputDistributionClass::Sparse => {
            // Sparse inputs benefit from activations that preserve zeros
            scores.insert("RELU".to_string(), 0.9);
            scores.insert("RELU6".to_string(), 0.85);
            scores.insert("ELU".to_string(), 0.8);
            scores.insert("GELU".to_string(), 0.75);
            scores.insert("SOFTPLUS".to_string(), 0.7);
            scores.insert("TANH".to_string(), 0.5); // Maps zeros to zeros, but loses sparsity pattern
            scores.insert("IDENTITY".to_string(), 0.6);
        }
        InputDistributionClass::Bounded => {
            // Bounded inputs match well with bounded activations
            scores.insert("LOGISTIC".to_string(), 0.9);
            scores.insert("HARD_TANH".to_string(), 0.85);
            scores.insert("TANH".to_string(), 0.8);
            scores.insert("SOFTSIGN".to_string(), 0.75);
            scores.insert("ARCTAN".to_string(), 0.7);
            scores.insert("RELU".to_string(), 0.5);
            scores.insert("IDENTITY".to_string(), 0.4); // May amplify out of bounds
        }
        InputDistributionClass::Uniform => {
            // Uniform distributions are flexible; prefer smooth activations
            scores.insert("TANH".to_string(), 0.75);
            scores.insert("IDENTITY".to_string(), 0.7);
            scores.insert("ELU".to_string(), 0.7);
            scores.insert("GELU".to_string(), 0.7);
            scores.insert("SOFTPLUS".to_string(), 0.65);
            scores.insert("RELU".to_string(), 0.6);
            scores.insert("LOGISTIC".to_string(), 0.6);
        }
        InputDistributionClass::Bimodal => {
            // Bimodal inputs may benefit from threshold-like activations
            scores.insert("TANH".to_string(), 0.7);
            scores.insert("BIPOLAR".to_string(), 0.65);
            scores.insert("HARD_TANH".to_string(), 0.7);
            scores.insert("IDENTITY".to_string(), 0.6);
            scores.insert("RELU".to_string(), 0.5);
        }
        InputDistributionClass::Unknown => {
            // Conservative scores for unknown distributions
            scores.insert("IDENTITY".to_string(), 0.5);
            scores.insert("TANH".to_string(), 0.5);
            scores.insert("RELU".to_string(), 0.5);
        }
    }

    // Apply gradient flow penalty for activations that would saturate
    apply_gradient_flow_penalty(&mut scores, distribution);

    scores
}

/// Apply penalty to activation scores based on gradient flow risk.
fn apply_gradient_flow_penalty(
    scores: &mut HashMap<String, f32>,
    distribution: &InputDistribution,
) {
    // Penalise TANH if inputs are far from zero (would saturate)
    if distribution.mean.abs() > 2.0 || distribution.max.abs() > 3.0 || distribution.min.abs() > 3.0
    {
        if let Some(score) = scores.get_mut("TANH") {
            *score *= 0.7;
        }
        if let Some(score) = scores.get_mut("LOGISTIC") {
            *score *= 0.7;
        }
    }

    // Penalise RELU if many inputs are negative (would lose information)
    if distribution.mean < 0.0 || distribution.min < -1.0 {
        let negative_fraction = if distribution.max > distribution.min {
            (0.0 - distribution.min) / (distribution.max - distribution.min)
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        if negative_fraction > 0.3 {
            if let Some(score) = scores.get_mut("RELU") {
                *score *= (1.0 - negative_fraction * 0.5).max(0.3);
            }
            if let Some(score) = scores.get_mut("ReLU6") {
                *score *= (1.0 - negative_fraction * 0.5).max(0.3);
            }
        }
    }
}

// =============================================================================
// Output Range Detection
// =============================================================================

/// Detect the output range requirement from recorded activations.
///
/// # Arguments
/// * `records` - Discovery records for the neuron.
///
/// # Returns
/// The detected output range requirement.
pub fn detect_output_range_requirements(records: &[DiscoverRecord]) -> OutputRangeRequirement {
    if records.len() < MIN_SAMPLES_FOR_ANALYSIS {
        return OutputRangeRequirement::Unknown;
    }

    let min = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::INFINITY, f32::min);
    let max = records
        .iter()
        .map(|r| r.activation)
        .fold(f32::NEG_INFINITY, f32::max);

    // Check for binary (only two distinct values close to 0 and 1)
    let unique_values: std::collections::HashSet<i32> = records
        .iter()
        .map(|r| (r.activation * 100.0) as i32) // Quantise to 2 decimal places
        .collect();

    if unique_values.len() <= 2 {
        let values: Vec<f32> = unique_values.iter().map(|&v| v as f32 / 100.0).collect();
        if values
            .iter()
            .all(|&v| (v - 0.0).abs() < 0.1 || (v - 1.0).abs() < 0.1)
        {
            return OutputRangeRequirement::Binary;
        }
    }

    // Check for unit interval [0, 1]
    if min >= -0.05 && max <= 1.05 {
        return OutputRangeRequirement::UnitInterval;
    }

    // Check for symmetric unit [-1, 1]
    if min >= -1.05 && max <= 1.05 {
        return OutputRangeRequirement::SymmetricUnit;
    }

    // If range is large, consider unbounded
    if (max - min) > 10.0 || max.abs() > 10.0 || min.abs() > 10.0 {
        return OutputRangeRequirement::Unbounded;
    }

    OutputRangeRequirement::Unknown
}

// =============================================================================
// Gradient Flow Analysis
// =============================================================================

/// Analyse the gradient flow risk for a given activation function and input distribution.
///
/// # Arguments
/// * `records` - Discovery records for the neuron.
/// * `squash` - The activation function name.
///
/// # Returns
/// A risk score from 0.0 (no risk) to 1.0 (severe gradient vanishing risk).
pub fn analyse_gradient_flow_risk(records: &[DiscoverRecord], squash: &str) -> f32 {
    if records.is_empty() {
        return 0.0;
    }

    match squash {
        "TANH" | "BIPOLAR_SIGMOID" => {
            // TANH saturates for |x| > 3
            let saturated_count = records.iter().filter(|r| r.activation.abs() > 0.95).count();
            saturated_count as f32 / records.len() as f32
        }
        "LOGISTIC" => {
            // LOGISTIC saturates for x < -5 or x > 5
            let saturated_count = records
                .iter()
                .filter(|r| r.activation < 0.05 || r.activation > 0.95)
                .count();
            saturated_count as f32 / records.len() as f32
        }
        "RELU" | "LEAKYRELU" => {
            // RELU has zero gradient for x < 0
            let dead_count = records.iter().filter(|r| r.activation <= 0.0).count();
            dead_count as f32 / records.len() as f32
        }
        "HARD_TANH" | "CLIPPED" => {
            // HARD_TANH saturates at exactly ±1
            let saturated_count = records
                .iter()
                .filter(|r| r.activation.abs() >= 0.99)
                .count();
            saturated_count as f32 / records.len() as f32
        }
        _ => 0.0, // Identity and unbounded activations have no gradient risk
    }
}

// =============================================================================
// Recommendation Generation
// =============================================================================

/// Generate a proactive activation function recommendation for a neuron.
///
/// # Arguments
/// * `records` - Discovery records for the neuron.
/// * `current_squash` - The current activation function name.
///
/// # Returns
/// An optional `ActivationRecommendation` if a better activation is found.
pub fn recommend_activation_function(
    records: &[DiscoverRecord],
    current_squash: &str,
) -> Option<ActivationRecommendation> {
    if records.len() < MIN_SAMPLES_FOR_ANALYSIS {
        return None;
    }

    let neuron_uuid = records
        .first()
        .map(|r| r.neuron_uuid.clone())
        .unwrap_or_default();

    // Analyse the input distribution
    let distribution = analyse_input_distribution(records);

    if distribution.class == InputDistributionClass::Unknown {
        return None;
    }

    // Get suitability scores for all activation functions
    let suitability = classify_activation_suitability(&distribution);

    // Get the current activation's score
    let current_score = get_activation_score(&suitability, current_squash);

    // Find the best activation
    let (best_squash, best_score) = suitability.iter().max_by(|a, b| a.1.total_cmp(b.1))?;

    // Calculate improvement
    let improvement = best_score - current_score;

    // Only recommend if improvement is significant
    if improvement < MIN_IMPROVEMENT_THRESHOLD {
        return None;
    }

    // Don't recommend same activation
    if best_squash == current_squash {
        return None;
    }

    // Generate rationale
    let rationale = generate_rationale(&distribution, current_squash, best_squash);

    // Compute confidence based on sample size and improvement magnitude
    let sample_confidence = (records.len() as f32 / 100.0).min(1.0);
    let improvement_confidence = (improvement * 5.0).min(1.0);
    let confidence = (sample_confidence * 0.5 + improvement_confidence * 0.5).clamp(0.0, 1.0);

    Some(ActivationRecommendation {
        neuron_uuid,
        current_squash: current_squash.to_string(),
        recommended_squash: best_squash.clone(),
        confidence,
        expected_improvement: improvement * 0.02, // Scale to realistic improvement
        rationale,
        input_distribution: distribution.class,
    })
}

/// Get the score for an activation function.
///
/// Squash names are pre-normalised to uppercase at deserialisation (Issue #753),
/// so a direct HashMap lookup suffices.
fn get_activation_score(suitability: &HashMap<String, f32>, squash: &str) -> f32 {
    suitability.get(squash).copied().unwrap_or(0.5)
}

/// Generate a human-readable rationale for the recommendation.
fn generate_rationale(
    distribution: &InputDistribution,
    current_squash: &str,
    recommended_squash: &str,
) -> String {
    let distribution_desc = match distribution.class {
        InputDistributionClass::Gaussian => "Gaussian (bell-curve) distribution",
        InputDistributionClass::Sparse => "sparse distribution (many zeros)",
        InputDistributionClass::Bounded => "bounded distribution",
        InputDistributionClass::Uniform => "uniform distribution",
        InputDistributionClass::Bimodal => "bimodal distribution (two clusters)",
        InputDistributionClass::Unknown => "unknown distribution",
    };

    format!(
        "Input shows {} (mean={:.2}, std_dev={:.2}). {} better matches this pattern than {}.",
        distribution_desc,
        distribution.mean,
        distribution.std_dev,
        recommended_squash,
        current_squash
    )
}

// =============================================================================
// Coordinated Structural Candidate Conversion
// =============================================================================

/// Convert an activation recommendation to a coordinated structural candidate.
///
/// # Arguments
/// * `recommendation` - The activation recommendation to convert.
///
/// # Returns
/// A `CoordinatedStructuralCandidateJson` with a `changeSquash` operation.
pub fn recommendation_to_coordinated_candidate(
    recommendation: &ActivationRecommendation,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: recommendation.neuron_uuid.clone(),
            squash: recommendation.recommended_squash.clone(),
        }],
        expected_creature_score_gain: recommendation.expected_improvement,
        comment: Some(format!(
            "Proactive activation recommendation: {} → {} (confidence {:.0}%). {}",
            recommendation.current_squash,
            recommendation.recommended_squash,
            recommendation.confidence * 100.0,
            recommendation.rationale
        )),
    }
}

/// Convert multiple recommendations to coordinated structural candidates.
pub fn recommendations_to_coordinated_candidates(
    recommendations: &[ActivationRecommendation],
) -> Vec<CoordinatedStructuralCandidateJson> {
    recommendations
        .iter()
        .map(recommendation_to_coordinated_candidate)
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(uuid: &str, idx: u32, activation: f32) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(activation),
            activation,
            errors: vec![0.01],
        }
    }

    #[test]
    fn test_distribution_analysis_basic() {
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| make_record("test", i, (i as f32 - 25.0) / 10.0))
            .collect();

        let dist = analyse_input_distribution(&records);

        assert!(dist.mean.abs() < 0.1, "Mean should be near zero");
        assert!(dist.std_dev > 0.5, "Should have variance");
    }

    #[test]
    fn test_sparse_detection() {
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let activation = if i % 4 == 0 { 1.0 } else { 0.0 };
                make_record("sparse", i, activation)
            })
            .collect();

        let dist = analyse_input_distribution(&records);
        assert!(dist.sparsity > 0.5, "Should detect sparsity");
    }

    #[test]
    fn test_suitability_scores_not_empty() {
        let dist = InputDistribution {
            class: InputDistributionClass::Gaussian,
            mean: 0.0,
            std_dev: 1.0,
            min: -3.0,
            max: 3.0,
            sparsity: 0.0,
            kurtosis: 3.0,
        };

        let scores = classify_activation_suitability(&dist);
        assert!(!scores.is_empty(), "Should have suitability scores");
    }

    #[test]
    fn test_gradient_risk_for_saturated_tanh() {
        let records: Vec<DiscoverRecord> = (0..50).map(|i| make_record("sat", i, 0.99)).collect();

        let risk = analyse_gradient_flow_risk(&records, "TANH");
        assert!(
            risk > 0.5,
            "Should detect high gradient risk for saturated TANH"
        );
    }
}
