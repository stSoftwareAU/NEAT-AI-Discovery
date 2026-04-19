//! Activation compatibility scoring for neuron candidates.
//!
//! Penalises candidate squash functions that are likely to compound clipping
//! when feeding into bounded target activations (`HARD_TANH`, `TANH`, `LOGISTIC`,
//! `CLIPPED`). This does not filter candidates outright — it only adjusts
//! scoring to deprioritise inherently poor combinations while preserving
//! exploration.
//!
//! Issue #1113.

/// Compute a compatibility score in (0, 1] for a candidate squash function
/// feeding into a target squash function.
///
/// - Returns `1.0` for fully compatible combinations (e.g., `IDENTITY` → anything,
///   or any candidate → unbounded target).
/// - Returns `0.3–0.5` for problematic combinations where the candidate's output
///   range interacts poorly with the target's clipping bounds.
/// - Returns `0.8` for moderately compatible combinations (e.g., `TANH` → `HARD_TANH`).
pub fn activation_compatibility_score(candidate_squash: &str, target_squash: &str) -> f32 {
    let target_class = classify_target(target_squash);
    if target_class == TargetClass::Unbounded {
        // Any candidate feeding into an unbounded target has no clipping risk.
        return 1.0;
    }

    let candidate_class = classify_candidate(candidate_squash);
    match (candidate_class, target_class) {
        // Fully flexible candidates are always compatible.
        (CandidateClass::Flexible, _) => 1.0,

        // Symmetric bounded candidates (TANH, SOFTSIGN, ArcTan, HARD_TANH, CLIPPED)
        // feeding into symmetric bounded targets — similar range, moderate risk.
        (CandidateClass::SymmetricBounded, TargetClass::SymmetricBounded) => 0.8,

        // Symmetric bounded candidate → asymmetric bounded target (LOGISTIC [0,1]):
        // the negative half of the candidate's output maps below the target's
        // lower bound. Moderate penalty.
        (CandidateClass::SymmetricBounded, TargetClass::AsymmetricBounded) => 0.6,

        // Non-negative candidate (ABSOLUTE, Softplus, ReLU6) → symmetric bounded
        // target (HARD_TANH [-1,1]): positive-only output wastes the entire
        // negative half of the target's range. Heavy penalty.
        (CandidateClass::NonNegative, TargetClass::SymmetricBounded) => 0.3,

        // Non-negative candidate → asymmetric bounded target (LOGISTIC [0,1]):
        // the output is at least in the correct sign range. Moderate penalty
        // because uncapped upside still clips at the target max.
        (CandidateClass::NonNegative, TargetClass::AsymmetricBounded) => 0.6,

        // Asymmetric candidates (GELU, ELU, Mish, BENT_IDENTITY) have output
        // distributions that don't align well with symmetric clipping. Their
        // negative tails are truncated or compressed differently to positives.
        (CandidateClass::Asymmetric, TargetClass::SymmetricBounded) => 0.5,

        // Asymmetric candidate → asymmetric bounded target: slightly better
        // alignment since both are asymmetric.
        (CandidateClass::Asymmetric, TargetClass::AsymmetricBounded) => 0.7,

        // Discrete candidates (BIPOLAR) → bounded target: step output is
        // inherently clipping-aware but loses gradient information.
        (CandidateClass::Discrete, TargetClass::SymmetricBounded) => 0.7,
        (CandidateClass::Discrete, TargetClass::AsymmetricBounded) => 0.5,

        // Catch-all: unbounded target is already handled above.
        (_, TargetClass::Unbounded) => 1.0,
    }
}

// ============================================================================
// Internal classification types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateClass {
    /// Unbounded, passes through any range (`IDENTITY`).
    Flexible,
    /// Symmetric bounded output, e.g., `TANH` [-1,1], `SOFTSIGN`, `ArcTan`, `HARD_TANH`, `CLIPPED`.
    SymmetricBounded,
    /// Non-negative output only, e.g., `ABSOLUTE` [0,∞), `Softplus` [0,∞), `ReLU6` [0,6].
    NonNegative,
    /// Asymmetric output — negative tail truncated or compressed differently
    /// from positive (`GELU`, `ELU`, `Mish`, `BENT_IDENTITY`).
    Asymmetric,
    /// Discrete or step-function output (`BIPOLAR`).
    Discrete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetClass {
    /// Unbounded target — no clipping risk.
    Unbounded,
    /// Symmetric bounded target with negative lower bound (`HARD_TANH`, `TANH`, `CLIPPED`, `SOFTSIGN`).
    SymmetricBounded,
    /// Asymmetric bounded target with non-negative lower bound (`LOGISTIC` [0,1]).
    AsymmetricBounded,
}

fn classify_candidate(squash: &str) -> CandidateClass {
    match squash {
        "IDENTITY" => CandidateClass::Flexible,
        "TANH" | "HARD_TANH" | "CLIPPED" | "SOFTSIGN" | "ArcTan" => {
            CandidateClass::SymmetricBounded
        }
        "ABSOLUTE" | "Softplus" | "ReLU6" => CandidateClass::NonNegative,
        "GELU" | "ELU" | "Mish" | "BENT_IDENTITY" => CandidateClass::Asymmetric,
        "BIPOLAR" => CandidateClass::Discrete,
        "LOGISTIC" => CandidateClass::NonNegative, // [0, 1]
        _ => CandidateClass::Flexible,             // Unknown → assume flexible (conservative)
    }
}

fn classify_target(squash: &str) -> TargetClass {
    match squash {
        "HARD_TANH" | "TANH" | "CLIPPED" | "SOFTSIGN" | "BIPOLAR" => TargetClass::SymmetricBounded,
        "LOGISTIC" => TargetClass::AsymmetricBounded,
        _ => TargetClass::Unbounded,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Issue-specified test cases
    // ------------------------------------------------------------------

    #[test]
    fn identity_to_hard_tanh_is_fully_compatible() {
        let score = activation_compatibility_score("IDENTITY", "HARD_TANH");
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "IDENTITY → HARD_TANH should be 1.0, got {score}"
        );
    }

    #[test]
    fn absolute_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("ABSOLUTE", "HARD_TANH");
        assert!(
            score <= 0.5,
            "ABSOLUTE → HARD_TANH should be heavily penalised, got {score}"
        );
    }

    #[test]
    fn softplus_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("Softplus", "HARD_TANH");
        assert!(
            score <= 0.5,
            "Softplus → HARD_TANH should be penalised, got {score}"
        );
    }

    #[test]
    fn tanh_to_identity_is_fully_compatible() {
        let score = activation_compatibility_score("TANH", "IDENTITY");
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "TANH → IDENTITY should be 1.0, got {score}"
        );
    }

    // ------------------------------------------------------------------
    // Additional coverage (≥6 combinations as required)
    // ------------------------------------------------------------------

    #[test]
    fn gelu_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("GELU", "HARD_TANH");
        assert!(
            score <= 0.6,
            "GELU → HARD_TANH should be moderately penalised, got {score}"
        );
    }

    #[test]
    fn elu_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("ELU", "HARD_TANH");
        assert!(
            score <= 0.6,
            "ELU → HARD_TANH should be moderately penalised, got {score}"
        );
    }

    #[test]
    fn tanh_to_hard_tanh_is_moderately_compatible() {
        let score = activation_compatibility_score("TANH", "HARD_TANH");
        assert!(
            (score - 0.8).abs() < f32::EPSILON,
            "TANH → HARD_TANH should be 0.8, got {score}"
        );
    }

    #[test]
    fn relu6_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("ReLU6", "HARD_TANH");
        assert!(
            score <= 0.5,
            "ReLU6 → HARD_TANH should be penalised, got {score}"
        );
    }

    #[test]
    fn logistic_to_hard_tanh_is_penalised() {
        let score = activation_compatibility_score("LOGISTIC", "HARD_TANH");
        assert!(
            score <= 0.5,
            "LOGISTIC → HARD_TANH: non-negative output into symmetric target, got {score}"
        );
    }

    #[test]
    fn identity_to_identity_is_fully_compatible() {
        let score = activation_compatibility_score("IDENTITY", "IDENTITY");
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "IDENTITY → IDENTITY should be 1.0, got {score}"
        );
    }

    #[test]
    fn mish_to_tanh_is_penalised() {
        let score = activation_compatibility_score("Mish", "TANH");
        assert!(
            score <= 0.6,
            "Mish → TANH should be moderately penalised, got {score}"
        );
    }

    // ------------------------------------------------------------------
    // Unbounded target — all candidates should be 1.0
    // ------------------------------------------------------------------

    #[test]
    fn any_candidate_to_unbounded_target_is_fully_compatible() {
        for candidate in &[
            "IDENTITY", "TANH", "ABSOLUTE", "Softplus", "GELU", "ELU", "Mish", "BIPOLAR",
        ] {
            let score = activation_compatibility_score(candidate, "IDENTITY");
            assert!(
                (score - 1.0).abs() < f32::EPSILON,
                "{candidate} → IDENTITY should be 1.0, got {score}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Score is always in valid range
    // ------------------------------------------------------------------

    #[test]
    fn all_spec_combinations_produce_valid_scores() {
        use crate::analysis::activation::ACTIVATION_SPECS;
        let targets = [
            "HARD_TANH",
            "TANH",
            "LOGISTIC",
            "CLIPPED",
            "IDENTITY",
            "GELU",
            "ELU",
        ];
        for spec in &ACTIVATION_SPECS {
            for &target in &targets {
                let score = activation_compatibility_score(spec.name, target);
                assert!(
                    score > 0.0 && score <= 1.0,
                    "Score for {} → {} must be in (0, 1], got {score}",
                    spec.name,
                    target,
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Unknown candidate squash defaults to flexible (1.0)
    // ------------------------------------------------------------------

    #[test]
    fn unknown_candidate_defaults_to_flexible() {
        let score = activation_compatibility_score("UNKNOWN_ACTIVATION", "HARD_TANH");
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "Unknown candidate should default to 1.0, got {score}"
        );
    }
}
