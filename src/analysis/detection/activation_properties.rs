//! Shared activation function classification helpers.
//!
//! Consolidates the activation function property queries that were previously
//! duplicated across `saturation.rs` and `weight_coherence.rs` (Issue #768).

/// Returns whether a squash function is bounded and can saturate.
///
/// Bounded activation functions have finite upper and lower limits, meaning
/// their output cannot grow without bound. When a neuron's pre-activation
/// input is extreme, a bounded function "saturates" — its output stops
/// changing even as the input keeps moving.
///
/// Unbounded functions like `RELU` and `IDENTITY` cannot saturate
/// (they have no ceiling). `RELU` can have a "dead zone" (all outputs at 0),
/// which is handled separately by [`can_have_dead_zone`].
///
/// # Examples
///
/// ```
/// use neat_ai_discovery::analysis::detection::activation_properties::is_bounded_squash;
///
/// assert!(is_bounded_squash("TANH"));
/// assert!(is_bounded_squash("LOGISTIC"));
/// assert!(!is_bounded_squash("RELU"));
/// assert!(!is_bounded_squash("IDENTITY"));
/// ```
pub fn is_bounded_squash(squash: &str) -> bool {
    matches!(
        squash,
        "TANH"
            | "LOGISTIC"
            | "HARD_TANH"
            | "CLIPPED"
            | "BIPOLAR"
            | "BIPOLAR_SIGMOID"
            | "STEP"
            | "SOFTSIGN"
            | "ISRU"
            | "ARCTAN"
            | "RELU6"
    )
}

/// Returns whether a squash function is a saturating type with bounded output.
///
/// This is a subset of [`is_bounded_squash`] focused on smooth, differentiable
/// activation functions commonly used in hidden layers. It identifies functions
/// whose gradients vanish at extreme input values, causing weight coherence
/// issues.
///
/// The input is case-insensitive — it uses `eq_ignore_ascii_case` to avoid
/// String allocations (Issue #771).
///
/// # Examples
///
/// ```
/// use neat_ai_discovery::analysis::detection::activation_properties::is_saturating_squash;
///
/// assert!(is_saturating_squash("TANH"));
/// assert!(is_saturating_squash("logistic"));
/// assert!(!is_saturating_squash("RELU"));
/// assert!(!is_saturating_squash("IDENTITY"));
/// ```
pub fn is_saturating_squash(squash: &str) -> bool {
    squash.eq_ignore_ascii_case("TANH")
        || squash.eq_ignore_ascii_case("LOGISTIC")
        || squash.eq_ignore_ascii_case("SIGMOID")
        || squash.eq_ignore_ascii_case("HARD_TANH")
        || squash.eq_ignore_ascii_case("CLIPPED")
        || squash.eq_ignore_ascii_case("SOFTSIGN")
}

/// Returns whether a squash function can have a dead zone (all outputs at zero).
///
/// ReLU-family activations output zero for all negative inputs. If a neuron's
/// bias or incoming weights push its pre-activation consistently negative, the
/// neuron becomes "dead" — producing zero output regardless of input.
///
/// # Examples
///
/// ```
/// use neat_ai_discovery::analysis::detection::activation_properties::can_have_dead_zone;
///
/// assert!(can_have_dead_zone("RELU"));
/// assert!(can_have_dead_zone("ELU"));
/// assert!(!can_have_dead_zone("TANH"));
/// assert!(!can_have_dead_zone("IDENTITY"));
/// ```
pub fn can_have_dead_zone(squash: &str) -> bool {
    matches!(squash, "RELU" | "LEAKYRELU" | "ELU" | "SELU")
}
