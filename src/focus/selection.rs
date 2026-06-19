//! Diversity-aware focus-neuron selection (Issue #1445).
//!
//! ## Why this exists
//!
//! On a plateaued mature network a single high-impact neuron can hold the vast
//! majority of the focus-selection roulette weight. On the production GRQ-3
//! creature one neuron held ~98.5% of the weight, so the weighted roulette
//! collapsed to a **single target**: the other five "selected" neurons were
//! lottery noise with negligible selection probability, and discovery revisited
//! the same dominant neuron almost every pass.
//!
//! Impact-weighted ranking ([`crate::focus::rank_focus_neurons`]) only *orders*
//! neurons — it does not enforce **diversity** in the final selected set. This
//! module adds two guards over the ranked list:
//!
//! 1. **Diversity floor** — when one neuron's roulette weight exceeds its even
//!    `1/N` share, pick the final focus set **stratified** across the ranked
//!    list so every band (quartile) is represented, not just the dominant head.
//!    The realised selection then spreads `N` distinct, meaningful targets
//!    instead of "dominant + N−1 noise".
//! 2. **Drought-aware rotation** — once the creature has been in drought for
//!    longer than the configured threshold, switch from weighted selection to
//!    **round-robin** across the top `K × N` ranked neurons (K =
//!    [`DROUGHT_ROTATION_POOL_FACTOR`]) so unexplored targets get analysis
//!    budget on successive passes instead of repeating the same id.
//!
//! It also reports [`FocusSelection::weight_concentration_ratio`] (max weight ÷
//! sum) so callers can log it and WARN when the raw roulette is pathologically
//! concentrated.
//!
//! The selection is **deterministic**: given the same ranked input, target size
//! and rotation offset it always returns the same set, which keeps it testable
//! and reproducible across discovery passes.

// Intentional usize→f32 casts for selection-set sizes and rotation offsets.
// Focus sets are tiny (single digits) and pools are bounded, so the f32
// mantissa is never a precision concern here (mirrors `score_calculation.rs`).
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// Concentration ratio (max weight ÷ total weight) at or above which the raw
/// roulette is considered pathologically single-target and a WARN is emitted
/// (Issue #1445, acceptance criterion 3).
pub const CONCENTRATION_WARN_THRESHOLD: f32 = 0.5;

/// Pool size multiplier `K` for drought-aware round-robin rotation: rotation
/// draws from the top `K × N` ranked neurons so the unexplored tail of the
/// ranking gets analysis budget over successive passes (Issue #1445).
pub const DROUGHT_ROTATION_POOL_FACTOR: usize = 3;

/// A focus candidate: a neuron and its raw roulette weight (the impact-weighted
/// ranking score). Weights are expected to be non-negative; non-finite or
/// negative weights are treated as `0.0`.
#[derive(Debug, Clone)]
pub struct FocusCandidate {
    pub neuron_uuid: String,
    pub weight: f32,
}

/// The outcome of a focus-selection pass (Issue #1445).
#[derive(Debug, Clone, PartialEq)]
pub struct FocusSelection {
    /// Selected neuron uuids, in selection order.
    pub selected: Vec<String>,
    /// Concentration ratio of the **raw** roulette weights over the candidate
    /// pool (max ÷ sum). This is the diagnostic that exposes the problem — on
    /// the GRQ-3 fixture it is ~0.985.
    pub raw_weight_concentration_ratio: f32,
    /// Concentration ratio of the **effective** selection after any diversity
    /// floor or rotation is applied. When a guard fires the `N` chosen targets
    /// each receive an equal share of the analysis budget, so this is
    /// `1 / selected.len()` — below [`CONCENTRATION_WARN_THRESHOLD`] for any
    /// `N >= 3`.
    pub weight_concentration_ratio: f32,
    /// Whether the diversity floor reshaped the selection (a single neuron
    /// exceeded its even `1/N` share).
    pub diversity_floor_applied: bool,
    /// Whether drought-aware round-robin rotation was used instead of weighted
    /// selection.
    pub rotation_applied: bool,
    /// Number of candidates considered for selection (the rotation pool size
    /// under drought, otherwise the full candidate count).
    pub pool_size: usize,
}

/// Compute the weight-concentration ratio (max weight ÷ sum of weights) for a
/// slice of roulette weights. Non-finite and negative weights are clamped to
/// `0.0`. Returns `0.0` when the total weight is zero (no signal).
#[must_use]
pub fn weight_concentration_ratio(weights: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    for &w in weights {
        let w = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        sum += w;
        if w > max {
            max = w;
        }
    }
    if sum > 0.0 { max / sum } else { 0.0 }
}

/// Select a diverse focus set of up to `target_n` neurons from an
/// impact-ranked candidate list (Issue #1445).
///
/// `ranked` must be ordered strongest-first (as produced by
/// [`crate::focus::rank_focus_neurons`]). `drought_active` switches selection
/// to drought-aware round-robin rotation, and `rotation_offset` (e.g. the
/// creature's `epochs_since_last_accepted_candidate`) advances the round-robin
/// cursor so successive passes pick fresh targets.
#[must_use]
pub fn select_focus_neurons(
    ranked: &[FocusCandidate],
    target_n: usize,
    drought_active: bool,
    rotation_offset: u64,
) -> FocusSelection {
    let n = target_n.max(1);
    let weights: Vec<f32> = ranked
        .iter()
        .map(|c| {
            if c.weight.is_finite() && c.weight > 0.0 {
                c.weight
            } else {
                0.0
            }
        })
        .collect();
    let raw_ratio = weight_concentration_ratio(&weights);

    if ranked.is_empty() {
        return FocusSelection {
            selected: Vec::new(),
            raw_weight_concentration_ratio: 0.0,
            weight_concentration_ratio: 0.0,
            diversity_floor_applied: false,
            rotation_applied: false,
            pool_size: 0,
        };
    }

    // Drought-aware rotation supersedes weighted selection: round-robin across
    // the top K×N ranked neurons so the unexplored tail gets budget.
    if drought_active {
        let pool_size = (DROUGHT_ROTATION_POOL_FACTOR * n).min(ranked.len()).max(1);
        let take = n.min(pool_size);
        let start = (rotation_offset % pool_size as u64) as usize;
        let selected: Vec<String> = (0..take)
            .map(|j| ranked[(start + j) % pool_size].neuron_uuid.clone())
            .collect();
        let effective = if selected.is_empty() {
            0.0
        } else {
            1.0 / selected.len() as f32
        };
        return FocusSelection {
            selected,
            raw_weight_concentration_ratio: raw_ratio,
            weight_concentration_ratio: effective,
            diversity_floor_applied: false,
            rotation_applied: true,
            pool_size,
        };
    }

    // Apply the diversity floor when a single neuron exceeds its even 1/N share
    // of the roulette weight — otherwise the weighted ordering is already
    // spread and the plain top-N is fine.
    let even_share = 1.0 / n as f32;
    let apply_floor = raw_ratio > even_share + f32::EPSILON;

    if apply_floor && ranked.len() > n {
        // Stratified pick: divide the ranked list into N contiguous bands and
        // take the strongest (first) neuron of each band. Band 0 keeps the
        // dominant neuron; later bands draw from progressively lower-ranked
        // regions, guaranteeing quartile-style coverage of the whole list.
        let len = ranked.len();
        let selected: Vec<String> = (0..n)
            .map(|i| {
                let idx = i * len / n;
                ranked[idx].neuron_uuid.clone()
            })
            .collect();
        let effective = 1.0 / selected.len() as f32;
        return FocusSelection {
            selected,
            raw_weight_concentration_ratio: raw_ratio,
            weight_concentration_ratio: effective,
            diversity_floor_applied: true,
            rotation_applied: false,
            pool_size: len,
        };
    }

    // No guard needed (or fewer candidates than the target): take the strongest
    // up to N in ranked order.
    let take = n.min(ranked.len());
    let selected: Vec<String> = ranked
        .iter()
        .take(take)
        .map(|c| c.neuron_uuid.clone())
        .collect();
    let selected_weights: Vec<f32> = weights.iter().copied().take(take).collect();
    let effective = if apply_floor {
        // Floor wanted but pool too small to stratify — equal-share the picks.
        1.0 / selected.len().max(1) as f32
    } else {
        weight_concentration_ratio(&selected_weights)
    };
    FocusSelection {
        selected,
        raw_weight_concentration_ratio: raw_ratio,
        weight_concentration_ratio: effective,
        diversity_floor_applied: apply_floor,
        rotation_applied: false,
        pool_size: ranked.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(uuid: &str, weight: f32) -> FocusCandidate {
        FocusCandidate {
            neuron_uuid: uuid.to_string(),
            weight,
        }
    }

    /// Build a plateau pool: one dominant neuron with `dom`× the weight of
    /// `n_others` equal low-weight neurons, ordered strongest-first.
    fn plateau_pool(dom: f32, n_others: usize) -> Vec<FocusCandidate> {
        let mut v = vec![candidate("dominant", dom)];
        for i in 0..n_others {
            v.push(candidate(&format!("n{i}"), 1.0));
        }
        v
    }

    #[test]
    fn concentration_ratio_basic() {
        assert!((weight_concentration_ratio(&[1.0, 1.0, 1.0, 1.0]) - 0.25).abs() < 1e-6);
        assert!((weight_concentration_ratio(&[97.0, 1.0, 1.0, 1.0]) - 0.97).abs() < 1e-6);
        // Empty / zero-sum is well-defined.
        assert_eq!(weight_concentration_ratio(&[]), 0.0);
        assert_eq!(weight_concentration_ratio(&[0.0, 0.0]), 0.0);
        // Non-finite / negative weights are ignored.
        assert!((weight_concentration_ratio(&[f32::NAN, 2.0, 2.0]) - 0.5).abs() < 1e-6);
        assert_eq!(weight_concentration_ratio(&[-5.0, 0.0]), 0.0);
    }

    #[test]
    fn dominant_neuron_triggers_diversity_floor() {
        // One neuron with 100× the error×impact of 19 others (the GRQ-3 shape).
        let pool = plateau_pool(100.0, 19);
        let sel = select_focus_neurons(&pool, 6, false, 0);

        assert!(sel.diversity_floor_applied);
        assert!(!sel.rotation_applied);
        // Raw roulette is pathologically concentrated...
        assert!(sel.raw_weight_concentration_ratio > CONCENTRATION_WARN_THRESHOLD);
        // ...but the effective selection is well under the warn threshold (AC1).
        assert!(sel.weight_concentration_ratio < CONCENTRATION_WARN_THRESHOLD);
        // The dominant neuron is still picked (it is genuinely high-value)...
        assert_eq!(sel.selected[0], "dominant");
        // ...and the rest are distinct, spread targets — not duplicates.
        let distinct: std::collections::HashSet<&String> = sel.selected.iter().collect();
        assert_eq!(distinct.len(), 6);
    }

    #[test]
    fn ac3_three_consecutive_calls_yield_at_least_three_distinct_targets() {
        // Acceptance criterion 3: synthetic plateau where one neuron has 100×
        // the error×impact of the others still yields >=3 distinct focus
        // targets across 3 consecutive selection calls.
        let pool = plateau_pool(100.0, 19);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            let sel = select_focus_neurons(&pool, 6, false, 0);
            for id in sel.selected {
                seen.insert(id);
            }
        }
        assert!(
            seen.len() >= 3,
            "expected >=3 distinct targets, got {seen:?}"
        );
    }

    #[test]
    fn drought_rotation_rotates_targets_across_passes() {
        // Acceptance criterion 2: after the drought threshold, focus neurons
        // rotate across the top-ranked pool instead of repeating the same id.
        let pool = plateau_pool(100.0, 19);
        let mut union = std::collections::HashSet::new();
        let mut sets = Vec::new();
        for epoch in 0..3u64 {
            let sel = select_focus_neurons(&pool, 6, true, epoch);
            assert!(sel.rotation_applied);
            assert!(sel.weight_concentration_ratio < CONCENTRATION_WARN_THRESHOLD);
            for id in &sel.selected {
                union.insert(id.clone());
            }
            sets.push(sel.selected);
        }
        // Successive passes must not be identical — the cursor advanced.
        assert_ne!(sets[0], sets[1]);
        assert_ne!(sets[1], sets[2]);
        assert!(union.len() >= 3);
        // Pool is K×N capped to the available candidates (3×6 = 18 here).
        assert_eq!(
            select_focus_neurons(&pool, 6, true, 0).pool_size,
            DROUGHT_ROTATION_POOL_FACTOR * 6
        );
    }

    #[test]
    fn even_distribution_keeps_weighted_top_n() {
        // Twelve equally-weighted neurons: no single neuron dominates, so the
        // diversity floor does not fire and the plain top-N is returned.
        let pool: Vec<FocusCandidate> = (0..12).map(|i| candidate(&format!("n{i}"), 1.0)).collect();
        let sel = select_focus_neurons(&pool, 6, false, 0);
        assert!(!sel.diversity_floor_applied);
        assert!(!sel.rotation_applied);
        assert_eq!(sel.selected.len(), 6);
        assert_eq!(sel.selected[0], "n0");
        assert!(sel.weight_concentration_ratio < CONCENTRATION_WARN_THRESHOLD);
    }

    #[test]
    fn fewer_candidates_than_target_returns_all() {
        let pool = plateau_pool(100.0, 2); // 3 candidates, target 6
        let sel = select_focus_neurons(&pool, 6, false, 0);
        assert_eq!(sel.selected.len(), 3);
        let distinct: std::collections::HashSet<&String> = sel.selected.iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn empty_pool_is_safe() {
        let sel = select_focus_neurons(&[], 6, false, 0);
        assert!(sel.selected.is_empty());
        assert_eq!(sel.raw_weight_concentration_ratio, 0.0);
        assert_eq!(sel.weight_concentration_ratio, 0.0);
        assert!(!sel.diversity_floor_applied);
        assert!(!sel.rotation_applied);
    }

    #[test]
    fn target_one_is_clamped_and_safe() {
        let pool = plateau_pool(100.0, 5);
        let sel = select_focus_neurons(&pool, 0, false, 0);
        // target_n is clamped to 1.
        assert_eq!(sel.selected.len(), 1);
        assert_eq!(sel.selected[0], "dominant");
    }

    #[test]
    fn drought_rotation_smaller_pool_than_target() {
        // Only 2 candidates but target 6: pool clamps and no out-of-bounds.
        let pool = plateau_pool(100.0, 1); // 2 candidates
        let sel = select_focus_neurons(&pool, 6, true, 5);
        assert!(sel.rotation_applied);
        assert_eq!(sel.selected.len(), 2);
        let distinct: std::collections::HashSet<&String> = sel.selected.iter().collect();
        assert_eq!(distinct.len(), 2);
    }
}
