//! Exploit/explore focus-neuron selection (Issue #1662, supersedes #1445).
//!
//! ## Why this exists
//!
//! Issue #1445 corrected a 98.5%-concentrated roulette by replacing weighted
//! selection with broad deterministic diversity — full-list *stratification*
//! outside drought and *top-`K × N` round-robin* during drought. That
//! over-corrected: on a production creature with ~1,661 eligible hidden neurons
//! and `N = 16`, stratification kept only **one** of sixteen focus slots in the
//! highest-ranked neighbourhood, moving expensive analysis budget away from the
//! neurons the ranking said were most likely to yield successful candidates —
//! without actually guaranteeing complete exploration (the drought pool was
//! capped at `3 × N`).
//!
//! This module replaces both behaviours with a deterministic **exploit/explore**
//! allocation over the impact-ranked list ([`crate::focus::rank_focus_neurons`],
//! whose weight already folds in error, impact, gradient/frequency,
//! reconstruction mismatch and the optional Bayesian success-history
//! multiplier):
//!
//! 1. **Exploitation majority** — most slots take the highest-ranked,
//!    highest-weight neurons (default ≥80% outside drought, always a strict
//!    majority). This exploits the ranking evidence instead of flattening it.
//! 2. **Bounded exploration quota** — the remaining slots (default 20%, at least
//!    one when the set has capacity) rotate deterministically through the
//!    **complete eligible neuron list** via a monotonic per-creature cursor.
//! 3. **Eventual full coverage** — the cursor is supplied by the caller and does
//!    **not** reset when a candidate succeeds, so across enough passes every
//!    eligible candidate-producing neuron is selected.
//! 4. **Drought stays exploitative** — drought widens the exploration quota
//!    ([`DROUGHT_EXPLORATION_FRACTION`]) but exploitation remains a strict
//!    majority (>50%).
//!
//! It also reports [`FocusSelection::weight_concentration_ratio`] (max weight ÷
//! sum) so callers can log it and WARN when the raw roulette is pathologically
//! concentrated.
//!
//! The selection is **deterministic**: identical ranked input, target size and
//! cursor always return the same set — no random selection — which keeps it
//! testable and reproducible across discovery passes.

// Intentional usize→f32 casts for selection-set sizes and cursor offsets. Focus
// sets are tiny (single digits) and pools are bounded, so the f32 mantissa is
// never a precision concern here (mirrors `score_calculation.rs`).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// Concentration ratio (max weight ÷ total weight) at or above which the raw
/// roulette is considered pathologically single-target and a WARN is emitted
/// (Issue #1445, retained as a diagnostic under #1662).
pub const CONCENTRATION_WARN_THRESHOLD: f32 = 0.5;

/// Fraction of the focus set reserved for deterministic exploration **outside**
/// drought (Issue #1662, acceptance criterion 1: exploitation keeps ≥80%).
pub const DEFAULT_EXPLORATION_FRACTION: f32 = 0.2;

/// Fraction of the focus set reserved for exploration **during** drought. Larger
/// than [`DEFAULT_EXPLORATION_FRACTION`] so a plateaued creature explores more,
/// but capped downstream so exploitation always keeps a strict majority (Issue
/// #1662, acceptance criterion 4).
pub const DROUGHT_EXPLORATION_FRACTION: f32 = 0.4;

/// A focus candidate: a neuron and its raw roulette weight (the impact-weighted
/// ranking score, already including any Bayesian success-history multiplier).
/// Weights are expected to be non-negative; non-finite or negative weights are
/// treated as `0.0`.
#[derive(Debug, Clone)]
pub struct FocusCandidate {
    pub neuron_uuid: String,
    pub weight: f32,
}

/// The outcome of a focus-selection pass (Issue #1662).
#[derive(Debug, Clone, PartialEq)]
pub struct FocusSelection {
    /// Selected neuron uuids, in selection order (exploitation head first, then
    /// the exploration picks).
    pub selected: Vec<String>,
    /// Concentration ratio of the **raw** roulette weights over the candidate
    /// pool (max ÷ sum). Exposes single-target collapse — ~0.985 on the GRQ-3
    /// fixture.
    pub raw_weight_concentration_ratio: f32,
    /// Concentration ratio of the **selected** set's weights (max ÷ sum). Lower
    /// than the raw ratio because the exploration quota spreads budget.
    pub weight_concentration_ratio: f32,
    /// Slots filled by ranking/history exploitation (the strongest neurons).
    pub exploitation_count: usize,
    /// Slots filled by deterministic exploration rotation over the eligible tail.
    pub exploration_count: usize,
    /// The monotonic per-creature cursor that seeded exploration this pass.
    pub exploration_cursor: u64,
    /// Total eligible candidate-producing neurons available for selection.
    pub eligible_pool_size: usize,
    /// Best-effort cumulative coverage: distinct eligible neurons the selection
    /// has reached by exploration/exploitation across cursors `0..=cursor`.
    /// Stateless upper estimate — the caller owns the authoritative history.
    pub cumulative_coverage: usize,
    /// Whether drought widened the exploration quota this pass.
    pub drought_active: bool,
    /// Number of candidates considered (== [`Self::eligible_pool_size`]). Kept
    /// for logging compatibility.
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

/// Resolve the exploration slot count for a focus set of `n` under the given
/// drought state. Guarantees exploitation keeps a strict majority: the returned
/// value never exceeds `(n - 1) / 2`, and it reserves at least one exploration
/// slot when the set has capacity (`n >= 3`).
fn exploration_quota(n: usize, drought_active: bool) -> usize {
    let fraction = if drought_active {
        DROUGHT_EXPLORATION_FRACTION
    } else {
        DEFAULT_EXPLORATION_FRACTION
    };
    // Strict exploitation majority: exploit = n - explore must exceed n / 2, so
    // explore < n / 2, i.e. explore <= (n - 1) / 2 in integer arithmetic.
    let max_explore = n.saturating_sub(1) / 2;
    let mut explore = (n as f32 * fraction).round() as usize;
    if max_explore >= 1 && explore == 0 {
        // Reserve at least one exploration slot when the set has capacity.
        explore = 1;
    }
    explore.min(max_explore)
}

/// Deterministically pick `explore` exploration indices from the eligible tail
/// `[exploit, pool_len)`, rotating by `cursor` so successive passes sweep fresh
/// neurons. The walk stays entirely within the tail — the contiguous
/// exploitation head `[0, exploit)` is never revisited, which satisfies the
/// no-duplicate-slot guarantee — and advances by `explore` per pass, so every
/// tail neuron is covered within `ceil(tail_len / explore)` passes and coverage
/// is eventual regardless of candidate success.
fn exploration_indices(pool_len: usize, exploit: usize, explore: usize, cursor: u64) -> Vec<usize> {
    if explore == 0 || exploit >= pool_len {
        return Vec::new();
    }
    let tail_len = pool_len - exploit;
    // `pool_len > n` at every call site guarantees `tail_len > explore`, so the
    // `explore` offsets below are distinct within a single pass.
    let base = (cursor.wrapping_mul(explore as u64) % tail_len as u64) as usize;
    (0..explore)
        .map(|k| exploit + (base + k) % tail_len)
        .collect()
}

/// Select a focus set of up to `target_n` neurons from an impact-ranked
/// candidate list using deterministic exploit/explore allocation (Issue #1662).
///
/// `ranked` must be ordered strongest-first (as produced by
/// [`crate::focus::rank_focus_neurons`]) and must already exclude proven-
/// ineligible neurons (input, constant, demonstrably zero-impact). `cursor` is a
/// monotonic per-creature focus-selection cursor that the caller advances every
/// pass and **never resets on candidate success**, so exploration eventually
/// covers every eligible neuron. `drought_active` widens the exploration quota
/// while keeping exploitation a strict majority.
#[must_use]
pub fn select_focus_neurons(
    ranked: &[FocusCandidate],
    target_n: usize,
    drought_active: bool,
    cursor: u64,
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
    let pool_len = ranked.len();

    if pool_len == 0 {
        return FocusSelection {
            selected: Vec::new(),
            raw_weight_concentration_ratio: 0.0,
            weight_concentration_ratio: 0.0,
            exploitation_count: 0,
            exploration_count: 0,
            exploration_cursor: cursor,
            eligible_pool_size: 0,
            cumulative_coverage: 0,
            drought_active,
            pool_size: 0,
        };
    }

    // When the eligible pool fits inside the focus set, analyse everything — no
    // exploit/explore split is needed and full coverage is trivially complete.
    if pool_len <= n {
        let selected: Vec<String> = ranked.iter().map(|c| c.neuron_uuid.clone()).collect();
        let effective = weight_concentration_ratio(&weights);
        return FocusSelection {
            selected,
            raw_weight_concentration_ratio: raw_ratio,
            weight_concentration_ratio: effective,
            exploitation_count: pool_len,
            exploration_count: 0,
            exploration_cursor: cursor,
            eligible_pool_size: pool_len,
            cumulative_coverage: pool_len,
            drought_active,
            pool_size: pool_len,
        };
    }

    let explore = exploration_quota(n, drought_active);
    let exploit = n - explore;

    // Exploitation: the strongest `exploit` neurons, in ranked order.
    let mut selected: Vec<String> = ranked
        .iter()
        .take(exploit)
        .map(|c| c.neuron_uuid.clone())
        .collect();
    let mut selected_indices: Vec<usize> = (0..exploit).collect();

    // Exploration: rotate through the eligible tail via the monotonic cursor.
    let explore_idx = exploration_indices(pool_len, exploit, explore, cursor);
    for &idx in &explore_idx {
        selected.push(ranked[idx].neuron_uuid.clone());
        selected_indices.push(idx);
    }
    let exploration_count = explore_idx.len();

    let selected_weights: Vec<f32> = selected_indices.iter().map(|&i| weights[i]).collect();
    let effective = weight_concentration_ratio(&selected_weights);

    // Best-effort cumulative coverage: the exploitation head is selected every
    // pass, and the exploration walk sweeps `explore` fresh tail neurons per
    // pass, so after `cursor + 1` passes it has covered up to the whole tail.
    let tail_len = pool_len - exploit;
    let passes = cursor.saturating_add(1);
    let tail_covered = passes.saturating_mul(explore as u64).min(tail_len as u64) as usize;
    let cumulative_coverage = (exploit + tail_covered).min(pool_len);

    FocusSelection {
        selected,
        raw_weight_concentration_ratio: raw_ratio,
        weight_concentration_ratio: effective,
        exploitation_count: exploit,
        exploration_count,
        exploration_cursor: cursor,
        eligible_pool_size: pool_len,
        cumulative_coverage,
        drought_active,
        pool_size: pool_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn candidate(uuid: &str, weight: f32) -> FocusCandidate {
        FocusCandidate {
            neuron_uuid: uuid.to_string(),
            weight,
        }
    }

    /// Build `n` candidates with strictly descending weights, ordered
    /// strongest-first (`c0` is the highest-ranked).
    fn descending_pool(n: usize) -> Vec<FocusCandidate> {
        (0..n)
            .map(|i| candidate(&format!("c{i}"), (n - i) as f32))
            .collect()
    }

    #[test]
    fn concentration_ratio_basic() {
        assert!((weight_concentration_ratio(&[1.0, 1.0, 1.0, 1.0]) - 0.25).abs() < 1e-6);
        assert!((weight_concentration_ratio(&[97.0, 1.0, 1.0, 1.0]) - 0.97).abs() < 1e-6);
        assert_eq!(weight_concentration_ratio(&[]), 0.0);
        assert_eq!(weight_concentration_ratio(&[0.0, 0.0]), 0.0);
        assert!((weight_concentration_ratio(&[f32::NAN, 2.0, 2.0]) - 0.5).abs() < 1e-6);
        assert_eq!(weight_concentration_ratio(&[-5.0, 0.0]), 0.0);
    }

    // TDD proof 1 (Issue #1662): likely-success majority. 100 descending-weight
    // candidates, focus size 10 => at least 8 come from the ranked exploitation
    // head and the highest-ranked candidate is retained.
    #[test]
    fn exploitation_majority_retains_ranked_head() {
        let pool = descending_pool(100);
        let sel = select_focus_neurons(&pool, 10, false, 0);

        assert_eq!(sel.selected.len(), 10);
        assert!(
            sel.exploitation_count >= 8,
            "expected >=8 exploitation slots, got {}",
            sel.exploitation_count
        );
        // The exploitation head is the strongest prefix of the ranked list.
        for (i, uuid) in sel.selected.iter().take(sel.exploitation_count).enumerate() {
            assert_eq!(uuid, &format!("c{i}"));
        }
        // The single highest-ranked candidate is always retained.
        assert_eq!(sel.selected[0], "c0");
        assert_eq!(sel.exploration_count, 10 - sel.exploitation_count);
    }

    // TDD proof 2 (Issue #1662): eventual coverage. Focus size 10, two
    // exploration slots; 50 consecutive monotonic cursor positions cover all
    // 100 eligible candidates while the exploitation majority stays present.
    #[test]
    fn exploration_reaches_full_coverage() {
        let pool = descending_pool(100);
        let mut covered: HashSet<String> = HashSet::new();
        for cursor in 0..50u64 {
            let sel = select_focus_neurons(&pool, 10, false, cursor);
            assert_eq!(sel.exploration_count, 2, "expected two exploration slots");
            assert!(
                sel.exploitation_count > sel.exploration_count,
                "exploitation majority must persist each pass"
            );
            for id in sel.selected {
                covered.insert(id);
            }
        }
        assert_eq!(
            covered.len(),
            100,
            "50 cursor positions must cover every eligible candidate"
        );
    }

    // TDD proof 3 (Issue #1662): the cursor survives candidate success — an
    // accepted candidate between passes does not restart exploration at rank 0.
    #[test]
    fn cursor_survives_success() {
        let pool = descending_pool(100);
        // Pass at cursor 7 (some exploration already happened).
        let before = select_focus_neurons(&pool, 10, false, 7);
        // A candidate is "accepted": drought resets, but the monotonic
        // focus-selection cursor keeps advancing rather than resetting to 0.
        let after_success = select_focus_neurons(&pool, 10, false, 8);
        let restart = select_focus_neurons(&pool, 10, false, 0);

        let explore_before: Vec<&String> = before
            .selected
            .iter()
            .skip(before.exploitation_count)
            .collect();
        let explore_after: Vec<&String> = after_success
            .selected
            .iter()
            .skip(after_success.exploitation_count)
            .collect();
        let explore_restart: Vec<&String> = restart
            .selected
            .iter()
            .skip(restart.exploitation_count)
            .collect();

        // Continuing from the advanced cursor differs from restarting at 0.
        assert_ne!(explore_after, explore_restart);
        // And it advances beyond the previous pass rather than repeating it.
        assert_ne!(explore_after, explore_before);
    }

    // TDD proof 4 (Issue #1662): drought remains exploitative. Drought increases
    // exploration within its cap but still assigns >50% of slots by ranking.
    #[test]
    fn drought_widens_exploration_but_keeps_majority() {
        let pool = descending_pool(100);
        let normal = select_focus_neurons(&pool, 10, false, 0);
        let drought = select_focus_neurons(&pool, 10, true, 0);

        assert!(
            drought.exploration_count > normal.exploration_count,
            "drought must widen exploration ({} !> {})",
            drought.exploration_count,
            normal.exploration_count
        );
        assert!(
            drought.exploitation_count * 2 > drought.selected.len(),
            "drought must keep a strict exploitation majority, got {}/{}",
            drought.exploitation_count,
            drought.selected.len()
        );
        assert!(drought.drought_active);
    }

    // TDD proof 5 (Issue #1662): history matters. Two otherwise-equal neurons
    // with different Bayesian success histories (reflected in weight) order
    // into/out of exploitation consistently.
    #[test]
    fn success_history_orders_into_exploitation() {
        // `high` has the better success history (larger weight) so it ranks
        // above `low`; the exploitation boundary (top 8 for N=10) sits between
        // them — `high` at rank 7, `low` at rank 8.
        let mut pool = descending_pool(20);
        pool[7] = candidate("high", 13.0);
        pool[8] = candidate("low", 12.5);
        // Re-sort strongest-first to mimic rank_focus_neurons ordering.
        pool.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

        let sel = select_focus_neurons(&pool, 10, false, 0);
        let head: HashSet<&String> = sel.selected.iter().take(sel.exploitation_count).collect();
        assert!(
            head.contains(&"high".to_string()),
            "higher success-history neuron must be exploited"
        );
        assert!(
            !head.contains(&"low".to_string()),
            "lower success-history neuron must fall outside the exploitation head"
        );
    }

    // TDD proof 6 (Issue #1662): no duplicate slots. A neuron chosen for
    // exploitation is skipped by the exploration cursor for that pass.
    #[test]
    fn no_duplicate_slots_across_passes() {
        let pool = descending_pool(30);
        for cursor in 0..40u64 {
            let sel = select_focus_neurons(&pool, 10, false, cursor);
            let distinct: HashSet<&String> = sel.selected.iter().collect();
            assert_eq!(
                distinct.len(),
                sel.selected.len(),
                "no neuron may take two slots in one pass (cursor {cursor})"
            );
        }
    }

    #[test]
    fn allocation_diagnostics_are_reported() {
        let pool = descending_pool(100);
        let sel = select_focus_neurons(&pool, 10, false, 3);
        assert_eq!(sel.eligible_pool_size, 100);
        assert_eq!(sel.exploration_cursor, 3);
        assert_eq!(sel.exploitation_count + sel.exploration_count, 10);
        // Cumulative coverage includes the always-exploited head plus the tail
        // swept so far, and never exceeds the pool.
        assert!(sel.cumulative_coverage >= sel.exploitation_count);
        assert!(sel.cumulative_coverage <= 100);
    }

    #[test]
    fn even_distribution_keeps_ranked_head() {
        // Twelve equally-weighted neurons: exploitation still takes the ranked
        // prefix; a single exploration slot is reserved.
        let pool: Vec<FocusCandidate> = (0..12).map(|i| candidate(&format!("n{i}"), 1.0)).collect();
        let sel = select_focus_neurons(&pool, 6, false, 0);
        assert_eq!(sel.selected.len(), 6);
        assert_eq!(sel.selected[0], "n0");
        assert!(sel.exploitation_count >= 5, "≥80% exploitation for N=6");
        assert!(sel.weight_concentration_ratio < CONCENTRATION_WARN_THRESHOLD);
    }

    #[test]
    fn fewer_candidates_than_target_returns_all() {
        let pool = descending_pool(3); // 3 candidates, target 6
        let sel = select_focus_neurons(&pool, 6, false, 0);
        assert_eq!(sel.selected.len(), 3);
        assert_eq!(sel.exploitation_count, 3);
        assert_eq!(sel.exploration_count, 0);
        let distinct: HashSet<&String> = sel.selected.iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn empty_pool_is_safe() {
        let sel = select_focus_neurons(&[], 6, false, 0);
        assert!(sel.selected.is_empty());
        assert_eq!(sel.raw_weight_concentration_ratio, 0.0);
        assert_eq!(sel.weight_concentration_ratio, 0.0);
        assert_eq!(sel.exploitation_count, 0);
        assert_eq!(sel.exploration_count, 0);
        assert_eq!(sel.eligible_pool_size, 0);
    }

    #[test]
    fn target_one_is_clamped_and_safe() {
        let pool = descending_pool(6);
        let sel = select_focus_neurons(&pool, 0, false, 0);
        // target_n is clamped to 1: one exploitation slot, no exploration.
        assert_eq!(sel.selected.len(), 1);
        assert_eq!(sel.selected[0], "c0");
        assert_eq!(sel.exploration_count, 0);
    }

    #[test]
    fn drought_smaller_pool_than_target() {
        // Only 2 candidates but target 6: everything is analysed, no split.
        let pool = descending_pool(2);
        let sel = select_focus_neurons(&pool, 6, true, 5);
        assert_eq!(sel.selected.len(), 2);
        assert_eq!(sel.exploration_count, 0);
        let distinct: HashSet<&String> = sel.selected.iter().collect();
        assert_eq!(distinct.len(), 2);
    }

    #[test]
    fn deterministic_for_same_inputs() {
        let pool = descending_pool(50);
        let a = select_focus_neurons(&pool, 10, false, 17);
        let b = select_focus_neurons(&pool, 10, false, 17);
        assert_eq!(a, b, "identical inputs must produce identical output");
    }
}
