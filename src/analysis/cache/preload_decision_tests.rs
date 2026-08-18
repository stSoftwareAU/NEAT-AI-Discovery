//! Unit tests for the analysis-cache eager-vs-lazy pre-load decision
//! (Issue #3176).
//!
//! These exercise the pure decision helpers directly against fixed memory
//! figures, so they run cross-platform with no parquet file and no live system
//! memory. They lock two guarantees:
//!
//! - the pre-load honours a supplied memory budget, and
//! - on the auto-detect path it shares the focus-ranking accounting primitive
//!   ([`parquet_preload_fits_available`]) — there is no divergent second
//!   heuristic (the old 50%-of-total-RAM cap is gone).

use super::*;

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * 1024 * 1024;

// =============================================================================
// Budget path (supplied --rustMemoryBudgetMB)
// =============================================================================

#[test]
fn budget_path_selects_preload_when_projection_fits_budget() {
    // 10 MB projection well under a 100 MB budget → eager.
    let (mode, reason) = decide_cache_preload_for_budget(10 * MB, 100);
    assert_eq!(mode, CachePreloadMode::Preload);
    assert_eq!(reason, CacheLazyReason::None);
}

#[test]
fn budget_path_selects_lazy_when_projection_exceeds_budget() {
    // 200 MB projection over a 100 MB budget → lazy with budget reason.
    let (mode, reason) = decide_cache_preload_for_budget(200 * MB, 100);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::Budget);
}

#[test]
fn budget_path_exact_boundary_is_eager() {
    // Projection exactly equal to the budget still pre-loads (<= is a fit).
    let (mode, reason) = decide_cache_preload_for_budget(100 * MB, 100);
    assert_eq!(mode, CachePreloadMode::Preload);
    assert_eq!(reason, CacheLazyReason::None);
}

// =============================================================================
// Auto-detect path (no budget) — corrected available-memory accounting
// =============================================================================

#[test]
fn available_memory_path_24gb_exemplar_selects_eager() {
    // Acceptance: the 24 GB exemplar host with a ~14.3 GB projection. With the
    // corrected reclaimable accounting (Issue #3173) the host reports ~20 GB
    // available; 14.3 GB fits within 20 GB − 1 GB margin → eager, no WARN.
    let projected = 14_297 * MB;
    let available = 20 * GB;
    let margin = GB;
    let (mode, reason) = decide_cache_preload_for_available_memory(projected, available, margin);
    assert_eq!(mode, CachePreloadMode::Preload);
    assert_eq!(reason, CacheLazyReason::None);
}

#[test]
fn available_memory_path_constrained_host_selects_lazy() {
    // A genuinely constrained host: only 11 GB reclaimable for the same 14.3 GB
    // projection → lazy fallback with a memory-pressure reason.
    let projected = 14_297 * MB;
    let available = 11 * GB;
    let margin = GB;
    let (mode, reason) = decide_cache_preload_for_available_memory(projected, available, margin);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::MemoryPressure);
}

#[test]
fn available_memory_path_exact_boundary_is_eager() {
    // projected == available − margin → fits exactly → eager.
    let available = 8 * GB;
    let margin = GB;
    let projected = available - margin; // 7 GB
    let (mode, _) = decide_cache_preload_for_available_memory(projected, available, margin);
    assert_eq!(mode, CachePreloadMode::Preload);
}

#[test]
fn available_memory_path_one_byte_over_boundary_is_lazy() {
    let available = 8 * GB;
    let margin = GB;
    let projected = available - margin + 1; // 7 GB + 1 byte
    let (mode, reason) = decide_cache_preload_for_available_memory(projected, available, margin);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::MemoryPressure);
}

#[test]
fn available_memory_path_margin_larger_than_available_is_lazy() {
    // Saturating subtraction: a margin larger than available yields 0 usable
    // bytes → lazy rather than an underflow that wrongly pre-loads.
    let (mode, reason) = decide_cache_preload_for_available_memory(GB, 500 * MB, GB);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::MemoryPressure);
}

// =============================================================================
// Shared accounting: no divergent second heuristic
// =============================================================================

#[test]
fn auto_detect_decision_matches_shared_fits_primitive() {
    // The auto-detect decision must agree with the shared
    // `parquet_preload_fits_available` primitive focus ranking uses, for every
    // combination — proving both phases share one accounting heuristic.
    let margins = [0u64, 512 * MB, GB];
    let availables = [GB, 8 * GB, 20 * GB];
    let projecteds = [0u64, 7 * GB, 14_297 * MB, 30 * GB];

    for &margin in &margins {
        for &available in &availables {
            for &projected in &projecteds {
                let (mode, _) =
                    decide_cache_preload_for_available_memory(projected, available, margin);
                let fits = parquet_preload_fits_available(projected, available, margin);
                let expected = if fits {
                    CachePreloadMode::Preload
                } else {
                    CachePreloadMode::Lazy
                };
                assert_eq!(
                    mode, expected,
                    "diverged from shared accounting for projected={projected} \
                     available={available} margin={margin}"
                );
            }
        }
    }
}

// =============================================================================
// Reason string mapping
// =============================================================================

#[test]
fn lazy_reason_as_str_is_stable() {
    assert_eq!(CacheLazyReason::None.as_str(), "none");
    assert_eq!(CacheLazyReason::Budget.as_str(), "budget");
    assert_eq!(CacheLazyReason::NoBudget.as_str(), "no_budget");
    assert_eq!(CacheLazyReason::MemoryPressure.as_str(), "memory_pressure");
    assert_eq!(CacheLazyReason::Unworkable.as_str(), "unworkable");
}

// =============================================================================
// Combined decision (Issue #4138) — named acceptance cases
// =============================================================================

#[test]
fn budget_supplied_sufficient_selects_preload() {
    // 8 GB host, 3547 MB projection, 4096 MB budget that fits.
    let projected = 3547 * MB;
    let (mode, reason, logged) =
        decide_analysis_cache_preload(projected, Some(4096), 8 * GB, GB, 8192);
    assert_eq!(mode, CachePreloadMode::Preload);
    assert_eq!(reason, CacheLazyReason::None);
    assert_eq!(
        logged, 4096,
        "logged budget must be the supplied (clamped) value, not 0"
    );
}

#[test]
fn budget_supplied_insufficient_selects_lazy() {
    let projected = 3547 * MB;
    let (mode, reason, logged) =
        decide_analysis_cache_preload(projected, Some(1024), 8 * GB, GB, 8192);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::Budget);
    assert_eq!(logged, 1024);
    assert_ne!(reason, CacheLazyReason::NoBudget);
}

#[test]
fn budget_absent_falls_back_to_available_memory() {
    let projected = 14_297 * MB;
    let (mode, reason, logged) =
        decide_analysis_cache_preload(projected, None, 11 * GB, GB, 24 * 1024);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(
        reason,
        CacheLazyReason::NoBudget,
        "absent budget must log reason=no_budget, not memory_pressure"
    );
    assert_eq!(logged, 0, "absent budget logs budget_mb=0");

    // The same projection on a host with enough available RAM pre-loads.
    let (mode_ok, reason_ok, logged_ok) =
        decide_analysis_cache_preload(projected, None, 20 * GB, GB, 24 * 1024);
    assert_eq!(mode_ok, CachePreloadMode::Preload);
    assert_eq!(reason_ok, CacheLazyReason::None);
    assert_eq!(logged_ok, 0);
}

#[test]
fn budget_above_host_memory_is_clamped() {
    // 5.47 GB budget on a host reporting 3457 MB total.
    let supplied = 5223u64;
    let host_total = 3457u64;
    assert_eq!(clamp_budget_mb_to_host(supplied, host_total), host_total);

    let projected = 200 * MB;
    let (mode, reason, logged) =
        decide_analysis_cache_preload(projected, Some(supplied), 8 * GB, GB, host_total);
    assert_eq!(logged, host_total, "decision must use the clamped budget");
    assert_ne!(logged, supplied);
    assert_eq!(mode, CachePreloadMode::Preload);
    assert_eq!(reason, CacheLazyReason::None);
}
