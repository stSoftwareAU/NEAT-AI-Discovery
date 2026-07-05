//! Issue #3172 — scale the focus-ranking wall-clock budget with mode + dataset
//! size so a legitimate lazy fallback finishes instead of aborting into
//! degraded recorded-error aggregation.
//!
//! Verifies that:
//! - Eager mode keeps the current fixed default budget (no regression).
//! - Lazy mode scales the budget above the default, growing with the projected
//!   dataset size, and is clamped to the supported maximum.
//! - An explicit `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` override wins
//!   verbatim for both modes and is never scaled.
//! - `0` still disables the budget; the unscaled `focus_ranking_budget_ms`
//!   accessor keeps its prior behaviour (regression guard).

use neat_ai_discovery::config::{
    DEFAULT_FOCUS_RANKING_BUDGET_MS, FOCUS_RANKING_BUDGET_MAX_MS, FOCUS_RANKING_BUDGET_MIN_MS,
    FOCUS_RANKING_LAZY_BUDGET_MULTIPLIER, effective_focus_ranking_budget_ms,
    focus_ranking_budget_ms, scale_default_focus_ranking_budget_ms,
};
use serial_test::serial;

const BUDGET_KEY: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS";

/// RAII guard that restores an env var on drop so env-mutating tests do not leak
/// into one another. Tests are `#[serial]` so no concurrent set/unset races.
struct EnvGuard {
    key: &'static str,
    prior: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::set_var(key, value) };
        Self { key, prior }
    }

    fn unset(key: &'static str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::remove_var(key) };
        Self { key, prior }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

// =============================================================================
// Pure scaling helper (no env) — the core of the fix.
// =============================================================================

#[test]
fn eager_scaling_keeps_the_default_budget() {
    // No regression: eager ignores dataset size and keeps the fixed default.
    assert_eq!(
        scale_default_focus_ranking_budget_ms(false, 0),
        DEFAULT_FOCUS_RANKING_BUDGET_MS
    );
    assert_eq!(
        scale_default_focus_ranking_budget_ms(false, 50_000),
        DEFAULT_FOCUS_RANKING_BUDGET_MS,
        "eager mode must not scale with dataset size",
    );
}

#[test]
fn lazy_scaling_exceeds_the_default_budget() {
    // The #3170 incident: ~14 297 MB projected aborted at the fixed 120 s.
    let scaled = scale_default_focus_ranking_budget_ms(true, 14_297);
    assert!(
        scaled > DEFAULT_FOCUS_RANKING_BUDGET_MS,
        "a large lazy run must earn more than the fixed default budget \
         (got {scaled} vs default {DEFAULT_FOCUS_RANKING_BUDGET_MS})",
    );
    // At minimum the lazy multiplier is applied before the per-MB term.
    assert!(
        scaled >= DEFAULT_FOCUS_RANKING_BUDGET_MS * FOCUS_RANKING_LAZY_BUDGET_MULTIPLIER,
        "lazy budget must be at least the multiplied default",
    );
}

#[test]
fn lazy_scaling_grows_with_projected_size() {
    let small = scale_default_focus_ranking_budget_ms(true, 100);
    let mid = scale_default_focus_ranking_budget_ms(true, 5_000);
    let large = scale_default_focus_ranking_budget_ms(true, 20_000);
    assert!(
        small <= mid && mid <= large,
        "budget must be monotone in size"
    );
    assert!(
        large > small,
        "a materially larger dataset must earn a larger budget",
    );
}

#[test]
fn lazy_scaling_is_clamped_to_the_supported_range() {
    // An enormous projection cannot exceed the one-hour ceiling.
    let huge = scale_default_focus_ranking_budget_ms(true, u64::MAX);
    assert_eq!(
        huge, FOCUS_RANKING_BUDGET_MAX_MS,
        "lazy budget must clamp to the supported maximum, not overflow",
    );
    // Even the smallest lazy run stays within [MIN, MAX].
    let tiny = scale_default_focus_ranking_budget_ms(true, 0);
    assert!((FOCUS_RANKING_BUDGET_MIN_MS..=FOCUS_RANKING_BUDGET_MAX_MS).contains(&tiny));
}

// =============================================================================
// Effective budget: override precedence + scaling (env).
// =============================================================================

#[test]
#[serial]
fn effective_default_scales_by_mode_when_unset() {
    let _g = EnvGuard::unset(BUDGET_KEY);
    let eager = effective_focus_ranking_budget_ms(false, 14_297).expect("eager enabled");
    let lazy = effective_focus_ranking_budget_ms(true, 14_297).expect("lazy enabled");
    assert_eq!(
        eager, DEFAULT_FOCUS_RANKING_BUDGET_MS,
        "unset + eager must return the unscaled default",
    );
    assert!(
        lazy > eager,
        "unset + lazy must scale above the eager default (lazy {lazy} vs eager {eager})",
    );
}

#[test]
#[serial]
fn explicit_override_wins_verbatim_and_is_never_scaled() {
    let _g = EnvGuard::set(BUDGET_KEY, "90000");
    // The operator's value is authoritative for BOTH modes — no scaling.
    assert_eq!(
        effective_focus_ranking_budget_ms(false, 0),
        Some(90_000),
        "explicit override must win for eager",
    );
    assert_eq!(
        effective_focus_ranking_budget_ms(true, 50_000),
        Some(90_000),
        "explicit override must win for lazy and must not be scaled by dataset size",
    );
}

#[test]
#[serial]
fn explicit_override_is_clamped_but_still_not_scaled() {
    // Above the max → clamped down to the ceiling, still not scaled by mode.
    let _g = EnvGuard::set(BUDGET_KEY, "999999999");
    assert_eq!(
        effective_focus_ranking_budget_ms(true, 20_000),
        Some(FOCUS_RANKING_BUDGET_MAX_MS),
    );
}

#[test]
#[serial]
fn explicit_zero_disables_budget_for_both_modes() {
    let _g = EnvGuard::set(BUDGET_KEY, "0");
    assert_eq!(effective_focus_ranking_budget_ms(false, 0), None);
    assert_eq!(
        effective_focus_ranking_budget_ms(true, 14_297),
        None,
        "0 is the deliberate opt-out even for a large lazy run",
    );
}

#[test]
#[serial]
fn invalid_override_falls_back_to_scaled_default() {
    let _g = EnvGuard::set(BUDGET_KEY, "not-a-number");
    // Garbage must not disable or pin the budget — the scaled default applies.
    let lazy = effective_focus_ranking_budget_ms(true, 14_297).expect("lazy enabled");
    assert!(
        lazy > DEFAULT_FOCUS_RANKING_BUDGET_MS,
        "invalid override must fall back to the scaled default for lazy",
    );
    assert_eq!(
        effective_focus_ranking_budget_ms(false, 0),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS),
        "invalid override must fall back to the unscaled default for eager",
    );
}

// =============================================================================
// Regression guard: the unscaled accessor keeps its prior behaviour.
// =============================================================================

#[test]
#[serial]
fn unscaled_accessor_unchanged_when_unset() {
    let _g = EnvGuard::unset(BUDGET_KEY);
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
}

#[test]
#[serial]
fn unscaled_accessor_honours_override_and_zero() {
    {
        let _g = EnvGuard::set(BUDGET_KEY, "45000");
        assert_eq!(focus_ranking_budget_ms(), Some(45_000));
    }
    {
        let _g = EnvGuard::set(BUDGET_KEY, "0");
        assert_eq!(focus_ranking_budget_ms(), None);
    }
}
