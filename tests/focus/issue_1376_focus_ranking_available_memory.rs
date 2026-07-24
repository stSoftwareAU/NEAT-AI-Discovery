//! Issue #1376: Eager/lazy focus-ranking decision based on real available RAM.
//!
//! The auto-detect (no explicit budget) path previously gated on a
//! 50%-of-total-RAM cap, which dropped mid-sized parquet files onto the slow
//! lazy path while GBs of RAM were free (the mid-sized-projection regression: ~1.6 GB
//! projection chosen lazy despite ~2990 MB available).
//!
//! These tests exercise the pure decision helper
//! `decide_loading_mode_for_available_memory` directly so the available-memory
//! branch is covered deterministically, including those regression numbers.

use neat_ai_discovery::config::{
    DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB, focus_ranking_memory_margin_mb,
};
use neat_ai_discovery::focus::{
    FocusLazyReason, FocusLoadingMode, decide_loading_mode_for_available_memory,
};
use serial_test::serial;

const MB: u64 = 1024 * 1024;
const MARGIN_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB";

/// RAII guard that sets/restores `MARGIN_ENV` for a single test.
struct MarginEnvGuard {
    previous: Option<String>,
}

impl MarginEnvGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var(MARGIN_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(MARGIN_ENV, value) };
        Self { previous }
    }

    fn unset() -> Self {
        let previous = std::env::var(MARGIN_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(MARGIN_ENV) };
        Self { previous }
    }
}

impl Drop for MarginEnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(MARGIN_ENV, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(MARGIN_ENV) },
        }
    }
}

#[test]
fn mid_sized_projection_numbers_choose_preload() {
    // Parquet 531.10 MB → ×3 ≈ 1593 MB projection, ~2990 MB available, default
    // 1 GB margin. usable = 2990 − 1024 = 1966 MB ≥ 1593 MB → Preload.
    let (mode, reason) = decide_loading_mode_for_available_memory(1593 * MB, 2990 * MB, 1024 * MB);
    assert_eq!(mode, FocusLoadingMode::Preload);
    assert_eq!(reason, FocusLazyReason::None);
}

#[test]
fn projection_over_usable_chooses_lazy_memory_pressure() {
    // 3.5 GB projection, 4 GB available, 1 GB margin → usable 3 GB → Lazy.
    let (mode, reason) = decide_loading_mode_for_available_memory(3584 * MB, 4096 * MB, 1024 * MB);
    assert_eq!(mode, FocusLoadingMode::Lazy);
    assert_eq!(reason, FocusLazyReason::MemoryPressure);
}

#[test]
fn boundary_equal_to_usable_preloads() {
    // projected == available − margin → fits (<=, not <).
    let (mode, reason) = decide_loading_mode_for_available_memory(3072 * MB, 4096 * MB, 1024 * MB);
    assert_eq!(mode, FocusLoadingMode::Preload);
    assert_eq!(reason, FocusLazyReason::None);

    // One byte over the usable budget tips to lazy.
    let (mode_over, reason_over) =
        decide_loading_mode_for_available_memory(3072 * MB + 1, 4096 * MB, 1024 * MB);
    assert_eq!(mode_over, FocusLoadingMode::Lazy);
    assert_eq!(reason_over, FocusLazyReason::MemoryPressure);
}

#[test]
fn margin_exceeding_available_falls_back_to_lazy_without_panic() {
    // Saturating subtraction: margin > available → 0 usable bytes → lazy for
    // any real projection (no underflow).
    let (mode, reason) = decide_loading_mode_for_available_memory(MB, 512 * MB, 1024 * MB);
    assert_eq!(mode, FocusLoadingMode::Lazy);
    assert_eq!(reason, FocusLazyReason::MemoryPressure);
}

#[test]
#[serial]
fn margin_config_unset_uses_default() {
    let _guard = MarginEnvGuard::unset();
    assert_eq!(
        focus_ranking_memory_margin_mb(),
        DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB
    );
    assert_eq!(DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB, 1024);
}

#[test]
#[serial]
fn margin_config_honours_override() {
    let _guard = MarginEnvGuard::set("2048");
    assert_eq!(focus_ranking_memory_margin_mb(), 2048);
}

#[test]
#[serial]
fn margin_config_honours_zero() {
    // Zero is a valid choice (reserve no margin), distinct from the budget env
    // var where 0 means "unset".
    let _guard = MarginEnvGuard::set("0");
    assert_eq!(focus_ranking_memory_margin_mb(), 0);
}

#[test]
#[serial]
fn margin_config_invalid_falls_back_to_default() {
    let _guard = MarginEnvGuard::set("not-a-number");
    assert_eq!(
        focus_ranking_memory_margin_mb(),
        DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB
    );
}
