//! Escape hatch for the incremental-analysis fingerprint skip (Issue #1781).
//!
//! `previous_neuron_fingerprints` (Issue #490) skips focus neurons whose
//! **structural** fingerprint is unchanged since the last pass. That is a sound
//! saving while discovery is healthy, but it is self-reinforcing during a
//! drought: the topology by definition does not change while nothing is being
//! accepted, so every focus neuron looks unchanged and the pass is dropped
//! whole — no candidates, no rejection breakdown, no diagnostic — *regardless
//! of freshly recorded activation data*.
//!
//! The escape hatch releases the cache once the creature's trailing streak of
//! empty passes reaches [`DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS`]. The
//! structural cache has demonstrably stopped paying for itself by then, so the
//! full focus set is re-analysed against the new recordings.

use super::discovery_mode::DiscoveryOutcomeLog;

/// Consecutive empty passes after which the fingerprint cache is bypassed.
///
/// Small on purpose: a couple of empty passes is normal noise, but by the third
/// the structural cache is provably not helping this creature.
pub const DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS: u32 = 3;

/// Whether the fingerprint cache should be bypassed for this pass.
///
/// Returns `true` once `log`'s trailing consecutive-failure streak reaches
/// `drought_epochs`. A `drought_epochs` of `0` disables the hatch entirely, and
/// an absent log (host supplied no outcome history) leaves the cache active so
/// existing behaviour is preserved.
#[must_use]
pub fn should_bypass_fingerprint_cache(
    log: Option<&DiscoveryOutcomeLog>,
    drought_epochs: u32,
) -> bool {
    if drought_epochs == 0 {
        return false;
    }
    log.is_some_and(|l| l.consecutive_trailing_failures() >= drought_epochs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_log_keeps_cache_active() {
        assert!(!should_bypass_fingerprint_cache(
            None,
            DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
        ));
    }

    #[test]
    fn streak_at_threshold_bypasses() {
        let log = DiscoveryOutcomeLog::from_outcomes(vec![true, false, false, false]);
        assert!(should_bypass_fingerprint_cache(
            Some(&log),
            DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
        ));
    }

    #[test]
    fn streak_below_threshold_keeps_cache_active() {
        let log = DiscoveryOutcomeLog::from_outcomes(vec![true, false, false]);
        assert!(!should_bypass_fingerprint_cache(
            Some(&log),
            DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
        ));
    }
}
