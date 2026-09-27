//! `ModuleOutcomeTracker` validation at the FFI boundary (Issue #2170).
//!
//! `analyze_parallel` accepts the caller's persisted tracker verbatim. serde
//! bypasses the invariants `record()` and `record_soft_failures()` keep, so a
//! corrupt or hostile payload can carry `successes > attempts` or a negative,
//! non-finite or huge `soft_failures`. `success_rate()` is total (Issue #2222),
//! but a gate computed from corrupt history is still wrong, so the boundary
//! rejects the tracker rather than silently repairing it.

use super::DiscoveryError;
use crate::analysis::module_weights::{ModuleOutcomeTracker, ModuleStats};
use crate::analysis::utils::char_prefix;

/// Most characters of a caller-supplied module name echoed in an error detail.
pub const MODULE_NAME_DETAIL_MAX_CHARS: usize = 64;

/// Upper bound on `soft_failures` (Issue #2223): `record_soft_failures` adds at
/// most 1.0 per filtered candidate, so this matches the `u32` range of the
/// sibling `attempts` and `candidatesProduced` counters.
const MAX_SOFT_FAILURES: f64 = u32::MAX as f64;

fn invalid(name: &str, problem: &str) -> DiscoveryError {
    DiscoveryError::InvalidInput {
        detail: format!(
            "moduleOutcomeTracker module \"{}\": {problem} (Issue #2170)",
            char_prefix(name, MODULE_NAME_DETAIL_MAX_CHARS)
        ),
    }
}

/// Reject a `ModuleStats` entry that breaks the invariants `record()` and
/// `record_soft_failures()` keep.
///
/// Returns `DiscoveryError::InvalidInput` (`data_validation`) naming the field
/// and at most [`MODULE_NAME_DETAIL_MAX_CHARS`] characters of `name`.
pub fn validate_module_stats(name: &str, stats: &ModuleStats) -> Result<(), DiscoveryError> {
    if stats.successes > stats.attempts {
        return Err(invalid(
            name,
            &format!(
                "successes ({}) exceeds attempts ({})",
                stats.successes, stats.attempts
            ),
        ));
    }
    let soft = stats.soft_failures;
    if !soft.is_finite() || soft < 0.0 || soft > MAX_SOFT_FAILURES {
        return Err(invalid(
            name,
            &format!("softFailures ({soft}) must be finite and within [0, {MAX_SOFT_FAILURES}]"),
        ));
    }
    Ok(())
}

/// Run [`validate_module_stats`] on every module in `tracker`.
pub fn validate_module_outcome_tracker(
    tracker: &ModuleOutcomeTracker,
) -> Result<(), DiscoveryError> {
    tracker
        .all_stats()
        .iter()
        .try_for_each(|(name, stats)| validate_module_stats(name, stats))
}
