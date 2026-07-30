//! Fail-loud candidate reconciliation (Issue #1802).
//!
//! Each sub-issue of #1782 wired one silent drop path into
//! [`RejectionBreakdown`], but nothing stopped the *next* one being added
//! silently: every one of the six paths the #1782 diagnosis found was a plain
//! `continue` that compiled, passed tests and shipped. This module supplies the
//! enforced invariant that replaces that convention.
//!
//! # The invariant
//!
//! Each analysis surface owns a [`CandidateLedger`] with two counters:
//!
//! * `considered` — incremented once where a batch of candidate proposals
//!   enters disposition (a *single* call site per batch, never per drop site).
//! * `accounted` — incremented once per candidate whose fate was recorded,
//!   whether that is an accept or a counted rejection.
//!
//! At the point the surface's metadata breakdown is finalised,
//! [`reconcile`] asserts
//!
//! ```text
//! considered == accounted
//! ```
//!
//! and, because `accounted` is only ever incremented alongside a
//! breakdown-bound counter or an accept, that is the surface-scoped form of the
//! #1802 identity `candidates_considered == candidates_returned +
//! sum(rejection_breakdown)`.
//!
//! # Fail-loud behaviour on mismatch
//!
//! A shortfall (`considered > accounted`) means candidates vanished with no
//! recorded verdict. Three things happen, in line with the repository's
//! never-fail-silently rule:
//!
//! 1. The residual is recorded on the breakdown under
//!    [`REJECTION_UNACCOUNTED_DROP`], so the drop is never invisible to
//!    downstream tooling — including on a release build where assertions are
//!    compiled out.
//! 2. A single `tracing::warn!` naming the surface and the unaccounted delta is
//!    emitted, greppable in production logs.
//! 3. Under [strict mode](strict_mode) a `debug_assert!` fires, so CI fails the
//!    PR that introduces the unaccounted path rather than only logging.
//!
//! An over-count (`accounted > considered`) is the opposite defect — a
//! disposition recorded twice, or recorded without a matching batch-formation
//! increment. It cannot hide a lost candidate, so nothing is added to the
//! breakdown, but it is warned about and fails strict mode all the same.
//!
//! A balanced pass does none of this: no log line, no breakdown entry, and two
//! relaxed atomic loads of overhead.
//!
//! # Surface scope
//!
//! The ledger covers the additive candidate populations where the #1782 drop
//! paths lived, and each surface documents its own unit:
//!
//! | Surface | Unit of `considered` |
//! | --- | --- |
//! | [`SURFACE_NEURON`] | add-neuron candidates formed by the GPU evaluators |
//! | [`SURFACE_SYNAPSE`] | helpful-synapse work items entering result collection |
//!
//! Harmful (synapse-removal) candidates are a separate population with their
//! own diagnostics and are deliberately outside the ledger; so is downstream
//! post-processing, whose truncation and de-duplication drops are already
//! counted under `budget_truncated` / `same_target_squash_duplicate`.

use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use crate::analysis::diagnostics::RejectionBreakdown;
use crate::analysis::diagnostics::rejection_reasons::REJECTION_UNACCOUNTED_DROP;

/// Surface name for add-neuron analysis, used in reconciliation logs and
/// assertion messages.
pub const SURFACE_NEURON: &str = "neuron";

/// Surface name for synapse analysis, used in reconciliation logs and assertion
/// messages.
pub const SURFACE_SYNAPSE: &str = "synapse";

// =============================================================================
// Strict mode
// =============================================================================

/// Strict mode follows [`crate::config::strict_candidate_reconciliation`].
const STRICT_FROM_ENV: u8 = 0;
/// Strict mode is forced off by a test override.
const STRICT_FORCED_OFF: u8 = 1;
/// Strict mode is forced on by a test override.
const STRICT_FORCED_ON: u8 = 2;

static STRICT_OVERRIDE: AtomicU8 = AtomicU8::new(STRICT_FROM_ENV);

/// Whether a reconciliation mismatch should trip a `debug_assert!`.
///
/// Defaults to [`crate::config::strict_candidate_reconciliation`] — on for debug
/// builds, so a new unaccounted drop path fails the PR that introduces it under
/// `cargo test`, and off for release builds. Tests can force the setting for the
/// duration of a case via [`StrictModeGuard`].
///
/// Assertions are compiled out of release builds, so on a production build this
/// only selects whether the `debug_assert!` *would* fire — the `tracing::warn!`
/// and the [`REJECTION_UNACCOUNTED_DROP`] breakdown entry are emitted either
/// way.
#[must_use]
pub fn strict_mode() -> bool {
    match STRICT_OVERRIDE.load(Ordering::Relaxed) {
        STRICT_FORCED_ON => true,
        STRICT_FORCED_OFF => false,
        _ => crate::config::strict_candidate_reconciliation(),
    }
}

/// Force [`strict_mode`] on or off, or pass `None` to fall back to the
/// environment variable.
///
/// Intended for tests that need to drive both the strict and warn-only paths
/// deterministically without mutating process environment state. Prefer
/// [`StrictModeGuard`], which restores the previous setting even when the case
/// panics.
pub fn set_strict_mode_override(forced: Option<bool>) {
    let value = match forced {
        Some(true) => STRICT_FORCED_ON,
        Some(false) => STRICT_FORCED_OFF,
        None => STRICT_FROM_ENV,
    };
    STRICT_OVERRIDE.store(value, Ordering::Relaxed);
}

/// Scoped [`strict_mode`] override that restores the previous setting on drop,
/// including when the scope unwinds from a failed assertion.
///
/// The override is process-global, so tests that use it must run serially.
#[derive(Debug)]
#[must_use = "the override is restored when the guard is dropped"]
pub struct StrictModeGuard {
    previous: u8,
}

impl StrictModeGuard {
    /// Force strict mode to `strict` until the guard is dropped.
    pub fn new(strict: bool) -> Self {
        let previous = STRICT_OVERRIDE.load(Ordering::Relaxed);
        set_strict_mode_override(Some(strict));
        Self { previous }
    }
}

impl Drop for StrictModeGuard {
    fn drop(&mut self) {
        STRICT_OVERRIDE.store(self.previous, Ordering::Relaxed);
    }
}

// =============================================================================
// Ledger
// =============================================================================

/// Per-surface, per-pass candidate accounting.
///
/// Shared across rayon workers via `Arc`. Both counters are plain relaxed
/// atomics: the increments sit in per-candidate loops, so they must not
/// allocate, lock, or build a `String`. Each orchestration call creates a fresh
/// ledger; it does not persist across passes.
#[derive(Debug, Default)]
pub struct CandidateLedger {
    considered: AtomicU32,
    accounted: AtomicU32,
}

impl CandidateLedger {
    /// Create a zeroed ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `count` candidates entering disposition.
    ///
    /// Call this exactly once per formed batch, at the point the batch comes
    /// into existence — never per drop site, or a new drop path could satisfy
    /// the invariant by incrementing both sides.
    #[inline]
    pub fn record_considered(&self, count: usize) {
        add_saturating(&self.considered, count);
    }

    /// Record `count` candidates whose verdict was accounted for — accepted, or
    /// rejected with a counter that reaches the rejection breakdown.
    #[inline]
    pub fn record_accounted(&self, count: usize) {
        add_saturating(&self.accounted, count);
    }

    /// Candidates that entered disposition on this surface.
    #[must_use]
    pub fn considered(&self) -> u32 {
        self.considered.load(Ordering::Relaxed)
    }

    /// Candidates whose verdict was accounted for on this surface.
    #[must_use]
    pub fn accounted(&self) -> u32 {
        self.accounted.load(Ordering::Relaxed)
    }
}

/// Add `count` to `counter`, clamping an implausibly large `usize` to
/// `u32::MAX` rather than wrapping.
#[inline]
fn add_saturating(counter: &AtomicU32, count: usize) {
    if count == 0 {
        return;
    }
    counter.fetch_add(u32::try_from(count).unwrap_or(u32::MAX), Ordering::Relaxed);
}

// =============================================================================
// Reconciliation
// =============================================================================

/// Outcome of reconciling one surface's [`CandidateLedger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reconciliation {
    /// Surface the ledger belongs to ([`SURFACE_NEURON`] / [`SURFACE_SYNAPSE`]).
    pub surface: &'static str,
    /// Candidates that entered disposition.
    pub considered: u32,
    /// Candidates whose verdict was accounted for.
    pub accounted: u32,
    /// Candidates that vanished with no recorded verdict (`considered -
    /// accounted`). Non-zero means a drop path is unaccounted for.
    pub unaccounted: u32,
    /// Verdicts recorded without a matching considered candidate (`accounted -
    /// considered`). Non-zero means a disposition is double counted.
    pub over_accounted: u32,
}

impl Reconciliation {
    /// Reconcile a raw pair of counts.
    #[must_use]
    pub fn new(surface: &'static str, considered: u32, accounted: u32) -> Self {
        Self {
            surface,
            considered,
            accounted,
            unaccounted: considered.saturating_sub(accounted),
            over_accounted: accounted.saturating_sub(considered),
        }
    }

    /// Whether every considered candidate was accounted for exactly once.
    #[must_use]
    pub fn balanced(&self) -> bool {
        self.unaccounted == 0 && self.over_accounted == 0
    }

    /// Signed `accounted - considered` delta, for logs that want one number.
    #[must_use]
    pub fn signed_delta(&self) -> i64 {
        i64::from(self.accounted) - i64::from(self.considered)
    }

    /// Diagnostic message naming the surface and the unaccounted delta.
    ///
    /// Used verbatim as the strict-mode assertion message, so it must be
    /// self-contained enough to diagnose from a CI log alone.
    #[must_use]
    pub fn message(&self) -> String {
        format!(
            "Issue #1802: candidate reconciliation failed on the {surface} surface — \
             {considered} candidate(s) entered disposition but {accounted} verdict(s) were \
             recorded ({unaccounted} unaccounted, {over_accounted} over-accounted, \
             delta {delta:+}). Every drop path must record its verdict on the surface's \
             CandidateLedger and in the rejection breakdown; a bare `continue` makes the \
             candidate vanish.",
            surface = self.surface,
            considered = self.considered,
            accounted = self.accounted,
            unaccounted = self.unaccounted,
            over_accounted = self.over_accounted,
            delta = self.signed_delta(),
        )
    }
}

/// Reconcile `ledger` against `breakdown` for `surface` (Issue #1802).
///
/// Call this once per surface, where the metadata rejection breakdown is
/// finalised. On a balanced pass this is two relaxed atomic loads and nothing
/// else — no log line, no breakdown entry.
///
/// On a mismatch it records the unaccounted residual under
/// [`REJECTION_UNACCOUNTED_DROP`], emits one `tracing::warn!` naming the surface
/// and the delta, and — under [`strict_mode`] — trips a `debug_assert!` so CI
/// fails rather than merely logging.
pub fn reconcile(
    surface: &'static str,
    ledger: &CandidateLedger,
    breakdown: &mut RejectionBreakdown,
) -> Reconciliation {
    let reconciliation = Reconciliation::new(surface, ledger.considered(), ledger.accounted());

    if !reconciliation.balanced() {
        // Record first: on a release build the assertion is compiled out, so
        // the breakdown entry is the only durable evidence the pass lost
        // candidates.
        breakdown.record_many_u32(REJECTION_UNACCOUNTED_DROP, reconciliation.unaccounted);
        tracing::warn!(
            surface = reconciliation.surface,
            considered = reconciliation.considered,
            accounted = reconciliation.accounted,
            unaccounted = reconciliation.unaccounted,
            over_accounted = reconciliation.over_accounted,
            "{}",
            reconciliation.message()
        );
    }

    if strict_mode() {
        debug_assert!(reconciliation.balanced(), "{}", reconciliation.message());
    }

    reconciliation
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn balanced_ledger_records_nothing_and_logs_nothing() {
        let ledger = CandidateLedger::new();
        ledger.record_considered(7);
        ledger.record_accounted(4);
        ledger.record_accounted(3);

        let mut breakdown = RejectionBreakdown::new();
        let result = reconcile(SURFACE_NEURON, &ledger, &mut breakdown);

        assert!(result.balanced());
        assert_eq!(result.considered, 7);
        assert_eq!(result.accounted, 7);
        assert_eq!(result.signed_delta(), 0);
        assert!(
            breakdown.is_empty(),
            "a clean pass must add no rejection-breakdown entry"
        );
    }

    /// The guard's whole purpose: a drop path added without accounting for it
    /// surfaces as an `unaccounted_drop` count naming the surface and delta.
    #[test]
    #[serial]
    fn uncounted_drop_surfaces_in_the_breakdown() {
        let _guard = StrictModeGuard::new(false);
        let ledger = CandidateLedger::new();
        ledger.record_considered(10);
        // Six accounted; four dropped by a hypothetical bare `continue`.
        ledger.record_accounted(6);

        let mut breakdown = RejectionBreakdown::new();
        let result = reconcile(SURFACE_SYNAPSE, &ledger, &mut breakdown);

        assert!(!result.balanced());
        assert_eq!(result.unaccounted, 4);
        assert_eq!(result.over_accounted, 0);
        assert_eq!(result.signed_delta(), -4);
        assert_eq!(
            breakdown.counts().get(REJECTION_UNACCOUNTED_DROP),
            Some(&4),
            "the unaccounted residual must be visible in the breakdown"
        );
        let message = result.message();
        assert!(
            message.contains(SURFACE_SYNAPSE),
            "the message must name the surface: {message}"
        );
        assert!(
            message.contains('4'),
            "the message must name the delta: {message}"
        );
    }

    /// An over-count cannot hide a lost candidate, so it must not inflate the
    /// breakdown — but it is still a defect and still reported.
    #[test]
    #[serial]
    fn over_accounting_is_reported_but_not_recorded() {
        let _guard = StrictModeGuard::new(false);
        let ledger = CandidateLedger::new();
        ledger.record_considered(2);
        ledger.record_accounted(5);

        let mut breakdown = RejectionBreakdown::new();
        let result = reconcile(SURFACE_NEURON, &ledger, &mut breakdown);

        assert!(!result.balanced());
        assert_eq!(result.unaccounted, 0);
        assert_eq!(result.over_accounted, 3);
        assert_eq!(result.signed_delta(), 3);
        assert!(
            breakdown.is_empty(),
            "over-accounting must not be recorded as a lost candidate"
        );
    }

    #[test]
    fn zero_counts_are_ignored() {
        let ledger = CandidateLedger::new();
        ledger.record_considered(0);
        ledger.record_accounted(0);
        assert_eq!(ledger.considered(), 0);
        assert_eq!(ledger.accounted(), 0);
    }

    /// Strict mode must be togglable without touching process environment
    /// state, so tests can drive both paths deterministically.
    #[test]
    #[serial]
    fn strict_mode_override_round_trips() {
        {
            let _on = StrictModeGuard::new(true);
            assert!(strict_mode());
        }
        {
            let _off = StrictModeGuard::new(false);
            assert!(!strict_mode());
        }
        assert_eq!(
            strict_mode(),
            crate::config::strict_candidate_reconciliation(),
            "with no override the setting must follow the environment variable"
        );
    }

    /// Strict mode is what turns the warn into a CI failure: the assertion
    /// message names the surface and the delta.
    #[test]
    #[serial]
    #[should_panic(expected = "candidate reconciliation failed on the neuron surface")]
    #[cfg(debug_assertions)]
    fn strict_mode_panics_on_unaccounted_drop() {
        let _guard = StrictModeGuard::new(true);
        let ledger = CandidateLedger::new();
        ledger.record_considered(3);
        let mut breakdown = RejectionBreakdown::new();
        let _ = reconcile(SURFACE_NEURON, &ledger, &mut breakdown);
    }
}
