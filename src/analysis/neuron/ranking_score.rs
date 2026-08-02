//! Reliability-weighted rank score for add-neuron candidates (Issue #1924).
//!
//! Add-neuron candidates used to be ordered by `expected_creature_score_gain`
//! alone. The candidates-cache study
//! (`docs/analysis/candidates-cache-study-1920.md`, findings B and C) measured
//! that field against the success indicator at **r = −0.608**: the engine's own
//! confidence predicts *failure*, so ranking on it selects losers. The share of
//! samples a candidate improves (`improved_count / total_count`) is the only
//! field in that corpus that tracks success positively (**r = +0.466**), and it
//! tracks realised gain *negatively* among successes (−0.871) — broadly-helpful
//! candidates land reliably but small.
//!
//! The rank score makes that trade-off explicit rather than implicit:
//!
//! ```text
//! band   = min(floor(improved_share × bands), bands - 1) / bands  // reliability
//! credit = min(gain / gain_reference, 1) / (bands + 1)            // gain, in-band
//! score  = band + credit
//! ```
//!
//! `credit` is strictly smaller than one band width, so a larger prediction can
//! never lift a candidate past a more reliable one — it only orders candidates
//! that are equally reliable. The band count
//! ([`neuron_ranking_reliability_bands`]) is the knob: one band reproduces the
//! old gain-descending order exactly, more bands hand more of the ordering to
//! reliability.

use crate::CandidateNeuronJson;
use crate::analysis::constants::{NEURON_RANKING_GAIN_REFERENCE, neuron_ranking_reliability_bands};

/// Share of evaluated samples a candidate improves, in `[0, 1]`.
///
/// Returns `0.0` when `total_count` is zero — a candidate measured on no
/// samples has demonstrated no reliability, and must not be ranked as though
/// it had.
#[must_use]
pub fn improved_share(improved_count: u32, total_count: u32) -> f64 {
    if total_count == 0 {
        return 0.0;
    }
    (f64::from(improved_count) / f64::from(total_count)).clamp(0.0, 1.0)
}

/// Rank score for an add-neuron candidate (Issue #1924), using the configured
/// band count.
///
/// See the [module docs](self) for the formula and the evidence behind it.
/// Non-finite or non-positive inputs earn no credit rather than propagating a
/// `NaN` into the ordering.
#[must_use]
pub fn neuron_rank_score(expected_creature_score_gain: f64, improved_share: f64) -> f64 {
    neuron_rank_score_with_bands(
        expected_creature_score_gain,
        improved_share,
        neuron_ranking_reliability_bands(),
    )
}

/// [`neuron_rank_score`] with an explicit band count.
///
/// Callers that score many candidates in a row (the sort comparators) read the
/// configured band count once and pass it in, so the environment is not
/// re-read per comparison.
#[must_use]
pub fn neuron_rank_score_with_bands(
    expected_creature_score_gain: f64,
    improved_share: f64,
    bands: u32,
) -> f64 {
    let bands = f64::from(bands.max(1));
    let share = if improved_share.is_finite() {
        improved_share.clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Equal-width buckets: a perfect share falls in the top bucket rather than
    // in a bucket of its own, and `bands == 1` leaves a single bucket, which is
    // the documented gain-only escape hatch.
    let band = (share * bands).floor().min(bands - 1.0) / bands;

    let reference = f64::from(NEURON_RANKING_GAIN_REFERENCE);
    let credited_gain = if expected_creature_score_gain.is_finite()
        && expected_creature_score_gain > 0.0
        && reference > 0.0
    {
        (expected_creature_score_gain / reference).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Divide by `bands + 1` so the tie-break is strictly narrower than a band:
    // a saturated gain can never reach the floor of the next band up.
    band + credited_gain / (bands + 1.0)
}

/// Rank score of a candidate, for a pre-read band count.
#[must_use]
pub(crate) fn candidate_rank_score(candidate: &CandidateNeuronJson, bands: u32) -> f64 {
    neuron_rank_score_with_bands(
        f64::from(candidate.expected_creature_score_gain),
        improved_share(candidate.improved_count, candidate.total_count),
        bands,
    )
}

/// Order add-neuron candidates best-first by [`neuron_rank_score`]
/// (Issue #1924).
///
/// Ties on the rank score fall back to `expected_creature_score_gain`
/// descending, so the ordering stays total and deterministic (`total_cmp`
/// handles `NaN`).
pub(crate) fn sort_candidates_by_rank(candidates: &mut [CandidateNeuronJson]) {
    let bands = neuron_ranking_reliability_bands();
    candidates.sort_by(|a, b| {
        candidate_rank_score(b, bands)
            .total_cmp(&candidate_rank_score(a, bands))
            .then_with(|| {
                b.expected_creature_score_gain
                    .total_cmp(&a.expected_creature_score_gain)
            })
    });
}

#[cfg(test)]
mod tests {
    use super::{improved_share, neuron_rank_score, neuron_rank_score_with_bands};
    use crate::analysis::constants::NEURON_RANKING_GAIN_REFERENCE;
    use serial_test::serial;

    const BANDS: u32 = 10;

    #[test]
    fn reliability_outranks_an_over_confident_prediction() {
        // The production shape: a huge prediction on a barely-better-than-coin-
        // flip candidate versus a tiny prediction that helps every sample.
        let over_confident = neuron_rank_score_with_bands(1.063e-2, 0.552, BANDS);
        let reliable = neuron_rank_score_with_bands(1.576e-7, 1.0, BANDS);
        assert!(
            reliable > over_confident,
            "reliable candidate must outrank the over-confident one: \
             {reliable} vs {over_confident}"
        );
    }

    #[test]
    fn a_saturated_gain_cannot_cross_a_band_boundary() {
        // Top of the 0.5 band with an unboundedly large prediction must still
        // sit below the bottom of the 0.6 band with no prediction credit.
        let saturated_low_band = neuron_rank_score_with_bands(1.0e6, 0.599, BANDS);
        let bottom_of_next_band = neuron_rank_score_with_bands(0.0, 0.600, BANDS);
        assert!(
            saturated_low_band < bottom_of_next_band,
            "gain credit must stay inside its band: {saturated_low_band} vs {bottom_of_next_band}"
        );
    }

    #[test]
    fn gain_breaks_ties_within_a_band() {
        let bigger = neuron_rank_score_with_bands(9.0e-4, 0.62, BANDS);
        let smaller = neuron_rank_score_with_bands(1.0e-5, 0.62, BANDS);
        assert!(
            bigger > smaller,
            "within one reliability band the larger gain must rank higher: \
             {bigger} vs {smaller}"
        );
    }

    #[test]
    fn predictions_past_the_reference_earn_no_extra_credit() {
        let reference = f64::from(NEURON_RANKING_GAIN_REFERENCE);
        let at_reference = neuron_rank_score_with_bands(reference, 0.62, BANDS);
        let far_past = neuron_rank_score_with_bands(reference * 1000.0, 0.62, BANDS);
        assert!(
            (at_reference - far_past).abs() < f64::EPSILON,
            "gain credit must saturate at the reference: {at_reference} vs {far_past}"
        );
    }

    #[test]
    fn single_band_reproduces_gain_ordering() {
        // The escape hatch: one band means reliability cannot separate anything,
        // so ordering is by gain alone — the pre-#1924 behaviour.
        let low_reliability_big_gain = neuron_rank_score_with_bands(1.0e-3, 0.1, 1);
        let high_reliability_small_gain = neuron_rank_score_with_bands(1.0e-9, 1.0, 1);
        assert!(
            low_reliability_big_gain > high_reliability_small_gain,
            "with one band the larger gain must win: \
             {low_reliability_big_gain} vs {high_reliability_small_gain}"
        );
    }

    #[test]
    fn unmeasured_candidates_score_zero_reliability() {
        assert!(
            (improved_share(0, 0) - 0.0).abs() < f64::EPSILON,
            "a candidate with no samples has no measured reliability"
        );
        let unmeasured = neuron_rank_score_with_bands(0.0, improved_share(0, 0), BANDS);
        assert!(
            (unmeasured - 0.0).abs() < f64::EPSILON,
            "no samples and no gain must score zero, got {unmeasured}"
        );
    }

    #[test]
    fn non_finite_inputs_do_not_poison_the_ordering() {
        for (gain, share) in [
            (f64::NAN, 0.8),
            (f64::INFINITY, 0.8),
            (-1.0, 0.8),
            (1.0e-4, f64::NAN),
        ] {
            let score = neuron_rank_score_with_bands(gain, share, BANDS);
            assert!(
                score.is_finite(),
                "score must stay finite for gain={gain} share={share}, got {score}"
            );
        }
    }

    #[test]
    fn improved_share_clamps_a_miscounted_candidate() {
        assert!((improved_share(20, 10) - 1.0).abs() < f64::EPSILON);
        assert!((improved_share(5, 10) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    #[serial]
    fn band_count_is_configurable() {
        // SAFETY: guarded by #[serial]; no other thread reads the environment
        // while this test runs.
        unsafe { std::env::set_var("NEAT_AI_DISCOVERY_NEURON_RANKING_BANDS", "1") };
        let big_gain = neuron_rank_score(1.0e-3, 0.1);
        let reliable = neuron_rank_score(1.0e-9, 1.0);
        // SAFETY: guarded by #[serial]; no other thread reads the environment
        // while this test runs.
        unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_NEURON_RANKING_BANDS") };
        assert!(
            big_gain > reliable,
            "the env override must collapse the ranking to gain order: {big_gain} vs {reliable}"
        );
    }
}
