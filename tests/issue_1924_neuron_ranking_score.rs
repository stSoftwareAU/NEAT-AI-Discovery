//! Issue #1924 — `add-neurons` were ranked on `expectedCreatureScoreGain`,
//! which anti-correlates with success (r = −0.61).
//!
//! The corpus below is the complete set of 13 cached `add-neurons` records
//! measured by the Issue #1920 candidates-cache study
//! (`docs/analysis/candidates-cache-study-1920.md`, findings B and C): the
//! prediction the engine sent, the sample counts behind it, and the
//! `scoreDelta` the controller actually realised.
//!
//! The acceptance measure the issue names is `r vs success` for the field the
//! ranking sorts on. These tests assert it directly: negative for the old key,
//! positive for the new one, over the same records.

use neat_ai_discovery::analysis::cache_study::stats::pearson;
use neat_ai_discovery::analysis::neuron::ranking_score::{improved_share, neuron_rank_score};

/// One cached `add-neurons` record: predicted gain, improved samples, total
/// samples, realised `scoreDelta`.
struct Record {
    expected_gain: f64,
    improved_count: u32,
    total_count: u32,
    score_delta: f64,
}

impl Record {
    fn succeeded(&self) -> bool {
        self.score_delta > 0.0
    }
}

/// The 13 `add-neurons` records in the production candidates cache on
/// 2026-08-02 (model hash `c1885aa6`), four of which succeeded.
const CORPUS: &[Record] = &[
    Record {
        expected_gain: 1.0626e-2,
        improved_count: 174,
        total_count: 315,
        score_delta: -2.0396e-5,
    },
    Record {
        expected_gain: 3.9766e-3,
        improved_count: 174,
        total_count: 315,
        score_delta: -7.3711e-6,
    },
    Record {
        expected_gain: 3.8972e-3,
        improved_count: 174,
        total_count: 315,
        score_delta: -8.3666e-6,
    },
    Record {
        expected_gain: 1.0180e-3,
        improved_count: 130,
        total_count: 206,
        score_delta: 2.2497e-6,
    },
    Record {
        expected_gain: 9.5263e-4,
        improved_count: 129,
        total_count: 206,
        score_delta: -1.7335e-5,
    },
    Record {
        expected_gain: 9.4921e-4,
        improved_count: 116,
        total_count: 206,
        score_delta: -1.2417e-5,
    },
    Record {
        expected_gain: 3.5983e-4,
        improved_count: 204,
        total_count: 204,
        score_delta: -5.6364e-8,
    },
    Record {
        expected_gain: 2.6381e-5,
        improved_count: 136,
        total_count: 206,
        score_delta: -5.8245e-6,
    },
    Record {
        expected_gain: 2.5649e-5,
        improved_count: 143,
        total_count: 206,
        score_delta: -3.2437e-6,
    },
    Record {
        expected_gain: 2.5175e-5,
        improved_count: 144,
        total_count: 206,
        score_delta: -2.7238e-6,
    },
    Record {
        expected_gain: 1.9088e-5,
        improved_count: 137,
        total_count: 205,
        score_delta: 2.9214e-6,
    },
    Record {
        expected_gain: 1.7601e-7,
        improved_count: 205,
        total_count: 205,
        score_delta: 8.2503e-8,
    },
    Record {
        expected_gain: 1.5759e-7,
        improved_count: 204,
        total_count: 204,
        score_delta: 5.4848e-7,
    },
];

fn success_indicators() -> Vec<f64> {
    CORPUS
        .iter()
        .map(|r| if r.succeeded() { 1.0 } else { 0.0 })
        .collect()
}

fn rank_scores() -> Vec<f64> {
    CORPUS
        .iter()
        .map(|r| {
            neuron_rank_score(
                r.expected_gain,
                improved_share(r.improved_count, r.total_count),
            )
        })
        .collect()
}

/// Mean 0-indexed position of the successes when the corpus is ordered
/// best-first by `key`. Lower is better: the ranking is putting the candidates
/// that actually landed nearer the top of the batch.
fn mean_success_position(keys: &[f64]) -> f64 {
    let mut ordered: Vec<usize> = (0..CORPUS.len()).collect();
    ordered.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
    let positions: Vec<u32> = ordered
        .iter()
        .enumerate()
        .filter(|&(_, &record)| CORPUS[record].succeeded())
        .map(|(position, _)| u32::try_from(position).expect("corpus is 13 records"))
        .collect();
    let count = u32::try_from(positions.len()).expect("at most 13 successes");
    f64::from(positions.iter().sum::<u32>()) / f64::from(count)
}

/// The defect: the field the ranking used to sort on predicts failure.
#[test]
fn expected_gain_anti_correlates_with_success() {
    let gains: Vec<f64> = CORPUS.iter().map(|r| r.expected_gain.log10()).collect();
    let r = pearson(&gains, &success_indicators()).expect("13 paired records");
    assert!(
        r < -0.5,
        "the study measured expectedCreatureScoreGain at r = -0.61 against success; got {r:.3}"
    );
}

/// The fix: the field the ranking now sorts on tracks success positively.
#[test]
fn rank_score_correlates_positively_with_success() {
    let r = pearson(&rank_scores(), &success_indicators()).expect("13 paired records");
    assert!(
        r > 0.4,
        "the rank score must track success at least as well as improvedShare (r = +0.466); \
         got {r:.3}"
    );
}

/// Ranking is about ordering, not just correlation: the successes must move up
/// the batch. Under the old key they sat at mean position 9.0 of 13.
#[test]
fn successes_rank_higher_than_under_the_gain_ordering() {
    let gains: Vec<f64> = CORPUS.iter().map(|r| r.expected_gain).collect();
    let before = mean_success_position(&gains);
    let after = mean_success_position(&rank_scores());
    assert!(
        after < before,
        "successes must rank nearer the top than under gain ordering: {after:.1} vs {before:.1}"
    );
}

/// The four biggest predictions in the corpus all lost score. None of them may
/// take the top slot of the batch any more.
#[test]
fn the_most_over_confident_prediction_is_no_longer_ranked_first() {
    let scores = rank_scores();
    let top = (0..CORPUS.len())
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("non-empty corpus");
    assert!(
        CORPUS[top].expected_gain < 1.0e-3,
        "the top-ranked candidate must not be one of the over-confident predictions; \
         got expected gain {:e}",
        CORPUS[top].expected_gain
    );
}

/// The explicit trade-off the issue asks for: reliability first, gain second.
#[test]
fn reliability_outranks_a_larger_prediction() {
    let over_confident = neuron_rank_score(1.0626e-2, improved_share(174, 315));
    let reliable = neuron_rank_score(1.5759e-7, improved_share(204, 204));
    assert!(
        reliable > over_confident,
        "a candidate improving every sample must outrank a 67,000x larger prediction \
         that improves 55% of them: {reliable} vs {over_confident}"
    );
}
