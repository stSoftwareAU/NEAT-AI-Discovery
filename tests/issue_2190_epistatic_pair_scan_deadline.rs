//! Issue #2190: the O(n²) epistatic pair scan honours the analysis deadline
//! and a ceiling on the candidates it emits, and reports an early return.
//!
//! Deterministic: an elapsed deadline is `UNIX_EPOCH`, a live one is a day
//! ahead — no sleeps and no timing thresholds.

use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::recommendation::epistatic::{
    EpistaticPairCandidate, MAX_EPISTATIC_PAIR_CANDIDATES, ScanTruncation, SourceContribution,
    SynergisticCandidate, build_source_contribution, detect_epistatic_pairs,
    detect_epistatic_pairs_with_deadline, detect_synergistic_candidates,
    detect_synergistic_candidates_with_deadline,
};
use neat_ai_discovery::analysis::samples::{HelpfulSample, HelpfulStats};

const SAMPLES_PER_SOURCE_PER_HALF: usize = 2;
const MIN_SAMPLES: usize = 40;

/// `n` sources that are pairwise fully complementary: source `i` fires only on
/// samples where `index % n == i`, twice in each half so every pair
/// cross-validates. Every one of the `n(n-1)/2` pairs is an epistatic pair.
fn complementary_sources(n: usize) -> Vec<SourceContribution> {
    // Floor keeps small sets above the detectors' minimum sample counts.
    let sample_count = (n * SAMPLES_PER_SOURCE_PER_HALF * 2).max(MIN_SAMPLES);
    (0..n)
        .map(|i| {
            let samples = (0..sample_count)
                .map(|idx| HelpfulSample {
                    activation: if idx % n == i { 1.0 } else { 0.0 },
                    avg_error: 0.3,
                    target_value: None,
                    target_activation: None,
                })
                .collect();
            build_source_contribution(
                &format!("source-{i}"),
                samples,
                HelpfulStats::default(),
                0.1,
                0.0,
            )
        })
        .collect()
}

fn elapsed() -> Option<SystemTime> {
    Some(SystemTime::UNIX_EPOCH)
}

fn far_future() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

fn pair_keys(pairs: &[(String, String)]) -> Vec<(String, String)> {
    let mut keys = pairs.to_vec();
    keys.sort();
    keys
}

#[test]
fn elapsed_deadline_stops_the_epistatic_pair_scan_before_any_pair() {
    let sources = complementary_sources(24);

    let scan = detect_epistatic_pairs_with_deadline("output-0", &sources, 1.0, None, &elapsed());

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(
        scan.candidates.is_empty(),
        "an elapsed deadline must stop the scan before the first row, got {} pairs",
        scan.candidates.len()
    );
}

#[test]
fn future_deadline_yields_every_pair_the_unbounded_scan_yields() {
    let n = 24;
    let sources = complementary_sources(n);

    let scan = detect_epistatic_pairs_with_deadline("output-0", &sources, 1.0, None, &far_future());
    let legacy = detect_epistatic_pairs("output-0", &sources, 1.0, None);

    assert_eq!(scan.truncation, None);
    assert_eq!(scan.candidates.len(), n * (n - 1) / 2);
    let as_keys = |v: &[EpistaticPairCandidate]| {
        pair_keys(
            &v.iter()
                .map(|p| (p.source_a_uuid.clone(), p.source_b_uuid.clone()))
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(as_keys(&scan.candidates), as_keys(&legacy));
}

#[test]
fn epistatic_pair_scan_stops_at_the_candidate_ceiling_and_reports_it() {
    // 48 fully complementary sources would yield 1,128 pairs.
    let n = 48;
    assert!(n * (n - 1) / 2 > MAX_EPISTATIC_PAIR_CANDIDATES);
    let sources = complementary_sources(n);

    let scan = detect_epistatic_pairs_with_deadline("output-0", &sources, 1.0, None, &far_future());

    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(scan.candidates.len(), MAX_EPISTATIC_PAIR_CANDIDATES);
}

#[test]
fn candidate_ceiling_not_reported_when_the_scan_completes() {
    // Two sources: one pair, well under the ceiling.
    let sources = complementary_sources(2);

    let scan = detect_epistatic_pairs_with_deadline("output-0", &sources, 1.0, None, &far_future());

    assert_eq!(scan.truncation, None);
    assert_eq!(scan.candidates.len(), 1);
}

#[test]
fn too_few_sources_is_not_a_truncation_even_past_the_deadline() {
    let sources = complementary_sources(1);

    let scan = detect_epistatic_pairs_with_deadline("output-0", &sources, 1.0, None, &elapsed());

    assert_eq!(scan.truncation, None);
    assert!(scan.candidates.is_empty());
}

#[test]
fn elapsed_deadline_stops_the_synergistic_scan() {
    let sources = complementary_sources(24);

    let scan =
        detect_synergistic_candidates_with_deadline("output-0", &sources, 1.0, None, &elapsed());

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(scan.candidates.is_empty());
}

#[test]
fn future_deadline_synergistic_scan_matches_the_unbounded_scan() {
    let sources = complementary_sources(24);

    let scan =
        detect_synergistic_candidates_with_deadline("output-0", &sources, 1.0, None, &far_future());
    let legacy = detect_synergistic_candidates("output-0", &sources, 1.0, None);

    assert_eq!(scan.truncation, None);
    let keys = |v: &[SynergisticCandidate]| {
        pair_keys(
            &v.iter()
                .map(|c| {
                    (
                        c.primary_source_uuid.clone(),
                        c.complement_source_uuid.clone(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(keys(&scan.candidates), keys(&legacy));
}
