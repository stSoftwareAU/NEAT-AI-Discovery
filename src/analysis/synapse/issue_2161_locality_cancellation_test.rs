//! Issue #2161 regression tests for `group_sources_by_locality`.
//!
//! The pairwise locality scan is O(n²) in the source count. Before this fix it
//! carried no cancellation point, so neither `analysis_deadline_ms` nor a host
//! cancellation request could interrupt it once started. These tests pin both
//! halves of the fix: the per-iteration deadline check, and the source-count
//! ceiling above which the quadratic scan is never entered at all.

use super::{
    MAX_SOURCES_FOR_LOCALITY_SCAN, SampleLocalityGroup, group_sources_by_locality,
    group_sources_by_locality_observed,
};
use crate::analysis::utils::OrderedNeuron;
use crate::types::DiscoverRecord;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Observation indices attributed to each synthetic source.
const RECORDS_PER_SOURCE: u32 = 4;

/// Observation-index layout for the synthetic sources built by [`build_sources`].
#[derive(Clone, Copy)]
enum ObsLayout {
    /// Every source observes the same indices, so the scan collapses them all
    /// into a single group.
    Shared,
    /// No two sources share an index, so no source is ever absorbed — the worst
    /// case for the pairwise scan, and the one the issue's trigger relies on.
    Disjoint,
}

fn build_sources(
    count: usize,
    layout: ObsLayout,
) -> (Vec<OrderedNeuron>, Vec<Arc<Vec<DiscoverRecord>>>) {
    let mut neurons = Vec::with_capacity(count);
    let mut records = Vec::with_capacity(count);

    for i in 0..count {
        let uuid = format!("source-{i}");
        let base = match layout {
            ObsLayout::Shared => 0,
            ObsLayout::Disjoint => {
                u32::try_from(i).expect("synthetic source count fits in u32") * RECORDS_PER_SOURCE
            }
        };
        let rows: Vec<DiscoverRecord> = (0..RECORDS_PER_SOURCE)
            .map(|k| DiscoverRecord::new(base + k, uuid.clone(), Some(1.0), 1.0, vec![0.1]))
            .collect();

        neurons.push(OrderedNeuron { uuid, index: i });
        records.push(Arc::new(rows));
    }

    (neurons, records)
}

fn as_pairs<'a>(
    neurons: &'a [OrderedNeuron],
    records: &[Arc<Vec<DiscoverRecord>>],
) -> Vec<(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)> {
    neurons
        .iter()
        .zip(records.iter())
        .map(|(n, r)| (n, Arc::clone(r)))
        .collect()
}

/// The grouping contract: every source appears in exactly one group, always.
fn assert_every_source_in_exactly_one_group(
    groups: &[SampleLocalityGroup<'_>],
    expected_sources: usize,
) {
    let mut seen: HashSet<usize> = HashSet::new();
    for group in groups {
        for (neuron, _) in &group.sources {
            assert!(
                seen.insert(neuron.index),
                "source {} appeared in more than one group",
                neuron.index
            );
        }
    }
    assert_eq!(seen.len(), expected_sources, "grouping dropped sources");
}

/// Count the pairwise overlap comparisons one grouping pass performs (Issue #2296).
///
/// The count is the scan's unit of work, so asserting on it is deterministic
/// where a wall-clock reading flakes under a loaded parallel test run.
fn count_overlap_comparisons(sources: &[(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)]) -> usize {
    let mut comparisons = 0usize;
    let groups = group_sources_by_locality_observed(sources, &None, || comparisons += 1);
    assert_every_source_in_exactly_one_group(&groups, sources.len());
    comparisons
}

#[test]
fn expired_deadline_stops_locality_scan_without_dropping_sources() {
    const SOURCE_COUNT: usize = 64;

    let (neurons, records) = build_sources(SOURCE_COUNT, ObsLayout::Shared);
    let sources = as_pairs(&neurons, &records);

    // Issue #1799: establish the positive precondition first — without a
    // deadline these sources really are collapsed by the pairwise scan, so the
    // assertion below is observing cancellation and not an inert fixture.
    let grouped = group_sources_by_locality(&sources, &None);
    assert_eq!(
        grouped.len(),
        1,
        "identical observation indices must collapse into a single group"
    );
    assert_every_source_in_exactly_one_group(&grouped, SOURCE_COUNT);

    let expired = Some(SystemTime::now() - Duration::from_secs(60));
    let cancelled = group_sources_by_locality(&sources, &expired);

    assert_eq!(
        cancelled.len(),
        SOURCE_COUNT,
        "an expired deadline must abandon the pairwise scan and emit single-source groups"
    );
    for group in &cancelled {
        assert_eq!(
            group.sources.len(),
            1,
            "every group emitted after cancellation holds exactly one source"
        );
    }
    assert_every_source_in_exactly_one_group(&cancelled, SOURCE_COUNT);
}

#[test]
fn locality_grouping_cost_does_not_grow_quadratically() {
    // Issue #1799: positive precondition — below the ceiling the observer sees
    // every pair of a disjoint fixture, so a zero count further down means the
    // scan was skipped and not that the counter is inert.
    const SCANNED: usize = 16;
    let (neurons, records) = build_sources(SCANNED, ObsLayout::Disjoint);
    assert_eq!(
        count_overlap_comparisons(&as_pairs(&neurons, &records)),
        SCANNED * (SCANNED - 1) / 2,
        "a disjoint fixture below the ceiling must compare every pair once"
    );

    // Issue #2296: count the scan's work instead of timing it. Above the
    // ceiling the quadratic scan must never be entered, so the comparison
    // count stays at zero however far the source count grows — a load spike
    // on a parallel test run cannot move it.
    let small = MAX_SOURCES_FOR_LOCALITY_SCAN + 1;
    let large = small * 2;
    let (neurons, records) = build_sources(large, ObsLayout::Disjoint);
    let large_sources = as_pairs(&neurons, &records);

    let small_comparisons = count_overlap_comparisons(&large_sources[..small]);
    let large_comparisons = count_overlap_comparisons(&large_sources);
    assert_eq!(
        (small_comparisons, large_comparisons),
        (0, 0),
        "locality scan entered above the {MAX_SOURCES_FOR_LOCALITY_SCAN}-source ceiling: \
         {small_comparisons} comparisons at {small} sources, {large_comparisons} at {large}"
    );
}
