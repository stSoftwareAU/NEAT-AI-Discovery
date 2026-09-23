//! Issue #2161 regression tests for `group_sources_by_locality`.
//!
//! The pairwise locality scan is O(n²) in the source count. Before this fix it
//! carried no cancellation point, so neither `analysis_deadline_ms` nor a host
//! cancellation request could interrupt it once started. These tests pin both
//! halves of the fix: the per-iteration deadline check, and the source-count
//! ceiling above which the quadratic scan is never entered at all.

use super::{MAX_SOURCES_FOR_LOCALITY_SCAN, SampleLocalityGroup, group_sources_by_locality};
use crate::analysis::utils::OrderedNeuron;
use crate::types::DiscoverRecord;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

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

fn min_grouping_time(sources: &[(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)]) -> Duration {
    const RUNS: usize = 5;

    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            let groups = group_sources_by_locality(sources, &None);
            let elapsed = start.elapsed();
            assert!(
                !groups.is_empty(),
                "grouping must always emit at least one group"
            );
            elapsed
        })
        .min()
        .expect("RUNS is non-zero")
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
    let small = MAX_SOURCES_FOR_LOCALITY_SCAN + 1;
    let large = small * 2;

    // Fixtures are built once, outside the timed region, and the smaller run
    // reuses a prefix of the larger so both time exactly the same kind of work.
    let (neurons, records) = build_sources(large, ObsLayout::Disjoint);
    let large_sources = as_pairs(&neurons, &records);
    let small_sources = &large_sources[..small];

    let t_small = min_grouping_time(small_sources).max(Duration::from_nanos(1));
    let t_large = min_grouping_time(&large_sources);

    // Two readings of the same work, never a reading against a wall-clock
    // constant: doubling the input may double the cost (linear) but must not
    // quadruple it (quadratic). The bound sits midway between the two.
    assert!(
        t_large <= t_small * 3,
        "locality grouping cost grew faster than linearly: {t_small:?} at {small} sources against {t_large:?} at {large} sources"
    );
}
