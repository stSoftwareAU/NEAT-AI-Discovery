//! Unit tests for the focus module's internal components.
//!
//! These tests exercise internal record provider behaviour that cannot be
//! tested through the public API alone.

use super::ranking::*;
use crate::types::DiscoverRecord;
use anyhow::anyhow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn lazy_provider_defers_loading_and_bounds_cache() -> anyhow::Result<()> {
    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", 2, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // No eager loads during initialisation
    assert_eq!(0, loads.load(Ordering::SeqCst));

    // First load hits the loader, subsequent load for same neuron is cached
    provider.get("a")?.expect("records should be present");
    assert_eq!(1, loads.load(Ordering::SeqCst));
    provider.get("a")?.expect("records should be cached");
    assert_eq!(1, loads.load(Ordering::SeqCst));

    // Loading a second neuron increments once and cache remains bounded
    provider.get("b")?.expect("records should be present");
    assert_eq!(2, loads.load(Ordering::SeqCst));
    assert!(provider.len() <= 2);
    Ok(())
}

/// Issue #1374: A cache sized to the ranking working set must materialise each
/// neuron **at most once** across the ranking pipeline's multiple passes.
///
/// The pathological lazy-mode cost was `passes × neurons × full-file-decode`
/// because the default 8-entry cache thrashed when the working set was larger.
/// With the cache sized to the working set, the per-neuron loader is invoked
/// `O(neurons)` times, not `O(passes × neurons)`.
#[test]
fn sized_cache_loads_each_neuron_at_most_once_across_passes() -> anyhow::Result<()> {
    const NEURONS: usize = 20;
    const PASSES: usize = 5;

    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", NEURONS, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // Simulate the ranking pipeline's repeated sweeps over every neuron.
    for _pass in 0..PASSES {
        for n in 0..NEURONS {
            provider
                .get(&format!("n{n}"))?
                .expect("records should be present");
        }
    }

    assert_eq!(
        NEURONS,
        loads.load(Ordering::SeqCst),
        "each neuron must be loaded exactly once across {PASSES} passes when the \
         cache is sized to the working set (O(neurons), not O(passes × neurons))"
    );
    Ok(())
}

/// Issue #1374: Seeding the cache from a single grouped parquet pass must let
/// every seeded neuron be served from cache, so the per-neuron full-file loader
/// is never invoked during ranking.
#[test]
fn seeded_cache_serves_neurons_without_invoking_loader() -> anyhow::Result<()> {
    const NEURONS: usize = 12;

    let loads = Arc::new(AtomicUsize::new(0));
    let provider = LazyRecordProvider::with_loader_for_tests("unused.parquet", NEURONS, {
        let loads = Arc::clone(&loads);
        Arc::new(move |_file, neuron_uuid| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: neuron_uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            }])
        })
    });

    // Warm the cache with one grouped batch (records intentionally unsorted to
    // verify seed sorts by obs_index).
    let mut grouped = std::collections::HashMap::new();
    for n in 0..NEURONS {
        let uuid = format!("n{n}");
        grouped.insert(
            uuid.clone(),
            vec![
                DiscoverRecord {
                    obs_index: 2,
                    neuron_uuid: uuid.clone(),
                    value: None,
                    activation: 0.0,
                    errors: vec![0.0],
                },
                DiscoverRecord {
                    obs_index: 0,
                    neuron_uuid: uuid.clone(),
                    value: None,
                    activation: 0.0,
                    errors: vec![0.0],
                },
            ],
        );
    }
    provider.seed(grouped)?;

    // Multiple passes over the seeded neurons must all be cache hits.
    for _pass in 0..5 {
        for n in 0..NEURONS {
            let records = provider
                .get(&format!("n{n}"))?
                .expect("seeded records should be present");
            assert_eq!(records[0].obs_index, 0, "seed must sort by obs_index");
            assert_eq!(records[1].obs_index, 2);
        }
    }

    assert_eq!(
        0,
        loads.load(Ordering::SeqCst),
        "seeded neurons must be served from cache without any per-neuron loads"
    );
    Ok(())
}

#[test]
fn lazy_provider_returns_loader_errors_with_context() {
    let provider = LazyRecordProvider::with_loader_for_tests("failing.parquet", 2, {
        Arc::new(|file, neuron_uuid| {
            Err(anyhow!(
                "Simulated parquet read failure for {neuron_uuid} in {file}"
            ))
        })
    });

    let err = provider
        .get("hidden-1")
        .expect_err("loader error should surface");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Simulated parquet read failure"),
        "expected loader error, got: {msg}"
    );
    assert!(
        msg.contains("hidden-1"),
        "neuron context should be present: {msg}"
    );
    assert!(
        msg.contains("failing.parquet"),
        "file context should be present: {msg}"
    );
}
