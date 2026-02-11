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
