//! Guard tests for Issue #2013: lazy analysis must honour the shared deadline
//! and skip when the projected pre-load is unworkably over budget.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * 1024 * 1024;

#[test]
fn budget_path_skips_when_projection_exceeds_budget_by_more_than_10x() {
    // observed host shape: projected 85113 MB vs budget 4004 MB ≈ 21×.
    let projected = 85_113 * MB;
    let budget_mb = 4004;
    let (mode, reason) = decide_cache_preload_for_budget(projected, budget_mb);
    assert_eq!(mode, CachePreloadMode::SkipUnworkable);
    assert_eq!(reason, CacheLazyReason::Unworkable);
}

#[test]
fn budget_path_stays_lazy_just_under_10x_overbook() {
    // 9× overbook → still lazy (recoverable), not skip.
    let budget_mb = 100;
    let projected = 9 * budget_mb * MB;
    let (mode, reason) = decide_cache_preload_for_budget(projected, budget_mb);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::Budget);
}

#[test]
fn budget_path_zero_budget_still_forces_lazy_not_skip() {
    // Operators / tests use budget=0 to force lazy; must not skip every file.
    let (mode, reason) = decide_cache_preload_for_budget(200 * MB, 0);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::Budget);
}

/// Issue #4138: a supplied budget that would pre-load still falls back to lazy
/// when host-available memory cannot hold the projection.
#[test]
fn lazy_engages_under_genuine_memory_pressure_with_budget_present() {
    let projected = 4 * 1024 * MB; // 4 GB
    let budget_mb = 8192; // 8 GB budget would otherwise pre-load
    let available = 2 * 1024 * MB; // 2 GB reclaimable
    let margin = GB;
    let (mode, reason, logged) =
        decide_analysis_cache_preload(projected, Some(budget_mb), available, margin, 8192);
    assert_eq!(mode, CachePreloadMode::Lazy);
    assert_eq!(reason, CacheLazyReason::MemoryPressure);
    assert_eq!(logged, budget_mb, "budget_mb in the log must stay non-zero");
    assert_ne!(reason, CacheLazyReason::NoBudget);
}

#[test]
fn lazy_get_returns_at_deadline() {
    // Simulated over-deadline analysis phase: a past deadline must abort
    // before the loader runs (AC: guard test).
    let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_loader = Arc::clone(&calls);
    let cache = RecordCache::with_loader_and_deadline(
        "unused.parquet",
        Arc::new(move |_file, _uuid| {
            calls_loader.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Vec::<DiscoverRecord>::new())
        }),
        Some(SystemTime::now() - Duration::from_secs(1)),
    );

    let err = cache
        .get("neuron-a")
        .expect_err("past deadline must abort lazy get");
    assert_eq!(
        err.downcast_ref::<AnalysisDeadlineExceeded>(),
        Some(&AnalysisDeadlineExceeded {
            neuron_uuid: "neuron-a".to_string()
        }),
        "a deadline bail must be the typed error so callers can tell it from a load failure"
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline exceeded") || msg.contains("Issue #2013"),
        "unexpected error: {msg}"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "loader must not run after the deadline"
    );
}

#[test]
fn lazy_get_runs_loader_when_deadline_in_future() {
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false]);
    let cache = RecordCache::with_loader_and_deadline(
        "unused.parquet",
        Arc::new(|_file, _uuid| Ok(Vec::<DiscoverRecord>::new())),
        Some(SystemTime::now() + Duration::from_secs(3600)),
    );
    let records = cache.get("neuron-a").expect("future deadline allows load");
    assert!(records.is_empty());
}

/// Issue #2408: orchestration builds the cache with the loading deadline, so
/// records that are already in the cache must stay readable after it passes.
#[test]
fn preloaded_records_served_after_loading_deadline() {
    let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_loader = Arc::clone(&calls);
    let cache = RecordCache::with_loader_and_deadline(
        "unused.parquet",
        Arc::new(move |_file, _uuid| {
            calls_loader.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Vec::<DiscoverRecord>::new())
        }),
        Some(SystemTime::now() - Duration::from_secs(1)),
    );
    let stored = Arc::new(vec![DiscoverRecord::new(
        0,
        "neuron-a".to_string(),
        Some(1.0),
        0.5,
        vec![0.1],
    )]);
    let cell = OnceLock::new();
    cell.set(Ok(Arc::clone(&stored))).ok();
    cache
        .cache
        .write()
        .insert("neuron-a".to_string(), Arc::new(cell));

    let records = cache
        .get("neuron-a")
        .expect("already-loaded records must be served after the deadline passes");
    assert!(Arc::ptr_eq(&records, &stored));
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "loader must not run for a pre-loaded neuron"
    );
}

/// Issue #2408: the deadline gates only cache misses, before the loader runs.
#[test]
fn loaded_records_still_served_once_deadline_passes() {
    // First poll (miss for neuron-a): not passed. Next poll (miss for
    // neuron-b): passed. A hit on neuron-a must not consume a poll.
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_loader = Arc::clone(&calls);
    let cache = RecordCache::with_loader_and_deadline(
        "unused.parquet",
        Arc::new(move |_file, uuid| {
            calls_loader.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![DiscoverRecord::new(
                0,
                uuid.to_string(),
                None,
                0.25,
                vec![],
            )])
        }),
        Some(SystemTime::now() + Duration::from_secs(3600)),
    );

    let first = cache.get("neuron-a").expect("first get loads");
    assert_eq!(first.len(), 1);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    let second = cache
        .get("neuron-a")
        .expect("loaded records stay readable once the deadline passes");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    let err = cache
        .get("neuron-b")
        .expect_err("an uncached neuron after the deadline must abort");
    assert_eq!(
        err.downcast_ref::<AnalysisDeadlineExceeded>(),
        Some(&AnalysisDeadlineExceeded {
            neuron_uuid: "neuron-b".to_string()
        })
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "loader must not run for the uncached neuron after the deadline"
    );
}
