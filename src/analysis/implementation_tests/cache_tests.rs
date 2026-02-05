//! Tests for record cache functionality.
//!
//! Tests cover:
//! - Record cache contention handling (loads once per neuron under contention)

use super::common::*;

#[test]
fn record_cache_loads_once_per_neuron_under_contention() {
    let load_counter = Arc::new(AtomicUsize::new(0));
    let loader_counter = Arc::clone(&load_counter);
    let loader = Arc::new(
        move |_file: &str, neuron_uuid: &str| -> Result<Vec<DiscoverRecord>> {
            loader_counter.fetch_add(1, AtomicOrdering::SeqCst);
            thread::sleep(Duration::from_millis(50));
            Ok(vec![DiscoverRecord::new(
                0,
                neuron_uuid.to_string(),
                None,
                0.0,
                Vec::new(),
            )])
        },
    );

    let cache = Arc::new(RecordCache::with_loader("unused.parquet", loader));
    let worker_count = 4;
    let barrier = Arc::new(Barrier::new(worker_count));
    let mut handles = Vec::new();
    for _ in 0..worker_count {
        let cache_clone = Arc::clone(&cache);
        let barrier_clone = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier_clone.wait();
            cache_clone
                .get("neuron-1")
                .expect("cache should load neuron records");
        }));
    }

    for handle in handles {
        handle.join().expect("worker thread should exit cleanly");
    }

    assert_eq!(
        load_counter.load(AtomicOrdering::SeqCst),
        1,
        "record cache should only hit the loader once even when multiple threads request the same neuron"
    );
}
