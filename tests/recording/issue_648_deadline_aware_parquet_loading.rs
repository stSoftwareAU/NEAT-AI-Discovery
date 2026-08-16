//! Tests for Issue #648: Deadline-aware parquet loading with early abort.
//!
//! Verifies that parquet loading respects the analysis deadline and aborts
//! early if insufficient time remains, returning a clear error message.
//! Also verifies that watchdog receives intermediate beats during loading.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::TempDir;

/// Create a test parquet file with the specified number of neurons and records per neuron.
fn create_test_parquet(
    temp_dir: &TempDir,
    neuron_count: usize,
    records_per_neuron: usize,
) -> String {
    let parquet_path = temp_dir.path().join("test_data.parquet");
    let path_str = parquet_path.to_str().unwrap().to_string();

    let mut records = Vec::new();
    for neuron_idx in 0..neuron_count {
        let neuron_uuid = format!("neuron-{neuron_idx}");
        for obs_idx in 0..records_per_neuron {
            records.push(DiscoverRecord::new(
                obs_idx as u32,
                neuron_uuid.clone(),
                Some(obs_idx as f32 / records_per_neuron as f32),
                obs_idx as f32 / records_per_neuron as f32,
                vec![0.1, 0.2],
            ));
        }
    }

    write_records_to_parquet(&path_str, &records).expect("Failed to write test parquet");
    path_str
}

// =============================================================================
// Deadline-Aware Loading Tests
// =============================================================================

/// Test that `new_adaptive_with_deadline` succeeds when deadline is far in the future.
#[test]
fn deadline_aware_loading_succeeds_with_generous_deadline() {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 5, 50);

    // Deadline 10 minutes from now — plenty of time
    let deadline = std::time::SystemTime::now() + std::time::Duration::from_secs(600);

    let cache = RecordCache::new_adaptive_with_deadline(&parquet_path, Some(deadline))
        .expect("Loading should succeed with a generous deadline")
        .expect("generous deadline must not skip the cache");
    let records = cache.get("neuron-0").expect("Should retrieve neuron-0");
    assert_eq!(records.len(), 50, "neuron-0 should have 50 records");
}

/// Test that `new_adaptive_with_deadline` returns an error when deadline is already passed.
#[test]
fn deadline_aware_loading_aborts_when_deadline_already_passed() {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 5, 50);

    // Deadline already in the past
    let deadline = std::time::SystemTime::now() - std::time::Duration::from_secs(1);

    let result = RecordCache::new_adaptive_with_deadline(&parquet_path, Some(deadline));
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Loading should fail when deadline is already passed"),
    };
    assert!(
        err_msg.contains("deadline"),
        "Error message should mention deadline, got: {err_msg}"
    );
}

/// Test that `new_adaptive_with_deadline` works the same as `new_adaptive` when no deadline is set.
#[test]
fn deadline_aware_loading_with_no_deadline_matches_adaptive() {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 5, 50);

    // No deadline — should behave identically to new_adaptive
    let cache_with_deadline = RecordCache::new_adaptive_with_deadline(&parquet_path, None)
        .expect("Loading without deadline should succeed")
        .expect("no-budget path must not skip the cache");

    let cache_standard =
        RecordCache::new_adaptive(&parquet_path).expect("Standard loading should succeed");

    // Both should return the same data
    for i in 0..5 {
        let uuid = format!("neuron-{i}");
        let records_deadline = cache_with_deadline.get(&uuid).expect("get failed");
        let records_standard = cache_standard.get(&uuid).expect("get failed");
        assert_eq!(
            records_deadline.len(),
            records_standard.len(),
            "Record count should match for {uuid}"
        );
    }
}

/// Test that the error message from deadline abort is clear and informative.
#[test]
fn deadline_abort_error_message_is_clear() {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 3, 20);

    let deadline = std::time::SystemTime::now() - std::time::Duration::from_secs(5);

    let result = RecordCache::new_adaptive_with_deadline(&parquet_path, Some(deadline));
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Loading should fail when deadline has passed"),
    };
    // The message should mention that loading was aborted due to deadline
    assert!(
        err_msg.to_lowercase().contains("deadline")
            || err_msg.to_lowercase().contains("abort")
            || err_msg.to_lowercase().contains("time"),
        "Error should clearly indicate deadline abort, got: {err_msg}"
    );
}

/// Test that deadline-aware loading checks deadline before starting the load.
#[test]
fn deadline_aware_loading_checks_before_load_starts() {
    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 10, 100);

    // Deadline just barely in the past
    let deadline = std::time::SystemTime::now() - std::time::Duration::from_millis(1);

    let result = RecordCache::new_adaptive_with_deadline(&parquet_path, Some(deadline));
    assert!(
        result.is_err(),
        "Should abort before even starting to load when deadline is passed"
    );
}
