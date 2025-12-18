//! Regression tests for Parquet read limiting.
//!
//! The Parquet reader supports an optional `max_obs` limit to avoid loading huge
//! recordings into memory.
//!
//! Important: The Parquet file may be written in different row orders.
//! - Observation-first order (all neurons for obs 0, then all neurons for obs 1, ...)
//! - Neuron-first order (all observations for neuron A, then all observations for neuron B, ...)
//!
//! When limiting by `max_obs`, we must ensure we still collect *all* records for
//! already-accepted observation indices, regardless of row ordering.

use neat_ai_discovery::parquet_format::{
    read_records_from_parquet_with_limit, write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::collections::HashSet;
use tempfile::NamedTempFile;

#[test]
fn read_records_from_parquet_with_limit_handles_neuron_first_ordering() {
    let temp_file = NamedTempFile::new().expect("Failed to create temp Parquet file");
    let file_path = temp_file.path().to_str().unwrap();

    // Write records in neuron-first order:
    // neuron-a: obs 0, obs 1
    // neuron-b: obs 0, obs 1
    // If `max_obs=1`, we want ALL rows for obs 0, including neuron-b's obs 0 row.
    let records = vec![
        DiscoverRecord::new(0, "neuron-a".to_string(), Some(1.0), 1.0, vec![0.0]),
        DiscoverRecord::new(1, "neuron-a".to_string(), Some(2.0), 2.0, vec![0.0]),
        DiscoverRecord::new(0, "neuron-b".to_string(), Some(3.0), 3.0, vec![0.0]),
        DiscoverRecord::new(1, "neuron-b".to_string(), Some(4.0), 4.0, vec![0.0]),
    ];

    write_records_to_parquet(file_path, &records).expect("Failed to write Parquet");

    let limited = read_records_from_parquet_with_limit(file_path, Some(1))
        .expect("Failed to read Parquet with limit");

    // We should have both neuron rows for obs 0.
    assert_eq!(
        limited.len(),
        2,
        "Expected all rows for the first accepted obs_index (0), even in neuron-first order"
    );

    // Ensure we only included obs 0.
    assert!(
        limited.iter().all(|r| r.obs_index == 0),
        "All returned rows should be for obs_index 0"
    );

    let uuids: HashSet<String> = limited.into_iter().map(|r| r.neuron_uuid).collect();
    assert!(uuids.contains("neuron-a"), "Missing neuron-a row for obs 0");
    assert!(uuids.contains("neuron-b"), "Missing neuron-b row for obs 0");
}
