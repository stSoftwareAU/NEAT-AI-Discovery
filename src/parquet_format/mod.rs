//! Parquet file format handling for discovery records
//!
//! Sub-modules:
//! - `schema` — Schema definitions and validation
//! - `writer` — Parquet writing and serialisation
//! - `reader` — Parquet reading and deserialisation

pub mod decode_budget;
pub mod footer;
mod reader;
pub mod schema;
pub mod shared_records;
mod writer;

// Re-export public API at the parquet_format level for backward compatibility
pub use reader::{
    ColumnProfile, read_all_records_from_parquet, read_all_records_grouped_by_neuron,
    read_all_records_grouped_by_neuron_bounded, read_all_records_grouped_by_neuron_with_deadline,
    read_all_records_grouped_by_neuron_with_deadline_and_profile,
    read_all_records_grouped_by_neuron_with_profile, read_records_from_parquet,
    read_records_from_parquet_with_limit, read_records_from_parquet_with_limit_and_budget,
    read_records_from_parquet_with_limit_and_profile, read_records_from_parquet_with_profile,
};
pub use schema::create_schema;
pub use writer::{ParquetRecordWriter, merge_parquet_files, write_records_to_parquet};

#[cfg(test)]
pub(crate) use writer::write_records_to_parquet_with_limit;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use arrow::datatypes::DataType;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_schema() {
        let schema = create_schema();
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "obs_index");
        assert_eq!(schema.field(1).name(), "neuron_uuid");
        assert_eq!(schema.field(2).name(), "value");
        assert_eq!(schema.field(3).name(), "activation");
        assert_eq!(schema.field(4).name(), "errors");
    }

    #[test]
    fn test_create_schema_uses_utf8_for_neuron_uuid() {
        let schema = create_schema();
        assert_eq!(
            schema.field(1).data_type(),
            &DataType::Utf8,
            "Neuron UUID column should use Utf8 to ensure Parquet compatibility"
        );
    }

    #[test]
    fn test_write_records_to_parquet() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-1".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "hidden-1".to_string(), Some(0.6), 0.8, vec![0.15]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        // Verify file was created
        assert!(std::path::Path::new(file_path).exists());
    }

    #[test]
    fn test_write_empty_records_fails() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let result = write_records_to_parquet(file_path, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No records"));
    }

    #[test]
    fn test_write_records_prevents_i32_overflow() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-1".to_string(), Some(0.5), 0.7, vec![0.1; 1000]),
            DiscoverRecord::new(1, "hidden-2".to_string(), Some(0.6), 0.8, vec![0.2; 1000]),
        ];

        let result = write_records_to_parquet(file_path, &records);
        assert!(
            result.is_ok(),
            "Should handle valid error counts within i32::MAX"
        );

        assert!(std::path::Path::new(file_path).exists());
    }

    #[test]
    fn test_write_records_validates_error_count_overflow() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| DiscoverRecord::new(i, format!("neuron-{i}"), Some(0.5), 0.7, vec![0.1; 10]))
            .collect();

        let result = write_records_to_parquet(file_path, &records);
        assert!(result.is_ok(), "Validation should allow valid error counts");
    }

    #[test]
    fn test_write_records_chunks_when_uuid_bytes_exceed_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-0000".to_string(), Some(0.5), 0.7, vec![0.1]),
            DiscoverRecord::new(1, "hidden-1111".to_string(), Some(0.6), 0.8, vec![0.2]),
        ];

        let result = write_records_to_parquet_with_limit(file_path, &records, 16);
        assert!(
            result.is_ok(),
            "Writer should chunk records when total neuron UUID bytes exceed the configured per-batch limit"
        );

        assert!(
            std::path::Path::new(file_path).exists(),
            "Parquet file should exist after chunked write"
        );
    }

    #[test]
    fn test_write_records_errors_when_single_uuid_exceeds_batch_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "hidden-0000".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let err = write_records_to_parquet_with_limit(file_path, &records, 5)
            .expect_err("Single record with UUID longer than limit should be rejected");
        assert!(
            err.to_string().contains("exceeds configured limit"),
            "Error message should explain the batch limit violation"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_too_short() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected empty neuron UUID to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains("\"\""),
            "Error message should include the invalid UUID: {msg}"
        );
        assert!(
            msg.contains("letters, digits, or hyphens"),
            "Error message should explain valid characters: {msg}"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_with_invalid_characters() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "invalid_UUID".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected neuron UUID with invalid characters to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains("\"invalid_UUID\""),
            "Error message should include the invalid UUID: {msg}"
        );
        assert!(
            msg.contains("letters, digits, or hyphens"),
            "Error message should explain valid characters: {msg}"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_too_long() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let long_uuid = "a".repeat(101);

        let records = vec![DiscoverRecord::new(
            0,
            long_uuid.clone(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected overly long neuron UUID to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains(&long_uuid[..60]),
            "Error message should include the invalid UUID preview: {msg}"
        );
        assert!(
            msg.contains("between 1 and 100 characters"),
            "Error message should explain the valid length range: {msg}"
        );
    }

    #[test]
    fn test_write_records_accepts_single_character_uppercase_neuron_uuid() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "A".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        assert!(
            result.is_ok(),
            "Single-character uppercase neuron UUID should be accepted"
        );
    }

    #[test]
    fn test_read_records_from_parquet_preserves_obs_index() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(5, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1]),
            DiscoverRecord::new(2, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.15]),
            DiscoverRecord::new(8, "neuron-1".to_string(), Some(0.7), 0.9, vec![0.2]),
            DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.4), 0.6, vec![0.05]),
            DiscoverRecord::new(3, "neuron-1".to_string(), Some(0.65), 0.85, vec![0.18]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        let read_records = read_records_from_parquet(file_path, "neuron-1").unwrap();

        assert_eq!(read_records.len(), 5, "Should have 5 records");
        let obs_indices: Vec<u32> = read_records.iter().map(|r| r.obs_index).collect();
        assert!(obs_indices.contains(&1), "Should contain obs_index 1");
        assert!(obs_indices.contains(&2), "Should contain obs_index 2");
        assert!(obs_indices.contains(&3), "Should contain obs_index 3");
        assert!(obs_indices.contains(&5), "Should contain obs_index 5");
        assert!(obs_indices.contains(&8), "Should contain obs_index 8");
    }

    #[test]
    fn test_write_records_chunks_when_error_value_count_exceeds_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-0000".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "hidden-1111".to_string(), Some(0.6), 0.8, vec![0.3, 0.4]),
        ];

        let mut writer =
            ParquetRecordWriter::new(file_path, schema::MAX_ARROW_OFFSET, 3, records.len())
                .expect("Writer should construct successfully");
        assert!(
            writer.write_records(&records).is_ok(),
            "Writer should chunk records when total error values exceed the configured per-batch limit"
        );
        writer.finish().unwrap();

        assert!(
            std::path::Path::new(file_path).exists(),
            "Parquet file should exist after chunked write"
        );
    }

    #[test]
    fn test_write_records_errors_when_single_record_error_count_exceeds_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "hidden-0000".to_string(),
            Some(0.5),
            0.7,
            vec![0.1, 0.2],
        )];

        let mut writer =
            ParquetRecordWriter::new(file_path, schema::MAX_ARROW_OFFSET, 1, records.len())
                .expect("Writer should construct successfully");
        let err = writer
            .write_records(&records)
            .expect_err("Single record with more error values than the limit should be rejected");
        assert!(
            err.to_string().contains("error value(s)"),
            "Error message should reference the per-batch error limit"
        );
    }

    #[test]
    fn test_read_all_records_grouped_by_neuron() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.15, 0.25]),
            DiscoverRecord::new(0, "neuron-2".to_string(), Some(0.3), 0.5, vec![0.05]),
            DiscoverRecord::new(1, "neuron-2".to_string(), Some(0.4), 0.6, vec![0.1]),
            DiscoverRecord::new(2, "neuron-2".to_string(), Some(0.5), 0.7, vec![0.15]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        let grouped = read_all_records_grouped_by_neuron(file_path).unwrap();

        let neuron1_records = grouped.get("neuron-1").expect("neuron-1 should exist");
        assert_eq!(neuron1_records.len(), 2, "neuron-1 should have 2 records");
        assert_eq!(neuron1_records[0].obs_index, 0);
        assert_eq!(neuron1_records[1].obs_index, 1);

        let neuron2_records = grouped.get("neuron-2").expect("neuron-2 should exist");
        assert_eq!(neuron2_records.len(), 3, "neuron-2 should have 3 records");
        assert_eq!(neuron2_records[0].obs_index, 0);
        assert_eq!(neuron2_records[1].obs_index, 1);
        assert_eq!(neuron2_records[2].obs_index, 2);

        let neuron1_single = read_records_from_parquet(file_path, "neuron-1").unwrap();
        assert_eq!(
            neuron1_single.len(),
            neuron1_records.len(),
            "Single read should return same number of records as grouped read"
        );

        let neuron2_single = read_records_from_parquet(file_path, "neuron-2").unwrap();
        assert_eq!(
            neuron2_single.len(),
            neuron2_records.len(),
            "Single read should return same number of records as grouped read"
        );
    }
}
