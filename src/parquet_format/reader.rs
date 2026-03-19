//! Parquet reading and deserialisation for discovery records

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;

use crate::types::DiscoverRecord;
use std::time::SystemTime;

/// Read all discovery records from a Parquet file, grouped by neuron UUID.
/// This is more efficient than calling `read_records_from_parquet` multiple times
/// when you need records for multiple neurons.
pub fn read_all_records_grouped_by_neuron(
    file_path: &str,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    read_all_records_grouped_by_neuron_with_deadline(file_path, None)
}

/// Read all discovery records from a Parquet file, grouped by neuron UUID,
/// with optional deadline checking and watchdog beats (Issue #648).
///
/// When a deadline is provided, the reader checks at each batch boundary
/// whether the deadline has passed. If so, it aborts early with a clear
/// error message. Watchdog beats are emitted every batch to prevent
/// the watchdog from triggering during long loads.
pub fn read_all_records_grouped_by_neuron_with_deadline(
    file_path: &str,
    deadline: Option<SystemTime>,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    // Check deadline before starting
    if let Some(dl) = deadline
        && SystemTime::now() >= dl
    {
        anyhow::bail!("Parquet loading aborted: deadline already passed before loading started");
    }

    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut grouped_records: HashMap<String, Vec<DiscoverRecord>> = HashMap::new();

    for (batch_count, batch_result) in reader.enumerate() {
        // Check deadline at each batch boundary (Issue #648)
        if let Some(dl) = deadline
            && SystemTime::now() >= dl
        {
            let neurons_loaded = grouped_records.len();
            tracing::warn!(
                batch_count,
                neurons_loaded,
                "Parquet loading aborted: deadline reached after {batch_count} batches \
                 ({neurons_loaded} neurons loaded)"
            );
            anyhow::bail!(
                "Parquet loading aborted: deadline reached after {batch_count} batches \
                 ({neurons_loaded} neurons loaded)"
            );
        }

        // Beat the watchdog periodically during loading (Issue #648)
        if batch_count.is_multiple_of(10) {
            crate::watchdog::beat(format!("parquet loading: batch {batch_count}"));
        }

        let batch = batch_result.context("Failed to read record batch")?;

        // Get columns - use schema field names to find correct columns instead of hardcoded indices
        let obs_index_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;
        let errors_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("Failed to cast errors column")?;

        // Collect all records and group by neuron UUID
        for i in 0..batch.num_rows() {
            let uuid = neuron_uuid_col.value(i).to_string();
            let obs_index = obs_index_col.value(i);
            let value = if value_col.is_null(i) {
                None
            } else {
                Some(value_col.value(i))
            };
            let activation = activation_col.value(i);

            // Extract errors array
            let errors_list = errors_col.value(i);
            let errors_array = errors_list
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast errors array")?;
            let errors: Vec<f32> = (0..errors_array.len())
                .map(|j| errors_array.value(j))
                .collect();

            let record = DiscoverRecord::new(obs_index, uuid.clone(), value, activation, errors);
            grouped_records.entry(uuid).or_default().push(record);
        }
    }

    // Note: Records are returned in Parquet read order (not sorted by obs_index)
    // TypeScript sorts records by obs_index after reading for cross-neuron matching
    Ok(grouped_records)
}

/// Read discovery records from a Parquet file, filtered by neuron UUID
pub fn read_records_from_parquet(
    file_path: &str,
    neuron_uuid: &str,
) -> Result<Vec<DiscoverRecord>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut records = Vec::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        // Get columns - use schema field names to find correct columns instead of hardcoded indices
        let obs_index_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;
        let errors_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("Failed to cast errors column")?;

        // Filter by neuron UUID and collect records
        for i in 0..batch.num_rows() {
            let uuid = neuron_uuid_col.value(i);
            if uuid == neuron_uuid {
                let obs_index = obs_index_col.value(i);
                let value = if value_col.is_null(i) {
                    None
                } else {
                    Some(value_col.value(i))
                };
                let activation = activation_col.value(i);

                // Extract errors array
                let errors_list = errors_col.value(i);
                let errors_array = errors_list
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Failed to cast errors array")?;
                let errors: Vec<f32> = (0..errors_array.len())
                    .map(|j| errors_array.value(j))
                    .collect();

                records.push(DiscoverRecord::new(
                    obs_index,
                    uuid.to_string(),
                    value,
                    activation,
                    errors,
                ));
            }
        }
    }

    // Note: Records are returned in Parquet read order (not sorted by obs_index)
    // TypeScript sorts records by obs_index after reading for cross-neuron matching
    Ok(records)
}

/// Read ALL discovery records from a Parquet file (no filtering).
///
/// Used by the visualisation snapshot export to get all recorded data.
pub fn read_all_records_from_parquet(file_path: &str) -> Result<Vec<DiscoverRecord>> {
    read_records_from_parquet_with_limit(file_path, None)
}

/// Read discovery records from a Parquet file with optional observation limit.
///
/// If `max_obs` is Some, returns records for the first `max_obs` distinct
/// `obs_index` values encountered while scanning the file.
///
/// Important: Parquet rows are not guaranteed to be stored in observation-first
/// order. They may be stored in neuron-first order (all obs for neuron A, then
/// all obs for neuron B). In that case, **we must not stop reading** as soon as
/// we see a new `obs_index` beyond the limit, because later rows may still
/// contain records for already-accepted observation indices.
///
/// To guarantee complete data for the accepted observations, this reader:
/// - Keeps a set of accepted `obs_index` values up to `max_obs`
/// - Scans the file and **filters** to only accepted observations
pub fn read_records_from_parquet_with_limit(
    file_path: &str,
    max_obs: Option<u32>,
) -> Result<Vec<DiscoverRecord>> {
    // Edge case: treat max_obs=0 as "return no rows".
    if matches!(max_obs, Some(0)) {
        return Ok(Vec::new());
    }

    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut records = Vec::new();
    let mut accepted_obs_indices: HashSet<u32> = HashSet::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        let obs_index_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;
        let errors_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("Failed to cast errors column")?;

        for i in 0..batch.num_rows() {
            let obs_index = obs_index_col.value(i);

            // Determine whether to include this row under the limit.
            let include_row = match max_obs {
                None => true,
                Some(max) => {
                    if accepted_obs_indices.contains(&obs_index) {
                        true
                    } else if accepted_obs_indices.len() < max as usize {
                        accepted_obs_indices.insert(obs_index);
                        true
                    } else {
                        // We've accepted enough observations. Skip new obs_index values,
                        // but keep scanning in case we encounter more rows for accepted
                        // observations (e.g. neuron-first ordering).
                        false
                    }
                }
            };

            if !include_row {
                continue;
            }

            let uuid = neuron_uuid_col.value(i);
            let value = if value_col.is_null(i) {
                None
            } else {
                Some(value_col.value(i))
            };
            let activation = activation_col.value(i);

            // Extract errors array
            let errors_list = errors_col.value(i);
            let errors_array = errors_list
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast errors array")?;
            let errors: Vec<f32> = (0..errors_array.len())
                .map(|j| errors_array.value(j))
                .collect();

            records.push(DiscoverRecord::new(
                obs_index,
                uuid.to_string(),
                value,
                activation,
                errors,
            ));
        }
    }

    Ok(records)
}
