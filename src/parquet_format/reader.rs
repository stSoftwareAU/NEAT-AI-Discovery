//! Parquet reading and deserialisation for discovery records

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;

use crate::DiscoveryError;
use crate::parquet_format::decode_budget::DecodeBudget;
use crate::types::DiscoverRecord;
use std::time::SystemTime;

/// Column projection profiles for Parquet reads (Issue #1073).
///
/// Different analysis paths need different subsets of columns.
/// Skipping unused columns (particularly the `errors` `ListArray`)
/// reduces I/O and deserialisation overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnProfile {
    /// All columns: `obs_index`, `neuron_uuid`, `value`, `activation`, `errors`.
    Full,
    /// Core analysis columns without errors: `obs_index`, `neuron_uuid`, `value`, `activation`.
    /// Records returned with this profile have an empty `errors` vec.
    WithoutErrors,
}

impl ColumnProfile {
    /// Root column indices for the discovery Parquet schema.
    fn root_indices(self) -> Vec<usize> {
        match self {
            Self::Full => vec![0, 1, 2, 3, 4],
            Self::WithoutErrors => vec![0, 1, 2, 3],
        }
    }

    /// Whether this profile includes the errors column.
    fn includes_errors(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Open a parquet file, returning a clear "file removed" error if the file
/// no longer exists on disk (Issue #1049).
///
/// Also rejects `.parquet.tmp` files which indicate incomplete writes from
/// the streaming writer (Issue #1085).
///
/// This distinguishes external deletion (e.g., host cleaned up temp directory)
/// from other I/O errors such as corruption or permission issues, allowing
/// callers to return partial results instead of crashing.
fn open_parquet_file(file_path: &str) -> Result<File> {
    // Reject .parquet.tmp files — these are incomplete writes (Issue #1085)
    if file_path.ends_with(".parquet.tmp") {
        tracing::warn!(
            file_path,
            "Skipping incomplete Parquet file (still has .tmp suffix). \
             This file was likely left behind by an interrupted recording session."
        );
        return Err(DiscoveryError::Io {
            detail: format!(
                "Parquet file '{file_path}' has a .tmp suffix indicating an incomplete write. \
                 It may have been left behind by an interrupted recording session."
            ),
        }
        .into());
    }

    File::open(file_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            anyhow::anyhow!(
                "Parquet file removed: the file '{file_path}' no longer exists on disk. \
                 It may have been deleted by the host while analysis was still active."
            )
        } else {
            anyhow::anyhow!("Failed to open Parquet file: {file_path}").context(e)
        }
    })
}

/// Expected column names for the discovery Parquet schema.
const EXPECTED_COLUMNS: &[&str] = &["obs_index", "neuron_uuid", "value", "activation", "errors"];

/// Validate that the Parquet file's schema matches the expected discovery schema
/// (Issue #1085). Checks that required columns exist and that the file has the
/// expected number of root columns for the requested projection profile.
///
/// This is a metadata-only check with no performance impact on valid files.
fn validate_parquet_schema(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    file_path: &str,
    profile: ColumnProfile,
) -> Result<()> {
    let parquet_schema = builder.parquet_schema();
    let num_columns = parquet_schema.num_columns();

    // The discovery schema has 5 root columns (errors has a nested child, so
    // the Parquet schema may report 6 columns). Check that we have at least
    // the minimum required columns for the requested profile.
    let required_count = match profile {
        ColumnProfile::Full => 5,
        ColumnProfile::WithoutErrors => 4,
    };

    if num_columns < required_count {
        return Err(DiscoveryError::Io {
            detail: format!(
                "Parquet file '{file_path}' schema mismatch: expected at least \
                 {required_count} columns but found {num_columns}. \
                 The file may not be a valid discovery Parquet file."
            ),
        }
        .into());
    }

    // Verify that the root column indices we need exist in the schema
    let root_indices = profile.root_indices();
    let max_index = root_indices.iter().copied().max().unwrap_or(0);

    // Use the Arrow schema from the builder to check column names
    let arrow_schema = builder.schema();
    let field_count = arrow_schema.fields().len();

    if max_index >= field_count {
        return Err(DiscoveryError::Io {
            detail: format!(
                "Parquet file '{file_path}' schema mismatch: expected column index \
                 {max_index} but file only has {field_count} columns. \
                 The file may not be a valid discovery Parquet file."
            ),
        }
        .into());
    }

    // Check that column names match expected names
    let columns_to_check = match profile {
        ColumnProfile::Full => EXPECTED_COLUMNS,
        ColumnProfile::WithoutErrors => &EXPECTED_COLUMNS[..4],
    };

    for (idx, expected_name) in root_indices.iter().zip(columns_to_check.iter()) {
        let actual_name = arrow_schema.field(*idx).name();
        if actual_name != *expected_name {
            return Err(DiscoveryError::Io {
                detail: format!(
                    "Parquet file '{file_path}' schema mismatch: expected column '{expected_name}' \
                     at index {idx} but found '{actual_name}'. \
                     The file may not be a valid discovery Parquet file."
                ),
            }
            .into());
        }
    }

    Ok(())
}

/// Read all discovery records from a Parquet file, grouped by neuron UUID.
/// This is more efficient than calling `read_records_from_parquet` multiple times
/// when you need records for multiple neurons.
pub fn read_all_records_grouped_by_neuron(
    file_path: &str,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    read_all_records_grouped_by_neuron_with_deadline(file_path, None)
}

/// Read all records grouped by neuron UUID, using a column projection profile
/// to skip unnecessary columns (Issue #1073).
pub fn read_all_records_grouped_by_neuron_with_profile(
    file_path: &str,
    profile: ColumnProfile,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    read_all_records_grouped_by_neuron_with_deadline_and_profile(file_path, None, profile)
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
    read_all_records_grouped_by_neuron_with_deadline_and_profile(
        file_path,
        deadline,
        ColumnProfile::Full,
    )
}

/// Read all discovery records grouped by neuron UUID with deadline checking
/// and column projection (Issue #1073).
///
/// When `profile` is `WithoutErrors`, the `errors` `ListArray` column is
/// skipped entirely during I/O and deserialisation. Records returned with
/// this profile have an empty `errors` vec.
pub fn read_all_records_grouped_by_neuron_with_deadline_and_profile(
    file_path: &str,
    deadline: Option<SystemTime>,
    profile: ColumnProfile,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    read_all_records_grouped_by_neuron_bounded(file_path, deadline, profile, None)
}

/// Read all discovery records grouped by neuron UUID, bounding the decode with
/// a cumulative byte budget (Issue #1869).
///
/// `budget_mb` is the caller's memory budget (the analysis phase forwards its
/// `max_analysis_memory_mb`). When it is `None` the ceiling falls back to
/// `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB`, then to half of total system RAM.
/// Every materialised record is charged inside the batch loop, so a file whose
/// decoded size far exceeds its compressed size aborts with a typed
/// memory-exhaustion error part-way through rather than after the allocation
/// has already been made.
pub fn read_all_records_grouped_by_neuron_bounded(
    file_path: &str,
    deadline: Option<SystemTime>,
    profile: ColumnProfile,
    budget_mb: Option<u64>,
) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
    // Check deadline before starting
    if let Some(dl) = deadline
        && SystemTime::now() >= dl
    {
        anyhow::bail!("Parquet loading aborted: deadline already passed before loading started");
    }

    let file = open_parquet_file(file_path)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    validate_parquet_schema(&builder, file_path, profile)?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), profile.root_indices());
    let reader = builder
        .with_projection(mask)
        .build()
        .context("Failed to build Parquet reader")?;

    let include_errors = profile.includes_errors();
    let mut grouped_records: HashMap<String, Vec<DiscoverRecord>> = HashMap::new();
    // Issue #1869: the bound that actually holds — charged per record below.
    let mut budget = DecodeBudget::resolve(budget_mb);

    for (batch_count, batch_result) in reader.enumerate() {
        // Issue #1047: Check cancellation flag at each batch boundary so
        // parquet loading stops promptly when the host sends SIGTERM.
        if crate::cancellation::is_cancelled() {
            let neurons_loaded = grouped_records.len();
            tracing::info!(
                batch_count,
                neurons_loaded,
                "Parquet loading cancelled by host after {batch_count} batches"
            );
            anyhow::bail!("Analysis cancelled by host");
        }

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

        // Column indices shift when projection is applied — use schema field names
        let schema = batch.schema();
        let obs_idx = schema
            .index_of("obs_index")
            .context("Missing obs_index column")?;
        let uuid_idx = schema
            .index_of("neuron_uuid")
            .context("Missing neuron_uuid column")?;
        let value_idx = schema.index_of("value").context("Missing value column")?;
        let act_idx = schema
            .index_of("activation")
            .context("Missing activation column")?;

        let obs_index_col = batch
            .column(obs_idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(uuid_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(value_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(act_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;

        let errors_col = if include_errors {
            let err_idx = schema.index_of("errors").context("Missing errors column")?;
            Some(
                batch
                    .column(err_idx)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("Failed to cast errors column")?,
            )
        } else {
            None
        };

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

            let errors = if let Some(errors_col) = errors_col {
                let errors_list = errors_col.value(i);
                let errors_array = errors_list
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Failed to cast errors array")?;
                (0..errors_array.len())
                    .map(|j| errors_array.value(j))
                    .collect()
            } else {
                Vec::new()
            };

            budget.charge_record(uuid.len(), errors.len(), file_path)?;

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
    read_records_from_parquet_with_profile(file_path, neuron_uuid, ColumnProfile::Full)
}

/// Read discovery records filtered by neuron UUID, using a column projection
/// profile to skip unnecessary columns (Issue #1073).
pub fn read_records_from_parquet_with_profile(
    file_path: &str,
    neuron_uuid: &str,
    profile: ColumnProfile,
) -> Result<Vec<DiscoverRecord>> {
    let file = open_parquet_file(file_path)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    validate_parquet_schema(&builder, file_path, profile)?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), profile.root_indices());
    let reader = builder
        .with_projection(mask)
        .build()
        .context("Failed to build Parquet reader")?;

    let include_errors = profile.includes_errors();
    let mut records = Vec::new();
    // Issue #1869: bound the retained rows even though most are filtered out.
    let mut budget = DecodeBudget::resolve(None);

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        let schema = batch.schema();
        let obs_idx = schema
            .index_of("obs_index")
            .context("Missing obs_index column")?;
        let uuid_idx = schema
            .index_of("neuron_uuid")
            .context("Missing neuron_uuid column")?;
        let value_idx = schema.index_of("value").context("Missing value column")?;
        let act_idx = schema
            .index_of("activation")
            .context("Missing activation column")?;

        let obs_index_col = batch
            .column(obs_idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(uuid_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(value_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(act_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;

        let errors_col = if include_errors {
            let err_idx = schema.index_of("errors").context("Missing errors column")?;
            Some(
                batch
                    .column(err_idx)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("Failed to cast errors column")?,
            )
        } else {
            None
        };

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

                let errors = if let Some(errors_col) = errors_col {
                    let errors_list = errors_col.value(i);
                    let errors_array = errors_list
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .context("Failed to cast errors array")?;
                    (0..errors_array.len())
                        .map(|j| errors_array.value(j))
                        .collect()
                } else {
                    Vec::new()
                };

                budget.charge_record(uuid.len(), errors.len(), file_path)?;

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
    read_records_from_parquet_with_limit_and_profile(file_path, max_obs, ColumnProfile::Full)
}

/// Read discovery records with optional observation limit and column projection
/// (Issue #1073).
pub fn read_records_from_parquet_with_limit_and_profile(
    file_path: &str,
    max_obs: Option<u32>,
    profile: ColumnProfile,
) -> Result<Vec<DiscoverRecord>> {
    read_records_from_parquet_with_limit_and_budget(file_path, max_obs, profile, None)
}

/// Read discovery records with an observation limit, column projection and a
/// cumulative decode budget (Issue #1869).
///
/// `max_obs` caps distinct **observations**, not rows: a file whose rows all
/// share one `obs_index` is unbounded even at `max_obs = 1`. The decode budget
/// is the bound that holds regardless of how the rows are distributed.
pub fn read_records_from_parquet_with_limit_and_budget(
    file_path: &str,
    max_obs: Option<u32>,
    profile: ColumnProfile,
    budget_mb: Option<u64>,
) -> Result<Vec<DiscoverRecord>> {
    // Edge case: treat max_obs=0 as "return no rows".
    if matches!(max_obs, Some(0)) {
        return Ok(Vec::new());
    }

    let file = open_parquet_file(file_path)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    validate_parquet_schema(&builder, file_path, profile)?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), profile.root_indices());
    let reader = builder
        .with_projection(mask)
        .build()
        .context("Failed to build Parquet reader")?;

    let include_errors = profile.includes_errors();
    let mut records = Vec::new();
    let mut accepted_obs_indices: HashSet<u32> = HashSet::new();
    // Issue #1869: max_obs bounds observations, not rows — this bounds bytes.
    let mut budget = DecodeBudget::resolve(budget_mb);

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        let schema = batch.schema();
        let obs_col_idx = schema
            .index_of("obs_index")
            .context("Missing obs_index column")?;
        let uuid_col_idx = schema
            .index_of("neuron_uuid")
            .context("Missing neuron_uuid column")?;
        let value_col_idx = schema.index_of("value").context("Missing value column")?;
        let act_col_idx = schema
            .index_of("activation")
            .context("Missing activation column")?;

        let obs_index_col = batch
            .column(obs_col_idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(uuid_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(value_col_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(act_col_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;

        let errors_col = if include_errors {
            let err_idx = schema.index_of("errors").context("Missing errors column")?;
            Some(
                batch
                    .column(err_idx)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("Failed to cast errors column")?,
            )
        } else {
            None
        };

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

            let errors = if let Some(errors_col) = errors_col {
                let errors_list = errors_col.value(i);
                let errors_array = errors_list
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Failed to cast errors array")?;
                (0..errors_array.len())
                    .map(|j| errors_array.value(j))
                    .collect()
            } else {
                Vec::new()
            };

            budget.charge_record(uuid.len(), errors.len(), file_path)?;

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
