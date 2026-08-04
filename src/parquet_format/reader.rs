//! Parquet reading and deserialisation for discovery records

use anyhow::{Context, Result};
use arrow::datatypes::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;

use crate::DiscoveryError;
use crate::parquet_format::batch_columns::DiscoveryBatchColumns;
use crate::parquet_format::decode_budget::DecodeBudget;
use crate::parquet_format::schema::create_schema;
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
    /// `neuron_uuid` + `activation` only (Issue #1923).
    ///
    /// The narrowest profile in the crate: enough to aggregate per-neuron
    /// activation statistics without materialising a single `DiscoverRecord`.
    /// It cannot be used with the record readers — only with
    /// [`crate::parquet_format::read_mean_abs_activation_by_neuron`].
    ActivationOnly,
}

impl ColumnProfile {
    /// Root column indices for the discovery Parquet schema.
    pub(crate) fn root_indices(self) -> Vec<usize> {
        match self {
            Self::Full => vec![0, 1, 2, 3, 4],
            Self::WithoutErrors => vec![0, 1, 2, 3],
            Self::ActivationOnly => vec![1, 3],
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
pub(crate) fn open_parquet_file(file_path: &str) -> Result<File> {
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

/// Compare a file's column type against the discovery schema's (Issue #1901).
///
/// List columns match on element type and element nullability only: the element
/// field's *name* varies between Parquet producers (`item` vs `element`) and has
/// no bearing on how the column decodes.
fn data_type_matches(actual: &DataType, expected: &DataType) -> bool {
    match (actual, expected) {
        (DataType::List(actual_item), DataType::List(expected_item)) => {
            actual_item.data_type() == expected_item.data_type()
                && actual_item.is_nullable() == expected_item.is_nullable()
        }
        _ => actual == expected,
    }
}

/// Validate that the Parquet file's schema matches the expected discovery schema
/// (Issue #1085). Checks that required columns exist and that the file has the
/// expected number of root columns for the requested projection profile.
///
/// Each column's Arrow [`DataType`] and nullability are also checked against
/// [`create_schema`] (Issue #1901). Arrow's `value()` accessors return the
/// physical buffer contents for a null slot rather than an error, so a nullable
/// `neuron_uuid` would decode as `""` and a nullable `activation` as `0.0` —
/// silently forming a bogus UUID group and depressing MSE/MAE scores. Only the
/// columns the writer declares nullable (`value`) may carry nulls; every other
/// column makes the whole file invalid rather than partially usable.
///
/// This is a metadata-only check with no performance impact on valid files.
pub(crate) fn validate_parquet_schema(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    file_path: &str,
    profile: ColumnProfile,
) -> Result<()> {
    let parquet_schema = builder.parquet_schema();
    let num_columns = parquet_schema.num_columns();

    // The discovery schema has 5 root columns (errors has a nested child, so
    // the Parquet schema may report 6 columns). Check that we have at least
    // the minimum required columns for the requested profile — the highest root
    // index it projects must exist (Issue #1923: profiles are no longer
    // guaranteed to be a leading prefix of the schema).
    let root_indices = profile.root_indices();
    let max_index = root_indices.iter().copied().max().unwrap_or(0);
    let required_count = max_index + 1;

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

    // Check that column names match expected names. The expected name for a
    // projected root index is simply the discovery schema's name at that index,
    // so this holds for any subset — prefix or not (Issue #1923).
    let expected_schema = create_schema();

    for idx in &root_indices {
        let expected_name = &EXPECTED_COLUMNS[*idx];
        let actual_field = arrow_schema.field(*idx);
        let actual_name = actual_field.name();
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

        // Names alone are not a schema (Issue #1901) — check type and nullability.
        let expected_field = expected_schema.field(*idx);
        if !data_type_matches(actual_field.data_type(), expected_field.data_type()) {
            let expected_type = expected_field.data_type();
            let actual_type = actual_field.data_type();
            return Err(DiscoveryError::Io {
                detail: format!(
                    "Parquet file '{file_path}' schema mismatch: column '{expected_name}' at index \
                     {idx} has type {actual_type:?} but the discovery schema requires \
                     {expected_type:?}."
                ),
            }
            .into());
        }

        if actual_field.is_nullable() && !expected_field.is_nullable() {
            return Err(DiscoveryError::Io {
                detail: format!(
                    "Parquet file '{file_path}' schema mismatch: column '{expected_name}' at index \
                     {idx} is nullable but the discovery schema requires it to be non-null. \
                     A null '{expected_name}' would silently decode as a default value, so the \
                     file is rejected rather than read."
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

        // Issue #2005: one shared decoder owns column resolution and per-row
        // decoding; this reader keeps only its grouping and budget policy.
        let columns = DiscoveryBatchColumns::resolve(&batch, include_errors)?;

        // Collect all records and group by neuron UUID
        for i in 0..columns.num_rows() {
            let record = columns.decode_row(i)?;

            budget.charge_record(record.neuron_uuid.len(), record.errors.len(), file_path)?;

            // Issue #2007: only a neuron's *first* row allocates a map key. The
            // `entry` API would clone the UUID on every row, and a recording
            // holds thousands of rows per neuron.
            if let Some(existing) = grouped_records.get_mut(&record.neuron_uuid) {
                existing.push(record);
            } else {
                grouped_records.insert(record.neuron_uuid.clone(), vec![record]);
            }
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

        // Issue #2005: shared column resolution and per-row decoding; this
        // reader keeps only its UUID filter and budget policy.
        let columns = DiscoveryBatchColumns::resolve(&batch, include_errors)?;

        // Filter by neuron UUID and collect records
        for i in 0..columns.num_rows() {
            if columns.neuron_uuid(i) != neuron_uuid {
                continue;
            }

            let record = columns.decode_row(i)?;
            budget.charge_record(record.neuron_uuid.len(), record.errors.len(), file_path)?;
            records.push(record);
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

        // Issue #2005: shared column resolution and per-row decoding; this
        // reader keeps only its observation limit and budget policy.
        let columns = DiscoveryBatchColumns::resolve(&batch, include_errors)?;

        for i in 0..columns.num_rows() {
            let obs_index = columns.obs_index(i);

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

            let record = columns.decode_row(i)?;
            budget.charge_record(record.neuron_uuid.len(), record.errors.len(), file_path)?;
            records.push(record);
        }
    }

    Ok(records)
}
