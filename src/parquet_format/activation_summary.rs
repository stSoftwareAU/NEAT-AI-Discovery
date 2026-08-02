//! Per-neuron activation statistics from a single projected Parquet pass
//! (Issue #1923).
//!
//! The structural removal path needs one number per candidate — the neuron's
//! **mean absolute activation** — to resolve the activation-weighted gate that
//! [`RemovalCandidateJson`](crate::RemovalCandidateJson) documents. Decoding
//! whole [`DiscoverRecord`](crate::types::DiscoverRecord)s to obtain it is what
//! Issue #1766 removed from the focus path: the `errors` `ListArray` dominates
//! the file, and materialising every record cost a ~2h stall.
//!
//! This reader instead:
//! - projects **two** columns (`neuron_uuid`, `activation`) via
//!   [`ColumnProfile::ActivationOnly`], skipping `errors` I/O entirely;
//! - materialises **nothing** — each row folds straight into a running sum, so
//!   memory is `O(wanted neurons)` rather than `O(rows)`, and the decode budget
//!   that bounds record reads has nothing to bound; and
//! - honours the caller's deadline at every record-batch boundary.

// Activations are f32; the running mean is accumulated in f64 for precision and
// narrowed back on the way out. Both directions are intentional.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

use super::reader::{ColumnProfile, open_parquet_file, validate_parquet_schema};

/// Recorded activation statistics for one neuron.
///
/// Only **finite** activations contribute. `sample_count` is therefore the
/// number of usable samples behind `mean_abs_activation`, not the row count —
/// a neuron whose every recorded activation is NaN yields no summary at all
/// rather than a mean of `0.0` that would read as "measured and inactive"
/// (Issue #1923).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationSummary {
    /// Mean of `|activation|` across every finite recorded sample.
    pub mean_abs_activation: f32,
    /// Number of finite samples the mean was computed from (always `>= 1`).
    pub sample_count: usize,
}

/// Running accumulator — kept in `f64` so a long record stream does not lose
/// precision against `f32` activations.
#[derive(Default, Clone, Copy)]
struct Accumulator {
    sum_abs: f64,
    count: usize,
}

/// Aggregate the mean absolute activation of each `wanted` neuron in one
/// streaming pass over `file_path` (Issue #1923).
///
/// Neurons absent from the file — or whose every recorded activation is
/// non-finite — are **omitted from the result** rather than reported with a
/// zero mean. The caller must distinguish "measured as inactive" from "never
/// measured"; collapsing the two is the exact failure this issue fixes.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, does not carry the
/// discovery schema, or the `deadline` expires mid-decode. Nothing is returned
/// partially: a caller must never mistake a truncated scan for a complete one.
pub fn read_mean_abs_activation_by_neuron(
    file_path: &str,
    wanted: &HashSet<&str>,
    deadline: Option<SystemTime>,
) -> Result<HashMap<String, ActivationSummary>> {
    if wanted.is_empty() {
        return Ok(HashMap::new());
    }

    if let Some(dl) = deadline
        && SystemTime::now() >= dl
    {
        anyhow::bail!(
            "Activation summary aborted: deadline already passed before reading '{file_path}'"
        );
    }

    let file = open_parquet_file(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let profile = ColumnProfile::ActivationOnly;
    validate_parquet_schema(&builder, file_path, profile)?;

    let mask = ProjectionMask::roots(builder.parquet_schema(), profile.root_indices());
    let reader = builder
        .with_projection(mask)
        .build()
        .context("Failed to build Parquet reader")?;

    let mut accumulators: HashMap<&str, Accumulator> = HashMap::with_capacity(wanted.len());

    for batch_result in reader {
        if let Some(dl) = deadline
            && SystemTime::now() >= dl
        {
            anyhow::bail!(
                "Activation summary aborted: deadline expired while reading '{file_path}'"
            );
        }

        let batch = batch_result.context("Failed to read record batch")?;
        let schema = batch.schema();
        let uuid_idx = schema
            .index_of("neuron_uuid")
            .context("Missing neuron_uuid column")?;
        let act_idx = schema
            .index_of("activation")
            .context("Missing activation column")?;

        let neuron_uuid_col = batch
            .column(uuid_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let activation_col = batch
            .column(act_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;

        for i in 0..batch.num_rows() {
            let Some(uuid) = wanted.get(neuron_uuid_col.value(i)) else {
                continue;
            };
            let activation = activation_col.value(i);
            if !activation.is_finite() {
                continue;
            }
            let entry = accumulators.entry(*uuid).or_default();
            entry.sum_abs += f64::from(activation.abs());
            entry.count += 1;
        }
    }

    Ok(accumulators
        .into_iter()
        .filter_map(|(uuid, acc)| {
            if acc.count == 0 {
                return None;
            }
            let mean = acc.sum_abs / acc.count as f64;
            Some((
                uuid.to_string(),
                ActivationSummary {
                    mean_abs_activation: mean as f32,
                    sample_count: acc.count,
                },
            ))
        })
        .collect())
}
