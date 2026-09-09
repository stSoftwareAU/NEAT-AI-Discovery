//! Record discovery data logic.
//!
//! Split into focused sub-modules (Issue #604, #942):
//! - `validation` — input validation and observation index resolution
//! - `processing` — record building from training data and Parquet writing
//! - `tests` — unit tests for record module

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod processing;
mod sizing;
#[cfg(test)]
mod tests;
mod validation;

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::RecordDiscoveryInput;
use crate::parquet_format::ParquetRecordWriter;

pub use sizing::records_per_sample;

/// Result of recording discovery data
#[derive(Debug)]
pub struct RecordResult {
    pub temp_dir: String,
    pub file: String,
}

/// Record discovery data from training records
///
/// This function processes training data and records neuron activations and errors.
/// For each training record, it collects all neuron data atomically and writes to Parquet.
/// Since the training dataset is already randomized, parallelization is allowed,
/// but each training record must be processed atomically (all neurons written together).
pub fn record_discovery_data(input: &RecordDiscoveryInput) -> Result<RecordResult> {
    // Create temp directory
    let temp_dir = Path::new(&input.temp_dir);
    let temp_dir_str = input.temp_dir.clone();
    fs::create_dir_all(temp_dir)
        .with_context(|| format!("Failed to create temp directory: {temp_dir_str}"))?;

    let (non_input_neuron_count, obs_indices) = validation::validate_and_resolve_indices(input)?;

    let records_per_sample = records_per_sample(non_input_neuron_count, input.creature.input)?;

    let estimated_total_records = input
        .training_data
        .len()
        .checked_mul(records_per_sample)
        .ok_or_else(|| anyhow::anyhow!("Discovery record count would overflow usize"))?;

    let parquet_file = temp_dir.join("discovery_data.parquet");
    let parquet_path = parquet_file
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?;

    let max_arrow_offset = i32::MAX as usize;
    let mut writer = ParquetRecordWriter::new(
        parquet_path,
        max_arrow_offset,
        max_arrow_offset,
        estimated_total_records,
    )
    .context("Failed to initialise Parquet writer")?;

    processing::process_training_data(input, &obs_indices, non_input_neuron_count, &mut writer)?;

    writer
        .finish()
        .context("Failed to finalise Parquet writer")?;

    Ok(RecordResult {
        temp_dir: input.temp_dir.clone(),
        file: "discovery_data.parquet".to_string(),
    })
}
