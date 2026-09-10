//! The preconditions discovery recording rejects, checked before any Parquet
//! file is opened.
//!
//! Recording fails when the creature has no non-input neuron (input neurons
//! are skipped, so such a creature yields nothing to record), when the
//! training data is empty, or when the derived records-per-sample is zero —
//! each reported as a distinct error rather than as an empty output file.
//! Observation indices are resolved here too: caller-supplied
//! `record_indices` must match the training-data length, be free of
//! duplicates, and fit in `u32`; otherwise sequential indices are generated.

use anyhow::Result;

use super::sizing::records_per_sample;
use crate::RecordDiscoveryInput;

/// Validate that the creature has non-input neurons and resolve observation indices.
///
/// Returns `(non_input_neuron_count, obs_indices)` on success.
pub fn validate_and_resolve_indices(input: &RecordDiscoveryInput) -> Result<(usize, Vec<u32>)> {
    let non_input_neuron_count = input
        .creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type != "input")
        .count();

    if non_input_neuron_count == 0 {
        return Err(anyhow::anyhow!(
            "Cannot record discovery data: creature has no non-input neurons. Input neurons are skipped during discovery recording. Discovery recording requires at least one hidden or output neuron to record activations and errors."
        ));
    }

    let obs_indices = resolve_observation_indices(input)?;

    let records_per_sample = records_per_sample(non_input_neuron_count, input.creature.input)?;
    if input.training_data.is_empty() || records_per_sample == 0 {
        return Err(anyhow::anyhow!(
            "No discovery records were generated from the training data"
        ));
    }

    Ok((non_input_neuron_count, obs_indices))
}

/// Resolve observation indices from the input.
///
/// If `record_indices` is provided, validates and converts them.
/// Otherwise, generates sequential indices from `0..training_data.len()`.
fn resolve_observation_indices(input: &RecordDiscoveryInput) -> Result<Vec<u32>> {
    if let Some(ref record_indices) = input.record_indices {
        if record_indices.len() != input.training_data.len() {
            return Err(anyhow::anyhow!(
                "record_indices length ({}) must match training_data length ({}) when provided",
                record_indices.len(),
                input.training_data.len()
            ));
        }

        let mut seen = std::collections::HashSet::new();
        let mut indices = Vec::with_capacity(record_indices.len());
        for &idx in record_indices {
            if !seen.insert(idx) {
                return Err(anyhow::anyhow!(
                    "Record index {idx} is duplicated in record_indices. Discovery recording requires unique indices."
                ));
            }
            let obs_index_u32 = u32::try_from(idx).map_err(|_| {
                anyhow::anyhow!(
                    "Record index {idx} exceeds maximum supported size ({max})",
                    max = u32::MAX
                )
            })?;
            indices.push(obs_index_u32);
        }
        Ok(indices)
    } else {
        if input.training_data.len() > u32::MAX as usize {
            return Err(anyhow::anyhow!(
                "Training data has {} records which exceeds maximum supported size ({})",
                input.training_data.len(),
                u32::MAX
            ));
        }

        (0..input.training_data.len())
            .map(|idx| {
                u32::try_from(idx).map_err(|_| {
                    anyhow::anyhow!("Observation index {} exceeds u32::MAX ({})", idx, u32::MAX)
                })
            })
            .collect::<Result<Vec<u32>>>()
    }
}
