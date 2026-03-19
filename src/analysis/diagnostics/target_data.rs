//! Target data structures for sample building during synapse analysis.
//!
//! Provides efficient pre-built target maps that allow matching source records
//! against target neuron data without rebuilding the map for each source.
//!
//! Includes single-source and group-based sample building for locality optimisation
//! (Issue #221).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use crate::analysis::samples::HelpfulSample;
use crate::types::DiscoverRecord;

// =============================================================================
// Target Data Structures for Sample Building
// =============================================================================

/// Target data for a single observation (used for matching with source records).
pub(crate) struct TargetData {
    pub(crate) avg_error: f32,
    pub(crate) value: Option<f32>,
    pub(crate) activation: f32,
}

/// Pre-built target map for efficient sample building across multiple sources.
/// This avoids rebuilding the `HashMap` for each source when analysing a single target.
pub(crate) struct TargetMap {
    pub(crate) map: HashMap<u32, TargetData>,
}

impl TargetMap {
    /// Build a target map from target records.
    /// This should be done ONCE per focus neuron, then reused for all sources.
    pub(crate) fn from_records(target_records: &[DiscoverRecord]) -> Self {
        let mut map: HashMap<u32, TargetData> = HashMap::with_capacity(target_records.len());
        for record in target_records {
            if record.errors.is_empty() || !record.activation.is_finite() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0;
            for error in &record.errors {
                if error.is_finite() {
                    sum += *error;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            map.insert(
                record.obs_index,
                TargetData {
                    avg_error: sum / count as f32,
                    value: record.value,
                    activation: record.activation,
                },
            );
        }
        Self { map }
    }

    /// Build samples by matching source records against this pre-built target map.
    /// This is much faster than `build_samples()` when processing multiple sources
    /// against the same target.
    pub(crate) fn build_samples_from(&self, from_records: &[DiscoverRecord]) -> Vec<HelpfulSample> {
        if self.map.is_empty() || from_records.is_empty() {
            return Vec::new();
        }

        let mut samples = Vec::with_capacity(from_records.len().min(self.map.len()));
        for record in from_records {
            if let Some(target) = self.map.get(&record.obs_index)
                && record.activation.is_finite()
                && target.avg_error.is_finite()
            {
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error: target.avg_error,
                    target_value: target.value,
                    target_activation: Some(target.activation),
                });
            }
        }

        samples
    }

    /// Build samples for multiple sources against this target map in a single pass.
    /// This is an optimisation for sources that share the same `obs_indices`.
    ///
    /// Instead of calling `build_samples_from` N times for N sources with identical
    /// `obs_indices`, we iterate through the target data once and build all sample
    /// vectors simultaneously.
    ///
    /// Returns a Vec of sample vectors, one per source, in the same order as
    /// `source_records`.
    ///
    /// # Issue #221: Sample Locality Optimisation
    ///
    /// When multiple sources share the same `obs_indices` (common for input neurons
    /// recorded together), this method reduces redundant target map lookups from
    /// O(sources × samples) to O(samples).
    pub(crate) fn build_samples_for_group(
        &self,
        source_records: &[(&str, &[DiscoverRecord])],
    ) -> Vec<Vec<HelpfulSample>> {
        if self.map.is_empty() || source_records.is_empty() {
            return vec![Vec::new(); source_records.len()];
        }

        // Build activation maps for each source: obs_index -> activation
        let activation_maps: Vec<HashMap<u32, f32>> = source_records
            .iter()
            .map(|(_, records)| {
                let mut map = HashMap::with_capacity(records.len());
                for record in *records {
                    if record.activation.is_finite() {
                        map.insert(record.obs_index, record.activation);
                    }
                }
                map
            })
            .collect();

        // Pre-allocate result vectors
        let mut results: Vec<Vec<HelpfulSample>> = source_records
            .iter()
            .map(|(_, records)| Vec::with_capacity(records.len().min(self.map.len())))
            .collect();

        // Single pass through target data, building samples for all sources
        for (&obs_index, target) in &self.map {
            if !target.avg_error.is_finite() {
                continue;
            }

            for (source_idx, activation_map) in activation_maps.iter().enumerate() {
                if let Some(&activation) = activation_map.get(&obs_index) {
                    results[source_idx].push(HelpfulSample {
                        activation,
                        avg_error: target.avg_error,
                        target_value: target.value,
                        target_activation: Some(target.activation),
                    });
                }
            }
        }

        results
    }
}
