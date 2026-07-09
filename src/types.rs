//! Type definitions for discovery records

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Represents a single discovery record for a neuron
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscoverRecord {
    /// Observation index (training record index)
    pub obs_index: u32,
    /// Stable neuron identity string (RFC 4122 UUID, `input-N`, or other
    /// descriptive identifier). Numeric integer IDs are not permitted (Issue #952).
    pub neuron_uuid: String,
    /// Neuron value (optional, can be None)
    pub value: Option<f32>,
    /// Neuron activation
    pub activation: f32,
    /// Array of error values
    pub errors: Vec<f32>,
}

impl DiscoverRecord {
    /// Create a new discovery record
    pub fn new(
        obs_index: u32,
        neuron_uuid: String,
        value: Option<f32>,
        activation: f32,
        errors: Vec<f32>,
    ) -> Self {
        Self {
            obs_index,
            neuron_uuid,
            value,
            activation,
            errors,
        }
    }
}

/// An `Arc`-shared view of one neuron's discovery records (Issue #1543).
///
/// The `RecordCache` already stores each neuron's records behind an
/// `Arc<Vec<DiscoverRecord>>`. Historically the bulk `load_records_for_*`
/// loaders deep-cloned the inner `Vec` for every one of the ~48 discovery
/// modules dispatched per `analyze_all` pass, materialising tens of GB of
/// transient copies on production-scale creatures. `SharedRecords` lets the
/// loaders hand out a cheap `Arc::clone` of the cache's existing allocation
/// instead.
///
/// Detection and recommendation modules accept `&[(String, impl
/// AsRef<[DiscoverRecord]>)]`, so both the shared production path
/// (`SharedRecords`) and owned test fixtures (`Vec<DiscoverRecord>`) satisfy the
/// same signature without any call-site churn. `Deref` and `AsRef` expose the
/// underlying slice so the wrapper is transparent at use sites.
#[derive(Debug, Clone)]
pub struct SharedRecords(Arc<Vec<DiscoverRecord>>);

impl SharedRecords {
    /// Wrap an `Arc`-shared record vector without copying the records.
    #[must_use]
    pub fn new(records: Arc<Vec<DiscoverRecord>>) -> Self {
        Self(records)
    }

    /// Borrow the shared allocation so callers can assert allocation identity
    /// (e.g. `Arc::ptr_eq` against the cache entry — the Issue #1543 regression
    /// guard that a future revert to deep-cloning would break).
    #[must_use]
    pub fn arc(&self) -> &Arc<Vec<DiscoverRecord>> {
        &self.0
    }
}

impl AsRef<[DiscoverRecord]> for SharedRecords {
    fn as_ref(&self) -> &[DiscoverRecord] {
        self.0.as_slice()
    }
}

impl std::ops::Deref for SharedRecords {
    type Target = [DiscoverRecord];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_record_creation() {
        let record = DiscoverRecord::new(
            0,
            "hidden-1".to_string(),
            Some(0.5),
            0.7,
            vec![0.1, 0.2, -0.1],
        );

        assert_eq!(record.obs_index, 0);
        assert_eq!(record.neuron_uuid, "hidden-1");
        assert_eq!(record.value, Some(0.5));
        assert_eq!(record.activation, 0.7);
        assert_eq!(record.errors, vec![0.1, 0.2, -0.1]);
    }

    #[test]
    fn test_discover_record_without_value() {
        let record = DiscoverRecord::new(1, "output-0".to_string(), None, 0.9, vec![0.05]);

        assert_eq!(record.obs_index, 1);
        assert_eq!(record.value, None);
    }
}
