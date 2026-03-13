//! Shared helper utilities for detection modules (Issue #804).
//!
//! Extracts common boilerplate patterns used across detection modules into
//! reusable functions, reducing duplication and improving maintainability.

use std::collections::HashMap;

use crate::types::DiscoverRecord;

/// Build a lookup map from neuron UUID strings to their discovery records.
///
/// Many detection modules need to look up records by neuron UUID. This
/// helper extracts the common pattern of converting `&[(String, Vec<DiscoverRecord>)]`
/// into a `HashMap<&str, &Vec<DiscoverRecord>>` for O(1) lookups.
///
/// # Arguments
/// * `neuron_records` - Slice of `(neuron_uuid, records)` tuples.
///
/// # Returns
/// A `HashMap` mapping borrowed UUID strings to borrowed record vectors.
pub fn build_record_map(
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> HashMap<&str, &Vec<DiscoverRecord>> {
    neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect()
}
