//! Neuron UUID string interning for reduced memory allocations.
//!
//! This module provides `NeuronIndex`, a string interning pool that maps neuron UUID strings
//! to compact `u32` indices. This eliminates redundant string allocations when building
//! HashMaps/HashSets with UUID keys during analysis.
//!
//! # Memory Savings
//!
//! For a creature with 500 neurons and 10,000 synapses:
//! - Without interning: Each `HashMap` entry clones the full UUID (~36 bytes per String)
//! - With interning: Each entry uses a `u32` index (4 bytes)
//!
//! Example savings for `existing_synapses` `HashSet`<(String, String)>:
//! - Before: 10,000 × 2 × ~36 bytes ≈ 720KB
//! - After:  10,000 × 2 × 4 bytes = 80KB (89% reduction)
//!
//! # Usage
//!
//! ```ignore
//! let mut index = NeuronIndex::new();
//!
//! // Intern neuron UUIDs
//! let idx1 = index.intern("hidden-abc-123");
//! let idx2 = index.intern("hidden-abc-123"); // Returns same index
//! assert_eq!(idx1, idx2);
//!
//! // Look up the original string
//! assert_eq!(index.get_uuid(idx1), Some("hidden-abc-123"));
//!
//! // Use indices as HashMap keys instead of String
//! let existing_synapses: HashSet<(u32, u32)> = /* ... */;
//! ```
//!
//! # Thread Safety
//!
//! `NeuronIndex` is NOT thread-safe. Create one per thread or wrap in a Mutex/RwLock
//! if shared access is required.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;
use std::sync::Arc;

/// A string interning pool for neuron identity strings.
///
/// Maps neuron identity strings to compact `u32` indices, enabling
/// memory-efficient HashMap/HashSet operations with integer keys instead of
/// String keys. Identity strings may be RFC 4122 UUIDs or stringified
/// integers from TypeScript runtime `neuron.id` (Issue #950) — no format
/// validation is performed.
#[derive(Debug, Clone)]
pub struct NeuronIndex {
    /// Maps interned UUID strings to their assigned index.
    uuid_to_index: HashMap<Arc<str>, u32>,
    /// Stores the interned strings in index order for reverse lookup.
    index_to_uuid: Vec<Arc<str>>,
}

impl Default for NeuronIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuronIndex {
    /// Creates a new, empty `NeuronIndex`.
    #[inline]
    pub fn new() -> Self {
        Self {
            uuid_to_index: HashMap::new(),
            index_to_uuid: Vec::new(),
        }
    }

    /// Creates a new `NeuronIndex` with the specified capacity.
    ///
    /// Use this when you know approximately how many unique UUIDs will be interned
    /// to avoid reallocations.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            uuid_to_index: HashMap::with_capacity(capacity),
            index_to_uuid: Vec::with_capacity(capacity),
        }
    }

    /// Interns a UUID string and returns its index.
    ///
    /// If the UUID has already been interned, returns the existing index.
    /// Otherwise, assigns a new index and stores the string.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` unique UUIDs are interned.
    #[inline]
    pub fn intern(&mut self, uuid: &str) -> u32 {
        // Fast path: check if already interned
        if let Some(&idx) = self.uuid_to_index.get(uuid) {
            return idx;
        }

        // Slow path: intern new string
        let idx = self.index_to_uuid.len() as u32;
        let arc: Arc<str> = uuid.into();
        self.uuid_to_index.insert(arc.clone(), idx);
        self.index_to_uuid.push(arc);
        idx
    }

    /// Gets the UUID string for a given index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get_uuid(&self, index: u32) -> Option<&str> {
        self.index_to_uuid.get(index as usize).map(|arc| &**arc)
    }

    /// Gets the index for a given UUID if it has been interned.
    ///
    /// Returns `None` if the UUID has not been interned.
    #[inline]
    pub fn get_index(&self, uuid: &str) -> Option<u32> {
        self.uuid_to_index.get(uuid).copied()
    }

    /// Returns the number of unique UUIDs that have been interned.
    #[inline]
    pub fn len(&self) -> usize {
        self.index_to_uuid.len()
    }

    /// Returns `true` if no UUIDs have been interned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index_to_uuid.is_empty()
    }

    /// Clears all interned UUIDs.
    #[inline]
    pub fn clear(&mut self) {
        self.uuid_to_index.clear();
        self.index_to_uuid.clear();
    }

    /// Returns an iterator over all interned UUIDs and their indices.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.index_to_uuid
            .iter()
            .enumerate()
            .map(|(idx, arc)| (idx as u32, &**arc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_index_is_empty() {
        let index = NeuronIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_intern_returns_sequential_indices() {
        let mut index = NeuronIndex::new();

        let idx0 = index.intern("hidden-0");
        let idx1 = index.intern("hidden-1");
        let idx2 = index.intern("hidden-2");

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_intern_same_uuid_returns_same_index() {
        let mut index = NeuronIndex::new();

        let idx1 = index.intern("hidden-abc-123");
        let idx2 = index.intern("hidden-abc-123");
        let idx3 = index.intern("hidden-abc-123");

        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
        assert_eq!(index.len(), 1); // Only one unique UUID
    }

    #[test]
    fn test_round_trip_index_to_uuid() {
        let mut index = NeuronIndex::new();

        let uuid = "output-xyz-789";
        let idx = index.intern(uuid);

        assert_eq!(index.get_uuid(idx), Some(uuid));
    }

    #[test]
    fn test_round_trip_uuid_to_index() {
        let mut index = NeuronIndex::new();

        let uuid = "input-0";
        let idx = index.intern(uuid);

        assert_eq!(index.get_index(uuid), Some(idx));
    }

    #[test]
    fn test_get_uuid_invalid_index() {
        let index = NeuronIndex::new();
        assert_eq!(index.get_uuid(0), None);
        assert_eq!(index.get_uuid(100), None);
    }

    #[test]
    fn test_get_index_unknown_uuid() {
        let index = NeuronIndex::new();
        assert_eq!(index.get_index("unknown-uuid"), None);
    }

    #[test]
    fn test_with_capacity() {
        let index = NeuronIndex::with_capacity(100);
        assert!(index.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut index = NeuronIndex::new();
        index.intern("hidden-1");
        index.intern("hidden-2");

        assert_eq!(index.len(), 2);

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.get_index("hidden-1"), None);
    }

    #[test]
    fn test_iter() {
        let mut index = NeuronIndex::new();
        index.intern("a");
        index.intern("b");
        index.intern("c");

        let items: Vec<(u32, &str)> = index.iter().collect();

        assert_eq!(items, vec![(0, "a"), (1, "b"), (2, "c")]);
    }

    #[test]
    fn test_interning_typical_uuid_formats() {
        let mut index = NeuronIndex::new();

        // Test various UUID formats used in NEAT-AI
        let uuids = [
            "input-0",
            "input-100",
            "hidden-abc123def-456",
            "output-xyz789",
            "constant-1",
            "550e8400-e29b-41d4-a716-446655440000", // Standard UUID v4 format
        ];

        for (expected_idx, uuid) in uuids.iter().enumerate() {
            let idx = index.intern(uuid);
            assert_eq!(idx, expected_idx as u32);
            assert_eq!(index.get_uuid(idx), Some(*uuid));
        }
    }

    #[test]
    fn test_clone() {
        let mut original = NeuronIndex::new();
        original.intern("a");
        original.intern("b");

        let cloned = original.clone();

        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get_index("a"), Some(0));
        assert_eq!(cloned.get_index("b"), Some(1));
    }

    #[test]
    fn test_default() {
        let index = NeuronIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn test_many_uuids() {
        let mut index = NeuronIndex::new();

        // Simulate a creature with 500 neurons
        for i in 0..500 {
            let uuid = format!("hidden-neuron-{i}");
            let idx = index.intern(&uuid);
            assert_eq!(idx, i);
        }

        assert_eq!(index.len(), 500);

        // Verify round-trip for some
        assert_eq!(index.get_uuid(0), Some("hidden-neuron-0"));
        assert_eq!(index.get_uuid(499), Some("hidden-neuron-499"));
    }

    #[test]
    fn test_synapse_pair_use_case() {
        // Simulate the use case from the issue: storing synapse pairs as (u32, u32)
        let mut index = NeuronIndex::new();

        // Intern source and target UUIDs
        let from_idx = index.intern("input-0");
        let to_idx = index.intern("hidden-abc");

        // Store as compact tuple
        let synapse_key: (u32, u32) = (from_idx, to_idx);

        // Verify sizes
        assert_eq!(std::mem::size_of_val(&synapse_key), 8); // 2 × 4 bytes

        // Compare to String-based key size (much larger due to heap allocation overhead)
        let string_key: (String, String) = ("input-0".to_string(), "hidden-abc".to_string());
        // String is 24 bytes on 64-bit systems (ptr + len + capacity), plus heap data
        assert!(std::mem::size_of_val(&string_key) > std::mem::size_of_val(&synapse_key));
    }
}
