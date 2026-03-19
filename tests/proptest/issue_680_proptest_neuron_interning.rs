//! Property-based tests for `NeuronIndex` UUID interning (Issue #680).
//!
//! Uses `proptest` to verify round-trip identity, sequential indexing, and
//! structural invariants of the string interning pool.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::intern::NeuronIndex;
use proptest::prelude::*;
use std::collections::HashSet;

/// Strategy for generating UUID-like strings.
fn uuid_string() -> impl Strategy<Value = String> {
    "[a-z0-9]{1,8}(-[a-z0-9]{1,8}){0,3}".prop_map(|s| s)
}

/// Strategy for generating a Vec of UUID-like strings.
fn uuid_vec(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(uuid_string(), min..=max)
}

// =============================================================================
// 1. Round-Trip Identity
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Interning a string and looking up by index must return the original string.
    #[test]
    fn round_trip_intern_get_uuid(uuid in uuid_string()) {
        let mut index = NeuronIndex::new();
        let idx = index.intern(&uuid);
        prop_assert_eq!(
            index.get_uuid(idx),
            Some(uuid.as_str()),
            "get_uuid(intern(s)) must equal s"
        );
    }

    /// Looking up by string after interning must return the assigned index.
    #[test]
    fn round_trip_intern_get_index(uuid in uuid_string()) {
        let mut index = NeuronIndex::new();
        let idx = index.intern(&uuid);
        prop_assert_eq!(
            index.get_index(&uuid),
            Some(idx),
            "get_index(s) must equal intern(s)"
        );
    }

    /// Interning the same string twice must return the same index (idempotency).
    #[test]
    fn intern_idempotent(uuid in uuid_string()) {
        let mut index = NeuronIndex::new();
        let idx1 = index.intern(&uuid);
        let idx2 = index.intern(&uuid);
        prop_assert_eq!(idx1, idx2, "Interning same string must return same index");
        prop_assert_eq!(index.len(), 1, "Should have exactly one entry");
    }
}

// =============================================================================
// 2. Sequential Index Assignment
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Indices must be sequential starting from 0, based on insertion order of unique strings.
    #[test]
    fn indices_are_sequential(uuids in uuid_vec(1, 50)) {
        let mut index = NeuronIndex::new();
        let mut seen = HashSet::new();
        let mut expected_next_idx = 0u32;

        for uuid in &uuids {
            let idx = index.intern(uuid);
            if seen.insert(uuid.clone()) {
                // First time seeing this string
                prop_assert_eq!(
                    idx, expected_next_idx,
                    "New string '{}' should get index {}",
                    uuid, expected_next_idx
                );
                expected_next_idx += 1;
            }
        }

        prop_assert_eq!(
            index.len(),
            seen.len(),
            "len() must equal number of unique strings"
        );
    }

    /// len() must equal the count of distinct strings interned.
    #[test]
    fn len_equals_unique_count(uuids in uuid_vec(0, 100)) {
        let mut index = NeuronIndex::new();
        for uuid in &uuids {
            index.intern(uuid);
        }

        let unique_count = uuids.iter().collect::<HashSet<_>>().len();
        prop_assert_eq!(
            index.len(),
            unique_count,
            "len() must equal count of unique strings"
        );
    }
}

// =============================================================================
// 3. Boundary and Error Conditions
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// get_uuid() must return None for out-of-bounds indices.
    #[test]
    fn get_uuid_out_of_bounds(
        uuids in uuid_vec(0, 20),
        extra_offset in 1u32..100,
    ) {
        let mut index = NeuronIndex::new();
        for uuid in &uuids {
            index.intern(uuid);
        }

        let out_of_bounds = index.len() as u32 + extra_offset;
        prop_assert_eq!(
            index.get_uuid(out_of_bounds),
            None,
            "get_uuid({}) should be None for index with {} entries",
            out_of_bounds, index.len()
        );
    }

    /// get_index() must return None for strings not yet interned.
    #[test]
    fn get_index_unknown_string(
        known in uuid_vec(0, 10),
        unknown in uuid_string(),
    ) {
        let mut index = NeuronIndex::new();
        for uuid in &known {
            index.intern(uuid);
        }

        // Only test if the unknown string is truly unknown
        if !known.contains(&unknown) {
            prop_assert_eq!(
                index.get_index(&unknown),
                None,
                "get_index('{}') should be None for unknown string",
                unknown
            );
        }
    }

    /// After clear(), len() must be 0 and all previous lookups must return None.
    #[test]
    fn clear_resets_everything(uuids in uuid_vec(1, 30)) {
        let mut index = NeuronIndex::new();
        let indices: Vec<u32> = uuids.iter().map(|u| index.intern(u)).collect();

        index.clear();

        prop_assert!(index.is_empty(), "Should be empty after clear");
        prop_assert_eq!(index.len(), 0, "len() should be 0 after clear");

        for uuid in &uuids {
            prop_assert_eq!(
                index.get_index(uuid),
                None,
                "get_index('{}') should be None after clear",
                uuid
            );
        }
        for &idx in &indices {
            prop_assert_eq!(
                index.get_uuid(idx),
                None,
                "get_uuid({}) should be None after clear",
                idx
            );
        }
    }
}

// =============================================================================
// 4. Iterator Consistency
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// iter() must yield exactly len() items, all with valid round-trip lookups.
    #[test]
    fn iter_consistent_with_lookups(uuids in uuid_vec(1, 50)) {
        let mut index = NeuronIndex::new();
        for uuid in &uuids {
            index.intern(uuid);
        }

        let items: Vec<(u32, String)> = index.iter().map(|(i, s)| (i, s.to_string())).collect();

        prop_assert_eq!(
            items.len(),
            index.len(),
            "iter() must yield len() items"
        );

        for (idx, uuid) in &items {
            prop_assert_eq!(
                index.get_uuid(*idx),
                Some(uuid.as_str()),
                "Round-trip via iter: get_uuid({}) should equal '{}'",
                idx, uuid
            );
            prop_assert_eq!(
                index.get_index(uuid),
                Some(*idx),
                "Round-trip via iter: get_index('{}') should equal {}",
                uuid, idx
            );
        }
    }
}
