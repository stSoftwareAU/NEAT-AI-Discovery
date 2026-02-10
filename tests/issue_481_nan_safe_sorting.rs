//! Tests for Issue #481/#483: NaN-safe floating-point sorting across all modules.
//!
//! These tests validate that the `cmp_f32_desc` and `cmp_f32_asc` helpers
//! produce correct, deterministic sort results — even when NaN values are
//! present. This prevents silent sort corruption that could mis-rank
//! discovery candidates.

use neat_ai_discovery::analysis::constants::{cmp_f32_asc, cmp_f32_desc};

// =============================================================================
// Basic ordering tests
// =============================================================================

/// Verify descending sort puts largest values first.
#[test]
fn test_cmp_f32_desc_basic_ordering() {
    let mut values = vec![1.0_f32, 3.0, 2.0, 5.0, 4.0];
    values.sort_by(cmp_f32_desc);
    assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

/// Verify ascending sort puts smallest values first.
#[test]
fn test_cmp_f32_asc_basic_ordering() {
    let mut values = vec![3.0_f32, 1.0, 4.0, 2.0, 5.0];
    values.sort_by(cmp_f32_asc);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

// =============================================================================
// NaN handling tests
// =============================================================================

/// Verify NaN values do not corrupt the sort order of real values.
/// With `partial_cmp().unwrap_or(Equal)`, NaN values silently corrupt
/// sort stability. With `total_cmp`, NaN has a deterministic position.
#[test]
fn test_cmp_f32_desc_nan_deterministic() {
    let mut values = [1.0_f32, f32::NAN, 3.0, f32::NAN, 2.0];
    values.sort_by(cmp_f32_desc);

    // NaN sorts as greater than all finite values in total_cmp, so in descending
    // order NaN comes first, then real values in descending order
    assert!(values[0].is_nan());
    assert!(values[1].is_nan());
    assert_eq!(values[2], 3.0);
    assert_eq!(values[3], 2.0);
    assert_eq!(values[4], 1.0);
}

/// Verify NaN values sort deterministically in ascending sort.
#[test]
fn test_cmp_f32_asc_nan_deterministic() {
    let mut values = [f32::NAN, 3.0_f32, 1.0, f32::NAN, 2.0];
    values.sort_by(cmp_f32_asc);

    // Real values in ascending order, then NaN at the end
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 2.0);
    assert_eq!(values[2], 3.0);
    assert!(values[3].is_nan());
    assert!(values[4].is_nan());
}

// =============================================================================
// Edge cases
// =============================================================================

/// Verify that negative zero and positive zero are handled correctly.
#[test]
fn test_cmp_f32_handles_negative_zero() {
    let mut values = [0.0_f32, -0.0, 1.0, -1.0];
    values.sort_by(cmp_f32_asc);
    // -1.0 first, then zeros, then 1.0
    assert_eq!(values[0], -1.0);
    assert_eq!(values[3], 1.0);
}

/// Verify infinity values sort correctly.
#[test]
fn test_cmp_f32_handles_infinity() {
    let mut values = [f32::INFINITY, 1.0, f32::NEG_INFINITY, 0.0];
    values.sort_by(cmp_f32_desc);
    assert_eq!(values[0], f32::INFINITY);
    assert_eq!(values[1], 1.0);
    assert_eq!(values[2], 0.0);
    assert_eq!(values[3], f32::NEG_INFINITY);
}

/// Verify that an all-NaN slice doesn't panic.
#[test]
fn test_cmp_f32_all_nan_no_panic() {
    let mut values = [f32::NAN, f32::NAN, f32::NAN];
    values.sort_by(cmp_f32_desc);
    // All NaN — just verify no panic
    assert!(values.iter().all(|v| v.is_nan()));
}

/// Verify empty slice doesn't panic.
#[test]
fn test_cmp_f32_empty_slice_no_panic() {
    let mut values: Vec<f32> = vec![];
    values.sort_by(cmp_f32_desc);
    assert!(values.is_empty());
}
