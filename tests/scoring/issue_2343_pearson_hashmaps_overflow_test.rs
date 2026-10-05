//! Integration test for Issue #2343: `pearson_correlation_hashmaps` returns NaN
//! on finite-input `f32` overflow instead of `0.0`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use neat_ai_discovery::analysis::detection::stats::pearson_correlation_hashmaps;

/// Issue #2343: finite `±2e30` values overflow the f32 covariance and
/// variance accumulators; the result must be 0.0, never NaN.
#[test]
fn hashmaps_f32_overflow_returns_zero() {
    let a: HashMap<u32, f32> = (0..10_u32)
        .map(|i| (i, if i.is_multiple_of(2) { 2.0e30 } else { -2.0e30 }))
        .collect();
    let b: HashMap<u32, f32> = (0..10_u32)
        .map(|i| (i, if i.is_multiple_of(2) { 1.0e10 } else { -5.0e9 }))
        .collect();
    assert!(
        a.values().chain(b.values()).all(|v| v.is_finite()),
        "the trigger must use only finite inputs"
    );

    let r = pearson_correlation_hashmaps(&a, &b, 3);
    assert_eq!(r, 0.0, "an overflowed correlation must be 0.0, got {r}");
}
