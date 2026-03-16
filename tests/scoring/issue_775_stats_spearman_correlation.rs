//! Integration tests for Spearman rank correlation in stats module (Issue #775).

use neat_ai_discovery::analysis::detection::stats::{compute_ranks, spearman_rank_correlation};

// ── compute_ranks ────────────────────────────────────────────────────────

#[test]
fn compute_ranks_empty_input() {
    let ranks = compute_ranks(&[]);
    assert!(ranks.is_empty());
}

#[test]
fn compute_ranks_single_element() {
    let ranks = compute_ranks(&[42.0]);
    assert_eq!(ranks, vec![1.0]);
}

#[test]
fn compute_ranks_sorted_values() {
    let ranks = compute_ranks(&[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(ranks, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn compute_ranks_reverse_sorted() {
    let ranks = compute_ranks(&[40.0, 30.0, 20.0, 10.0]);
    assert_eq!(ranks, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn compute_ranks_with_ties() {
    // Values: [3, 1, 1, 2] → sorted ranks: 1→1.5, 1→1.5, 2→3, 3→4
    let ranks = compute_ranks(&[3.0, 1.0, 1.0, 2.0]);
    assert_eq!(ranks[0], 4.0); // 3.0 is rank 4
    assert_eq!(ranks[1], 1.5); // 1.0 tied at ranks 1-2 → avg 1.5
    assert_eq!(ranks[2], 1.5); // 1.0 tied at ranks 1-2 → avg 1.5
    assert_eq!(ranks[3], 3.0); // 2.0 is rank 3
}

#[test]
fn compute_ranks_all_tied() {
    let ranks = compute_ranks(&[5.0, 5.0, 5.0]);
    // All share average rank of (1+2+3)/3 = 2.0
    assert_eq!(ranks, vec![2.0, 2.0, 2.0]);
}

// ── spearman_rank_correlation ────────────────────────────────────────────

#[test]
fn spearman_empty_input_returns_zero() {
    assert_eq!(spearman_rank_correlation(&[], &[]), 0.0);
}

#[test]
fn spearman_single_element_returns_zero() {
    assert_eq!(spearman_rank_correlation(&[1.0], &[2.0]), 0.0);
}

#[test]
fn spearman_perfect_positive() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let rho = spearman_rank_correlation(&x, &y);
    assert!((rho - 1.0).abs() < 1e-5, "expected ~1.0, got {rho}");
}

#[test]
fn spearman_perfect_negative() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
    let rho = spearman_rank_correlation(&x, &y);
    assert!((rho + 1.0).abs() < 1e-5, "expected ~-1.0, got {rho}");
}

#[test]
fn spearman_monotonic_nonlinear() {
    // y = x^3 is monotonically increasing, so Spearman should be +1.0
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f32> = x.iter().map(|v| v * v * v).collect();
    let rho = spearman_rank_correlation(&x, &y);
    assert!(
        (rho - 1.0).abs() < 1e-5,
        "expected ~1.0 for monotonic nonlinear, got {rho}"
    );
}

#[test]
fn spearman_with_ties() {
    // Ties in input: ranks should use average
    let x = vec![1.0, 2.0, 2.0, 3.0];
    let y = vec![10.0, 20.0, 20.0, 30.0];
    let rho = spearman_rank_correlation(&x, &y);
    assert!(
        (rho - 1.0).abs() < 1e-5,
        "expected ~1.0 with ties, got {rho}"
    );
}

#[test]
fn spearman_zero_variance_returns_zero() {
    let x = vec![5.0, 5.0, 5.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(spearman_rank_correlation(&x, &y), 0.0);
}

#[test]
fn spearman_no_monotonic_relationship() {
    // U-shaped relationship: not monotonic
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = vec![10.0, 5.0, 2.0, 1.0, 2.0, 5.0, 10.0];
    let rho = spearman_rank_correlation(&x, &y);
    // Should be near zero for symmetric U-shape
    assert!(rho.abs() < 0.3, "expected near-zero for U-shape, got {rho}");
}
