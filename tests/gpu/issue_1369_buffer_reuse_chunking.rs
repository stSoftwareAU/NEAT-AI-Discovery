//! Tests for reusing GPU buffers across batch chunks (Issue #1369).
//!
//! `evaluate_helpful_batch` reuses a pool of GPU buffers across sample-set
//! chunks instead of allocating fresh buffers per chunk. The key invariant the
//! optimisation must preserve is that results stay **bit-identical** regardless
//! of how the input sample sets are split into chunks: each sample set is
//! computed wholly within one chunk, so the per-set statistics must not depend
//! on the batch size that controls chunking.
//!
//! These tests force several different chunkings of the same input and assert
//! the resulting statistics are bit-for-bit identical. Buffer-reuse corruption
//! (stale data leaking between chunks, wrong sub-range written/copied) would
//! break this invariant.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for test sample generation.
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analysis::samples::{HelpfulSample, HelpfulStats};

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build a deterministic helpful sample set of the requested length.
///
/// The `seed` varies the waveform per set so different sets produce distinct
/// statistics, exercising the buffer-reuse path with genuinely different data.
fn make_samples(len: usize, seed: usize) -> Vec<HelpfulSample> {
    (0..len)
        .map(|i| {
            let t = (i as f32 + seed as f32 * 1.7) / (len as f32 + 1.0);
            HelpfulSample {
                activation: (t * 9.0 - 4.5) + (t * 5.0 + seed as f32).sin() * 0.7,
                avg_error: (t * 3.0).cos() * 0.4 - 0.05,
                target_value: Some(t * 0.6),
                target_activation: Some(t.tanh()),
            }
        })
        .collect()
}

/// Assert two `HelpfulStats` are bit-identical (exact float bit patterns).
fn assert_stats_bit_identical(a: &HelpfulStats, b: &HelpfulStats, ctx: &str) {
    assert_eq!(a.positive_count, b.positive_count, "{ctx}: positive_count");
    assert_eq!(a.negative_count, b.negative_count, "{ctx}: negative_count");
    let pairs = [
        (
            a.positive_improvement_sum,
            b.positive_improvement_sum,
            "positive_improvement_sum",
        ),
        (
            a.negative_improvement_sum,
            b.negative_improvement_sum,
            "negative_improvement_sum",
        ),
        (
            a.positive_activation_sum,
            b.positive_activation_sum,
            "positive_activation_sum",
        ),
        (
            a.negative_activation_sum,
            b.negative_activation_sum,
            "negative_activation_sum",
        ),
        (a.error_sq_sum, b.error_sq_sum, "error_sq_sum"),
        (
            a.activation_sq_sum,
            b.activation_sq_sum,
            "activation_sq_sum",
        ),
        (
            a.error_activation_sum,
            b.error_activation_sum,
            "error_activation_sum",
        ),
    ];
    for (lhs, rhs, field) in pairs {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{ctx}: field {field} differs ({lhs} vs {rhs})"
        );
    }
}

/// Evaluate the same input with a given batch size (chunk granularity).
fn eval_with_batch_size(batch_size: usize, batch: &[&[HelpfulSample]]) -> Vec<HelpfulStats> {
    let gpu = GpuAnalyzer::new_with_batch_size(batch_size)
        .expect("GPU analyser should initialise on a GPU-equipped machine");
    gpu.evaluate_helpful_batch(batch)
        .expect("helpful batch evaluation should succeed")
}

/// Core regression: results must be bit-identical no matter how sample sets are
/// chunked. Mixes empty, small, medium and a reduction-path (>= 10k) set so both
/// the per-sample and the GPU-reduction code paths reuse pooled buffers.
#[test]
fn test_helpful_batch_chunking_is_bit_identical() {
    skip_without_gpu!();

    let s_empty: Vec<HelpfulSample> = Vec::new();
    let s_small = make_samples(7, 1);
    let s_med = make_samples(513, 2);
    let s_large = make_samples(12_000, 3); // >= GPU_REDUCTION_THRESHOLD -> reduction path
    let s_tiny = make_samples(1, 4);
    let s_mid2 = make_samples(300, 5);

    let batch: Vec<&[HelpfulSample]> = vec![
        &s_empty, &s_small, &s_med, &s_large, &s_tiny, &s_mid2, &s_small,
    ];

    // Single chunk (batch_size large enough to hold everything).
    let reference = eval_with_batch_size(64, &batch);
    assert_eq!(
        reference.len(),
        batch.len(),
        "result count must match input"
    );

    // Multiple chunkings that all force buffer reuse across chunks.
    for &bs in &[1usize, 2, 3, 5] {
        let got = eval_with_batch_size(bs, &batch);
        assert_eq!(
            got.len(),
            reference.len(),
            "batch_size {bs}: result count must match"
        );
        for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_stats_bit_identical(g, r, &format!("batch_size {bs}, set {i}"));
        }
    }
}

/// Reusing the pool across many small chunks must not accumulate or leak state:
/// the same single sample set repeated must yield identical stats every time.
#[test]
fn test_helpful_batch_repeated_set_no_state_leak() {
    skip_without_gpu!();

    let set = make_samples(128, 9);
    let batch: Vec<&[HelpfulSample]> = vec![&set, &set, &set, &set, &set];

    // batch_size 2 -> chunks [2,2,1]; pooled buffers are rewritten each chunk.
    let got = eval_with_batch_size(2, &batch);
    assert_eq!(got.len(), batch.len());
    for (i, stats) in got.iter().enumerate().skip(1) {
        assert_stats_bit_identical(stats, &got[0], &format!("repeat {i}"));
    }
    // Sanity: the chosen data actually produces non-trivial statistics.
    assert!(
        got[0].error_sq_sum > 0.0,
        "test data should yield a positive error_sq_sum"
    );
}
