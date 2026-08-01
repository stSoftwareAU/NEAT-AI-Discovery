//! Bound on the snapshot exporter's dense columnar allocation (Issue #1869).
//!
//! The exporter materialises `activation`, `value` and `errors` vectors of
//! length `distinct_obs` **for every neuron**, regardless of how sparse the
//! recording actually is. That is `O(distinct_neurons × distinct_obs)` and is
//! not bounded by the decoded record count, so it is checked explicitly before
//! the first vector is allocated.

use anyhow::Result;

use crate::DiscoveryError;
use crate::parquet_format::decode_budget::DecodeBudget;

/// Bytes one `(neuron, observation)` cell occupies across the three dense
/// columns: `f32` activation, `Option<f32>` value, and an empty `Vec<f32>`
/// header for the errors list.
pub const DENSE_CELL_BYTES: u64 =
    (size_of::<f32>() + size_of::<Option<f32>>() + size_of::<Vec<f32>>()) as u64;

/// Project the bytes the dense snapshot grid occupies.
#[must_use]
pub fn projected_dense_snapshot_bytes(neuron_count: usize, obs_count: usize) -> u64 {
    (neuron_count as u64)
        .saturating_mul(obs_count as u64)
        .saturating_mul(DENSE_CELL_BYTES)
}

/// Fail loud when the dense grid would exceed the decode ceiling.
///
/// Uses the same ceiling as the Parquet decode (an explicit budget, then
/// `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB`, then half of total RAM), so a
/// sparse-but-wide recording cannot turn a bounded decode into an unbounded
/// export.
pub fn ensure_dense_snapshot_fits(neuron_count: usize, obs_count: usize) -> Result<()> {
    let Some(limit) = DecodeBudget::resolve(None).limit_bytes() else {
        return Ok(());
    };
    let projected = projected_dense_snapshot_bytes(neuron_count, obs_count);
    if projected <= limit {
        return Ok(());
    }

    const BYTES_PER_MB: u64 = 1024 * 1024;
    Err(DiscoveryError::MemoryExhausted {
        detail: format!(
            "Visualisation snapshot too large: a dense {neuron_count} neuron × \
             {obs_count} observation grid needs {projected_mb} MB, above the \
             {limit_mb} MB decode budget. Lower maxObs, export fewer neurons, or \
             raise NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB.",
            projected_mb = projected.div_ceil(BYTES_PER_MB),
            limit_mb = limit.div_ceil(BYTES_PER_MB),
        ),
    }
    .into())
}
