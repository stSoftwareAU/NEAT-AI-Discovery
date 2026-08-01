//! Cumulative decode budget for Parquet reads (Issue #1869).
//!
//! Parquet's dictionary and RLE encodings routinely achieve far better than
//! the 3:1 ratio the pre-load projection assumes, so a projection can never be
//! a *bound*. This budget is the bound: every record materialised by the reader
//! is charged against it inside the batch loop, and the decode fails loud with
//! a typed [`DiscoveryError::MemoryExhausted`] the moment the ceiling is
//! reached — before the allocation that would exhaust the host completes.

use anyhow::Result;

use crate::DiscoveryError;
use crate::analysis::utils::get_memory_info;
use crate::parquet_format::footer::RECORD_STRUCT_BYTES;

/// One megabyte in bytes.
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Fraction of total system RAM used as the decode ceiling when no explicit
/// budget and no environment override are supplied. Mirrors the 50 % cap the
/// pre-load memory check already applies.
const DEFAULT_TOTAL_RAM_FRACTION: u64 = 2;

/// Estimated in-memory bytes for one materialised `DiscoverRecord`.
///
/// Counts the struct itself, the heap allocation backing the neuron UUID, and
/// four bytes per `f32` error. Allocator slack and the per-neuron `HashMap`
/// entry are deliberately excluded — the figure is a floor on the true cost,
/// which keeps the budget conservative in the caller's favour.
#[must_use]
pub fn estimated_record_bytes(uuid_len: usize, errors_len: usize) -> u64 {
    RECORD_STRUCT_BYTES
        .saturating_add(uuid_len as u64)
        .saturating_add((errors_len as u64).saturating_mul(4))
}

/// Default decode ceiling derived from total system RAM.
///
/// Returns half of `total_system_bytes`, or `None` when the total is unknown
/// (`0`) — an unknown figure cannot produce a meaningful ceiling, so the caller
/// decodes unbounded rather than guessing a limit that might reject valid work.
#[must_use]
pub const fn default_decode_limit_bytes(total_system_bytes: u64) -> Option<u64> {
    if total_system_bytes == 0 {
        None
    } else {
        Some(total_system_bytes / DEFAULT_TOTAL_RAM_FRACTION)
    }
}

/// Cumulative byte budget enforced while decoding a Parquet file.
#[derive(Debug)]
pub struct DecodeBudget {
    limit_bytes: Option<u64>,
    consumed_bytes: u64,
}

impl DecodeBudget {
    /// Create a budget with an explicit byte ceiling (`None` = unbounded).
    #[must_use]
    pub const fn new(limit_bytes: Option<u64>) -> Self {
        Self {
            limit_bytes,
            consumed_bytes: 0,
        }
    }

    /// Resolve the ceiling for a decode, in priority order:
    ///
    /// 1. the caller's `budget_mb` (the analysis phase forwards its
    ///    `max_analysis_memory_mb`),
    /// 2. `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB`,
    /// 3. half of total system RAM.
    #[must_use]
    pub fn resolve(budget_mb: Option<u64>) -> Self {
        if let Some(mb) = budget_mb.filter(|mb| *mb > 0) {
            return Self::new(Some(mb.saturating_mul(BYTES_PER_MB)));
        }
        if let Some(mb) = crate::config::max_parquet_decode_mb() {
            return Self::new(Some(mb.saturating_mul(BYTES_PER_MB)));
        }
        let (_available, total) = get_memory_info();
        Self::new(default_decode_limit_bytes(total))
    }

    /// Charge one materialised record against the budget.
    ///
    /// Returns a typed [`DiscoveryError::MemoryExhausted`] once the cumulative
    /// decode reaches the ceiling, so the caller aborts mid-decode instead of
    /// completing an allocation the host cannot hold.
    pub fn charge_record(
        &mut self,
        uuid_len: usize,
        errors_len: usize,
        file_path: &str,
    ) -> Result<()> {
        self.consumed_bytes = self
            .consumed_bytes
            .saturating_add(estimated_record_bytes(uuid_len, errors_len));

        let Some(limit) = self.limit_bytes else {
            return Ok(());
        };
        if self.consumed_bytes <= limit {
            return Ok(());
        }

        Err(DiscoveryError::MemoryExhausted {
            detail: format!(
                "Parquet decode budget exceeded while reading '{file_path}': \
                 decoded records already need {consumed_mb} MB, above the \
                 {limit_mb} MB decode budget. Parquet compresses this schema \
                 far better than 3:1, so the file's on-disk size does not bound \
                 its decoded size. Raise the analysis memory budget, set \
                 NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB, or reduce \
                 discoverySampleRate so fewer rows are recorded.",
                consumed_mb = self.consumed_bytes.div_ceil(BYTES_PER_MB),
                limit_mb = limit.div_ceil(BYTES_PER_MB),
            ),
        }
        .into())
    }

    /// Bytes charged so far.
    #[must_use]
    pub const fn consumed_bytes(&self) -> u64 {
        self.consumed_bytes
    }

    /// The ceiling in bytes, or `None` when the decode is unbounded.
    #[must_use]
    pub const fn limit_bytes(&self) -> Option<u64> {
        self.limit_bytes
    }
}
