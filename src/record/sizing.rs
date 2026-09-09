//! The one overflow-checked "records per sample" derivation (Issue #2047).

use anyhow::Result;

/// Derive how many discovery records a single training sample produces.
///
/// The count is `non_input_neuron_count + creature_input`: one record per
/// non-input neuron, plus one per input neuron (input activations are recorded
/// for GPU-assisted analysis).
///
/// The addition is **`checked_add`, never a bare `+`** (Issue #1867):
/// `creature.input` is caller-supplied, and release builds set no
/// `overflow-checks`, so a value near `usize::MAX` would wrap silently and hand
/// the Parquet writer — and every per-observation `Vec::with_capacity` — a size
/// unrelated to the real record count. An overflow is reported as an error
/// instead.
///
/// A zero result is *not* an error here: the helper only derives the size.
/// Rejecting a zero count belongs to `validation::validate_and_resolve_indices`,
/// which reports it alongside the empty-training-data case.
pub fn records_per_sample(non_input_neuron_count: usize, creature_input: usize) -> Result<usize> {
    non_input_neuron_count
        .checked_add(creature_input)
        .ok_or_else(|| anyhow::anyhow!("Discovery records per sample would overflow usize"))
}
