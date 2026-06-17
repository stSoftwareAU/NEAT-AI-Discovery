//! Per-change-type prediction calibration correction (Issue #1131, #1162).
//!
//! The compiled calibration constants (`NEURON_PREDICTION_CALIBRATION`,
//! `SYNAPSE_PREDICTION_CALIBRATION`, `COORDINATED_PREDICTION_CALIBRATION`) are
//! fixed at build time and reflect the historical average over-estimation gap
//! between `expectedCreatureScoreGain` and the actual post-apply creature
//! score delta. Individual creatures can drift from that average: a creature
//! whose recent failure cache shows a 1300× over-estimate for `add-neurons`
//! should discount neuron predictions more than the global constant does.
//!
//! This module computes a scalar correction factor per change-type (e.g.,
//! `add-neurons`, `add-synapses`, `coordinated-structural`) from recent
//! entries of the failure cache. The correction is applied
//! **multiplicatively** to the existing calibration constant — it never
//! overrides the constant, and it is floored at
//! [`MIN_CALIBRATION_CORRECTION`] so a run of failures cannot collapse
//! predictions to zero.
//!
//! ## Per-(change-type, target-squash) tracking (Issue #1162)
//!
//! In practice the prediction-vs-actual gap depends strongly on the target
//! neuron's activation function. Failures against an unbounded squash like
//! `SELU` can over-estimate by 1000× while bounded squashes such as
//! `HARD_TANH` over-estimate by a much smaller factor. Grouping failure
//! entries by `change_type` alone loses that signal.
//!
//! When the failure-cache entries supply `targetNeuronInfo.squash`, this
//! module also computes a *specific* correction keyed by
//! `(change_type, target_squash)`. The lookup
//! [`CalibrationCorrection::correction_for`] returns the specific value
//! when at least [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries are
//! available, otherwise it falls back to the per-`change_type` correction,
//! and finally to [`NEUTRAL_CORRECTION`] (= 1.0) when neither is known.
//!
//! ## Cold-start prior for non-invertible target squashes (Issue #1192)
//!
//! Non-invertible / periodic activations — SINE, COSINE, GAUSSIAN, SQUARE,
//! ABSOLUTE — are intrinsically riskier as add-neuron / add-synapse targets:
//! a small change in the incoming weighted sum can flip the output sign
//! entirely. Without the cold-start prior, candidates aimed at one of these
//! targets would receive [`NEUTRAL_CORRECTION`] (= 1.0) until at least
//! [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] failures have been observed for the
//! exact `(change_type, squash)` bucket — meaning the *first* three failures
//! against, say, a SINE target cannot be demoted in advance.
//!
//! [`CalibrationCorrection::correction_for`] therefore returns the
//! conservative prior from
//! [`risky_squash_prior`]
//! when the target squash is in
//! [`RISKY_TARGET_SQUASHES`](crate::analysis::constants::RISKY_TARGET_SQUASHES)
//! and the learnt EWMA is not yet available for that bucket. Once the
//! per-key sample count reaches the warmup threshold, the learnt EWMA takes
//! over as it does for every other squash. Non-risky squashes (`ReLU` family,
//! Sigmoid, Tanh, …) keep the existing 1.0 cold-start default.
//!
//! # Formula
//!
//! Given failure cache entries with `expected_error_reduction` and
//! `actual_error_reduction`, the per-entry ratio is
//! `actual / expected`. An exponentially-weighted moving average (EWMA)
//! of these ratios is computed so recent outcomes dominate the estimate.
//! The EWMA is then clamped to
//! `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]` (= `[0.001, 1.0]`).
//!
//! Entries where `expected_error_reduction == 0` are skipped.
//!
//! # Integration
//!
//! The correction is loaded at the FFI entry point (`analyze_parallel_internal`)
//! and plumbed through `analyze_all` into the post-processing passes in
//! `synapse::post_processing` and `neuron::post_processing`. Each call site
//! multiplies the base calibration constant by
//! `correction.get_correction(change_type)` before invoking the logistic /
//! flat calibration function.

use serde::Deserialize;
use std::collections::HashMap;

use crate::analysis::constants::{is_risky_target_squash, risky_squash_prior};
use crate::analysis::scoring::sample_creature_disconnect::{
    SAMPLE_DISCONNECT_PENALTY, detect_disconnect_entry,
};

// =============================================================================
// Constants
// =============================================================================

/// Minimum correction factor (lower clamp).
///
/// A run of failures with `actual / expected` at or near zero clamps to this
/// floor rather than collapsing predictions to zero. 0.001 means a creature
/// whose whole recent history shows ~1000× over-estimates still receives a
/// small non-zero prediction — enough for exploration, while staying far
/// below the neutral factor of 1.0.
pub const MIN_CALIBRATION_CORRECTION: f32 = 0.001;

/// Neutral correction factor (upper clamp, returned for empty / unknown caches).
///
/// Predictions are only ever discounted further by this correction — never
/// inflated. The upper clamp of 1.0 ensures the correction can never make the
/// calibration constant larger than its compiled value.
pub const NEUTRAL_CORRECTION: f32 = 1.0;

/// EWMA smoothing coefficient applied to each successive entry (0 < α ≤ 1).
///
/// `α = 0.3` gives recent outcomes roughly 3× the weight of outcomes 3 entries
/// ago. A full-cache override only happens with ~20 entries all moving in the
/// same direction.
pub const CALIBRATION_CORRECTION_EWMA_ALPHA: f32 = 0.3;

/// Minimum number of usable failure-cache entries required for a
/// `(change_type, target_squash)` group before its specific correction is
/// preferred over the per-`change_type` fallback (Issue #1162).
///
/// Three entries is enough to establish that a (`change_type`, squash)
/// bucket is consistently mis-calibrated without leaking a single noisy
/// outlier into the per-candidate prediction. Smaller groups fall through
/// to the per-`change_type` correction.
pub const MIN_SPECIFIC_TARGET_SQUASH_SAMPLES: usize = 3;

/// Minimum number of usable failure-cache entries required for a
/// `(change_type, variant_key)` group before its specific correction is
/// applied to variant generators (Issue #1163).
///
/// Three entries is the same threshold as
/// [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`]: enough to establish a per-variant
/// trend without letting a single noisy outlier dominate the static
/// `expected_multiplier`. Smaller groups fall through to the static value
/// (the issue specifies the static multiplier as a ceiling).
pub const MIN_SPECIFIC_VARIANT_KEY_SAMPLES: usize = 3;

// =============================================================================
// Change-type identifiers
// =============================================================================

/// Change-type key for `add-neurons` candidates.
pub const CHANGE_TYPE_ADD_NEURONS: &str = "add-neurons";

/// Change-type key for `add-synapses` candidates.
pub const CHANGE_TYPE_ADD_SYNAPSES: &str = "add-synapses";

/// Change-type key for `coordinated-structural` candidates.
pub const CHANGE_TYPE_COORDINATED_STRUCTURAL: &str = "coordinated-structural";

/// Change-type key for `remove-neuron` (harmful-neuron) candidates (Issue #1425).
///
/// A single-op `RemoveNeuron` coordinated candidate is recorded in the failure
/// cache under this change type. Grouping these failures separately from the
/// generic `coordinated-structural` bucket lets the failure-cache EWMA learn
/// the remove-neuron-specific over-prediction (the harmful-neuron path
/// historically over-predicted gain by ~800×) and feed it back via
/// [`CalibrationCorrection::correction_for`].
pub const CHANGE_TYPE_REMOVE_NEURON: &str = "remove-neuron";

// =============================================================================
// Failure cache entry
// =============================================================================

/// A single failure-cache record: what we predicted vs what actually happened.
///
/// `change_type` identifies which prediction bucket this outcome belongs to.
/// Stable keys (see the `CHANGE_TYPE_*` constants) let the same cache feed
/// back into the correction for the same class of candidate next run.
///
/// `target_squash` is the activation function of the target neuron the
/// candidate was applied to (Issue #1162). It is parsed from
/// `targetNeuronInfo.squash` in the failure-cache JSON when present and is
/// `None` for legacy entries that omit it. When supplied it lets the
/// calibration learn that, say, `add-neurons` against `SELU` targets needs a
/// much smaller correction multiplier than the generic `add-neurons` group.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", from = "FailureCacheEntryRaw")]
pub struct FailureCacheEntry {
    /// Candidate classification (e.g., `add-neurons`, `add-synapses`,
    /// `coordinated-structural`). Used as the correction key.
    pub change_type: String,

    /// What we predicted — the `expectedErrorReduction` at the time of proposal.
    pub expected_error_reduction: f32,

    /// What actually happened — the post-apply `actualErrorReduction`. May be
    /// negative for candidates that harmed the network.
    pub actual_error_reduction: f32,

    /// Target neuron's activation function (squash), when reported by the
    /// upstream emitter. Populated from `targetNeuronInfo.squash` in the
    /// failure-cache JSON, or from a top-level `targetSquash` field.
    /// `None` for entries that lack target metadata, preserving backward
    /// compatibility (Issue #1162).
    #[serde(default)]
    pub target_squash: Option<String>,

    /// Variant key identifying which `variant_generation` strategy produced
    /// the candidate (Issue #1163).
    ///
    /// When supplied this lets the calibration learn that, say, the
    /// `gentle-nudge` neuron variant against a particular history of
    /// failures should have its static `expected_multiplier` discounted.
    /// `None` for legacy entries and for original (non-variant) candidates.
    #[serde(default)]
    pub variant_key: Option<String>,

    /// Stable UUID of the target neuron the candidate was applied to
    /// (Issue #1194). Populated from `targetNeuronInfo.uuid` in the
    /// failure-cache JSON, or from a top-level `targetUuid` field when
    /// supplied. `None` for legacy entries that omit target metadata.
    #[serde(default)]
    pub target_uuid: Option<String>,

    /// Number of recorded samples that improved under this candidate
    /// (Issue #1195). Populated from `improvedCount` in the failure-cache
    /// JSON when present. `None` for legacy entries that omit per-sample
    /// statistics.
    #[serde(default)]
    pub improved_count: Option<u32>,

    /// Total number of recorded samples evaluated for this candidate
    /// (Issue #1195). Populated from `totalCount` in the failure-cache
    /// JSON when present. `None` for legacy entries that omit per-sample
    /// statistics.
    #[serde(default)]
    pub total_count: Option<u32>,
}

/// Wire-format helper for [`FailureCacheEntry`] (Issue #1162).
///
/// Allows the failure JSON to either nest the squash under
/// `targetNeuronInfo` (the upstream NEAT-AI shape) or supply a top-level
/// `targetSquash` field, while keeping the in-memory struct flat.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FailureCacheEntryRaw {
    change_type: String,
    expected_error_reduction: f32,
    actual_error_reduction: f32,
    #[serde(default)]
    target_neuron_info: Option<TargetNeuronInfoRaw>,
    #[serde(default)]
    target_squash: Option<String>,
    #[serde(default)]
    variant_key: Option<String>,
    #[serde(default)]
    variant_info: Option<VariantInfoRaw>,
    /// Issue #1194: optional top-level `targetUuid` shorthand for the target
    /// neuron's UUID.
    #[serde(default)]
    target_uuid: Option<String>,
    /// Issue #1195: per-sample success counters populated by the upstream
    /// emitter. Both fields are optional for backward compatibility.
    #[serde(default)]
    improved_count: Option<u32>,
    #[serde(default)]
    total_count: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TargetNeuronInfoRaw {
    #[serde(default)]
    squash: Option<String>,
    /// Stable UUID of the target neuron (Issue #1194). Optional for backward
    /// compatibility with legacy failure-cache payloads.
    #[serde(default)]
    uuid: Option<String>,
}

/// Wire-format helper for nested `variantInfo.key` payloads (Issue #1163).
///
/// The upstream NEAT-AI emitter may nest the variant identifier under a
/// `variantInfo` object. This struct lets us accept either layout while the
/// in-memory [`FailureCacheEntry`] keeps a flat `variant_key`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VariantInfoRaw {
    #[serde(default)]
    key: Option<String>,
}

impl From<FailureCacheEntryRaw> for FailureCacheEntry {
    fn from(raw: FailureCacheEntryRaw) -> Self {
        // Capture nested target neuron info before splitting fields off below.
        let target_neuron_info = raw.target_neuron_info;
        // Prefer an explicit top-level `targetSquash`, fall back to the
        // nested `targetNeuronInfo.squash` shape that NEAT-AI emits.
        let target_squash = raw.target_squash.or_else(|| {
            target_neuron_info
                .as_ref()
                .and_then(|info| info.squash.clone())
        });
        // Issue #1194: prefer top-level `targetUuid`, fall back to
        // `targetNeuronInfo.uuid`.
        let target_uuid = raw
            .target_uuid
            .or_else(|| target_neuron_info.and_then(|info| info.uuid));
        // Issue #1163: prefer top-level `variantKey`, fall back to
        // `variantInfo.key`. Both are optional for backward compatibility.
        let variant_key = raw
            .variant_key
            .or_else(|| raw.variant_info.and_then(|info| info.key));
        Self {
            change_type: raw.change_type,
            expected_error_reduction: raw.expected_error_reduction,
            actual_error_reduction: raw.actual_error_reduction,
            target_squash,
            variant_key,
            target_uuid,
            improved_count: raw.improved_count,
            total_count: raw.total_count,
        }
    }
}

// =============================================================================
// CalibrationCorrection
// =============================================================================

/// Per-change-type scalar correction factors derived from recent outcomes.
///
/// A value of 1.0 means "apply the compiled calibration constant as-is".
/// A value of 0.001 (the floor) means "this change-type has been over-estimating
/// so much recently that predictions should be heavily discounted beyond
/// the compiled constant".
///
/// Two layers of correction are tracked (Issue #1162):
///
/// - `corrections` — keyed by `change_type` alone. Computed from every
///   usable entry of that change-type. This is the **fallback** layer used
///   when no specific correction is available for the requested target
///   squash.
/// - `specific_corrections` — keyed by `(change_type, target_squash)`.
///   Only populated for groups with at least
///   [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries, so that a single
///   noisy outcome cannot dominate per-candidate calibration. This is the
///   **specific** layer queried by [`Self::correction_for`].
#[derive(Debug, Clone, Default)]
pub struct CalibrationCorrection {
    corrections: HashMap<String, f32>,
    specific_corrections: HashMap<(String, String), f32>,
    /// Per-variant corrections keyed by `(change_type, variant_key)`
    /// (Issue #1163). Only populated for groups with at least
    /// [`MIN_SPECIFIC_VARIANT_KEY_SAMPLES`] usable entries.
    variant_corrections: HashMap<(String, String), f32>,
    /// Sample-vs-creature disconnect penalties keyed by
    /// `(change_type, target_squash, variant_key)` (Issue #1195).
    ///
    /// Each detected disconnect multiplies the penalty for the matching
    /// triple by [`SAMPLE_DISCONNECT_PENALTY`] (`0.5`); the cumulative
    /// product is clamped to
    /// `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`. Triples with
    /// no detected disconnects are absent from the map and resolve to
    /// [`NEUTRAL_CORRECTION`] on lookup.
    disconnect_penalties: HashMap<(String, String, String), f32>,
}

impl CalibrationCorrection {
    /// Returns a neutral correction where every change-type maps to 1.0.
    #[must_use]
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Builds a correction map from the supplied failure cache.
    ///
    /// # Algorithm
    ///
    /// 1. Group entries by `change_type` (fallback layer) and by
    ///    `(change_type, target_squash)` (specific layer, Issue #1162).
    /// 2. Skip entries with `expected_error_reduction == 0` (undefined ratio)
    ///    and entries whose ratio is non-finite.
    /// 3. For each group, compute an EWMA of `actual / expected` in the
    ///    order supplied. Callers pass entries oldest-first; the final EWMA
    ///    value therefore reflects the most recent outcomes most strongly.
    /// 4. Clamp the EWMA to
    ///    `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`.
    /// 5. Only specific groups with at least
    ///    [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries are kept; the
    ///    rest fall through to the per-`change_type` fallback at lookup.
    ///
    /// Empty caches, or caches with no usable entries for a change-type,
    /// produce a [`Self::neutral`] correction for that change-type on lookup.
    #[must_use]
    pub fn from_failure_cache(cache: &[FailureCacheEntry]) -> Self {
        // Group entries by change_type (fallback) and by (change_type,
        // target_squash) (specific), preserving supplied order within each
        // group so the EWMA reflects oldest -> newest.
        let mut grouped: HashMap<&str, Vec<f32>> = HashMap::new();
        let mut grouped_specific: HashMap<(&str, &str), Vec<f32>> = HashMap::new();
        // Issue #1163: per-variant grouping keyed by (change_type, variant_key).
        let mut grouped_variant: HashMap<(&str, &str), Vec<f32>> = HashMap::new();
        for entry in cache {
            // Skip entries with undefined ratios.
            if entry.expected_error_reduction == 0.0 {
                continue;
            }
            let ratio = entry.actual_error_reduction / entry.expected_error_reduction;
            // Guard against NaN or infinities leaking into the EWMA.
            if !ratio.is_finite() {
                continue;
            }
            grouped
                .entry(entry.change_type.as_str())
                .or_default()
                .push(ratio);
            if let Some(squash) = entry.target_squash.as_deref() {
                grouped_specific
                    .entry((entry.change_type.as_str(), squash))
                    .or_default()
                    .push(ratio);
            }
            if let Some(variant) = entry.variant_key.as_deref() {
                grouped_variant
                    .entry((entry.change_type.as_str(), variant))
                    .or_default()
                    .push(ratio);
            }
        }

        let mut corrections: HashMap<String, f32> = HashMap::new();
        for (change_type, ratios) in grouped {
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            corrections.insert(change_type.to_string(), clamped);
        }

        let mut specific_corrections: HashMap<(String, String), f32> = HashMap::new();
        for ((change_type, squash), ratios) in grouped_specific {
            // Issue #1162: only retain specific corrections backed by enough
            // samples; smaller groups fall through to the per-change_type
            // fallback at lookup time.
            if ratios.len() < MIN_SPECIFIC_TARGET_SQUASH_SAMPLES {
                continue;
            }
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            specific_corrections.insert((change_type.to_string(), squash.to_string()), clamped);
        }

        let mut variant_corrections: HashMap<(String, String), f32> = HashMap::new();
        for ((change_type, variant), ratios) in grouped_variant {
            // Issue #1163: only retain per-variant corrections backed by enough
            // samples; smaller groups fall through to the static multiplier
            // ceiling at lookup time.
            if ratios.len() < MIN_SPECIFIC_VARIANT_KEY_SAMPLES {
                continue;
            }
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            variant_corrections.insert((change_type.to_string(), variant.to_string()), clamped);
        }

        // Issue #1195: scan the cache for sample-vs-creature disconnects and
        // record a multiplicative penalty for the matching
        // (change_type, target_squash, variant_key) triple. Each disconnect
        // halves the surviving penalty; the cumulative product is clamped to
        // the standard correction floor so a stream of failures cannot
        // collapse predictions to zero.
        let mut disconnect_penalties: HashMap<(String, String, String), f32> = HashMap::new();
        for entry in cache {
            if !detect_disconnect_entry(entry) {
                continue;
            }
            // Disconnects without a full triple key cannot be filed; the
            // detector still fired (the event will be emitted by callers)
            // but no entry-level penalty is recorded.
            let (Some(squash), Some(variant)) =
                (entry.target_squash.as_deref(), entry.variant_key.as_deref())
            else {
                continue;
            };
            let key = (
                entry.change_type.clone(),
                squash.to_string(),
                variant.to_string(),
            );
            let current = *disconnect_penalties
                .get(&key)
                .unwrap_or(&NEUTRAL_CORRECTION);
            let next = (current * SAMPLE_DISCONNECT_PENALTY)
                .clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            disconnect_penalties.insert(key, next);
        }

        Self {
            corrections,
            specific_corrections,
            variant_corrections,
            disconnect_penalties,
        }
    }

    /// Look up the correction factor for a given change-type.
    ///
    /// Returns [`NEUTRAL_CORRECTION`] (1.0) when the change-type is absent —
    /// this preserves the compiled calibration constant unchanged for
    /// change-types with no recent failure data.
    ///
    /// This is the per-`change_type` fallback. Prefer
    /// [`Self::correction_for`] when the candidate's target squash is known.
    #[must_use]
    pub fn get_correction(&self, change_type: &str) -> f32 {
        self.corrections
            .get(change_type)
            .copied()
            .unwrap_or(NEUTRAL_CORRECTION)
    }

    /// Look up the most specific available correction factor (Issue #1162, #1192).
    ///
    /// Resolution order:
    ///
    /// 1. If `target_squash` is supplied **and** at least
    ///    [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable failure-cache entries
    ///    exist for `(change_type, target_squash)`, return that specific
    ///    learnt correction (the EWMA from Issue #1162).
    /// 2. Otherwise, if `target_squash` is supplied and is in the
    ///    `RISKY_TARGET_SQUASHES` set (non-invertible / periodic activations
    ///    such as SINE, COSINE, GAUSSIAN, SQUARE, ABSOLUTE), return the
    ///    conservative cold-start prior from
    ///    [`risky_squash_prior`]
    ///    (Issue #1192). This applies whenever the per-key sample count is
    ///    below the warmup threshold — including when no specific data has
    ///    been recorded at all — so an add-neuron / add-synapse candidate
    ///    aimed at a periodic target receives a conservative discount before
    ///    the learnt EWMA exists.
    /// 3. Otherwise, return the per-`change_type` correction (existing
    ///    behaviour).
    /// 4. Otherwise, return [`NEUTRAL_CORRECTION`] (= 1.0) so the compiled
    ///    calibration constant is used unchanged.
    ///
    /// Callers should pass `None` for `target_squash` when the candidate's
    /// target squash is unknown — in that case the cold-start risky-target
    /// prior cannot be applied and resolution skips straight to step 3.
    #[must_use]
    pub fn correction_for(&self, change_type: &str, target_squash: Option<&str>) -> f32 {
        if let Some(squash) = target_squash {
            // 1. Learnt EWMA wins once we have enough samples for this
            //    (change_type, target_squash) bucket.
            if let Some(value) = self
                .specific_corrections
                .get(&(change_type.to_string(), squash.to_string()))
                .copied()
            {
                return value;
            }
            // 2. Cold-start prior for non-invertible / periodic targets
            //    (Issue #1192). Applied only when the learnt EWMA is not
            //    yet available for the bucket.
            if is_risky_target_squash(squash) {
                return risky_squash_prior();
            }
        }
        // 3. Per-change_type fallback (or neutral via `get_correction`).
        self.get_correction(change_type)
    }

    /// Returns a reference to the underlying correction map, for metadata
    /// export at the FFI boundary.
    #[must_use]
    pub fn as_map(&self) -> &HashMap<String, f32> {
        &self.corrections
    }

    /// Returns a reference to the per-(`change_type`, `target_squash`) map
    /// (Issue #1162). Exposed for diagnostic tests; the FFI metadata only
    /// includes the per-`change_type` map for backward compatibility.
    #[must_use]
    pub fn specific_as_map(&self) -> &HashMap<(String, String), f32> {
        &self.specific_corrections
    }

    /// Returns a reference to the per-(`change_type`, `variant_key`) map
    /// (Issue #1163). Exposed for diagnostic tests; the FFI metadata only
    /// includes the per-`change_type` map for backward compatibility.
    #[must_use]
    pub fn variant_as_map(&self) -> &HashMap<(String, String), f32> {
        &self.variant_corrections
    }

    /// Look up the per-variant correction factor (Issue #1163).
    ///
    /// Resolution:
    ///
    /// 1. If `variant_key` is supplied **and** at least
    ///    [`MIN_SPECIFIC_VARIANT_KEY_SAMPLES`] usable failure-cache entries
    ///    exist for `(change_type, variant_key)`, return that calibrated
    ///    correction.
    /// 2. Otherwise return [`NEUTRAL_CORRECTION`] (= 1.0). This neutral value
    ///    means the static `expected_multiplier` is used unchanged at the
    ///    call site (the issue specifies the static value as a ceiling).
    ///
    /// Callers wanting the per-variant correction combined with the static
    /// multiplier should compute `min(static, variant_correction_for(...))`.
    #[must_use]
    pub fn variant_correction_for(&self, change_type: &str, variant_key: &str) -> f32 {
        self.variant_corrections
            .get(&(change_type.to_string(), variant_key.to_string()))
            .copied()
            .unwrap_or(NEUTRAL_CORRECTION)
    }

    /// Look up the sample-vs-creature disconnect penalty for the given
    /// `(change_type, target_squash, variant_key)` triple (Issue #1195).
    ///
    /// Returns [`NEUTRAL_CORRECTION`] (= 1.0) when no disconnect has been
    /// recorded for the triple — i.e. the calibration is unaffected by
    /// the new penalty layer.
    #[must_use]
    pub fn disconnect_penalty_for(
        &self,
        change_type: &str,
        target_squash: &str,
        variant_key: &str,
    ) -> f32 {
        self.disconnect_penalties
            .get(&(
                change_type.to_string(),
                target_squash.to_string(),
                variant_key.to_string(),
            ))
            .copied()
            .unwrap_or(NEUTRAL_CORRECTION)
    }

    /// Returns a reference to the per-(`change_type`, `target_squash`,
    /// `variant_key`) disconnect-penalty map (Issue #1195). Exposed for
    /// diagnostic tests; the FFI metadata only includes the per-`change_type`
    /// map for backward compatibility.
    #[must_use]
    pub fn disconnect_penalties_as_map(&self) -> &HashMap<(String, String, String), f32> {
        &self.disconnect_penalties
    }

    /// Look up the calibration correction for a fully-specified
    /// `(change_type, target_squash, variant_key)` triple, with the
    /// sample-vs-creature disconnect penalty applied (Issue #1195).
    ///
    /// The returned value is `correction_for(change_type, target_squash) *
    /// disconnect_penalty_for(change_type, target_squash, variant_key)`,
    /// clamped to `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`. When
    /// no disconnect has been recorded for the triple this is identical
    /// to [`Self::correction_for`] — preserving existing behaviour for
    /// candidates the new layer has not yet observed.
    #[must_use]
    pub fn correction_for_triple(
        &self,
        change_type: &str,
        target_squash: &str,
        variant_key: &str,
    ) -> f32 {
        let base = self.correction_for(change_type, Some(target_squash));
        let penalty = self.disconnect_penalty_for(change_type, target_squash, variant_key);
        (base * penalty).clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION)
    }

    /// True when no change-type has a correction recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }
}

// =============================================================================
// EWMA helper
// =============================================================================

/// Compute the exponentially-weighted moving average of a slice.
///
/// Starts from the first observation (no seed bias) and applies
/// `ewma = (1 - α) * prev + α * x` for each subsequent observation. Returns
/// [`NEUTRAL_CORRECTION`] for empty input.
fn ewma(values: &[f32], alpha: f32) -> f32 {
    if values.is_empty() {
        return NEUTRAL_CORRECTION;
    }
    let mut acc = values[0];
    for &x in &values[1..] {
        acc = (1.0 - alpha) * acc + alpha * x;
    }
    acc
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(change_type: &str, predicted: f32, actual: f32) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: predicted,
            actual_error_reduction: actual,
            target_squash: None,
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
        }
    }

    fn entry_with_squash(
        change_type: &str,
        predicted: f32,
        actual: f32,
        squash: &str,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: predicted,
            actual_error_reduction: actual,
            target_squash: Some(squash.to_string()),
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
        }
    }

    fn entry_with_variant(
        change_type: &str,
        predicted: f32,
        actual: f32,
        variant: &str,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: predicted,
            actual_error_reduction: actual,
            target_squash: None,
            variant_key: Some(variant.to_string()),
            target_uuid: None,
            improved_count: None,
            total_count: None,
        }
    }

    #[test]
    fn empty_cache_returns_neutral_correction() {
        let correction = CalibrationCorrection::from_failure_cache(&[]);
        assert!(correction.is_empty());
        assert!(
            (correction.get_correction(CHANGE_TYPE_ADD_NEURONS) - NEUTRAL_CORRECTION).abs() < 1e-9
        );
        assert!(
            (correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES) - NEUTRAL_CORRECTION).abs() < 1e-9
        );
    }

    #[test]
    fn all_negative_actuals_clamp_to_minimum() {
        // Every entry is a net-harm outcome; raw ratio is negative.
        let cache: Vec<FailureCacheEntry> = (0..10)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 0.001, -0.0005))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (value - MIN_CALIBRATION_CORRECTION).abs() < 1e-9,
            "expected MIN_CALIBRATION_CORRECTION (={MIN_CALIBRATION_CORRECTION}), got {value}"
        );
    }

    #[test]
    fn uniform_cache_returns_that_mean_ratio() {
        // 20 entries with identical ratio of 1/1000. EWMA of a constant sequence
        // is that constant.
        let cache: Vec<FailureCacheEntry> = (0..20)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.000_001))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        // Expected ratio = 0.000001 / 0.001 = 0.001, which equals the floor.
        assert!(
            (value - 0.001).abs() < 1e-6,
            "expected ≈ 0.001, got {value}"
        );
    }

    #[test]
    fn mixed_cache_returns_ewma_of_ratios() {
        // Alternating entries: ratios of 0.5 and 0.1. Final EWMA value with
        // α = 0.3 is deterministic and can be recomputed here.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.1),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.1),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES);

        // Manual EWMA with α = 0.3:
        //   start = 0.5
        //   step1 = 0.7*0.5 + 0.3*0.1 = 0.38
        //   step2 = 0.7*0.38 + 0.3*0.5 = 0.416
        //   step3 = 0.7*0.416 + 0.3*0.1 = 0.3212
        let expected = 0.321_2_f32;
        assert!(
            (value - expected).abs() < 1e-4,
            "expected ≈ {expected}, got {value}"
        );
    }

    #[test]
    fn zero_predicted_entries_are_skipped() {
        // A zero-predicted entry has an undefined ratio and must not affect the
        // correction. Mix one in amongst legitimate entries.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.0, 0.0),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.005),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.005),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        // Both usable entries have ratio 0.5, so EWMA = 0.5.
        assert!((value - 0.5).abs() < 1e-6, "expected 0.5, got {value}");
    }

    #[test]
    fn non_finite_ratios_are_ignored() {
        // Infinite or NaN ratios (e.g., extremely tiny predicted near-zero) must
        // not contaminate the EWMA.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, f32::MIN_POSITIVE, f32::MAX),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!((value - 0.2).abs() < 1e-6, "expected 0.2, got {value}");
    }

    #[test]
    fn different_change_types_are_independent() {
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.2),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.2),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.8),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.8),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!((correction.get_correction(CHANGE_TYPE_ADD_NEURONS) - 0.2).abs() < 1e-6);
        assert!((correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES) - 0.8).abs() < 1e-6);
        // Unknown change-type falls back to neutral.
        assert!(
            (correction.get_correction(CHANGE_TYPE_COORDINATED_STRUCTURAL) - NEUTRAL_CORRECTION)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn correction_above_one_clamps_to_neutral() {
        // A lucky run where actual exceeded predicted must never inflate the
        // base calibration constant — the correction only ever discounts.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.01),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.01),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION (=1.0), got {value}"
        );
    }

    // =========================================================================
    // Issue #1162 — per-(change_type, target_squash) calibration tracking
    // =========================================================================

    #[test]
    fn only_change_type_data_is_used_as_fallback() {
        // No entry carries a target_squash, so the specific layer is empty
        // and `correction_for` falls back to the per-change_type value.
        let cache: Vec<FailureCacheEntry> = (0..6)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(correction.specific_as_map().is_empty());

        // Specific lookup with any squash falls back to the change_type EWMA.
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (value - 0.4).abs() < 1e-6,
            "expected fallback ≈ 0.4, got {value}"
        );

        // Without a target_squash the result also matches the change_type EWMA.
        let none_value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, None);
        assert!((none_value - 0.4).abs() < 1e-6);
    }

    #[test]
    fn specific_data_is_preferred_when_threshold_is_met() {
        // Three or more entries against SELU establish a specific correction
        // distinct from the change_type fallback.
        let mut cache = vec![
            // Generic add-neurons history dominated by mid-range outcomes.
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
        ];
        // SELU-specific entries that consistently massively over-estimate.
        for _ in 0..MIN_SPECIFIC_TARGET_SQUASH_SAMPLES {
            cache.push(entry_with_squash(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "SELU",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);

        // Generic fallback reflects the union of both groups (SELU entries
        // also feed the per-change_type EWMA), but `correction_for(SELU)`
        // must return the SELU-specific value (≈ 0.001) rather than the
        // higher fallback.
        let selu = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (selu - 0.001).abs() < 1e-4,
            "expected SELU-specific ≈ 0.001, got {selu}"
        );

        // A different squash with no specific data falls back.
        let fallback = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("HARD_TANH"));
        let direct_fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (fallback - direct_fallback).abs() < 1e-9,
            "expected HARD_TANH to fall back to {direct_fallback}, got {fallback}"
        );
        // The fallback should be larger than the SELU-specific value.
        assert!(
            fallback > selu,
            "fallback ({fallback}) should exceed SELU-specific ({selu})"
        );
    }

    #[test]
    fn insufficient_specific_samples_fall_back_to_change_type() {
        // Fewer than MIN_SPECIFIC_TARGET_SQUASH_SAMPLES SELU entries — the
        // specific bucket must NOT be retained, even though the data exists.
        const _: () = assert!(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES >= 2);
        let mut cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
        ];
        for _ in 0..(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES - 1) {
            cache.push(entry_with_squash(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "SELU",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            correction.specific_as_map().is_empty(),
            "specific corrections must not be retained below the sample threshold"
        );

        // Lookup with the SELU squash returns the per-change_type fallback.
        let lookup = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        let fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (lookup - fallback).abs() < 1e-9,
            "expected fallback {fallback}, got {lookup}"
        );
    }

    #[test]
    fn nan_and_zero_divisor_specific_entries_are_skipped() {
        // The two zero / non-finite entries against SELU must NOT count toward
        // the sample threshold, so the SELU-specific bucket stays empty and
        // lookups fall back to the per-change_type value.
        let cache = vec![
            // Zero predicted -> undefined ratio.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 0.0, 0.0, "SELU"),
            // Non-finite ratio.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, f32::MIN_POSITIVE, f32::MAX, "SELU"),
            // One legitimate SELU entry — still below the threshold.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.001, "SELU"),
            // Generic change_type entries to seed the fallback.
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            correction.specific_as_map().is_empty(),
            "non-finite / zero entries must not count toward the specific threshold"
        );

        let lookup = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        let fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (lookup - fallback).abs() < 1e-9,
            "non-finite / zero specific entries must not affect lookup"
        );
    }

    #[test]
    fn target_squash_parsed_from_target_neuron_info() {
        // Issue #1162: failure-cache JSON nests squash under
        // `targetNeuronInfo.squash`. Round-trip a representative payload.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001,
            "targetNeuronInfo": { "squash": "SELU" }
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.change_type, CHANGE_TYPE_ADD_NEURONS);
        assert_eq!(parsed.target_squash.as_deref(), Some("SELU"));
    }

    #[test]
    fn target_squash_optional_for_legacy_entries() {
        // Legacy entries that omit `targetNeuronInfo` must still parse.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert!(parsed.target_squash.is_none());
    }

    #[test]
    fn correction_for_returns_neutral_when_nothing_known() {
        let correction = CalibrationCorrection::neutral();
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION, got {value}"
        );
    }

    // =========================================================================
    // Issue #1163 — per-(change_type, variant_key) calibration tracking
    // =========================================================================

    #[test]
    fn variant_key_parsed_from_top_level_field() {
        // Top-level `variantKey` should be honoured.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001,
            "variantKey": "gentle-nudge"
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.variant_key.as_deref(), Some("gentle-nudge"));
    }

    #[test]
    fn variant_key_parsed_from_nested_variant_info() {
        // Nested `variantInfo.key` is also accepted (Issue #1163).
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001,
            "variantInfo": { "key": "micro-nudge" }
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.variant_key.as_deref(), Some("micro-nudge"));
    }

    #[test]
    fn improved_and_total_counts_parsed_from_top_level_fields() {
        // Issue #1195: failure-cache JSON may carry `improvedCount` and
        // `totalCount` describing per-sample success counters. They must
        // round-trip into the optional fields on the in-memory struct.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": -0.5,
            "improvedCount": 1032,
            "totalCount": 1036
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.improved_count, Some(1032));
        assert_eq!(parsed.total_count, Some(1036));
    }

    #[test]
    fn improved_and_total_counts_optional_for_legacy_entries() {
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": -0.5
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert!(parsed.improved_count.is_none());
        assert!(parsed.total_count.is_none());
    }

    #[test]
    fn variant_key_optional_for_legacy_entries() {
        // Entries without any variant identifier must still parse.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert!(parsed.variant_key.is_none());
    }

    #[test]
    fn variant_correction_neutral_when_no_history() {
        // No variant history -> neutral so the static multiplier wins.
        let correction = CalibrationCorrection::neutral();
        let value = correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "gentle-nudge");
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION, got {value}"
        );
    }

    #[test]
    fn insufficient_variant_history_falls_back_to_neutral() {
        // Fewer than MIN_SPECIFIC_VARIANT_KEY_SAMPLES variant entries — no
        // calibrated value should be retained.
        const _: () = assert!(MIN_SPECIFIC_VARIANT_KEY_SAMPLES >= 2);
        let mut cache = Vec::new();
        for _ in 0..(MIN_SPECIFIC_VARIANT_KEY_SAMPLES - 1) {
            cache.push(entry_with_variant(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "gentle-nudge",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            correction.variant_as_map().is_empty(),
            "variant corrections must not be retained below the sample threshold"
        );
        let lookup = correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "gentle-nudge");
        assert!(
            (lookup - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION fallback, got {lookup}"
        );
    }

    #[test]
    fn long_variant_failure_history_drives_calibrated_value() {
        // 10 SELU-against-Gentle-Nudge failures with 1000× over-estimation should
        // produce a calibrated value at the floor (0.001).
        let cache: Vec<FailureCacheEntry> = (0..10)
            .map(|_| entry_with_variant(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.000_001, "gentle-nudge"))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "gentle-nudge");
        assert!(
            (value - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
            "expected MIN_CALIBRATION_CORRECTION, got {value}"
        );
    }

    #[test]
    fn long_variant_success_history_clamps_at_neutral() {
        // A run where every variant outcome over-shot the prediction must be
        // clamped to NEUTRAL_CORRECTION (= 1.0). The calibrated value never
        // inflates predictions above the static `expected_multiplier` ceiling.
        let cache: Vec<FailureCacheEntry> = (0..10)
            .map(|_| entry_with_variant(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.01, "gentle-nudge"))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "gentle-nudge");
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION ceiling, got {value}"
        );
    }

    #[test]
    fn variant_key_buckets_are_independent() {
        // Two variants for the same change_type must keep separate corrections.
        let mut cache = Vec::new();
        for _ in 0..MIN_SPECIFIC_VARIANT_KEY_SAMPLES {
            cache.push(entry_with_variant(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "gentle-nudge",
            ));
        }
        for _ in 0..MIN_SPECIFIC_VARIANT_KEY_SAMPLES {
            cache.push(entry_with_variant(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.5,
                "conservative",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let gentle = correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "gentle-nudge");
        let conservative =
            correction.variant_correction_for(CHANGE_TYPE_ADD_NEURONS, "conservative");
        assert!(
            (gentle - 0.001).abs() < 1e-4,
            "gentle-nudge should be near floor, got {gentle}"
        );
        assert!(
            (conservative - 0.5).abs() < 1e-4,
            "conservative should be near 0.5, got {conservative}"
        );
        assert!(
            conservative > gentle,
            "conservative ({conservative}) should exceed gentle-nudge ({gentle})"
        );
    }

    #[test]
    fn deterministic_for_same_cache() {
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.001),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.001),
        ];
        let a = CalibrationCorrection::from_failure_cache(&cache);
        let b = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            (a.get_correction(CHANGE_TYPE_ADD_NEURONS) - b.get_correction(CHANGE_TYPE_ADD_NEURONS))
                .abs()
                < 1e-12
        );
    }

    // =========================================================================
    // Issue #1192 — cold-start prior for non-invertible target squashes
    // =========================================================================

    use crate::analysis::constants::{
        MAX_RISKY_SQUASH_PRIOR, MIN_RISKY_SQUASH_PRIOR, RISKY_SQUASH_PRIOR_DEFAULT,
        RISKY_TARGET_SQUASHES,
    };
    use serial_test::serial;

    /// Risky-target squashes match the non-invertible activations documented
    /// in `src/activations.rs` (the issue scopes the new prior to SINE,
    /// COSINE, GAUSSIAN, SQUARE, ABSOLUTE).
    #[test]
    fn risky_target_squashes_cover_documented_non_invertibles() {
        let expected = ["SINE", "COSINE", "GAUSSIAN", "SQUARE", "ABSOLUTE"];
        for name in expected {
            assert!(
                RISKY_TARGET_SQUASHES.contains(&name),
                "{name} should be in RISKY_TARGET_SQUASHES"
            );
        }
        // Sanity check: monotone activations must NOT be in the risky set.
        for name in ["RELU", "SIGMOID", "TANH", "GELU", "ELU", "IDENTITY"] {
            assert!(
                !RISKY_TARGET_SQUASHES.contains(&name),
                "{name} must not be flagged as risky"
            );
        }
    }

    /// SINE target with zero failure samples receives the conservative cold-
    /// start prior, not the global default of 1.0.
    #[test]
    #[serial]
    fn sine_target_with_zero_samples_receives_conservative_prior() {
        // SAFETY: env vars guarded by serial_test. Single-threaded under #[serial].
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        let correction = CalibrationCorrection::neutral();
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (value - RISKY_SQUASH_PRIOR_DEFAULT).abs() < 1e-6,
            "expected risky prior {RISKY_SQUASH_PRIOR_DEFAULT}, got {value}"
        );
        assert!(
            (value - NEUTRAL_CORRECTION).abs() > 1e-3,
            "risky prior must be visibly below neutral 1.0"
        );
    }

    /// SINE target with three or more failure samples uses the learnt EWMA,
    /// not the cold-start prior — the prior only governs cold-start.
    #[test]
    #[serial]
    fn sine_target_with_enough_samples_uses_learnt_ewma() {
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        // Three SINE failures with ratio 0.5 — well above both the prior (0.25)
        // and the floor (0.001). The EWMA must win over the prior so the
        // observed-better-than-prior outcome is reflected.
        let cache: Vec<FailureCacheEntry> = (0..MIN_SPECIFIC_TARGET_SQUASH_SAMPLES)
            .map(|_| entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5, "SINE"))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (value - 0.5).abs() < 1e-4,
            "expected learnt EWMA ≈ 0.5, got {value}"
        );
        assert!(
            (value - RISKY_SQUASH_PRIOR_DEFAULT).abs() > 0.1,
            "learnt EWMA must dominate the cold-start prior once warmed up"
        );
    }

    /// SINE target with fewer than three samples still receives the prior —
    /// the warmup threshold gates EWMA, not the prior.
    #[test]
    #[serial]
    fn sine_target_below_threshold_still_uses_prior() {
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        const _: () = assert!(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES >= 2);
        let cache: Vec<FailureCacheEntry> = (0..(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES - 1))
            .map(|_| entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.001, "SINE"))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (value - RISKY_SQUASH_PRIOR_DEFAULT).abs() < 1e-6,
            "below-threshold SINE entries must fall through to the prior, got {value}"
        );
    }

    /// `ReLU` (monotone, non-risky) target with zero samples still receives
    /// the global default of 1.0 — the cold-start prior is only for the
    /// risky squash set.
    #[test]
    #[serial]
    fn relu_target_with_zero_samples_uses_global_default() {
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        let correction = CalibrationCorrection::neutral();
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("RELU"));
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION (=1.0), got {value}"
        );
    }

    /// All five risky squashes receive the conservative prior at cold start.
    #[test]
    #[serial]
    fn every_risky_squash_receives_prior_at_cold_start() {
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        let correction = CalibrationCorrection::neutral();
        for &squash in RISKY_TARGET_SQUASHES {
            let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some(squash));
            assert!(
                (value - RISKY_SQUASH_PRIOR_DEFAULT).abs() < 1e-6,
                "{squash} should receive the cold-start prior, got {value}"
            );
        }
    }

    /// The per-`change_type` fallback (no `target_squash` supplied) is unaffected
    /// by the cold-start prior — the prior only fires when a risky target
    /// squash is provided.
    #[test]
    #[serial]
    fn missing_target_squash_skips_risky_prior() {
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
        let correction = CalibrationCorrection::neutral();
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, None);
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "missing target_squash must keep the per-change_type fallback"
        );
    }

    /// `NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR` overrides the default and is
    /// clamped to `[MIN_RISKY_SQUASH_PRIOR, MAX_RISKY_SQUASH_PRIOR]`.
    #[test]
    #[serial]
    fn env_var_overrides_and_clamps_risky_prior() {
        let correction = CalibrationCorrection::neutral();

        // Sensible mid-range override.
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR", "0.10");
        }
        let mid = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (mid - 0.10).abs() < 1e-6,
            "expected env override 0.10, got {mid}"
        );

        // Below the lower clamp -> clamped to MIN_RISKY_SQUASH_PRIOR (0.001).
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR", "0.0");
        }
        let low = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (low - MIN_RISKY_SQUASH_PRIOR).abs() < 1e-6,
            "expected lower clamp {MIN_RISKY_SQUASH_PRIOR}, got {low}"
        );

        // Above the upper clamp -> clamped to MAX_RISKY_SQUASH_PRIOR (1.0).
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR", "5.0");
        }
        let high = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (high - MAX_RISKY_SQUASH_PRIOR).abs() < 1e-6,
            "expected upper clamp {MAX_RISKY_SQUASH_PRIOR}, got {high}"
        );

        // Unparsable -> default.
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR", "not-a-number");
        }
        let bad = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
        assert!(
            (bad - RISKY_SQUASH_PRIOR_DEFAULT).abs() < 1e-6,
            "unparsable env var must fall back to the default"
        );

        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR");
        }
    }
}
