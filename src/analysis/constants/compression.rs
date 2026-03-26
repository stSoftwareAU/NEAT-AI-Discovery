//! Candidate compression thresholds (Issue #921).
//!
//! Constants governing when and how synapse candidates targeting the same
//! neuron are compressed into a single coordinated hidden-neuron candidate.

// =============================================================================
// Candidate Compression (Issue #921)
// =============================================================================

/// Minimum number of candidates sharing a target neuron to attempt compression.
///
/// Groups with fewer than this many distinct source neurons are not worth
/// compressing — a single synapse candidate is simpler and has lower
/// operation-count discount penalty.
///
/// ## Valid Range
/// Must be >= 2. Values above 3 may miss useful compression opportunities.
pub const MIN_COMPRESSED_SOURCES: usize = 2;

/// Maximum number of input synapses per compressed candidate.
///
/// Caps the number of inputs feeding into a single compressed hidden neuron.
/// Higher values increase the operation count (N+2 operations for N inputs),
/// which compounds the `COORDINATED_OPERATION_DISCOUNT` penalty.
///
/// ## Valid Range
/// Must be >= `MIN_COMPRESSED_SOURCES` and <= 8. Values above 5 receive
/// severe discount penalties (0.65^6 ≈ 0.075).
pub const MAX_COMPRESSION_INPUTS: usize = 5;

/// Saturation threshold for non-linear candidate compression (Issue #922).
///
/// When the estimated combined pre-activation exceeds this fraction of the
/// squash function's output range, a saturation discount is applied. This
/// accounts for diminished returns when TANH/GELU inputs are pushed into
/// saturated regimes where additional signal produces little change.
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values below 0.7 may over-discount useful
/// candidates. Values above 0.95 provide insufficient saturation correction.
pub const COMPRESSION_SATURATION_THRESHOLD: f32 = 0.9;

/// Minimum combined benefit ratio for non-linear compressed candidates (Issue #922).
///
/// The estimated combined gain through a non-linear squash must exceed the
/// best individual candidate gain multiplied by this ratio. This ensures the
/// interaction effect captured by TANH/GELU is meaningful and justifies the
/// additional operation-count penalty.
///
/// Matches `MIN_COMBINED_BENEFIT_RATIO` in `fan_in.rs` (1.05 = 5% improvement).
///
/// ## Valid Range
/// Must be > 1.0. Values above 1.20 may filter too aggressively.
pub const COMPRESSION_MIN_BENEFIT_RATIO: f32 = 1.05;
