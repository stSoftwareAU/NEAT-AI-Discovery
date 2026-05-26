//! Tests for Issue #424: Consolidate discovery constants into central module.
//!
//! Verifies that the central `constants` module provides a single source of
//! truth for discovery thresholds, and that analysis modules consume those
//! constants consistently.
//!
//! Issue #1295: the original file asserted `assert_eq!(constants::FOO, <literal>)`
//! for each consolidated constant. Those assertions were circular magic-value
//! tests (the literal in the test matched the literal in the source by
//! construction) and have been replaced with:
//!
//! - **Compile-only re-export and invariant checks** (`const _: T = constants::FOO;`
//!   and `const _: () = assert!(...)`) that confirm each consolidated constant
//!   is still reachable via the central module and that its declared envelope
//!   is preserved.
//! - **Behavioural tests** that exercise detection modules through the
//!   consolidated constants — these are the tests that genuinely guard the
//!   "single source of truth" contract.

use neat_ai_discovery::analysis::constants;

// =============================================================================
// Compile-time re-export checks — confirm every consolidated constant is
// still reachable via the central `constants` module.
// =============================================================================

const _: usize = constants::MIN_NEURON_SAMPLE_COUNT;
const _: usize = constants::MIN_DISCOVERY_SAMPLE_COUNT;
const _: [f32; 3] = constants::CANDIDATE_SENTINELS;
const _: f32 = constants::MIN_SENTINEL_FRACTION;
const _: f32 = constants::SENTINEL_TOLERANCE;
const _: f32 = constants::MIN_SENTINEL_GAP;
const _: f32 = constants::MIN_SOURCE_STD_DEV;
const _: usize = constants::DIVERSIFY_TOP_K;

// =============================================================================
// Compile-time invariant checks — pure-constant relationships that guard
// the consolidated module against accidental tuning past a safe envelope.
// =============================================================================

// Sample thresholds — pattern detection needs at least as many samples as
// the per-neuron statistical floor, and that floor must support a meaningful
// statistical calculation.
const _: () = assert!(
    constants::MIN_DISCOVERY_SAMPLE_COUNT >= constants::MIN_NEURON_SAMPLE_COUNT,
    "discovery sample minimum must be >= neuron sample minimum"
);
const _: () = assert!(
    constants::MIN_NEURON_SAMPLE_COUNT >= 2,
    "neuron sample minimum must be >= 2 for any statistical calculation"
);

// Diversification — a zero or negative top-k would disable diversification.
const _: () = assert!(
    constants::DIVERSIFY_TOP_K > 0,
    "DIVERSIFY_TOP_K must be positive to enable diversification"
);

// Source variance — a zero floor would admit constant sources, defeating
// variance filtering.
const _: () = assert!(
    constants::MIN_SOURCE_STD_DEV > 0.0,
    "MIN_SOURCE_STD_DEV must be strictly positive — a zero floor would \
     admit constant sources"
);

// Sentinel contract — tolerance must be smaller than the inter-sentinel gap
// (otherwise tolerance windows overlap and a value could match two
// sentinels at once), and the minimum fraction must lie strictly inside
// (0, 1).
const _: () = assert!(
    constants::SENTINEL_TOLERANCE < constants::MIN_SENTINEL_GAP,
    "SENTINEL_TOLERANCE must be smaller than MIN_SENTINEL_GAP to avoid \
     overlapping tolerance windows"
);
const _: () = assert!(
    constants::MIN_SENTINEL_FRACTION > 0.0 && constants::MIN_SENTINEL_FRACTION < 1.0,
    "MIN_SENTINEL_FRACTION must lie strictly inside (0, 1)"
);

// =============================================================================
// Behavioural tests — exercise the consolidated constants through real
// detection modules. These survive value tuning because they assert on
// observable behaviour, not on magic numbers.
// =============================================================================

/// Verify that saturated neuron detection still works (uses `MIN_NEURON_SAMPLE_COUNT`
/// indirectly via sample count filtering).
#[test]
fn test_saturation_detection_uses_centralised_constants() {
    use neat_ai_discovery::analysis::detection::saturation::detect_saturated_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let neurons = vec![("h1".to_string(), "LOGISTIC".to_string(), 0.0_f32)];

    // Fewer than MIN_NEURON_SAMPLE_COUNT (10) samples — should return no detections
    let too_few_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..5)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.999,
                errors: vec![0.1],
            })
            .collect(),
    )];

    let detected = detect_saturated_neurons(&neurons, &too_few_records);
    assert!(
        detected.is_empty(),
        "Should not detect saturation with fewer than MIN_NEURON_SAMPLE_COUNT samples"
    );
}

/// Verify that dead neuron detection still works with centralised constants.
#[test]
fn test_dead_neuron_detection_uses_centralised_constants() {
    use neat_ai_discovery::analysis::detection::dead_neuron::detect_dead_neurons;
    use neat_ai_discovery::types::DiscoverRecord;
    use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "LOGISTIC".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "LOGISTIC".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h1".into(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    // Fewer than MIN_DISCOVERY_SAMPLE_COUNT (20) samples — should return no detections
    let too_few: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..5)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            })
            .collect(),
    )];

    let detected = detect_dead_neurons(&creature, &too_few, None);
    assert!(
        detected.is_empty(),
        "Should not detect dead neurons with fewer than MIN_DISCOVERY_SAMPLE_COUNT samples"
    );
}

/// Every pair of `CANDIDATE_SENTINELS` is separated by at least
/// `MIN_SENTINEL_GAP`. This is a runtime check because iterating sentinel
/// pairs is not const-eval-able on stable Rust.
#[test]
fn test_candidate_sentinels_are_well_separated() {
    let sentinels = constants::CANDIDATE_SENTINELS;
    for (i, a) in sentinels.iter().enumerate() {
        for b in sentinels.iter().skip(i + 1) {
            assert!(
                (a - b).abs() >= constants::MIN_SENTINEL_GAP,
                "sentinels {a} and {b} must differ by at least \
                 MIN_SENTINEL_GAP={}",
                constants::MIN_SENTINEL_GAP
            );
        }
    }
}
