//! Regression coverage for hostile detection-threshold environment overrides.

use neat_ai_discovery::analysis::detection::noise_signal::detect_noisy_neurons;
use neat_ai_discovery::config::{dominance_threshold, gradient_threshold, noise_signal_threshold};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson};
use serial_test::serial;

const FALLBACK: f32 = 0.75;
const HOSTILE_VALUES: [&str; 7] = ["inf", "-inf", "NaN", "0", "-0", "-1", "1e999"];

fn with_env<T>(name: &str, value: Option<&str>, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();
    // SAFETY: Every caller is serialised so no other test mutates this variable.
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
    let result = body();
    // SAFETY: Every caller is serialised so no other test mutates this variable.
    unsafe {
        match previous {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
    result
}

fn assert_hostile_values_fall_back(name: &str, accessor: fn(f32) -> f32) {
    for value in HOSTILE_VALUES {
        let actual = with_env(name, Some(value), || accessor(FALLBACK));
        assert_eq!(
            actual, FALLBACK,
            "{name}={value:?} must use the caller-supplied fallback"
        );
    }
    for value in ["", "malformed"] {
        let actual = with_env(name, Some(value), || accessor(FALLBACK));
        assert_eq!(actual, FALLBACK, "{name}={value:?} must use the fallback");
    }
    let actual = with_env(name, None, || accessor(FALLBACK));
    assert_eq!(actual, FALLBACK, "unset {name} must use the fallback");
}

#[test]
#[serial]
fn noise_signal_threshold_rejects_non_finite_and_non_positive_overrides() {
    assert_hostile_values_fall_back(
        "NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD",
        noise_signal_threshold,
    );
}

#[test]
#[serial]
fn dominance_threshold_rejects_non_finite_and_non_positive_overrides() {
    assert_hostile_values_fall_back("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD", dominance_threshold);
}

#[test]
#[serial]
fn gradient_threshold_rejects_non_finite_and_non_positive_overrides() {
    assert_hostile_values_fall_back("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD", gradient_threshold);
}

fn noisy_neuron_fixture() -> (CreatureJson, Vec<DiscoverRecord>) {
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "hidden-noisy".to_string(),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
        input: 1,
        output: 1,
    };
    let records = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { -1.0 } else { 1.0 };
            let error = if i % 4 < 2 { -0.5 } else { 0.5 };
            DiscoverRecord::new(
                i,
                "hidden-noisy".to_string(),
                Some(activation),
                activation,
                vec![error],
            )
        })
        .collect();
    (creature, records)
}

#[test]
#[serial]
fn noisy_neuron_detection_uses_default_when_threshold_is_nan() {
    let (creature, records) = noisy_neuron_fixture();
    let records = [("hidden-noisy".to_string(), records)];

    let default_result = with_env(
        "NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD",
        Some("NaN"),
        || detect_noisy_neurons(&creature, &records),
    );
    assert!(
        default_result.is_empty(),
        "the fixture's ratio is below the default threshold of 2.0"
    );

    let lower_threshold_result = with_env(
        "NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD",
        Some("0.1"),
        || detect_noisy_neurons(&creature, &records),
    );
    assert_eq!(
        lower_threshold_result.len(),
        1,
        "positive control proves the fixture reaches the detector gate"
    );
}
