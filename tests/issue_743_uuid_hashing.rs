//! Tests for Issue #743: Deterministic UUID hashing without intermediate String.
//!
//! Verifies that `deterministic_coordinated_neuron_uuid()` produces consistent,
//! deterministic results for the same inputs and distinct results for different
//! inputs.

use neat_ai_discovery::analysis::deterministic_coordinated_neuron_uuid;

#[test]
fn deterministic_same_inputs_produce_same_uuid() {
    let uuid1 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    let uuid2 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    assert_eq!(uuid1, uuid2, "Same inputs must produce identical UUIDs");
}

#[test]
fn deterministic_different_inputs_produce_different_uuids() {
    let uuid1 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    let uuid2 =
        deterministic_coordinated_neuron_uuid("source-abc", "target-xyz", "TANH", 0.5, -0.3, 0.1);
    assert_ne!(
        uuid1, uuid2,
        "Different squash must produce different UUIDs"
    );

    let uuid3 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.6,
        -0.3,
        0.1,
    );
    assert_ne!(
        uuid1, uuid3,
        "Different weight must produce different UUIDs"
    );

    let uuid4 = deterministic_coordinated_neuron_uuid(
        "source-different",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    assert_ne!(
        uuid1, uuid4,
        "Different source must produce different UUIDs"
    );
}

#[test]
fn deterministic_uuid_has_correct_prefix() {
    let uuid = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    assert!(
        uuid.starts_with("coordinated-hidden-"),
        "UUID must start with 'coordinated-hidden-' prefix, got: {uuid}"
    );
}

#[test]
fn deterministic_uuid_has_correct_length() {
    let uuid = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    // "coordinated-hidden-" (19 chars) + 16 hex digits = 35 chars
    assert_eq!(
        uuid.len(),
        35,
        "UUID should be 35 chars (19 prefix + 16 hex), got {} for: {uuid}",
        uuid.len()
    );
}

#[test]
fn deterministic_uuid_hex_suffix_is_valid() {
    let uuid = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.5,
        -0.3,
        0.1,
    );
    let hex_part = &uuid["coordinated-hidden-".len()..];
    assert_eq!(hex_part.len(), 16);
    assert!(
        hex_part.chars().all(|c| c.is_ascii_hexdigit()),
        "Hex suffix must contain only hex digits, got: {hex_part}"
    );
}

#[test]
fn deterministic_uuid_float_sensitivity() {
    // Very close but distinct float values must produce different UUIDs
    let uuid1 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.1,
        0.2,
        0.3,
    );
    let uuid2 = deterministic_coordinated_neuron_uuid(
        "source-abc",
        "target-xyz",
        "LOGISTIC",
        0.1 + f32::EPSILON,
        0.2,
        0.3,
    );
    assert_ne!(
        uuid1, uuid2,
        "Floats differing by EPSILON must produce different UUIDs"
    );
}
