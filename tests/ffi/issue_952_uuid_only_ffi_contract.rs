//! Issue #952 — Enforce UUID-only FFI discovery contract.
//!
//! The FFI contract requires stable UUID strings for all neuron/synapse
//! identity fields. Runtime integer IDs (e.g. `"0"`, `"42"`) must be
//! rejected at the FFI boundary to prevent NaN/dangling-reference
//! regressions and brittle cache compatibility.
//!
//! These tests verify:
//! 1. Numeric-only neuron IDs are rejected at deserialisation.
//! 2. Valid UUID formats (RFC 4122, `input-N`, descriptive strings) are accepted.
//! 3. Discovery responses include a `schemaVersion` field.
//! 4. Coordinated structural operations use UUID-based references.

use neat_ai_discovery::{CreatureJson, NeuronData};

// ============================================================================
// Validation: reject purely numeric neuron IDs
// ============================================================================

/// Creature JSON with purely numeric neuron UUIDs must be rejected.
/// Numeric IDs are internal-only and must never cross the FFI boundary.
#[test]
fn reject_creature_with_numeric_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "2", "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
            {"uuid": "3", "type": "output", "squash": "LOGISTIC", "bias": -0.2}
        ],
        "synapses": [
            {"fromUUID": "0", "toUUID": "2", "weight": 0.5}
        ],
        "input": 2,
        "output": 1
    }"#;

    let result = serde_json::from_str::<CreatureJson>(json);
    assert!(
        result.is_err(),
        "purely numeric neuron UUIDs must be rejected at deserialisation"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("numeric") || err_msg.contains("integer"),
        "error message should mention numeric/integer IDs: {err_msg}"
    );
}

/// Synapse JSON with numeric fromUUID/toUUID must be rejected.
#[test]
fn reject_synapse_with_numeric_from_uuid() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "550e8400-e29b-41d4-a716-446655440000", "type": "output", "squash": "LOGISTIC"}
        ],
        "synapses": [
            {"fromUUID": "42", "toUUID": "550e8400-e29b-41d4-a716-446655440000", "weight": 0.5}
        ],
        "input": 2,
        "output": 1
    }"#;

    let result = serde_json::from_str::<CreatureJson>(json);
    assert!(result.is_err(), "numeric synapse fromUUID must be rejected");
}

/// `NeuronData` with numeric `neuron_uuid` must be rejected.
#[test]
fn reject_neuron_data_with_numeric_id() {
    let json = r#"{"neuron_uuid": "7", "activation": 0.85, "errors": [0.01, -0.02]}"#;

    let result = serde_json::from_str::<NeuronData>(json);
    assert!(
        result.is_err(),
        "numeric neuron_uuid in NeuronData must be rejected"
    );
}

/// Large numeric IDs must also be rejected.
#[test]
fn reject_large_numeric_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "100000", "type": "input", "squash": "IDENTITY"},
            {"uuid": "100001", "type": "input", "squash": "IDENTITY"},
            {"uuid": "999999", "type": "hidden", "squash": "TANH", "bias": 0.0},
            {"uuid": "1000000", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "100000", "toUUID": "999999", "weight": 1.0}
        ],
        "input": 2,
        "output": 1
    }"#;

    let result = serde_json::from_str::<CreatureJson>(json);
    assert!(
        result.is_err(),
        "large numeric neuron UUIDs must be rejected"
    );
}

// ============================================================================
// Validation: accept valid UUID formats
// ============================================================================

/// RFC 4122 UUIDs must be accepted.
#[test]
fn accept_rfc4122_uuids() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "550e8400-e29b-41d4-a716-446655440000", "type": "hidden", "squash": "RELU", "bias": 0.1},
            {"uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "550e8400-e29b-41d4-a716-446655440000", "weight": 0.5},
            {"fromUUID": "550e8400-e29b-41d4-a716-446655440000", "toUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson =
        serde_json::from_str(json).expect("RFC 4122 UUIDs must be accepted");

    assert_eq!(creature.neurons.len(), 4);
    assert_eq!(
        creature.neurons[2].uuid,
        "550e8400-e29b-41d4-a716-446655440000"
    );
}

/// `input-N` format neuron UUIDs must be accepted (used for input neurons).
#[test]
fn accept_input_neuron_format() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "hidden-abc123", "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
            {"uuid": "output-def456", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-abc123", "weight": 0.5},
            {"fromUUID": "hidden-abc123", "toUUID": "output-def456", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson =
        serde_json::from_str(json).expect("input-N format UUIDs must be accepted");

    assert_eq!(creature.neurons[0].uuid, "input-0");
    assert_eq!(creature.neurons[1].uuid, "input-1");
}

/// `NeuronData` with RFC UUID must be accepted.
#[test]
fn accept_neuron_data_with_uuid() {
    let json = r#"{"neuron_uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "activation": 0.85, "errors": [0.01]}"#;

    let data: NeuronData =
        serde_json::from_str(json).expect("RFC UUID in NeuronData must be accepted");
    assert_eq!(data.neuron_uuid, "a1b2c3d4-e5f6-4789-abcd-ef0123456789");
}

// ============================================================================
// FFI record_discovery: numeric IDs rejected
// ============================================================================

/// The FFI `record_discovery_internal` must reject numeric neuron IDs
/// and return a structured error response.
#[test]
fn ffi_record_discovery_rejects_numeric_ids() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    let input_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "5", "type": "hidden", "squash": "LOGISTIC", "bias": 0.0}},
                    {{"uuid": "10", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "0", "toUUID": "5", "weight": 0.5}},
                    {{"fromUUID": "5", "toUUID": "10", "weight": 0.8}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "5", "activation": 0.6, "errors": [0.01]}},
                        {{"neuron_uuid": "10", "activation": 0.7, "errors": [-0.05]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let result = neat_ai_discovery::record_discovery_internal(&input_json)
        .expect("internal call must not panic");

    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");
    assert_eq!(
        parsed["success"], false,
        "recording with numeric IDs must fail: {result}"
    );
}

// ============================================================================
// Schema version in responses
// ============================================================================

/// Discovery responses must include a `schemaVersion` field.
#[test]
fn record_discovery_response_includes_schema_version() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    let input_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "input-0", "toUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "weight": 0.5}},
                    {{"fromUUID": "input-1", "toUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "weight": -0.3}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "activation": 0.7, "errors": [-0.05]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let result = neat_ai_discovery::record_discovery_internal(&input_json)
        .expect("internal call must not panic");

    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");
    assert_eq!(parsed["success"], true, "recording must succeed: {result}");
    assert!(
        parsed.get("schemaVersion").is_some(),
        "response must include schemaVersion field: {result}"
    );
    assert!(
        parsed["schemaVersion"].is_string(),
        "schemaVersion must be a string"
    );
}

/// Version response must include schemaVersion.
#[test]
fn get_version_response_includes_schema_version() {
    let result =
        neat_ai_discovery::get_library_version_internal().expect("version call must not panic");

    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");
    assert!(
        parsed.get("schemaVersion").is_some(),
        "version response must include schemaVersion: {result}"
    );
}

// ============================================================================
// Coordinated structural operations use UUID references
// ============================================================================

/// Coordinated structural operations must serialise with UUID-based
/// neuron references (neuronUuid, fromNeuronUuid, toNeuronUuid).
#[test]
fn coordinated_op_serialises_uuid_references() {
    use neat_ai_discovery::CoordinatedStructuralOpJson;

    let op = CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: "a1b2c3d4-e5f6-4789-abcd-ef0123456789".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.1,
        insert_before_neuron_uuid: Some("b2c3d4e5-f6a7-4890-bcde-f01234567890".to_string()),
    };

    let json = serde_json::to_string(&op).expect("serialisation must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // All identity fields must use UUID format
    assert!(
        parsed.get("neuronUuid").is_some(),
        "AddNeuron must have neuronUuid field"
    );
    assert!(
        parsed.get("insertBeforeNeuronUuid").is_some(),
        "AddNeuron must have insertBeforeNeuronUuid field"
    );

    // Verify no numeric-only values in identity fields
    let neuron_uuid = parsed["neuronUuid"].as_str().unwrap();
    assert!(
        !neuron_uuid.chars().all(|c| c.is_ascii_digit()),
        "neuronUuid must not be a purely numeric string"
    );
}
