//! Issue #950 — FFI tests: neuron identity strings in the FFI pipeline.
//!
//! Originally these tests verified that stringified-integer neuron IDs were
//! accepted. Issue #952 tightened the FFI contract to UUID-only: purely
//! numeric IDs are now rejected at the FFI boundary.
//!
//! These tests now verify that:
//! 1. Valid UUID-format neuron IDs work through the full pipeline.
//! 2. Neuron interning handles diverse string formats correctly.
//! 3. `DiscoverRecord` preserves UUID strings without normalisation.

use neat_ai_discovery::{CreatureJson, NeuronData, RecordDiscoveryInput};

// ============================================================================
// FFI deserialisation: valid neuron ID formats
// ============================================================================

/// Creature JSON with RFC 4122 UUID neuron identifiers must deserialise correctly.
#[test]
fn deserialise_creature_with_uuid_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
            {"uuid": "b2c3d4e5-f6a7-4890-bcde-f01234567890", "type": "output", "squash": "LOGISTIC", "bias": -0.2}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "weight": 0.5},
            {"fromUUID": "input-1", "toUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "weight": -0.3},
            {"fromUUID": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "toUUID": "b2c3d4e5-f6a7-4890-bcde-f01234567890", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("UUID IDs must deserialise");

    assert_eq!(creature.neurons.len(), 4);
    assert_eq!(creature.neurons[0].uuid, "input-0");
    assert_eq!(
        creature.neurons[2].uuid,
        "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
    );
    assert_eq!(creature.synapses[0].from_uuid, "input-0");
    assert_eq!(
        creature.synapses[2].to_uuid,
        "b2c3d4e5-f6a7-4890-bcde-f01234567890"
    );
}

/// Creature JSON with mixed UUID styles (RFC 4122 + descriptive) must deserialise.
#[test]
fn deserialise_creature_with_mixed_uuid_formats() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "550e8400-e29b-41d4-a716-446655440000", "type": "hidden", "squash": "RELU", "bias": 0.1},
            {"uuid": "hidden-output-main", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "550e8400-e29b-41d4-a716-446655440000", "weight": 0.5},
            {"fromUUID": "550e8400-e29b-41d4-a716-446655440000", "toUUID": "hidden-output-main", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("mixed UUIDs must deserialise");

    assert_eq!(
        creature.neurons[2].uuid,
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(creature.neurons[3].uuid, "hidden-output-main");
    assert_eq!(
        creature.synapses[0].to_uuid,
        "550e8400-e29b-41d4-a716-446655440000"
    );
}

/// `NeuronData` with UUID `neuron_uuid` must deserialise correctly.
#[test]
fn deserialise_neuron_data_with_uuid_ids() {
    let json = r#"{"neuron_uuid": "a1b2c3d4-e5f6-4789-abcd-ef0123456789", "activation": 0.85, "errors": [0.01, -0.02]}"#;

    let data: NeuronData = serde_json::from_str(json).expect("UUID neuron_uuid must deserialise");

    assert_eq!(data.neuron_uuid, "a1b2c3d4-e5f6-4789-abcd-ef0123456789");
    assert!((data.activation - 0.85).abs() < f32::EPSILON);
}

/// Descriptive string neuron IDs must work.
#[test]
fn deserialise_descriptive_string_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "hidden-layer1-node0", "type": "hidden", "squash": "TANH", "bias": 0.0},
            {"uuid": "output-main", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-layer1-node0", "weight": 1.0},
            {"fromUUID": "hidden-layer1-node0", "toUUID": "output-main", "weight": -0.5}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson =
        serde_json::from_str(json).expect("descriptive string IDs must deserialise");

    assert_eq!(creature.neurons[2].uuid, "hidden-layer1-node0");
    assert_eq!(creature.synapses[1].from_uuid, "hidden-layer1-node0");
}

// ============================================================================
// Recording pipeline: UUID neuron IDs through record + read
// ============================================================================

/// Recording and reading back discovery data with UUID neuron IDs must
/// produce correct, retrievable records.
#[test]
fn record_and_read_with_uuid_neuron_ids() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    let hidden_uuid = "a1b2c3d4-e5f6-4789-abcd-ef0123456789";
    let output_uuid = "b2c3d4e5-f6a7-4890-bcde-f01234567890";

    let input_json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": hidden_uuid, "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
                {"uuid": output_uuid, "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": hidden_uuid, "weight": 0.5},
                {"fromUUID": "input-1", "toUUID": hidden_uuid, "weight": -0.3},
                {"fromUUID": hidden_uuid, "toUUID": output_uuid, "weight": 0.8}
            ],
            "input": 2,
            "output": 1
        },
        "training_data": [
            {
                "input": [0.5, 0.3],
                "output": [0.7],
                "neuron_data": [
                    {"neuron_uuid": hidden_uuid, "activation": 0.6, "errors": [0.01]},
                    {"neuron_uuid": output_uuid, "activation": 0.7, "errors": [-0.05]}
                ]
            },
            {
                "input": [0.1, 0.9],
                "output": [0.4],
                "neuron_data": [
                    {"neuron_uuid": hidden_uuid, "activation": 0.45, "errors": [0.02]},
                    {"neuron_uuid": output_uuid, "activation": 0.38, "errors": [0.03]}
                ]
            }
        ],
        "temp_dir": temp_path
    });

    let input: RecordDiscoveryInput =
        serde_json::from_value(input_json).expect("input must deserialise");

    let result =
        neat_ai_discovery::record::record_discovery_data(&input).expect("recording must succeed");

    let parquet_path = std::path::Path::new(&result.temp_dir).join(&result.file);
    let parquet_str = parquet_path.to_str().unwrap();

    // Read records for hidden neuron UUID
    let records_hidden =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, hidden_uuid)
            .expect("reading records for hidden neuron must succeed");
    assert_eq!(
        records_hidden.len(),
        2,
        "hidden neuron should have 2 records (one per training sample)"
    );
    for r in &records_hidden {
        assert_eq!(r.neuron_uuid, hidden_uuid);
    }

    // Read records for output neuron UUID
    let records_output =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, output_uuid)
            .expect("reading records for output neuron must succeed");
    assert_eq!(records_output.len(), 2);
    for r in &records_output {
        assert_eq!(r.neuron_uuid, output_uuid);
    }

    // Querying a non-existent UUID must return empty, not an error
    let records_missing = neat_ai_discovery::parquet_format::read_records_from_parquet(
        parquet_str,
        "non-existent-uuid",
    )
    .expect("querying non-existent neuron must not error");
    assert!(records_missing.is_empty());
}

/// Recording with descriptive string IDs must correctly associate each
/// record with its neuron.
#[test]
fn record_and_read_with_descriptive_id_formats() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    let rfc_uuid = "a1b2c3d4-e5f6-4789-abcd-ef0123456789";
    let descriptive_id = "output-main";

    let input_json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": rfc_uuid, "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": descriptive_id, "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": rfc_uuid, "weight": 1.0},
                {"fromUUID": rfc_uuid, "toUUID": descriptive_id, "weight": 0.5}
            ],
            "input": 2,
            "output": 1
        },
        "training_data": [
            {
                "input": [0.5, 0.3],
                "output": [0.7],
                "neuron_data": [
                    {"neuron_uuid": rfc_uuid, "activation": 0.6, "errors": [0.01]},
                    {"neuron_uuid": descriptive_id, "activation": 0.7, "errors": [-0.05]}
                ]
            }
        ],
        "temp_dir": temp_path
    });

    let input: RecordDiscoveryInput =
        serde_json::from_value(input_json).expect("descriptive-format input must deserialise");

    let result =
        neat_ai_discovery::record::record_discovery_data(&input).expect("recording must succeed");

    let parquet_path = std::path::Path::new(&result.temp_dir).join(&result.file);
    let parquet_str = parquet_path.to_str().unwrap();

    // Read by RFC UUID
    let records_rfc =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, rfc_uuid)
            .expect("reading by RFC UUID must succeed");
    assert_eq!(records_rfc.len(), 1);
    assert_eq!(records_rfc[0].neuron_uuid, rfc_uuid);

    // Read by descriptive ID
    let records_desc =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, descriptive_id)
            .expect("reading by descriptive ID must succeed");
    assert_eq!(records_desc.len(), 1);
    assert_eq!(records_desc[0].neuron_uuid, descriptive_id);

    // Grouped read must separate by exact string identity
    let grouped =
        neat_ai_discovery::parquet_format::read_all_records_grouped_by_neuron(parquet_str)
            .expect("grouped read must succeed");
    assert!(
        grouped.contains_key(rfc_uuid),
        "grouped records must contain RFC UUID key"
    );
    assert!(
        grouped.contains_key(descriptive_id),
        "grouped records must contain descriptive key"
    );
}

// ============================================================================
// Neuron interning: diverse string IDs
// ============================================================================

/// `NeuronIndex` must correctly intern UUID-format neuron IDs.
#[test]
fn intern_uuid_neuron_ids() {
    use neat_ai_discovery::intern::NeuronIndex;

    let mut index = NeuronIndex::new();

    // Intern UUID-format IDs
    let idx_input = index.intern("input-0");
    let idx_hidden = index.intern("a1b2c3d4-e5f6-4789-abcd-ef0123456789");
    let idx_output = index.intern("output-main");
    let idx_descriptive = index.intern("hidden-layer1-node0");

    // Must assign distinct indices
    assert_ne!(idx_input, idx_hidden);
    assert_ne!(idx_hidden, idx_output);
    assert_ne!(idx_output, idx_descriptive);

    // Round-trip must preserve exact string
    assert_eq!(index.get_uuid(idx_input), Some("input-0"));
    assert_eq!(
        index.get_uuid(idx_hidden),
        Some("a1b2c3d4-e5f6-4789-abcd-ef0123456789")
    );
    assert_eq!(index.get_uuid(idx_output), Some("output-main"));

    // Re-interning same string must return same index
    assert_eq!(
        index.intern("a1b2c3d4-e5f6-4789-abcd-ef0123456789"),
        idx_hidden
    );
}

/// `NeuronIndex` must treat different UUID formats as distinct entries.
#[test]
fn intern_different_uuid_formats_are_distinct() {
    use neat_ai_discovery::intern::NeuronIndex;

    let mut index = NeuronIndex::new();

    let idx_rfc = index.intern("550e8400-e29b-41d4-a716-446655440000");
    let idx_descriptive = index.intern("hidden-42");
    let idx_input = index.intern("input-42");

    // All three must be distinct
    assert_ne!(idx_rfc, idx_descriptive);
    assert_ne!(idx_rfc, idx_input);
    assert_ne!(idx_descriptive, idx_input);

    // Each round-trips correctly
    assert_eq!(
        index.get_uuid(idx_rfc),
        Some("550e8400-e29b-41d4-a716-446655440000")
    );
    assert_eq!(index.get_uuid(idx_descriptive), Some("hidden-42"));
    assert_eq!(index.get_uuid(idx_input), Some("input-42"));
}

// ============================================================================
// DiscoverRecord: UUID neuron IDs
// ============================================================================

/// `DiscoverRecord` must accept UUID-format neuron identifiers.
#[test]
fn discover_record_with_uuid_neuron_id() {
    use neat_ai_discovery::types::DiscoverRecord;

    let record = DiscoverRecord::new(
        0,
        "a1b2c3d4-e5f6-4789-abcd-ef0123456789".to_string(),
        Some(0.5),
        0.6,
        vec![0.01],
    );

    assert_eq!(record.neuron_uuid, "a1b2c3d4-e5f6-4789-abcd-ef0123456789");
    assert_eq!(record.obs_index, 0);
}

/// `DiscoverRecord` must preserve both RFC UUID and descriptive strings
/// without any normalisation or conversion.
#[test]
fn discover_record_preserves_id_format() {
    use neat_ai_discovery::types::DiscoverRecord;

    let descriptive =
        DiscoverRecord::new(0, "hidden-node-7".to_string(), Some(0.5), 0.6, vec![0.01]);
    let rfc = DiscoverRecord::new(
        1,
        "a1b2c3d4-e5f6-4789-abcd-ef0123456789".to_string(),
        Some(0.5),
        0.6,
        vec![0.01],
    );

    // Each format is preserved as-is
    assert_eq!(descriptive.neuron_uuid, "hidden-node-7");
    assert_eq!(rfc.neuron_uuid, "a1b2c3d4-e5f6-4789-abcd-ef0123456789");

    // They are distinct identities
    assert_ne!(descriptive.neuron_uuid, rfc.neuron_uuid);
}

// ============================================================================
// FFI entry point: record_discovery_internal with UUID IDs
// ============================================================================

/// The FFI `record_discovery_internal` entry point must accept UUID neuron
/// IDs and return `success: true`.
#[test]
fn ffi_record_discovery_with_uuid_ids() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    let input_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "hidden-a1b2c3", "type": "hidden", "squash": "LOGISTIC", "bias": 0.0}},
                    {{"uuid": "output-d4e5f6", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "input-0", "toUUID": "hidden-a1b2c3", "weight": 0.5}},
                    {{"fromUUID": "input-1", "toUUID": "hidden-a1b2c3", "weight": -0.3}},
                    {{"fromUUID": "hidden-a1b2c3", "toUUID": "output-d4e5f6", "weight": 0.8}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "hidden-a1b2c3", "activation": 0.6, "errors": [0.01]}},
                        {{"neuron_uuid": "output-d4e5f6", "activation": 0.7, "errors": [-0.05]}}
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
        parsed["success"], true,
        "recording with UUID IDs must succeed: {result}"
    );
}

/// The FFI `read_discovery_records_internal` entry point must accept a UUID
/// `neuron_uuid` for querying.
#[test]
fn ffi_read_discovery_with_uuid_neuron_id() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    // First, record some data with UUID IDs
    let record_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "input-0", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "input-1", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "output-abc123", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "input-0", "toUUID": "output-abc123", "weight": 0.5}},
                    {{"fromUUID": "input-1", "toUUID": "output-abc123", "weight": -0.3}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "output-abc123", "activation": 0.65, "errors": [0.02]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let record_result = neat_ai_discovery::record_discovery_internal(&record_json)
        .expect("recording must not panic");
    let record_parsed: serde_json::Value = serde_json::from_str(&record_result).unwrap();
    assert_eq!(record_parsed["success"], true, "recording must succeed");

    // Now read using UUID neuron ID
    let parquet_file = format!("{temp_path}/discovery_data.parquet");
    let read_json = serde_json::json!({
        "parquet_file": parquet_file,
        "neuron_uuid": "output-abc123"
    });

    let read_result =
        neat_ai_discovery::read_discovery_records(&serde_json::to_string(&read_json).unwrap())
            .expect("read must not panic");

    let read_parsed: serde_json::Value = serde_json::from_str(&read_result).unwrap();
    assert_eq!(
        read_parsed["success"], true,
        "reading with UUID neuron ID must succeed: {read_result}"
    );
}
