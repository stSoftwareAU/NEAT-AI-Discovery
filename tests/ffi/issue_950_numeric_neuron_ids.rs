//! Issue #950 — FFI tests: do not treat runtime neuron id strings as RFC UUIDs.
//!
//! NEAT-AI public creature exports use stable wire-format strings for neuron
//! identity (`neuron.uuid`, synapse `fromUUID`/`toUUID`). However, TypeScript
//! bridges (e.g. `creatureToRustFormat`) may stringify **numeric runtime ids**
//! for the Rust side. These strings are **not** RFC 4122 UUIDs.
//!
//! These tests verify that the FFI boundary, recording pipeline, and neuron
//! interning all handle stringified-integer neuron identifiers correctly,
//! without assuming a UUID format.

use neat_ai_discovery::{CreatureJson, NeuronData, RecordDiscoveryInput};

// ============================================================================
// FFI deserialisation: stringified-integer neuron IDs
// ============================================================================

/// Creature JSON with purely numeric neuron UUIDs must deserialise correctly.
/// This guards against any future UUID-format validation rejecting runtime ids.
#[test]
fn deserialise_creature_with_numeric_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "2", "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
            {"uuid": "3", "type": "output", "squash": "LOGISTIC", "bias": -0.2}
        ],
        "synapses": [
            {"fromUUID": "0", "toUUID": "2", "weight": 0.5},
            {"fromUUID": "1", "toUUID": "2", "weight": -0.3},
            {"fromUUID": "2", "toUUID": "3", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("numeric IDs must deserialise");

    assert_eq!(creature.neurons.len(), 4);
    assert_eq!(creature.neurons[0].uuid, "0");
    assert_eq!(creature.neurons[2].uuid, "2");
    assert_eq!(creature.synapses[0].from_uuid, "0");
    assert_eq!(creature.synapses[2].to_uuid, "3");
}

/// Creature JSON mixing RFC 4122 UUIDs with numeric IDs must deserialise.
/// This is the realistic scenario after `normaliseCreatureExport`: hidden
/// neurons have RFC UUIDs while TypeScript runtime ids are numeric.
#[test]
fn deserialise_creature_with_mixed_uuid_formats() {
    let json = r#"{
        "neurons": [
            {"uuid": "0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "550e8400-e29b-41d4-a716-446655440000", "type": "hidden", "squash": "RELU", "bias": 0.1},
            {"uuid": "42", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "0", "toUUID": "550e8400-e29b-41d4-a716-446655440000", "weight": 0.5},
            {"fromUUID": "550e8400-e29b-41d4-a716-446655440000", "toUUID": "42", "weight": 0.8}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("mixed IDs must deserialise");

    assert_eq!(
        creature.neurons[2].uuid,
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(creature.neurons[3].uuid, "42");
    assert_eq!(
        creature.synapses[0].to_uuid,
        "550e8400-e29b-41d4-a716-446655440000"
    );
}

/// `NeuronData` with numeric `neuron_uuid` must deserialise correctly.
/// This is the format TypeScript sends for pre-computed activations.
#[test]
fn deserialise_neuron_data_with_numeric_ids() {
    let json = r#"{"neuron_uuid": "7", "activation": 0.85, "errors": [0.01, -0.02]}"#;

    let data: NeuronData =
        serde_json::from_str(json).expect("numeric neuron_uuid must deserialise");

    assert_eq!(data.neuron_uuid, "7");
    assert!((data.activation - 0.85).abs() < f32::EPSILON);
}

/// Large numeric IDs (matching TypeScript's `neuron.id` counter) must work.
/// Runtime IDs can be arbitrarily large integers stringified.
#[test]
fn deserialise_large_numeric_neuron_ids() {
    let json = r#"{
        "neurons": [
            {"uuid": "100000", "type": "input", "squash": "IDENTITY"},
            {"uuid": "100001", "type": "input", "squash": "IDENTITY"},
            {"uuid": "999999", "type": "hidden", "squash": "TANH", "bias": 0.0},
            {"uuid": "1000000", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "100000", "toUUID": "999999", "weight": 1.0},
            {"fromUUID": "999999", "toUUID": "1000000", "weight": -0.5}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson =
        serde_json::from_str(json).expect("large numeric IDs must deserialise");

    assert_eq!(creature.neurons[2].uuid, "999999");
    assert_eq!(creature.synapses[1].from_uuid, "999999");
}

// ============================================================================
// Recording pipeline: numeric neuron IDs through record + read
// ============================================================================

/// Recording and reading back discovery data with purely numeric neuron IDs
/// must produce correct, retrievable records. This exercises the full pipeline:
/// JSON → `RecordDiscoveryInput` → Parquet write → Parquet read.
#[test]
fn record_and_read_with_numeric_neuron_ids() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    let input_json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "2", "type": "hidden", "squash": "LOGISTIC", "bias": 0.1},
                {"uuid": "3", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "0", "toUUID": "2", "weight": 0.5},
                {"fromUUID": "1", "toUUID": "2", "weight": -0.3},
                {"fromUUID": "2", "toUUID": "3", "weight": 0.8}
            ],
            "input": 2,
            "output": 1
        },
        "training_data": [
            {
                "input": [0.5, 0.3],
                "output": [0.7],
                "neuron_data": [
                    {"neuron_uuid": "2", "activation": 0.6, "errors": [0.01]},
                    {"neuron_uuid": "3", "activation": 0.7, "errors": [-0.05]}
                ]
            },
            {
                "input": [0.1, 0.9],
                "output": [0.4],
                "neuron_data": [
                    {"neuron_uuid": "2", "activation": 0.45, "errors": [0.02]},
                    {"neuron_uuid": "3", "activation": 0.38, "errors": [0.03]}
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

    // Read records for numeric neuron ID "2"
    let records_2 = neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, "2")
        .expect("reading records for neuron '2' must succeed");
    assert_eq!(
        records_2.len(),
        2,
        "neuron '2' should have 2 records (one per training sample)"
    );
    for r in &records_2 {
        assert_eq!(r.neuron_uuid, "2");
    }

    // Read records for numeric neuron ID "3"
    let records_3 = neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, "3")
        .expect("reading records for neuron '3' must succeed");
    assert_eq!(records_3.len(), 2);
    for r in &records_3 {
        assert_eq!(r.neuron_uuid, "3");
    }

    // Querying a non-existent numeric ID must return empty, not an error
    let records_99 =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, "99")
            .expect("querying non-existent neuron must not error");
    assert!(records_99.is_empty());
}

/// Recording with mixed UUID formats (RFC 4122 + numeric) must correctly
/// associate each record with its neuron. This is the realistic scenario
/// after `normaliseCreatureExport`.
#[test]
fn record_and_read_with_mixed_id_formats() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    let rfc_uuid = "a1b2c3d4-e5f6-4789-abcd-ef0123456789";

    let input_json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": rfc_uuid, "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "42", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "0", "toUUID": rfc_uuid, "weight": 1.0},
                {"fromUUID": rfc_uuid, "toUUID": "42", "weight": 0.5}
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
                    {"neuron_uuid": "42", "activation": 0.7, "errors": [-0.05]}
                ]
            }
        ],
        "temp_dir": temp_path
    });

    let input: RecordDiscoveryInput =
        serde_json::from_value(input_json).expect("mixed-format input must deserialise");

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

    // Read by numeric ID
    let records_42 =
        neat_ai_discovery::parquet_format::read_records_from_parquet(parquet_str, "42")
            .expect("reading by numeric ID must succeed");
    assert_eq!(records_42.len(), 1);
    assert_eq!(records_42[0].neuron_uuid, "42");

    // Grouped read must separate by exact string identity
    let grouped =
        neat_ai_discovery::parquet_format::read_all_records_grouped_by_neuron(parquet_str)
            .expect("grouped read must succeed");
    assert!(
        grouped.contains_key(rfc_uuid),
        "grouped records must contain RFC UUID key"
    );
    assert!(
        grouped.contains_key("42"),
        "grouped records must contain numeric key '42'"
    );
}

// ============================================================================
// Neuron interning: numeric string IDs
// ============================================================================

/// `NeuronIndex` must correctly intern stringified-integer neuron IDs.
/// These are NOT parsed to integers — they are opaque strings.
#[test]
fn intern_numeric_neuron_ids() {
    use neat_ai_discovery::intern::NeuronIndex;

    let mut index = NeuronIndex::new();

    // Intern purely numeric IDs
    let idx_0 = index.intern("0");
    let idx_1 = index.intern("1");
    let idx_42 = index.intern("42");
    let idx_999 = index.intern("999999");

    // Must assign distinct indices
    assert_ne!(idx_0, idx_1);
    assert_ne!(idx_1, idx_42);
    assert_ne!(idx_42, idx_999);

    // Round-trip must preserve exact string
    assert_eq!(index.get_uuid(idx_0), Some("0"));
    assert_eq!(index.get_uuid(idx_42), Some("42"));
    assert_eq!(index.get_uuid(idx_999), Some("999999"));

    // Re-interning same string must return same index
    assert_eq!(index.intern("42"), idx_42);
}

/// `NeuronIndex` must treat "42" (numeric) and RFC UUIDs as distinct entries.
/// No implicit format conversion should occur.
#[test]
fn intern_mixed_format_ids_are_distinct() {
    use neat_ai_discovery::intern::NeuronIndex;

    let mut index = NeuronIndex::new();

    let idx_numeric = index.intern("42");
    let idx_rfc = index.intern("550e8400-e29b-41d4-a716-446655440000");
    let idx_prefixed = index.intern("hidden-42");

    // All three must be distinct
    assert_ne!(idx_numeric, idx_rfc);
    assert_ne!(idx_numeric, idx_prefixed);
    assert_ne!(idx_rfc, idx_prefixed);

    // Each round-trips correctly
    assert_eq!(index.get_uuid(idx_numeric), Some("42"));
    assert_eq!(
        index.get_uuid(idx_rfc),
        Some("550e8400-e29b-41d4-a716-446655440000")
    );
    assert_eq!(index.get_uuid(idx_prefixed), Some("hidden-42"));
}

// ============================================================================
// DiscoverRecord: numeric neuron IDs
// ============================================================================

/// `DiscoverRecord` must accept numeric string neuron UUIDs.
/// This is the core data type that flows through the pipeline.
#[test]
fn discover_record_with_numeric_neuron_id() {
    use neat_ai_discovery::types::DiscoverRecord;

    let record = DiscoverRecord::new(0, "42".to_string(), Some(0.5), 0.6, vec![0.01]);

    assert_eq!(record.neuron_uuid, "42");
    assert_eq!(record.obs_index, 0);
}

/// `DiscoverRecord` must preserve both numeric and RFC UUID strings
/// without any normalisation or conversion.
#[test]
fn discover_record_preserves_id_format() {
    use neat_ai_discovery::types::DiscoverRecord;

    let numeric = DiscoverRecord::new(0, "7".to_string(), Some(0.5), 0.6, vec![0.01]);
    let rfc = DiscoverRecord::new(
        1,
        "a1b2c3d4-e5f6-4789-abcd-ef0123456789".to_string(),
        Some(0.5),
        0.6,
        vec![0.01],
    );

    // Each format is preserved as-is
    assert_eq!(numeric.neuron_uuid, "7");
    assert_eq!(rfc.neuron_uuid, "a1b2c3d4-e5f6-4789-abcd-ef0123456789");

    // They are distinct identities
    assert_ne!(numeric.neuron_uuid, rfc.neuron_uuid);
}

// ============================================================================
// FFI entry point: record_discovery_internal with numeric IDs
// ============================================================================

/// The FFI `record_discovery_internal` entry point must accept numeric neuron
/// IDs and return `success: true`. This tests the full JSON-in → JSON-out path.
#[test]
fn ffi_record_discovery_with_numeric_ids() {
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
                    {{"fromUUID": "1", "toUUID": "5", "weight": -0.3}},
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
        parsed["success"], true,
        "recording with numeric IDs must succeed: {result}"
    );
}

/// The FFI `read_discovery_records_internal` entry point must accept a numeric
/// `neuron_uuid` for querying.
#[test]
fn ffi_read_discovery_with_numeric_neuron_id() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    // First, record some data with numeric IDs
    let record_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "0", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "1", "type": "input", "squash": "IDENTITY", "bias": 0.0}},
                    {{"uuid": "77", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "0", "toUUID": "77", "weight": 0.5}},
                    {{"fromUUID": "1", "toUUID": "77", "weight": -0.3}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "77", "activation": 0.65, "errors": [0.02]}}
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

    // Now read using numeric neuron ID
    let parquet_file = format!("{temp_path}/discovery_data.parquet");
    let read_json = serde_json::json!({
        "parquet_file": parquet_file,
        "neuron_uuid": "77"
    });

    let read_result =
        neat_ai_discovery::read_discovery_records(&serde_json::to_string(&read_json).unwrap())
            .expect("read must not panic");

    let read_parsed: serde_json::Value = serde_json::from_str(&read_result).unwrap();
    assert_eq!(
        read_parsed["success"], true,
        "reading with numeric neuron ID must succeed: {read_result}"
    );
}
