//! Schema definitions and validation for discovery Parquet records

use anyhow::Result;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub(crate) const MAX_ARROW_OFFSET: usize = i32::MAX as usize;
pub(crate) const MIN_NEURON_UUID_LENGTH: usize = 1;
pub(crate) const MAX_NEURON_UUID_LENGTH: usize = 100;
const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parquet schema for discovery records
pub fn create_schema() -> Schema {
    Schema::new(vec![
        Field::new("obs_index", DataType::UInt32, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
        Field::new("value", DataType::Float32, true), // nullable
        Field::new("activation", DataType::Float32, false),
        Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ])
}

pub(crate) fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }

    if max_bytes == 0 {
        return "...".to_string();
    }

    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }

    if end == 0 {
        return "...".to_string();
    }

    let slice = &value[..end];
    format!("{slice}...")
}

pub(crate) fn validate_neuron_uuid(uuid: &str) -> Result<()> {
    let len = uuid.len();
    let preview = truncate_utf8(uuid, 120);

    if len < MIN_NEURON_UUID_LENGTH {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" ({len} characters). UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    if len > MAX_NEURON_UUID_LENGTH {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" ({len} characters). UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    if let Some((index, ch)) = uuid
        .chars()
        .enumerate()
        .find(|(_, c)| !matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '-'))
    {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" because it contains an invalid character '{ch}' at position {index}. UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    Ok(())
}
