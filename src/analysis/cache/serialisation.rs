//! Binary serialisation and LZ4 compression for discovery records (Issue #420, #484).
//!
//! Provides compact binary serialisation of `DiscoverRecord` values for use with
//! LZ4-compressed caching. Defensive deserialisation returns errors on truncated
//! or malformed data rather than panicking.

use crate::types::DiscoverRecord;
use anyhow::{Context, Result, bail};

/// Serialise records to a compact binary format for LZ4 compression.
///
/// Format per record:
/// - obs_index: u32 (4 bytes)
/// - uuid_len: u16 (2 bytes)
/// - uuid: [u8; uuid_len]
/// - has_value: u8 (1 byte)
/// - value: f32 (4 bytes, only if has_value)
/// - activation: f32 (4 bytes)
/// - errors_len: u16 (2 bytes)
/// - errors: [f32; errors_len]
pub(crate) fn serialise_records(records: &[DiscoverRecord]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(records.len() * 32);
    for r in records {
        buf.extend_from_slice(&r.obs_index.to_le_bytes());
        let uuid_bytes = r.neuron_uuid.as_bytes();
        buf.extend_from_slice(&(uuid_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(uuid_bytes);
        match r.value {
            Some(v) => {
                buf.push(1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            None => {
                buf.push(0);
            }
        }
        buf.extend_from_slice(&r.activation.to_le_bytes());
        buf.extend_from_slice(&(r.errors.len() as u16).to_le_bytes());
        for &e in &r.errors {
            buf.extend_from_slice(&e.to_le_bytes());
        }
    }
    buf
}

/// Deserialise records from the compact binary format.
///
/// Returns `Err` if the data is truncated or malformed, allowing the caller
/// to fall back to a fresh parquet read rather than panicking (Issue #484).
pub(crate) fn deserialise_records(data: &[u8]) -> Result<Vec<DiscoverRecord>> {
    let mut records = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        // obs_index: u32 (4 bytes)
        let end = pos + 4;
        if end > data.len() {
            bail!("Cache truncated at offset {pos}: expected 4 bytes for obs_index");
        }
        let obs_index = u32::from_le_bytes(
            data[pos..end]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Cache byte conversion failed at offset {pos}"))?,
        );
        pos = end;

        // uuid_len: u16 (2 bytes)
        let end = pos + 2;
        if end > data.len() {
            bail!("Cache truncated at offset {pos}: expected 2 bytes for uuid_len");
        }
        let uuid_len = u16::from_le_bytes(
            data[pos..end]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Cache byte conversion failed at offset {pos}"))?,
        ) as usize;
        pos = end;

        // uuid: [u8; uuid_len]
        if pos + uuid_len > data.len() {
            bail!("Cache truncated at offset {pos}: expected {uuid_len} bytes for UUID");
        }
        let neuron_uuid = String::from_utf8_lossy(&data[pos..pos + uuid_len]).to_string();
        pos += uuid_len;

        // has_value: u8 (1 byte)
        if pos >= data.len() {
            bail!("Cache truncated at offset {pos}: expected 1 byte for has_value flag");
        }
        let has_value = data[pos];
        pos += 1;

        // value: f32 (4 bytes, only if has_value == 1)
        let value = if has_value == 1 {
            let end = pos + 4;
            if end > data.len() {
                bail!("Cache truncated at offset {pos}: expected 4 bytes for value");
            }
            let v =
                f32::from_le_bytes(data[pos..end].try_into().map_err(|_| {
                    anyhow::anyhow!("Cache byte conversion failed at offset {pos}")
                })?);
            pos = end;
            Some(v)
        } else {
            None
        };

        // activation: f32 (4 bytes)
        let end = pos + 4;
        if end > data.len() {
            bail!("Cache truncated at offset {pos}: expected 4 bytes for activation");
        }
        let activation = f32::from_le_bytes(
            data[pos..end]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Cache byte conversion failed at offset {pos}"))?,
        );
        pos = end;

        // errors_len: u16 (2 bytes)
        let end = pos + 2;
        if end > data.len() {
            bail!("Cache truncated at offset {pos}: expected 2 bytes for errors_len");
        }
        let errors_len = u16::from_le_bytes(
            data[pos..end]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Cache byte conversion failed at offset {pos}"))?,
        ) as usize;
        pos = end;

        // errors: [f32; errors_len]
        let errors_bytes = errors_len * 4;
        if pos + errors_bytes > data.len() {
            bail!(
                "Cache truncated at offset {pos}: expected {errors_bytes} bytes for {errors_len} errors"
            );
        }
        let mut errors = Vec::with_capacity(errors_len);
        for i in 0..errors_len {
            let end = pos + 4;
            errors.push(f32::from_le_bytes(data[pos..end].try_into().map_err(
                |_| {
                    anyhow::anyhow!(
                        "Cache byte conversion failed at offset {pos} (error index {i})"
                    )
                },
            )?));
            pos = end;
        }

        records.push(DiscoverRecord::new(
            obs_index,
            neuron_uuid,
            value,
            activation,
            errors,
        ));
    }
    Ok(records)
}

/// A compressed entry in the LRU cache.
///
/// Records are serialised and LZ4-compressed when stored, then decompressed
/// on access. This trades CPU time for memory, allowing more neurons to fit
/// in the cache before eviction is needed.
pub(crate) struct CompressedCacheEntry {
    /// LZ4-compressed serialised record data.
    pub(crate) compressed_data: Vec<u8>,
    /// Number of records (for quick stats without decompression).
    #[allow(dead_code)]
    pub(crate) record_count: usize,
    /// Size of the compressed data in bytes (what we actually store).
    pub(crate) compressed_size: usize,
    /// Last access time for LRU ordering.
    pub(crate) last_access: std::time::Instant,
}

impl CompressedCacheEntry {
    pub(crate) fn new(records: &[DiscoverRecord]) -> Self {
        let serialised = serialise_records(records);
        let compressed = lz4_flex::compress_prepend_size(&serialised);
        let compressed_size = compressed.len();
        Self {
            compressed_data: compressed,
            record_count: records.len(),
            compressed_size,
            last_access: std::time::Instant::now(),
        }
    }

    pub(crate) fn decompress(&self) -> Result<Vec<DiscoverRecord>> {
        let decompressed = lz4_flex::decompress_size_prepended(&self.compressed_data)
            .map_err(|e| anyhow::anyhow!("LZ4 decompression failed — cache corruption: {e}"))?;
        deserialise_records(&decompressed).context("Failed to deserialise cached records")
    }

    pub(crate) fn touch(&mut self) {
        self.last_access = std::time::Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Issue #484: Defensive binary deserialisation tests

    #[test]
    fn deserialise_records_round_trip() {
        let records = vec![
            DiscoverRecord::new(0, "uuid-a".to_string(), Some(1.5), 0.8, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "uuid-b".to_string(), None, 0.3, vec![]),
            DiscoverRecord::new(
                2,
                "uuid-c".to_string(),
                Some(-0.5),
                1.0,
                vec![0.4, 0.5, 0.6],
            ),
        ];
        let serialised = serialise_records(&records);
        let result = deserialise_records(&serialised);
        assert!(result.is_ok(), "Valid data should deserialise successfully");
        let deserialized = result.unwrap();
        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized[0].obs_index, 0);
        assert_eq!(deserialized[0].neuron_uuid, "uuid-a");
        assert_eq!(deserialized[0].value, Some(1.5));
        assert_eq!(deserialized[0].activation, 0.8);
        assert_eq!(deserialized[0].errors, vec![0.1, 0.2]);
        assert_eq!(deserialized[1].value, None);
        assert_eq!(deserialized[2].errors.len(), 3);
    }

    #[test]
    fn deserialise_records_empty_data() {
        let result = deserialise_records(&[]);
        assert!(result.is_ok(), "Empty data should return Ok with empty vec");
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn deserialise_records_truncated_at_obs_index() {
        // Only 2 bytes instead of the 4 needed for obs_index
        let result = deserialise_records(&[0x01, 0x00]);
        assert!(result.is_err(), "Truncated obs_index should return Err");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("truncated") || err_msg.contains("offset"),
            "Error should mention truncation: {err_msg}"
        );
    }

    #[test]
    fn deserialise_records_truncated_at_uuid() {
        // Valid obs_index (4 bytes) + uuid_len=10 (2 bytes) but no uuid data
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // obs_index
        data.extend_from_slice(&10u16.to_le_bytes()); // uuid_len = 10
        // Missing: 10 bytes of UUID data

        let result = deserialise_records(&data);
        assert!(result.is_err(), "Truncated UUID should return Err");
    }

    #[test]
    fn deserialise_records_truncated_at_value() {
        // Build a record that is truncated in the value field
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // obs_index
        let uuid = b"test";
        data.extend_from_slice(&(uuid.len() as u16).to_le_bytes());
        data.extend_from_slice(uuid);
        data.push(1); // has_value = true
        // Missing: 4 bytes for value f32

        let result = deserialise_records(&data);
        assert!(result.is_err(), "Truncated value field should return Err");
    }

    #[test]
    fn deserialise_records_truncated_at_activation() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes());
        let uuid = b"test";
        data.extend_from_slice(&(uuid.len() as u16).to_le_bytes());
        data.extend_from_slice(uuid);
        data.push(0); // has_value = false (no value bytes)
        // Missing: 4 bytes for activation f32

        let result = deserialise_records(&data);
        assert!(result.is_err(), "Truncated activation should return Err");
    }

    #[test]
    fn deserialise_records_truncated_at_errors() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes());
        let uuid = b"test";
        data.extend_from_slice(&(uuid.len() as u16).to_le_bytes());
        data.extend_from_slice(uuid);
        data.push(0); // has_value = false
        data.extend_from_slice(&0.5f32.to_le_bytes()); // activation
        data.extend_from_slice(&5u16.to_le_bytes()); // errors_len = 5
        // Missing: 20 bytes (5 × f32) for errors

        let result = deserialise_records(&data);
        assert!(result.is_err(), "Truncated errors array should return Err");
    }

    #[test]
    fn decompress_returns_error_on_corrupted_lz4() {
        let entry = CompressedCacheEntry {
            compressed_data: vec![0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x01],
            record_count: 1,
            compressed_size: 6,
            last_access: std::time::Instant::now(),
        };
        let result = entry.decompress();
        assert!(result.is_err(), "Corrupted LZ4 data should return Err");
    }

    #[test]
    fn decompress_valid_data_round_trip() {
        let records = vec![DiscoverRecord::new(
            0,
            "uuid-x".to_string(),
            Some(1.0),
            0.5,
            vec![0.1],
        )];
        let entry = CompressedCacheEntry::new(&records);
        let result = entry.decompress();
        assert!(result.is_ok(), "Valid compressed data should decompress");
        let decompressed = result.unwrap();
        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].neuron_uuid, "uuid-x");
    }
}
