//! Binary serialisation and LZ4 compression for discovery records (Issue #420, #484).
//!
//! Provides compact binary serialisation of `DiscoverRecord` values for use with
//! LZ4-compressed caching. Defensive deserialisation returns errors on truncated
//! or malformed data rather than panicking.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::types::DiscoverRecord;
use anyhow::{Context, Result, bail};

/// Append a `u16` little-endian length prefix, guarding against overflow.
///
/// The wire format stores `uuid` and `errors` lengths as `u16`. A bare
/// `len as u16` cast silently truncates any length ≥ 65 536, which would make
/// the deserialiser read back the wrong number of bytes and mis-parse every
/// subsequent record — silent cache corruption. Returning an error instead
/// keeps the on-disk format unchanged while refusing to write a value that
/// cannot round-trip (Issue #1366).
fn push_len_u16(buf: &mut Vec<u8>, len: usize, field: &str) -> Result<()> {
    if len > u16::MAX as usize {
        bail!(
            "Cannot serialise {field}: length {len} exceeds u16 maximum {} — would overflow the length prefix and corrupt the cache",
            u16::MAX
        );
    }
    buf.extend_from_slice(&(len as u16).to_le_bytes());
    Ok(())
}

/// Serialise records to a compact binary format for LZ4 compression.
///
/// Format per record:
/// - `obs_index`: u32 (4 bytes)
/// - `uuid_len`: u16 (2 bytes)
/// - uuid: [u8; `uuid_len`]
/// - `has_value`: u8 (1 byte)
/// - value: f32 (4 bytes, only if `has_value`)
/// - activation: f32 (4 bytes)
/// - `errors_len`: u16 (2 bytes)
/// - errors: [f32; `errors_len`]
///
/// Returns `Err` if a `uuid` or `errors` length exceeds `u16::MAX`, rather than
/// silently truncating the length prefix and corrupting the cache (Issue #1366).
pub(crate) fn serialise_records(records: &[DiscoverRecord]) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(records.len() * 32);
    for r in records {
        buf.extend_from_slice(&r.obs_index.to_le_bytes());
        let uuid_bytes = r.neuron_uuid.as_bytes();
        push_len_u16(&mut buf, uuid_bytes.len(), "neuron_uuid")?;
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
        push_len_u16(&mut buf, r.errors.len(), "errors")?;
        for &e in &r.errors {
            buf.extend_from_slice(&e.to_le_bytes());
        }
    }
    Ok(buf)
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
        let neuron_uuid = String::from_utf8_lossy(&data[pos..pos + uuid_len]).into_owned();
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
    /// Size of the compressed data in bytes (what we actually store).
    pub(crate) compressed_size: usize,
    /// Last access time for LRU ordering.
    pub(crate) last_access: std::time::Instant,
}

impl CompressedCacheEntry {
    pub(crate) fn new(records: &[DiscoverRecord]) -> Result<Self> {
        let serialised = serialise_records(records)?;
        let compressed = lz4_flex::compress_prepend_size(&serialised);
        let compressed_size = compressed.len();
        Ok(Self {
            compressed_data: compressed,
            compressed_size,
            last_access: std::time::Instant::now(),
        })
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
        let serialised = serialise_records(&records).expect("normal records should serialise");
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
    fn deserialise_records_invalid_utf8_uuid_uses_lossy_replacement() {
        // Issue #1456: a UUID field carrying invalid UTF-8 bytes drives
        // String::from_utf8_lossy into the Cow::Owned branch. into_owned()
        // must yield the same lossy-decoded value as before.
        let invalid_uuid = [0x66, 0x6f, 0x6f, 0xff, 0x62, 0x61, 0x72]; // "foo\u{FFFD}bar"
        let mut data = Vec::new();
        data.extend_from_slice(&7u32.to_le_bytes()); // obs_index
        data.extend_from_slice(&(invalid_uuid.len() as u16).to_le_bytes()); // uuid_len
        data.extend_from_slice(&invalid_uuid); // uuid (invalid UTF-8)
        data.push(0); // has_value = false
        data.extend_from_slice(&0.5f32.to_le_bytes()); // activation
        data.extend_from_slice(&0u16.to_le_bytes()); // errors_len = 0

        let result = deserialise_records(&data).expect("invalid UTF-8 UUID should still decode");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].neuron_uuid,
            String::from_utf8_lossy(&invalid_uuid),
            "neuron_uuid should match the lossy-decoded UUID with the replacement character"
        );
        assert!(result[0].neuron_uuid.contains('\u{FFFD}'));
        assert_eq!(result[0].obs_index, 7);
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
            compressed_size: 6,
            last_access: std::time::Instant::now(),
        };
        let result = entry.decompress();
        assert!(result.is_err(), "Corrupted LZ4 data should return Err");
    }

    // Issue #1366: Guard against u16 length-prefix overflow.

    #[test]
    fn serialise_records_rejects_oversize_errors() {
        // An errors vector longer than u16::MAX cannot have its length encoded
        // in the 2-byte prefix and must be rejected rather than truncated.
        let oversize = vec![0.0f32; u16::MAX as usize + 1];
        let records = vec![DiscoverRecord::new(
            0,
            "uuid".to_string(),
            None,
            0.5,
            oversize,
        )];
        let result = serialise_records(&records);
        assert!(
            result.is_err(),
            "Over-length errors vector must return Err, not a mis-deserialising buffer"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("errors") && err.contains("u16"),
            "Error should explain the errors length prefix overflow: {err}"
        );
    }

    #[test]
    fn serialise_records_rejects_oversize_uuid() {
        // A UUID longer than u16::MAX bytes overflows the 2-byte prefix.
        let oversize_uuid = "a".repeat(u16::MAX as usize + 1);
        let records = vec![DiscoverRecord::new(0, oversize_uuid, None, 0.5, vec![0.1])];
        let result = serialise_records(&records);
        assert!(
            result.is_err(),
            "Over-length UUID must return Err, not a mis-deserialising buffer"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("neuron_uuid") && err.contains("u16"),
            "Error should explain the uuid length prefix overflow: {err}"
        );
    }

    #[test]
    fn serialise_records_max_length_boundary_is_accepted() {
        // Exactly u16::MAX still fits the prefix and must round-trip cleanly.
        let max_errors = vec![0.0f32; u16::MAX as usize];
        let records = vec![DiscoverRecord::new(
            0,
            "uuid".to_string(),
            None,
            0.5,
            max_errors,
        )];
        let serialised =
            serialise_records(&records).expect("u16::MAX-length errors must serialise");
        let deserialised = deserialise_records(&serialised).expect("must round-trip");
        assert_eq!(deserialised.len(), 1);
        assert_eq!(deserialised[0].errors.len(), u16::MAX as usize);
    }

    #[test]
    fn serialise_records_normal_sizes_are_byte_identical() {
        // Normal-sized records must serialise to exactly the bytes the wire
        // format defines, confirming the guard left the format unchanged.
        let records = vec![DiscoverRecord::new(
            7,
            "ab".to_string(),
            Some(1.5),
            0.25,
            vec![0.5, -0.5],
        )];
        let serialised = serialise_records(&records).expect("normal records should serialise");

        let mut expected = Vec::new();
        expected.extend_from_slice(&7u32.to_le_bytes()); // obs_index
        expected.extend_from_slice(&2u16.to_le_bytes()); // uuid_len
        expected.extend_from_slice(b"ab"); // uuid
        expected.push(1); // has_value
        expected.extend_from_slice(&1.5f32.to_le_bytes()); // value
        expected.extend_from_slice(&0.25f32.to_le_bytes()); // activation
        expected.extend_from_slice(&2u16.to_le_bytes()); // errors_len
        expected.extend_from_slice(&0.5f32.to_le_bytes());
        expected.extend_from_slice(&(-0.5f32).to_le_bytes());

        assert_eq!(
            serialised, expected,
            "Wire format must be byte-identical for normal-sized records"
        );
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
        let entry = CompressedCacheEntry::new(&records).expect("normal records should compress");
        let result = entry.decompress();
        assert!(result.is_ok(), "Valid compressed data should decompress");
        let decompressed = result.unwrap();
        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].neuron_uuid, "uuid-x");
    }
}
