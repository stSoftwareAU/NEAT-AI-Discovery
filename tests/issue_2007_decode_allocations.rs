//! Issue #2007: the Parquet read paths decode without per-row allocations.
//!
//! The shared decoder no longer slices a fresh Arrow array per row to read the
//! `errors` list, and the grouped reader no longer clones each row's
//! `neuron_uuid` to key the map. Both are pure performance changes, so these
//! tests pin the *behaviour* that must survive them: every read path must still
//! decode exactly the rows it decoded before, with the same error lists, the
//! same grouping and the same per-neuron ordering.
//!
//! The interleaved fixture is deliberate — a decoder that mis-tracks list
//! offsets or reuses the previous row's group only shows up when consecutive
//! rows belong to different neurons and carry different-length error lists.

use neat_ai_discovery::analysis::cache::StreamingRecordCache;
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_from_parquet, read_all_records_grouped_by_neuron,
    read_all_records_grouped_by_neuron_with_profile, read_records_from_parquet,
    read_records_from_parquet_with_limit_and_profile, write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::NamedTempFile;

const ALPHA: &str = "11111111-1111-1111-1111-111111111111";
const BETA: &str = "22222222-2222-2222-2222-222222222222";

/// Rows alternating between two neurons, with error lists of length 0, 1, 2 and
/// 3 so a mis-computed list offset shifts a neighbouring row's values.
fn interleaved_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord::new(0, ALPHA.to_string(), Some(0.5), 0.75, vec![0.1, 0.2]),
        DiscoverRecord::new(0, BETA.to_string(), None, -0.25, vec![]),
        DiscoverRecord::new(1, ALPHA.to_string(), Some(-1.5), 0.0, vec![0.3]),
        DiscoverRecord::new(1, BETA.to_string(), Some(0.25), 0.5, vec![0.6, 0.7, 0.8]),
        DiscoverRecord::new(2, ALPHA.to_string(), Some(2.5), 1.5, vec![]),
        DiscoverRecord::new(2, BETA.to_string(), Some(-0.75), -1.0, vec![0.9, 1.0]),
    ]
}

fn write_interleaved_parquet() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".parquet").expect("temp file");
    write_records_to_parquet(
        file.path().to_str().expect("utf-8 path"),
        &interleaved_records(),
    )
    .expect("write");
    file
}

/// The rows of `interleaved_records` belonging to `neuron`, in file order.
fn expected_for(neuron: &str) -> Vec<DiscoverRecord> {
    interleaved_records()
        .into_iter()
        .filter(|record| record.neuron_uuid == neuron)
        .collect()
}

#[test]
fn grouping_keeps_every_row_with_its_own_neuron_and_error_list() {
    let file = write_interleaved_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let grouped = read_all_records_grouped_by_neuron(path).expect("read grouped");

    assert_eq!(grouped.len(), 2, "two distinct neurons were written");
    assert_eq!(grouped[ALPHA], expected_for(ALPHA));
    assert_eq!(grouped[BETA], expected_for(BETA));
}

#[test]
fn every_read_path_decodes_the_same_rows() {
    let file = write_interleaved_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let all = read_all_records_from_parquet(path).expect("read all");
    assert_eq!(all, interleaved_records(), "unfiltered read is file order");

    let filtered = read_records_from_parquet(path, BETA).expect("read filtered");
    assert_eq!(filtered, expected_for(BETA), "the UUID filter path agrees");

    let limited = read_records_from_parquet_with_limit_and_profile(path, None, ColumnProfile::Full)
        .expect("read limited");
    assert_eq!(limited, interleaved_records(), "the limit path agrees");
}

#[test]
fn the_observation_limit_keeps_whole_observations() {
    let file = write_interleaved_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let limited =
        read_records_from_parquet_with_limit_and_profile(path, Some(2), ColumnProfile::Full)
            .expect("read limited");

    let expected: Vec<DiscoverRecord> = interleaved_records()
        .into_iter()
        .filter(|record| record.obs_index < 2)
        .collect();
    assert_eq!(limited, expected, "both rows of each accepted observation");
}

#[test]
fn projecting_errors_away_yields_empty_lists_not_shifted_ones() {
    let file = write_interleaved_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    let grouped =
        read_all_records_grouped_by_neuron_with_profile(path, ColumnProfile::WithoutErrors)
            .expect("read without errors");

    for neuron in [ALPHA, BETA] {
        let decoded = &grouped[neuron];
        let expected = expected_for(neuron);
        assert_eq!(decoded.len(), expected.len());
        for (decoded, expected) in decoded.iter().zip(expected.iter()) {
            assert!(decoded.errors.is_empty(), "errors are projected away");
            assert_eq!(decoded.obs_index, expected.obs_index);
            assert_eq!(decoded.neuron_uuid, expected.neuron_uuid);
            assert_eq!(decoded.value, expected.value);
        }
    }
}

#[test]
fn the_streaming_block_loader_decodes_the_same_rows() {
    let file = write_interleaved_parquet();
    let path = file.path().to_str().expect("utf-8 path");

    // One block per two rows, so ALPHA's rows span every block.
    let cache = StreamingRecordCache::new(path, Some(2), Some(1)).expect("streaming cache");

    assert_eq!(*cache.get(ALPHA).expect("alpha"), expected_for(ALPHA));
    assert_eq!(*cache.get(BETA).expect("beta"), expected_for(BETA));
}
