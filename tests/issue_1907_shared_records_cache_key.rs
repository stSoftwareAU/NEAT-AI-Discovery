//! Issue #1907: the shared-decode cache key must describe the bytes actually
//! cached, and an unknown file identity must never be cacheable.
//!
//! Three properties are exercised here:
//!
//! 1. A replacement file that reproduces the original's `(len, mtime)` is still
//!    a different file — the device/inode in the key keeps it out of the cache.
//! 2. A path that cannot be `stat`ed never seeds the cache, so two consecutive
//!    failures cannot match each other.
//! 3. The once-per-cycle contract from Issue #1406 still holds: an unchanged
//!    file is served from the cache without a second decode.
//!
//! The in-window rewrite case (the file changing between the pre-decode `stat`
//! and the decode) needs to inject into the stat → decode window, so it lives
//! with the module as
//! `shared_records::tests::rewrite_between_stat_and_decode_does_not_serve_stale_identity`.
//!
//! These tests share the process-global cache, so they are `#[serial]`.

use neat_ai_discovery::parquet_format::shared_records::{
    decodes_for_path, invalidate, load_grouped_records_shared,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use std::fs;
use std::sync::Arc;
use tempfile::NamedTempFile;

fn sample_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord::new(0, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1]),
        DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.2]),
        DiscoverRecord::new(0, "neuron-2".to_string(), Some(0.3), 0.5, vec![0.05]),
    ]
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

/// A different file that happens to share the original's byte length and
/// modification time must not be served from the cache: the key carries the
/// device/inode, so the replacement is a miss.
#[test]
#[serial]
fn replacement_file_with_matching_len_and_mtime_is_not_a_cache_hit() {
    let tmp = write_temp_parquet(&sample_records());
    let path = tmp.path().to_str().unwrap().to_string();
    invalidate();

    let first = load_grouped_records_shared(&path, None).expect("first load decodes");
    assert_eq!(decodes_for_path(&path), 1, "first load should decode once");

    // Build a distinct file (new inode) with byte-identical content, stamp it
    // with the original's mtime, and swap it in. Path, length and mtime all
    // still match — only the inode differs.
    let original = fs::metadata(&path).expect("stat original");
    let mtime = original.modified().expect("original mtime");
    let replacement = tmp.path().with_extension("replacement");
    fs::copy(&path, &replacement).expect("copy parquet");
    fs::File::options()
        .write(true)
        .open(&replacement)
        .expect("open replacement")
        .set_modified(mtime)
        .expect("stamp replacement mtime");
    fs::rename(&replacement, &path).expect("swap replacement in");

    let swapped = fs::metadata(&path).expect("stat replacement");
    assert_eq!(
        swapped.len(),
        original.len(),
        "replacement length must match"
    );
    assert_eq!(
        swapped.modified().expect("replacement mtime"),
        mtime,
        "replacement mtime must match",
    );

    let second = load_grouped_records_shared(&path, None).expect("second load decodes");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "a different file must not be served from the previous file's cache entry",
    );
}

/// A path that cannot be `stat`ed is not cacheable: `(path, 0, None)` must never
/// become a hittable key, so two loads of a missing path both miss.
#[test]
#[serial]
fn missing_path_never_seeds_cache() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("does-not-exist.parquet");
    let path = missing.to_str().unwrap();
    invalidate();

    assert!(
        load_grouped_records_shared(path, None).is_err(),
        "loading a missing path must fail",
    );
    assert_eq!(
        decodes_for_path(path),
        0,
        "a failed stat must not seed a cache entry",
    );

    assert!(
        load_grouped_records_shared(path, None).is_err(),
        "the second load of a missing path must also fail — never a cache hit",
    );
    assert_eq!(
        decodes_for_path(path),
        0,
        "two failed stats must not match each other",
    );
}

/// The Issue #1406 once-per-cycle contract: an unchanged file is still served
/// from the cache, and the `decodes` counter still reports a single decode.
#[test]
#[serial]
fn unchanged_file_still_hits_cache() {
    let tmp = write_temp_parquet(&sample_records());
    let path = tmp.path().to_str().unwrap();
    invalidate();

    let first = load_grouped_records_shared(path, None).expect("first load decodes");
    assert_eq!(decodes_for_path(path), 1, "first load should decode once");

    let second = load_grouped_records_shared(path, None).expect("second load hits");
    assert_eq!(
        decodes_for_path(path),
        1,
        "an unchanged file must not decode a second time",
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "the cache hit must hand back the identical shared decode",
    );
    assert_eq!(first.get("neuron-1").expect("neuron-1 present").len(), 2);

    invalidate();
    assert_eq!(decodes_for_path(path), 0, "invalidate clears the slot");
}
