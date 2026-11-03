//! Regression coverage for merge temporary-path collisions (Issue #2450).

use neat_ai_discovery::parquet_format::{
    merge_parquet_files, read_all_records_from_parquet, write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn fixture() -> (TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().expect("temporary directory");
    let output = dir.path().join("merged.parquet");
    let input = dir.path().join("input.parquet");
    let records = vec![DiscoverRecord::new(
        7,
        "neuron-a".to_string(),
        Some(0.5),
        0.7,
        vec![0.1, 0.2],
    )];
    write_records_to_parquet(input.to_str().expect("UTF-8 path"), &records)
        .expect("write input recording");
    (dir, output, input)
}

fn path_string(path: &Path) -> String {
    path.to_str().expect("UTF-8 path").to_owned()
}

#[test]
fn input_at_legacy_temp_path_is_refused_without_changing_bytes() {
    let (_dir, output, input) = fixture();
    fs::copy(&input, &output).expect("seed destination recording");
    let original_output = fs::read(&output).expect("read original destination");
    let legacy_temp = PathBuf::from(format!("{}.tmp", output.display()));
    fs::copy(&input, &legacy_temp).expect("copy valid input to legacy temp path");
    let original = fs::read(&legacy_temp).expect("read original bytes");

    let error = merge_parquet_files(&path_string(&output), &[path_string(&legacy_temp)])
        .expect_err("legacy temporary path input must be rejected");

    assert!(
        format!("{error:#}").contains("temporary"),
        "error should identify the temporary-path alias: {error:#}"
    );
    assert_eq!(fs::read(&legacy_temp).expect("re-read input"), original);
    assert_eq!(
        fs::read(&output).expect("re-read destination"),
        original_output
    );
}

#[cfg(unix)]
#[test]
fn pre_existing_temp_symlink_and_target_are_unchanged() {
    use std::os::unix::fs::symlink;

    let (dir, output, input) = fixture();
    let victim = dir.path().join("victim.txt");
    fs::write(&victim, b"victim!!").expect("write victim");
    let original = fs::read(&victim).expect("read victim");
    let legacy_temp = PathBuf::from(format!("{}.tmp", output.display()));
    symlink(&victim, &legacy_temp).expect("create legacy temp symlink");

    merge_parquet_files(&path_string(&output), &[path_string(&input)])
        .expect("merge succeeds without using occupied legacy temp path");

    assert_eq!(fs::read(&victim).expect("re-read victim"), original);
    assert!(
        fs::symlink_metadata(&legacy_temp)
            .expect("inspect legacy temp")
            .file_type()
            .is_symlink(),
        "legacy temp symlink must remain in place"
    );
}

#[test]
fn occupied_regular_temp_path_is_preserved_during_successful_merge() {
    let (_dir, output, input) = fixture();
    let legacy_temp = PathBuf::from(format!("{}.tmp", output.display()));
    fs::write(&legacy_temp, b"occupied temp bytes").expect("write occupied path");
    let original = fs::read(&legacy_temp).expect("read occupied path");

    merge_parquet_files(&path_string(&output), &[path_string(&input)])
        .expect("merge succeeds with a unique temporary path");

    assert_eq!(
        fs::read(&legacy_temp).expect("re-read occupied path"),
        original
    );
    let merged = read_all_records_from_parquet(&path_string(&output)).expect("read merged output");
    assert_eq!(merged.len(), 1);
}

#[test]
fn failed_merge_preserves_occupied_temp_path_and_cleans_its_own_temp() {
    let (dir, output, input) = fixture();
    let legacy_temp = PathBuf::from(format!("{}.tmp", output.display()));
    fs::write(&legacy_temp, b"keep this file").expect("write occupied path");
    let original = fs::read(&legacy_temp).expect("read occupied path");
    let missing = dir.path().join("missing.parquet");

    let error = merge_parquet_files(
        &path_string(&output),
        &[path_string(&input), path_string(&missing)],
    )
    .expect_err("missing input must fail the merge");

    assert!(format!("{error:#}").contains("missing.parquet"));
    assert_eq!(
        fs::read(&legacy_temp).expect("re-read occupied path"),
        original
    );
    let leftovers: Vec<_> = fs::read_dir(dir.path())
        .expect("read temporary directory")
        .map(|entry| entry.expect("directory entry").file_name())
        .filter(|name| name.to_string_lossy().starts_with("merged.parquet.tmp."))
        .collect();
    assert!(
        leftovers.is_empty(),
        "merge temp files remain: {leftovers:?}"
    );
}

#[cfg(unix)]
#[test]
fn resolved_alias_to_legacy_temp_path_is_refused_unchanged() {
    use std::os::unix::fs::symlink;

    let (_dir, output, input) = fixture();
    let legacy_temp = PathBuf::from(format!("{}.tmp", output.display()));
    fs::copy(&input, &legacy_temp).expect("copy valid input to legacy temp path");
    let alias = legacy_temp.with_file_name("input-alias.parquet");
    symlink(&legacy_temp, &alias).expect("create input alias");
    let original = fs::read(&legacy_temp).expect("read original bytes");

    let error = merge_parquet_files(&path_string(&output), &[path_string(&alias)])
        .expect_err("resolved legacy temp alias must be refused");

    assert!(format!("{error:#}").contains("temporary"));
    assert_eq!(fs::read(&legacy_temp).expect("re-read input"), original);
    assert!(
        fs::symlink_metadata(&alias)
            .expect("inspect alias")
            .file_type()
            .is_symlink()
    );
}
