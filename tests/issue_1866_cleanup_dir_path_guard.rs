//! Issue #1866 — `cleanup_discovery_dir` must refuse arbitrary caller paths.
//!
//! The FFI entry point `cleanup_discovery_dir` passed its caller-supplied
//! `tempDir` straight to `fs::remove_dir_all`, so any host-side path bug
//! became an unconstrained recursive delete (CWE-73). The sibling scanner
//! `clean_orphaned_discovery_dirs` already had this guard (Issue #1218);
//! these tests pin the same protection onto the directly FFI-exposed
//! function and onto the scanner's internal delegation.

use neat_ai_discovery::discovery_cleanup::{
    CleanupOutcome, DISCOVERY_DATA_FILE_NAME, LOCK_FILE_NAME, cleanup_discovery_dir,
};
use std::fs::{self, File};
use std::io;
use tempfile::TempDir;

/// A directory that is neither under a discovery root nor carries discovery
/// contents must be rejected, and must survive untouched.
#[test]
fn test_cleanup_rejects_unrelated_directory() {
    let temp = TempDir::new().unwrap();
    let victim = temp.path().join("Documents");
    fs::create_dir(&victim).unwrap();
    let precious = victim.join("thesis.txt");
    File::create(&precious).unwrap();

    let err = cleanup_discovery_dir(victim.to_str().unwrap()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains(".discovery"),
        "error must explain the allowlist: {err}"
    );
    assert!(victim.exists(), "unrelated directory must survive");
    assert!(precious.exists(), "unrelated files must survive");
}

/// The canonical bad inputs from Issue #1218, now applied to the direct
/// entry point: `/`, `/tmp` and a `/var/folders/...` style path.
#[test]
fn test_cleanup_rejects_root_and_tmp() {
    for bad in ["/", "/tmp", "/var/folders/xx", "/etc"] {
        let err = cleanup_discovery_dir(bad).unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidInput,
            "expected InvalidInput for temp_dir={bad:?}"
        );
    }
}

/// An empty path must be rejected rather than silently reported as gone.
#[test]
fn test_cleanup_rejects_empty_path() {
    let err = cleanup_discovery_dir("").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

/// A `..` component could escape a legitimate discovery root, so any path
/// containing one is rejected outright.
#[test]
fn test_cleanup_rejects_parent_dir_traversal() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    let victim = temp.path().join("Documents");
    fs::create_dir(&victim).unwrap();

    let traversal = format!("{}/session/../../Documents", root.display());
    let err = cleanup_discovery_dir(&traversal).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(victim.exists(), "traversal target must survive");
}

/// A directory under a `.discovery` root is accepted even when it holds no
/// discovery marker files (this is how the orphan scanner delegates).
#[test]
fn test_cleanup_accepts_child_of_discovery_root() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    let session = root.join("session-abc123");
    fs::create_dir(&session).unwrap();
    File::create(session.join("scratch.bin")).unwrap();

    let outcome = cleanup_discovery_dir(session.to_str().unwrap()).unwrap();
    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!session.exists());
    assert!(root.exists(), "the discovery root itself must remain");
}

/// A directory outside a discovery root is still accepted when it carries
/// the discovery lock file — that content proves it is ours.
#[test]
fn test_cleanup_accepts_directory_with_lock_file() {
    let temp = TempDir::new().unwrap();
    let session = temp.path().join("session-xyz");
    fs::create_dir(&session).unwrap();
    File::create(session.join(LOCK_FILE_NAME)).unwrap();

    let outcome = cleanup_discovery_dir(session.to_str().unwrap()).unwrap();
    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!session.exists());
}

/// Likewise for a directory carrying the discovery parquet data file.
#[test]
fn test_cleanup_accepts_directory_with_parquet_data() {
    let temp = TempDir::new().unwrap();
    let session = temp.path().join("session-parquet");
    fs::create_dir(&session).unwrap();
    File::create(session.join(DISCOVERY_DATA_FILE_NAME)).unwrap();

    let outcome = cleanup_discovery_dir(session.to_str().unwrap()).unwrap();
    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!session.exists());
}

/// A discovery path that has already been removed by another actor stays a
/// benign `AlreadyGone`, so the guard does not break the Issue #1100
/// race-suppression contract.
#[test]
fn test_cleanup_missing_discovery_path_still_already_gone() {
    let temp = TempDir::new().unwrap();
    let missing = temp.path().join(".discovery").join("does-not-exist");

    let outcome = cleanup_discovery_dir(missing.to_str().unwrap()).unwrap();
    assert_eq!(outcome, CleanupOutcome::AlreadyGone);
}

/// A regular file is not a discovery directory and must be rejected.
#[test]
fn test_cleanup_rejects_regular_file() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    let file = root.join("not-a-dir.txt");
    File::create(&file).unwrap();

    let err = cleanup_discovery_dir(file.to_str().unwrap()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(file.exists(), "file must not be removed");
}

/// A symlinked `temp_dir` must be refused via `symlink_metadata` so
/// `remove_dir_all` can never touch the link target's contents.
#[cfg(unix)]
#[test]
fn test_cleanup_rejects_symlinked_temp_dir() {
    use std::os::unix::fs as unix_fs;

    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();

    let outside = temp.path().join("outside-target");
    fs::create_dir(&outside).unwrap();
    let canary = outside.join("canary.dat");
    File::create(&canary).unwrap();

    let link = root.join("symlinked-session");
    unix_fs::symlink(&outside, &link).unwrap();

    let err = cleanup_discovery_dir(link.to_str().unwrap()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(outside.exists(), "symlink target must not be deleted");
    assert!(canary.exists(), "files under symlink target must survive");
    assert!(link.exists(), "the symlink itself must remain");
}
