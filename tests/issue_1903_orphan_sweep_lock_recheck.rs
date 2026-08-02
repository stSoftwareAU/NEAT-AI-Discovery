//! Issue #1903 — the orphan sweep must re-check the lock before deleting.
//!
//! The sweep decided a directory was abandoned in `is_directory_orphaned` and
//! deleted it in a later, separate call. A discovery session that claimed the
//! directory inside that window lost its in-flight data. These tests pin the
//! three behaviours that close it:
//!
//! * a lock created inside the window aborts the removal with
//!   [`CleanupOutcome::Claimed`],
//! * a `discovery.lock` that is a dangling symlink is *not* orphaned (the probe
//!   fails closed), and
//! * the claimed-mid-sweep case is counted separately from `removed`.

use std::fs::{self, File};
use std::time::{Duration, SystemTime};

use neat_ai_discovery::discovery_cleanup::{
    CleanupOutcome, LOCK_FILE_NAME, OrphanCleanupResult, clean_orphaned_discovery_dirs,
    clean_orphaned_discovery_dirs_since, cleanup_orphaned_discovery_dir, is_directory_orphaned,
};
use tempfile::TempDir;

/// Create a `.discovery` root inside `temp` so every path passes the
/// Issue #1218 / #1866 allowlists.
fn make_discovery_root(temp: &TempDir) -> std::path::PathBuf {
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    root
}

#[test]
fn lock_created_after_orphan_check_survives_sweep() {
    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let session = base.join("session-claimed");
    fs::create_dir(&session).unwrap();
    File::create(session.join("discovery_data.parquet")).unwrap();

    // The sweep's check: no lock, so the directory looks abandoned.
    assert!(is_directory_orphaned(&session));

    // The race window: a discovery session claims the directory before the
    // sweep gets round to removing it.
    File::create(session.join(LOCK_FILE_NAME)).unwrap();

    let outcome = cleanup_orphaned_discovery_dir(session.to_str().unwrap()).unwrap();

    assert_eq!(
        outcome,
        CleanupOutcome::Claimed,
        "a directory claimed inside the window must report Claimed"
    );
    assert!(session.exists(), "the claimed directory must survive");
    assert!(
        session.join("discovery_data.parquet").exists(),
        "in-flight data must survive"
    );
}

#[test]
fn genuinely_orphaned_directory_is_still_removed_by_the_sweep_path() {
    // The re-check must not block the case it was built to allow.
    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let session = base.join("session-orphan");
    fs::create_dir(&session).unwrap();
    File::create(session.join("discovery_data.parquet")).unwrap();

    let outcome = cleanup_orphaned_discovery_dir(session.to_str().unwrap()).unwrap();

    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!session.exists());
}

#[cfg(unix)]
#[test]
fn dangling_symlink_lock_is_not_orphaned() {
    use std::os::unix::fs as unix_fs;

    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let session = base.join("session-dangling-lock");
    fs::create_dir(&session).unwrap();
    File::create(session.join("discovery_data.parquet")).unwrap();

    // `discovery.lock` exists as a symlink, but its target does not. The old
    // `Path::exists()` probe followed the link and reported "no lock".
    unix_fs::symlink(
        session.join("no-such-lock-target"),
        session.join(LOCK_FILE_NAME),
    )
    .unwrap();

    assert!(
        !is_directory_orphaned(&session),
        "a dangling-symlink lock must fail closed, not read as absent"
    );

    // …and the sweep must therefore leave it alone.
    let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
    assert_eq!(result.removed, 0);
    assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    assert!(session.exists(), "the directory must survive the sweep");
}

#[test]
fn claimed_counter_reported_separately() {
    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let session = base.join("session-mid-scan");
    fs::create_dir(&session).unwrap();
    File::create(session.join("discovery_data.parquet")).unwrap();

    // Age floor: sweeping with a scan-start an hour in the past makes this
    // freshly created directory "touched after the scan started", exactly as a
    // directory created mid-scan would be.
    let floor = SystemTime::now() - Duration::from_secs(3600);
    let result = clean_orphaned_discovery_dirs_since(base.to_str().unwrap(), floor).unwrap();

    assert_eq!(result.claimed, 1, "claimed must be counted on its own");
    assert_eq!(result.removed, 0, "claimed must not be counted as removed");
    assert_eq!(result.already_gone, 0);
    assert!(
        result.errors.is_empty(),
        "claimed is not an error: {:?}",
        result.errors
    );
    assert!(session.exists(), "the directory must survive the sweep");

    // With the real scan-start the same directory is genuinely orphaned and is
    // swept, so `claimed` is not a blanket veto.
    let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
    assert_eq!(result.removed, 1);
    assert_eq!(result.claimed, 0);
    assert!(!session.exists());
}

#[test]
fn claimed_defaults_to_zero() {
    let result = OrphanCleanupResult::default();
    assert_eq!(result.claimed, 0);
    assert_eq!(result.removed, 0);
}
