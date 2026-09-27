//! Issue #2256 — the orphan sweep must honour the host's lock-file spelling.
//!
//! The sweep keyed liveness on `discovery.lock`, but NEAT-AI
//! (`src/discovery/DiscoveryCleanup.ts`) writes `.discovery.lock`. A live
//! NEAT-AI session therefore looked abandoned and was deleted mid-run
//! (CWE-706). Either spelling must now mark a directory as in use, everywhere
//! the lock is probed.

use std::fs::{self, File};
use std::path::PathBuf;

use neat_ai_discovery::discovery_cleanup::{
    CleanupOutcome, HOST_LOCK_FILE_NAME, LOCK_FILE_NAME, clean_orphaned_discovery_dirs,
    cleanup_discovery_dir, cleanup_orphaned_discovery_dir, is_directory_orphaned,
};
use tempfile::TempDir;

/// A `.discovery` root inside `temp`, so every path passes the #1218 / #1866
/// allowlists.
fn make_discovery_root(temp: &TempDir) -> PathBuf {
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    root
}

/// A `<root>/<uuid>/` session directory holding only the given lock file.
fn make_session(root: &std::path::Path, name: &str, lock: Option<&str>) -> PathBuf {
    let session = root.join(name);
    fs::create_dir(&session).unwrap();
    if let Some(lock) = lock {
        File::create(session.join(lock)).unwrap();
    }
    session
}

#[test]
fn host_spelled_lock_survives_orphan_sweep() {
    let temp = TempDir::new().unwrap();
    let root = make_discovery_root(&temp);
    let session = make_session(
        &root,
        "0b9f3c1e-4a6d-4f55-9c2e-5d8a7b1e2f30",
        Some(".discovery.lock"),
    );

    let result = clean_orphaned_discovery_dirs(root.to_str().unwrap()).unwrap();

    assert!(session.exists(), "a live NEAT-AI session must not be swept");
    assert_eq!(result.removed, 0, "nothing may be removed");
    // Not merely rescued by the age floor: the lock itself must keep it live.
    assert_eq!(
        result.claimed, 0,
        "the lock, not the age floor, must spare it"
    );
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn either_lock_spelling_marks_directory_in_use() {
    let temp = TempDir::new().unwrap();
    let root = make_discovery_root(&temp);

    assert_eq!(HOST_LOCK_FILE_NAME, ".discovery.lock");
    for lock in [LOCK_FILE_NAME, HOST_LOCK_FILE_NAME] {
        let session = make_session(&root, &format!("session-{lock}"), Some(lock));
        assert!(!is_directory_orphaned(&session), "{lock} must mark it live");
    }

    let unlocked = make_session(&root, "session-unlocked", None);
    assert!(
        is_directory_orphaned(&unlocked),
        "no lock at all is orphaned"
    );
}

#[test]
fn sweep_still_removes_unlocked_sibling() {
    let temp = TempDir::new().unwrap();
    let root = make_discovery_root(&temp);
    let host = make_session(&root, "session-host", Some(HOST_LOCK_FILE_NAME));
    let ours = make_session(&root, "session-ours", Some(LOCK_FILE_NAME));
    let orphan = make_session(&root, "session-orphan", None);
    File::create(orphan.join("discovery_data.parquet")).unwrap();
    // Backdate the orphan so the age floor cannot spare it on a coarse clock.
    File::open(&orphan)
        .unwrap()
        .set_modified(std::time::SystemTime::UNIX_EPOCH)
        .unwrap();

    let result = clean_orphaned_discovery_dirs(root.to_str().unwrap()).unwrap();

    assert!(
        host.exists() && ours.exists(),
        "locked sessions must survive"
    );
    assert!(!orphan.exists(), "a lock-less directory is still swept");
    assert_eq!(result.removed, 1);
}

#[test]
fn recheck_honours_host_spelled_lock() {
    let temp = TempDir::new().unwrap();
    let root = make_discovery_root(&temp);
    let session = make_session(&root, "session-claimed", Some(HOST_LOCK_FILE_NAME));

    let outcome = cleanup_orphaned_discovery_dir(session.to_str().unwrap()).unwrap();

    assert_eq!(outcome, CleanupOutcome::Claimed);
    assert!(session.exists(), "the #1903 re-check must spare it");
}

#[test]
fn host_spelled_lock_identifies_discovery_dir() {
    // No `.discovery` path component, so only the contents check can accept it.
    let temp = TempDir::new().unwrap();
    let session = temp.path().join("session");
    fs::create_dir(&session).unwrap();
    File::create(session.join(HOST_LOCK_FILE_NAME)).unwrap();

    let outcome = cleanup_discovery_dir(session.to_str().unwrap()).unwrap();

    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!session.exists());
}
