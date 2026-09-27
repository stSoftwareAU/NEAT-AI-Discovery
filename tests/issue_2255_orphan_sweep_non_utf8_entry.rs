//! Issue #2255 — the orphan sweep must remove the exact entry it vetted.
//!
//! The sweep checked the orphan and #1903 age-floor guards on the real
//! `PathBuf`, then removed `path.display().to_string()`. That rendering is
//! lossy: a non-UTF-8 name such as `orphan-\xff` became `orphan-\u{FFFD}`, so
//! the removal hit a *different* directory — one the age floor had never
//! vetted — and the real orphan survived while being counted as `already_gone`.
#![cfg(unix)]

use std::ffi::OsStr;
use std::fs::{self, File};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use neat_ai_discovery::discovery_cleanup::{
    CleanupOutcome, clean_orphaned_discovery_dirs_since, cleanup_orphaned_discovery_dir,
};
use tempfile::TempDir;

fn make_discovery_root(temp: &TempDir) -> PathBuf {
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    root
}

/// Backdate a directory's modification time so it predates the sweep's floor.
fn backdate(dir: &Path, when: SystemTime) {
    File::open(dir).unwrap().set_modified(when).unwrap();
}

#[test]
fn sweep_removes_non_utf8_orphan_and_spares_lossy_sibling() {
    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let now = SystemTime::now();
    let scan_started = now - Duration::from_secs(3600);

    // A genuine orphan with a non-UTF-8 name, last touched before the scan.
    let orphan = base.join(OsStr::from_bytes(b"orphan-\xff"));
    fs::create_dir(&orphan).unwrap();
    File::create(orphan.join("discovery_data.parquet")).unwrap();
    backdate(&orphan, now - Duration::from_secs(7200));

    // Its lossy rendering names this sibling, touched after the scan started,
    // so the #1903 age floor must keep it out of the candidate set.
    let canary = base.join("orphan-\u{FFFD}");
    fs::create_dir(&canary).unwrap();
    File::create(canary.join("discovery_data.parquet")).unwrap();

    let result = clean_orphaned_discovery_dirs_since(base.to_str().unwrap(), scan_started).unwrap();

    assert!(
        canary.exists(),
        "sibling touched after the scan started must survive the sweep"
    );
    assert!(!orphan.exists(), "non-UTF-8 orphan must be removed");
    assert_eq!(result.removed, 1, "{result:?}");
    assert_eq!(result.claimed, 1, "{result:?}");
    assert_eq!(result.already_gone, 0, "{result:?}");
    assert!(result.errors.is_empty(), "{result:?}");
}

#[test]
fn orphaned_removal_accepts_non_utf8_path() {
    let temp = TempDir::new().unwrap();
    let base = make_discovery_root(&temp);

    let orphan = base.join(OsStr::from_bytes(b"session-\xfe\xff"));
    fs::create_dir(&orphan).unwrap();

    let outcome = cleanup_orphaned_discovery_dir(&orphan).unwrap();
    assert_eq!(outcome, CleanupOutcome::Removed);
    assert!(!orphan.exists());
}
