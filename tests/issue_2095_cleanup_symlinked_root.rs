//! Chunk 11 probe (a) — a symlinked discovery **root** (Issues #2095, #2234).
//!
//! `clean_orphaned_discovery_dirs` gates `base_dir` on its final component
//! containing `.discovery`, then follows it: `exists()`, `is_dir()` and
//! `read_dir` all resolve a symlinked root, and each child path is
//! `<link>/<child>`. The sweep accepts that — the root defaults to `.discovery`
//! in the caller's own working directory, so only the caller can plant the link,
//! and a link to a scratch volume is a legitimate layout.
//!
//! What must hold is the blast radius: only lock-less child directories *of the
//! link target* go. The target itself, the link, a locked session, and a canary
//! outside both the link and its target all survive.

#![cfg(unix)]

use std::fs::{self, File};
use std::os::unix::fs as unix_fs;

use neat_ai_discovery::discovery_cleanup::{LOCK_FILE_NAME, clean_orphaned_discovery_dirs};
use tempfile::TempDir;

#[test]
fn symlinked_root_sweeps_only_orphans_under_its_target() {
    let temp = TempDir::new().unwrap();

    // The real store the `.discovery` link points at.
    let target = temp.path().join("scratch-store");
    fs::create_dir(&target).unwrap();
    let orphan = target.join("session-orphan");
    fs::create_dir(&orphan).unwrap();
    File::create(orphan.join("discovery_data.parquet")).unwrap();
    let active = target.join("session-active");
    fs::create_dir(&active).unwrap();
    File::create(active.join(LOCK_FILE_NAME)).unwrap();
    let target_file = target.join("keep.dat");
    File::create(&target_file).unwrap();

    // A canary beside the link and its target, shaped like an orphan so a
    // sweep that escaped the root would delete it.
    let canary = temp.path().join("canary-outside");
    fs::create_dir(&canary).unwrap();
    let canary_file = canary.join("canary.dat");
    File::create(&canary_file).unwrap();

    let link = temp.path().join(".discovery");
    unix_fs::symlink(&target, &link).unwrap();

    let result = clean_orphaned_discovery_dirs(link.to_str().unwrap()).unwrap();

    assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    assert_eq!(result.removed, 1, "only the lock-less child is swept");
    assert!(!orphan.exists(), "the orphan under the target is removed");

    assert!(
        canary.exists(),
        "canary outside the link and target must survive"
    );
    assert!(canary_file.exists(), "canary contents must survive");
    assert!(
        active.exists(),
        "a locked session under the target must survive"
    );
    assert!(target.exists(), "the link target itself must survive");
    assert!(
        target_file.exists(),
        "a plain file under the target must survive"
    );
    assert!(
        fs::symlink_metadata(&link).unwrap().is_symlink(),
        "the root link itself must remain a symlink"
    );
}

#[test]
fn symlinked_root_still_skips_symlinked_children() {
    // Following the root must not extend to following a child link: the
    // per-entry `file_type()` check still refuses it, so a canary reached only
    // through a child symlink survives.
    let temp = TempDir::new().unwrap();
    let target = temp.path().join("scratch-store");
    fs::create_dir(&target).unwrap();

    let canary = temp.path().join("canary-outside");
    fs::create_dir(&canary).unwrap();
    let canary_file = canary.join("canary.dat");
    File::create(&canary_file).unwrap();
    unix_fs::symlink(&canary, target.join("child-link")).unwrap();

    let link = temp.path().join(".discovery");
    unix_fs::symlink(&target, &link).unwrap();

    let result = clean_orphaned_discovery_dirs(link.to_str().unwrap()).unwrap();

    assert_eq!(result.removed, 0);
    assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    assert!(canary.exists(), "canary behind a child link must survive");
    assert!(canary_file.exists(), "canary contents must survive");
}
