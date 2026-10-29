//! Chunk 11 probe (c) — crafted entry names (Issues #2095, #2234).
//!
//! `read_dir` yields single-component names, joined under the root, so an entry
//! can never be absolute or climb out with `..`. This test pins the observable
//! half: orphans with unicode, dotted, dash-led and control-character names are
//! removed, and a lock-less canary directory beside the root survives.
//!
//! The non-UTF-8 case is a filed finding (#2255) and ships its own failing-first
//! test with the fix. The `..` refusal is pinned by
//! `tests/issue_1866_cleanup_dir_path_guard.rs::test_cleanup_rejects_parent_dir_traversal`.

#![cfg(unix)]

use std::fs::{self, File};

use neat_ai_discovery::discovery_cleanup::clean_orphaned_discovery_dirs;
use tempfile::TempDir;

/// Names a crafted or careless host could give a session directory.
const CRAFTED_NAMES: [&str; 5] = [
    "séance-🧠-会話",
    "...",
    "-rf",
    "line\nbreak",
    "\u{202E}gpj.exe",
];

#[test]
fn crafted_orphan_names_are_removed_and_the_canary_beside_the_root_survives() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();

    let orphans: Vec<_> = CRAFTED_NAMES
        .iter()
        .map(|name| {
            let dir = root.join(name);
            fs::create_dir(&dir).unwrap();
            File::create(dir.join("discovery_data.parquet")).unwrap();
            dir
        })
        .collect();

    // Beside the root, not under it — lock-less, so an escaping sweep would take it.
    let canary = temp.path().join("canary-beside-root");
    fs::create_dir(&canary).unwrap();
    let canary_file = canary.join("canary.dat");
    File::create(&canary_file).unwrap();

    let result = clean_orphaned_discovery_dirs(root.to_str().unwrap()).unwrap();

    assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    assert_eq!(result.removed as usize, CRAFTED_NAMES.len());
    assert_eq!(result.already_gone, 0);
    for orphan in &orphans {
        assert!(!orphan.exists(), "{orphan:?} must be removed");
    }
    assert!(root.exists(), "the root itself must survive");
    assert!(canary.exists(), "the canary beside the root must survive");
    assert!(canary_file.exists(), "the canary's contents must survive");
}
