//! Chunk 11 probe (c) — crafted entry names (Issues #2095, #2234).
//!
//! `read_dir` yields single-component names, joined under the root, so an entry
//! can never be absolute or climb out with `..`. These tests pin the observable
//! half: orphans with unicode, dotted, dash-led and control-character names are
//! removed, and a lock-less canary directory beside the root survives.
//!
//! The non-UTF-8 case is a filed finding (#2255) and ships its own failing-first
//! test with the fix.

use std::fs::{self, File};

use neat_ai_discovery::discovery_cleanup::{clean_orphaned_discovery_dirs, cleanup_discovery_dir};
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

#[test]
fn parent_dir_component_is_refused_and_the_canary_it_names_survives() {
    // `assert_is_discovery_dir` refuses `Component::ParentDir` before any
    // probe, so a marker-carrying path cannot climb to a sibling.
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    let canary = temp.path().join("canary-beside-root");
    fs::create_dir(&canary).unwrap();
    let canary_file = canary.join("canary.dat");
    File::create(&canary_file).unwrap();

    let escaping = root.join("..").join("canary-beside-root");
    let err = cleanup_discovery_dir(escaping.to_str().unwrap()).unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(canary.exists(), "the canary named via `..` must survive");
    assert!(canary_file.exists(), "the canary's contents must survive");
}
