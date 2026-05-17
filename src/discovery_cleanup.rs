//! Discovery directory cleanup with race-condition safety (Issue #1100).
//!
//! This module provides atomic cleanup of discovery temp directories and
//! orphan scanning that avoids the race condition between async cleanup
//! and the orphan scanner.
//!
//! ## The Problem
//!
//! Previously, cleanup removed the lock file *before* the directory:
//!
//! ```text
//! 1. removeDiscoveryLockFile()   -- lock file gone
//! 2. Deno.remove(tempDir)        -- directory still exists briefly
//! ```
//!
//! This created a window where the orphan scanner could see the directory
//! without a lock file, classify it as orphaned, and delete it. The original
//! async cleanup would then fail with `NotFound`.
//!
//! ## The Fix
//!
//! `cleanup_discovery_dir()` removes the entire directory in one recursive
//! call. The lock file is inside the directory, so it is removed atomically
//! from the orphan scanner's perspective. `NotFound` errors are suppressed
//! because another actor may have already cleaned the directory.

use std::fs;
use std::io;
use std::path::Path;

/// Conventional lock-file name placed inside a discovery temp directory
/// to indicate it is still in use.
pub const LOCK_FILE_NAME: &str = "discovery.lock";

/// Marker substring that must appear in the final path component of any
/// `base_dir` accepted by [`clean_orphaned_discovery_dirs`] (Issue #1218).
///
/// This is a defence-in-depth allowlist: the orphan scanner recursively
/// removes every subdirectory of `base_dir` that lacks a
/// [`LOCK_FILE_NAME`] file, so a caller bug or misconfiguration that
/// passed e.g. `/tmp`, `$HOME`, or `/var/folders/...` would mass-delete
/// unrelated subdirectories. Requiring the marker localises that blast
/// radius to genuine discovery roots.
pub const DISCOVERY_DIR_MARKER: &str = ".discovery";

/// Result of cleaning up a single discovery directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupOutcome {
    /// The directory was successfully removed.
    Removed,
    /// The directory was already gone (another actor removed it first).
    AlreadyGone,
}

/// Atomically clean up a discovery temp directory (Issue #1100).
///
/// Removes the entire directory tree in a single recursive call so that
/// the lock file is never absent while the directory still exists. If the
/// directory has already been removed (e.g. by the orphan scanner), this
/// returns `Ok(CleanupOutcome::AlreadyGone)` instead of an error.
pub fn cleanup_discovery_dir(temp_dir: &str) -> io::Result<CleanupOutcome> {
    let path = Path::new(temp_dir);

    match fs::remove_dir_all(path) {
        Ok(()) => {
            tracing::debug!(
                path = %temp_dir,
                "Discovery temp directory cleaned up"
            );
            Ok(CleanupOutcome::Removed)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            tracing::debug!(
                path = %temp_dir,
                "Discovery temp directory already removed by another actor"
            );
            Ok(CleanupOutcome::AlreadyGone)
        }
        Err(err) => {
            tracing::warn!(
                path = %temp_dir,
                error = %err,
                "Failed to clean up discovery temp directory"
            );
            Err(err)
        }
    }
}

/// Check whether a discovery directory is orphaned.
///
/// A directory is considered orphaned when its lock file is absent, meaning
/// no active discovery process owns it. Directories that still contain a
/// lock file are actively in use and must not be removed.
pub fn is_directory_orphaned(dir: &Path) -> bool {
    let lock_path = dir.join(LOCK_FILE_NAME);
    !lock_path.exists()
}

/// Result of scanning and cleaning orphaned discovery directories.
#[derive(Debug, Clone, Default)]
pub struct OrphanCleanupResult {
    /// Number of orphaned directories successfully removed.
    pub removed: u32,
    /// Number of directories that were already gone when removal was attempted.
    pub already_gone: u32,
    /// Number of directories that failed to remove (with error details).
    pub errors: Vec<String>,
}

/// Scan a base directory for orphaned discovery temp directories and remove them
/// (Issue #1100).
///
/// A subdirectory is considered orphaned when it has no `discovery.lock` file.
/// Removal suppresses `NotFound` errors because the async cleanup actor may
/// have removed the directory between the orphan check and the removal call.
///
/// # Arguments
///
/// * `base_dir` - The parent directory that contains discovery temp directories
///   (e.g. `.discovery/`).
pub fn clean_orphaned_discovery_dirs(base_dir: &str) -> io::Result<OrphanCleanupResult> {
    let base_path = Path::new(base_dir);

    // Defence in depth (Issue #1218): refuse to operate on anything whose
    // final path component does not contain `DISCOVERY_DIR_MARKER`. Without
    // this gate, a future caller bug that derived `base_dir` from a
    // mis-configured environment variable or an untrusted field (e.g.
    // defaulting to `os.tmpdir()` or `$HOME`) would mass-delete unrelated
    // subdirectories. This check is enforced before the existence/is_dir
    // probes so callers cannot bypass it by passing a non-existent path.
    let looks_like_discovery_root = base_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains(DISCOVERY_DIR_MARKER));
    if !looks_like_discovery_root {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "base_dir must be a discovery root (final path component must contain {DISCOVERY_DIR_MARKER}): {base_dir}"
            ),
        ));
    }

    if !base_path.exists() {
        return Ok(OrphanCleanupResult::default());
    }

    if !base_path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Base path is not a directory: {base_dir}"),
        ));
    }

    let mut result = OrphanCleanupResult::default();

    let entries = match fs::read_dir(base_path) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(result);
        }
        Err(err) => return Err(err),
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                result
                    .errors
                    .push(format!("Failed to read directory entry: {err}"));
                continue;
            }
        };

        let path = entry.path();

        // Defence in depth (Issue #1218): skip symlinks. `path.is_dir()`
        // follows symlinks, and `fs::remove_dir_all` on a symlinked
        // directory has had platform-dependent / version-dependent
        // behaviour in the past where it could delete the link target's
        // contents. Discovery sessions never create symlinked roots, so
        // a symlink here is always anomalous.
        match entry.file_type() {
            Ok(ft) if ft.is_symlink() => {
                tracing::warn!(
                    path = %path.display(),
                    "Refusing to follow symlink during orphan scan"
                );
                continue;
            }
            Ok(_) => {}
            Err(err) => {
                result
                    .errors
                    .push(format!("Failed to read file type for {path:?}: {err}"));
                continue;
            }
        }

        if !path.is_dir() {
            continue;
        }

        if !is_directory_orphaned(&path) {
            continue;
        }

        let dir_str = path.display().to_string();
        match cleanup_discovery_dir(&dir_str) {
            Ok(CleanupOutcome::Removed) => {
                tracing::info!(
                    path = %dir_str,
                    "Removed orphaned discovery directory"
                );
                result.removed += 1;
            }
            Ok(CleanupOutcome::AlreadyGone) => {
                result.already_gone += 1;
            }
            Err(err) => {
                result.errors.push(format!(
                    "Failed to remove orphaned directory {dir_str}: {err}"
                ));
            }
        }
    }

    if result.removed > 0 || !result.errors.is_empty() {
        tracing::info!(
            base_dir = %base_dir,
            removed = result.removed,
            already_gone = result.already_gone,
            errors = result.errors.len(),
            "Orphan discovery directory scan complete"
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_cleanup_removes_directory() {
        let base = TempDir::new().unwrap();
        let discovery_dir = base.path().join("discovery-abc123");
        fs::create_dir(&discovery_dir).unwrap();

        // Place a lock file and a data file inside
        File::create(discovery_dir.join(LOCK_FILE_NAME)).unwrap();
        File::create(discovery_dir.join("discovery_data.parquet")).unwrap();

        let result = cleanup_discovery_dir(discovery_dir.to_str().unwrap()).unwrap();
        assert_eq!(result, CleanupOutcome::Removed);
        assert!(!discovery_dir.exists());
    }

    #[test]
    fn test_cleanup_already_gone_returns_ok() {
        let base = TempDir::new().unwrap();
        let nonexistent = base.path().join("does-not-exist");

        let result = cleanup_discovery_dir(nonexistent.to_str().unwrap()).unwrap();
        assert_eq!(result, CleanupOutcome::AlreadyGone);
    }

    #[test]
    fn test_is_directory_orphaned_no_lock_file() {
        let base = TempDir::new().unwrap();
        let dir = base.path().join("orphan-dir");
        fs::create_dir(&dir).unwrap();

        assert!(is_directory_orphaned(&dir));
    }

    #[test]
    fn test_is_directory_orphaned_with_lock_file() {
        let base = TempDir::new().unwrap();
        let dir = base.path().join("active-dir");
        fs::create_dir(&dir).unwrap();
        File::create(dir.join(LOCK_FILE_NAME)).unwrap();

        assert!(!is_directory_orphaned(&dir));
    }

    /// Create a discovery-root child of `parent` whose name contains the
    /// [`DISCOVERY_DIR_MARKER`] so it passes the entry-point allowlist
    /// (Issue #1218). All orphan-scan tests run inside such a root.
    fn make_discovery_root(parent: &Path) -> std::path::PathBuf {
        let root = parent.join(".discovery");
        fs::create_dir(&root).unwrap();
        root
    }

    #[test]
    fn test_orphan_scan_removes_only_orphaned_dirs() {
        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());

        // Create an orphaned directory (no lock file)
        let orphan = base.join("orphan-001");
        fs::create_dir(&orphan).unwrap();
        File::create(orphan.join("discovery_data.parquet")).unwrap();

        // Create an active directory (has lock file)
        let active = base.join("active-002");
        fs::create_dir(&active).unwrap();
        File::create(active.join(LOCK_FILE_NAME)).unwrap();
        File::create(active.join("discovery_data.parquet")).unwrap();

        let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
        assert_eq!(result.removed, 1);
        assert!(result.errors.is_empty());

        // Orphaned dir should be gone
        assert!(!orphan.exists());
        // Active dir should remain
        assert!(active.exists());
    }

    #[test]
    fn test_orphan_scan_nonexistent_base_dir() {
        let result = clean_orphaned_discovery_dirs("/nonexistent/path/.discovery").unwrap();
        assert_eq!(result.removed, 0);
        assert_eq!(result.already_gone, 0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_orphan_scan_empty_base_dir() {
        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());
        let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
        assert_eq!(result.removed, 0);
        assert_eq!(result.already_gone, 0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_orphan_scan_skips_files() {
        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());

        // Create a regular file (not a directory) in the base dir
        File::create(base.join("not-a-dir.txt")).unwrap();

        let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
        assert_eq!(result.removed, 0);
        // The file should still exist
        assert!(base.join("not-a-dir.txt").exists());
    }

    #[test]
    fn test_orphan_scan_rejects_non_discovery_root() {
        // Defence-in-depth check (Issue #1218): a base_dir whose final
        // path component does not contain ".discovery" must be rejected
        // with InvalidInput, even if it is a real, existing directory
        // populated with subdirectories that look orphaned.
        let temp = TempDir::new().unwrap();
        let bogus = temp.path().join("not-a-discovery-root");
        fs::create_dir(&bogus).unwrap();
        let victim = bogus.join("some-other-tool-data");
        fs::create_dir(&victim).unwrap();
        File::create(victim.join("important.dat")).unwrap();

        let err = clean_orphaned_discovery_dirs(bogus.to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains(".discovery"),
            "error message must explain the allowlist: {err}"
        );
        // The victim directory must be untouched.
        assert!(victim.exists());
        assert!(victim.join("important.dat").exists());
    }

    #[test]
    fn test_orphan_scan_rejects_root_and_tmp() {
        // Spot-check the canonical bad inputs called out in Issue #1218:
        // a future caller bug that derives base_dir from os.tmpdir() or
        // $HOME must not be able to mass-delete unrelated subdirectories.
        for bad in ["/", "/tmp", "/var/folders/xx"] {
            let err = clean_orphaned_discovery_dirs(bad).unwrap_err();
            assert_eq!(
                err.kind(),
                io::ErrorKind::InvalidInput,
                "expected InvalidInput for base_dir={bad:?}"
            );
        }
    }

    #[test]
    fn test_orphan_scan_accepts_suffixed_discovery_root() {
        // The marker is a `contains` check, so e.g. `my-project.discovery`
        // and `.discovery-cache` should both be accepted.
        let temp = TempDir::new().unwrap();
        for name in ["my-project.discovery", ".discovery-cache"] {
            let base = temp.path().join(name);
            fs::create_dir(&base).unwrap();
            let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
            assert_eq!(result.removed, 0);
            assert!(result.errors.is_empty());
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_orphan_scan_skips_symlinked_subdirs() {
        // Defence-in-depth (Issue #1218): a symlink inside the discovery
        // root must not be followed by the orphan scanner. Otherwise
        // `fs::remove_dir_all` on the symlinked path could (on some
        // platforms / older Rust versions) delete the link target's
        // contents.
        use std::os::unix::fs as unix_fs;

        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());

        // Real target that lives outside the discovery root and must
        // remain untouched.
        let outside = temp.path().join("outside-target");
        fs::create_dir(&outside).unwrap();
        let canary = outside.join("canary.dat");
        File::create(&canary).unwrap();

        // Symlink inside the discovery root pointing at the outside target.
        let link = base.join("symlinked-session");
        unix_fs::symlink(&outside, &link).unwrap();

        let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
        // The symlink itself was skipped — not counted as removed.
        assert_eq!(result.removed, 0);
        assert!(result.errors.is_empty());
        // Critically: the outside target and its contents survive.
        assert!(outside.exists(), "symlink target must not be deleted");
        assert!(canary.exists(), "files under symlink target must survive");
    }

    #[test]
    fn test_concurrent_cleanup_no_not_found_error() {
        // Simulates the race condition: two actors try to clean the same dir.
        // Both should succeed without errors.
        let base = TempDir::new().unwrap();
        let discovery_dir = base.path().join("concurrent-dir");
        fs::create_dir(&discovery_dir).unwrap();
        File::create(discovery_dir.join("data.parquet")).unwrap();

        let dir_str = discovery_dir.to_str().unwrap().to_string();

        // Actor 1 removes the directory
        let result1 = cleanup_discovery_dir(&dir_str).unwrap();
        assert_eq!(result1, CleanupOutcome::Removed);

        // Actor 2 tries to remove the same directory — should not error
        let result2 = cleanup_discovery_dir(&dir_str).unwrap();
        assert_eq!(result2, CleanupOutcome::AlreadyGone);
    }

    #[test]
    fn test_concurrent_orphan_scan_with_async_cleanup() {
        // Simulates the exact race from the issue:
        // 1. Discovery A finishes, cleanup_discovery_dir removes the dir atomically
        // 2. Discovery B starts, orphan scanner runs — dir is already gone
        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());

        // Discovery A's directory (being cleaned up)
        let dir_a = base.join("discovery-a");
        fs::create_dir(&dir_a).unwrap();
        File::create(dir_a.join(LOCK_FILE_NAME)).unwrap();
        File::create(dir_a.join("discovery_data.parquet")).unwrap();

        // Discovery B's active directory
        let dir_b = base.join("discovery-b");
        fs::create_dir(&dir_b).unwrap();
        File::create(dir_b.join(LOCK_FILE_NAME)).unwrap();

        // Step 1: Discovery A does atomic cleanup (removes entire dir including lock)
        let cleanup_result = cleanup_discovery_dir(dir_a.to_str().unwrap()).unwrap();
        assert_eq!(cleanup_result, CleanupOutcome::Removed);

        // Step 2: Discovery B's orphan scanner runs — dir_a is gone, dir_b is active
        let orphan_result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();

        // dir_a doesn't show up because it's already gone (not in readdir)
        // dir_b is not orphaned (has lock file)
        assert_eq!(orphan_result.removed, 0);
        assert!(orphan_result.errors.is_empty());
        assert!(dir_b.exists(), "Active directory B should still exist");
    }

    #[test]
    fn test_lock_file_lifecycle_correct() {
        // Verifies acceptance criterion: lock file exists while discovery is
        // active, removed only after directory is gone.
        let base = TempDir::new().unwrap();
        let dir = base.path().join("lifecycle-test");
        fs::create_dir(&dir).unwrap();

        let lock_path = dir.join(LOCK_FILE_NAME);

        // Phase 1: Discovery active — lock file present
        File::create(&lock_path).unwrap();
        assert!(
            !is_directory_orphaned(&dir),
            "Should not be orphaned while lock exists"
        );

        // Phase 2: Atomic cleanup — removes entire dir including lock
        let result = cleanup_discovery_dir(dir.to_str().unwrap()).unwrap();
        assert_eq!(result, CleanupOutcome::Removed);

        // Both directory and lock file are gone simultaneously
        assert!(!dir.exists(), "Directory should be gone");
        assert!(
            !lock_path.exists(),
            "Lock file should be gone with directory"
        );
    }

    #[test]
    fn test_multiple_orphaned_dirs_cleaned() {
        let temp = TempDir::new().unwrap();
        let base = make_discovery_root(temp.path());

        // Create three orphaned directories
        for i in 0..3 {
            let dir = base.join(format!("orphan-{i}"));
            fs::create_dir(&dir).unwrap();
            File::create(dir.join("data.parquet")).unwrap();
        }

        // Create one active directory
        let active = base.join("active-dir");
        fs::create_dir(&active).unwrap();
        File::create(active.join(LOCK_FILE_NAME)).unwrap();

        let result = clean_orphaned_discovery_dirs(base.to_str().unwrap()).unwrap();
        assert_eq!(result.removed, 3);
        assert!(result.errors.is_empty());
        assert!(active.exists(), "Active directory should remain");
    }
}
