//! Private scratch directory for the external sampler's output (Issue #1905).
//!
//! The sampler is handed a path to write to, and whatever lands there is read
//! back and echoed into the thread dump — which lands in the operator's log. The
//! old path was `$TMPDIR/neat_ai_discovery.sample.<pid>.<millis>.txt`: both
//! components observable, in a world-writable directory, opened by an external
//! process that never got to pass `O_EXCL`. A local user who won the race chose
//! what the operator read.
//!
//! Two defences, because neither is sufficient alone:
//!
//! 1. [`SampleDir`] creates a per-invocation directory with mode `0700`, which
//!    makes the inner filename irrelevant to an attacker, and removes it on
//!    `Drop` — so every exit path, including the kill-on-timeout one, cleans up.
//! 2. [`read_guarded`] refuses anything at the output path that is not a regular
//!    file owned by this effective uid, so a planted symlink is reported rather
//!    than followed.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Owner-only access: the directory's mode is what makes the inner filename
/// uninteresting to an attacker.
#[cfg(unix)]
const PRIVATE_DIR_MODE: u32 = 0o700;

/// Fixed name of the capture file inside the private directory.
const CAPTURE_FILE_NAME: &str = "sample.txt";

/// How many name collisions to tolerate before giving up on a capture.
const MAX_CREATE_ATTEMPTS: u32 = 8;

/// A per-invocation directory holding one sampler capture, removed on `Drop`.
pub(super) struct SampleDir {
    dir: PathBuf,
}

impl SampleDir {
    /// Create an owner-only directory under the system temp directory.
    ///
    /// Creation is non-recursive, so an already-present path fails with
    /// `AlreadyExists` rather than silently adopting whatever is there; the
    /// caller simply tries the next name.
    pub(super) fn create(pid: u32) -> io::Result<Self> {
        let base = std::env::temp_dir();
        let mut last_err = None;

        for attempt in 0..MAX_CREATE_ATTEMPTS {
            let dir = base.join(format!(
                "neat_ai_discovery.sample.{pid}.{}.{attempt}.d",
                unique_suffix()
            ));
            match create_private_dir(&dir) {
                Ok(()) => return Ok(Self { dir }),
                Err(e) => last_err = Some(e),
            }
        }

        Err(last_err.unwrap_or_else(|| {
            io::Error::other("no private capture directory could be created".to_string())
        }))
    }

    /// Where the sampler is told to write.
    pub(super) fn capture_path(&self) -> PathBuf {
        self.dir.join(CAPTURE_FILE_NAME)
    }

    #[cfg(test)]
    pub(super) fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for SampleDir {
    fn drop(&mut self) {
        match std::fs::remove_dir_all(&self.dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            // Fail loud: leaked scratch directories on a fleet host are the
            // observable symptom, so say so rather than accumulating silently.
            Err(e) => eprintln!(
                "[NEAT-AI-Discovery][debug] WARNING: failed to remove sampler capture directory {}: {e}",
                self.dir.display()
            ),
        }
    }
}

/// Create `dir` exclusively, owner-only where the platform has file modes.
///
/// The mode is re-applied explicitly because `mkdir` masks the requested mode
/// with the process umask — the same belt-and-braces as the runtime-directory
/// preparation in `analysis::utils::platform` (Issue #1904).
#[cfg(unix)]
fn create_private_dir(dir: &Path) -> io::Result<()> {
    use std::fs::{DirBuilder, Permissions, set_permissions};
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    DirBuilder::new()
        .mode(PRIVATE_DIR_MODE)
        .recursive(false)
        .create(dir)?;
    set_permissions(dir, Permissions::from_mode(PRIVATE_DIR_MODE))
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> io::Result<()> {
    std::fs::DirBuilder::new().recursive(false).create(dir)
}

/// Read `path` only when it is a regular file owned by this effective uid.
///
/// [`std::fs::symlink_metadata`] does not traverse the final component, so a
/// symlink planted at the capture path is classified as a symlink and refused
/// instead of being followed to whatever it points at.
pub(super) fn read_guarded(path: &Path) -> io::Result<String> {
    let metadata = std::fs::symlink_metadata(path)?;

    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is a {}, not a regular file",
                path.display(),
                describe(&metadata)
            ),
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;

        // SAFETY: `geteuid` takes no arguments, touches no caller-owned memory
        // and cannot fail; it is only `unsafe` because it is an FFI call.
        let euid = unsafe { libc::geteuid() };
        let owner = metadata.uid();
        if owner != euid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "{} is owned by uid {owner}, not by this process (uid {euid})",
                    path.display()
                ),
            ));
        }
    }

    std::fs::read_to_string(path)
}

/// Human-readable file kind, for the refusal message.
fn describe(metadata: &std::fs::Metadata) -> &'static str {
    if metadata.is_symlink() {
        "symlink"
    } else if metadata.is_dir() {
        "directory"
    } else {
        "special file"
    }
}

/// A per-invocation suffix: wall-clock nanoseconds plus a process-local counter,
/// so two dumps in the same nanosecond still get distinct names.
fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{nanos}.{seq}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The containing directory is owner-only — that is what makes the inner
    /// filename irrelevant to an attacker who can write to the temp directory.
    #[test]
    #[cfg(unix)]
    fn the_capture_directory_is_created_with_mode_0700() {
        use std::os::unix::fs::PermissionsExt;

        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let metadata = std::fs::metadata(dir.path()).expect("metadata");

        assert!(metadata.is_dir(), "the capture path must be a directory");
        assert_eq!(
            metadata.permissions().mode() & 0o777,
            0o700,
            "the capture directory must be owner-only"
        );
        assert_eq!(
            dir.capture_path().parent(),
            Some(dir.path()),
            "the capture file must live inside the private directory"
        );
    }

    /// Two captures never collide, so a second dump cannot adopt the first's
    /// directory.
    #[test]
    fn two_capture_directories_are_distinct() {
        let first = SampleDir::create(std::process::id()).expect("first");
        let second = SampleDir::create(std::process::id()).expect("second");
        assert_ne!(first.path(), second.path());
    }

    /// The directory is removed on `Drop`, contents and all — this is what makes
    /// the kill-on-timeout path leak-free.
    #[test]
    fn dropping_the_capture_directory_removes_it_and_its_contents() {
        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let path = dir.path().to_path_buf();
        let capture = dir.capture_path();
        std::fs::write(&capture, "Call graph:\n").expect("write capture");

        drop(dir);

        assert!(
            !path.exists(),
            "the capture directory must not survive drop"
        );
        assert!(!capture.exists(), "the capture file must go with it");
    }

    /// A regular file we own reads back verbatim.
    #[test]
    fn a_regular_file_we_own_is_read() {
        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let capture = dir.capture_path();
        std::fs::write(&capture, "Call graph:\n").expect("write capture");

        assert_eq!(
            read_guarded(&capture).expect("our own file is readable"),
            "Call graph:\n"
        );
    }

    /// A symlink at the capture path is refused rather than followed — the
    /// planted-symlink attack this issue is about.
    #[test]
    #[cfg(unix)]
    fn a_symlink_at_the_capture_path_is_refused() {
        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let target = dir.path().join("secret.txt");
        std::fs::write(&target, "Thread_666: secret\n").expect("write target");

        let capture = dir.capture_path();
        std::os::unix::fs::symlink(&target, &capture).expect("plant symlink");

        let err = read_guarded(&capture).expect_err("a symlink must be refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("symlink"),
            "the refusal must name the reason: {err}"
        );
    }

    /// A directory at the capture path is refused too — anything that is not a
    /// regular file is a refusal, not a special case.
    #[test]
    fn a_directory_at_the_capture_path_is_refused() {
        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let capture = dir.capture_path();
        std::fs::create_dir(&capture).expect("plant directory");

        let err = read_guarded(&capture).expect_err("a directory must be refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// A missing capture is an ordinary `NotFound`, so the caller can tell it
    /// apart from a refusal.
    #[test]
    fn a_missing_capture_is_not_found() {
        let dir = SampleDir::create(std::process::id()).expect("create capture dir");
        let err = read_guarded(&dir.capture_path()).expect_err("nothing was written");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
