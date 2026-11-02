//! Isolated symlink regression: this binary has exactly one input-writer caller.
// Keep it separate so the process-wide input counter deterministically starts at zero.
#[path = "../fuzz/footer_input.rs"]
mod footer_input;

#[cfg(unix)]
use footer_input::with_input;
#[cfg(unix)]
use std::fs;

#[test]
#[cfg(unix)]
fn input_is_private_and_preserves_preexisting_symlink_targets() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let sentinel = outside.path().join("sentinel");
    fs::write(&sentinel, b"unchanged").unwrap();
    let link = root
        .path()
        .join(format!("neat-ai-footer-fuzz-{}-0", std::process::id()));
    symlink(&sentinel, &link).unwrap();
    let old_link = root.path().join(format!(
        "neat-ai-footer-fuzz-{}-0.parquet",
        std::process::id()
    ));
    symlink(&sentinel, &old_link).unwrap();
    let mut input_path = None;
    with_input(root.path(), b"input", |path| {
        assert_eq!(
            path.parent().unwrap().file_name().unwrap(),
            format!("neat-ai-footer-fuzz-{}-1", std::process::id()).as_str()
        );
        assert!(path.starts_with(root.path()));
        assert_eq!(fs::read(path).unwrap(), b"input");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(path.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        input_path = Some(path.to_path_buf());
    })
    .unwrap();
    assert!(!input_path.unwrap().exists());
    assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
    assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    assert!(
        fs::symlink_metadata(old_link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
}
