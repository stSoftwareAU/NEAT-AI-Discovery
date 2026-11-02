//! Security regression coverage for the footer fuzz input writer.
#[path = "../fuzz/footer_input.rs"]
mod footer_input;

use footer_input::{MAX_INPUT_BYTES, with_input};
use std::fs;

#[test]
fn input_limit_is_inclusive_and_oversized_inputs_do_not_touch_disk() {
    let root = tempfile::tempdir().unwrap();
    with_input(root.path(), &vec![42; MAX_INPUT_BYTES], |path| {
        assert!(path.starts_with(root.path()));
        assert_eq!(fs::read(path).unwrap(), vec![42; MAX_INPUT_BYTES]);
    })
    .unwrap();
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    with_input(
        &root.path().join("missing"),
        &vec![0; MAX_INPUT_BYTES + 1],
        |_| {
            panic!("oversized input reached parser");
        },
    )
    .unwrap();
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

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
