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
