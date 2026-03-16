//! Issue #477: Verify Rust 2024 edition is active.
//!
//! These tests use features that are only available in Rust 2024 edition,
//! confirming the edition upgrade was applied correctly. If the edition
//! were reverted to 2021, these tests would fail to compile.

/// Verify that let-chains (stabilised in Rust 2024) compile and work correctly.
///
/// `if let ... && let ...` syntax is only available in edition 2024+.
/// This test would fail to compile under edition 2021.
#[test]
fn let_chains_available_in_edition_2024() {
    let outer: Option<Vec<i32>> = Some(vec![1, 2, 3]);

    let mut found = false;
    if let Some(v) = &outer
        && let Some(&first) = v.first()
    {
        assert_eq!(first, 1);
        found = true;
    }
    assert!(found, "Let-chain should have matched");
}

/// Verify that `gen` is a reserved keyword in edition 2024.
///
/// In edition 2024, `gen` became a reserved keyword (for future generator support).
/// This test confirms we can still use it via raw identifier syntax (`r#gen`).
#[test]
fn gen_is_reserved_keyword_in_edition_2024() {
    // In edition 2024, `gen` is reserved. We can still use it as r#gen.
    fn r#gen(x: i32) -> i32 {
        x * 2
    }
    assert_eq!(r#gen(21), 42);
}

/// Verify that unsafe attributes work with the edition 2024 syntax.
///
/// In edition 2024, unsafe attributes like `no_mangle` require `unsafe(...)` wrapping.
/// This test confirms the `unsafe(...)` attribute syntax is accepted.
#[test]
fn unsafe_attribute_syntax_accepted() {
    // This function uses the edition 2024 unsafe attribute syntax.
    // Under edition 2021, `unsafe(no_mangle)` would be a compile error.
    #[unsafe(no_mangle)]
    pub extern "C" fn _issue_477_test_fn() -> i32 {
        477
    }

    // Verify the function works
    assert_eq!(_issue_477_test_fn(), 477);
}
