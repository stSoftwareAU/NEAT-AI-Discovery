//! Tests that validate the cache tier selection examples documented in
//! `docs/CACHE_TUNING.md`. These tests call the real `select_loading_strategy`
//! function with the scenarios from the documentation and verify the expected
//! tier is selected.
//!
//! Issue #1041: Ensures documentation stays in sync with implementation.

use neat_ai_discovery::analysis::cache::{LoadingStrategy, select_loading_strategy};

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

/// Documented example: 10 MB file, 8 GB RAM selects `PreloadAll`.
#[test]
fn doc_example_small_file_preload() {
    let strategy = select_loading_strategy(10 * MB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::PreloadAll),
        "10 MB file with 8 GB RAM should select PreloadAll, got {strategy:?}"
    );
}

/// Documented example: 500 MB file, 8 GB RAM selects `PreloadAll`
/// (expanded 1.5 GB < 2 GB threshold).
#[test]
fn doc_example_medium_file_preload() {
    let strategy = select_loading_strategy(500 * MB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::PreloadAll),
        "500 MB file with 8 GB RAM should select PreloadAll, got {strategy:?}"
    );
}

/// Documented example: 1 GB file, 8 GB RAM selects `LruCache`.
#[test]
fn doc_example_1gb_file_lru() {
    let strategy = select_loading_strategy(GB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::LruCache { .. }),
        "1 GB file with 8 GB RAM should select LruCache, got {strategy:?}"
    );
}

/// Documented example: 4 GB file, 8 GB RAM selects `Streaming`.
#[test]
fn doc_example_large_file_streaming() {
    let strategy = select_loading_strategy(4 * GB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::Streaming),
        "4 GB file with 8 GB RAM should select Streaming, got {strategy:?}"
    );
}

/// Documented example: 100 MB file, 1 GB RAM selects `LruCache`.
#[test]
fn doc_example_constrained_ram_lru() {
    let strategy = select_loading_strategy(100 * MB, GB);
    assert!(
        matches!(strategy, LoadingStrategy::LruCache { .. }),
        "100 MB file with 1 GB RAM should select LruCache, got {strategy:?}"
    );
}

/// Documented example: 500 MB file, 1 GB RAM selects `Streaming`.
#[test]
fn doc_example_constrained_ram_streaming() {
    let strategy = select_loading_strategy(500 * MB, GB);
    assert!(
        matches!(strategy, LoadingStrategy::Streaming),
        "500 MB file with 1 GB RAM should select Streaming, got {strategy:?}"
    );
}

/// Verify the documented decompression ratio (3x) is used in tier selection.
/// A file whose expanded size (`file_size * 3`) is just under `available_memory / 4`
/// should select `PreloadAll`.
#[test]
fn decompression_ratio_boundary_preload() {
    // available / 4 = 2 GB. estimated_expanded = file_size * 3.
    // file_size * 3 < 2 GB → file_size < 682 MB → use 600 MB
    let strategy = select_loading_strategy(600 * MB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::PreloadAll),
        "600 MB file (expanded 1.8 GB) with 8 GB RAM should select PreloadAll, got {strategy:?}"
    );
}

/// Verify that at the boundary between `PreloadAll` and LRU, a file whose
/// expanded size exceeds `available_memory / 4` selects LRU.
#[test]
fn decompression_ratio_boundary_lru() {
    // available / 4 = 2 GB. file_size * 3 > 2 GB → file_size > 682 MB → use 700 MB
    let strategy = select_loading_strategy(700 * MB, 8 * GB);
    assert!(
        matches!(strategy, LoadingStrategy::LruCache { .. }),
        "700 MB file (expanded 2.1 GB) with 8 GB RAM should select LruCache, got {strategy:?}"
    );
}
