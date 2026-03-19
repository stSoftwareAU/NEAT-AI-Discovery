//! Loading strategy selection for parquet files (Issue #215).
//!
//! Automatically selects between preload, LRU, and streaming modes based on
//! file size and available system memory.

#![allow(clippy::cast_possible_wrap)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
/// Loading strategy for parquet files based on file size and available memory.
///
/// Issue #215: Automatic selection between different caching modes to support
/// larger files while maintaining good performance for smaller ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadingStrategy {
    /// Load everything into memory upfront.
    /// Best for small files where `estimated_expanded` < `available_memory` / 4.
    PreloadAll,

    /// Keep frequently-accessed neurons in memory with LRU eviction.
    /// Best for medium-sized files that fit in memory but benefit from bounded usage.
    LruCache {
        /// Maximum bytes to keep in cache.
        capacity_bytes: usize,
    },

    /// Load records on-demand without caching.
    /// Best for very large files that exceed available memory.
    Streaming,
}

/// Decompression ratio estimate (parquet typically expands 2-4x).
const DECOMPRESSION_RATIO: u64 = 3;

/// Select the optimal loading strategy based on file size and available memory.
///
/// # Strategy Selection Logic
///
/// - **`PreloadAll`**: When estimated expanded size < `available_memory` / 4
///   (small files that easily fit in memory with plenty of headroom)
///
/// - **`LruCache`**: When estimated expanded size < `available_memory`
///   (medium files that fit but benefit from bounded memory usage)
///
/// - **Streaming**: When estimated expanded size >= `available_memory`
///   (large files that would exceed available memory)
///
/// # Arguments
///
/// * `file_size_bytes` - Size of the parquet file in bytes
/// * `available_memory_bytes` - Available system memory in bytes
pub fn select_loading_strategy(
    file_size_bytes: u64,
    available_memory_bytes: u64,
) -> LoadingStrategy {
    let estimated_expanded = file_size_bytes.saturating_mul(DECOMPRESSION_RATIO);

    // Small file threshold: fits comfortably in 1/4 of available memory
    let small_threshold = available_memory_bytes / 4;

    if estimated_expanded < small_threshold {
        LoadingStrategy::PreloadAll
    } else if estimated_expanded < available_memory_bytes {
        // Medium file: use LRU cache with half of available memory
        let capacity_bytes = (available_memory_bytes / 2) as usize;
        LoadingStrategy::LruCache { capacity_bytes }
    } else {
        // Large file: use streaming
        LoadingStrategy::Streaming
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_strategy_preload_for_small_files() {
        // 10MB file, 8GB available
        let file_size = 10 * 1024 * 1024;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::PreloadAll));
    }

    #[test]
    fn loading_strategy_lru_for_medium_files() {
        // 1GB file, 8GB available
        let file_size = 1024 * 1024 * 1024_u64;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::LruCache { .. }));
    }

    #[test]
    fn loading_strategy_streaming_for_large_files() {
        // 4GB file, 8GB available
        let file_size = 4 * 1024 * 1024 * 1024_u64;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::Streaming));
    }

    #[test]
    fn lru_capacity_scales_with_memory() {
        // 2GB file, 16GB available
        // Estimated expanded = 2GB * 3 = 6GB
        // Small threshold = 16GB / 4 = 4GB
        // 6GB > 4GB but < 16GB, so LruCache is selected
        let file_size = 2 * 1024 * 1024 * 1024_u64;
        let available = 16 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        if let LoadingStrategy::LruCache { capacity_bytes } = strategy {
            // Should be approximately 8GB (half of available)
            let expected = 8 * 1024 * 1024 * 1024_usize;
            let tolerance = expected / 10;
            assert!(
                (capacity_bytes as i64 - expected as i64).unsigned_abs() < tolerance as u64,
                "Capacity {capacity_bytes} should be ~{expected}"
            );
        } else {
            panic!("Expected LruCache strategy, got {strategy:?}");
        }
    }
}
