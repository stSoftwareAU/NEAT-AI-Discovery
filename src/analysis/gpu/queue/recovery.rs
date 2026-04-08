//! GPU device-lost recovery with automatic retry (Issue #647)
//!
//! When the GPU device is lost (e.g., macOS Metal timeout, driver reset, resource
//! exhaustion), this module detects the error, re-initialises the `GpuAnalyzer`,
//! and retries the failed work item. This allows unattended workers to recover
//! from transient GPU issues without restarting the entire discovery process.

/// Default maximum number of consecutive device-lost recovery attempts before
/// propagating the error to the caller.
pub const DEFAULT_GPU_RETRY_LIMIT: u32 = 3;

/// Environment variable name for configuring the GPU retry limit.
pub const GPU_RETRY_LIMIT_ENV: &str = "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT";

/// Initial backoff delay between GPU retry attempts (milliseconds).
pub const DEFAULT_BACKOFF_INITIAL_MS: u64 = 10;

/// Maximum backoff delay cap between GPU retry attempts (milliseconds).
pub const DEFAULT_BACKOFF_MAX_MS: u64 = 1_000;

/// Calculate the exponential backoff delay for a given retry attempt.
///
/// Uses exponential doubling: `initial_ms * 2^(attempt - 1)`, capped at
/// `max_ms`. Attempt numbers start at 1.
///
/// # Examples
///
/// With defaults (10ms initial, 1000ms cap):
/// - Attempt 1: 10ms
/// - Attempt 2: 20ms
/// - Attempt 3: 40ms
/// - Attempt 7: 640ms
/// - Attempt 8: 1000ms (capped)
pub fn backoff_delay_ms(attempt: u32, initial_ms: u64, max_ms: u64) -> u64 {
    let exponent = attempt.saturating_sub(1);
    let delay = initial_ms.saturating_mul(1u64.checked_shl(exponent).unwrap_or(u64::MAX));
    delay.min(max_ms)
}

/// Get the configured GPU retry limit.
///
/// Delegates to [`crate::config::gpu_retry_limit()`].
pub fn get_gpu_retry_limit() -> u32 {
    crate::config::gpu_retry_limit()
}

/// Check whether an error indicates the GPU device has been lost or is in an
/// unrecoverable state that warrants re-initialisation.
///
/// This inspects the error message chain for patterns known to indicate device
/// loss in wgpu (Metal, Vulkan, and DX12 backends).
pub fn is_device_lost_error(error: &anyhow::Error) -> bool {
    let msg = format!("{error:#}").to_lowercase();

    // wgpu device-lost error patterns across backends
    msg.contains("device is lost")
        || msg.contains("device lost")
        || msg.contains("device was lost")
        || msg.contains("gpu device poll error")
        || msg.contains("internal error")
        || msg.contains("out of memory")
        || msg.contains("allocation failed")
        || msg.contains("command buffer")
        || msg.contains("too many command buffers")
        || msg.contains("device creation failed")
        || msg.contains("gpu driver")
        || msg.contains("driver may be unresponsive")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_retry_limit() {
        assert_eq!(DEFAULT_GPU_RETRY_LIMIT, 3);
    }

    #[test]
    fn test_retry_limit_env_var_name() {
        assert_eq!(GPU_RETRY_LIMIT_ENV, "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT");
    }

    #[test]
    fn test_is_device_lost_error_detects_known_patterns() {
        let cases = vec![
            "Device is lost",
            "wgpu: device lost during operation",
            "GPU device poll error (warm-up): device was lost",
            "Out of memory allocating GPU buffer",
            "Internal error in GPU pipeline",
            "Too many command buffers in flight",
            "GPU driver may be unresponsive",
        ];

        for msg in cases {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                is_device_lost_error(&err),
                "Expected device-lost detection for: {msg}"
            );
        }
    }

    #[test]
    fn test_is_device_lost_error_ignores_non_device_errors() {
        let cases = vec![
            "Invalid input data",
            "Sample count too low",
            "Timeout waiting for response",
            "Channel disconnected",
        ];

        for msg in cases {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                !is_device_lost_error(&err),
                "Should not detect device-lost for: {msg}"
            );
        }
    }

    #[test]
    fn test_is_device_lost_error_case_insensitive() {
        let err = anyhow::anyhow!("DEVICE IS LOST");
        assert!(is_device_lost_error(&err));
    }

    #[test]
    fn test_is_device_lost_error_detects_allocation_failed() {
        let err = anyhow::anyhow!("GPU allocation failed for buffer");
        assert!(
            is_device_lost_error(&err),
            "Expected device-lost detection for allocation failed"
        );
    }

    #[test]
    fn test_backoff_delay_initial_attempt() {
        assert_eq!(backoff_delay_ms(1, 10, 1_000), 10);
    }

    #[test]
    fn test_backoff_delay_doubles_each_attempt() {
        assert_eq!(backoff_delay_ms(1, 10, 1_000), 10);
        assert_eq!(backoff_delay_ms(2, 10, 1_000), 20);
        assert_eq!(backoff_delay_ms(3, 10, 1_000), 40);
        assert_eq!(backoff_delay_ms(4, 10, 1_000), 80);
        assert_eq!(backoff_delay_ms(5, 10, 1_000), 160);
        assert_eq!(backoff_delay_ms(6, 10, 1_000), 320);
        assert_eq!(backoff_delay_ms(7, 10, 1_000), 640);
    }

    #[test]
    fn test_backoff_delay_respects_cap() {
        assert_eq!(backoff_delay_ms(8, 10, 1_000), 1_000);
        assert_eq!(backoff_delay_ms(9, 10, 1_000), 1_000);
        assert_eq!(backoff_delay_ms(20, 10, 1_000), 1_000);
    }

    #[test]
    fn test_backoff_delay_zero_attempt_returns_initial() {
        // Attempt 0 is treated as attempt 1 (saturating_sub prevents underflow)
        assert_eq!(backoff_delay_ms(0, 10, 1_000), 10);
    }

    #[test]
    fn test_backoff_delay_with_default_constants() {
        assert_eq!(
            backoff_delay_ms(1, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS),
            10
        );
        assert_eq!(
            backoff_delay_ms(3, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS),
            40
        );
        // Should cap at 1000ms
        assert_eq!(
            backoff_delay_ms(10, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS),
            1_000
        );
    }

    #[test]
    fn test_backoff_delay_large_attempt_does_not_overflow() {
        // Even with very large attempt numbers, should not panic
        let result = backoff_delay_ms(100, 10, 1_000);
        assert_eq!(result, 1_000);
    }

    #[test]
    fn test_default_backoff_constants() {
        assert_eq!(DEFAULT_BACKOFF_INITIAL_MS, 10);
        assert_eq!(DEFAULT_BACKOFF_MAX_MS, 1_000);
    }
}
