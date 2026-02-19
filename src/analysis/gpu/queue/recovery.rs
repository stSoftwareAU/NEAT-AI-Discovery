//! GPU device-lost recovery with automatic retry (Issue #647)
//!
//! When the GPU device is lost (e.g., macOS Metal timeout, driver reset, resource
//! exhaustion), this module detects the error, re-initialises the `GpuAnalyzer`,
//! and retries the failed work item. This allows unattended workers to recover
//! from transient GPU issues without restarting the entire discovery process.

use std::sync::OnceLock;

/// Default maximum number of consecutive device-lost recovery attempts before
/// propagating the error to the caller.
pub const DEFAULT_GPU_RETRY_LIMIT: u32 = 3;

/// Environment variable name for configuring the GPU retry limit.
pub const GPU_RETRY_LIMIT_ENV: &str = "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT";

/// Get the configured GPU retry limit.
///
/// Reads from `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` environment variable on first
/// call and caches the result. Falls back to `DEFAULT_GPU_RETRY_LIMIT` if the
/// variable is not set or contains an invalid value.
pub fn get_gpu_retry_limit() -> u32 {
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var(GPU_RETRY_LIMIT_ENV)
            .ok()
            .and_then(|val| val.parse::<u32>().ok())
            .filter(|&n| n <= 10)
            .unwrap_or(DEFAULT_GPU_RETRY_LIMIT)
    })
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
        || msg.contains("a]location failed")
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
}
