## Summary

Add periodic cleanup of orphaned streaming sessions to prevent memory leaks when the host process crashes or fails to call `finish_discovery_session()` / `cancel_discovery_session()`. Closes #1084.

**Changes:**

- Added `created_at: Instant` field to `RecordingSession` to track session creation time
- Added `cleanup_stale_sessions()` function that removes sessions exceeding the configurable TTL
- Call `cleanup_stale_sessions()` at the start of `start_session()` (piggybacks on existing calls)
- Added `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` environment variable (default 3600s / 1 hour, clamped to 60-86400)
- Stale sessions are logged at `warn` level with session ID and age before removal
- Temp files are cleaned up via the existing `Drop` implementation on `RecordingSession`
- No impact on normal session lifecycle (finish/cancel still work as before)

## Evidence
All existing streaming tests continue to pass unchanged. Four new unit tests verify the cleanup behaviour.

## Test Plan
- `test_cleanup_stale_sessions_removes_expired` — verifies sessions older than TTL are removed
- `test_cleanup_stale_sessions_preserves_fresh` — verifies fresh sessions survive cleanup
- `test_cleanup_stale_sessions_cleans_temp_files` — verifies temp files are cleaned up when stale sessions are removed
- `test_start_session_triggers_cleanup` — verifies cleanup is automatically triggered when starting a new session
- `session_ttl_default_values` — verifies config constants
- `session_ttl_returns_valid_value` — verifies config accessor returns a value within valid range
