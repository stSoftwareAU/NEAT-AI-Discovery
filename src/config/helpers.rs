//! Boolean environment variable parsing utilities.

/// Parse a boolean-style environment variable.
///
/// Truthy values: `"1"`, `"true"`, `"yes"` (case-insensitive).
/// Falsy values: `"0"`, `"false"`, `"no"` (case-insensitive), unset, or empty.
pub(crate) fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Parse an optional boolean-style environment variable.
///
/// Returns `Some(true)` for truthy, `Some(false)` for falsy, `None` if unset.
pub(crate) fn parse_optional_bool_env(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|v| {
        let v = v.trim().to_lowercase();
        match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        }
    })
}
