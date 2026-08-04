//! Environment variable parsing utilities shared by every config accessor.

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

/// Parse a numeric `NEAT_AI_DISCOVERY_*` override (Issue #2006).
///
/// This is the single definition of how an environment override becomes a
/// number: **missing → `None`, surrounding whitespace tolerated, unparsable →
/// `None`.** Whitespace tolerance matters because operators supply values from
/// shell heredocs (trailing newline) and YAML env blocks (`VAR: " 5 "`).
///
/// The per-knob policy that follows — `unwrap_or`, `filter`, `clamp` — is *not*
/// shared and stays at each call site.
pub(crate) fn parse_env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::parse_env;
    use serial_test::serial;

    const ENV: &str = "NEAT_AI_DISCOVERY_PARSE_ENV_HELPER_TEST";

    /// Set `ENV` to `value`, run `body`, then remove it again.
    ///
    /// `ENV` is unique to these tests, so nothing outside them observes it.
    fn with_env<T>(value: &str, body: impl FnOnce() -> T) -> T {
        // SAFETY: Serialised via #[serial].
        unsafe { std::env::set_var(ENV, value) };
        let result = body();
        // SAFETY: Serialised via #[serial].
        unsafe { std::env::remove_var(ENV) };
        result
    }

    #[test]
    #[serial]
    fn unset_variable_is_none() {
        // SAFETY: Serialised via #[serial].
        unsafe { std::env::remove_var(ENV) };
        assert_eq!(parse_env::<u64>(ENV), None);
    }

    #[test]
    #[serial]
    fn parses_a_clean_value() {
        assert_eq!(with_env("42", || parse_env::<u64>(ENV)), Some(42));
        assert_eq!(with_env("-1.5", || parse_env::<f32>(ENV)), Some(-1.5));
    }

    #[test]
    #[serial]
    fn tolerates_surrounding_whitespace() {
        assert_eq!(with_env("  42  ", || parse_env::<u64>(ENV)), Some(42));
        assert_eq!(with_env("42\n", || parse_env::<usize>(ENV)), Some(42));
        assert_eq!(with_env("\t0.25\t", || parse_env::<f32>(ENV)), Some(0.25));
    }

    #[test]
    #[serial]
    fn unparsable_and_empty_values_are_none() {
        assert_eq!(with_env("abc", || parse_env::<u64>(ENV)), None);
        assert_eq!(with_env("", || parse_env::<u64>(ENV)), None);
        assert_eq!(with_env("   ", || parse_env::<u64>(ENV)), None);
        // Wrong type for the target: a float never parses as an integer.
        assert_eq!(with_env("3.5", || parse_env::<u32>(ENV)), None);
        // Negative values do not fit an unsigned target.
        assert_eq!(with_env("-3", || parse_env::<u32>(ENV)), None);
        // Overflow of the target type is unparsable, not a saturating clamp.
        assert_eq!(with_env("999", || parse_env::<u8>(ENV)), None);
    }

    #[test]
    #[serial]
    fn interior_whitespace_is_not_stripped() {
        // Only the ends are trimmed — `4 2` is a typo, not the number 42.
        assert_eq!(with_env(" 4 2 ", || parse_env::<u64>(ENV)), None);
    }
}
