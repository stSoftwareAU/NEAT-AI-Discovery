//! Environment variable parsing utilities shared by every config accessor.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock, PoisonError};

/// Opt-outs already logged in this process, keyed by `(env, raw value)`
/// (Issue #2468).
///
/// Readers run several times per discovery pass, so each distinct opt-out is
/// logged once per process rather than on every call.
static WARNED_OPT_OUTS: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();

/// Returns `true` the first time `(env, raw)` is seen in this process.
fn first_opt_out_sighting(env: &str, raw: &str) -> bool {
    WARNED_OPT_OUTS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert((env.to_string(), raw.to_string()))
}

/// Log a `WARN` that an explicitly set environment value disables a guard
/// (Issue #2468).
///
/// Emitted at most once per distinct `(env, raw)` per process. `raw` is the
/// trimmed value as the operator wrote it, and `guard` names what is switched
/// off, e.g. `"the analysis wall-clock reserve"`.
pub(crate) fn warn_guard_disabled(env: &str, raw: &str, guard: &str) {
    if first_opt_out_sighting(env, raw) {
        tracing::warn!(env, value = raw, guard, "{env}={raw:?} disables {guard}");
    }
}

/// Log a `WARN` that an environment value does not parse, so the guard named
/// by `guard` is left off or uncapped (Issue #2468).
///
/// Emitted at most once per distinct `(env, raw)` per process. `guard` is the
/// clause that follows the value, e.g. `"the watchdog stays disabled"`.
pub(crate) fn warn_guard_invalid(env: &str, raw: &str, guard: &str) {
    if first_opt_out_sighting(env, raw) {
        tracing::warn!(
            env,
            value = raw,
            guard,
            "{env}={raw:?} is not a valid value; {guard}"
        );
    }
}

/// The trimmed raw value of `name`, or an empty string when unset (Issue #2468).
///
/// Used only on the opt-out path, to name the value the operator actually set.
pub(crate) fn env_trimmed(name: &str) -> String {
    std::env::var(name)
        .map(|v| v.trim().to_string())
        .unwrap_or_default()
}

/// Parse a boolean-style environment variable.
///
/// Truthy values: `"1"`, `"true"`, `"yes"` (case-insensitive, surrounding
/// whitespace trimmed). Every other value is false: `"0"`, `"false"`, `"no"`,
/// unset, empty, and any unrecognised value such as `"on"`, `"y"` or `"enabled"`.
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
    use super::{parse_env, warn_guard_disabled, warn_guard_invalid};
    use serial_test::serial;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::Registry;

    /// Counts `WARN` events emitted while it is the active subscriber.
    struct WarnCounter(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for WarnCounter {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            if *event.metadata().level() == tracing::Level::WARN {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Run `body` and return how many `WARN` events it emitted on this thread.
    fn count_warns(body: impl FnOnce()) -> usize {
        let warns = Arc::new(AtomicUsize::new(0));
        let subscriber = Registry::default().with(WarnCounter(Arc::clone(&warns)));
        tracing::subscriber::with_default(subscriber, body);
        warns.load(Ordering::SeqCst)
    }

    #[test]
    fn opt_out_warns_once_per_distinct_env_and_value() {
        // Unique to this test: the dedupe set is process-wide.
        const DEDUPE_ENV: &str = "NEAT_AI_DISCOVERY_HELPER_DEDUPE_DISABLED_TEST";
        let warns = count_warns(|| {
            warn_guard_disabled(DEDUPE_ENV, "0", "the helper test guard");
            // Same (env, value): suppressed.
            warn_guard_disabled(DEDUPE_ENV, "0", "the helper test guard");
            // Different value for the same env: warns again.
            warn_guard_disabled(DEDUPE_ENV, "00", "the helper test guard");
        });
        assert_eq!(warns, 2);
    }

    #[test]
    fn invalid_value_warns_once_per_distinct_env_and_value() {
        const DEDUPE_ENV: &str = "NEAT_AI_DISCOVERY_HELPER_DEDUPE_INVALID_TEST";
        let warns = count_warns(|| {
            warn_guard_invalid(DEDUPE_ENV, "3O", "the helper test guard");
            warn_guard_invalid(DEDUPE_ENV, "3O", "the helper test guard");
            warn_guard_invalid(DEDUPE_ENV, "3o", "the helper test guard");
        });
        assert_eq!(warns, 2);
    }

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
