//! Finding #2463 — `get_constant_source_threshold` ignored an explicit
//! `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD=1e-7`. Fixed by Issue #2472.
//!
//! `analysis::samples::get_constant_source_threshold` used to honour the env
//! override only when it differed from `DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD`
//! (`src/analysis/samples/thresholds.rs`, the `Some(v) if v != DEFAULT` arm). An
//! explicit `1e-7`, which equals the default, fell through to the dynamic threshold
//! and returned `DEFAULT x max(1, avg / 0.05)`. With `source_std_dev_avg = Some(0.5)`
//! that is about `1e-6`, not the `1e-7` the operator asked for.
//!
//! The production function now delegates to its twin,
//! `config::constant_source_threshold_with_dynamic` (`src/config/user_facing.rs`),
//! which honours an explicit `1e-7` and returns `Some(1e-7)`. The two functions
//! agree on every input in this file.
//!
//! `explicit_default_override_is_honoured_like_its_twin` is the regression test for
//! the finding. The other tests pin the rest of the behaviour table (unset, empty,
//! unparsable, whitespace-padded, a non-default override and `0`) for both functions.

use neat_ai_discovery::analysis::samples::get_constant_source_threshold;
use neat_ai_discovery::config::{
    DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, constant_source_threshold_with_dynamic,
};
use serial_test::serial;
use std::ffi::OsString;

/// The env var both functions read.
const VAR: &str = "NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD";

/// Source variance used throughout. Its dynamic threshold is about `1e-6`,
/// which is distinct from the `1e-7` default, so the fixture can tell the two apart.
const AVG: Option<f32> = Some(0.5);

/// RAII guard that restores an env var on drop. Every caller is `#[serial]`,
/// so the process-wide mutation is safe.
struct EnvGuard {
    key: String,
    prior: Option<OsString>,
}

impl EnvGuard {
    fn set(key: &str, value: &str) -> Self {
        let prior = std::env::var_os(key);
        // SAFETY: serialised via #[serial]
        unsafe { std::env::set_var(key, value) };
        Self {
            key: key.to_string(),
            prior,
        }
    }

    fn unset(key: &str) -> Self {
        let prior = std::env::var_os(key);
        // SAFETY: serialised via #[serial]
        unsafe { std::env::remove_var(key) };
        Self {
            key: key.to_string(),
            prior,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: serialised via #[serial]
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Compare two `Option<f32>` thresholds: both `None`, or both `Some` and within
/// `1e-12` of each other.
fn close(actual: Option<f32>, expected: Option<f32>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(a), Some(e)) => (a - e).abs() < 1e-12,
        _ => false,
    }
}

/// Run both functions with `VAR` set to `value`, or unset when `value` is `None`.
/// Returns `(get_constant_source_threshold, constant_source_threshold_with_dynamic)`.
fn both_under(value: Option<&str>) -> (Option<f32>, Option<f32>) {
    let _guard = match value {
        Some(v) => EnvGuard::set(VAR, v),
        None => EnvGuard::unset(VAR),
    };
    (
        get_constant_source_threshold(AVG),
        constant_source_threshold_with_dynamic(AVG),
    )
}

/// The dynamic value both functions return when no override applies.
fn dynamic_value() -> Option<f32> {
    let (production, twin) = both_under(None);
    assert!(
        close(production, twin),
        "with {VAR} unset, get_constant_source_threshold returned {production:?} but \
         constant_source_threshold_with_dynamic returned {twin:?}; they must agree"
    );
    production
}

#[test]
#[serial]
fn unset_returns_the_dynamic_value_in_both_functions() {
    let dynamic = dynamic_value();
    let value = dynamic.expect("the dynamic threshold must be Some for a positive variance");
    assert!(
        (value - DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD).abs() > 1e-8,
        "fixture cannot discriminate: the dynamic value {value:e} is too close to the \
         default {DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD:e}, so the default-override \
         probe would prove nothing"
    );
    assert!(
        value > 5e-7 && value < 5e-6,
        "with source_std_dev_avg Some(0.5) and {VAR} unset, the dynamic threshold must be \
         about 1e-6, got {value:e}"
    );
}

#[test]
#[serial]
fn empty_and_unparsable_overrides_fall_back_to_the_dynamic_value() {
    let dynamic = dynamic_value();
    for raw in ["", "abc"] {
        let (production, twin) = both_under(Some(raw));
        assert!(
            close(production, dynamic),
            "{VAR}={raw:?}: get_constant_source_threshold returned {production:?}, \
             expected the unset dynamic value {dynamic:?}"
        );
        assert!(
            close(twin, dynamic),
            "{VAR}={raw:?}: constant_source_threshold_with_dynamic returned {twin:?}, \
             expected the unset dynamic value {dynamic:?}"
        );
    }
}

#[test]
#[serial]
fn explicit_non_default_override_is_honoured_by_both_functions() {
    let (production, twin) = both_under(Some("2e-7"));
    let expected = Some(2e-7_f32);
    assert!(
        close(production, expected),
        "{VAR}=\"2e-7\": get_constant_source_threshold returned {production:?}, expected {expected:?}"
    );
    assert!(
        close(twin, expected),
        "{VAR}=\"2e-7\": constant_source_threshold_with_dynamic returned {twin:?}, expected {expected:?}"
    );
}

#[test]
#[serial]
fn explicit_zero_disables_folding_in_both_functions() {
    let (production, twin) = both_under(Some("0"));
    assert!(
        close(production, None),
        "{VAR}=\"0\": get_constant_source_threshold returned {production:?}, expected None (folding disabled)"
    );
    assert!(
        close(twin, None),
        "{VAR}=\"0\": constant_source_threshold_with_dynamic returned {twin:?}, expected None (folding disabled)"
    );
}

/// Finding #2463, fixed by Issue #2472. The twin honours an explicit `1e-7` and
/// returns `Some(1e-7)`. The production function now delegates to the twin, so it
/// returns the same explicit value rather than the dynamic value (about `1e-6`).
#[test]
#[serial]
fn explicit_default_override_is_honoured_like_its_twin() {
    let expected = Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    let (production, twin) = both_under(Some("1e-7"));
    assert!(
        close(twin, expected),
        "{VAR}=\"1e-7\": constant_source_threshold_with_dynamic returned {twin:?}, expected Some(1e-7)"
    );
    assert!(
        close(production, expected),
        "finding #2463: {VAR}=\"1e-7\" with source_std_dev_avg Some(0.5): \
         get_constant_source_threshold returned {production:?}, expected Some(1e-7) \
         (its twin constant_source_threshold_with_dynamic returned {twin:?}); the explicit \
         override equal to the default was ignored and the dynamic threshold used instead"
    );
}

/// Finding #2463 (whitespace variant): surrounding whitespace is trimmed and an
/// explicit `1e-7` is honoured by both functions.
#[test]
#[serial]
fn explicit_default_override_with_whitespace_is_honoured() {
    let expected = Some(1e-7_f32);
    let (production, twin) = both_under(Some(" 1e-7 "));
    assert!(
        close(production, expected),
        "{VAR}=\" 1e-7 \": get_constant_source_threshold returned {production:?}, expected {expected:?}"
    );
    assert!(
        close(twin, expected),
        "{VAR}=\" 1e-7 \": constant_source_threshold_with_dynamic returned {twin:?}, expected {expected:?}"
    );
}
