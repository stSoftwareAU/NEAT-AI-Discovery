//! Issue #1916: the internal `stSoftwareAU/*` quarantine bypass in
//! `renovate.json` was both keyed on a removed config key and positioned
//! where a later rule overrode it.
//!
//! Two independent faults, both silent:
//!
//!   1. **Ordering.** Renovate merges `packageRules` in array order and the
//!      *last* matching rule wins. The bypass sat before the
//!      `github-actions` rule, so a first-party `stSoftwareAU/*` Action —
//!      the one dependency class this repository actually consumes from
//!      stSoftwareAU — resolved to `24h`, not the documented `0`.
//!   2. **Removed key.** `matchSourceUrlPrefixes` was deprecated in favour
//!      of `matchSourceUrls` and removed in Renovate 40. On a current
//!      Renovate the rule is a validation error or a silently non-matching
//!      no-op, so neither the bypass nor the crates.io rule selects
//!      anything.
//!
//! These tests resolve the effective `minimumReleaseAge` the way Renovate
//! does — last matching rule wins — and assert on the *outcome* for
//! representative dependencies, so they keep working if the rules are
//! rewritten as long as the resolved policy holds. `rule_matches` refuses
//! to evaluate an unrecognised `match*` key rather than treating it as
//! "matches everything", so reintroducing `matchSourceUrlPrefixes` fails
//! loudly instead of quietly disabling a control.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

/// A dependency as Renovate would present it to the rule matcher.
struct Dependency<'a> {
    manager: &'a str,
    datasource: Option<&'a str>,
    source_url: Option<&'a str>,
}

/// Match a `matchSourceUrls` glob against a source URL.
///
/// Only the `<prefix>/**` form this repository uses is supported; any
/// other pattern panics rather than silently failing to match.
fn source_url_glob_matches(pattern: &str, source_url: &str) -> bool {
    let prefix = pattern.strip_suffix("**").unwrap_or_else(|| {
        panic!(
            "unsupported matchSourceUrls pattern `{pattern}` — this repository \
             only uses the `<prefix>/**` form (Issue #1916)"
        )
    });
    source_url.starts_with(prefix)
}

/// Does `rule` select `dep`?
///
/// A rule with no `match*` selector matches everything. An unrecognised
/// `match*` key is a hard error: an unmatched selector would widen the
/// rule to every dependency, which is exactly how a removed key such as
/// `matchSourceUrlPrefixes` turns a control into a no-op.
fn rule_matches(rule: &Value, dep: &Dependency<'_>) -> bool {
    let object = rule.as_object().expect("packageRule must be an object");
    for key in object.keys() {
        assert!(
            !key.starts_with("match")
                || matches!(
                    key.as_str(),
                    "matchManagers" | "matchSourceUrls" | "matchDatasources"
                ),
            "packageRule uses unsupported selector `{key}` — \
             `matchSourceUrlPrefixes` was removed in Renovate 40 and must not \
             be reintroduced (Issue #1916)"
        );
    }

    if let Some(managers) = object.get("matchManagers").and_then(Value::as_array)
        && !managers
            .iter()
            .any(|manager| manager.as_str() == Some(dep.manager))
    {
        return false;
    }

    if let Some(datasources) = object.get("matchDatasources").and_then(Value::as_array) {
        let Some(datasource) = dep.datasource else {
            return false;
        };
        if !datasources
            .iter()
            .any(|candidate| candidate.as_str() == Some(datasource))
        {
            return false;
        }
    }

    if let Some(patterns) = object.get("matchSourceUrls").and_then(Value::as_array) {
        let Some(source_url) = dep.source_url else {
            return false;
        };
        if !patterns.iter().any(|pattern| {
            source_url_glob_matches(
                pattern
                    .as_str()
                    .expect("matchSourceUrls entry must be a string"),
                source_url,
            )
        }) {
            return false;
        }
    }

    true
}

/// Resolve the effective `minimumReleaseAge` for `dep` using Renovate's
/// merge order: rules are applied in array order and the last matching
/// rule wins.
fn effective_release_age<'a>(config: &'a Value, dep: &Dependency<'_>) -> Option<&'a str> {
    let mut resolved = None;
    for rule in config["packageRules"]
        .as_array()
        .expect("packageRules array")
    {
        if rule_matches(rule, dep) {
            resolved = rule.get("minimumReleaseAge").and_then(Value::as_str);
        }
    }
    resolved
}

fn load_renovate_config() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("renovate.json");
    let contents = fs::read_to_string(&path).expect("read renovate.json");
    serde_json::from_str(&contents).expect("renovate.json must be valid JSON")
}

/// Recursively collect every object key used anywhere in `value`.
fn collect_keys(value: &Value, keys: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                keys.push(key.clone());
                collect_keys(child, keys);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_keys(item, keys);
            }
        }
        _ => {}
    }
}

#[test]
fn removed_match_source_url_prefixes_key_is_absent() {
    let config = load_renovate_config();
    let mut keys = Vec::new();
    collect_keys(&config, &mut keys);
    assert!(
        !keys.iter().any(|key| key == "matchSourceUrlPrefixes"),
        "renovate.json must not use `matchSourceUrlPrefixes` — it was \
         deprecated in favour of `matchSourceUrls` and removed in \
         Renovate 40, where it is a validation error or a silently \
         non-matching rule (Issue #1916)"
    );
}

#[test]
fn internal_bypass_is_the_last_package_rule() {
    let config = load_renovate_config();
    let rules = config["packageRules"]
        .as_array()
        .expect("packageRules array");
    let last = rules.last().expect("packageRules must not be empty");
    assert_eq!(
        last.get("matchSourceUrls"),
        Some(&json!(["https://github.com/stSoftwareAU/**"])),
        "the internal stSoftwareAU/* bypass must be the final packageRule — \
         the last matching rule wins, so any rule placed after it overrides \
         the bypass back to the quarantine window (Issue #1916)"
    );
    assert_eq!(
        last.get("minimumReleaseAge").and_then(Value::as_str),
        Some("0"),
        "the final packageRule must be the zero-window internal bypass \
         (Issue #1916)"
    );
}

#[test]
fn internal_first_party_dependencies_resolve_to_a_zero_window() {
    let config = load_renovate_config();
    for manager in ["github-actions", "cargo"] {
        let dep = Dependency {
            manager,
            datasource: None,
            source_url: Some("https://github.com/stSoftwareAU/NEAT-AI-Discovery"),
        };
        assert_eq!(
            effective_release_age(&config, &dep),
            Some("0"),
            "a first-party stSoftwareAU/* `{manager}` dependency must bypass \
             the quarantine (Issue #1916)"
        );
    }
}

#[test]
fn external_dependencies_still_resolve_to_the_24h_window() {
    let config = load_renovate_config();
    for (manager, datasource, source_url) in [
        (
            "github-actions",
            None,
            Some("https://github.com/actions/checkout"),
        ),
        ("cargo", None, Some("https://crates.io/crates/serde")),
        ("cargo", None, Some("https://github.com/serde-rs/serde")),
        ("cargo", None, None),
        ("pip_requirements", None, None),
        ("custom.regex", Some("npm"), None),
    ] {
        let dep = Dependency {
            manager,
            datasource,
            source_url,
        };
        assert_eq!(
            effective_release_age(&config, &dep),
            Some("24h"),
            "external `{manager}` dependency ({source_url:?}) must keep the \
             24h quarantine window (Issue #1234)"
        );
    }
}

#[test]
fn crates_io_and_default_cargo_rules_still_carry_24h() {
    let config = load_renovate_config();
    let rules = config["packageRules"]
        .as_array()
        .expect("packageRules array");

    let crates_io = rules
        .iter()
        .find(|rule| {
            rule.get("matchSourceUrls")
                .and_then(Value::as_array)
                .is_some_and(|patterns| {
                    patterns
                        .iter()
                        .any(|pattern| pattern.as_str() == Some("https://crates.io/**"))
                })
        })
        .expect("renovate.json must carry a crates.io cargo packageRule");
    assert_eq!(
        crates_io.get("minimumReleaseAge").and_then(Value::as_str),
        Some("24h"),
        "the crates.io rule must keep the 24h quarantine (Issue #1234)"
    );

    let cargo_default = rules
        .iter()
        .find(|rule| {
            rule.get("matchManagers") == Some(&json!(["cargo"]))
                && rule.get("matchSourceUrls").is_none()
        })
        .expect("renovate.json must carry a default cargo packageRule");
    assert_eq!(
        cargo_default
            .get("minimumReleaseAge")
            .and_then(Value::as_str),
        Some("24h"),
        "the default cargo rule must keep the 24h quarantine (Issue #1234)"
    );
}

#[test]
fn moving_the_bypass_before_a_manager_rule_is_detected() {
    // The pre-fix layout: the bypass sits before the github-actions rule,
    // which then overrides it. Reproduces the reported defect.
    let mut config = load_renovate_config();
    let rules = config["packageRules"]
        .as_array_mut()
        .expect("packageRules array");
    let bypass = rules.pop().expect("bypass rule");
    let github_actions_index = rules
        .iter()
        .position(|rule| rule.get("matchManagers") == Some(&json!(["github-actions"])))
        .expect("github-actions rule");
    rules.insert(github_actions_index, bypass);

    let dep = Dependency {
        manager: "github-actions",
        datasource: None,
        source_url: Some("https://github.com/stSoftwareAU/some-action"),
    };
    assert_eq!(
        effective_release_age(&config, &dep),
        Some("24h"),
        "this fixture must reproduce the Issue #1916 defect — with the \
         bypass placed before the github-actions rule, the bypass is \
         overridden; the real config must therefore keep it last"
    );
}

#[test]
#[serial_test::serial]
fn reintroducing_the_removed_prefix_key_panics_the_matcher() {
    let mut config = load_renovate_config();
    let rules = config["packageRules"]
        .as_array_mut()
        .expect("packageRules array");
    let last = rules
        .last_mut()
        .expect("bypass rule")
        .as_object_mut()
        .expect("object");
    let patterns = last.remove("matchSourceUrls").expect("matchSourceUrls");
    last.insert("matchSourceUrlPrefixes".to_string(), patterns);

    let dep = Dependency {
        manager: "cargo",
        datasource: None,
        source_url: Some("https://github.com/stSoftwareAU/NEAT-AI-Discovery"),
    };
    // Silence the expected panic's backtrace so the test output stays readable.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        effective_release_age(&config, &dep)
    }));
    std::panic::set_hook(previous_hook);
    assert!(
        outcome.is_err(),
        "a rule keyed on the removed `matchSourceUrlPrefixes` must fail \
         loudly rather than be silently ignored (Issue #1916)"
    );
}

#[test]
fn ci_validates_the_renovate_config() {
    let workflow =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/renovate-validate.yml");
    let body = fs::read_to_string(&workflow).unwrap_or_else(|e| {
        panic!(
            "read {} — CI must run renovate-config-validator so an invalid \
             or migrated-away key fails the PR (Issue #1916): {e}",
            workflow.display()
        )
    });
    assert!(
        body.contains("renovate-config-validator --strict"),
        "the validator must run with --strict so deprecated keys Renovate \
         would auto-migrate are reported as failures (Issue #1916)"
    );
    assert!(
        body.contains("--ignore-scripts renovate@"),
        "the renovate CLI must be installed at a pinned version with \
         lifecycle scripts disabled (Issue #1484)"
    );
}
