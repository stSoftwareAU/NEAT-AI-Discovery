//! Issue #1234: Dependency-update quarantine declared but never enforced.
//!
//! `bump-deps.sh` documents a `VIBE_BUMP_QUARANTINE_HOURS` window (default
//! 24h) intended to "dodge fast-flagged supply-chain attacks". Previously
//! neither tool actually filtered upgrades by publish age — a poisoned
//! version published shortly before the bump ran would land in
//! `Cargo.toml` immediately.
//!
//! This test enforces the configuration-level contract for the fix:
//!
//!   1. A Renovate config (`renovate.json`) exists at the repo root, and
//!      configures `minimumReleaseAge` of at least 24 hours for every
//!      dependency class the policy covers — defence in depth in case
//!      Renovate is enabled.
//!   2. The weekly scheduled upgrade workflow has been removed
//!      (Issue #1282). `bump-deps.sh` still runs on the per-PR path and
//!      is the single place where the quarantine gate is enforced.
//!   3. `bump-deps.sh` has wiring that fetches publish times from
//!      crates.io and reverts in-quarantine bumps — the helper is no
//!      longer dead code.
//!
//! Issue #1915 (business-logic change to check 1): the original
//! `renovate.json` assertions were substring matches over the whole file,
//! so *any* surviving `"24h"` anywhere satisfied them — deleting
//! `minimumReleaseAge` from the cargo rule left the assertion green while
//! the control was gone. The semgrep rule that would otherwise catch this
//! is excluded in `.github/workflows/semgrep.yml` (it hardcodes a 7-day
//! threshold and cannot express this org's 24h standard), so the substring
//! check was the only thing behind the exclusion. It is replaced below by
//! `audit_quarantine_policy`, which parses the JSON and checks each rule
//! structurally.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Structural quarantine-policy auditor (Issue #1915)
// ---------------------------------------------------------------------------

/// The quarantine window every covered `packageRule` must meet, in minutes.
const MIN_RELEASE_AGE_MINUTES: u64 = 24 * 60;

/// Dependency classes the quarantine policy covers, named by the Renovate
/// manager a `packageRule` must match to hold them.
const COVERED_MANAGERS: [&str; 3] = ["cargo", "github-actions", "custom.regex"];

/// Source-URL prefix identifying first-party deps exempt from the window.
const INTERNAL_SOURCE_PREFIX: &str = "https://github.com/stSoftwareAU/";

/// Parse a Renovate duration (`"24h"`, `"1 day"`, `"0"`) into minutes.
///
/// Unknown units are an error rather than a silently-accepted zero: an
/// unparsable window must fail the audit loudly, not pass it.
fn parse_release_age_minutes(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let boundary = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, unit) = trimmed.split_at(boundary);
    let value: u64 = digits
        .parse()
        .map_err(|_| format!("`{raw}` does not start with a whole number"))?;
    let unit = unit.trim().to_ascii_lowercase();
    let minutes_per_unit = match unit.as_str() {
        // A bare number is only meaningful as the documented "disabled" 0.
        "" if value == 0 => 1,
        "" => return Err(format!("`{raw}` omits a unit")),
        "minute" | "minutes" => 1,
        "h" | "hour" | "hours" => 60,
        "d" | "day" | "days" => 60 * 24,
        "w" | "week" | "weeks" => 60 * 24 * 7,
        other => return Err(format!("`{raw}` uses an unrecognised unit `{other}`")),
    };
    Ok(value * minutes_per_unit)
}

/// Is this rule the internal `stSoftwareAU/*` bypass — the only rule
/// permitted to set a zero window?
fn is_internal_bypass_rule(rule: &Value) -> bool {
    let Some(prefixes) = rule.get("matchSourceUrlPrefixes").and_then(Value::as_array) else {
        return false;
    };
    !prefixes.is_empty()
        && prefixes.iter().all(|prefix| {
            prefix
                .as_str()
                .is_some_and(|prefix| prefix.starts_with(INTERNAL_SOURCE_PREFIX))
        })
}

/// Audit a parsed `renovate.json` against the Issue #1234 quarantine policy.
///
/// Returns one message per violation; an empty vector means compliant.
fn audit_quarantine_policy(config: &Value) -> Vec<String> {
    let mut violations = Vec::new();

    let Some(rules) = config.get("packageRules").and_then(Value::as_array) else {
        violations.push("renovate.json has no `packageRules` array".to_string());
        return violations;
    };

    let mut covered: BTreeSet<&str> = BTreeSet::new();
    let mut internal_bypass_present = false;

    for (index, rule) in rules.iter().enumerate() {
        let label = rule
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("<no description>");
        let raw = match rule.get("minimumReleaseAge") {
            Some(Value::String(raw)) => raw.as_str(),
            Some(other) => {
                violations.push(format!(
                    "packageRule[{index}] ({label}) sets a non-string `minimumReleaseAge` ({other})"
                ));
                continue;
            }
            None => {
                violations.push(format!(
                    "packageRule[{index}] ({label}) does not set `minimumReleaseAge` — \
                     every rule must carry the quarantine window explicitly (Issue #1234)"
                ));
                continue;
            }
        };
        let minutes = match parse_release_age_minutes(raw) {
            Ok(minutes) => minutes,
            Err(reason) => {
                violations.push(format!("packageRule[{index}] ({label}): {reason}"));
                continue;
            }
        };

        if minutes == 0 {
            if is_internal_bypass_rule(rule) {
                internal_bypass_present = true;
            } else {
                violations.push(format!(
                    "packageRule[{index}] ({label}) disables the quarantine with \
                     `minimumReleaseAge: \"{raw}\"` but is not the {INTERNAL_SOURCE_PREFIX} \
                     internal-dependency bypass"
                ));
            }
            continue;
        }

        if minutes < MIN_RELEASE_AGE_MINUTES {
            violations.push(format!(
                "packageRule[{index}] ({label}) sets `minimumReleaseAge: \"{raw}\"` \
                 ({minutes} minutes), below the {MIN_RELEASE_AGE_MINUTES}-minute \
                 (24h) standard (Issue #1234)"
            ));
            continue;
        }

        for manager in rule
            .get("matchManagers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            covered.insert(manager);
        }
    }

    for manager in COVERED_MANAGERS {
        if !covered.contains(manager) {
            violations.push(format!(
                "no packageRule holds `{manager}` dependencies for at least 24h \
                 (Issue #1234)"
            ));
        }
    }

    if !internal_bypass_present {
        violations.push(format!(
            "no packageRule grants the {INTERNAL_SOURCE_PREFIX} internal-dependency \
             bypass (Issue #1234)"
        ));
    }

    violations
}

/// Read and parse the repository's `renovate.json`.
fn load_renovate_config() -> Value {
    let renovate = Path::new(env!("CARGO_MANIFEST_DIR")).join("renovate.json");
    assert!(
        renovate.is_file(),
        "renovate.json must exist at the repository root (Issue #1234) — \
         it is the defence-in-depth gate that delays externally-published \
         dependencies by at least 24h"
    );
    let contents = fs::read_to_string(&renovate).expect("read renovate.json");
    serde_json::from_str(&contents).expect("renovate.json must be valid JSON")
}

/// The crates.io cargo rule the substring assertions used to "protect".
fn crates_io_rule_index(config: &Value) -> usize {
    config["packageRules"]
        .as_array()
        .expect("packageRules array")
        .iter()
        .position(|rule| {
            rule.get("matchSourceUrlPrefixes")
                .and_then(Value::as_array)
                .is_some_and(|prefixes| {
                    prefixes
                        .iter()
                        .any(|prefix| prefix.as_str() == Some("https://crates.io/"))
                })
        })
        .expect("renovate.json must carry a crates.io cargo packageRule")
}

#[test]
fn renovate_json_configures_minimum_release_age() {
    let config = load_renovate_config();
    let violations = audit_quarantine_policy(&config);
    assert!(
        violations.is_empty(),
        "renovate.json violates the 24h quarantine policy (Issue #1234): {violations:#?}"
    );
}

#[test]
fn removing_the_cargo_rule_window_is_detected() {
    let mut config = load_renovate_config();
    let index = crates_io_rule_index(&config);
    config["packageRules"][index]
        .as_object_mut()
        .expect("packageRule object")
        .remove("minimumReleaseAge");

    // The old substring assertion would still pass here: other rules keep
    // the literal "24h" alive. That is exactly the gap Issue #1915 closes.
    assert!(
        config.to_string().contains("24h"),
        "fixture must keep `24h` elsewhere so this proves the structural \
         check catches what a substring match cannot"
    );

    let violations = audit_quarantine_policy(&config);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("does not set `minimumReleaseAge`")),
        "dropping the cargo rule's window must be reported: {violations:#?}"
    );
}

#[test]
fn lowering_a_window_below_24h_is_detected() {
    for lowered in ["1h", "60 minutes", "23h"] {
        let mut config = load_renovate_config();
        let index = crates_io_rule_index(&config);
        config["packageRules"][index]["minimumReleaseAge"] = json!(lowered);
        let violations = audit_quarantine_policy(&config);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("below the")),
            "`{lowered}` is under the 24h standard and must be reported: {violations:#?}"
        );
    }
}

#[test]
fn zeroing_a_covered_rule_is_detected() {
    let mut config = load_renovate_config();
    let index = crates_io_rule_index(&config);
    config["packageRules"][index]["minimumReleaseAge"] = json!("0");
    let violations = audit_quarantine_policy(&config);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("disables the quarantine")),
        "zeroing the cargo rule must be reported: {violations:#?}"
    );
}

#[test]
fn a_new_zero_age_rule_outside_the_internal_bypass_is_detected() {
    for smuggled in [
        json!({
            "description": "Smuggled bypass for all cargo deps.",
            "matchManagers": ["cargo"],
            "minimumReleaseAge": "0"
        }),
        json!({
            "description": "Smuggled bypass keyed on a third-party source.",
            "matchSourceUrlPrefixes": ["https://github.com/attacker/"],
            "minimumReleaseAge": "0"
        }),
        json!({
            "description": "Smuggled numeric zero.",
            "matchManagers": ["github-actions"],
            "minimumReleaseAge": 0
        }),
    ] {
        let mut config = load_renovate_config();
        config["packageRules"]
            .as_array_mut()
            .expect("packageRules array")
            .push(smuggled.clone());
        let violations = audit_quarantine_policy(&config);
        assert!(
            !violations.is_empty(),
            "a new zero-age rule ({smuggled}) must be reported"
        );
    }
}

#[test]
fn dropping_a_covered_dependency_class_is_detected() {
    for manager in COVERED_MANAGERS {
        let mut config = load_renovate_config();
        config["packageRules"]
            .as_array_mut()
            .expect("packageRules array")
            .retain(|rule| {
                !rule
                    .get("matchManagers")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .any(|entry| entry.as_str() == Some(manager))
            });
        let violations = audit_quarantine_policy(&config);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(manager)),
            "dropping every `{manager}` rule must be reported: {violations:#?}"
        );
    }
}

#[test]
fn dropping_the_internal_bypass_is_detected() {
    let mut config = load_renovate_config();
    config["packageRules"]
        .as_array_mut()
        .expect("packageRules array")
        .retain(|rule| !is_internal_bypass_rule(rule));
    let violations = audit_quarantine_policy(&config);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("internal-dependency")),
        "losing the stSoftwareAU bypass must be reported: {violations:#?}"
    );
}

#[test]
fn release_age_parser_accepts_the_documented_forms() {
    assert_eq!(parse_release_age_minutes("24h"), Ok(24 * 60));
    assert_eq!(parse_release_age_minutes("1 day"), Ok(24 * 60));
    assert_eq!(parse_release_age_minutes("3 days"), Ok(3 * 24 * 60));
    assert_eq!(parse_release_age_minutes("1 week"), Ok(7 * 24 * 60));
    assert_eq!(parse_release_age_minutes(" 90 minutes "), Ok(90));
    assert_eq!(parse_release_age_minutes("0"), Ok(0));
}

#[test]
fn release_age_parser_rejects_unparseable_windows() {
    for raw in ["", "soon", "24 fortnights", "h24", "7"] {
        assert!(
            parse_release_age_minutes(raw).is_err(),
            "`{raw}` must not parse as a valid quarantine window"
        );
    }
}

#[test]
fn scheduled_upgrade_workflow_has_been_removed() {
    // Business logic change (Issue #1282): the weekly Cargo dependency
    // upgrade workflow has been removed entirely. Dependency bumps now
    // happen on the per-PR path via `bump-deps.sh`, which continues to
    // enforce the quarantine gate. This test previously asserted that
    // the workflow invoked `bump-deps.sh`; it now asserts the workflow
    // file is absent so the cron path cannot be reintroduced silently.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflow = root.join(".github/workflows/upgrade-dependencies.yml");
    assert!(
        !workflow.exists(),
        ".github/workflows/upgrade-dependencies.yml must not exist \
         (Issue #1282) — the weekly scheduled upgrade workflow has \
         been removed in favour of per-PR bumps via bump-deps.sh"
    );
}

#[test]
fn bump_deps_script_enforces_quarantine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("bump-deps.sh");
    let contents = fs::read_to_string(&script).expect("read bump-deps.sh");
    // The helper must be invoked from the bump phase (not only defined
    // in the header). We require an actual crates.io API lookup helper
    // plus a use site that reverts an in-quarantine bump.
    assert!(
        contents.contains("crates.io/api/v1/crates"),
        "bump-deps.sh must query crates.io for publish time \
         (Issue #1234) — the existing quarantine plumbing currently \
         never reaches the API"
    );
    assert!(
        contents.contains("fetch_publish_epoch")
            || contents.contains("publish_epoch")
            || contents.contains("published_epoch"),
        "bump-deps.sh must define and call a publish-time fetch helper \
         (Issue #1234)"
    );
    assert!(
        contents.contains("revert") || contents.contains("rollback"),
        "bump-deps.sh must revert in-quarantine bumps rather than just \
         logging them (Issue #1234)"
    );
}
