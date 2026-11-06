//! Issue #2532: run the useful cases from the formerly un-gated shell suites
//! through Cargo's integration-test gate.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn write_executable(path: &Path, body: &str) {
    fs::write(path, body).expect("write executable fixture");
    let mut permissions = fs::metadata(path)
        .expect("stat executable fixture")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).expect("make fixture executable");
}

fn lockfile(packages: &[(&str, &str)]) -> String {
    let mut contents = String::from("version = 4\n\n");
    for (name, version) in packages {
        contents.push_str(&format!(
            "[[package]]\nname = \"{name}\"\nversion = \"{version}\"\n\n"
        ));
    }
    contents
}

fn lock_version(lock: &str, name: &str) -> Option<String> {
    let mut package_name = None;
    for line in lock.lines() {
        if let Some(value) = line
            .strip_prefix("name = \"")
            .and_then(|line| line.strip_suffix('\"'))
        {
            package_name = Some(value);
        } else if let Some(value) = line
            .strip_prefix("version = \"")
            .and_then(|line| line.strip_suffix('\"'))
            && package_name == Some(name)
        {
            return Some(value.to_owned());
        }
    }
    None
}

fn source_bump(snippet: &str, bin: Option<&Path>, fixture: Option<&Path>) -> Output {
    let mut command = Command::new("bash");
    command.arg("-c").arg(format!(
        "source '{}' && {snippet}",
        root().join("bump-deps.sh").display()
    ));
    command
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .stdin(Stdio::null());
    if let Some(bin) = bin {
        command.env("PATH", format!("{}:/usr/bin:/bin", bin.display()));
    }
    if let Some(fixture) = fixture {
        command.env("BUMP_DEPS_TEST_FIXTURE", fixture);
    }
    command.output().expect("run bump-deps.sh helper")
}

fn run_bump(args: &[&str], quarantine_hours: Option<&str>) -> Output {
    let mut command = Command::new(root().join("bump-deps.sh"));
    command.args(args).stdin(Stdio::null());
    if let Some(hours) = quarantine_hours {
        command.env("VIBE_BUMP_QUARANTINE_HOURS", hours);
    }
    command.output().expect("run bump-deps.sh CLI")
}

fn write_publish_fixtures(dir: &Path, packages: &[(&str, &str, bool)]) {
    for (name, version, young) in packages {
        let timestamp = if *young {
            "2025-05-31T19:00:00.000000+00:00"
        } else {
            "2025-05-19T12:00:00.000000+00:00"
        };
        fs::write(
            dir.join(format!("{name}-{version}.json")),
            format!(r#"{{"version":{{"num":"{version}","created_at":"{timestamp}"}}}}"#),
        )
        .expect("write crates.io publish-time fixture");
    }
}

fn cargo_stub(bin: &Path, rules: &Path, log: &Path) {
    let body = r###"#!/bin/bash
set -eu
printf '%s\n' "$*" >> 'LOG_PATH'
key="$*"
while IFS='|' read -r match result packages; do
  [[ -z "$match" ]] && continue
  if [[ "$key" == *"$match"* ]]; then
    if [[ "$result" == fail ]]; then echo 'error: failed to select a version for the requirement' >&2; exit 1; fi
    if [[ "$result" == noop ]]; then exit 0; fi
    lock="$(dirname "$(printf '%s\n' "$key" | sed -n 's/.*--manifest-path \([^ ]*\).*/\1/p')")/Cargo.lock"
    awk -v updates="$packages" '
      BEGIN { n = split(updates, rows, " "); for (i=1; i<=n; i++) { split(rows[i], pair, "="); replacement[pair[1]]=pair[2] } }
      function emit() { if (name != "") { print "[[package]]"; print "name = \"" name "\""; print "version = \"" ((name in replacement) ? replacement[name] : version) "\""; print "" } }
      /^\[\[package\]\]$/ { emit(); name=""; version=""; next }
      /^name = / { name=$3; gsub(/"/, "", name); next }
      /^version = / { version=$3; gsub(/"/, "", version); next }
      END { emit() }
    ' "$lock" > "$lock.next"
    mv "$lock.next" "$lock"
    for item in $packages; do name="$(echo "$item" | cut -d= -f1)"; version="$(echo "$item" | cut -d= -f2-)"; if ! grep -q "name = \"$name\"" "$lock"; then printf '[[package]]\nname = \"%s\"\nversion = \"%s\"\n\n' "$name" "$version" >> "$lock"; fi; done
    exit 0
  fi
done < 'RULE_PATH'
echo 'unexpected cargo update invocation' >&2
exit 2
"###
    .replace("LOG_PATH", &log.display().to_string())
    .replace("RULE_PATH", &rules.display().to_string());
    write_executable(&bin.join("cargo"), &body);
}

#[test]
fn reapply_holds_young_partners_and_transitives_while_keeping_safe_updates() {
    let tmp = tempfile::tempdir().unwrap();
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&fixtures).unwrap();
    write_publish_fixtures(
        &fixtures,
        &[
            ("js-sys", "0.3.106", true),
            ("wasm-bindgen", "0.2.106", true),
            ("wasm-bindgen-futures", "0.4.79", false),
            ("serde", "1.0.201", false),
            ("zerocopy", "0.8.2", false),
            ("zerocopy-derive", "0.8.2", false),
            ("tokio", "1.41.0", false),
            ("evil", "0.1.0", true),
            ("mio", "1.0.1", false),
        ],
    );
    let project = tmp.path().join("project");
    let bin = tmp.path().join("bin");
    fs::create_dir_all(&project).unwrap();
    fs::create_dir_all(&bin).unwrap();
    let manifest = project.join("Cargo.toml");
    fs::write(
        &manifest,
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let pristine = tmp.path().join("pristine.lock");
    fs::write(
        &pristine,
        lockfile(&[
            ("js-sys", "0.3.105"),
            ("wasm-bindgen", "0.2.105"),
            ("wasm-bindgen-futures", "0.4.78"),
            ("serde", "1.0.200"),
            ("zerocopy-derive", "0.8.1"),
            ("zerocopy", "0.8.1"),
            ("tokio", "1.40.0"),
            ("mio", "1.0.0"),
        ]),
    )
    .unwrap();
    fs::write(
        project.join("Cargo.lock"),
        lockfile(&[
            ("js-sys", "0.3.106"),
            ("wasm-bindgen", "0.2.106"),
            ("wasm-bindgen-futures", "0.4.79"),
            ("serde", "1.0.201"),
            ("zerocopy-derive", "0.8.2"),
            ("zerocopy", "0.8.2"),
            ("tokio", "1.41.0"),
            ("mio", "1.0.1"),
            ("evil", "0.1.0"),
        ]),
    )
    .unwrap();
    let before = tmp.path().join("before.tsv");
    fs::write(&before, "js-sys\t0.3.105\nwasm-bindgen\t0.2.105\nwasm-bindgen-futures\t0.4.78\nserde\t1.0.200\nzerocopy-derive\t0.8.1\nzerocopy\t0.8.1\ntokio\t1.40.0\nmio\t1.0.0\n").unwrap();
    let rules = tmp.path().join("rules");
    fs::write(&rules, "--workspace|noop|\nwasm-bindgen-futures|ok|wasm-bindgen-futures=0.4.79 js-sys=0.3.106 wasm-bindgen=0.2.106\nserde|ok|serde=1.0.201\nzerocopy-derive|ok|zerocopy-derive=0.8.2 zerocopy=0.8.2\nzerocopy|ok|zerocopy=0.8.2 zerocopy-derive=0.8.2\ntokio|ok|tokio=1.41.0 evil=0.1.0\nmio|fail|\n").unwrap();
    let log = tmp.path().join("cargo.log");
    cargo_stub(&bin, &rules, &log);
    let output = source_bump(
        &format!(
            "bump_deps::reapply_safe_lock_changes '{}' '{}' '{}' 1748736000 24",
            manifest.display(),
            pristine.display(),
            before.display()
        ),
        Some(&bin),
        Some(&fixtures),
    );
    assert!(
        output.status.success(),
        "lock reapplication failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let final_lock = fs::read_to_string(project.join("Cargo.lock")).unwrap();
    for (name, version) in [
        ("js-sys", "0.3.105"),
        ("wasm-bindgen", "0.2.105"),
        ("wasm-bindgen-futures", "0.4.78"),
        ("serde", "1.0.201"),
        ("zerocopy", "0.8.2"),
        ("zerocopy-derive", "0.8.2"),
        ("tokio", "1.40.0"),
        ("mio", "1.0.0"),
    ] {
        assert_eq!(
            lock_version(&final_lock, name).as_deref(),
            Some(version),
            "wrong final lock version for {name}"
        );
    }
    assert_eq!(lock_version(&final_lock, "evil"), None);
    for expected in [
        "pinned\tjs-sys\t0.3.105\t0.3.106\t5h",
        "skipped\tevil\t-\t0.1.0\t5h",
        "kept\tserde\t1.0.200\t1.0.201",
        "rejected\twasm-bindgen-futures\t0.4.78\t0.4.79\t",
        "rejected\ttokio\t1.40.0\t1.41.0\t",
        "rejected\tmio\t1.0.0\t1.0.1\tcargo update failed",
    ] {
        assert!(
            stdout.lines().any(|line| line.starts_with(expected)),
            "missing diagnostic {expected:?} in {stdout}"
        );
    }
    assert!(
        stdout.contains("js-sys@0.3.106"),
        "rejection must name the young lockstep partner"
    );
    assert!(
        stdout.contains("evil@0.1.0"),
        "rejection must name the young transitive package"
    );
}

#[test]
fn bump_cli_help_config_and_invalid_options_are_reported() {
    for args in [&["--help"][..], &["-h"][..]] {
        let output = run_bump(args, None);
        assert!(output.status.success());
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("Usage:"));
        assert!(text.contains("VIBE_BUMP_QUARANTINE_HOURS"));
    }
    let unknown = run_bump(&["--not-a-real-flag"], None);
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("Unknown"));
    let defaults = run_bump(&["--print-config"], None);
    assert!(defaults.status.success());
    let defaults_text = String::from_utf8_lossy(&defaults.stdout);
    assert!(defaults_text.contains("quarantine_hours = 24"));
    assert!(defaults_text.contains("Cargo.toml"));
    let override_value = run_bump(&["--print-config"], Some("48"));
    assert!(override_value.status.success());
    assert!(String::from_utf8_lossy(&override_value.stdout).contains("quarantine_hours = 48"));
    let invalid = run_bump(&["--print-config"], Some("abc"));
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("VIBE_BUMP_QUARANTINE_HOURS"));
    let internal = run_bump(&["--list-internal-deps"], None);
    assert!(internal.status.success());
    assert!(String::from_utf8_lossy(&internal.stdout).contains("0 internal deps"));
}

#[test]
fn bump_no_network_dry_run_is_read_only_and_repeatable() {
    let cargo_toml = root().join("Cargo.toml");
    let cargo_lock = root().join("Cargo.lock");
    let manifest_before = fs::read(&cargo_toml).unwrap();
    let lock_before = fs::read(&cargo_lock).unwrap();
    let first = run_bump(&["--dry-run", "--no-network"], None);
    let second = run_bump(&["--dry-run", "--no-network"], None);
    assert!(
        first.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(first.status.code(), second.status.code());
    assert!(String::from_utf8_lossy(&first.stdout).contains("--no-network"));
    assert_eq!(fs::read(cargo_toml).unwrap(), manifest_before);
    assert_eq!(fs::read(cargo_lock).unwrap(), lock_before);
}

#[test]
fn bump_source_helpers_parse_diff_restore_and_read_lock_versions() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = tmp.path().join("Cargo.toml");
    fs::write(&manifest, "[package]\nname=\"demo\"\n\n[dependencies]\nserde = \"1.0.225\"\narrow = { version = \"58.4.0\", default-features = false }\nparquet = \"58.3.0\"\n\n[dev-dependencies]\ntempfile = \"3.27\"\n").unwrap();
    let deps = source_bump(
        &format!("bump_deps::extract_dep_versions '{}'", manifest.display()),
        None,
        None,
    );
    assert!(deps.status.success());
    let deps = String::from_utf8_lossy(&deps.stdout);
    for expected in [
        "serde\t1.0.225",
        "arrow\t58.4.0",
        "parquet\t58.3.0",
        "tempfile\t3.27",
    ] {
        assert!(
            deps.lines().any(|line| line == expected),
            "missing {expected:?} in {deps}"
        );
    }
    let before = tmp.path().join("before.tsv");
    let after = tmp.path().join("after.tsv");
    fs::write(&before, "serde\t1.0.190\nparquet\t58.3.0\narrow\t58.3.0\n").unwrap();
    fs::write(&after, "serde\t1.0.225\nparquet\t58.3.0\narrow\t58.4.0\n").unwrap();
    let changed = source_bump(
        &format!(
            "bump_deps::compute_changed_deps '{}' '{}'",
            before.display(),
            after.display()
        ),
        None,
        None,
    );
    assert!(changed.status.success());
    let changed = String::from_utf8_lossy(&changed.stdout);
    assert!(
        changed
            .lines()
            .any(|line| line == "serde\t1.0.190\t1.0.225")
    );
    assert!(changed.lines().any(|line| line == "arrow\t58.3.0\t58.4.0"));
    assert!(!changed.lines().any(|line| line.starts_with("parquet\t")));
    let restored = source_bump(
        &format!(
            "bump_deps::revert_dep_line '{}' serde 1.0.190 && bump_deps::revert_dep_line '{}' arrow 58.3.0",
            manifest.display(),
            manifest.display()
        ),
        None,
        None,
    );
    assert!(restored.status.success());
    let manifest_text = fs::read_to_string(manifest).unwrap();
    assert!(manifest_text.contains("serde = \"1.0.190\""));
    assert!(manifest_text.contains("version = \"58.3.0\""));
    assert!(manifest_text.contains("parquet = \"58.3.0\""));
}

#[test]
fn bump_publish_fixture_and_lock_helpers_fail_loud_and_cover_transitives() {
    let tmp = tempfile::tempdir().unwrap();
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&fixtures).unwrap();
    fs::write(
        fixtures.join("serde-1.0.225.json"),
        r#"{"version":{"num":"1.0.225","created_at":"2025-05-19T12:00:00.000000+00:00"}}"#,
    )
    .unwrap();
    let publish = source_bump(
        "bump_deps::fetch_publish_epoch serde 1.0.225",
        None,
        Some(&fixtures),
    );
    assert!(publish.status.success());
    assert_eq!(
        String::from_utf8_lossy(&publish.stdout).trim(),
        "1747656000"
    );
    let missing = source_bump(
        "bump_deps::fetch_publish_epoch absent 9.9.9",
        None,
        Some(&fixtures),
    );
    assert!(!missing.status.success());
    let now = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "source '{}' && bump_deps::current_epoch",
            root().join("bump-deps.sh").display()
        ))
        .env("BUMP_DEPS_SOURCE_ONLY", "1")
        .env("BUMP_DEPS_NOW_EPOCH", "1700000000")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(now.status.success());
    assert_eq!(String::from_utf8_lossy(&now.stdout).trim(), "1700000000");
    let lock = tmp.path().join("Cargo.lock");
    fs::write(
        &lock,
        lockfile(&[
            ("serde", "1.0.190"),
            ("quote", "1.0.35"),
            ("proc-macro2", "1.0.90"),
        ]),
    )
    .unwrap();
    let versions = source_bump(
        &format!("bump_deps::extract_lock_versions '{}'", lock.display()),
        None,
        None,
    );
    assert!(versions.status.success());
    let versions = String::from_utf8_lossy(&versions.stdout);
    for expected in ["serde\t1.0.190", "quote\t1.0.35", "proc-macro2\t1.0.90"] {
        assert!(versions.lines().any(|line| line == expected));
    }
    let before = tmp.path().join("before.tsv");
    let after = tmp.path().join("after.tsv");
    fs::write(&before, "serde\t1.0.190\nquote\t1.0.35\n").unwrap();
    fs::write(&after, "serde\t1.0.225\nquote\t1.0.35\nsneaky\t0.1.0\n").unwrap();
    let added = source_bump(
        &format!(
            "bump_deps::compute_new_deps '{}' '{}'",
            before.display(),
            after.display()
        ),
        None,
        None,
    );
    assert!(added.status.success());
    assert_eq!(
        String::from_utf8_lossy(&added.stdout).trim(),
        "sneaky\t0.1.0"
    );
}

#[test]
fn bump_planner_quarantines_young_updates_new_packages_and_unknown_publish_times() {
    let tmp = tempfile::tempdir().unwrap();
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&fixtures).unwrap();
    write_publish_fixtures(
        &fixtures,
        &[
            ("quote", "1.0.40", true),
            ("sneaky", "0.1.0", true),
            ("serde", "1.0.225", false),
        ],
    );
    let before = tmp.path().join("before.tsv");
    let after = tmp.path().join("after.tsv");
    fs::write(&before, "serde\t1.0.190\nquote\t1.0.35\n").unwrap();
    fs::write(&after, "serde\t1.0.225\nquote\t1.0.40\nsneaky\t0.1.0\n").unwrap();
    let planned = source_bump(
        &format!(
            "bump_deps::plan_lock_quarantine '{}' '{}' 1748736000 24",
            before.display(),
            after.display()
        ),
        None,
        Some(&fixtures),
    );
    assert!(planned.status.success());
    let plan = String::from_utf8_lossy(&planned.stdout);
    assert!(
        plan.lines()
            .any(|line| line == "revert\tquote\t1.0.35\t1.0.40\t5h"),
        "young version should be reverted: {plan}"
    );
    assert!(
        plan.lines()
            .any(|line| line == "block\tsneaky\t-\t0.1.0\t5h"),
        "young transitive should be blocked: {plan}"
    );
    assert!(
        !plan.lines().any(|line| line.contains("\tserde\t")),
        "aged update should remain safe: {plan}"
    );
    fs::write(&before, "ghost\t1.0.0\n").unwrap();
    fs::write(&after, "ghost\t9.9.9\n").unwrap();
    let unknown = source_bump(
        &format!(
            "bump_deps::plan_lock_quarantine '{}' '{}' 1748736000 24",
            before.display(),
            after.display()
        ),
        None,
        Some(&fixtures),
    );
    assert!(
        unknown.status.success(),
        "unknown publish time is a fail-closed plan, not a parser failure"
    );
    assert!(
        String::from_utf8_lossy(&unknown.stdout).contains("revert\tghost\t1.0.0\t9.9.9\tunknown")
    );
}

#[test]
fn fully_aged_refresh_is_unchanged_and_does_not_invoke_cargo() {
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("project");
    let bin = tmp.path().join("bin");
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&project).unwrap();
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(&fixtures).unwrap();
    write_publish_fixtures(&fixtures, &[("serde", "1.0.201", false)]);
    let manifest = project.join("Cargo.toml");
    fs::write(&manifest, "").unwrap();
    let pristine = tmp.path().join("pristine.lock");
    fs::write(&pristine, lockfile(&[("serde", "1.0.200")])).unwrap();
    let current = lockfile(&[("serde", "1.0.201")]);
    let lock = project.join("Cargo.lock");
    fs::write(&lock, &current).unwrap();
    let before = tmp.path().join("before.tsv");
    fs::write(&before, "serde\t1.0.200\n").unwrap();
    let log = tmp.path().join("cargo.log");
    let rules = tmp.path().join("rules");
    fs::write(&rules, "").unwrap();
    cargo_stub(&bin, &rules, &log);
    let output = source_bump(
        &format!(
            "bump_deps::reapply_safe_lock_changes '{}' '{}' '{}' 1748736000 24",
            manifest.display(),
            pristine.display(),
            before.display()
        ),
        Some(&bin),
        Some(&fixtures),
    );
    assert!(output.status.success());
    assert!(
        output.stdout.is_empty(),
        "no quarantined changes should produce no report"
    );
    assert_eq!(fs::read_to_string(lock).unwrap(), current);
    assert!(!log.exists() || fs::read_to_string(log).unwrap().is_empty());
}

#[test]
fn manifest_that_requires_a_young_package_fails_with_the_package_cause() {
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("project");
    let bin = tmp.path().join("bin");
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&project).unwrap();
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(&fixtures).unwrap();
    write_publish_fixtures(&fixtures, &[("js-sys", "0.3.106", true)]);
    let manifest = project.join("Cargo.toml");
    fs::write(
        &manifest,
        "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n",
    )
    .unwrap();
    let pristine = tmp.path().join("pristine.lock");
    fs::write(&pristine, lockfile(&[("js-sys", "0.3.105")])).unwrap();
    fs::write(
        project.join("Cargo.lock"),
        lockfile(&[("js-sys", "0.3.106")]),
    )
    .unwrap();
    let before = tmp.path().join("before.tsv");
    fs::write(&before, "js-sys\t0.3.105\n").unwrap();
    let rules = tmp.path().join("rules");
    fs::write(&rules, "--workspace|ok|js-sys=0.3.106\n").unwrap();
    let log = tmp.path().join("cargo.log");
    cargo_stub(&bin, &rules, &log);
    let output = source_bump(
        &format!(
            "bump_deps::reapply_safe_lock_changes '{}' '{}' '{}' 1748736000 24",
            manifest.display(),
            pristine.display(),
            before.display()
        ),
        Some(&bin),
        Some(&fixtures),
    );
    assert!(
        !output.status.success(),
        "unsafe manifest resolution must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("js-sys@0.3.106"),
        "failure must state the young package cause: {stderr}"
    );
}

#[test]
fn safe_lockstep_family_is_reapplied_in_one_multi_package_update() {
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("project");
    let bin = tmp.path().join("bin");
    let fixtures = tmp.path().join("fixtures");
    fs::create_dir_all(&project).unwrap();
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(&fixtures).unwrap();
    write_publish_fixtures(
        &fixtures,
        &[
            ("js-sys", "0.3.107", false),
            ("wasm-bindgen", "0.2.129", false),
            ("web-sys", "0.3.107", false),
            ("pair-a", "1.0.1", false),
            ("pair-b", "1.0.1", true),
        ],
    );
    let manifest = project.join("Cargo.toml");
    fs::write(&manifest, "").unwrap();
    let pristine = tmp.path().join("pristine.lock");
    fs::write(
        &pristine,
        lockfile(&[
            ("js-sys", "0.3.105"),
            ("wasm-bindgen", "0.2.105"),
            ("web-sys", "0.3.105"),
            ("pair-a", "1.0.0"),
            ("pair-b", "1.0.0"),
        ]),
    )
    .unwrap();
    fs::write(
        project.join("Cargo.lock"),
        lockfile(&[
            ("js-sys", "0.3.107"),
            ("wasm-bindgen", "0.2.129"),
            ("web-sys", "0.3.107"),
            ("pair-a", "1.0.1"),
            ("pair-b", "1.0.1"),
        ]),
    )
    .unwrap();
    let before = tmp.path().join("before.tsv");
    fs::write(
        &before,
        "js-sys\t0.3.105\nwasm-bindgen\t0.2.105\nweb-sys\t0.3.105\npair-a\t1.0.0\npair-b\t1.0.0\n",
    )
    .unwrap();
    let rules = tmp.path().join("rules");
    fs::write(&rules, "--workspace|noop|\n-p pair-a@1.0.0|ok|js-sys=0.3.107 wasm-bindgen=0.2.129 web-sys=0.3.107\njs-sys|noop|\nwasm-bindgen|noop|\nweb-sys|noop|\npair-a|noop|\n").unwrap();
    let log = tmp.path().join("cargo.log");
    cargo_stub(&bin, &rules, &log);
    let output = source_bump(
        &format!(
            "bump_deps::reapply_safe_lock_changes '{}' '{}' '{}' 1748736000 24",
            manifest.display(),
            pristine.display(),
            before.display()
        ),
        Some(&bin),
        Some(&fixtures),
    );
    assert!(
        output.status.success(),
        "lockstep reapply failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let final_lock = fs::read_to_string(project.join("Cargo.lock")).unwrap();
    assert_eq!(
        lock_version(&final_lock, "js-sys").as_deref(),
        Some("0.3.107")
    );
    assert_eq!(
        lock_version(&final_lock, "wasm-bindgen").as_deref(),
        Some("0.2.129")
    );
    assert_eq!(
        lock_version(&final_lock, "web-sys").as_deref(),
        Some("0.3.107")
    );
    assert_eq!(
        lock_version(&final_lock, "pair-a").as_deref(),
        Some("1.0.0")
    );
    assert_eq!(
        lock_version(&final_lock, "pair-b").as_deref(),
        Some("1.0.0")
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("kept\tpair-a"));
    assert!(String::from_utf8_lossy(&output.stdout).contains("rejected\tpair-a"));
    let calls = fs::read_to_string(log).unwrap();
    assert!(
        calls.lines().any(|line| line.contains("-p js-sys@0.3.105")
            && line.contains("-p wasm-bindgen@0.2.105")
            && line.contains("-p web-sys@0.3.105")),
        "the stuck lockstep family must be updated together: {calls}"
    );
}

struct BenchmarkSandbox {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    bin: PathBuf,
    log: PathBuf,
}

impl BenchmarkSandbox {
    fn new(benchmarks: &[&str], cargo_body: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("project");
        let scripts = project.join("scripts");
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&scripts).unwrap();
        fs::create_dir_all(&bin).unwrap();
        let mut manifest = String::from("[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n");
        for bench in benchmarks {
            manifest.push_str(&format!(
                "\n[[bench]]\nname = \"{bench}\"\nharness = false\n"
            ));
        }
        fs::write(project.join("Cargo.toml"), manifest).unwrap();
        fs::copy(
            root().join("scripts/benchmark-ci.sh"),
            scripts.join("benchmark-ci.sh"),
        )
        .unwrap();
        fs::copy(
            root().join("scripts/benchmark_threshold.sh"),
            scripts.join("benchmark_threshold.sh"),
        )
        .unwrap();
        let log = tmp.path().join("cargo.log");
        write_executable(
            &bin.join("cargo"),
            &cargo_body.replace("LOG_PATH", &log.display().to_string()),
        );
        Self {
            _tmp: tmp,
            root: project,
            bin,
            log,
        }
    }

    fn run(&self, args: &[&str], mode: Option<&str>) -> Output {
        let mut command = Command::new(self.root.join("scripts/benchmark-ci.sh"));
        command
            .args(args)
            .env("PATH", format!("{}:/usr/bin:/bin", self.bin.display()))
            .stdin(Stdio::null());
        if let Some(mode) = mode {
            command.env("BENCHMARK_CI_MODE", mode);
        }
        command.output().expect("run benchmark-ci.sh")
    }
}

#[test]
fn benchmark_cli_validation_and_discovery_work_in_an_isolated_manifest() {
    let sandbox = BenchmarkSandbox::new(&["alpha", "beta"], "#!/bin/bash\nexit 0\n");
    for args in [&["--help"][..], &["-h"][..]] {
        let out = sandbox.run(args, None);
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout);
        for item in ["Usage:", "compile-only", "compare", "threshold", "--list"] {
            assert!(text.contains(item), "help omits {item}");
        }
    }
    let unknown = sandbox.run(&["--invalid-flag"], None);
    assert_eq!(unknown.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&unknown.stdout).contains("Unknown option"));
    let missing = sandbox.run(&["--threshold"], None);
    assert_eq!(missing.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&missing.stdout).contains("requires a value"));
    let invalid = sandbox.run(&["--threshold", "abc"], None);
    assert_eq!(invalid.status.code(), Some(1));
    let invalid_text = format!(
        "{}{}",
        String::from_utf8_lossy(&invalid.stdout),
        String::from_utf8_lossy(&invalid.stderr)
    );
    assert!(invalid_text.contains("positive integer or decimal"));
    let list = sandbox.run(&["--list"], None);
    assert!(list.status.success());
    let text = String::from_utf8_lossy(&list.stdout);
    assert!(text.contains("Discovered benchmark suites (2)"));
    assert!(text.contains("  - alpha"));
    assert!(text.contains("  - beta"));
    let empty = BenchmarkSandbox::new(&[], "#!/bin/bash\nexit 0\n");
    let no_benchmarks = empty.run(&["--list"], None);
    assert!(!no_benchmarks.status.success());
    assert!(String::from_utf8_lossy(&no_benchmarks.stdout).contains("No benchmark targets"));
}

#[test]
fn benchmark_ci_discovers_real_manifest_targets() {
    let output = Command::new(root().join("scripts/benchmark-ci.sh"))
        .arg("--list")
        .stdin(Stdio::null())
        .output()
        .expect("run benchmark-ci.sh against the repository manifest");
    assert!(
        output.status.success(),
        "benchmark discovery failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listed = String::from_utf8_lossy(&output.stdout);
    assert!(listed.contains("synapse_counts"));
    assert!(listed.contains("neuron_interning"));
    assert!(
        listed
            .lines()
            .filter(|line| line.starts_with("  - "))
            .count()
            >= 10
    );
}

#[test]
fn benchmark_mode_selection_and_compile_results_use_cargo_stub() {
    let sandbox = BenchmarkSandbox::new(
        &["alpha", "beta"],
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> 'LOG_PATH'\nif [[ \"$*\" == *alpha* ]]; then exit 0; else echo 'compile failed for beta' >&2; exit 1; fi\n",
    );
    let invalid_mode = sandbox.run(&[], Some("invalid_mode"));
    assert_eq!(invalid_mode.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&invalid_mode.stdout).contains("Unknown mode"));
    let compared = sandbox.run(&["--list"], Some("compare"));
    assert!(
        compared.status.success(),
        "--list validates and exits before mode execution"
    );
    let compile = sandbox.run(&["--compile-only"], Some("compare"));
    assert_eq!(compile.status.code(), Some(1));
    let text = String::from_utf8_lossy(&compile.stdout);
    assert!(text.contains("Mode: compile"));
    assert!(text.contains("Compiled: 1 / 2"));
    assert!(text.contains("Failed:   1"));
    assert!(text.contains("beta"));
    let calls = fs::read_to_string(&sandbox.log).unwrap();
    assert!(
        calls
            .lines()
            .any(|line| line == "bench --bench alpha --no-run")
    );
    assert!(
        calls
            .lines()
            .any(|line| line == "bench --bench beta --no-run")
    );
    let success = BenchmarkSandbox::new(&["alpha", "beta"], "#!/bin/bash\nexit 0\n");
    let compile = success.run(&[], Some("compile"));
    assert!(compile.status.success());
    assert!(
        String::from_utf8_lossy(&compile.stdout).contains("All benchmarks compile successfully")
    );
}

#[test]
fn benchmark_compare_mode_delegates_with_validated_threshold() {
    let sandbox = BenchmarkSandbox::new(&["alpha"], "#!/bin/bash\nexit 0\n");
    fs::create_dir_all(sandbox.root.join("target/criterion")).unwrap();
    let compare = sandbox.root.join("benchmark_compare.sh");
    write_executable(
        &compare,
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$COMPARE_LOG\"\n",
    );
    let out = Command::new(sandbox.root.join("scripts/benchmark-ci.sh"))
        .args(["--compare", "--threshold", "12"])
        .env("PATH", format!("{}:/usr/bin:/bin", sandbox.bin.display()))
        .env("COMPARE_LOG", sandbox._tmp.path().join("compare.log"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        fs::read_to_string(sandbox._tmp.path().join("compare.log"))
            .unwrap()
            .trim(),
        "--threshold 12"
    );
}
