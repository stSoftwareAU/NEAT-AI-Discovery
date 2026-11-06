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
        {
            if package_name == Some(name) {
                return Some(value.to_owned());
            }
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
    write_executable(
        &bin.join("cargo"),
        &format!(
            "#!/bin/bash\nset -eu\nprintf '%s\\n' \"$*\" >> 'LOG_PATH'\nkey=\"$*\"\nwhile IFS='|' read -r match result packages; do\n  [[ -z \"$match\" ]] && continue\n  if [[ \"$key\" == *\"$match\"* ]]; then\n    if [[ \"$result\" == fail ]]; then echo 'error: failed to select a version for the requirement' >&2; exit 1; fi\n    if [[ \"$result\" == noop ]]; then exit 0; fi\n    lock=\"$(dirname \"$(printf '%s\\n' \"$key\" | sed -n 's/.*--manifest-path \\([^ ]*\\).*/\\1/p')\")/Cargo.lock\"\n    awk -v updates=\"$packages\" 'BEGIN {{ n = split(updates, rows, " "); for (i=1; i<=n; i++) {{ split(rows[i], pair, "="); replacement[pair[1]]=pair[2] }} }} function emit() {{ if (name != "") {{ print "[[package]]"; print "name = \\\"" name "\\\""; print "version = \\\"" ((name in replacement) ? replacement[name] : version) "\\\""; print "" }} }} /^\\[\\[package\\]\\]$/ {{ emit(); name=""; version=""; next }} /^name = / {{ name=$3; gsub(/\\\"/, "", name); next }} /^version = / {{ version=$3; gsub(/\\\"/, "", version); next }} END {{ emit() }}' \"$lock\" > \"$lock.next\"\n    mv \"$lock.next\" \"$lock\"\n    for item in $packages; do name=\"${{item%%=*}}\"; version=\"${{item#*=}}\"; if ! grep -q \"name = \\\"$name\\\"\" \"$lock\"; then printf '[[package]]\\nname = \\\"%s\\\"\\nversion = \\\"%s\\\"\\n\\n' \"$name\" \"$version\" >> \"$lock\"; fi; done\n    exit 0\n  fi\ndone < 'RULE_PATH'\necho 'unexpected cargo update invocation' >&2\nexit 2\n"
                .replace("LOG_PATH", &log.display().to_string())
                .replace("RULE_PATH", &rules.display().to_string()),
        ),
    );
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
    fs::write(&rules, "--workspace|noop|\n-p wasm-bindgen@0.2.105|ok|js-sys=0.3.107 wasm-bindgen=0.2.129 web-sys=0.3.107\njs-sys|noop|\nwasm-bindgen|noop|\nweb-sys|noop|\npair-a|noop|\n").unwrap();
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
