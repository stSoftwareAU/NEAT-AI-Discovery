//! Tests for the production discovery candidates cache study (Issue #1920).

use neat_ai_discovery::analysis::cache_study::corpus::{
    Outcome, RecordSource, load_live, parse_cache_path,
};
use neat_ai_discovery::analysis::cache_study::git_history::{load_from_history, parse_deleted_log};
use neat_ai_discovery::analysis::cache_study::stats::{VANISHING_GAIN, pearson, study};
use neat_ai_discovery::analysis::cache_study::{load_corpus, render_markdown, study_checkout};
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Writes one candidate record under `<outcome>/<hash>/<strategy>/<key>.json`.
fn write_record(
    root: &Path,
    outcome: &str,
    hash: &str,
    strategy: &str,
    key: &str,
    score_delta: f64,
    impact: f64,
) {
    let dir = root.join(outcome).join(hash).join(strategy);
    std::fs::create_dir_all(&dir).expect("create cache dir");
    let body = serde_json::json!({
        "wireSchemaVersion": 2,
        "key": key,
        "changeType": strategy,
        "description": "test record",
        "originalScore": 0.32,
        "candidateScore": 0.32 + score_delta,
        "scoreDelta": score_delta,
        "originalError": 0.68,
        "error": 0.68 - score_delta,
        "timestamp": "2026-07-30T14:47:46.194333008Z",
        "discoveryVersion": "0.74.188",
        "rustRequest": {
            "removalCandidate": {
                "neuronUuid": key,
                "impact": impact,
                "meanActivation": 0.5,
            }
        },
    });
    std::fs::write(
        dir.join(format!("{key}.json")),
        serde_json::to_string_pretty(&body).expect("serialise record"),
    )
    .expect("write record");
}

/// A cache with three successes (increasing gain with increasing impact) and
/// two failures, spread over two strategies.
fn sample_cache() -> TempDir {
    let temp = TempDir::new().expect("temp dir");
    let root = temp.path();
    write_record(
        root,
        "success",
        "c188",
        "remove-low-impact",
        "s1",
        1e-8,
        1e-14,
    );
    write_record(
        root,
        "success",
        "c188",
        "remove-low-impact",
        "s2",
        1e-7,
        1e-13,
    );
    write_record(
        root,
        "success",
        "c188",
        "remove-low-impact",
        "s3",
        1e-6,
        1e-12,
    );
    write_record(
        root,
        "failures",
        "c188",
        "remove-low-impact",
        "f1",
        -1e-5,
        1e-9,
    );
    write_record(root, "failures", "c188", "add-neurons", "f2", -2e-5, 1e-8);
    temp
}

#[test]
fn parses_valid_cache_paths() {
    let parsed = parse_cache_path("success/c1885aa6/remove-low-impact/v2_key.json")
        .expect("valid success path");
    assert_eq!(parsed.outcome, Outcome::Success);
    assert_eq!(parsed.model_hash, "c1885aa6");
    assert_eq!(parsed.strategy, "remove-low-impact");

    let failure =
        parse_cache_path("failures/abc/add-neurons/v2_key.json").expect("valid failure path");
    assert_eq!(failure.outcome, Outcome::Failure);
}

#[test]
fn rejects_paths_that_are_not_cache_records() {
    assert!(parse_cache_path("README.md").is_none());
    assert!(parse_cache_path("success/c188/remove-low-impact/notes.txt").is_none());
    assert!(parse_cache_path("archive/c188/remove-low-impact/v2_key.json").is_none());
    assert!(parse_cache_path("success/c188/v2_key.json").is_none());
}

#[test]
fn loads_every_live_record_with_its_classification() {
    let temp = sample_cache();
    let entries = load_live(temp.path()).expect("load live corpus");

    assert_eq!(entries.len(), 5);
    assert_eq!(
        entries
            .iter()
            .filter(|e| e.outcome == Outcome::Success)
            .count(),
        3
    );
    assert!(entries.iter().all(|e| e.source == RecordSource::Live));
    assert!(entries.iter().all(|e| e.model_hash == "c188"));
}

#[test]
fn malformed_record_fails_loudly_naming_the_file() {
    let temp = sample_cache();
    let bad = temp
        .path()
        .join("success/c188/remove-low-impact/broken.json");
    std::fs::write(&bad, "{ not json").expect("write malformed record");

    let error = load_live(temp.path()).expect_err("malformed record must fail loudly");
    let message = format!("{error:#}");
    assert!(
        message.contains("broken.json"),
        "error must name the offending file, got: {message}"
    );
}

#[test]
fn rejects_a_directory_that_is_not_a_cache_checkout() {
    let temp = TempDir::new().expect("temp dir");
    let error = load_live(temp.path()).expect_err("non-cache directory must fail loudly");
    assert!(format!("{error:#}").contains("discovery cache checkout"));
}

#[test]
fn study_summarises_volume_and_gain() {
    let temp = sample_cache();
    let entries = load_live(temp.path()).expect("load live corpus");
    let result = study(&entries);

    assert_eq!(result.total, 5);
    assert_eq!(result.live, 5);
    assert_eq!(result.recovered, 0);
    assert_eq!(result.overall.successes, 3);
    assert_eq!(result.overall.failures, 2);
    assert!((result.overall.success_rate().expect("rate") - 0.6).abs() < 1e-12);
    assert!((result.overall.median_success_delta - 1e-7).abs() < 1e-18);
    assert!((result.overall.max_success_delta - 1e-6).abs() < 1e-18);
    assert_eq!(
        result.vanishing_successes, 3,
        "all fixture gains are <= VANISHING_GAIN ({VANISHING_GAIN:e})"
    );

    // Strategy breakdown is ordered largest-first.
    assert_eq!(result.by_strategy[0].label, "remove-low-impact");
    assert_eq!(result.by_strategy[0].total(), 4);
    assert_eq!(result.by_strategy[1].label, "add-neurons");
    assert_eq!(result.by_strategy[1].successes, 0);

    assert_eq!(result.by_day.len(), 1);
    assert_eq!(result.by_day[0].label, "2026-07-30");
    assert_eq!(result.by_version[0].label, "0.74.188");
    assert_eq!(result.by_request_kind[0].label, "removalCandidate");
}

#[test]
fn predictor_detects_a_monotone_relationship() {
    let temp = sample_cache();
    let entries = load_live(temp.path()).expect("load live corpus");
    let result = study(&entries);

    let impact = result
        .predictors
        .iter()
        .find(|p| p.label == "removalCandidate.impact")
        .expect("impact predictor present");
    assert_eq!(impact.gain_samples, 3);
    let r = impact.r_vs_log_gain.expect("correlation computable");
    assert!(
        r > 0.99,
        "impact rises with gain in the fixture, expected r ~ 1.0, got {r}"
    );
}

#[test]
fn pearson_handles_degenerate_inputs() {
    assert!(
        (pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).expect("perfect fit") - 1.0).abs() < 1e-12
    );
    assert!(
        (pearson(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]).expect("inverse fit") + 1.0).abs() < 1e-12
    );
    assert!(pearson(&[1.0, 2.0], &[1.0, 2.0]).is_none(), "too few pairs");
    assert!(
        pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none(),
        "zero variance"
    );
    assert!(pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none(), "ragged");
}

#[test]
fn parses_deleted_log_output() {
    let log = "commit aaa111\nsuccess/h1/remove-low-impact/a.json\nfailures/h1/add-neurons/b.json\n\ncommit bbb222\nsuccess/h2/change-squash/c.json\n";
    let deleted = parse_deleted_log(log);

    assert_eq!(deleted.len(), 3);
    assert_eq!(deleted[0].commit, "aaa111");
    assert_eq!(deleted[0].path, "success/h1/remove-low-impact/a.json");
    assert_eq!(deleted[2].commit, "bbb222");
}

fn git(root: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .status()
        .expect("run git");
    assert!(status.success(), "git {args:?} failed");
}

#[test]
fn recovers_wiped_records_from_git_history() {
    let temp = sample_cache();
    let root = temp.path();
    git(root, &["init", "--quiet"]);
    git(root, &["config", "user.email", "test@example.com"]);
    git(root, &["config", "user.name", "Cache Study Test"]);
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "seed cache"]);

    // Simulate a "Clean up OLD discovery caches" commit wiping an old hash.
    write_record(root, "success", "old99", "change-squash", "o1", 5e-7, 1e-11);
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "old hash records"]);
    std::fs::remove_dir_all(root.join("success/old99")).expect("wipe old hash");
    git(root, &["add", "-A"]);
    git(
        root,
        &["commit", "--quiet", "-m", "Clean up OLD discovery caches."],
    );

    let recovered = load_from_history(root).expect("recover deleted records");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].model_hash, "old99");
    assert_eq!(recovered[0].strategy, "change-squash");
    assert_eq!(recovered[0].source, RecordSource::GitHistory);

    // The merged corpus adds the wiped record without duplicating live ones.
    let live_only = load_corpus(root, false).expect("live corpus");
    let widened = load_corpus(root, true).expect("widened corpus");
    assert_eq!(live_only.len(), 5);
    assert_eq!(widened.len(), 6);

    let result = study(&widened);
    assert_eq!(result.live, 5);
    assert_eq!(result.recovered, 1);
    assert_eq!(result.by_model_hash.len(), 2);
}

#[test]
fn report_renders_the_headline_tables() {
    let temp = sample_cache();
    let markdown = study_checkout(temp.path(), false).expect("render report");

    for heading in [
        "# Discovery Candidates Cache Study",
        "## Overview",
        "## Volume by strategy",
        "## Volume by model hash",
        "## Volume by day",
        "## Volume by discovery version",
        "## Volume by request kind",
        "## Gain-size predictors",
    ] {
        assert!(markdown.contains(heading), "report must contain {heading}");
    }
    assert!(markdown.contains("| remove-low-impact | 4 | 3 | 1 | 75.0% |"));
    assert!(markdown.contains("`removalCandidate.impact`"));
}

#[test]
fn report_handles_an_empty_corpus_without_panicking() {
    let markdown = render_markdown(&study(&[]));
    assert!(markdown.contains("| Records | 0 |"));
    assert!(markdown.contains("_No records._"));
}
