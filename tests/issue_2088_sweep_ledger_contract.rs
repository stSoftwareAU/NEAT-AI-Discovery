//! Contract tests for the security-sweep coverage ledger (Issue #2088).
//!
//! Before this ledger existed there was no way to tell a swept chunk from an
//! unswept one: every overflow tracker restarted from zero, so the same surface
//! could be swept twice while another was never swept at all. These tests are
//! the gate that keeps the ledger trustworthy:
//!
//! * the machine-readable index parses and covers every overflow chunk,
//! * a claimed sweep is falsifiable — it pins a baseline commit SHA,
//! * prose records and index entries match **both ways**, so a record with no
//!   index entry (invisible to the next automated run) fails loudly.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Machine-readable sweep index — the only file every chunk sweep touches.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";
/// Human-readable ledger rules.
const README: &str = "docs/audits/README.md";
/// Per-chunk record skeleton.
const TEMPLATE: &str = "docs/audits/security-sweep-TEMPLATE.md";
/// Directory holding the whole ledger.
const LEDGER_DIR: &str = "docs/audits";
/// Filename prefix of a per-chunk prose record.
const RECORD_PREFIX: &str = "security-sweep-chunk-";

/// Chunk ids the overflow tracker (Issue #2083) left unreached.
const REQUIRED_CHUNKS: [&str; 9] = ["2", "4", "7", "8a", "8b", "9", "11", "13", "16"];

/// Index keys, in the order every entry must spell them. A stable order keeps
/// concurrent chunk sweeps conflicting on one line rather than the whole file.
const KEY_ORDER: [&str; 7] = [
    "id",
    "name",
    "exposure",
    "issue",
    "last_swept",
    "baseline_commit",
    "record",
];

/// Exposure taxonomy carried over from Issue #2083.
const EXPOSURES: [&str; 3] = ["internal", "local", "network"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

fn index() -> Value {
    serde_json::from_str(&read(INDEX)).unwrap_or_else(|e| panic!("{INDEX} must be valid JSON: {e}"))
}

fn chunk_entries() -> Vec<Value> {
    match index().get("chunks") {
        Some(Value::Array(entries)) => entries.clone(),
        other => panic!("{INDEX} must carry a `chunks` array, found {other:?}"),
    }
}

fn string_field(entry: &Value, key: &str) -> String {
    entry
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("entry {entry} must carry a string `{key}`"))
        .to_string()
}

/// Chunk ids are strings (`8a`, `8b`), and record filenames pad them to two
/// digits (`08a`). Normalise both sides before comparing.
fn normalise_id(raw: &str) -> String {
    raw.trim_start_matches('0').to_ascii_lowercase()
}

/// Every `docs/audits/security-sweep-chunk-*.md` record, repo-relative, sorted.
fn record_files() -> Vec<String> {
    let dir = repo_root().join(LEDGER_DIR);
    let mut records: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()))
        .map(|entry| entry.expect("directory entry must be readable").path())
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .filter_map(|p| file_name(&p))
        .filter(|name| name.starts_with(RECORD_PREFIX))
        .map(|name| format!("{LEDGER_DIR}/{name}"))
        .collect();
    records.sort();
    records
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name().map(|n| n.to_string_lossy().into_owned())
}

/// The chunk id encoded in a record filename, e.g. `08a` from
/// `security-sweep-chunk-08a-analysis-detection-neuron.md`.
fn id_from_record(rel: &str) -> String {
    let name = rel
        .rsplit('/')
        .next()
        .unwrap_or(rel)
        .trim_start_matches(RECORD_PREFIX);
    let token = name.split('-').next().unwrap_or_default();
    assert!(
        !token.is_empty(),
        "record {rel} must be named {RECORD_PREFIX}<NN>-<slug>.md"
    );
    normalise_id(token)
}

/// A baseline commit is the falsifiability anchor — a reader must be able to run
/// `git diff <sha>..HEAD -- <files>` against it.
fn is_commit_sha(value: &str) -> bool {
    value.len() >= 7 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    value.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && value.chars().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        })
}

#[test]
fn ledger_files_exist() {
    for rel in [README, INDEX, TEMPLATE] {
        let path = repo_root().join(rel);
        assert!(
            path.is_file(),
            "{rel} must exist — the ledger is the only way to tell a swept chunk from an unswept one"
        );
    }
}

#[test]
fn coverage_index_parses_and_covers_every_overflow_chunk() {
    let ids: Vec<String> = chunk_entries()
        .iter()
        .map(|entry| normalise_id(&string_field(entry, "id")))
        .collect();

    for required in REQUIRED_CHUNKS {
        assert!(
            ids.contains(&normalise_id(required)),
            "chunk {required} (Issue #2083) must have an index entry; found {ids:?}"
        );
    }

    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "each chunk id must appear exactly once in {INDEX}: {ids:?}"
    );
}

#[test]
fn every_entry_carries_the_required_fields_with_a_known_exposure() {
    for entry in chunk_entries() {
        let id = string_field(&entry, "id");
        for key in KEY_ORDER {
            assert!(
                entry.get(key).is_some(),
                "chunk {id} must carry `{key}` — a partial record cannot be audited"
            );
        }
        let exposure = string_field(&entry, "exposure");
        assert!(
            EXPOSURES.contains(&exposure.as_str()),
            "chunk {id} exposure `{exposure}` must be one of {EXPOSURES:?}"
        );
        assert!(
            !string_field(&entry, "name").trim().is_empty(),
            "chunk {id} must carry a human name"
        );
    }
}

#[test]
fn a_claimed_sweep_pins_a_baseline_commit_and_a_record() {
    for entry in chunk_entries() {
        let id = string_field(&entry, "id");
        let swept = &entry["last_swept"];
        if swept.is_null() {
            // "Never recorded" is a datum in its own right: it must not carry
            // half a claim.
            assert!(
                entry["baseline_commit"].is_null() && entry["record"].is_null(),
                "chunk {id} has no sweep date, so it must not claim a baseline commit or a record"
            );
            continue;
        }
        let date = string_field(&entry, "last_swept");
        assert!(
            is_iso_date(&date),
            "chunk {id} last_swept `{date}` must be an ISO YYYY-MM-DD date"
        );
        let sha = string_field(&entry, "baseline_commit");
        assert!(
            is_commit_sha(&sha),
            "chunk {id} claims a sweep on {date} but `{sha}` is not a commit SHA — a record with no SHA is worthless"
        );
        assert!(
            !string_field(&entry, "record").trim().is_empty(),
            "chunk {id} claims a sweep on {date} but names no prose record"
        );
    }
}

#[test]
fn every_entry_is_one_line_with_a_stable_key_order() {
    let raw = read(INDEX);
    let entry_lines: Vec<&str> = raw.lines().filter(|line| line.contains("\"id\"")).collect();
    assert_eq!(
        entry_lines.len(),
        chunk_entries().len(),
        "each chunk entry must sit on its own line in {INDEX} so concurrent sweeps conflict on one line"
    );

    for line in entry_lines {
        let mut cursor = 0usize;
        for key in KEY_ORDER {
            let needle = format!("\"{key}\"");
            let Some(offset) = line[cursor..].find(&needle) else {
                panic!(
                    "entry line must spell keys in the order {KEY_ORDER:?}, missing `{key}`: {line}"
                );
            };
            cursor = cursor + offset + needle.len();
        }
    }
}

#[test]
fn prose_records_and_index_entries_match_both_ways() {
    let entries = chunk_entries();
    let indexed: Vec<String> = entries
        .iter()
        .filter(|entry| !entry["record"].is_null())
        .map(|entry| string_field(entry, "record"))
        .collect();

    // A prose record with no index entry is invisible to the next automated run.
    for record in record_files() {
        assert!(
            indexed.contains(&record),
            "{record} has no entry in {INDEX} — the next automated sweep cannot see it"
        );
        let file_id = id_from_record(&record);
        // Null-safe: `string_field` panics on an unswept entry's `record: null`,
        // and the scan passes those to reach a record filed further down the
        // index. Match on the optional string instead.
        let entry = entries
            .iter()
            .find(|entry| entry["record"].as_str() == Some(record.as_str()))
            .expect("record was just found in the index");
        assert_eq!(
            normalise_id(&string_field(entry, "id")),
            file_id,
            "{record} is filed under a different chunk id than its index entry"
        );
    }

    // An index entry naming a record that does not exist is a false claim.
    for record in indexed {
        assert!(
            repo_root().join(&record).is_file(),
            "{INDEX} names {record}, which does not exist"
        );
    }
}

#[test]
fn readme_states_the_per_chunk_file_rule_and_the_commit_sha_requirement() {
    let readme = read(README).to_lowercase();
    assert!(
        readme.contains("security-sweep-chunk-"),
        "{README} must name the per-chunk record filename pattern"
    );
    assert!(
        readme.contains("one file per chunk"),
        "{README} must state the one-file-per-chunk rule — a shared append-only document guarantees merge conflicts"
    );
    assert!(
        readme.contains("commit sha"),
        "{README} must state that every record pins a baseline commit SHA"
    );
    assert!(
        readme.contains("git diff"),
        "{README} must show how to falsify a record against its baseline commit"
    );
}

#[test]
fn template_carries_every_required_record_field() {
    let template = read(TEMPLATE).to_lowercase();
    for field in [
        "chunk id",
        "exposure",
        "baseline commit",
        "sweep date",
        "defect classes",
        "outcome",
        "issues filed",
    ] {
        assert!(
            template.contains(field),
            "{TEMPLATE} must prompt for `{field}` so the eight chunk issues produce comparable records"
        );
    }
}

#[test]
fn security_policy_links_the_sweep_ledger() {
    let security = read("SECURITY.md");
    assert!(
        security.contains(README),
        "SECURITY.md must point at {README} so a responder can find the sweep ledger"
    );
}
