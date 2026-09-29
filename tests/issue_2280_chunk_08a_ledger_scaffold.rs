//! Contract tests for the staged chunk 8a sweep record skeleton (Issue #2280).
//!
//! The record is staged under `docs/audits/in-progress/` — invisible to
//! `record_files()` in `tests/issue_2088_sweep_ledger_contract.rs`, which reads
//! only the top level of `docs/audits` — while audit sub-issues #2217, #2150,
//! #2218, #2152 and #2153 fill their sections. These tests deliberately do not
//! assert `pending`: the sub-issues flip those outcomes. Finalisation (#2154)
//! `git mv`s the file to the top level in the commit that fills the `"8a"`
//! index entry, and must delete or retarget
//! `no_top_level_chunk_08a_record_exists` and
//! `the_chunk_8a_index_entry_is_still_all_null` in that same commit — both
//! assert facts that finalisation flips.

use std::path::PathBuf;

/// The staged chunk 8a prose record.
const RECORD: &str = "docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md";
/// Machine-readable sweep index — its `"8a"` entry stays null until finalisation.
const INDEX: &str = "docs/audits/lib-sweep-coverage.json";
/// The top-level audits directory `record_files()` scans.
const AUDITS_DIR: &str = "docs/audits";
/// The file-name prefix a premature top-level record would carry.
const TOP_LEVEL_PREFIX: &str = "security-sweep-chunk-08a-";

/// The two directories chunk 8a sweeps, scanned non-recursively.
const SCOPE_DIRS: [&str; 2] = ["src/analysis/detection", "src/analysis/neuron"];

/// Total files across both scope directories — also the row count in
/// `## Files swept`.
const ROW_COUNT: usize = 52;

/// One `###` section per audit sub-issue, in record order, with the files it
/// owns. Owning issues: shared — none (scaffolding only); graph — #2217;
/// pairwise — #2150; per-neuron-a — #2218; per-neuron-b — #2152; neuron —
/// #2153.
const SECTIONS: [(&str, &[&str]); 6] = [
    (
        "shared",
        &[
            "src/analysis/detection/mod.rs",
            "src/analysis/detection/helpers.rs",
            "src/analysis/detection/stats.rs",
            "src/analysis/detection/topology_cache.rs",
            "src/analysis/detection/activation_properties.rs",
        ],
    ),
    (
        "graph",
        &[
            "src/analysis/detection/topology.rs",
            "src/analysis/detection/skip_connection.rs",
            "src/analysis/detection/dead_neuron.rs",
            "src/analysis/detection/compound_degradation.rs",
            "src/analysis/detection/redundant_path.rs",
            "src/analysis/detection/bottleneck.rs",
            "src/analysis/detection/low_impact_neuron.rs",
            "src/analysis/detection/cross_detection_synthesis.rs",
        ],
    ),
    (
        "pairwise",
        &[
            "src/analysis/detection/correlated_error.rs",
            "src/analysis/detection/weight_coherence.rs",
            "src/analysis/detection/co_adaptation.rs",
            "src/analysis/detection/symmetry_breaking.rs",
            "src/analysis/detection/fanin_polarity_conflict.rs",
            "src/analysis/detection/opposing_synapse.rs",
            "src/analysis/detection/output_conflict.rs",
            "src/analysis/detection/hard_sample_cluster.rs",
            "src/analysis/detection/sentinel_cluster.rs",
        ],
    ),
    (
        "per-neuron-a",
        &[
            "src/analysis/detection/activation_mismatch.rs",
            "src/analysis/detection/bias_perturbation.rs",
            "src/analysis/detection/bimodal_neuron.rs",
            "src/analysis/detection/bounded_range.rs",
            "src/analysis/detection/dormant_synapse.rs",
            "src/analysis/detection/error_dispersion.rs",
            "src/analysis/detection/error_plateau.rs",
            "src/analysis/detection/high_error_squash_exploration.rs",
            "src/analysis/detection/input_sensitivity.rs",
            "src/analysis/detection/monotonicity.rs",
            "src/analysis/detection/noise_signal.rs",
            "src/analysis/detection/observation_range.rs",
            "src/analysis/detection/observation_utilisation.rs",
        ],
    ),
    (
        "per-neuron-b",
        &[
            "src/analysis/detection/operating_point.rs",
            "src/analysis/detection/oscillating_neuron.rs",
            "src/analysis/detection/output_range_compression.rs",
            "src/analysis/detection/output_squash_mismatch.rs",
            "src/analysis/detection/restricted_range.rs",
            "src/analysis/detection/saturation.rs",
            "src/analysis/detection/sentinel_gating.rs",
            "src/analysis/detection/squash_weight_rescale.rs",
            "src/analysis/detection/topology_diversification.rs",
            "src/analysis/detection/unbounded_capping.rs",
            "src/analysis/detection/weight_magnitude_reset.rs",
            "src/analysis/detection/weight_polarity_flip.rs",
        ],
    ),
    (
        "neuron",
        &[
            "src/analysis/neuron/evaluation.rs",
            "src/analysis/neuron/mod.rs",
            "src/analysis/neuron/post_processing.rs",
            "src/analysis/neuron/preparation.rs",
            "src/analysis/neuron/ranking_score.rs",
        ],
    ),
];

fn read(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .lines()
        .scan(0usize, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find(|(_, line)| line.trim_end() == heading)
        .map_or_else(
            || panic!("{RECORD} must carry the heading `{heading}`"),
            |(at, _)| at,
        );
    let level = heading.chars().take_while(|c| *c == '#').count();
    let body_start = start + heading.len();

    let mut cursor = body_start;
    let end = loop {
        let Some(offset) = doc[cursor..].find("\n#") else {
            break doc.len();
        };
        let at = cursor + offset + 1;
        let depth = doc[at..].chars().take_while(|c| *c == '#').count();
        if depth <= level {
            break at;
        }
        cursor = at;
    };
    &doc[body_start..end]
}

/// The repo-relative paths in the first cell of each table row under `body`.
fn row_paths(body: &str) -> Vec<String> {
    body.lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .filter_map(|line| {
            let first = line.trim().trim_matches('|').split('|').next()?;
            let path = first.trim().trim_matches('`');
            path.starts_with("src/").then(|| path.to_string())
        })
        .collect()
}

#[test]
fn the_record_is_staged_under_in_progress() {
    let doc = read(RECORD);
    assert!(
        !doc.is_empty(),
        "{RECORD} must exist and be non-empty — the staged skeleton is what the audit \
         sub-issues fill in"
    );
}

#[test]
fn no_top_level_chunk_08a_record_exists() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(AUDITS_DIR);
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("failed to read a {AUDITS_DIR} entry: {e}"));
        let name = entry.file_name();
        let name = name.to_string_lossy();
        assert!(
            !name.starts_with(TOP_LEVEL_PREFIX),
            "{AUDITS_DIR}/{name} must not exist at the top level while the chunk 8a index \
             entry is still null — the #2088 contract would pick it up as a swept record"
        );
    }
}

#[test]
fn files_swept_carries_exactly_six_section_markers_in_order() {
    let doc = read(RECORD);
    let mut found = Vec::new();
    let mut cursor = 0usize;
    while let Some(offset) = doc[cursor..].find("<!-- section:") {
        let at = cursor + offset;
        let rest = &doc[at + "<!-- section:".len()..];
        let end = rest
            .find(" -->")
            .unwrap_or_else(|| panic!("malformed section marker in {RECORD} at byte {at}"));
        found.push(rest[..end].trim().to_string());
        cursor = at + "<!-- section:".len() + end;
    }
    let expected: Vec<&str> = SECTIONS.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        found, expected,
        "{RECORD} must carry exactly the six `<!-- section: NAME -->` markers, in order, so \
         each sub-issue's rows land in a disjoint, unambiguous region"
    );
}

#[test]
fn files_swept_has_exactly_52_rows() {
    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    assert_eq!(
        row_paths(files_swept).len(),
        ROW_COUNT,
        "`## Files swept` in {RECORD} must carry exactly {ROW_COUNT} rows — one per in-scope \
         file, no more and no fewer"
    );
}

#[test]
fn each_in_scope_file_has_exactly_one_row_under_its_owning_section() {
    let mut on_disk: Vec<String> = Vec::new();
    for dir in SCOPE_DIRS {
        let abs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir);
        let entries = std::fs::read_dir(&abs)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", abs.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("failed to read a {dir} entry: {e}"));
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".rs") {
                on_disk.push(format!("{dir}/{name}"));
            }
        }
    }
    on_disk.sort();

    let mut ledgered: Vec<String> = SECTIONS
        .iter()
        .flat_map(|(_, files)| files.iter().map(ToString::to_string))
        .collect();
    ledgered.sort();

    assert_eq!(
        on_disk, ledgered,
        "the files under {SCOPE_DIRS:?} must exactly match the union of SECTIONS in \
         tests/issue_2280_chunk_08a_ledger_scaffold.rs — a file added to or removed from the \
         tree fails here until the ledger's SECTIONS list is updated to match"
    );
    assert_eq!(
        on_disk.len(),
        ROW_COUNT,
        "the enumerated in-scope file count must equal ROW_COUNT"
    );

    let doc = read(RECORD);
    let files_swept = section(&doc, "## Files swept");
    let all_rows = row_paths(files_swept);

    for (name, files) in SECTIONS {
        let heading = format!("### {name}");
        let owned = row_paths(section(files_swept, &heading));
        for file in files {
            let under_heading = owned.iter().filter(|p| p == file).count();
            assert_eq!(
                under_heading, 1,
                "{file} must have exactly one row under `{heading}` — a missing row is never \
                 swept, and a duplicate invites two conflicting outcomes"
            );
            let anywhere = all_rows.iter().filter(|p| p == file).count();
            assert_eq!(
                anywhere, 1,
                "{file} must appear exactly once in `## Files swept`, not also under another section"
            );
        }
    }
}

#[test]
fn the_chunk_8a_index_entry_is_still_all_null() {
    let index = read(INDEX);
    let entry = index
        .lines()
        .find(|line| line.contains(r#""id": "8a""#))
        .unwrap_or_else(|| panic!("{INDEX} must carry a chunk 8a entry"));
    assert!(
        entry.contains(r#""last_swept": null"#),
        "the chunk 8a entry in {INDEX} must still carry `\"last_swept\": null` while the \
         record is staged under in-progress/ and unfinalised: {entry}"
    );
    assert!(
        entry.contains(r#""baseline_commit": null"#),
        "the chunk 8a entry in {INDEX} must still carry `\"baseline_commit\": null`: {entry}"
    );
    assert!(
        entry.contains(r#""record": null"#),
        "the chunk 8a entry in {INDEX} must still carry `\"record\": null`: {entry}"
    );
}
