//! Contract tests for the chunk 9a-2b shader-layer sweep (Issue #2291).
//!
//! Pins what the slice writes into the shaders region of the chunk 9 record:
//! a pipeline-binding-order table (one row per `build_compute_pipeline` call,
//! checked here against the WGSL `@binding` declarations naga parses from the
//! embedded shader constants), a GPU constants table whose values match the
//! compiled constants, a used/unused verdict for every `pub use` that
//! `src/analysis/gpu/mod.rs` really declares, a ledger row per finding and
//! the three inventory rows flipped off `pending`.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules.

use std::collections::BTreeSet;
use std::path::PathBuf;

use naga::{AddressSpace, StorageAccess};
use neat_ai_discovery::analysis::gpu::device::{
    GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS, GPU_BUFFER_MAP_TIMEOUT_SECS,
    GPU_INIT_TIMEOUT_SECS as DEVICE_GPU_INIT_TIMEOUT_SECS,
};
use neat_ai_discovery::analysis::gpu::shaders::{
    ACTIVATION_REDUCE_SHADER, ACTIVATION_SHADER, BIAS_SHADER, GPU_INIT_TIMEOUT_SECS,
    GPU_REDUCTION_THRESHOLD, GPU_SHUTDOWN_TIMEOUT_SECS, HARMFUL_REDUCE_SHADER, HARMFUL_SHADER,
    HELPFUL_REDUCE_SHADER, HELPFUL_SHADER, RELU_SHADER, WORKGROUP_SIZE,
};

const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";
const GPU_MOD: &str = "src/analysis/gpu/mod.rs";

const BINDING_TABLE: &str = "#### Pipeline binding order";
const CONSTANTS_TABLE: &str = "#### GPU constants";
const SURFACE_TABLE: &str = "#### Module surface";

/// One bind-group entry kind, as `pipeline_builder::BindingKind` spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Read,
    ReadWrite,
    Uniform,
}

impl Kind {
    /// The WGSL access spelling the record uses for this kind.
    fn wgsl(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::ReadWrite => "read_write",
            Self::Uniform => "uniform",
        }
    }
}

/// `pipeline_builder::STANDARD_BINDINGS`.
const STANDARD: &[Kind] = &[Kind::Read, Kind::ReadWrite, Kind::Uniform];
/// `pipeline_builder::BIAS_BINDINGS`.
const BIAS: &[Kind] = &[Kind::Read, Kind::Read, Kind::ReadWrite, Kind::Uniform];

/// A `build_compute_pipeline` call site the record must carry a row for.
struct CallSite {
    at: &'static str,
    wgsl: &'static str,
    source: &'static str,
    slice: &'static str,
    kinds: &'static [Kind],
}

/// The eight call sites named in Issue #2291.
fn call_sites() -> [CallSite; 8] {
    let site = |at, wgsl, source, slice, kinds| CallSite {
        at,
        wgsl,
        source,
        slice,
        kinds,
    };
    [
        site(
            "bias_evaluation.rs:30",
            "bias.wgsl",
            BIAS_SHADER,
            "BIAS_BINDINGS",
            BIAS,
        ),
        site(
            "activation_evaluation.rs:35",
            "activation.wgsl",
            ACTIVATION_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "activation_evaluation.rs:53",
            "activation_reduce.wgsl",
            ACTIVATION_REDUCE_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "harmful_evaluation.rs:38",
            "harmful.wgsl",
            HARMFUL_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "harmful_evaluation.rs:56",
            "harmful_reduce.wgsl",
            HARMFUL_REDUCE_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "helpful_evaluation.rs:36",
            "helpful.wgsl",
            HELPFUL_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "helpful_evaluation.rs:54",
            "helpful_reduce.wgsl",
            HELPFUL_REDUCE_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
        site(
            "relu_evaluation.rs:33",
            "relu.wgsl",
            RELU_SHADER,
            "STANDARD_BINDINGS",
            STANDARD,
        ),
    ]
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// The text of a Markdown section, from its heading line to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let mut offset = 0;
    let mut start = None;
    for line in doc.split_inclusive('\n') {
        if line.trim_end() == heading {
            start = Some(offset + line.len());
            break;
        }
        offset += line.len();
    }
    let start = start.unwrap_or_else(|| panic!("{RECORD} must carry the heading `{heading}`"));
    let level = heading.chars().take_while(|c| *c == '#').count();
    let mut end = doc.len();
    let mut cursor = start;
    for line in doc[start..].split_inclusive('\n') {
        let depth = line.chars().take_while(|c| *c == '#').count();
        if depth > 0 && depth <= level {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &doc[start..end]
}

/// The text between `<!-- section: <from> -->` and `<!-- section: <to> -->`.
fn region<'a>(body: &'a str, from: &str, to: &str) -> &'a str {
    let open = format!("<!-- section: {from} -->");
    let close = format!("<!-- section: {to} -->");
    let start = body
        .find(&open)
        .unwrap_or_else(|| panic!("missing `{open}`"))
        + open.len();
    let end = start
        + body[start..]
            .find(&close)
            .unwrap_or_else(|| panic!("missing `{close}` after `{open}`"));
    &body[start..end]
}

/// The data rows of every Markdown table in `body`, as trimmed cells. The
/// header row and the `| --- |` separator are skipped.
fn table_rows(body: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut in_table = false;
    for line in body.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            in_table = false;
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_string())
            .collect();
        if !in_table {
            in_table = true; // header row
            continue;
        }
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue;
        }
        rows.push(cells);
    }
    rows
}

/// Every `#<digits>` issue reference in `text`.
fn issue_refs(text: &str) -> Vec<u32> {
    text.split('#')
        .skip(1)
        .filter_map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .collect()
}

/// The `(binding index, kind)` pairs a shader declares in `@group(0)`,
/// ordered by index.
fn shader_bindings(label: &str, source: &str) -> Vec<(u32, Kind)> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{label} must parse as WGSL: {e}"));
    let mut bindings: Vec<(u32, Kind)> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| {
            let binding = var.binding.as_ref()?;
            assert_eq!(binding.group, 0, "{label}: every binding sits in group 0");
            let kind = match var.space {
                AddressSpace::Uniform => Kind::Uniform,
                AddressSpace::Storage { access } if access.contains(StorageAccess::STORE) => {
                    Kind::ReadWrite
                }
                AddressSpace::Storage { .. } => Kind::Read,
                other => panic!("{label}: unexpected bound address space {other:?}"),
            };
            Some((binding.binding, kind))
        })
        .collect();
    bindings.sort_by_key(|(index, _)| *index);
    bindings
}

/// Every name a `pub use` in `source` brings into scope (the alias for
/// `X as Y`), in declaration order. Parses statements, not lines, so grouped
/// and multi-line imports are expanded.
fn pub_use_names(source: &str) -> Vec<String> {
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or_default())
        .collect::<Vec<_>>()
        .join(" ");
    let mut names = Vec::new();
    for statement in code.split(';') {
        let Some(at) = statement.find("pub use ") else {
            continue;
        };
        let path = &statement[at + "pub use ".len()..];
        let leaves: Vec<&str> = match (path.find('{'), path.rfind('}')) {
            (Some(open), Some(close)) => path[open + 1..close].split(',').collect(),
            _ => vec![path.rsplit("::").next().unwrap_or_default()],
        };
        for leaf in leaves {
            let leaf = leaf.trim();
            if leaf.is_empty() {
                continue;
            }
            let name = leaf.rsplit(" as ").next().unwrap_or(leaf).trim();
            names.push(name.to_string());
        }
    }
    names
}

fn strip_ticks(cell: &str) -> String {
    cell.replace('`', "")
}

#[test]
fn shaders_inventory_rows_record_an_outcome_and_reason() {
    let doc = read(RECORD);
    let shaders = section(section(&doc, "## Files swept"), "### shaders");
    let rows = table_rows(shaders);
    for path in [
        "src/analysis/gpu/mod.rs",
        "src/analysis/gpu/pipeline_builder.rs",
        "src/analysis/gpu/shaders.rs",
    ] {
        let row = rows
            .iter()
            .find(|r| strip_ticks(&r[0]) == path)
            .unwrap_or_else(|| panic!("shaders inventory must carry a row for {path}"));
        let outcome = row.last().expect("row has an outcome cell");
        assert!(
            !outcome.contains("pending"),
            "{path} must no longer be pending: {outcome}"
        );
        let (verdict, reason) = outcome
            .split_once(" — ")
            .unwrap_or_else(|| panic!("{path} outcome must read `<outcome> — <reason>`"));
        assert!(
            !verdict.trim().is_empty() && !reason.trim().is_empty(),
            "{path}: {outcome}"
        );
        if path.ends_with("pipeline_builder.rs") {
            assert!(
                outcome.contains("min_binding_size: None"),
                "pipeline_builder row must note `min_binding_size: None` defers size \
                 validation to draw time: {outcome}"
            );
        }
    }
}

#[test]
fn binding_table_matches_every_call_site_shader_and_slice() {
    let doc = read(RECORD);
    let rows = table_rows(section(&doc, BINDING_TABLE));
    assert_eq!(
        rows.len(),
        8,
        "{BINDING_TABLE} must carry exactly one row per call site"
    );
    for site in call_sites() {
        let row = rows
            .iter()
            .find(|r| r[0].contains(site.at))
            .unwrap_or_else(|| panic!("{BINDING_TABLE} must carry a row for {}", site.at));
        assert_eq!(
            row.len(),
            6,
            "{}: call site | shader | slice | @binding | bind group | verdict",
            site.at
        );
        assert!(row[1].contains(site.wgsl), "{}: shader cell", site.at);
        assert!(row[2].contains(site.slice), "{}: slice cell", site.at);
        assert!(!row[4].is_empty(), "{}: bind-group cell", site.at);
        assert!(!row[5].is_empty(), "{}: verdict cell", site.at);

        // Real code: the shader the call builds declares exactly the slice.
        let declared = shader_bindings(site.wgsl, site.source);
        let expected: Vec<(u32, Kind)> = site
            .kinds
            .iter()
            .copied()
            .zip(0..)
            .map(|(k, i)| (i, k))
            .collect();
        assert_eq!(
            declared, expected,
            "{}: {} bindings must match {}",
            site.at, site.wgsl, site.slice
        );
        // The record cites each index with its access mode.
        for (index, kind) in declared {
            let cited = format!("@binding({index})");
            let at = row[3]
                .find(&cited)
                .unwrap_or_else(|| panic!("{}: shader cell must cite {cited}", site.at));
            let after = &row[3][at + cited.len()..];
            let this_entry = after.split("@binding(").next().unwrap_or_default();
            let access = kind.wgsl();
            let names_access = if kind == Kind::Read {
                this_entry.contains("read") && !this_entry.contains("read_write")
            } else {
                this_entry.contains(access)
            };
            assert!(
                names_access,
                "{}: {cited} must be recorded as `{access}`: {this_entry}",
                site.at
            );
        }
    }
}

#[test]
fn constants_table_values_match_the_compiled_constants() {
    let doc = read(RECORD);
    let rows = table_rows(section(&doc, CONSTANTS_TABLE));
    let expected: [(&str, u64); 6] = [
        ("WORKGROUP_SIZE", u64::from(WORKGROUP_SIZE)),
        ("GPU_REDUCTION_THRESHOLD", GPU_REDUCTION_THRESHOLD as u64),
        ("GPU_SHUTDOWN_TIMEOUT_SECS", GPU_SHUTDOWN_TIMEOUT_SECS),
        (
            "GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS",
            GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS,
        ),
        ("GPU_BUFFER_MAP_TIMEOUT_SECS", GPU_BUFFER_MAP_TIMEOUT_SECS),
        ("GPU_INIT_TIMEOUT_SECS", GPU_INIT_TIMEOUT_SECS),
    ];
    for (name, value) in expected {
        let row = rows
            .iter()
            .find(|r| strip_ticks(&r[0]).split_whitespace().next() == Some(name))
            .unwrap_or_else(|| panic!("{CONSTANTS_TABLE} must carry a row for {name}"));
        assert_eq!(
            row.len(),
            5,
            "{name}: constant | defined at | value | checked against | verdict"
        );
        let digits: String = strip_ticks(&row[2])
            .replace('_', "")
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        assert_eq!(
            digits.parse::<u64>().ok(),
            Some(value),
            "{name}: recorded value must match the compiled constant"
        );
        assert!(!row[4].is_empty(), "{name}: verdict cell");
    }

    // The duplicate is recorded as a finding (with its issue) or a refutation.
    let init = rows
        .iter()
        .find(|r| strip_ticks(&r[0]).starts_with("GPU_INIT_TIMEOUT_SECS"))
        .expect("GPU_INIT_TIMEOUT_SECS row");
    let verdict = &init[4];
    assert!(
        (verdict.contains("finding") && !issue_refs(verdict).is_empty())
            || verdict.contains("refuted"),
        "the duplicated GPU_INIT_TIMEOUT_SECS needs a finding with its issue or a refutation: \
         {verdict}"
    );
    assert!(
        init[1].contains("shaders.rs") && init[1].contains("device.rs"),
        "the row must cite both copies: {}",
        init[1]
    );
    assert_eq!(
        GPU_INIT_TIMEOUT_SECS, DEVICE_GPU_INIT_TIMEOUT_SECS,
        "both copies agree today; the recorded finding is that nothing pins them"
    );
}

#[test]
fn workgroup_size_matches_every_wgsl_file() {
    let dir = repo_root().join("src/shaders");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).expect("src/shaders must be readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("wgsl readable");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{} must parse: {e}", path.display()));
        for entry_point in &module.entry_points {
            assert_eq!(
                entry_point.workgroup_size,
                [WORKGROUP_SIZE, 1, 1],
                "{}::{} must use WORKGROUP_SIZE",
                path.display(),
                entry_point.name
            );
            seen += 1;
        }
    }
    assert_eq!(
        seen, 10,
        "the record's constants table checks 10 `@workgroup_size` lines"
    );
}

#[test]
fn module_surface_gives_every_pub_use_a_verdict() {
    let names = pub_use_names(&read(GPU_MOD));
    let declared: BTreeSet<String> = names.iter().cloned().collect();
    assert_eq!(
        declared.len(),
        names.len(),
        "{GPU_MOD} re-exports each name once"
    );
    assert!(
        declared.contains("SHADER_GPU_INIT_TIMEOUT_SECS"),
        "aliases resolve"
    );

    let doc = read(RECORD);
    let rows = table_rows(section(&doc, SURFACE_TABLE));
    let recorded: BTreeSet<String> = rows.iter().map(|r| strip_ticks(&r[0])).collect();
    assert_eq!(
        recorded.len(),
        rows.len(),
        "{SURFACE_TABLE} lists each name once"
    );
    assert_eq!(
        recorded, declared,
        "{SURFACE_TABLE} must list exactly the names `pub use` brings into {GPU_MOD}"
    );

    for row in &rows {
        assert_eq!(
            row.len(),
            4,
            "{}: re-export | mod.rs line | verdict | evidence",
            row[0]
        );
        let verdict = row[2].as_str();
        if verdict.starts_with("used") {
            let evidence = strip_ticks(&row[3]);
            let outside = evidence
                .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '(' || c == ')')
                .map(|token| token.split(':').next().unwrap_or_default())
                .find(|path| {
                    (path.starts_with("src/")
                        || path.starts_with("tests/")
                        || path.starts_with("benches/"))
                        && !path.starts_with("src/analysis/gpu/")
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{}: a `used` verdict must cite a user outside src/analysis/gpu/",
                        row[0]
                    )
                });
            assert!(
                repo_root().join(outside).is_file(),
                "{}: cited user {outside} must exist",
                row[0]
            );
        } else {
            assert!(
                verdict.starts_with("unused"),
                "{}: verdict must start with used or unused: {verdict}",
                row[0]
            );
            assert!(!row[3].is_empty(), "{}: unused needs a reason", row[0]);
        }
    }
}

#[test]
fn ledger_shaders_region_carries_the_slice_finding() {
    let doc = read(RECORD);
    let ledger = region(section(&doc, "## Ledger"), "shaders", "evaluation");
    // Region rows carry no header of their own: every data row starts `| SEC-`.
    let rows: Vec<Vec<String>> = ledger
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("| SEC-"))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect();
    assert!(
        !rows.is_empty(),
        "the shaders ledger region must carry rows"
    );
    let mut ids = BTreeSet::new();
    let mut cwe_1041 = Vec::new();
    for row in &rows {
        let id = &row[0];
        let hex = id
            .strip_prefix("SEC-")
            .unwrap_or_else(|| panic!("ledger id must start SEC-: {id}"));
        assert!(
            hex.len() == 12
                && hex
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "ledger id must be SEC- + 12 lowercase hex: {id}"
        );
        assert!(ids.insert(id.clone()), "ledger ids must be unique: {id}");
        let status = row.last().expect("status cell");
        assert!(
            !issue_refs(status).is_empty(),
            "{id}: status must link its issue: {status}"
        );
        if row.iter().any(|c| c == "CWE-1041") {
            cwe_1041.push(row.clone());
        }
    }
    assert_eq!(
        cwe_1041.len(),
        1,
        "the shaders region must carry one CWE-1041 row for the duplicate constant"
    );
    let row = &cwe_1041[0];
    assert!(
        row[1].contains("shaders.rs:143"),
        "anchored at the shaders.rs copy: {}",
        row[1]
    );

    // The constants table and the ledger name the same issue.
    let constants = table_rows(section(&doc, CONSTANTS_TABLE));
    let init = constants
        .iter()
        .find(|r| strip_ticks(&r[0]).starts_with("GPU_INIT_TIMEOUT_SECS"))
        .expect("GPU_INIT_TIMEOUT_SECS row");
    assert_eq!(
        issue_refs(row.last().expect("status")),
        issue_refs(&init[4]),
        "ledger status and constants verdict must link the same issue"
    );
}

#[test]
fn refuted_shaders_region_covers_bindings_constants_and_surface() {
    let doc = read(RECORD);
    let refuted = region(
        section(&doc, "## Refuted / not findings"),
        "shaders",
        "evaluation",
    );
    for topic in ["binding", "WORKGROUP_SIZE", "re-export"] {
        assert!(
            refuted
                .lines()
                .any(|line| line.starts_with('|') && line.contains(topic)),
            "the shaders refutations must carry a `{topic}` row"
        );
    }
    let audit = section(&doc, "## Audit sections");
    assert!(
        !audit.contains("Pending — 9a-2b"),
        "the 9a-2b placeholder must be replaced by its audit prose"
    );
}
