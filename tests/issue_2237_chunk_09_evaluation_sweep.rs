//! Contract tests for the chunk 9 evaluation sweep, helpful + harmful halves
//! (Issue #2237).
//!
//! Pins what the slice writes into the evaluation region of the chunk 9
//! record against the real code it describes: the first-failing set lengths
//! are recomputed from `wgpu::Limits::default()` and the host struct sizes, the
//! byte cap is called to show it never bounds one set, the queue's device-lost
//! classifier is fed the exact messages the record quotes. It also pins the
//! `copy_size` chain, the per-module check tables, the ledger rows and the two
//! inventory outcomes. Line numbers are baseline-relative, so they are not
//! re-read from the live source: the record's "Verify this record" diff is the
//! drift check.
//! `tests/issue_2088_sweep_ledger_contract.rs` covers the ledger-wide rules.

use std::collections::BTreeSet;
use std::path::PathBuf;

use neat_ai_discovery::analysis::gpu::GPU_MAX_BATCH_ALLOC_BYTES;
use neat_ai_discovery::analysis::gpu::is_device_lost_error;
use neat_ai_discovery::analysis::gpu::shaders::WORKGROUP_SIZE;
use neat_ai_discovery::analysis::samples::{
    GpuHelpfulSample, HarmfulContribution, HelpfulContribution,
};
use neat_ai_discovery::analysis::utils::cap_gpu_batch_size_by_bytes;

const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";
const HELPFUL: &str = "src/analysis/gpu/helpful_evaluation.rs";
const HARMFUL: &str = "src/analysis/gpu/harmful_evaluation.rs";

const SHARED: &str = "#### Dispatch and binding limits — shared verdict (Issue #2237)";
const HELPFUL_TABLE: &str = "#### `helpful_evaluation.rs` (Issue #2237)";
const HARMFUL_TABLE: &str = "#### `harmful_evaluation.rs` (Issue #2237)";

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

/// Data rows of every Markdown table in `body`; header and separator skipped.
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
            in_table = true;
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

/// Region rows with no header of their own, e.g. every `| SEC-` ledger row.
fn prefixed_rows(body: &str, prefix: &str) -> Vec<Vec<String>> {
    body.lines()
        .map(str::trim)
        .filter(|line| line.starts_with(prefix))
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
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

/// Whether `text` cites a `name.rs:<line>` / `name.wgsl:<line>` location.
fn cites_file_line(text: &str) -> bool {
    [".rs:", ".wgsl:"].iter().any(|ext| {
        text.match_indices(ext).any(|(at, _)| {
            text[at + ext.len()..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        })
    })
}

/// `n` with thousands separators, as the record spells set lengths.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Smallest set length whose `stride`-byte buffer exceeds `limit` bytes.
fn first_failing_len(limit: u64, stride: usize) -> u64 {
    limit / stride as u64 + 1
}

/// `text` with every whitespace run collapsed to one space, so a quote that
/// the Markdown wraps across lines still matches.
fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn evaluation_region(doc: &str) -> &str {
    region(section(doc, "## Audit sections"), "evaluation", "device")
}

#[test]
fn limit_table_matches_wgpu_default_limits_and_struct_strides() {
    let limits = wgpu::Limits::default();
    let binding = limits.max_storage_buffer_binding_size;
    let buffer = limits.max_buffer_size;
    let helpful = std::mem::size_of::<HelpfulContribution>();
    let harmful = std::mem::size_of::<HarmfulContribution>();
    let sample = std::mem::size_of::<GpuHelpfulSample>();
    let dispatch =
        u64::from(limits.max_compute_workgroups_per_dimension) * u64::from(WORKGROUP_SIZE) + 1;

    let doc = read(RECORD);
    let rows = table_rows(section(&doc, SHARED));
    let expect = |module: &str, stride: &str, len: u64| {
        let want = grouped(len);
        assert!(
            rows.iter()
                .any(|r| r[0] == module && r[2].contains(stride) && r[4] == want),
            "the limits table must carry a {module} row with stride {stride} failing at {want}"
        );
    };
    expect(
        "helpful",
        &format!("{helpful} B"),
        first_failing_len(binding, helpful),
    );
    expect(
        "helpful",
        &format!("{helpful} B"),
        first_failing_len(buffer, helpful),
    );
    expect(
        "helpful",
        &format!("{sample} B"),
        first_failing_len(binding, sample),
    );
    expect(
        "harmful",
        &format!("{harmful} B"),
        first_failing_len(binding, harmful),
    );
    expect(
        "harmful",
        &format!("{harmful} B"),
        first_failing_len(buffer, harmful),
    );
    for module in ["helpful", "harmful"] {
        expect(module, "256 per workgroup", dispatch);
    }

    // The verdict the table records: the contribution binding, not the
    // dispatch limit, trips first in both modules.
    for stride in [helpful, harmful] {
        assert!(first_failing_len(binding, stride) < dispatch);
    }
    let first: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[5].contains("trips first"))
        .collect();
    assert_eq!(first.len(), 2, "one first-tripping row per module");
    assert!(
        first
            .iter()
            .all(|r| r[3] == "`max_storage_buffer_binding_size`")
    );
}

#[test]
fn byte_cap_bounds_set_count_never_one_set_length() {
    let limits = wgpu::Limits::default();
    let stride = std::mem::size_of::<HelpfulContribution>();
    let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>() + 2 * stride;
    let oversized = usize::try_from(first_failing_len(
        limits.max_storage_buffer_binding_size,
        stride,
    ))
    .expect("fits usize");

    // At the first failing length, and far beyond it, the cap shrinks the
    // chunk to one set but still admits that whole oversized set.
    for len in [oversized, oversized * 10] {
        assert_eq!(
            cap_gpu_batch_size_by_bytes(64, len, bytes_per_sample, GPU_MAX_BATCH_ALLOC_BYTES),
            1,
            "a {len}-sample set is admitted alone, not refused or split"
        );
    }

    let doc = read(RECORD);
    let shared = section(&doc, SHARED);
    assert!(shared.contains("`src/analysis/utils/memory.rs:633`"));
    assert!(shared.contains("**number of sample sets**"));
}

#[test]
fn queue_classification_verdict_matches_is_device_lost_error() {
    let doc = read(RECORD);
    let shared = flat(section(&doc, SHARED));
    assert!(shared.contains("as a panic, not an `Err`"));
    assert!(shared.contains("`queue/recovery.rs:56`"));
    assert!(shared.contains("`\"internal error\"`"));

    // The messages the record says reach (or would reach) the classifier are
    // not device loss, so no re-initialisation follows.
    let quoted = [
        "GPU response channel closed unexpectedly — the GPU thread may have exited or panicked",
        "GPU work queue channel closed",
        "wgpu error: Validation Error\n\nCaused by:\n  In Device::create_bind_group, label = \
         'helpful-bind-group-pool'\n    Buffer binding 1 range 134217744 exceeds \
         `max_*_buffer_binding_size` limit 134217728",
    ];
    for msg in quoted {
        if !msg.starts_with("wgpu error") {
            assert!(shared.contains(msg), "the record must quote: {msg}");
        }
        assert!(
            !is_device_lost_error(&anyhow::anyhow!("{msg}")),
            "must not classify as device loss: {msg}"
        );
    }
    // #2313: the map-wait timeout text *is* device loss, which the callback
    // panic preempts.
    assert!(is_device_lost_error(&anyhow::anyhow!(
        "GPU batch buffer mapping timed out after 0.0s. The GPU driver may be unresponsive."
    )));
}

#[test]
fn helpful_and_harmful_tables_give_checks_1_to_6_a_verdict() {
    let doc = read(RECORD);
    // The slice writes only inside its own region, so concurrent PRs do not clash.
    let own = evaluation_region(&doc);
    for heading in [SHARED, HELPFUL_TABLE, HARMFUL_TABLE] {
        assert!(
            own.contains(heading),
            "`{heading}` must sit in the evaluation region"
        );
    }
    for heading in [HELPFUL_TABLE, HARMFUL_TABLE] {
        let rows = table_rows(section(&doc, heading));
        assert_eq!(rows.len(), 6, "{heading}: one row per check 1–6");
        for (i, row) in rows.iter().enumerate() {
            let check = i + 1;
            assert!(
                row[0].starts_with(&format!("{check} — ")),
                "{heading}: row {check} must be check {check}: {}",
                row[0]
            );
            let verdict = &row[1];
            let ok = if verdict.starts_with("finding") {
                !issue_refs(verdict).is_empty()
            } else if verdict.starts_with("refuted") {
                cites_file_line(verdict)
            } else {
                check == 2 && verdict.starts_with("n/a") && verdict.contains("#2238")
            };
            assert!(
                ok,
                "{heading}: check {check} verdict `{verdict}` must be finding #N or refuted with file:line"
            );
            assert!(!row[2].is_empty(), "{heading}: check {check} evidence");
        }
    }
}

#[test]
fn copy_size_chain_is_written_out_with_line_numbers() {
    let doc = read(RECORD);
    let helpful = section(&doc, HELPFUL_TABLE);
    let chain = flat(
        &helpful[helpful
            .find("**`copy_size` chain (check 4).**")
            .expect("the copy_size chain must be written out")..],
    );
    // Each step names its code and its baseline line, in chain order.
    let steps = [
        ("num_workgroups = (samples.len() as u32)", "(L374)"),
        ("partial_sums_size =", "(L375–L377)"),
        ("used.push((slot_idx, partial_sums_size", "(L405)"),
        ("encoder.copy_buffer_to_buffer(", "(L426)"),
        ("slice(0..copy_size)", "mapped at L441"),
        ("read at L461", "read at L461"),
    ];
    let mut cursor = 0;
    for (code, cite) in steps {
        let at = chain[cursor..]
            .find(code)
            .unwrap_or_else(|| panic!("the chain must name `{code}` after the step before it"))
            + cursor;
        assert!(chain[at..].contains(cite), "`{code}` must cite {cite}");
        cursor = at;
    }
    assert!(chain.contains("**Verdict: refuted.**"));
}

#[test]
fn ledger_and_refuted_regions_carry_the_slice_rows() {
    let doc = read(RECORD);
    let ledger = region(section(&doc, "## Ledger"), "evaluation", "device");
    let rows = prefixed_rows(ledger, "| SEC-");
    assert!(
        rows.len() >= 2,
        "the evaluation ledger region must carry both findings"
    );
    let mut ids = BTreeSet::new();
    let mut linked = BTreeSet::new();
    for row in &rows {
        let hex = row[0].strip_prefix("SEC-").expect("SEC- id");
        assert!(
            hex.len() == 12
                && hex
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "ledger id must be SEC- + 12 lowercase hex: {}",
            row[0]
        );
        assert!(ids.insert(row[0].clone()), "duplicate ledger id {}", row[0]);
        assert!(cites_file_line(&row[1]), "{}: file:line cell", row[0]);
        assert!(row[2].starts_with("CWE-"), "{}: CWE cell", row[0]);
        linked.extend(issue_refs(row.last().expect("status")));
    }

    // Every finding a module table names is linked from the ledger.
    for heading in [HELPFUL_TABLE, HARMFUL_TABLE] {
        for row in table_rows(section(&doc, heading)) {
            if row[1].starts_with("finding") {
                for n in issue_refs(&row[1]) {
                    assert!(linked.contains(&n), "{heading}: #{n} needs a ledger row");
                }
            }
        }
    }

    let refuted = region(
        section(&doc, "## Refuted / not findings"),
        "evaluation",
        "device",
    );
    let rows = prefixed_rows(refuted, "| ");
    assert!(
        !rows.is_empty(),
        "the evaluation refuted region must carry rows"
    );
    for row in &rows {
        assert_eq!(
            row.len(),
            3,
            "candidate | refuting file:line | why: {row:?}"
        );
        assert!(
            cites_file_line(&row[1]),
            "refuting cell must cite file:line: {}",
            row[1]
        );
    }
}

#[test]
fn helpful_and_harmful_inventory_rows_are_no_longer_pending() {
    let doc = read(RECORD);
    let evaluation = section(&doc, "### evaluation");
    for file in [HELPFUL, HARMFUL] {
        let cited = format!("| `{file}` |");
        let row = evaluation
            .lines()
            .find(|l| l.starts_with(&cited))
            .unwrap_or_else(|| panic!("{file} must have an evaluation inventory row"));
        let outcome = row
            .trim_end_matches('|')
            .rsplit('|')
            .next()
            .unwrap_or_default();
        assert!(
            !outcome.contains("pending"),
            "{file} must not read pending: {outcome}"
        );
        assert!(
            outcome.contains("#2313") && outcome.contains("#2314"),
            "{file}: {outcome}"
        );
    }
}
