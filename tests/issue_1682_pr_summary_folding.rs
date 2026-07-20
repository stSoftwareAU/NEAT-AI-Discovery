//! Issue #1682 — fold unabsorbed PR-summary learnings into the live docs, then
//! delete the folded summaries and codify the retention rule.
//!
//! The `docs/archive/pr-summaries/` archive held durable learnings — notably
//! first-class *negative results* — that were not reflected anywhere in the live
//! docs. These tests lock the folded learnings into their target docs so they
//! cannot be lost again, and assert the folded summaries were deleted only after
//! their learnings landed (capture is the precondition for deletion):
//!
//!   A. SIMD / vectorisation outcomes (#1009 `SoA` negative result, #1006
//!      auto-vectorisation-first policy, #1075 tanh-path caveat) → `BENCHMARKS.md`.
//!   B. The `cargo upgrade --incompatible` breaking-major-bump trap (wgpu 29→30,
//!      resolved by #1594) → `AGENTS.md` quality gate + `GPU_GUIDE.md`.
//!   C. Release-profile LTO trade-off numbers (#741) → `BENCHMARKS.md`.
//!   Meta. The fold-then-delete retention rule → `pr-summaries/README.md`.

use std::path::Path;

const BENCHMARKS: &str = include_str!("../docs/BENCHMARKS.md");
const GPU_GUIDE: &str = include_str!("../docs/GPU_GUIDE.md");
const AGENTS: &str = include_str!("../AGENTS.md");
const ARCHIVE_README: &str = include_str!("../docs/archive/pr-summaries/README.md");

// ============================================================================
// Theme A — SIMD / vectorisation outcomes folded into BENCHMARKS.md.
// ============================================================================

#[test]
fn benchmarks_has_optimisation_outcomes_section() {
    assert!(
        BENCHMARKS.contains("Optimisation outcomes"),
        "BENCHMARKS.md must carry an 'Optimisation outcomes' section (Issue #1682)"
    );
}

#[test]
fn benchmarks_records_soa_negative_result() {
    // #1009: SoA layout / compiler hints gave no meaningful improvement; the
    // blockers are algorithmic (is_finite branches / function pointers).
    assert!(
        BENCHMARKS.contains("SoA") && BENCHMARKS.contains("#1009"),
        "BENCHMARKS.md must record the #1009 SoA/data-layout negative result (Issue #1682)"
    );
    assert!(
        BENCHMARKS.to_lowercase().contains("is_finite"),
        "BENCHMARKS.md must name is_finite() branches as the vectorisation blocker (Issue #1682)"
    );
    assert!(
        BENCHMARKS.to_lowercase().contains("negative result"),
        "BENCHMARKS.md must label the SoA outcome a negative result (Issue #1682)"
    );
}

#[test]
fn benchmarks_records_auto_vectorisation_first_policy() {
    // #1006: rely on compiler auto-vectorisation before explicit SIMD.
    assert!(
        BENCHMARKS.to_lowercase().contains("auto-vectoris") && BENCHMARKS.contains("#1006"),
        "BENCHMARKS.md must record the #1006 auto-vectorisation-first policy (Issue #1682)"
    );
}

#[test]
fn benchmarks_records_tanh_path_caveat() {
    // #1075: branch elimination helped value-domain paths but was noise on
    // tanh-dominated paths.
    assert!(
        BENCHMARKS.to_lowercase().contains("tanh") && BENCHMARKS.contains("#1075"),
        "BENCHMARKS.md must record the #1075 tanh-path branch-elimination caveat (Issue #1682)"
    );
}

// ============================================================================
// Theme C — release-profile LTO trade-off numbers folded into BENCHMARKS.md.
// ============================================================================

#[test]
fn benchmarks_records_lto_tradeoff() {
    // #741: lto = "fat" + codegen-units = 1 measured -56%/-22%/-8% runtime for a
    // 12s → 3m07s release-compile cost.
    assert!(
        BENCHMARKS.contains("#741"),
        "BENCHMARKS.md must attribute the LTO trade-off to #741 (Issue #1682)"
    );
    assert!(
        BENCHMARKS.contains("lto = \"fat\"") && BENCHMARKS.contains("codegen-units = 1"),
        "BENCHMARKS.md must name the lto=fat + codegen-units=1 release config (Issue #1682)"
    );
    for pct in ["56 %", "22 %", "8 %"] {
        assert!(
            BENCHMARKS.contains(pct),
            "BENCHMARKS.md must record the {pct} LTO pipeline-runtime figure (Issue #1682)"
        );
    }
}

// ============================================================================
// Theme B — the cargo-upgrade breaking-major-bump trap folded into the docs.
// ============================================================================

#[test]
fn agents_quality_gate_warns_about_incompatible_major_bumps() {
    // The note must live near the `cargo upgrade --incompatible` gate step.
    assert!(
        AGENTS.contains("cargo upgrade --incompatible"),
        "AGENTS.md must keep the cargo upgrade --incompatible quality-gate step (Issue #1682)"
    );
    assert!(
        AGENTS.contains("wgpu") && AGENTS.contains("#1594"),
        "AGENTS.md must record the wgpu 29→30 breaking-bump trap resolved by #1594 (Issue #1682)"
    );
}

#[test]
fn gpu_guide_records_wgpu_30_migration() {
    // The migration is complete (#1594); the durable API-breakage learning stays.
    assert!(
        GPU_GUIDE.contains("wgpu 30") && GPU_GUIDE.contains("#1594"),
        "GPU_GUIDE.md must record the completed wgpu 30 migration (Issue #1682)"
    );
    assert!(
        GPU_GUIDE.contains("get_mapped_range"),
        "GPU_GUIDE.md must name the get_mapped_range() Result API breakage (Issue #1682)"
    );
}

// ============================================================================
// Meta — retention rule codified in the archive README.
// ============================================================================

#[test]
fn archive_readme_codifies_fold_then_delete_rule() {
    let lower = ARCHIVE_README.to_lowercase();
    assert!(
        lower.contains("fold") && lower.contains("delete"),
        "archive README must codify the fold-then-delete retention rule (Issue #1682)"
    );
    assert!(
        lower.contains("negative result"),
        "archive README rule must protect negative results from being dropped (Issue #1682)"
    );
    // The old promise of blanket indefinite retention must be gone.
    assert!(
        !ARCHIVE_README.contains("retained for historical reference but are not"),
        "archive README must not keep the blanket indefinite-retention promise (Issue #1682)"
    );
}

// ============================================================================
// Capture is the precondition for deletion — folded summaries are gone.
// ============================================================================

#[test]
fn folded_summaries_were_deleted_after_capture() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/archive/pr-summaries");
    for n in [
        1009, 1006, 1075, 741, 1482, 1484, 1485, 1517, 1518, 1519, 1521, 1532, 1544, 1566,
    ] {
        let path = format!("{dir}/pr-summary-{n}.md");
        assert!(
            !Path::new(&path).exists(),
            "pr-summary-{n}.md must be deleted once its learning is folded into the live docs \
             (Issue #1682)"
        );
    }
    // The migration record (#1594) is the source of the folded wgpu learning and
    // is intentionally retained.
    assert!(
        Path::new(&format!("{dir}/pr-summary-1594.md")).exists(),
        "pr-summary-1594.md (wgpu 30 migration record) must be retained (Issue #1682)"
    );
}
