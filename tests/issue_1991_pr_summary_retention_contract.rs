//! Issue #1991 — the PR-summary retention rule stated itself three
//! contradictory ways, and seven durable learnings lived only in the archive.
//!
//! Part 1 pins the retention rule to one answer across all three sites that
//! state it (`docs/archive/pr-summaries/README.md`, `docs/archive/README.md`,
//! `scripts/check-pr-summary-location.sh`): **fold, then delete**, with capture
//! as the precondition. A worker must never again have to guess whether
//! deletion is mandatory or forbidden.
//!
//! Part 2 locks each folded learning into its target doc so it cannot be lost
//! when the summary that carried it is deleted:
//!
//!   1. The nightly toolchain deliberately floats (#1912) → `README.md`.
//!   2. A process-wide singleton needs an injectable value seam (#1929/#1930)
//!      → `CONTRIBUTING.md`.
//!   3. The GPU-queue exit-channel fixture trap (#1930) → `docs/GPU_GUIDE.md`.
//!   4. `Sender::receiver_count()` does not exist in crossbeam 0.5 (#1929)
//!      → `docs/GPU_GUIDE.md`.
//!   5. Raising `SAMPLE_TIMEOUT_SECS` was evaluated and rejected (#1934)
//!      → `docs/GPU_GUIDE.md`.
//!   6. Direction of fix — wire a half-wired documented capability (#1937)
//!      → `docs/archive/README.md`.
//!   7. Symbol-anchored doc references (#1942) → `CONTRIBUTING.md`.
//!
//! Where a learning makes a claim about the code, the test proves the claim
//! against the real code first and only then asserts the prose agrees.

use std::path::Path;
use std::process::Command;

use neat_ai_discovery::debug::SAMPLE_TIMEOUT_SECS;

const README: &str = include_str!("../README.md");
const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
const GPU_GUIDE: &str = include_str!("../docs/GPU_GUIDE.md");
const ARCHIVE_README: &str = include_str!("../docs/archive/README.md");
const SUMMARIES_README: &str = include_str!("../docs/archive/pr-summaries/README.md");
const LOCATION_GUARD: &str = include_str!("../scripts/check-pr-summary-location.sh");

/// Collapse hard line wraps so a wrapped sentence still matches.
fn unwrapped(doc: &str) -> String {
    doc.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Text of the markdown section introduced by `heading`, up to the next heading
/// of the same or a higher level. Lines inside fenced code blocks are ignored
/// when looking for that boundary — a shell comment such as `# Install …` is
/// not a heading.
fn section(doc: &str, heading: &str) -> String {
    let level = heading.chars().filter(|c| *c == '#').count();
    let mut lines = doc.lines().skip_while(|line| line.trim_end() != heading);
    lines
        .next()
        .expect("doc must contain the heading {heading}");

    let mut fenced = false;
    let mut body = Vec::new();
    for line in lines {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        let depth = line.chars().take_while(|c| *c == '#').count();
        if !fenced && depth > 0 && depth <= level && line[depth..].starts_with(' ') {
            break;
        }
        body.push(line);
    }
    body.join("\n")
}

/// The tier table row describing `docs/archive/pr-summaries/*.md`.
fn transient_tier_row() -> &'static str {
    ARCHIVE_README
        .lines()
        .find(|line| line.starts_with("| `docs/archive/pr-summaries/*.md`"))
        .expect("the archive README must carry a transient tier row")
}

// ============================================================================
// 1. One retention rule, stated the same way at all three sites.
// ============================================================================

#[test]
fn the_archive_tier_table_says_fold_then_delete() {
    let row = unwrapped(transient_tier_row());
    assert!(
        !row.contains("leave the summary alone"),
        "the transient tier row must not contradict the fold-then-delete rule (Issue #1991): {row}"
    );
    assert!(
        row.contains("delete"),
        "the transient tier row must state that a folded summary is deleted (Issue #1991): {row}"
    );
    assert!(
        row.contains("pr-summaries/README.md"),
        "the transient tier row must point at the canonical retention rule (Issue #1991): {row}"
    );
}

#[test]
fn the_location_guard_forbids_deleting_only_unfolded_summaries() {
    let text = unwrapped(LOCATION_GUARD);
    assert!(
        !text.contains("never delete — the learnings must be preserved"),
        "the guard must not state an absolute never-delete rule (Issue #1991)"
    );
    assert!(
        text.contains("never delete an unfolded summary"),
        "the guard must forbid deleting an *unfolded* summary, not every summary (Issue #1991)"
    );
    assert!(
        LOCATION_GUARD.contains("docs/archive/pr-summaries/README.md"),
        "the guard must point at the canonical retention rule (Issue #1991)"
    );
}

#[test]
fn the_canonical_retention_rule_keeps_capture_as_the_precondition() {
    let text = unwrapped(SUMMARIES_README);
    assert!(
        text.contains("precondition"),
        "the retention rule must keep capture as the precondition for deletion (Issue #1991)"
    );
    assert!(
        text.contains("No negative result may be dropped"),
        "the retention rule must keep the no-dropped-negative-result guarantee (Issue #1991)"
    );
}

#[test]
fn the_location_guard_passes_on_the_committed_tree() {
    let output = Command::new("bash")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/check-pr-summary-location.sh"
        ))
        .output()
        .expect("the location guard must be runnable");
    assert!(
        output.status.success(),
        "the location guard must pass on the committed tree (Issue #1991): {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// 2. Learning 1 — the nightly toolchain deliberately floats (#1912).
// ============================================================================

#[test]
fn readme_records_why_the_nightly_toolchain_is_not_date_pinned() {
    let fuzzing = unwrapped(&section(README, "### 🔀 Fuzz Testing"));
    assert!(
        fuzzing.contains("#1912"),
        "the Fuzz Testing section must attribute the floating-nightly decision to #1912 \
         (Issue #1991)"
    );
    assert!(
        fuzzing.contains("nightly-YYYY-MM-DD"),
        "the Fuzz Testing section must name the date-pinned form that must not be adopted \
         (Issue #1991)"
    );
    assert!(
        fuzzing.contains("-Z sanitizer"),
        "the Fuzz Testing section must give the sanitiser reason the pin goes stale \
         (Issue #1991)"
    );
    assert!(
        fuzzing.contains("signed channel") || fuzzing.contains("rustup's signed"),
        "the Fuzz Testing section must record why the floating channel is not a supply-chain \
         hole (Issue #1991)"
    );
}

#[test]
fn the_pinning_gate_cross_notes_the_nightly_exemption() {
    let gate = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/quality/cargo_install_pinning.sh"
    ))
    .expect("the pinning gate must be readable");
    assert!(
        unwrapped(&gate).contains("rustup toolchain"),
        "the pinning gate must state that the rule covers `cargo install`, not the rustup \
         toolchain channel (Issue #1991)"
    );
    assert!(
        gate.contains("README.md"),
        "the pinning gate must point at the README for the floating-nightly rationale \
         (Issue #1991)"
    );
}

// ============================================================================
// 3. Learning 2 — a process-wide singleton needs an injectable value seam.
// ============================================================================

#[test]
fn contributing_requires_a_value_seam_for_process_wide_singletons() {
    let organisation = unwrapped(&section(CONTRIBUTING, "### Test Organisation"));
    assert!(
        organisation.contains("#[serial]"),
        "Test Organisation must still describe the #[serial] convention (Issue #1991)"
    );
    assert!(
        organisation.contains("is not enough") || organisation.contains("not sufficient"),
        "Test Organisation must record that #[serial] alone does not contain a tripped \
         singleton (Issue #1991)"
    );
    assert!(
        organisation.contains("seam"),
        "Test Organisation must prescribe an injectable value seam (Issue #1991)"
    );
    for issue in ["#1929", "#1930"] {
        assert!(
            organisation.contains(issue),
            "Test Organisation must attribute the value-seam rule to {issue} (Issue #1991)"
        );
    }
}

#[test]
fn the_breaker_seam_the_convention_describes_really_exists() {
    // Proves the prose: a queue can be pointed at an isolated breaker, so a
    // trip in one test cannot refuse GPU work for the rest of the binary.
    use neat_ai_discovery::analysis::gpu::{GpuCircuitBreaker, GpuTripReason, global_gpu_breaker};

    let isolated = GpuCircuitBreaker::new();
    assert!(
        !isolated.is_tripped(),
        "a fresh isolated breaker must start closed"
    );
    isolated.trip(GpuTripReason::BatchTimeout);
    assert!(
        isolated.is_tripped(),
        "the isolated breaker must trip on demand"
    );
    assert!(
        !global_gpu_breaker().is_tripped(),
        "tripping an isolated breaker must not trip the process-wide one — this is the seam \
         CONTRIBUTING.md documents (Issue #1991)"
    );
}

// ============================================================================
// 4. Learning 3 — the GPU-queue exit-channel fixture trap (#1930).
// ============================================================================

#[test]
fn gpu_guide_records_the_exit_channel_fixture_trap() {
    let text = unwrapped(GPU_GUIDE);
    assert!(
        text.contains("exit-channel sender"),
        "GPU_GUIDE.md must name the exit-channel sender as the fixture trap (Issue #1991)"
    );
    assert!(
        text.contains("abandoned_threads"),
        "GPU_GUIDE.md must record that the trap fabricates an abandoned_threads report \
         (Issue #1991)"
    );
    assert!(
        text.contains("regression signal") || text.contains("regression indicator"),
        "GPU_GUIDE.md must record that the trap makes the documented breaker regression \
         signal lie (Issue #1991)"
    );
}

// ============================================================================
// 5. Learning 4 — crossbeam 0.5 has no `Sender::receiver_count()` (#1929).
// ============================================================================

#[test]
fn gpu_guide_records_why_liveness_is_arc_based() {
    let stale = unwrapped(GPU_GUIDE);
    assert!(
        stale.contains("receiver_count()"),
        "GPU_GUIDE.md must name the API that does not exist (Issue #1991)"
    );
    assert!(
        stale.contains("0.5"),
        "GPU_GUIDE.md must pin the missing API to the crossbeam 0.5 line (Issue #1991)"
    );
    assert!(
        stale.contains("CallerGuard") || stale.contains("Arc"),
        "GPU_GUIDE.md must name the Arc/Weak alternative actually used (Issue #1991)"
    );
}

#[test]
fn crossbeam_is_still_on_the_line_the_note_describes() {
    // The note is only useful while the dependency is unchanged; if crossbeam
    // moves off 0.5 this fails loudly so the note gets revisited.
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"))
        .expect("Cargo.lock must be readable");
    let version = lock
        .split("name = \"crossbeam-channel\"")
        .nth(1)
        .and_then(|rest| rest.split("version = \"").nth(1))
        .and_then(|rest| rest.split('"').next())
        .expect("Cargo.lock must pin crossbeam-channel");
    assert!(
        version.starts_with("0.5"),
        "GPU_GUIDE.md's note is scoped to crossbeam-channel 0.5.x, found {version} \
         (Issue #1991)"
    );
}

// ============================================================================
// 6. Learning 5 — raising `SAMPLE_TIMEOUT_SECS` was rejected (#1934).
// ============================================================================

#[test]
fn gpu_guide_records_the_rejected_sample_timeout_increase() {
    let dump = unwrapped(&section(
        GPU_GUIDE,
        "#### The dump degrades, it never disappears (Issue #1934)",
    ));
    assert!(
        dump.contains("SAMPLE_TIMEOUT_SECS"),
        "the degraded-dump section must name the constant that was evaluated (Issue #1991)"
    );
    assert!(
        dump.contains("rejected") || dump.contains("deliberately not"),
        "the degraded-dump section must record that raising the bound was rejected \
         (Issue #1991)"
    );
    assert!(
        dump.contains("retry") || dump.contains("retrying"),
        "the degraded-dump section must record that retrying the sampler was rejected too \
         (Issue #1991)"
    );
}

#[test]
fn the_sample_timeout_is_still_the_bound_the_note_defends() {
    assert_eq!(
        SAMPLE_TIMEOUT_SECS, 5,
        "the rejected-increase note is written against a 5 s bound (Issue #1991)"
    );
    assert!(
        GPU_GUIDE.contains("5 s"),
        "GPU_GUIDE.md must quote the real 5 s bound (Issue #1991)"
    );
}

// ============================================================================
// 7. Learning 6 — direction of fix: wire it, do not delete it (#1937).
// ============================================================================

#[test]
fn the_archive_readme_records_the_direction_of_fix_rule() {
    let text = unwrapped(ARCHIVE_README);
    assert!(
        text.contains("#1937"),
        "the archive README must attribute the direction-of-fix rule to #1937 (Issue #1991)"
    );
    assert!(
        text.contains("half-wired"),
        "the archive README must name the half-wired capability case (Issue #1991)"
    );
    assert!(
        text.contains("wired") && text.contains("not deleted"),
        "the archive README must state that the capability is wired, not deleted (Issue #1991)"
    );
}

// ============================================================================
// 8. Learning 7 — symbol-anchored doc references (#1942).
// ============================================================================

#[test]
fn contributing_requires_symbol_anchored_doc_references() {
    let text = unwrapped(CONTRIBUTING);
    assert!(
        text.contains("#1942"),
        "CONTRIBUTING.md must attribute the symbol-anchor rule to #1942 (Issue #1991)"
    );
    assert!(
        text.contains("`<file>.rs::<function>`"),
        "CONTRIBUTING.md must give the symbol-anchored citation form (Issue #1991)"
    );
    assert!(
        text.contains("rot") || text.contains("rots"),
        "CONTRIBUTING.md must state why bare line numbers are forbidden (Issue #1991)"
    );
}

// ============================================================================
// Capture is the precondition for deletion — folded summaries are gone.
// ============================================================================

#[test]
fn folded_summaries_were_deleted_after_capture() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/archive/pr-summaries");
    for n in [1912, 1929, 1930, 1934, 1937, 1942] {
        let path = format!("{dir}/pr-summary-{n}.md");
        assert!(
            !Path::new(&path).exists(),
            "pr-summary-{n}.md must be deleted once its learning is folded into the live docs \
             (Issue #1991)"
        );
    }
}
