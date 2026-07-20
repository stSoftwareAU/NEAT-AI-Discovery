//! Issue #1681 — README and CONTRIBUTING were stale on the FFI symbol location,
//! the available-RAM floor, the version-bump trigger/policy, the Deno permission
//! example, and the repository's mission wording.
//!
//! These tests lock the human docs to the code and to `ci.yml` so the facts
//! cannot drift again:
//!   1. The authoritative FFI symbol list lives under `src/ffi/`, not `src/lib.rs`.
//!   2. The available-RAM row states the real per-platform floor (matched to the
//!      `DEFAULT_MIN_AVAILABLE_MEMORY_GB` constant on this platform).
//!   3. The version-bump docs describe the real CI trigger (every PR, not
//!      `src/`-change detection) and no longer contradict `AGENTS.md` on manual
//!      bumps for direct commits.
//!   4. The Deno permission example grants `--allow-write` (the library writes
//!      Parquet files).
//!   5. The related-repositories row describes proposing mutation candidates, not
//!      hyper-parameter search.

use neat_ai_discovery::analysis::utils::memory::DEFAULT_MIN_AVAILABLE_MEMORY_GB;

const README: &str = include_str!("../README.md");
const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
const AGENTS: &str = include_str!("../AGENTS.md");
const LIB_RS: &str = include_str!("../src/lib.rs");
const FFI_ANALYSIS: &str = include_str!("../src/ffi/analysis.rs");

// ============================================================================
// 1. FFI symbol location — src/ffi/, not src/lib.rs.
// ============================================================================

#[test]
fn readme_points_ffi_symbols_at_src_ffi_not_lib_rs() {
    // The stale claim (list lives in src/lib.rs as #[no_mangle]) must be gone.
    assert!(
        !README.contains("`src/lib.rs` as `#[no_mangle]"),
        "README must not claim the FFI symbol list lives in src/lib.rs (Issue #1681)"
    );
    // The corrected home and the real attribute spelling must be present.
    assert!(
        README.contains("src/ffi/"),
        "README FFI summary must point at src/ffi/ (Issue #1681)"
    );
    assert!(
        README.contains("#[unsafe(no_mangle)]"),
        "README must use the real `#[unsafe(no_mangle)]` attribute spelling (Issue #1681)"
    );
}

#[test]
fn code_confirms_ffi_symbols_live_under_src_ffi() {
    // Ties the doc claim to reality: src/lib.rs exports no FFI symbols; src/ffi/ does.
    assert!(
        !LIB_RS.contains("no_mangle"),
        "src/lib.rs must contain no #[no_mangle] FFI symbols (Issue #1681)"
    );
    assert!(
        FFI_ANALYSIS.contains("#[unsafe(no_mangle)]"),
        "src/ffi/ must carry the #[unsafe(no_mangle)] FFI symbols (Issue #1681)"
    );
}

// ============================================================================
// 2. Available-RAM floor — matches the per-platform code constant.
// ============================================================================

#[test]
fn readme_available_ram_row_states_per_platform_floor() {
    assert!(
        README.contains("0.5 GB (macOS) / 1 GB (Linux)"),
        "README available-RAM row must state the per-platform floor (Issue #1681)"
    );
    // The bare, macOS-wrong "1 GB" row must be gone.
    assert!(
        !README.contains("| **Available RAM** | 1 GB |"),
        "README must not keep the flat 1 GB available-RAM row (wrong for macOS, Issue #1681)"
    );
}

#[test]
fn documented_floor_matches_code_constant_on_this_platform() {
    #[cfg(target_os = "macos")]
    let expected = 0.5_f64;
    #[cfg(not(target_os = "macos"))]
    let expected = 1.0_f64;
    assert!(
        (DEFAULT_MIN_AVAILABLE_MEMORY_GB - expected).abs() < 1e-9,
        "DEFAULT_MIN_AVAILABLE_MEMORY_GB ({DEFAULT_MIN_AVAILABLE_MEMORY_GB}) must match the value \
         documented in the README available-RAM row for this platform (Issue #1681)"
    );
}

// ============================================================================
// 3. Version-bump trigger and policy — match ci.yml and AGENTS.md.
// ============================================================================

#[test]
fn version_bump_docs_describe_every_pr_not_src_detection() {
    // The stale "src/`-change detection" trigger claim must be gone from both docs.
    assert!(
        !README.contains("`src/` changes are detected"),
        "README must not claim CI bumps only when src/ changes are detected (Issue #1681)"
    );
    assert!(
        !CONTRIBUTING.contains("`src/` changes are detected"),
        "CONTRIBUTING must not claim CI bumps only when src/ changes are detected (Issue #1681)"
    );
    // The real trigger — every pull request — must be stated.
    assert!(
        README.contains("every pull request"),
        "README must state CI bumps on every pull request (Issue #1681)"
    );
    assert!(
        CONTRIBUTING.contains("every pull request"),
        "CONTRIBUTING must state CI bumps on every pull request (Issue #1681)"
    );
    // AGENTS.md already describes the every-PR trigger; that consistency anchor stays.
    assert!(
        AGENTS.contains("every PR"),
        "AGENTS.md must keep the every-PR trigger description (Issue #1681)"
    );
}

#[test]
fn manual_bump_policy_is_consistent_across_docs() {
    // The flat, context-free prohibitions that contradicted AGENTS.md must be gone.
    assert!(
        !CONTRIBUTING.contains("Do not manually bump versions."),
        "CONTRIBUTING must not flatly forbid manual bumps (contradicts AGENTS.md, Issue #1681)"
    );
    assert!(
        !README.contains("Do not manually edit version numbers; CI handles patch bumps."),
        "README must not flatly forbid manual bumps (contradicts AGENTS.md, Issue #1681)"
    );
    // Both human docs must state the direct-commit caveat that reconciles them
    // with AGENTS.md (manual bump required when CI does not run).
    assert!(
        README.contains("directly"),
        "README must document the direct-commit manual-bump caveat (Issue #1681)"
    );
    assert!(
        CONTRIBUTING.contains("directly"),
        "CONTRIBUTING must document the direct-commit manual-bump caveat (Issue #1681)"
    );
}

// ============================================================================
// 4. Deno permission example — grants --allow-write.
// ============================================================================

#[test]
fn readme_deno_example_grants_allow_write() {
    assert!(
        README
            .contains("deno run --allow-env --allow-ffi --allow-read --allow-write your-script.ts"),
        "README Deno example must grant --allow-write (the library writes Parquet, Issue #1681)"
    );
    assert!(
        !README.contains("deno run --allow-env --allow-ffi --allow-read your-script.ts"),
        "README Deno example must not omit --allow-write (Issue #1681)"
    );
}

// ============================================================================
// 5. Mission wording — proposes mutation candidates, not hyper-parameter search.
// ============================================================================

#[test]
fn readme_related_repo_role_describes_mutation_candidates() {
    assert!(
        !README.contains("search architectures and hyper-parameters"),
        "README must not describe this repo as hyper-parameter search (Issue #1681)"
    );
    assert!(
        README.contains("mutation candidate"),
        "README related-repositories row must describe proposing mutation candidates (Issue #1681)"
    );
}
