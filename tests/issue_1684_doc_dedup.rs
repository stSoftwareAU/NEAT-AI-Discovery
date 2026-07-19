//! Issue #1684 — deduplicate drifting copies across the doc set.
//!
//! The same material was maintained in two or more places (README discovery /
//! cost tables, per-doc env-var tables, and several cross-doc reference
//! duplicates), contrary to the repo's single-source rule — and some copies had
//! already drifted. These tests lock in the consolidation: each topic keeps one
//! authoritative home and the other files link to it, so the copies cannot be
//! reintroduced and drift again.

const README: &str = include_str!("../README.md");
const CONFIGURATION: &str = include_str!("../docs/CONFIGURATION.md");
const IMPACT: &str = include_str!("../docs/IMPACT_CALCULATION.md");
const CACHE_TUNING: &str = include_str!("../docs/CACHE_TUNING.md");
const GPU_GUIDE: &str = include_str!("../docs/GPU_GUIDE.md");
const ANALYSIS_DEEP_DIVE: &str = include_str!("../docs/ANALYSIS_DEEP_DIVE.md");
const DROUGHT_PLAYBOOK: &str = include_str!("../docs/DROUGHT_PLAYBOOK.md");
const FOCUS_SELECTION_DOC: &str = include_str!("../docs/FOCUS_SELECTION.md");
const FFI_API: &str = include_str!("../docs/FFI_API.md");
const STREAMING_GUIDE: &str = include_str!("../docs/STREAMING_GUIDE.md");
const DISCOVERIES_README: &str = include_str!("../docs/discoveries/README.md");

// ============================================================================
// README dedup — discovery tables, cost table, remove-neuron maths.
// ============================================================================

#[test]
fn readme_replaces_discovery_tables_with_a_single_link() {
    assert!(
        README.contains("docs/DISCOVERY_TYPES.md#discovery-type-summary"),
        "README must link to the authoritative Discovery Type Summary"
    );
    // The per-row anchor links that duplicated the summary table must be gone.
    assert!(
        !README.contains("docs/DISCOVERY_TYPES.md#saturated-neuron-detection"),
        "README must not re-list the per-type discovery table rows (drift seed)"
    );
    assert!(
        !README.contains("docs/DISCOVERY_TYPES.md#dormant-synapse-detection"),
        "README must not re-list the per-type discovery table rows (drift seed)"
    );
}

#[test]
fn readme_drops_the_cost_function_table_and_keeps_the_link() {
    assert!(
        README.contains("docs/COST_FUNCTION_NOTES.md"),
        "README must link to the authoritative cost-function reference"
    );
    assert!(
        !README.contains("| `MSE` | linear residual"),
        "README must not carry its own cost-function table (COST_FUNCTION_NOTES.md is the home)"
    );
}

#[test]
fn remove_neuron_maths_lives_in_impact_calculation_not_only_the_readme() {
    // The Δw / residual maths now has a live doc home.
    assert!(
        IMPACT.contains("cov(a_c, a_s) / var(a_s)"),
        "IMPACT_CALCULATION.md must document the remove-neuron Δw formula"
    );
    assert!(
        IMPACT.contains("ActivationCovariance"),
        "IMPACT_CALCULATION.md must document the compact sufficient statistic"
    );
    // The README keeps only a short summary plus a link — not the full maths.
    assert!(
        README.contains(
            "docs/IMPACT_CALCULATION.md#remove-neuron-weight-redistribution-compensation-issue-1559"
        ),
        "README must link to the IMPACT_CALCULATION.md home for the compensation maths"
    );
    assert!(
        !README.contains("cov(a_c, a_s) / var(a_s)"),
        "README must not inline the full Δw formula (moved to IMPACT_CALCULATION.md)"
    );
}

// ============================================================================
// Env-var single source — new knobs land in CONFIGURATION.md; #3172 note folded.
// ============================================================================

#[test]
fn configuration_documents_the_previously_undocumented_runtime_knobs() {
    for var in [
        "NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES",
        "NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER",
        "NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH",
        "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT",
        "NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD",
        "NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER",
    ] {
        assert!(
            CONFIGURATION.contains(var),
            "docs/CONFIGURATION.md must document `{var}` (Issue #1684 single source)"
        );
    }
}

#[test]
fn configuration_folds_in_the_lazy_budget_scaling_note() {
    // The #3172 lazy-mode scaling that a sibling doc documented must now live in
    // the single source of truth so the two cannot disagree.
    assert!(
        CONFIGURATION.contains("Issue #3172"),
        "CONFIGURATION.md FOCUS_RANKING_BUDGET_MS row must document the #3172 lazy scaling"
    );
    assert!(
        CONFIGURATION.contains("scaled for lazy"),
        "CONFIGURATION.md must note the budget is scaled in lazy mode"
    );
}

/// Anti-drift guard: every `NEAT_AI_DISCOVERY_*` variable still named in the
/// docs whose env tables were reduced to links must also live in the canonical
/// reference, so those files can never again name a knob the source omits.
#[test]
fn every_variable_named_in_reduced_docs_is_in_the_canonical_reference() {
    for doc in [
        CACHE_TUNING,
        GPU_GUIDE,
        DROUGHT_PLAYBOOK,
        FOCUS_SELECTION_DOC,
    ] {
        for var in discovery_vars(doc) {
            assert!(
                CONFIGURATION.contains(&var),
                "`{var}` is named in a reduced doc but missing from docs/CONFIGURATION.md"
            );
        }
    }
}

// ============================================================================
// Parallel env tables reduced to links.
// ============================================================================

#[test]
fn cache_tuning_env_table_is_reduced_to_a_link() {
    assert!(
        CACHE_TUNING.contains("CONFIGURATION.md#streaming--parquet"),
        "CACHE_TUNING.md must link to the canonical Streaming & Parquet reference"
    );
    assert!(
        !CACHE_TUNING.contains("| `NEAT_AI_DISCOVERY_PRELOAD_ALL` | off | Force PreloadAll"),
        "CACHE_TUNING.md must not carry its own env-var reference table"
    );
}

#[test]
fn gpu_guide_streaming_block_is_reduced_to_a_link() {
    assert!(
        GPU_GUIDE.contains("CONFIGURATION.md#streaming--parquet"),
        "GPU_GUIDE.md must link to the canonical Streaming & Parquet reference"
    );
    assert!(
        !GPU_GUIDE.contains("export NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS=100"),
        "GPU_GUIDE.md must not carry its own streaming env-var example block"
    );
}

#[test]
fn focus_selection_env_table_is_reduced_to_a_link() {
    assert!(
        FOCUS_SELECTION_DOC.contains("CONFIGURATION.md#focus-selection--ranking"),
        "FOCUS_SELECTION.md must link to the canonical Focus selection & ranking reference"
    );
    // The broken pointer at the old README table must be gone.
    assert!(
        !FOCUS_SELECTION_DOC.contains("../README.md#-environment-variables"),
        "FOCUS_SELECTION.md must not point at the removed README env-var table"
    );
}

// ============================================================================
// Cross-doc duplicates — one home, links from the rest.
// ============================================================================

#[test]
fn cache_tier_logic_has_one_home_and_the_others_link_to_it() {
    assert!(
        CACHE_TUNING.contains("## Tier Selection Logic"),
        "CACHE_TUNING.md is the home for tier-selection logic"
    );
    assert!(
        ANALYSIS_DEEP_DIVE.contains("CACHE_TUNING.md#tier-selection-logic"),
        "ANALYSIS_DEEP_DIVE.md must link to the cache tier home"
    );
    assert!(
        GPU_GUIDE.contains("CACHE_TUNING.md#tier-selection-logic"),
        "GPU_GUIDE.md must link to the cache tier home"
    );
    // The duplicated ÷4 / half-memory tables must be gone from the linkers.
    assert!(
        !ANALYSIS_DEEP_DIVE.contains("PreloadAll (< 2GB = 8GB ÷ 4)"),
        "ANALYSIS_DEEP_DIVE.md must not duplicate the tier-selection thresholds"
    );
}

#[test]
fn shutdown_sequence_home_is_ffi_api() {
    assert!(
        CACHE_TUNING.contains("FFI_API.md#analysis-lifecycle-guard-issue-1048"),
        "CACHE_TUNING.md must link to the FFI_API shutdown-sequence home"
    );
}

#[test]
fn drought_diagnostic_schema_home_is_ffi_api() {
    // FFI_API is the field-schema home and must be complete (previously-missing
    // dominant-failure fields folded in).
    assert!(
        FFI_API.contains("dominantFailedModule") && FFI_API.contains("predictedVsActualGapP50"),
        "FFI_API.md must document the full droughtDiagnostic field schema"
    );
    assert!(
        DROUGHT_PLAYBOOK.contains("FFI_API.md#drought-diagnostic-metadata-issue-1202"),
        "DROUGHT_PLAYBOOK.md must link to the FFI_API droughtDiagnostic schema home"
    );
}

#[test]
fn streaming_api_reference_home_is_streaming_guide() {
    assert!(
        FFI_API.contains("STREAMING_GUIDE.md"),
        "FFI_API.md must link to the STREAMING_GUIDE home"
    );
    // The 50 MB size-estimation snippet had two copies; the FFI_API copy is gone.
    assert!(
        !FFI_API.contains("const FLUSH_THRESHOLD = 50 * 1024 * 1024"),
        "FFI_API.md must not duplicate the size-estimation snippet (STREAMING_GUIDE is the home)"
    );
    assert!(
        STREAMING_GUIDE.contains("50"),
        "STREAMING_GUIDE.md remains the home for the size-estimation guidance"
    );
}

#[test]
fn production_success_rates_home_is_discovery_types() {
    assert!(
        DISCOVERIES_README.contains("DISCOVERY_TYPES.md#production-success-rates"),
        "discoveries/README.md must link to the authoritative success-rate table"
    );
    assert!(
        !DISCOVERIES_README.contains("| Remove Low-Impact | **17.6%** | 369 candidates |"),
        "discoveries/README.md must not carry its own success-rate table"
    );
}

// ============================================================================
// Helper — extract distinct `NEAT_AI_DISCOVERY_*` identifiers from text.
// ============================================================================

fn discovery_vars(text: &str) -> Vec<String> {
    const PREFIX: &str = "NEAT_AI_DISCOVERY_";
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(rel) = text[i..].find(PREFIX) {
        let start = i + rel;
        let mut end = start + PREFIX.len();
        while end < bytes.len()
            && (bytes[end].is_ascii_uppercase()
                || bytes[end].is_ascii_digit()
                || bytes[end] == b'_')
        {
            end += 1;
        }
        let mut var = &text[start..end];
        while let Some(stripped) = var.strip_suffix('_') {
            var = stripped;
        }
        let owned = var.to_string();
        if !out.contains(&owned) {
            out.push(owned);
        }
        i = end.max(start + 1);
    }
    out
}
