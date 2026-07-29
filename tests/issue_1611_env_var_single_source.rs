//! Issue #1611 — the `NEAT_AI_DISCOVERY_*` environment-variable reference had
//! drifted between two hand-maintained tables (`README.md` and `AGENTS.md`).
//!
//! These tests enforce a single source of truth: `docs/CONFIGURATION.md` is the
//! one authoritative table, and both `README.md` and `AGENTS.md` point at it
//! rather than duplicating it. The dynamic checks guard against future drift by
//! asserting every variable named anywhere in the README/AGENTS prose is also
//! documented in the canonical reference.

const CONFIGURATION: &str = include_str!("../docs/CONFIGURATION.md");
const README: &str = include_str!("../README.md");
const AGENTS: &str = include_str!("../AGENTS.md");
const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");

/// Canonical union of every user-facing variable that previously lived in the
/// README table, the AGENTS table, or (newly) neither. `docs/CONFIGURATION.md`
/// must document all of them.
const CANONICAL_VARS: &[&str] = &[
    "NEAT_AI_DISCOVERY_LIB_PATH",
    "NEAT_AI_DISCOVERY_VERBOSE",
    "NEAT_AI_DISCOVERY_GPU_BATCH_SIZE",
    "NEAT_AI_DISCOVERY_GPU_TIMING",
    "NEAT_AI_DISCOVERY_QUIET_GPU",
    "NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS",
    "NEAT_AI_DISCOVERY_PREFETCH_DEPTH",
    "NEAT_AI_DISCOVERY_PRELOAD_ALL",
    "NEAT_AI_DISCOVERY_BLOCK_SIZE",
    "NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS",
    "NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE",
    "NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY",
    "NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS",
    "NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS",
    "NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD",
    "NEAT_AI_DISCOVERY_CPU_PRE_REJECT",
    "NEAT_AI_DISCOVERY_HIDDEN_SQUASH_PRUNE",
    "NEAT_AI_DISCOVERY_MH_TEMPERATURE",
    "NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL",
    "NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB",
    "NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS",
    "NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS",
    "NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS",
    "NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION",
    "NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS",
    "NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS",
    "NEAT_AI_DISCOVERY_SESSION_TTL_SECS",
    "NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES",
    "NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET",
    "NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET",
    "NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH",
    "NEAT_AI_DISCOVERY_MAX_ACTIVATION_CONFIGS_PER_TARGET",
    "NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET",
    "NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN",
    "NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE",
    "NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR",
    "NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD",
    "NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD",
    "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS",
    "NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS",
    "NEAT_AI_DISCOVERY_NOVELTY_SUPPRESSION_RATIO",
    "NEAT_AI_DISCOVERY_NOVELTY_GAIN_RELAXATION",
    "NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR",
    "NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB",
    "NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION",
];

/// Extract every distinct `NEAT_AI_DISCOVERY_*` identifier appearing in `text`.
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
        // Trim a trailing underscore so `FOO_` matches `FOO`.
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

#[test]
fn configuration_doc_documents_every_canonical_variable() {
    for var in CANONICAL_VARS {
        assert!(
            CONFIGURATION.contains(var),
            "docs/CONFIGURATION.md must document `{var}` (Issue #1611 single source of truth)"
        );
    }
}

#[test]
fn configuration_doc_includes_the_previously_undocumented_variable() {
    assert!(
        CONFIGURATION.contains("NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET"),
        "docs/CONFIGURATION.md must document NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET, \
         which was documented in neither table before Issue #1611"
    );
}

#[test]
fn readme_points_at_the_canonical_configuration_doc() {
    assert!(
        README.contains("docs/CONFIGURATION.md"),
        "README.md must link to docs/CONFIGURATION.md instead of duplicating the table"
    );
}

#[test]
fn agents_points_at_the_canonical_configuration_doc() {
    assert!(
        AGENTS.contains("docs/CONFIGURATION.md"),
        "AGENTS.md must link to docs/CONFIGURATION.md instead of duplicating the table"
    );
}

/// The whole point of the fix: the reference table must not be duplicated. Both
/// README and AGENTS previously carried a full multi-column env-var table; those
/// tables must be gone so they cannot drift again.
#[test]
fn readme_no_longer_carries_the_duplicate_env_var_table() {
    assert!(
        !README.contains("| Variable | Default | Description |"),
        "README.md must not carry its own env-var reference table (Issue #1611)"
    );
}

#[test]
fn agents_no_longer_carries_the_duplicate_env_var_table() {
    assert!(
        !AGENTS.contains("| Variable | Purpose |"),
        "AGENTS.md must not carry its own env-var reference table (Issue #1611)"
    );
}

/// Anti-drift guard: any variable still named in the README or AGENTS prose
/// (e.g. troubleshooting rows) must also live in the canonical reference, so the
/// two can never again document a knob the source of truth omits.
#[test]
fn every_variable_named_in_readme_or_agents_is_in_the_canonical_doc() {
    for var in discovery_vars(README)
        .into_iter()
        .chain(discovery_vars(AGENTS))
    {
        assert!(
            CONFIGURATION.contains(&var),
            "`{var}` is named in README.md/AGENTS.md but missing from docs/CONFIGURATION.md \
             (Issue #1611 anti-drift)"
        );
    }
}

/// CONTRIBUTING.md must tell contributors to document new variables in exactly
/// one place so the tables cannot drift apart again.
#[test]
fn contributing_documents_the_single_source_rule() {
    assert!(
        CONTRIBUTING.contains("docs/CONFIGURATION.md"),
        "CONTRIBUTING.md must name docs/CONFIGURATION.md as the single home for env vars"
    );
}
