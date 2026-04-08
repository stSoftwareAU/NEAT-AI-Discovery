//! Focus target filtering and validation for neuron analysis.
//!
//! Provides:
//! - `filter_focus_targets_for_neuron_analysis` — classify focus targets by neuron type
//! - `require_unique_focus` — validate that focus neurons are non-empty and unique

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::analysis::activation::is_threshold_activation;

/// Type alias matching the shared UUID key type from neuron preparation (Issue #1036).
type SharedUuid = Arc<str>;

// =============================================================================
// Focus Target Filtering
// =============================================================================

/// Result of filtering focus targets for add-neuron analysis.
///
/// This is split out to keep the rules testable without needing a GPU (the main analysis path
/// asserts GPU availability).
#[derive(Debug, Default)]
pub(crate) struct FocusTargetFilterResult {
    pub(crate) focus_order: Vec<String>,
    pub(crate) skipped_hidden: Vec<String>,
    pub(crate) skipped_input: Vec<String>,
    pub(crate) skipped_constant: Vec<String>,
    pub(crate) threshold_targets: Vec<String>,
}

/// Filter and classify focus targets for add-neuron analysis.
///
/// Notes (Dec 2025):
/// - By default we allow both output and hidden focus targets (hidden will be impact-discounted later).
/// - When `output_only_targets` is enabled, hidden (and unknown treated-as-hidden) targets are filtered out.
/// - STEP/BIPOLAR targets are tracked in `threshold_targets` for visibility (and potential future branching).
pub(crate) fn filter_focus_targets_for_neuron_analysis(
    unique_focus: &[&String],
    neuron_type_map: &HashMap<SharedUuid, String>,
    neuron_squash_map: &HashMap<SharedUuid, String>,
    output_only_targets: bool,
) -> FocusTargetFilterResult {
    let mut result = FocusTargetFilterResult::default();

    // Helper: record STEP/BIPOLAR targets consistently across output/hidden/unknown.
    let mut record_threshold_target = |uuid: &String| {
        if let Some(squash) = neuron_squash_map.get(uuid.as_str())
            && is_threshold_activation(squash)
        {
            result.threshold_targets.push(uuid.clone());
        }
    };

    result.focus_order = unique_focus
        .iter()
        .filter_map(|uuid| {
            // By default we analyse both output and hidden focus targets (hidden will be discounted).
            // If `output_only_targets` is set, hidden targets are filtered.
            let neuron_type = neuron_type_map.get(uuid.as_str()).map(String::as_str);
            match neuron_type {
                Some("output") => {
                    record_threshold_target(uuid);
                    Some((*uuid).clone())
                }
                Some("hidden") => {
                    if output_only_targets {
                        result.skipped_hidden.push((*uuid).clone());
                        None
                    } else {
                        record_threshold_target(uuid);
                        Some((*uuid).clone())
                    }
                }
                Some("input") => {
                    // Input neurons are observation sources, not computation nodes.
                    result.skipped_input.push((*uuid).clone());
                    None
                }
                Some("constant") => {
                    // Constant neurons don't receive inputs - filter them out.
                    result.skipped_constant.push((*uuid).clone());
                    None
                }
                Some(unknown_type) => {
                    // Unknown type - treat as hidden.
                    tracing::warn!(
                        neuron_type = %unknown_type,
                        neuron_uuid = %uuid,
                        "Unknown neuron type encountered — treating as hidden neuron"
                    );
                    if output_only_targets {
                        result.skipped_hidden.push((*uuid).clone());
                        None
                    } else {
                        record_threshold_target(uuid);
                        Some((*uuid).clone())
                    }
                }
                None => {
                    // Unknown UUID - this is likely a bug, skip it.
                    tracing::warn!(
                        neuron_uuid = %uuid,
                        "Unknown neuron UUID in focus list (not found in creature) — skipping"
                    );
                    None
                }
            }
        })
        .collect();

    result
}

// =============================================================================
// Focus Validation
// =============================================================================

/// Validate that focus neurons list is not empty and contains no duplicates.
///
/// Rust side refuses to run if `focus_neurons` is empty or contains duplicates.
/// Controllers **must** validate and de-duplicate targets before calling into FFI
/// so any upstream issues are surfaced promptly.
pub(crate) fn require_unique_focus<'a>(
    focus_neurons: &'a [String],
    context: &str,
) -> anyhow::Result<Vec<&'a String>> {
    if focus_neurons.is_empty() {
        return Err(anyhow::anyhow!(
            "{context} needs at least one focus neuron. The Deno controller supplied an empty `focus_neurons` array, so there is nothing to analyse. Please fix the upstream request and retry after setting `NEAT_AI_DISCOVERY_VERBOSE=1` if you need extra logging."
        ));
    }

    let mut seen_targets: HashSet<&str> = HashSet::new();
    let mut unique_focus: Vec<&String> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();

    for target_uuid in focus_neurons {
        if seen_targets.insert(target_uuid.as_str()) {
            unique_focus.push(target_uuid);
        } else {
            duplicates.push(target_uuid.clone());
        }
    }

    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        let joined = duplicates.join(", ");
        return Err(anyhow::anyhow!(
            "{context} received duplicate focus neurons ({joined}). Each target must be unique so we can map diagnostics back to the Deno request. We are refusing to continue so the upstream behaviour can be corrected."
        ));
    }

    Ok(unique_focus)
}
