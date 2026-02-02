//! Issue #375: Generic discovery module dispatch pattern.
//!
//! Extracts the boilerplate from `analyze_all()` into a reusable dispatch function.
//! Each discovery module provides a `DiscoveryModule` implementation that defines
//! how to collect records, detect candidates, and convert them to coordinated
//! structural operations.

use crate::analysis::cache::RecordCache;
use crate::analysis::shared::AnalyzeSynapsesResult;
use crate::analysis::utils;
use crate::observability::PhaseTimer;
use crate::{CoordinatedStructuralCandidateJson, CreatureJson};
use std::sync::Arc;

/// Defines a single discovery detection module that can be dispatched
/// generically within `analyze_all()`.
///
/// Each module specifies:
/// - A human-readable name (for logging and watchdog beats)
/// - A phase timer name (for profiling)
/// - How to collect neuron records from the cache
/// - How to detect candidates and convert them to coordinated structural operations
pub trait DiscoveryModule {
    /// Human-readable name used in watchdog beats and verbose logging.
    /// Example: `"saturation"`, `"bottleneck"`.
    fn name(&self) -> &'static str;

    /// Phase timer identifier for profiling.
    /// Example: `"saturation_detection"`, `"bottleneck_detection"`.
    fn phase_name(&self) -> &'static str;

    /// Run detection and return coordinated structural candidates.
    ///
    /// This method encapsulates the entire detect → convert pipeline:
    /// 1. Collect relevant neuron UUIDs
    /// 2. Retrieve records from the shared cache
    /// 3. Call the module-specific detect function
    /// 4. Convert detected items to coordinated structural candidates
    ///
    /// Returns `(detection_count, candidates)` for verbose logging.
    fn detect_and_convert(
        &self,
        creature: &CreatureJson,
        shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>);
}

/// Collect records from the shared cache for a set of neuron UUIDs.
///
/// This is a helper used by `DiscoveryModule` implementations to avoid
/// duplicating the cache-lookup boilerplate.
pub fn collect_records_for_uuids(
    uuids: &[String],
    shared_cache: &Arc<RecordCache>,
) -> Vec<(String, Vec<crate::types::DiscoverRecord>)> {
    uuids
        .iter()
        .filter_map(|uuid| {
            shared_cache
                .get(uuid)
                .ok()
                .map(|records| (uuid.clone(), records.as_ref().to_vec()))
        })
        .collect()
}

/// Collect records from the shared cache for hidden neurons specified
/// as `(uuid, squash, bias)` tuples.
pub fn collect_records_for_hidden_neurons(
    hidden_neurons: &[(String, String, f32)],
    shared_cache: &Arc<RecordCache>,
) -> Vec<(String, Vec<crate::types::DiscoverRecord>)> {
    hidden_neurons
        .iter()
        .filter_map(|(uuid, _, _)| {
            shared_cache
                .get(uuid)
                .ok()
                .map(|records| (uuid.clone(), records.as_ref().to_vec()))
        })
        .collect()
}

/// Dispatch a single discovery module: watchdog beats, phase timer, detection,
/// verbose logging, and merge into the synapse result.
///
/// This replaces the ~30–50 line boilerplate block that was repeated for each
/// of the 9 discovery modules in `analyze_all()`.
pub fn dispatch_discovery_module(
    module: &dyn DiscoveryModule,
    creature: &CreatureJson,
    shared_cache: &Arc<RecordCache>,
    syn: &mut AnalyzeSynapsesResult,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
) {
    let name = module.name();
    crate::watchdog::beat(format!("analysis::analyze_all → {name} detection starting"));
    let _timer = PhaseTimer::new(module.phase_name());

    let (detection_count, candidates) = module.detect_and_convert(creature, shared_cache);

    if !candidates.is_empty() {
        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] {} detection: found {} detected, {} candidate(s)",
                capitalise_first(name),
                detection_count,
                candidates.len()
            );
        }

        super::merge_coordinated_structural_replacements(
            syn,
            candidates,
            max_synapse_candidates,
            diversify,
        );
    }

    crate::watchdog::beat(format!("analysis::analyze_all → {name} detection finished"));
}

/// Dispatch all registered discovery modules in sequence.
pub fn dispatch_all_discovery_modules(
    modules: &[&dyn DiscoveryModule],
    creature: &CreatureJson,
    shared_cache: &Arc<RecordCache>,
    syn: &mut AnalyzeSynapsesResult,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
) {
    for module in modules {
        dispatch_discovery_module(
            *module,
            creature,
            shared_cache,
            syn,
            max_synapse_candidates,
            diversify,
        );
    }
}

/// Capitalise the first character of a string for display.
fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

pub mod modules;

#[cfg(test)]
mod tests;
