//! Discovery module specification builders (Issue #562, #595).
//!
//! This module constructs the vector of `DiscoveryModuleSpec` entries that are
//! dispatched in parallel by `run_discovery_modules_parallel`. Each spec wraps a
//! detection module's load-records → detect → convert-to-candidates pipeline.
//!
//! Sub-modules group related specs by concern:
//! - `neuron_specs` — neuron-focused dispatch specs
//! - `synapse_specs` — synapse-focused dispatch specs
//! - `structural_specs` — structural discovery specs
//! - `scoring_specs` — scoring and recommendation specs
//!
//! ## `discovery_spec!` macro (Issue #773)
//!
//! The `discovery_spec!` macro eliminates the repetitive clone-guard-detect-convert
//! boilerplate shared by ~33 discovery module specs. See `macros.rs` for details.

#[macro_use]
mod macros;
mod neuron_specs;
mod scoring_specs;
mod structural_specs;
mod synapse_specs;

use std::sync::Arc;

use super::cost_function_hint::CostFunctionHint;
use super::detection::topology_cache::CreatureTopologyCache;
use super::task_descriptor::TaskDescriptor;
use super::{
    cache, candidate_clustering, candidate_diversity, discovery_dispatch, ensemble_scoring,
    module_weights::{self, CandidateBudgetConfig, ModuleOutcomeTracker},
    shared, utils,
};

/// Build all discovery module specs for parallel dispatch.
///
/// Each module follows the detect → convert-to-candidates pipeline. The returned
/// specs are consumed by `run_discovery_modules_parallel` which runs the detection
/// closures concurrently and merges results sequentially.
///
/// Issue #754: A shared `CreatureTopologyCache` is pre-computed once and passed
/// to all detection modules, eliminating redundant `HashMap` / `HashSet`
/// construction across 30+ modules.
pub(crate) fn build_discovery_module_specs(
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    topo: &Arc<CreatureTopologyCache>,
    cost_hint: CostFunctionHint,
    task_descriptor: TaskDescriptor,
) -> Vec<discovery_dispatch::DiscoveryModuleSpec> {
    let mut modules: Vec<discovery_dispatch::DiscoveryModuleSpec> = Vec::with_capacity(32);

    neuron_specs::append_neuron_specs(
        &mut modules,
        creature,
        hidden_neurons,
        shared_cache,
        topo,
        cost_hint,
    );
    synapse_specs::append_synapse_specs(&mut modules, creature, hidden_neurons, shared_cache, topo);
    structural_specs::append_structural_specs(
        &mut modules,
        creature,
        hidden_neurons,
        shared_cache,
        topo,
    );
    scoring_specs::append_scoring_specs(
        &mut modules,
        creature,
        shared_cache,
        cost_hint,
        task_descriptor,
    );

    modules
}

/// Prepare and run discovery module detection phases without merging (Issue #1004).
///
/// Builds module specs, allocates candidate budgets, and runs all detection
/// closures in parallel. Returns the detection results for deferred merging
/// via [`discovery_dispatch::merge_discovery_module_results`].
///
/// This separation allows the detection phase to overlap with other concurrent
/// work (e.g., candidate compression) before merging results sequentially.
///
/// ## Deadline enforcement (Issue #1029)
///
/// The `deadline` parameter is forwarded to [`discovery_dispatch::detect_discovery_modules_parallel`]
/// so that modules are skipped when the analysis time budget is exhausted,
/// preventing the detection phase from running indefinitely.
///
/// ## Removed: starvation wrapper (Issue #1793)
///
/// A `..._with_starvation` twin of this function used to take a
/// `ModuleStarvationTracker` and a `current_epoch`, and this function delegated
/// to it with a hard-coded `None` / `0`. The tracker was never populated in
/// production, so the pair was removed rather than wired — see the decision on
/// Issue #1793. The population-wide [`ModuleOutcomeTracker`] gate below is the
/// surviving per-module suppressor.
///
/// ## Creature-scale module tiering (Issue #1547)
///
/// When the creature is large (`hidden_neurons.len()` exceeds
/// [`crate::config::module_tiering_hidden_neuron_threshold`]) and no drought /
/// novelty escalation is active (`escalation_active == false`), expensive-tier
/// discovery modules are filtered out of the dispatched set before detection.
/// On escalation passes, below the threshold, or with tiering disabled
/// (threshold `0`), the full set is dispatched exactly as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_and_detect_discovery_modules(
    creature: &Arc<crate::CreatureJson>,
    hidden_neurons: &Arc<Vec<(String, String, f32)>>,
    shared_cache: &Arc<cache::RecordCache>,
    tracker: &ModuleOutcomeTracker,
    deadline: Option<std::time::SystemTime>,
    cost_hint: CostFunctionHint,
    task_descriptor: TaskDescriptor,
    escalation_active: bool,
) -> discovery_dispatch::DiscoveryModuleDetectionResults {
    // Issue #754: Pre-compute topology cache once for all detection modules.
    let topo = Arc::new(CreatureTopologyCache::new(creature));
    let mut modules = build_discovery_module_specs(
        creature,
        hidden_neurons,
        shared_cache,
        &topo,
        cost_hint,
        task_descriptor,
    );

    // Issue #1547: Creature-scale module tiering — on large creatures outside
    // escalation, drop expensive-tier modules before dispatch so post-processing
    // budget is not spent on super-linear scans. Escalation passes keep the full
    // set (and log that they did, so a missing re-enable is diagnosable).
    apply_module_tiering(&mut modules, hidden_neurons.len(), escalation_active);

    // Issue #967: Allocate candidate budgets based on module success rates.
    let config = CandidateBudgetConfig::default();
    let module_names: Vec<String> = modules.iter().map(|m| m.module_name.clone()).collect();
    let budgets = module_weights::allocate_candidate_budgets(&module_names, tracker, &config);
    for module in &mut modules {
        if let Some(&budget) = budgets.get(&module.module_name) {
            module.max_candidates = budget;
        }
    }

    discovery_dispatch::detect_discovery_modules_parallel(modules, deadline, Some(tracker))
}

/// Filter expensive-tier discovery modules out of the dispatch set on large
/// creatures outside escalation (Issue #1547).
///
/// The predicate is [`super::module_tiering::should_skip_module`], so this
/// filtering shares exactly the logic exercised by the unit and integration
/// tests. When escalation is active on a large creature, nothing is filtered and
/// a re-enable line is logged — its absence during a drought pass is the
/// documented diagnostic signal.
fn apply_module_tiering(
    modules: &mut Vec<discovery_dispatch::DiscoveryModuleSpec>,
    hidden_neuron_count: usize,
    escalation_active: bool,
) {
    use super::module_tiering;

    let threshold = crate::config::module_tiering_hidden_neuron_threshold();

    if !module_tiering::tiering_applies(hidden_neuron_count, threshold, escalation_active) {
        // Large-and-escalating is the case worth logging: the full set is
        // deliberately kept so the escalation pass can try everything (#1422/#1423).
        if escalation_active && threshold > 0 && hidden_neuron_count > threshold {
            tracing::info!(
                hidden_neuron_count,
                threshold,
                module_count = modules.len(),
                "Issue #1547: drought/novelty escalation active — full discovery module \
                 set re-enabled on large creature"
            );
        }
        return;
    }

    let before = modules.len();
    let mut skipped: Vec<String> = Vec::new();
    modules.retain(|m| {
        let skip = module_tiering::should_skip_module(
            &m.module_name,
            hidden_neuron_count,
            threshold,
            escalation_active,
        );
        if skip {
            skipped.push(m.module_name.clone());
        }
        !skip
    });

    if !skipped.is_empty() {
        tracing::info!(
            hidden_neuron_count,
            threshold,
            skipped_count = skipped.len(),
            dispatched_count = modules.len(),
            total_count = before,
            skipped_modules = ?skipped,
            "Issue #1547: creature-scale tiering — expensive discovery modules skipped on \
             large creature (no escalation active)"
        );
    }
}

/// Synthesise cross-detection candidates for co-flagged neurons (Issue #963).
///
/// When multiple detection modules flag the same neuron with compatible operations,
/// a combined coordinated candidate is synthesised containing all operations.
/// Individual candidates are always preserved — synthesis is additive.
pub(crate) fn synthesise_cross_detection_candidates(syn: &mut shared::AnalyzeSynapsesResult) {
    use crate::observability::PhaseTimer;

    if syn.coordinated_structural_candidates.is_empty() {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → cross-detection synthesis starting");
    let _timer = PhaseTimer::new("cross_detection_synthesis");

    let before_count = syn.coordinated_structural_candidates.len();

    let result = super::detection::cross_detection_synthesis::synthesise_cross_detection_candidates(
        std::mem::take(&mut syn.coordinated_structural_candidates),
    );

    syn.coordinated_structural_candidates = result.candidates;

    if result.synthesised_count > 0 && utils::verbose_enabled() {
        tracing::debug!(
            before = before_count,
            after = syn.coordinated_structural_candidates.len(),
            synthesised = result.synthesised_count,
            "Cross-detection synthesis: synthesised combined candidate(s) for co-flagged neuron(s)"
        );
    }

    // Update metadata to reflect synthesised candidates.
    syn.metadata.candidates_returned = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();

    crate::watchdog::beat("analysis::analyze_all → cross-detection synthesis finished");
}

/// Perform cross-module deduplication on coordinated structural candidates.
pub(crate) fn deduplicate_cross_module_candidates(syn: &mut shared::AnalyzeSynapsesResult) {
    use crate::observability::PhaseTimer;

    if syn.coordinated_structural_candidates.is_empty() {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → cross-module deduplication starting");
    let _dedup_timer = PhaseTimer::new("cross_module_deduplication");

    let before_count = syn.coordinated_structural_candidates.len();
    let dedup_result = candidate_clustering::deduplicate_cross_module_candidates(std::mem::take(
        &mut syn.coordinated_structural_candidates,
    ));
    syn.coordinated_structural_candidates = dedup_result.candidates;

    if dedup_result.duplicates_removed > 0 && utils::verbose_enabled() {
        tracing::debug!(
            duplicates_removed = dedup_result.duplicates_removed,
            before_count = before_count,
            remaining = syn.coordinated_structural_candidates.len(),
            "Cross-module deduplication: removed duplicate(s) from coordinated candidate(s)"
        );
    }

    if dedup_result.conflicts_detected > 0 && utils::verbose_enabled() {
        tracing::debug!(
            conflicts_detected = dedup_result.conflicts_detected,
            "Cross-module deduplication: neuron conflict(s) detected (remove vs modify)"
        );
    }

    // Update metadata to reflect the deduplicated count.
    syn.metadata.candidates_returned = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();

    crate::watchdog::beat("analysis::analyze_all → cross-module deduplication finished");
}

/// Perform candidate clustering to reduce redundant ablation tests.
pub(crate) fn cluster_synapse_candidates(
    syn: &mut shared::AnalyzeSynapsesResult,
    creature: &crate::CreatureJson,
) {
    use crate::observability::PhaseTimer;

    crate::watchdog::beat("analysis::analyze_all → candidate clustering starting");
    let _clustering_timer = PhaseTimer::new("candidate_clustering");

    let mut clusterable: Vec<candidate_clustering::ClusterableCandidate> = Vec::new();

    for c in &syn.helpful_synapses {
        clusterable.push(candidate_clustering::ClusterableCandidate {
            from_neuron_uuid: c.from_neuron_uuid.clone(),
            to_neuron_uuid: c.to_neuron_uuid.clone(),
            expected_improvement: c.expected_creature_score_gain,
            neuron_type: creature
                .neurons
                .iter()
                .find(|n| n.uuid == c.from_neuron_uuid)
                .map_or_else(|| "input".to_string(), |n| n.neuron_type.clone()),
        });
    }

    for c in &syn.harmful_synapses {
        clusterable.push(candidate_clustering::ClusterableCandidate {
            from_neuron_uuid: c.from_neuron_uuid.clone(),
            to_neuron_uuid: c.to_neuron_uuid.clone(),
            expected_improvement: c.expected_creature_score_gain,
            neuron_type: creature
                .neurons
                .iter()
                .find(|n| n.uuid == c.from_neuron_uuid)
                .map_or_else(|| "input".to_string(), |n| n.neuron_type.clone()),
        });
    }

    let clusters = candidate_clustering::cluster_candidates(&clusterable);

    if !clusters.is_empty() && utils::verbose_enabled() {
        let total_clustered: usize = clusters.iter().map(|c| c.member_count).sum();
        tracing::debug!(
            cluster_count = clusters.len(),
            clustered_candidates = total_clustered,
            total_candidates = clusterable.len(),
            "Candidate clustering: cluster(s) covering candidate(s)"
        );
    }

    syn.candidate_clusters = clusters;

    crate::watchdog::beat("analysis::analyze_all → candidate clustering finished");
}

/// Apply diversity-aware reranking to coordinated structural candidates (Issue #610).
///
/// Penalises candidates that are structurally similar to higher-ranked candidates,
/// promoting diverse mutation exploration over redundant clusters.
pub(crate) fn apply_diversity_reranking(syn: &mut shared::AnalyzeSynapsesResult) {
    use crate::observability::PhaseTimer;

    if syn.coordinated_structural_candidates.len() <= 1 {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → diversity reranking starting");
    let _timer = PhaseTimer::new("diversity_reranking");

    let before_order: Vec<f32> = syn
        .coordinated_structural_candidates
        .iter()
        .map(|c| c.expected_creature_score_gain)
        .collect();

    let config = candidate_diversity::DiversityConfig::default();
    syn.coordinated_structural_candidates = candidate_diversity::rerank_with_diversity(
        std::mem::take(&mut syn.coordinated_structural_candidates),
        &config,
    );

    if utils::verbose_enabled() {
        let after_order: Vec<f32> = syn
            .coordinated_structural_candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect();
        let reordered = before_order != after_order;
        tracing::debug!(
            candidates = syn.coordinated_structural_candidates.len(),
            reordered,
            "Diversity reranking: applied structural diversity penalty"
        );
    }

    crate::watchdog::beat("analysis::analyze_all → diversity reranking finished");
}

/// Apply ensemble scoring to combine predictions across discovery modules (Issue #572).
///
/// Groups coordinated structural candidates by their target neuron/synapse,
/// boosts candidates that multiple modules agree on, and penalises candidates
/// where modules disagree on the direction of change.
pub(crate) fn apply_ensemble_scoring(
    syn: &mut shared::AnalyzeSynapsesResult,
    tracker: &ModuleOutcomeTracker,
) {
    use crate::observability::PhaseTimer;

    if syn.coordinated_structural_candidates.is_empty() {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → ensemble scoring starting");
    let _timer = PhaseTimer::new("ensemble_scoring");

    let before_count = syn.coordinated_structural_candidates.len();

    let result = ensemble_scoring::apply_ensemble_scoring(
        std::mem::take(&mut syn.coordinated_structural_candidates),
        tracker,
    );

    syn.coordinated_structural_candidates = result.candidates;

    if utils::verbose_enabled() {
        tracing::debug!(
            before = before_count,
            after = syn.coordinated_structural_candidates.len(),
            ensemble = result.ensemble_candidates,
            single_module = result.single_module_candidates,
            "Ensemble scoring: combined cross-module predictions"
        );
    }

    // Update metadata to reflect post-ensemble counts.
    syn.metadata.candidates_returned = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();

    crate::watchdog::beat("analysis::analyze_all → ensemble scoring finished");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    fn test_creature() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "h1".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "o1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![SynapseJson {
                from_uuid: "h1".to_string(),
                to_uuid: "o1".to_string(),
                weight: 0.5,
                synapse_type: None,
            }],
            input: 1,
            output: 1,
        }
    }

    fn empty_cache() -> cache::RecordCache {
        cache::RecordCache::with_loader(
            "test.parquet",
            Arc::new(
                |_file: &str, _uuid: &str| -> anyhow::Result<Vec<DiscoverRecord>> { Ok(vec![]) },
            ),
        )
    }

    /// Verify that `build_discovery_module_specs` produces the expected number
    /// of module specs and that each spec has a non-empty name and phase.
    #[test]
    fn test_build_discovery_module_specs_produces_all_modules() {
        let creature = Arc::new(test_creature());
        let hidden: Arc<Vec<(String, String, f32)>> =
            Arc::new(vec![("h1".to_string(), "TANH".to_string(), 0.0)]);
        let cache = Arc::new(empty_cache());
        let topo = Arc::new(
            super::super::detection::topology_cache::CreatureTopologyCache::new(&creature),
        );

        let specs = build_discovery_module_specs(
            &creature,
            &hidden,
            &cache,
            &topo,
            CostFunctionHint::Unknown,
            TaskDescriptor::neutral(),
        );

        // We expect 48 modules across all four spec groups (batch-successful
        // is disabled by default, Issue #1059; merge-redundant-neuron added in
        // Issue #1633).
        assert_eq!(
            specs.len(),
            48,
            "Expected 48 discovery module specs, got {}",
            specs.len()
        );

        // Every spec must have a non-empty module name and phase name.
        for (i, spec) in specs.iter().enumerate() {
            assert!(
                !spec.module_name.is_empty(),
                "Module spec {i} has empty module_name"
            );
            assert!(
                !spec.phase_name.is_empty(),
                "Module spec {i} has empty phase_name"
            );
        }
    }

    /// Verify that all module specs have unique phase names (no duplicates).
    #[test]
    fn test_discovery_module_specs_have_unique_phase_names() {
        let creature = Arc::new(test_creature());
        let hidden: Arc<Vec<(String, String, f32)>> =
            Arc::new(vec![("h1".to_string(), "TANH".to_string(), 0.0)]);
        let cache = Arc::new(empty_cache());
        let topo = Arc::new(
            super::super::detection::topology_cache::CreatureTopologyCache::new(&creature),
        );

        let specs = build_discovery_module_specs(
            &creature,
            &hidden,
            &cache,
            &topo,
            CostFunctionHint::Unknown,
            TaskDescriptor::neutral(),
        );
        let mut phase_names: Vec<&str> = specs.iter().map(|s| s.phase_name).collect();
        let total = phase_names.len();
        phase_names.sort();
        phase_names.dedup();
        assert_eq!(
            phase_names.len(),
            total,
            "Duplicate phase names found in discovery module specs"
        );
    }

    /// Verify that detection closures for empty hidden neurons return None
    /// (modules with hidden guards should short-circuit).
    #[test]
    fn test_discovery_specs_with_empty_hidden_return_none() {
        let creature = Arc::new(test_creature());
        let hidden: Arc<Vec<(String, String, f32)>> = Arc::new(vec![]);
        let cache = Arc::new(empty_cache());
        let topo = Arc::new(
            super::super::detection::topology_cache::CreatureTopologyCache::new(&creature),
        );

        let specs = build_discovery_module_specs(
            &creature,
            &hidden,
            &cache,
            &topo,
            CostFunctionHint::Unknown,
            TaskDescriptor::neutral(),
        );

        // With empty hidden neurons and no records, all modules should return None.
        for spec in specs {
            let result = (spec.detect_fn)();
            assert!(
                result.is_none(),
                "Module '{}' returned Some with empty hidden neurons and no records",
                spec.module_name,
            );
        }
    }
}
