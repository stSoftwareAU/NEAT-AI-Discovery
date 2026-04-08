//! Diagnostic tracking and rejection reason structures for analysis.
//!
//! This module provides visibility into why candidates were rejected during discovery:
//! - `NEAT_AI_DISCOVERY_VERBOSE=1` enables detailed logging
//! - JSON responses include `diagnostics` arrays
//! - Critical for debugging "no candidates found" situations
//!
//! The rejection tracking is essential for understanding discovery behaviour without
//! having to dig through logs.
//!
//! **Extracted from implementation.rs as part of Issue #271**
//!
//! Sub-modules (Issue #524):
//! - `rejection` — Synapse rejection tracking (`RejectionReason`, `TargetDiagnostics`)
//! - `neuron_tracking` — Neuron rejection tracking (`NeuronDiagnostics`)
//! - `target_data` — Target data structures for sample building (`TargetMap`)
//! - `focus_filter` — Focus target filtering and validation

#![allow(clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod focus_filter;
pub(crate) mod mcmc_diagnostics;
mod neuron_tracking;
mod rejection;
mod target_data;

// Re-export all public items to maintain the existing API
pub(crate) use focus_filter::{
    FocusTargetFilterResult, filter_focus_targets_for_neuron_analysis, require_unique_focus,
};
pub(crate) use neuron_tracking::NeuronDiagnostics;
pub(crate) use rejection::{TargetDiagnostics, ThresholdContext};
pub(crate) use target_data::TargetMap;

// RejectionReason is only used by test modules (implementation_tests, inline tests)
#[cfg(test)]
pub(crate) use rejection::RejectionReason;

use crate::analysis::utils::verbose_enabled;
use crate::focus::{RecordProvider, compute_impacts_public, compute_impacts_with_activations};
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// RecordCacheProvider - Adapter for focus impact calculation
// =============================================================================

/// Adapter to allow `RecordCache` to be used where focus impact code expects a `RecordProvider`.
///
/// This lets us compute squash-aware impacts using *recorded activations* for selection squashes
/// (MINIMUM/MAXIMUM/IF) during candidate discounting, rather than falling back to the conservative
/// 1/N probability model.
pub(crate) struct RecordCacheProvider<'a> {
    pub(crate) cache: &'a super::cache::RecordCache,
}

impl RecordProvider for RecordCacheProvider<'_> {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        let records = self.cache.get(neuron_uuid)?;
        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some(records))
        }
    }

    fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Compute neuron impact scores for candidate discounting.
///
/// We prefer activation-based selection statistics when available so MINIMUM/MAXIMUM/IF neurons
/// don't get incorrectly diluted via the 1/N fallback.
pub(crate) fn compute_impact_scores_for_discounting(
    creature: &crate::CreatureJson,
    cache: &super::cache::RecordCache,
) -> HashMap<String, f32> {
    let provider = RecordCacheProvider { cache };
    match compute_impacts_with_activations(creature, &provider) {
        Ok(scores) => scores,
        Err(err) => {
            if verbose_enabled() {
                tracing::debug!(
                    reason = %err,
                    "Falling back to conservative impact calculation (no activation-based selection stats)"
                );
            }
            compute_impacts_public(creature)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_reason_display() {
        assert_eq!(
            format!("{}", RejectionReason::NoSamples),
            "no overlapping discovery samples"
        );
        assert_eq!(
            format!("{}", RejectionReason::ZeroImprovement),
            "no consistent improvement in GPU stats"
        );
        assert_eq!(
            format!("{}", RejectionReason::BelowThreshold),
            "expected improvement below threshold"
        );
    }

    #[test]
    fn test_target_diagnostics_tracks_candidate() {
        let targets = ["target-1".to_string()];
        let target_refs: Vec<&String> = targets.iter().collect();
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = TargetDiagnostics::new_for_tests(
            &target_refs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );

        diagnostics.set_target_record_count("target-1", 100);
        diagnostics.set_total_eligible_sources("target-1", 50);
        diagnostics.record_candidate_attempt("target-1", true);
        diagnostics.mark_candidate_selected("target-1");

        let entry = diagnostics.entry_for("target-1").unwrap();
        assert_eq!(entry.target_record_count, 100);
        assert_eq!(entry.total_eligible_sources, 50);
        assert_eq!(entry.evaluated_candidates, 1);
        assert!(entry.had_candidate);
    }

    #[test]
    fn test_neuron_diagnostics_tracks_filtered() {
        let targets = ["hidden-1".to_string(), "output-1".to_string()];
        let target_refs: Vec<&String> = targets.iter().collect();
        // Issue #216: Methods now take &self, not &mut self
        let diagnostics = NeuronDiagnostics::new_for_tests(
            &target_refs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );

        diagnostics.mark_hidden_filtered("hidden-1");
        diagnostics.mark_candidate_selected("output-1");

        let hidden_entry = diagnostics.entry_for("hidden-1").unwrap();
        assert!(hidden_entry.hidden_filtered);
        assert!(!hidden_entry.had_candidate);

        let output_entry = diagnostics.entry_for("output-1").unwrap();
        assert!(!output_entry.hidden_filtered);
        assert!(output_entry.had_candidate);
    }

    #[test]
    fn test_require_unique_focus_empty() {
        let result = require_unique_focus(&[], "test");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least one focus neuron")
        );
    }

    #[test]
    fn test_require_unique_focus_duplicates() {
        let focus = vec![
            "uuid-1".to_string(),
            "uuid-2".to_string(),
            "uuid-1".to_string(),
        ];
        let result = require_unique_focus(&focus, "test");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn test_require_unique_focus_valid() {
        let focus = vec!["uuid-1".to_string(), "uuid-2".to_string()];
        let result = require_unique_focus(&focus, "test").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], "uuid-1");
        assert_eq!(*result[1], "uuid-2");
    }

    #[test]
    fn test_filter_focus_targets_output_only() {
        let focus = [
            "output-1".to_string(),
            "hidden-1".to_string(),
            "input-0".to_string(),
        ];
        let focus_refs: Vec<&String> = focus.iter().collect();

        let mut type_map: HashMap<std::sync::Arc<str>, String> = HashMap::new();
        type_map.insert(std::sync::Arc::from("output-1"), "output".to_string());
        type_map.insert(std::sync::Arc::from("hidden-1"), "hidden".to_string());
        type_map.insert(std::sync::Arc::from("input-0"), "input".to_string());

        let squash_map = HashMap::new();

        let result =
            filter_focus_targets_for_neuron_analysis(&focus_refs, &type_map, &squash_map, true);

        assert_eq!(result.focus_order, vec!["output-1"]);
        assert_eq!(result.skipped_hidden, vec!["hidden-1"]);
        assert_eq!(result.skipped_input, vec!["input-0"]);
    }

    #[test]
    fn test_filter_focus_targets_allow_hidden() {
        let focus = ["output-1".to_string(), "hidden-1".to_string()];
        let focus_refs: Vec<&String> = focus.iter().collect();

        let mut type_map: HashMap<std::sync::Arc<str>, String> = HashMap::new();
        type_map.insert(std::sync::Arc::from("output-1"), "output".to_string());
        type_map.insert(std::sync::Arc::from("hidden-1"), "hidden".to_string());

        let squash_map = HashMap::new();

        let result =
            filter_focus_targets_for_neuron_analysis(&focus_refs, &type_map, &squash_map, false);

        assert_eq!(result.focus_order, vec!["output-1", "hidden-1"]);
        assert!(result.skipped_hidden.is_empty());
    }

    #[test]
    fn test_target_map_from_records() {
        let records = vec![
            DiscoverRecord {
                neuron_uuid: "target".to_string(),
                obs_index: 0,
                activation: 0.5,
                errors: vec![0.1, 0.2],
                value: Some(0.3),
            },
            DiscoverRecord {
                neuron_uuid: "target".to_string(),
                obs_index: 1,
                activation: 0.7,
                errors: vec![0.3],
                value: None,
            },
        ];

        let target_map = TargetMap::from_records(&records);

        assert_eq!(target_map.map.len(), 2);
        let data_0 = target_map.map.get(&0).unwrap();
        assert!((data_0.avg_error - 0.15).abs() < 0.01); // (0.1 + 0.2) / 2
        assert_eq!(data_0.value, Some(0.3));
    }

    #[test]
    fn test_target_map_build_samples() {
        let target_records = vec![DiscoverRecord {
            neuron_uuid: "target".to_string(),
            obs_index: 0,
            activation: 0.5,
            errors: vec![0.1],
            value: Some(0.3),
        }];

        let source_records = vec![DiscoverRecord {
            neuron_uuid: "source".to_string(),
            obs_index: 0,
            activation: 0.8,
            errors: vec![],
            value: None,
        }];

        let target_map = TargetMap::from_records(&target_records);
        let samples = target_map.build_samples_from(&source_records);

        assert_eq!(samples.len(), 1);
        assert!((samples[0].activation - 0.8).abs() < 0.01);
        assert!((samples[0].avg_error - 0.1).abs() < 0.01);
    }

    // =============================================================================
    // Concurrent Diagnostic Insertion Tests (Issue #216)
    // =============================================================================
    //
    // These tests verify that diagnostic aggregation works correctly when accessed
    // concurrently from multiple threads. The diagnostics structs use DashMap
    // internally for lock-free concurrent access.

    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_target_diagnostics_concurrent_record_count() {
        // Test concurrent updates to target record counts from multiple threads
        let targets: Vec<String> = (0..64).map(|i| format!("target-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(TargetDiagnostics::new_for_tests(&target_refs));

        let handles: Vec<_> = (0..64)
            .map(|i| {
                let diag = Arc::clone(&diagnostics);
                let target_uuid = format!("target-{i}");
                thread::spawn(move || {
                    // Each thread sets record count for its own target
                    diag.set_target_record_count(&target_uuid, (i + 1) * 100);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all updates were recorded correctly
        for i in 0..64 {
            let target_uuid = format!("target-{i}");
            let entry = diagnostics.entry_for(&target_uuid).unwrap();
            assert_eq!(
                entry.target_record_count,
                (i + 1) * 100,
                "Target {target_uuid} has incorrect record count"
            );
        }
    }

    #[test]
    fn test_target_diagnostics_concurrent_candidate_attempts() {
        // Test concurrent candidate attempt recording from multiple threads
        let targets: Vec<String> = (0..8).map(|i| format!("target-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(TargetDiagnostics::new_for_tests(&target_refs));

        // Each thread will record 10 candidate attempts for each target
        let num_threads = 8;
        let attempts_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let diag = Arc::clone(&diagnostics);
                let targets_clone = targets.clone();
                thread::spawn(move || {
                    for target_uuid in &targets_clone {
                        for _ in 0..attempts_per_thread {
                            diag.record_candidate_attempt(target_uuid, true);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all targets received the expected number of candidate attempts
        let expected_attempts = num_threads * attempts_per_thread;
        for target_uuid in &targets {
            let entry = diagnostics.entry_for(target_uuid).unwrap();
            assert_eq!(
                entry.evaluated_candidates, expected_attempts as u32,
                "Target {target_uuid} has incorrect evaluated_candidates count"
            );
            assert_eq!(
                entry.candidates_with_samples, expected_attempts as u32,
                "Target {target_uuid} has incorrect candidates_with_samples count"
            );
        }
    }

    #[test]
    fn test_target_diagnostics_concurrent_mark_selected() {
        // Test concurrent marking of candidates as selected
        let targets: Vec<String> = (0..32).map(|i| format!("target-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(TargetDiagnostics::new_for_tests(&target_refs));

        let handles: Vec<_> = (0..32)
            .map(|i| {
                let diag = Arc::clone(&diagnostics);
                let target_uuid = format!("target-{i}");
                thread::spawn(move || {
                    diag.mark_candidate_selected(&target_uuid);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all targets were marked as having candidates
        for i in 0..32 {
            let target_uuid = format!("target-{i}");
            let entry = diagnostics.entry_for(&target_uuid).unwrap();
            assert!(
                entry.had_candidate,
                "Target {target_uuid} should have had_candidate=true"
            );
        }
    }

    #[test]
    fn test_neuron_diagnostics_concurrent_record_count() {
        // Test concurrent updates to neuron record counts from multiple threads
        let targets: Vec<String> = (0..64).map(|i| format!("neuron-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(NeuronDiagnostics::new_for_tests(&target_refs));

        let handles: Vec<_> = (0..64)
            .map(|i| {
                let diag = Arc::clone(&diagnostics);
                let target_uuid = format!("neuron-{i}");
                thread::spawn(move || {
                    diag.set_target_record_count(&target_uuid, (i + 1) * 50);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all updates were recorded correctly
        for i in 0..64 {
            let target_uuid = format!("neuron-{i}");
            let entry = diagnostics.entry_for(&target_uuid).unwrap();
            assert_eq!(
                entry.target_record_count,
                (i + 1) * 50,
                "Neuron {target_uuid} has incorrect record count"
            );
        }
    }

    #[test]
    fn test_neuron_diagnostics_concurrent_candidate_attempts() {
        // Test concurrent candidate attempt recording from multiple threads
        let targets: Vec<String> = (0..8).map(|i| format!("neuron-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(NeuronDiagnostics::new_for_tests(&target_refs));

        // Each thread will record 10 candidate attempts for each target
        let num_threads = 8;
        let attempts_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let diag = Arc::clone(&diagnostics);
                let targets_clone = targets.clone();
                thread::spawn(move || {
                    for target_uuid in &targets_clone {
                        for _ in 0..attempts_per_thread {
                            diag.record_candidate_attempt(target_uuid, true);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all targets received the expected number of candidate attempts
        let expected_attempts = num_threads * attempts_per_thread;
        for target_uuid in &targets {
            let entry = diagnostics.entry_for(target_uuid).unwrap();
            assert_eq!(
                entry.evaluated_sources, expected_attempts as u32,
                "Neuron {target_uuid} has incorrect evaluated_sources count"
            );
            assert_eq!(
                entry.sources_with_samples, expected_attempts as u32,
                "Neuron {target_uuid} has incorrect sources_with_samples count"
            );
        }
    }

    #[test]
    fn test_neuron_diagnostics_concurrent_filtered_marking() {
        // Test concurrent marking of neurons as filtered
        let targets: Vec<String> = (0..30).map(|i| format!("neuron-{i}")).collect();
        let target_refs: Vec<&str> = targets.iter().map(std::string::String::as_str).collect();
        let diagnostics = Arc::new(NeuronDiagnostics::new_for_tests(&target_refs));

        let handles: Vec<_> = (0..30)
            .map(|i| {
                let diag = Arc::clone(&diagnostics);
                let target_uuid = format!("neuron-{i}");
                thread::spawn(move || {
                    // Distribute different filter types across threads
                    match i % 3 {
                        0 => diag.mark_hidden_filtered(&target_uuid),
                        1 => diag.mark_input_filtered(&target_uuid),
                        _ => diag.mark_constant_filtered(&target_uuid),
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify each neuron was marked with the correct filter
        for i in 0..30 {
            let target_uuid = format!("neuron-{i}");
            let entry = diagnostics.entry_for(&target_uuid).unwrap();
            match i % 3 {
                0 => assert!(
                    entry.hidden_filtered,
                    "Neuron {target_uuid} should have hidden_filtered=true"
                ),
                1 => assert!(
                    entry.input_filtered,
                    "Neuron {target_uuid} should have input_filtered=true"
                ),
                _ => assert!(
                    entry.constant_filtered,
                    "Neuron {target_uuid} should have constant_filtered=true"
                ),
            }
        }
    }
}
