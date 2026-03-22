//! Scoring, confidence, and statistical analysis integration tests.

#[path = "../common/mod.rs"]
mod common;

mod cross_validation_test;
mod issue_192_error_distribution_analysis;
mod issue_465_candidate_outcome_cache;
mod issue_465_source_type_scoring;
mod issue_486_neuron_error_distribution;
mod issue_506_score_prediction_pessimism_discount;
mod issue_527_confidence_metrics;
mod issue_572_ensemble_candidate_scoring;
mod issue_605_calibration_tracking;
mod issue_651_error_classification;
mod issue_750_stats_deduplication;
mod issue_753_squash_normalisation;
mod issue_767_stats_pearson_correlation;
mod issue_775_stats_spearman_correlation;
mod issue_805_numeric_safety;
mod issue_887_activation_neuron_boost;
mod issue_888_weight_constraints;
mod issue_891_prediction_calibration;
