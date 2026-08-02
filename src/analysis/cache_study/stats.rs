//! Aggregation over the candidates corpus: volume and gain-size statistics.
//!
//! Answers the two questions in Issue #1920 — why so few candidates land per
//! run (volume), and which record fields predict a bigger score delta (gain).

// Counts are aggregated into f64 means; the corpus is thousands of records at
// most, far below f64's exact-integer range.
#![allow(clippy::cast_precision_loss)]

use std::collections::BTreeMap;

use super::corpus::{CorpusEntry, Outcome, RecordSource};

/// Deltas at or below this magnitude are "vanishing" — the gain-size concern
/// that prompted the study (successes land at ~1e-7).
pub const VANISHING_GAIN: f64 = 1e-6;

/// Success/failure counts and gain summary for one grouping key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupStats {
    /// Group label (strategy, model hash, day, version or request kind).
    pub label: String,
    /// Records filed under `success/`.
    pub successes: usize,
    /// Records filed under `failures/`.
    pub failures: usize,
    /// Mean `scoreDelta` across the successes in this group.
    pub mean_success_delta: f64,
    /// Median `scoreDelta` across the successes in this group.
    pub median_success_delta: f64,
    /// Largest `scoreDelta` among the successes in this group.
    pub max_success_delta: f64,
    /// Summed `scoreDelta` across the successes — the group's total contribution.
    pub total_success_delta: f64,
}

impl GroupStats {
    /// Records in the group, both outcomes.
    #[must_use]
    pub fn total(&self) -> usize {
        self.successes + self.failures
    }

    /// Share of the group's records that succeeded, or `None` when empty.
    #[must_use]
    pub fn success_rate(&self) -> Option<f64> {
        match self.total() {
            0 => None,
            total => Some(self.successes as f64 / total as f64),
        }
    }
}

/// How strongly one numeric record field tracks gain size and success.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    /// Field label, e.g. `removalCandidate.impact`.
    pub label: String,
    /// Successful records where the field and a positive delta were both present.
    pub gain_samples: usize,
    /// Pearson correlation against `log10(scoreDelta)` over successes.
    pub r_vs_log_gain: Option<f64>,
    /// Records of either outcome where the field was present.
    pub outcome_samples: usize,
    /// Pearson correlation against the success indicator (1 = success, 0 = failure).
    pub r_vs_success: Option<f64>,
}

/// The complete study over a loaded corpus.
#[derive(Debug, Clone)]
pub struct CacheStudy {
    /// Records in the corpus.
    pub total: usize,
    /// Records read from the working tree.
    pub live: usize,
    /// Records recovered from git history.
    pub recovered: usize,
    /// Corpus-wide success/failure and gain summary.
    pub overall: GroupStats,
    /// Successes whose `scoreDelta` is at or below [`VANISHING_GAIN`].
    pub vanishing_successes: usize,
    /// Per-strategy breakdown, largest group first.
    pub by_strategy: Vec<GroupStats>,
    /// Per-model-hash breakdown, largest group first.
    pub by_model_hash: Vec<GroupStats>,
    /// Per-day breakdown, chronological.
    pub by_day: Vec<GroupStats>,
    /// Per-`discoveryVersion` breakdown, chronological by version string.
    pub by_version: Vec<GroupStats>,
    /// Per-`rustRequest` payload kind breakdown, largest group first.
    pub by_request_kind: Vec<GroupStats>,
    /// Gain-size and success predictors.
    pub predictors: Vec<Predictor>,
}

/// A numeric field pulled off a record, plus how to scale it for correlation.
struct PredictorSpec {
    label: &'static str,
    extract: fn(&CorpusEntry) -> Option<f64>,
    /// Correlate `log10(|value|)` instead of the raw value — used for fields
    /// that span many orders of magnitude (impacts, expected gains).
    log_scale: bool,
}

const PREDICTORS: &[PredictorSpec] = &[
    PredictorSpec {
        label: "removalCandidate.impact",
        extract: |e| e.record.request_number(&["removalCandidate", "impact"]),
        log_scale: true,
    },
    PredictorSpec {
        label: "removalCandidate.meanActivation",
        extract: |e| {
            e.record
                .request_number(&["removalCandidate", "meanActivation"])
        },
        log_scale: false,
    },
    PredictorSpec {
        label: "neuronCandidate.expectedCreatureScoreGain",
        extract: |e| {
            e.record
                .request_number(&["neuronCandidate", "expectedCreatureScoreGain"])
        },
        log_scale: true,
    },
    PredictorSpec {
        label: "neuronCandidate.targetNeuronImpact",
        extract: |e| {
            e.record
                .request_number(&["neuronCandidate", "targetNeuronImpact"])
        },
        log_scale: true,
    },
    PredictorSpec {
        label: "neuronCandidate.improvedShare",
        extract: improved_share,
        log_scale: false,
    },
    // Issue #1924: the value `add-neurons` candidates are now ranked on,
    // recomputed from each record's own `expectedCreatureScoreGain` and
    // improved share by the production scorer. The acceptance measure —
    // `r vs success` for the ranking field — is therefore measured on the same
    // arithmetic discovery ranks with.
    PredictorSpec {
        label: "neuronCandidate.rankScore",
        extract: rank_score,
        log_scale: false,
    },
    PredictorSpec {
        label: "expectedErrorReduction",
        extract: |e| e.record.expected_error_reduction,
        log_scale: true,
    },
    PredictorSpec {
        label: "sampleSize",
        extract: |e| e.record.sample_size,
        log_scale: true,
    },
    PredictorSpec {
        label: "originalScore",
        extract: |e| Some(e.record.original_score),
        log_scale: false,
    },
];

/// The production add-neuron rank score for a cached record (Issue #1924).
///
/// Returns `None` when the record carries no `neuronCandidate` prediction or
/// no sample counts — a missing predictor must not be scored as zero.
fn rank_score(entry: &CorpusEntry) -> Option<f64> {
    let gain = entry
        .record
        .request_number(&["neuronCandidate", "expectedCreatureScoreGain"])?;
    let share = improved_share(entry)?;
    Some(crate::analysis::neuron::ranking_score::neuron_rank_score(
        gain, share,
    ))
}

fn improved_share(entry: &CorpusEntry) -> Option<f64> {
    let improved = entry
        .record
        .request_number(&["neuronCandidate", "improvedCount"])?;
    let total = entry
        .record
        .request_number(&["neuronCandidate", "totalCount"])?;
    (total > 0.0).then(|| improved / total)
}

/// Builds the full study from a loaded corpus.
#[must_use]
pub fn study(entries: &[CorpusEntry]) -> CacheStudy {
    CacheStudy {
        total: entries.len(),
        live: count_source(entries, RecordSource::Live),
        recovered: count_source(entries, RecordSource::GitHistory),
        overall: group_stats("all records", entries.iter()),
        vanishing_successes: entries
            .iter()
            .filter(|e| e.outcome == Outcome::Success && e.record.score_delta <= VANISHING_GAIN)
            .count(),
        by_strategy: grouped_by(entries, |e| e.strategy.clone(), SortOrder::BySize),
        by_model_hash: grouped_by(entries, |e| e.model_hash.clone(), SortOrder::BySize),
        by_day: grouped_by(entries, |e| e.record.day().to_string(), SortOrder::ByLabel),
        by_version: grouped_by(
            entries,
            |e| e.record.discovery_version.clone(),
            SortOrder::ByLabel,
        ),
        by_request_kind: grouped_by(entries, |e| e.record.request_kind(), SortOrder::BySize),
        predictors: predictors(entries),
    }
}

fn count_source(entries: &[CorpusEntry], source: RecordSource) -> usize {
    entries.iter().filter(|e| e.source == source).count()
}

enum SortOrder {
    BySize,
    ByLabel,
}

fn grouped_by(
    entries: &[CorpusEntry],
    key: impl Fn(&CorpusEntry) -> String,
    order: SortOrder,
) -> Vec<GroupStats> {
    let mut buckets: BTreeMap<String, Vec<&CorpusEntry>> = BTreeMap::new();
    for entry in entries {
        buckets.entry(key(entry)).or_default().push(entry);
    }
    let mut stats: Vec<GroupStats> = buckets
        .into_iter()
        .map(|(label, group)| group_stats(&label, group.into_iter()))
        .collect();
    if matches!(order, SortOrder::BySize) {
        // Descending by size, ties broken by label so output is deterministic.
        stats.sort_by(|a, b| {
            b.total()
                .cmp(&a.total())
                .then_with(|| a.label.cmp(&b.label))
        });
    }
    stats
}

fn group_stats<'a>(label: &str, entries: impl Iterator<Item = &'a CorpusEntry>) -> GroupStats {
    let mut stats = GroupStats {
        label: label.to_string(),
        ..GroupStats::default()
    };
    let mut deltas = Vec::new();
    for entry in entries {
        match entry.outcome {
            Outcome::Success => {
                stats.successes += 1;
                deltas.push(entry.record.score_delta);
            }
            Outcome::Failure => stats.failures += 1,
        }
    }
    if !deltas.is_empty() {
        stats.total_success_delta = deltas.iter().sum();
        stats.mean_success_delta = stats.total_success_delta / deltas.len() as f64;
        stats.median_success_delta = median(&mut deltas);
        stats.max_success_delta = deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    stats
}

/// Median of the slice; reorders it in place. Returns `NaN` when empty.
#[must_use]
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Pearson correlation coefficient, or `None` when there are fewer than three
/// pairs or either series has zero variance.
#[must_use]
pub fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 3 {
        return None;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denominator = (var_x * var_y).sqrt();
    if denominator <= 0.0 || !denominator.is_finite() {
        return None;
    }
    let r = covariance / denominator;
    r.is_finite().then_some(r)
}

fn scaled(value: f64, log_scale: bool) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    if !log_scale {
        return Some(value);
    }
    let magnitude = value.abs();
    (magnitude > 0.0).then(|| magnitude.log10())
}

fn predictors(entries: &[CorpusEntry]) -> Vec<Predictor> {
    PREDICTORS
        .iter()
        .map(|spec| predictor(entries, spec))
        .filter(|p| p.gain_samples > 0 || p.outcome_samples > 0)
        .collect()
}

fn predictor(entries: &[CorpusEntry], spec: &PredictorSpec) -> Predictor {
    let (mut gain_x, mut gain_y) = (Vec::new(), Vec::new());
    let (mut outcome_x, mut outcome_y) = (Vec::new(), Vec::new());
    for entry in entries {
        let Some(raw) = (spec.extract)(entry) else {
            continue;
        };
        let Some(value) = scaled(raw, spec.log_scale) else {
            continue;
        };
        outcome_x.push(value);
        outcome_y.push(f64::from(u8::from(entry.outcome == Outcome::Success)));
        if entry.outcome == Outcome::Success && entry.record.score_delta > 0.0 {
            gain_x.push(value);
            gain_y.push(entry.record.score_delta.log10());
        }
    }
    Predictor {
        label: spec.label.to_string(),
        gain_samples: gain_x.len(),
        r_vs_log_gain: pearson(&gain_x, &gain_y),
        outcome_samples: outcome_x.len(),
        r_vs_success: pearson(&outcome_x, &outcome_y),
    }
}
