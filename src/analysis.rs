use crate::parquet_format::read_records_from_parquet;
use crate::types::DiscoverRecord;
use crate::{AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson, CandidateSynapseJson};
use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc};
use wgpu::util::DeviceExt;

const EPSILON: f32 = 1e-8;
const WORKGROUP_SIZE: u32 = 256;
const MIN_NEURON_SAMPLE_COUNT: usize = 10;
const GPU_BATCH_SIZE: usize = 32; // Batch multiple GPU operations together for better utilization

#[cfg(test)]
static FORCE_GPU_ADAPTER_FAILURE: AtomicBool = AtomicBool::new(false);

pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub gpu_used: bool,
}

pub struct AnalyzeNeuronsResult {
    pub helpful_neurons: Vec<CandidateNeuronJson>,
    pub gpu_used: bool,
}

struct OrderedNeuron {
    uuid: String,
    index: usize,
}

struct RecordCache<'a> {
    parquet_file: &'a str,
    cache: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

#[derive(Clone, Copy)]
enum RejectionReason {
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectionReason::NoSamples => write!(f, "no overlapping discovery samples"),
            RejectionReason::ZeroImprovement => write!(f, "no consistent improvement in GPU stats"),
            RejectionReason::BelowThreshold => write!(f, "expected improvement below threshold"),
        }
    }
}

#[derive(Clone)]
struct RejectionDetail {
    source_uuid: String,
    reason: RejectionReason,
    sample_count: usize,
    source_record_count: usize,
    improved_count: u32,
    worsened_count: u32,
    expected_improvement: f32,
    threshold: f32,
    weight: Option<f32>,
}

impl RejectionDetail {
    fn score(&self) -> f32 {
        self.expected_improvement
    }
}

struct ThresholdContext {
    sample_count: usize,
    expected_improvement: f32,
    threshold: f32,
    improved_count: u32,
    worsened_count: u32,
    weight: f32,
}

struct TargetDiagnosticEntry {
    target_uuid: String,
    target_record_count: usize,
    evaluated_candidates: u32,
    candidates_with_samples: u32,
    had_candidate: bool,
    best_rejection: Option<RejectionDetail>,
}

impl TargetDiagnosticEntry {
    fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            evaluated_candidates: 0,
            candidates_with_samples: 0,
            had_candidate: false,
            best_rejection: None,
        }
    }

    fn update_best(&mut self, detail: RejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

struct TargetDiagnostics {
    enabled: bool,
    entries: HashMap<String, TargetDiagnosticEntry>,
}

impl TargetDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let enabled = std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok();
        let mut entries = HashMap::new();
        if enabled {
            for target in targets {
                entries.insert(target.to_string(), TargetDiagnosticEntry::new(target));
            }
        }
        Self { enabled, entries }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_candidates += 1;
            if had_samples {
                entry.candidates_with_samples += 1;
            }
        }
    }

    fn record_no_samples(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        source_record_count: usize,
    ) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::NoSamples,
                sample_count: 0,
                source_record_count,
                improved_count: 0,
                worsened_count: 0,
                expected_improvement: f32::NEG_INFINITY,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    fn record_zero_improvement(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        sample_count: usize,
        positive_count: u32,
        negative_count: u32,
    ) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::ZeroImprovement,
                sample_count,
                source_record_count: sample_count,
                improved_count: positive_count.max(negative_count),
                worsened_count: positive_count.min(negative_count),
                expected_improvement: 0.0,
                threshold: 0.0,
                weight: None,
            });
        }
    }

    fn record_below_threshold(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        context: ThresholdContext,
    ) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(RejectionDetail {
                source_uuid: source_uuid.to_string(),
                reason: RejectionReason::BelowThreshold,
                sample_count: context.sample_count,
                source_record_count: context.sample_count,
                improved_count: context.improved_count,
                worsened_count: context.worsened_count,
                expected_improvement: context.expected_improvement,
                threshold: context.threshold,
                weight: Some(context.weight),
            });
        }
    }

    fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    fn emit_logs(&self) {
        if !self.enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            if entry.evaluated_candidates == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had no eligible upstream neurons to evaluate.",
                    entry.target_uuid
                );
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} potential synapses but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_candidates
                    );
                    continue;
                }
            };

            match best.reason {
                RejectionReason::NoSamples => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no aligned samples were available (source records {}, target records {}).",
                        entry.target_uuid,
                        best.source_uuid,
                        best.source_record_count,
                        entry.target_record_count
                    );
                }
                RejectionReason::ZeroImprovement => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} saw {} aligned samples from {} but GPU stats reported zero consistent improvements (positive {}, negative {}).",
                        entry.target_uuid,
                        best.sample_count,
                        best.source_uuid,
                        best.improved_count,
                        best.worsened_count
                    );
                }
                RejectionReason::BelowThreshold => {
                    if let Some(weight) = best.weight {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}, suggested weight {:.4}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count,
                            weight
                        );
                    } else {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best candidate from {} improved {:.4} but remained below threshold {:.4} (improved {}, worsened {}).",
                            entry.target_uuid,
                            best.source_uuid,
                            best.expected_improvement,
                            best.threshold,
                            best.improved_count,
                            best.worsened_count
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn entry_for(&self, target_uuid: &str) -> Option<&TargetDiagnosticEntry> {
        self.entries.get(target_uuid)
    }
}

#[derive(Clone, Copy)]
enum NeuronRejectionReason {
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
}

#[derive(Clone)]
struct NeuronRejectionDetail {
    source_uuid: String,
    orientation: Option<&'static str>,
    reason: NeuronRejectionReason,
    sample_count: usize,
    improved_count: u32,
    worsened_count: u32,
    expected_improvement: f32,
    threshold: f32,
    outgoing_weight: Option<f32>,
}

impl NeuronRejectionDetail {
    fn score(&self) -> f32 {
        self.expected_improvement
    }
}

struct NeuronDiagnosticEntry {
    target_uuid: String,
    target_record_count: usize,
    evaluated_sources: u32,
    sources_with_samples: u32,
    had_candidate: bool,
    best_rejection: Option<NeuronRejectionDetail>,
}

impl NeuronDiagnosticEntry {
    fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            evaluated_sources: 0,
            sources_with_samples: 0,
            had_candidate: false,
            best_rejection: None,
        }
    }

    fn update_best(&mut self, detail: NeuronRejectionDetail) {
        match &self.best_rejection {
            Some(current) => {
                if detail.score() > current.score()
                    || (detail.score() == current.score()
                        && detail.sample_count > current.sample_count)
                {
                    self.best_rejection = Some(detail);
                }
            }
            None => self.best_rejection = Some(detail),
        }
    }
}

struct NeuronDiagnostics {
    enabled: bool,
    entries: HashMap<String, NeuronDiagnosticEntry>,
}

impl NeuronDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let enabled = std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok();
        let mut entries = HashMap::new();
        if enabled {
            for target in targets {
                entries.insert(target.to_string(), NeuronDiagnosticEntry::new(target));
            }
        }
        Self { enabled, entries }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_sources += 1;
            if had_samples {
                entry.sources_with_samples += 1;
            }
        }
    }

    fn record_no_samples(&mut self, target_uuid: &str, source_uuid: &str) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(NeuronRejectionDetail {
                source_uuid: source_uuid.to_string(),
                orientation: None,
                reason: NeuronRejectionReason::NoSamples,
                sample_count: 0,
                improved_count: 0,
                worsened_count: 0,
                expected_improvement: f32::NEG_INFINITY,
                threshold: 0.0,
                outgoing_weight: None,
            });
        }
    }

    fn record_rejection(
        &mut self,
        target_uuid: &str,
        source_uuid: &str,
        summary: &ReluOrientationSummary,
        threshold: f32,
    ) {
        if !self.enabled {
            return;
        }
        let reason = match summary.failure {
            Some(ReluFailure::NotEnoughSamples) => NeuronRejectionReason::NotEnoughActivations,
            Some(ReluFailure::WeightInvalid) => NeuronRejectionReason::WeightDegenerate,
            Some(ReluFailure::BelowThreshold) | None => NeuronRejectionReason::BelowThreshold,
        };
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.update_best(NeuronRejectionDetail {
                source_uuid: source_uuid.to_string(),
                orientation: Some(summary.orientation_name()),
                reason,
                sample_count: summary.sample_count,
                improved_count: summary.improved_count,
                worsened_count: summary.worsened_count,
                expected_improvement: summary.expected_improvement,
                threshold,
                outgoing_weight: summary.outgoing_weight,
            });
        }
    }

    fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    fn emit_logs(&self) {
        if !self.enabled {
            return;
        }

        for entry in self.entries.values() {
            if entry.had_candidate {
                continue;
            }

            if entry.evaluated_sources == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had no upstream neurons to analyse.",
                    entry.target_uuid
                );
                continue;
            }

            let best = match &entry.best_rejection {
                Some(detail) => detail,
                None => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} evaluated {} upstream neurons but recorded no diagnostics.",
                        entry.target_uuid, entry.evaluated_sources
                    );
                    continue;
                }
            };

            let orientation = best.orientation.unwrap_or("unknown");
            match best.reason {
                NeuronRejectionReason::NoSamples => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no overlapping samples were found.",
                        entry.target_uuid, best.source_uuid
                    );
                }
                NeuronRejectionReason::NotEnoughActivations => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} saw fewer than {} aligned samples for {} ({}) so the ReLU neuron could not be evaluated.",
                        entry.target_uuid,
                        MIN_NEURON_SAMPLE_COUNT,
                        best.source_uuid,
                        orientation
                    );
                }
                NeuronRejectionReason::WeightDegenerate => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} computed a degenerate weight for {} ({}) so the candidate was discarded (samples {}).",
                        entry.target_uuid,
                        best.source_uuid,
                        orientation,
                        best.sample_count
                    );
                }
                NeuronRejectionReason::BelowThreshold => {
                    if let Some(weight) = best.outgoing_weight {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best ReLU candidate from {} ({}) improved {:.4} but stayed below threshold {:.4} (samples {}, improved {}, worsened {}, weight {:.4}).",
                            entry.target_uuid,
                            best.source_uuid,
                            orientation,
                            best.expected_improvement,
                            best.threshold,
                            best.sample_count,
                            best.improved_count,
                            best.worsened_count,
                            weight
                        );
                    } else {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {} best ReLU candidate from {} ({}) improved {:.4} but stayed below threshold {:.4} (samples {}, improved {}, worsened {}).",
                            entry.target_uuid,
                            best.source_uuid,
                            orientation,
                            best.expected_improvement,
                            best.threshold,
                            best.sample_count,
                            best.improved_count,
                            best.worsened_count
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn entry_for(&self, target_uuid: &str) -> Option<&NeuronDiagnosticEntry> {
        self.entries.get(target_uuid)
    }
}

impl<'a> RecordCache<'a> {
    fn new(parquet_file: &'a str) -> Self {
        Self {
            parquet_file,
            cache: HashMap::new(),
        }
    }

    fn get(&mut self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        use std::collections::hash_map::Entry;

        match self.cache.entry(neuron_uuid.to_string()) {
            Entry::Occupied(entry) => Ok(Arc::clone(entry.get())),
            Entry::Vacant(entry) => {
                let mut records = read_records_from_parquet(self.parquet_file, neuron_uuid)
                    .with_context(|| {
                        format!("Failed to read discovery records for neuron {neuron_uuid}")
                    })?;
                records.sort_by_key(|record| record.obs_index);
                let arc = Arc::new(records);
                entry.insert(Arc::clone(&arc));
                Ok(arc)
            }
        }
    }
}

fn require_unique_focus<'a>(focus_neurons: &'a [String], context: &str) -> Result<Vec<&'a String>> {
    if focus_neurons.is_empty() {
        return Err(anyhow!(
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
        return Err(anyhow!(
            "{context} received duplicate focus neurons ({joined}). Each target must be unique so we can map diagnostics back to the Deno request. We are refusing to continue so the upstream behaviour can be corrected."
        ));
    }

    Ok(unique_focus)
}

#[derive(Clone, Copy)]
struct HelpfulSample {
    activation: f32,
    avg_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHelpfulSample {
    activation: f32,
    avg_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTargetRecord {
    obs_index: u32,
    avg_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFromRecord {
    obs_index: u32,
    activation: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatchingUniforms {
    target_count: u32,
    from_count: u32,
    pad0: u32,
    pad1: u32,
}

impl From<HelpfulSample> for GpuHelpfulSample {
    fn from(value: HelpfulSample) -> Self {
        Self {
            activation: value.activation,
            avg_error: value.avg_error,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulContribution {
    positive_flag: u32,
    negative_flag: u32,
    positive_improvement: f32,
    negative_improvement: f32,
    positive_activation: f32,
    negative_activation: f32,
    pad0: f32,
    pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HelpfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulContribution {
    harmful_flag: u32,
    helpful_flag: u32,
    error_magnitude: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HarmfulUniforms {
    length: u32,
    pad0: u32,
    epsilon: f32,
    weight: f32,
}

#[derive(Default)]
struct HelpfulStats {
    positive_count: u32,
    negative_count: u32,
    positive_improvement_sum: f32,
    negative_improvement_sum: f32,
    positive_activation_sum: f32,
    negative_activation_sum: f32,
}

#[derive(Clone, Copy)]
enum ReluOrientation {
    Positive,
    Negative,
}

struct ReluStats {
    orientation: ReluOrientation,
    samples: Vec<(f32, f32)>,
    activation_sq_sum: f32,
    error_activation_sum: f32,
}

impl ReluStats {
    fn new(orientation: ReluOrientation) -> Self {
        Self {
            orientation,
            samples: Vec::new(),
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
        }
    }

    fn push(&mut self, relu_activation: f32, error: f32) {
        self.samples.push((relu_activation, error));
        self.activation_sq_sum += relu_activation * relu_activation;
        self.error_activation_sum += relu_activation * error;
    }

    fn evaluate(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        threshold: f32,
    ) -> ReluOrientationEvaluation {
        let sample_count = self.samples.len();
        if sample_count < MIN_NEURON_SAMPLE_COUNT || self.activation_sq_sum <= EPSILON {
            return ReluOrientationEvaluation {
                summary: ReluOrientationSummary::insufficient(self.orientation, sample_count),
                candidate: None,
            };
        }

        let mut outgoing_weight = self.error_activation_sum / (self.activation_sq_sum + EPSILON);
        if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
            return ReluOrientationEvaluation {
                summary: ReluOrientationSummary::degenerate(self.orientation, sample_count),
                candidate: None,
            };
        }
        outgoing_weight = outgoing_weight.clamp(-5.0, 5.0);

        let mut improved_count = 0u32;
        let mut worsened_count = 0u32;
        for (relu_activation, error) in &self.samples {
            let new_error = error - outgoing_weight * relu_activation;
            if new_error.abs() + EPSILON < error.abs() {
                improved_count += 1;
            } else if new_error.abs() > error.abs() + EPSILON {
                worsened_count += 1;
            }
        }

        let total_count = self.samples.len() as u32;
        debug_assert!(total_count > 0);

        let expected_improvement =
            (improved_count as f32 - worsened_count as f32) / total_count as f32;
        if expected_improvement <= threshold {
            return ReluOrientationEvaluation {
                summary: ReluOrientationSummary::below_threshold(
                    self.orientation,
                    sample_count,
                    improved_count,
                    worsened_count,
                    expected_improvement,
                    outgoing_weight,
                ),
                candidate: None,
            };
        }

        let incoming_weight = match self.orientation {
            ReluOrientation::Positive => 1.0,
            ReluOrientation::Negative => -1.0,
        };

        ReluOrientationEvaluation {
            summary: ReluOrientationSummary::successful(
                self.orientation,
                sample_count,
                improved_count,
                worsened_count,
                expected_improvement,
                outgoing_weight,
            ),
            candidate: Some(CandidateNeuronJson {
                source_neuron_uuid: source_uuid.to_string(),
                target_neuron_uuid: target_uuid.to_string(),
                incoming_weight,
                outgoing_weight,
                squash: "ReLU".to_string(),
                bias: 0.0,
                expected_improvement_percentage: expected_improvement,
                improved_count,
                total_count,
            }),
        }
    }
}

#[derive(Clone, Copy)]
enum ReluFailure {
    NotEnoughSamples,
    WeightInvalid,
    BelowThreshold,
}

#[derive(Clone)]
struct ReluOrientationSummary {
    orientation: ReluOrientation,
    sample_count: usize,
    improved_count: u32,
    worsened_count: u32,
    expected_improvement: f32,
    outgoing_weight: Option<f32>,
    failure: Option<ReluFailure>,
}

impl ReluOrientationSummary {
    fn insufficient(orientation: ReluOrientation, sample_count: usize) -> Self {
        Self {
            orientation,
            sample_count,
            improved_count: 0,
            worsened_count: 0,
            expected_improvement: f32::NEG_INFINITY,
            outgoing_weight: None,
            failure: Some(ReluFailure::NotEnoughSamples),
        }
    }

    fn degenerate(orientation: ReluOrientation, sample_count: usize) -> Self {
        Self {
            orientation,
            sample_count,
            improved_count: 0,
            worsened_count: 0,
            expected_improvement: f32::NEG_INFINITY,
            outgoing_weight: None,
            failure: Some(ReluFailure::WeightInvalid),
        }
    }

    fn below_threshold(
        orientation: ReluOrientation,
        sample_count: usize,
        improved_count: u32,
        worsened_count: u32,
        expected_improvement: f32,
        outgoing_weight: f32,
    ) -> Self {
        Self {
            orientation,
            sample_count,
            improved_count,
            worsened_count,
            expected_improvement,
            outgoing_weight: Some(outgoing_weight),
            failure: Some(ReluFailure::BelowThreshold),
        }
    }

    fn successful(
        orientation: ReluOrientation,
        sample_count: usize,
        improved_count: u32,
        worsened_count: u32,
        expected_improvement: f32,
        outgoing_weight: f32,
    ) -> Self {
        Self {
            orientation,
            sample_count,
            improved_count,
            worsened_count,
            expected_improvement,
            outgoing_weight: Some(outgoing_weight),
            failure: None,
        }
    }

    fn orientation_name(&self) -> &'static str {
        match self.orientation {
            ReluOrientation::Positive => "positive",
            ReluOrientation::Negative => "negative",
        }
    }
}

struct ReluOrientationEvaluation {
    summary: ReluOrientationSummary,
    candidate: Option<CandidateNeuronJson>,
}

struct ReluEvaluationResult {
    candidate: Option<CandidateNeuronJson>,
    best_summary: Option<ReluOrientationSummary>,
}

struct ActivationCandidateSpec {
    name: &'static str,
    orientations: &'static [f32],
    scales: &'static [f32],
    activation: fn(f32) -> f32,
    min_improvement: f32,
}

const ORIENTATIONS_BIDIRECTIONAL: [f32; 2] = [1.0, -1.0];
const SCALES_WIDE: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const SCALES_SMOOTH: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.797_884_6 * (x + 0.044_715 * x_cubed);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn elu_activation(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

fn selu_activation(x: f32) -> f32 {
    const SELU_ALPHA: f32 = 1.673_263_2;
    const SELU_LAMBDA: f32 = 1.050_701;
    if x >= 0.0 {
        SELU_LAMBDA * x
    } else {
        SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
    }
}

fn softplus_activation(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logistic_activation(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

const ACTIVATION_SPECS: [ActivationCandidateSpec; 6] = [
    ActivationCandidateSpec {
        name: "GELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: gelu_activation,
        min_improvement: 0.08,
    },
    ActivationCandidateSpec {
        name: "ELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: elu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "SELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: selu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "Softplus",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: softplus_activation,
        min_improvement: 0.07,
    },
    ActivationCandidateSpec {
        name: "LOGISTIC",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: logistic_activation,
        min_improvement: 0.05,
    },
    ActivationCandidateSpec {
        name: "TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: tanh_activation,
        min_improvement: 0.0,
    },
];

#[derive(Default)]
struct HarmfulStats {
    harmful_count: u32,
    helpful_count: u32,
    harmful_error_sum: f32,
}

fn cpu_helpful_stats(samples: &[HelpfulSample]) -> HelpfulStats {
    let mut stats = HelpfulStats::default();

    for sample in samples {
        if sample.activation.abs() <= EPSILON || sample.avg_error.abs() <= EPSILON {
            continue;
        }

        let required_sign = -sample.avg_error.signum() * sample.activation.signum();
        let improvement = sample.avg_error.abs();
        let activation_mag = sample.activation.abs();

        if required_sign > 0.0 {
            stats.positive_count += 1;
            stats.positive_improvement_sum += improvement;
            stats.positive_activation_sum += activation_mag;
        } else if required_sign < 0.0 {
            stats.negative_count += 1;
            stats.negative_improvement_sum += improvement;
            stats.negative_activation_sum += activation_mag;
        }
    }

    stats
}

fn cpu_harmful_stats(samples: &[HelpfulSample], weight: f32) -> HarmfulStats {
    let mut stats = HarmfulStats::default();

    for sample in samples {
        if sample.activation.abs() <= EPSILON || sample.avg_error.abs() <= EPSILON {
            continue;
        }
        let signal = sample.activation * weight;
        let signal_sign = signal.signum();
        let error_sign = sample.avg_error.signum();

        if signal_sign == 0.0 || error_sign == 0.0 {
            continue;
        }

        if signal_sign == error_sign {
            stats.harmful_count += 1;
            stats.harmful_error_sum += sample.avg_error.abs();
        } else {
            stats.helpful_count += 1;
        }
    }

    stats
}

struct GpuAnalyzer {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    helpful_layout: Option<wgpu::BindGroupLayout>,
    helpful_pipeline: Option<wgpu::ComputePipeline>,
    harmful_layout: Option<wgpu::BindGroupLayout>,
    harmful_pipeline: Option<wgpu::ComputePipeline>,
    matching_layout: Option<wgpu::BindGroupLayout>,
    matching_pipeline: Option<wgpu::ComputePipeline>,
    gpu_used: bool,
}

impl GpuAnalyzer {
    fn cpu_fallback() -> Self {
        Self {
            device: None,
            queue: None,
            helpful_layout: None,
            helpful_pipeline: None,
            harmful_layout: None,
            harmful_pipeline: None,
            matching_layout: None,
            matching_pipeline: None,
            gpu_used: false,
        }
    }

    fn new(require_gpu: bool) -> Result<Self> {
        let instance = wgpu::Instance::default();
        #[cfg(test)]
        let adapter = if FORCE_GPU_ADAPTER_FAILURE.load(AtomicOrdering::SeqCst) {
            None
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        };

        #[cfg(not(test))]
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(adapter) => {
                #[cfg(not(test))]
                {
                    // Log adapter info for diagnostics (only in non-test builds)
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU adapter found: {:?}",
                            adapter.get_info()
                        );
                    }
                }
                adapter
            }
            None => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] No GPU adapter available, {}",
                            if require_gpu {
                                "failing (GPU required)"
                            } else {
                                "falling back to CPU"
                            }
                        );
                    }
                }
                if require_gpu {
                    return Err(anyhow!("No GPU adapter available for discovery analysis"));
                }
                return Ok(Self::cpu_fallback());
            }
        };

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )) {
            Ok(result) => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU device initialised successfully: {:?}",
                            result.0.features()
                        );
                    }
                }
                result
            }
            Err(err) => {
                #[cfg(not(test))]
                {
                    if std::env::var("NEAT_AI_DISCOVERY_GPU_DEBUG").is_ok() {
                        eprintln!(
                            "[NEAT-AI-Discovery] GPU device initialisation failed: {err}, {}",
                            if require_gpu {
                                "failing (GPU required)"
                            } else {
                                "falling back to CPU"
                            }
                        );
                    }
                }
                if require_gpu {
                    return Err(anyhow!(
                        "Failed to initialise GPU device for discovery analysis: {err}"
                    ));
                }
                return Ok(Self::cpu_fallback());
            }
        };

        let (helpful_layout, helpful_pipeline) =
            Self::build_helpful_pipeline(&device, "helpful-synapse-pipeline");
        let (harmful_layout, harmful_pipeline) =
            Self::build_harmful_pipeline(&device, "harmful-synapse-pipeline");
        let (matching_layout, matching_pipeline) =
            Self::build_matching_pipeline(&device, "matching-pipeline");

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            matching_layout: Some(matching_layout),
            matching_pipeline: Some(matching_pipeline),
            gpu_used: true,
        })
    }

    fn build_helpful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_harmful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_matching_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matching-shader"),
            source: wgpu::ShaderSource::Wgsl(MATCHING_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matching-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn evaluate_helpful(&self, samples: &[HelpfulSample]) -> Result<HelpfulStats> {
        if samples.is_empty() {
            return Ok(HelpfulStats::default());
        }

        if !self.gpu_used {
            return Ok(cpu_helpful_stats(samples));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for helpful analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for helpful analysis")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("helpful-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("helpful-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = HelpfulUniforms {
            length: samples.len() as u32,
            pad0: 0,
            epsilon: EPSILON,
            pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("helpful-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: helpful_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contributions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("helpful-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helpful-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("helpful-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("helpful-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(helpful_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &contributions_buffer,
            0,
            &staging_buffer,
            0,
            contribution_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });
        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(anyhow!("Failed to map helpful contributions buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive helpful map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

        let mut stats = HelpfulStats::default();
        for contribution in contributions {
            stats.positive_count += contribution.positive_flag;
            stats.negative_count += contribution.negative_flag;
            stats.positive_improvement_sum += contribution.positive_improvement;
            stats.negative_improvement_sum += contribution.negative_improvement;
            stats.positive_activation_sum += contribution.positive_activation;
            stats.negative_activation_sum += contribution.negative_activation;
        }

        drop(data);
        staging_buffer.unmap();

        stats = Self::fallback_helpful_stats(stats, samples);

        Ok(stats)
    }

    fn evaluate_harmful(&self, samples: &[HelpfulSample], weight: f32) -> Result<HarmfulStats> {
        if samples.is_empty() {
            return Ok(HarmfulStats::default());
        }

        if !self.gpu_used {
            return Ok(cpu_harmful_stats(samples, weight));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for harmful analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for harmful analysis")?;
        let harmful_layout = self
            .harmful_layout
            .as_ref()
            .context("GPU layout not initialised for harmful analysis")?;
        let harmful_pipeline = self
            .harmful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for harmful analysis")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![HarmfulContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = HarmfulUniforms {
            length: samples.len() as u32,
            pad0: 0,
            epsilon: EPSILON,
            weight,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("harmful-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: harmful_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contributions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("harmful-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<HarmfulContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("harmful-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("harmful-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("harmful-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(harmful_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &contributions_buffer,
            0,
            &staging_buffer,
            0,
            contribution_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });
        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(anyhow!("Failed to map harmful contributions buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive harmful map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let contributions: &[HarmfulContribution] = bytemuck::cast_slice(&data);

        let mut stats = HarmfulStats::default();
        for contribution in contributions {
            stats.harmful_count += contribution.harmful_flag;
            stats.helpful_count += contribution.helpful_flag;
            stats.harmful_error_sum += contribution.error_magnitude;
        }

        drop(data);
        staging_buffer.unmap();

        if stats.harmful_count <= stats.helpful_count {
            let cpu_stats = cpu_harmful_stats(samples, weight);
            if cpu_stats.harmful_count > cpu_stats.helpful_count {
                stats = cpu_stats;
            }
        }

        Ok(stats)
    }

    fn fallback_helpful_stats(mut stats: HelpfulStats, samples: &[HelpfulSample]) -> HelpfulStats {
        if (stats.positive_count == 0 && stats.negative_count == 0) && !samples.is_empty() {
            let cpu_stats = cpu_helpful_stats(samples);
            if cpu_stats.positive_count > 0 || cpu_stats.negative_count > 0 {
                stats = cpu_stats;
            }
        }

        stats
    }

    fn gpu_used(&self) -> bool {
        self.gpu_used
    }

    /// Batch evaluate multiple helpful operations to improve GPU utilization
    /// Returns a vector of stats in the same order as the input samples
    fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
    ) -> Result<Vec<HelpfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        if !self.gpu_used {
            // CPU fallback - process sequentially
            return Ok(samples_batch
                .iter()
                .map(|samples| cpu_helpful_stats(samples))
                .collect());
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for batched helpful analysis")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for batched helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched helpful analysis")?;

        // Process in batches to avoid excessive memory usage
        let mut all_results = Vec::with_capacity(samples_batch.len());

        for batch_chunk in samples_batch.chunks(GPU_BATCH_SIZE) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_encoders = Vec::new();
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_sample_refs = Vec::new();

            // Prepare all operations in this batch
            for samples in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    continue;
                }
                empty_flags.push(false);

                batch_sample_refs.push(*samples);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("helpful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HelpfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    pad1: 0.0,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: helpful_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sample_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: contributions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("helpful-bind-group-batch"),
                });

                let contribution_size =
                    (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("helpful-staging-buffer-batch"),
                    size: contribution_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("helpful-command-encoder-batch"),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("helpful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(helpful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                encoder.copy_buffer_to_buffer(
                    &contributions_buffer,
                    0,
                    &staging_buffer,
                    0,
                    contribution_size,
                );

                batch_encoders.push(encoder);
                batch_staging_buffers.push(staging_buffer);
                batch_contribution_sizes.push((contribution_size, samples.len()));
            }

            // Submit all operations in this batch at once
            if !batch_encoders.is_empty() {
                let command_buffers: Vec<_> =
                    batch_encoders.into_iter().map(|e| e.finish()).collect();
                queue.submit(command_buffers);
            }

            // Wait for all results (single poll for entire batch)
            let mut batch_results = Vec::with_capacity(batch_contribution_sizes.len());
            for (staging_buffer, (_contribution_size, _sample_len), samples_ref) in
                batch_staging_buffers
                    .into_iter()
                    .zip(batch_contribution_sizes)
                    .zip(batch_sample_refs)
                    .map(|((buffer, size), samples)| (buffer, size, samples))
            {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });

                // Poll until this specific buffer is ready
                loop {
                    device.poll(wgpu::Maintain::Poll);
                    match receiver.try_recv() {
                        Ok(Ok(())) => break,
                        Ok(Err(err)) => {
                            return Err(anyhow!(
                                "Failed to map helpful contributions buffer: {err}"
                            ));
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            // Continue polling
                            continue;
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            return Err(anyhow!("Failed to receive helpful map_async completion"));
                        }
                    }
                }

                let data = buffer_slice.get_mapped_range();
                let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

                let mut stats = HelpfulStats::default();
                for contribution in contributions {
                    stats.positive_count += contribution.positive_flag;
                    stats.negative_count += contribution.negative_flag;
                    stats.positive_improvement_sum += contribution.positive_improvement;
                    stats.negative_improvement_sum += contribution.negative_improvement;
                    stats.positive_activation_sum += contribution.positive_activation;
                    stats.negative_activation_sum += contribution.negative_activation;
                }

                drop(data);
                staging_buffer.unmap();

                // Fallback check
                stats = Self::fallback_helpful_stats(stats, samples_ref);

                batch_results.push(stats);
            }

            let merged_results = Self::merge_batch_results(&empty_flags, batch_results);
            all_results.extend(merged_results);
        }

        Ok(all_results)
    }

    fn merge_batch_results(
        empty_flags: &[bool],
        computed_stats: Vec<HelpfulStats>,
    ) -> Vec<HelpfulStats> {
        let expected_non_empty = empty_flags.iter().filter(|flag| !**flag).count();
        debug_assert_eq!(
            expected_non_empty,
            computed_stats.len(),
            "Computed stats should match number of non-empty sample sets"
        );

        let mut results = Vec::with_capacity(empty_flags.len());
        let mut stats_iter = computed_stats.into_iter();

        for &is_empty in empty_flags {
            if is_empty {
                results.push(HelpfulStats::default());
            } else if let Some(stats) = stats_iter.next() {
                results.push(stats);
            } else {
                // Safety guard: if counts mismatch, preserve ordering by inserting default.
                results.push(HelpfulStats::default());
            }
        }

        results
    }

    /// GPU-accelerated matching of activations to errors by obs_index
    /// This replaces the CPU-based build_samples function for better GPU utilization
    fn build_samples_gpu(
        &self,
        target_records: &[DiscoverRecord],
        from_records: &[DiscoverRecord],
    ) -> Result<Vec<HelpfulSample>> {
        if target_records.is_empty() || from_records.is_empty() {
            return Ok(Vec::new());
        }

        if !self.gpu_used {
            // CPU fallback
            return Ok(build_samples(target_records, from_records));
        }

        let device = self
            .device
            .as_ref()
            .context("GPU device not initialised for sample matching")?;
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for sample matching")?;
        let matching_layout = self
            .matching_layout
            .as_ref()
            .context("GPU matching layout not initialised")?;
        let matching_pipeline = self
            .matching_pipeline
            .as_ref()
            .context("GPU matching pipeline not initialised")?;

        // Prepare target records: compute avg_error and create GPU structures
        let mut gpu_targets: Vec<GpuTargetRecord> = Vec::new();
        for record in target_records {
            if record.errors.is_empty() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0;
            for error in &record.errors {
                if error.is_finite() {
                    sum += *error;
                    count += 1;
                }
            }
            if count > 0 {
                gpu_targets.push(GpuTargetRecord {
                    obs_index: record.obs_index,
                    avg_error: sum / count as f32,
                });
            }
        }

        if gpu_targets.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by obs_index for binary search (should already be sorted, but ensure it)
        gpu_targets.sort_by_key(|r| r.obs_index);

        // Prepare from records
        let gpu_froms: Vec<GpuFromRecord> = from_records
            .iter()
            .map(|r| GpuFromRecord {
                obs_index: r.obs_index,
                activation: r.activation,
            })
            .collect();

        // Create GPU buffers
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-target-buffer"),
            contents: bytemuck::cast_slice(&gpu_targets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let from_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-from-buffer"),
            contents: bytemuck::cast_slice(&gpu_froms),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let samples_zeroed = vec![GpuHelpfulSample::zeroed(); from_records.len()];
        let samples_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-samples-buffer"),
            contents: bytemuck::cast_slice(&samples_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = MatchingUniforms {
            target_count: gpu_targets.len() as u32,
            from_count: from_records.len() as u32,
            pad0: 0,
            pad1: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: matching_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: from_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("matching-bind-group"),
        });

        let sample_size = (std::mem::size_of::<GpuHelpfulSample>() * from_records.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matching-staging-buffer"),
            size: sample_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matching-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matching-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(matching_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (from_records.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&samples_buffer, 0, &staging_buffer, 0, sample_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });
        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(anyhow!("Failed to map matching samples buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive matching map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let gpu_samples: &[GpuHelpfulSample] = bytemuck::cast_slice(&data);

        // Retain only finite samples; GPU matching emits NaN for invalid rows
        let mut samples = Vec::new();
        for gpu_sample in gpu_samples {
            if gpu_sample.activation.is_finite() && gpu_sample.avg_error.is_finite() {
                samples.push(HelpfulSample {
                    activation: gpu_sample.activation,
                    avg_error: gpu_sample.avg_error,
                });
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(samples)
    }
}

const HELPFUL_SHADER: &str = include_str!("shaders/helpful.wgsl");

const HARMFUL_SHADER: &str = include_str!("shaders/harmful.wgsl");

const MATCHING_SHADER: &str = include_str!("shaders/matching.wgsl");

fn build_ordered_neurons(creature: &crate::CreatureJson) -> Vec<OrderedNeuron> {
    let mut ordered = Vec::with_capacity(creature.input + creature.neurons.len());

    for input_index in 0..creature.input {
        ordered.push(OrderedNeuron {
            uuid: format!("input-{input_index}"),
            index: input_index,
        });
    }

    for (offset, neuron) in creature.neurons.iter().enumerate() {
        ordered.push(OrderedNeuron {
            uuid: neuron.uuid.clone(),
            index: creature.input + offset,
        });
    }

    ordered
}

fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    let mut error_map: HashMap<u32, f32> = HashMap::with_capacity(target_records.len());
    for record in target_records {
        if record.errors.is_empty() {
            continue;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for error in &record.errors {
            if error.is_finite() {
                sum += *error;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        error_map.insert(record.obs_index, sum / count as f32);
    }

    if error_map.is_empty() {
        return Vec::new();
    }

    let mut samples = Vec::new();
    for record in from_records {
        if let Some(avg_error) = error_map.get(&record.obs_index) {
            if record.activation.is_finite() && avg_error.is_finite() {
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error: *avg_error,
                });
            }
        }
    }

    samples
}

fn upsert_candidate(
    map: &mut HashMap<(String, String, String), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
    );
    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            if candidate.expected_improvement_percentage
                > entry.get().expected_improvement_percentage
            {
                entry.insert(candidate);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

fn evaluate_relu_candidate(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
) -> Result<ReluEvaluationResult> {
    if samples.is_empty() {
        return Ok(ReluEvaluationResult {
            candidate: None,
            best_summary: None,
        });
    }

    // Trigger the helpful analysis pipeline so we honour GPU requirements, even
    // though the detailed ReLU statistics are still evaluated on the CPU.
    let _ = analyzer.evaluate_helpful(samples)?;

    let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
    let mut negative_stats = ReluStats::new(ReluOrientation::Negative);

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        let activation = sample.activation;
        let error = sample.avg_error;

        let relu_positive = activation.max(0.0);
        if relu_positive > EPSILON {
            positive_stats.push(relu_positive, error);
        }

        let relu_negative = (-activation).max(0.0);
        if relu_negative > EPSILON {
            negative_stats.push(relu_negative, error);
        }
    }

    let positive_eval = positive_stats.evaluate(source_uuid, target_uuid, threshold);
    let negative_eval = negative_stats.evaluate(source_uuid, target_uuid, threshold);
    let evaluations = [positive_eval, negative_eval];

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_candidate_score = f32::NEG_INFINITY;
    let mut best_summary: Option<ReluOrientationSummary> = None;
    let mut best_summary_score = f32::NEG_INFINITY;

    for eval in evaluations.into_iter() {
        if best_summary.is_none() || eval.summary.expected_improvement > best_summary_score {
            best_summary_score = eval.summary.expected_improvement;
            best_summary = Some(eval.summary.clone());
        }

        if let Some(candidate) = eval.candidate {
            if best_candidate.is_none() || eval.summary.expected_improvement > best_candidate_score
            {
                best_candidate_score = eval.summary.expected_improvement;
                best_candidate = Some(candidate);
            }
        }
    }

    Ok(ReluEvaluationResult {
        candidate: best_candidate,
        best_summary,
    })
}

fn evaluate_activation_candidate(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    spec: &ActivationCandidateSpec,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    // Trigger GPU path if required.
    let _ = analyzer.evaluate_helpful(samples)?;

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = threshold;
    let mut fallback_candidate: Option<CandidateNeuronJson> = None;
    let mut fallback_score = f32::MIN;
    let mut outputs = Vec::with_capacity(samples.len());

    for &orientation in spec.orientations {
        for &scale in spec.scales {
            outputs.clear();
            let incoming_weight = orientation * scale;
            let mut sum_activation_sq = 0.0;
            let mut sum_error_activation = 0.0;
            let mut valid = true;

            for sample in samples {
                let pre_activation = incoming_weight * sample.activation;
                let output = (spec.activation)(pre_activation);
                if !output.is_finite() {
                    valid = false;
                    break;
                }
                outputs.push(output);
                sum_activation_sq += output * output;
                sum_error_activation += output * sample.avg_error;
            }

            if !valid || outputs.len() != samples.len() || sum_activation_sq <= EPSILON {
                continue;
            }

            let mut outgoing_weight = sum_error_activation / (sum_activation_sq + EPSILON);
            if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
                continue;
            }
            outgoing_weight = outgoing_weight.clamp(-5.0, 5.0);

            let mut improved_count = 0u32;
            let mut worsened_count = 0u32;
            for (sample, output) in samples.iter().zip(outputs.iter()) {
                let new_error = sample.avg_error - outgoing_weight * output;
                if new_error.abs() + EPSILON < sample.avg_error.abs() {
                    improved_count += 1;
                } else if new_error.abs() > sample.avg_error.abs() + EPSILON {
                    worsened_count += 1;
                }
            }

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            let expected_improvement_percentage =
                (improved_count as f32 - worsened_count as f32) / total_count as f32;

            if expected_improvement_percentage > fallback_score {
                fallback_score = expected_improvement_percentage;
                fallback_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: 0.0,
                    expected_improvement_percentage,
                    improved_count,
                    total_count,
                });
            }

            let improvement_cutoff = threshold.min(spec.min_improvement);

            if expected_improvement_percentage <= improvement_cutoff
                || improved_count < MIN_NEURON_SAMPLE_COUNT as u32
            {
                continue;
            }

            if let Some(candidate) = &fallback_candidate {
                if expected_improvement_percentage > best_score {
                    best_score = expected_improvement_percentage;
                    best_candidate = Some(CandidateNeuronJson {
                        source_neuron_uuid: candidate.source_neuron_uuid.clone(),
                        target_neuron_uuid: candidate.target_neuron_uuid.clone(),
                        incoming_weight: candidate.incoming_weight,
                        outgoing_weight: candidate.outgoing_weight,
                        squash: candidate.squash.clone(),
                        bias: candidate.bias,
                        expected_improvement_percentage: candidate.expected_improvement_percentage,
                        improved_count: candidate.improved_count,
                        total_count: candidate.total_count,
                    });
                }
            }
        }
    }

    Ok(best_candidate.or(fallback_candidate))
}

pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    let require_gpu = input.require_gpu.unwrap_or(cfg!(target_os = "macos"));
    let analyzer = GpuAnalyzer::new(require_gpu)?;

    let threshold = input.improvement_threshold.unwrap_or(0.1);
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let mut order_map: HashMap<&str, usize> = HashMap::new();
    for neuron in &ordered_neurons {
        order_map.insert(neuron.uuid.as_str(), neuron.index);
    }

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    let mut cache = RecordCache::new(&input.parquet_file);
    let mut helpful_map: HashMap<(String, String, String), CandidateNeuronJson> = HashMap::new();

    let mut diagnostics = NeuronDiagnostics::new(&unique_focus);

    for target_uuid in unique_focus {
        let target_records_arc = match cache.get(target_uuid) {
            Ok(records) => records,
            Err(err) => {
                if cfg!(debug_assertions) {
                    eprintln!("Failed to load target neuron records for {target_uuid}: {err}");
                }
                continue;
            }
        };
        if target_records_arc.is_empty() {
            diagnostics.set_target_record_count(target_uuid, 0);
            continue;
        }
        let target_records = target_records_arc.as_ref();
        diagnostics.set_target_record_count(target_uuid, target_records.len());

        let target_index = match order_map.get(target_uuid.as_str()) {
            Some(index) => *index,
            None => continue,
        };

        for source in ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
        {
            let source_uuid = source.uuid.as_str();
            let from_records_arc = match cache.get(source_uuid) {
                Ok(records) => records,
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("Failed to load source neuron records for {source_uuid}: {err}");
                    }
                    continue;
                }
            };
            if from_records_arc.is_empty() {
                diagnostics.record_candidate_attempt(target_uuid, false);
                diagnostics.record_no_samples(target_uuid, source_uuid);
                continue;
            }
            let from_records = from_records_arc.as_ref();

            let samples = analyzer.build_samples_gpu(target_records, from_records)?;
            diagnostics.record_candidate_attempt(target_uuid, !samples.is_empty());
            if samples.is_empty() {
                diagnostics.record_no_samples(target_uuid, source_uuid);
                continue;
            }

            let relu_result =
                evaluate_relu_candidate(&analyzer, source_uuid, target_uuid, &samples, threshold)?;

            if let Some(candidate) = relu_result.candidate {
                diagnostics.mark_candidate_selected(target_uuid);
                upsert_candidate(&mut helpful_map, candidate);
            } else if let Some(summary) = relu_result.best_summary.as_ref() {
                diagnostics.record_rejection(target_uuid, source_uuid, summary, threshold);
            }

            for spec in ACTIVATION_SPECS.iter() {
                if let Some(candidate) = evaluate_activation_candidate(
                    &analyzer,
                    source_uuid,
                    target_uuid,
                    &samples,
                    threshold,
                    spec,
                )? {
                    diagnostics.mark_candidate_selected(target_uuid);
                    upsert_candidate(&mut helpful_map, candidate);
                }
            }
        }
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();
    helpful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
    }

    diagnostics.emit_logs();

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used: analyzer.gpu_used(),
    })
}

pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    let require_gpu = input.require_gpu.unwrap_or(cfg!(target_os = "macos"));
    let analyzer = GpuAnalyzer::new(require_gpu)?;

    let ordered_neurons = build_ordered_neurons(&input.creature);
    let mut order_map: HashMap<&str, usize> = HashMap::new();
    for neuron in &ordered_neurons {
        order_map.insert(neuron.uuid.as_str(), neuron.index);
    }

    let existing_synapses: HashSet<(String, String)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.from_uuid.clone(), synapse.to_uuid.clone()))
        .collect();

    let mut synapses_by_target: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    for synapse in &input.creature.synapses {
        synapses_by_target
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;

    let mut cache = RecordCache::new(&input.parquet_file);
    let mut diagnostics = TargetDiagnostics::new(&unique_focus);

    let mut helpful_results: Vec<CandidateSynapseJson> = Vec::new();
    let mut harmful_results: Vec<CandidateSynapseJson> = Vec::new();
    let mut helpful_fallback: Option<CandidateSynapseJson> = None;

    let threshold = input.improvement_threshold.unwrap_or(0.1);

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
    }

    let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();

    for target_uuid in &unique_focus {
        let target_records_arc = cache.get(target_uuid)?;
        if target_records_arc.is_empty() {
            diagnostics.set_target_record_count(target_uuid, 0);
            continue;
        }
        let target_records = target_records_arc.as_ref();
        diagnostics.set_target_record_count(target_uuid, target_records.len());

        let target_index = match order_map.get(target_uuid.as_str()) {
            Some(index) => *index,
            None => continue,
        };

        // Collect helpful candidates for batching
        for source in ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
        {
            let source_uuid = source.uuid.as_str();

            if existing_synapses.contains(&(source_uuid.to_string(), target_uuid.to_string())) {
                continue;
            }

            let from_records_arc = cache.get(source_uuid)?;
            if from_records_arc.is_empty() {
                diagnostics.record_candidate_attempt(target_uuid, false);
                diagnostics.record_no_samples(target_uuid, source_uuid, 0);
                continue;
            }
            let from_records = from_records_arc.as_ref();

            let samples = analyzer.build_samples_gpu(target_records, from_records)?;
            diagnostics.record_candidate_attempt(target_uuid, !samples.is_empty());
            if samples.is_empty() {
                diagnostics.record_no_samples(target_uuid, source_uuid, from_records.len());
                continue;
            }

            helpful_work_batch.push(HelpfulWork {
                source_uuid: source_uuid.to_string(),
                target_uuid: target_uuid.to_string(),
                samples,
            });
        }
    }

    // Process helpful work in batches for better GPU utilization
    let helpful_samples_refs: Vec<&[HelpfulSample]> = helpful_work_batch
        .iter()
        .map(|w| w.samples.as_slice())
        .collect();
    let helpful_stats_batch = analyzer.evaluate_helpful_batch(&helpful_samples_refs)?;

    // Process results
    for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
        let positive_is_better = stats.positive_count >= stats.negative_count;
        let improved_count = if positive_is_better {
            stats.positive_count
        } else {
            stats.negative_count
        };
        if improved_count == 0 {
            diagnostics.record_zero_improvement(
                &work.target_uuid,
                &work.source_uuid,
                work.samples.len(),
                stats.positive_count,
                stats.negative_count,
            );
            continue;
        }

        let worsen_count = if positive_is_better {
            stats.negative_count
        } else {
            stats.positive_count
        };
        let total_count = work.samples.len() as u32;
        if total_count == 0 {
            continue;
        }

        let improvement_sum = if positive_is_better {
            stats.positive_improvement_sum
        } else {
            stats.negative_improvement_sum
        };

        let activation_sum = if positive_is_better {
            stats.positive_activation_sum
        } else {
            stats.negative_activation_sum
        };

        let mut weight = 0.0;
        if activation_sum.abs() > EPSILON {
            let raw_weight = improvement_sum / (activation_sum + 1e-8);
            weight = if positive_is_better {
                -raw_weight
            } else {
                raw_weight
            };
            weight = weight.clamp(-1.0, 1.0);
        }

        let expected_improvement_percentage =
            (improved_count as f32 - worsen_count as f32) / total_count as f32;

        if expected_improvement_percentage <= threshold {
            diagnostics.record_below_threshold(
                &work.target_uuid,
                &work.source_uuid,
                ThresholdContext {
                    sample_count: work.samples.len(),
                    expected_improvement: expected_improvement_percentage,
                    threshold,
                    improved_count,
                    worsened_count: worsen_count,
                    weight,
                },
            );
            if helpful_fallback.as_ref().is_none_or(|existing| {
                existing.expected_improvement_percentage < expected_improvement_percentage
            }) {
                helpful_fallback = Some(CandidateSynapseJson {
                    from_neuron_uuid: work.source_uuid.clone(),
                    to_neuron_uuid: work.target_uuid.clone(),
                    weight,
                    expected_improvement_percentage,
                    improved_count,
                    total_count,
                });
            }
            continue;
        }

        diagnostics.mark_candidate_selected(&work.target_uuid);
        helpful_results.push(CandidateSynapseJson {
            from_neuron_uuid: work.source_uuid.clone(),
            to_neuron_uuid: work.target_uuid.clone(),
            weight,
            expected_improvement_percentage,
            improved_count,
            total_count,
        });
    }

    // Harmful synapses (existing connections) - process per target
    for target_uuid in &unique_focus {
        let target_records_arc = cache.get(target_uuid)?;
        if target_records_arc.is_empty() {
            continue;
        }
        let target_records = target_records_arc.as_ref();

        if let Some(existing) = synapses_by_target.get(target_uuid.as_str()) {
            for synapse in existing {
                let from_records_arc = cache.get(synapse.from_uuid.as_str())?;
                if from_records_arc.is_empty() {
                    continue;
                }
                let from_records = from_records_arc.as_ref();
                let samples = analyzer.build_samples_gpu(target_records, from_records)?;
                if samples.is_empty() {
                    continue;
                }

                let stats = analyzer.evaluate_harmful(&samples, synapse.weight)?;
                let total_count = samples.len() as u32;
                if total_count == 0 {
                    continue;
                }

                let expected_improvement_percentage =
                    (stats.harmful_count as f32 - stats.helpful_count as f32) / total_count as f32;

                let candidate = CandidateSynapseJson {
                    from_neuron_uuid: synapse.from_uuid.clone(),
                    to_neuron_uuid: synapse.to_uuid.clone(),
                    weight: synapse.weight,
                    expected_improvement_percentage,
                    improved_count: stats.harmful_count,
                    total_count,
                };
                harmful_results.push(candidate);
            }
        }
    }

    if helpful_results.is_empty() {
        if let Some(candidate) = helpful_fallback.take() {
            diagnostics.mark_candidate_selected(&candidate.to_neuron_uuid);
            helpful_results.push(candidate);
        }
    }
    helpful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });
    harmful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
        harmful_results.truncate(limit);
    }

    diagnostics.emit_logs();

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        gpu_used: analyzer.gpu_used(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_format::write_records_to_parquet;
    use crate::types::DiscoverRecord;
    use crate::{AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
    use std::sync::atomic::Ordering as AtomicOrdering;
    use tempfile::tempdir;

    struct ForceGpuFailureGuard;

    impl ForceGpuFailureGuard {
        fn new() -> Self {
            FORCE_GPU_ADAPTER_FAILURE.store(true, AtomicOrdering::SeqCst);
            Self
        }
    }

    impl Drop for ForceGpuFailureGuard {
        fn drop(&mut self) {
            FORCE_GPU_ADAPTER_FAILURE.store(false, AtomicOrdering::SeqCst);
        }
    }

    #[test]
    fn cpu_fallback_when_gpu_not_required() {
        let _guard = ForceGpuFailureGuard::new();
        let analyzer =
            GpuAnalyzer::new(false).expect("CPU analysis should be available when GPU is optional");

        assert!(
            !analyzer.gpu_used(),
            "GPU should not be reported as used when we fall back to CPU analysis"
        );

        let samples = vec![
            HelpfulSample {
                activation: 0.8,
                avg_error: -0.4,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: 0.3,
            },
        ];

        let helpful_stats = analyzer
            .evaluate_helpful(&samples)
            .expect("CPU helpful analysis should succeed");
        assert!(
            helpful_stats.positive_count > 0 || helpful_stats.negative_count > 0,
            "CPU analysis should produce non-zero helpful counts"
        );

        let harmful_stats = analyzer
            .evaluate_harmful(&samples, 0.5)
            .expect("CPU harmful analysis should succeed");
        assert!(
            harmful_stats.harmful_count > 0 || harmful_stats.helpful_count > 0,
            "CPU analysis should produce non-zero harmful counts"
        );
    }

    #[test]
    fn gpu_matching_filters_non_finite_values() {
        let analyzer =
            GpuAnalyzer::new(false).expect("GPU analyser creation should succeed in tests");

        if !analyzer.gpu_used() {
            eprintln!("Skipping GPU filtering test because the GPU is unavailable");
            return;
        }

        let huge = f32::MAX;
        let target_records = vec![
            DiscoverRecord::new(0, "target".to_string(), None, 0.0, vec![0.5, -0.25]),
            DiscoverRecord::new(1, "target".to_string(), None, 0.0, vec![huge, huge]),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, f32::INFINITY, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, 1.0, Vec::new()),
        ];

        let cpu_samples = build_samples(&target_records, &from_records);
        assert!(
            cpu_samples.is_empty(),
            "CPU matching should exclude non-finite samples"
        );

        let gpu_samples = analyzer
            .build_samples_gpu(&target_records, &from_records)
            .expect("GPU matching should succeed");

        assert!(
            gpu_samples.is_empty(),
            "GPU matching should exclude non-finite samples"
        );
    }

    #[test]
    fn gpu_matching_retains_legitimate_zero_samples() {
        let analyzer =
            GpuAnalyzer::new(false).expect("GPU analyser creation should succeed in tests");

        if !analyzer.gpu_used() {
            eprintln!("Skipping zero sample retention test because the GPU is unavailable");
            return;
        }

        let target_records = vec![DiscoverRecord::new(
            42,
            "target".to_string(),
            None,
            0.0,
            vec![0.0, 0.0],
        )];
        let from_records = vec![DiscoverRecord::new(
            42,
            "from".to_string(),
            None,
            0.0,
            Vec::new(),
        )];

        let cpu_samples = build_samples(&target_records, &from_records);
        assert_eq!(
            cpu_samples.len(),
            1,
            "CPU matching should include legitimate zero-valued samples"
        );

        let gpu_samples = analyzer
            .build_samples_gpu(&target_records, &from_records)
            .expect("GPU matching should succeed");

        assert_eq!(
            gpu_samples.len(),
            cpu_samples.len(),
            "GPU matching should retain legitimate zero-valued samples"
        );

        let cpu_sample = cpu_samples[0];
        let gpu_sample = gpu_samples[0];
        assert_eq!(
            cpu_sample.activation, gpu_sample.activation,
            "Zero activation should be preserved by GPU matching"
        );
        assert_eq!(
            cpu_sample.avg_error, gpu_sample.avg_error,
            "Zero average error should be preserved by GPU matching"
        );
    }

    #[test]
    fn merge_batch_results_preserves_order_with_empty_samples() {
        let flags = vec![false, true, false, true];
        let merged = GpuAnalyzer::merge_batch_results(
            &flags,
            vec![
                HelpfulStats {
                    positive_count: 1,
                    ..HelpfulStats::default()
                },
                HelpfulStats {
                    positive_count: 2,
                    ..HelpfulStats::default()
                },
            ],
        );

        assert_eq!(
            merged.len(),
            flags.len(),
            "Merged results should match input batch length"
        );
        assert_eq!(
            merged[0].positive_count, 1,
            "First non-empty sample should remain first"
        );
        assert_eq!(
            merged[1].positive_count, 0,
            "Empty samples should produce default stats"
        );
        assert_eq!(
            merged[2].positive_count, 2,
            "Second non-empty sample should remain in original position"
        );
        assert_eq!(
            merged[3].positive_count, 0,
            "Trailing empty samples should also produce defaults"
        );
    }

    #[test]
    fn helpful_batch_fallback_uses_original_samples() {
        let stats = HelpfulStats::default();
        let samples = vec![
            HelpfulSample {
                activation: 0.9,
                avg_error: -0.3,
            },
            HelpfulSample {
                activation: -0.7,
                avg_error: 0.6,
            },
        ];

        let corrected = GpuAnalyzer::fallback_helpful_stats(stats, &samples);
        let expected = cpu_helpful_stats(&samples);

        assert!(
            expected.positive_count > 0 || expected.negative_count > 0,
            "CPU evaluation should observe helpful samples"
        );
        assert_eq!(
            corrected.positive_count, expected.positive_count,
            "Fallback should mirror CPU positive count when GPU result is empty"
        );
        assert_eq!(
            corrected.negative_count, expected.negative_count,
            "Fallback should mirror CPU negative count when GPU result is empty"
        );
    }

    #[test]
    fn diagnostics_prefers_higher_expected_improvement() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 1_500);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 0);

        diagnostics.record_candidate_attempt("output-0", true);
        diagnostics.record_below_threshold(
            "output-0",
            "hidden-1",
            ThresholdContext {
                sample_count: 42,
                expected_improvement: 0.05,
                threshold: 0.1,
                improved_count: 30,
                worsened_count: 12,
                weight: -0.25,
            },
        );

        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        let reason = entry.best_rejection.as_ref().map(|detail| detail.reason);
        assert!(
            matches!(reason, Some(RejectionReason::BelowThreshold)),
            "Expected below-threshold reason to persist when it has the highest score"
        );
    }

    #[test]
    fn diagnostics_marks_candidate_selection() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.mark_candidate_selected("output-0");
        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        assert!(
            entry.had_candidate,
            "Entry should record that a candidate was selected"
        );
    }

    #[test]
    fn relu_evaluation_identifies_below_threshold_reason() {
        let mut stats = ReluStats::new(ReluOrientation::Positive);
        for _ in 0..(MIN_NEURON_SAMPLE_COUNT + 2) {
            stats.push(1.0, 0.05);
        }
        let evaluation = stats.evaluate("source", "target", 2.0);
        assert!(
            evaluation.candidate.is_none(),
            "Expected candidate to fall below the threshold"
        );
        assert!(
            matches!(
                evaluation.summary.failure,
                Some(ReluFailure::BelowThreshold)
            ),
            "Summary should record the below-threshold failure"
        );
    }

    #[test]
    fn relu_evaluation_keeps_summary_and_candidate_in_sync_on_ties() {
        let _guard = ForceGpuFailureGuard::new();
        let analyzer =
            GpuAnalyzer::new(false).expect("CPU analysis should be available when GPU is optional");

        let mut samples = Vec::new();
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 1.0,
                avg_error: -1.0,
            });
        }
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -1.0,
                avg_error: 1.0,
            });
        }

        let result = evaluate_relu_candidate(&analyzer, "input-0", "output-0", &samples, 0.0)
            .expect("ReLU evaluation should succeed with balanced samples");

        let summary = result
            .best_summary
            .expect("Expected a summary for the best orientation");
        let candidate = result
            .candidate
            .expect("Expected a candidate neuron for tied orientations");

        let summary_orientation = summary.orientation_name();
        let candidate_orientation = if candidate.incoming_weight > 0.0 {
            "positive"
        } else {
            "negative"
        };

        assert_eq!(
            summary_orientation, candidate_orientation,
            "Summary orientation should match the selected candidate orientation when scores tie",
        );
    }

    #[test]
    fn neuron_diagnostics_records_relu_rejection() {
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        let summary = ReluOrientationSummary::below_threshold(
            ReluOrientation::Positive,
            MIN_NEURON_SAMPLE_COUNT,
            12,
            4,
            0.05,
            0.25,
        );
        diagnostics.record_rejection("output-0", "hidden-1", &summary, 0.1);
        let entry = diagnostics
            .entry_for("output-0")
            .expect("diagnostics entry should exist");
        let detail = entry
            .best_rejection
            .as_ref()
            .expect("best rejection should be recorded");
        assert!(
            matches!(detail.reason, NeuronRejectionReason::BelowThreshold),
            "Expected below-threshold reason"
        );
        assert_eq!(
            detail.orientation,
            Some("positive"),
            "Orientation should be preserved"
        );
    }

    #[test]
    fn analyze_neurons_rejects_duplicate_focus_targets() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = (MIN_NEURON_SAMPLE_COUNT + 5) as u32;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-source".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![1.0],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-source".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
        };

        let err = analyze_neurons(&input)
            .err()
            .expect("Neuron analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_rejects_duplicate_focus_targets() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 16;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![-0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
            }],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
        };

        let err = analyze_synapses(&input)
            .err()
            .expect("Synapse analysis should refuse duplicate focus neurons");
        let message = format!("{err}");
        assert!(
            message.contains("duplicate focus neurons"),
            "Expected duplicate focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_synapses_requires_focus_targets() {
        let _guard = ForceGpuFailureGuard::new();

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file: "unused.parquet".to_string(),
            creature,
            focus_neurons: Vec::new(),
            improvement_threshold: None,
            max_candidates: None,
            require_gpu: Some(false),
        };

        let err = analyze_synapses(&input)
            .err()
            .expect("Synapse analysis should refuse empty focus lists");
        let message = format!("{err}");
        assert!(
            message.contains("at least one focus neuron"),
            "Expected missing focus error, got: {message}",
        );
    }

    #[test]
    fn analyze_neurons_respects_gpu_requirement() {
        let _guard = ForceGpuFailureGuard::new();

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        let input = AnalyzeNeuronsInput {
            parquet_file: "non-existent.parquet".to_string(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: None,
            max_candidates: None,
            require_gpu: Some(true),
        };

        let err = analyze_neurons(&input)
            .err()
            .expect("Expected neuron analysis to fail when GPU is required but unavailable");
        let message = format!("{err}");
        assert!(
            message.contains("GPU"),
            "Expected GPU related error message, got: {message}"
        );
    }
}
