use crate::parquet_format::read_records_from_parquet;
use crate::types::DiscoverRecord;
use crate::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson,
    CandidateSynapseJson,
};
use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::{RngCore, SeedableRng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use wgpu::util::DeviceExt;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const EPSILON: f32 = 1e-8;
const WORKGROUP_SIZE: u32 = 256;
const MIN_NEURON_SAMPLE_COUNT: usize = 10;
const GPU_BATCH_SIZE: usize = 32; // Batch multiple GPU operations together for better utilization

fn build_deadline(deadline_ms: Option<u64>) -> Option<SystemTime> {
    deadline_ms.and_then(|target_ms| {
        let since_epoch = Duration::from_millis(target_ms);
        UNIX_EPOCH.checked_add(since_epoch)
    })
}

fn deadline_passed(deadline: &Option<SystemTime>) -> bool {
    #[cfg(test)]
    {
        if let Some(value) = deadline_override::next_override_value() {
            return value;
        }
    }

    matches!(deadline, Some(limit) if SystemTime::now() >= *limit)
}

#[cfg(test)]
mod deadline_override {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::sync::{Mutex, MutexGuard};
    use std::thread::ThreadId;

    static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());
    static OVERRIDE_SEQUENCE: Mutex<VecDeque<bool>> = Mutex::new(VecDeque::new());
    static OVERRIDE_ACTIVE: AtomicBool = AtomicBool::new(false);
    static OVERRIDE_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);

    pub(super) struct DeadlineOverrideGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl DeadlineOverrideGuard {
        pub(super) fn with_sequence(sequence: Vec<bool>) -> Self {
            let lock = OVERRIDE_LOCK
                .lock()
                .expect("Deadline override lock should not be poisoned");
            {
                let mut queue = OVERRIDE_SEQUENCE
                    .lock()
                    .expect("Deadline override queue should not be poisoned");
                queue.clear();
                for value in sequence.into_iter() {
                    queue.push_back(value);
                }
            }
            {
                let mut thread_slot = OVERRIDE_THREAD
                    .lock()
                    .expect("Deadline override thread slot should not be poisoned");
                *thread_slot = Some(std::thread::current().id());
            }
            OVERRIDE_ACTIVE.store(true, AtomicOrdering::SeqCst);
            Self { _lock: lock }
        }
    }

    impl Drop for DeadlineOverrideGuard {
        fn drop(&mut self) {
            OVERRIDE_ACTIVE.store(false, AtomicOrdering::SeqCst);
            {
                let mut thread_slot = OVERRIDE_THREAD
                    .lock()
                    .expect("Deadline override thread slot should not be poisoned");
                *thread_slot = None;
            }
            let mut queue = OVERRIDE_SEQUENCE
                .lock()
                .expect("Deadline override queue should not be poisoned");
            queue.clear();
        }
    }

    pub(super) fn next_override_value() -> Option<bool> {
        if !OVERRIDE_ACTIVE.load(AtomicOrdering::SeqCst) {
            return None;
        }
        let current_thread = std::thread::current().id();
        {
            let thread_slot = OVERRIDE_THREAD
                .lock()
                .expect("Deadline override thread slot should not be poisoned");
            if thread_slot.as_ref() != Some(&current_thread) {
                return None;
            }
        }

        let mut queue = OVERRIDE_SEQUENCE
            .lock()
            .expect("Deadline override queue should not be poisoned");
        queue.pop_front()
    }
}

fn verbose_enabled() -> bool {
    std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok()
}

fn deterministic_shuffle_seed(label: &str, focus: &[String]) -> Option<u64> {
    if std::env::var("NEAT_AI_DISCOVERY_DETERMINISTIC").is_ok() {
        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        focus.hash(&mut hasher);
        Some(hasher.finish())
    } else {
        None
    }
}

enum ShuffleRng {
    Thread(rand::rngs::ThreadRng),
    Std(Box<StdRng>),
}

impl RngCore for ShuffleRng {
    fn next_u32(&mut self) -> u32 {
        match self {
            ShuffleRng::Thread(rng) => rng.next_u32(),
            ShuffleRng::Std(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            ShuffleRng::Thread(rng) => rng.next_u64(),
            ShuffleRng::Std(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            ShuffleRng::Thread(rng) => rng.fill_bytes(dest),
            ShuffleRng::Std(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            ShuffleRng::Thread(rng) => rng.try_fill_bytes(dest),
            ShuffleRng::Std(rng) => rng.try_fill_bytes(dest),
        }
    }
}

fn build_shuffle_rng(label: &str, focus: &[String]) -> ShuffleRng {
    if let Some(seed) = deterministic_shuffle_seed(label, focus) {
        ShuffleRng::Std(Box::new(StdRng::seed_from_u64(seed)))
    } else {
        ShuffleRng::Thread(thread_rng())
    }
}

#[cfg(test)]
static FORCE_GPU_ADAPTER_FAILURE: AtomicBool = AtomicBool::new(false);

pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<SynapseNoCandidateSummary>,
}

pub struct AnalyzeNeuronsResult {
    pub helpful_neurons: Vec<CandidateNeuronJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<NeuronNoCandidateSummary>,
}

pub struct AnalyzeAllResult {
    pub synapse: Option<AnalyzeSynapsesResult>,
    pub neuron: Option<AnalyzeNeuronsResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynapseNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

#[derive(Debug, Clone)]
pub struct SynapseNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub sample_count: Option<usize>,
    pub source_record_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub suggested_weight: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SynapseNoCandidateSummary {
    pub target_uuid: String,
    pub reason: SynapseNoCandidateReason,
    pub evaluated_candidates: u32,
    pub candidates_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<SynapseNoCandidateDetail>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuronNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
}

#[derive(Debug, Clone)]
pub struct NeuronNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub orientation: Option<String>,
    pub sample_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub outgoing_weight: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct NeuronNoCandidateSummary {
    pub target_uuid: String,
    pub reason: NeuronNoCandidateReason,
    pub evaluated_sources: u32,
    pub sources_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<NeuronNoCandidateDetail>,
}

struct OrderedNeuron {
    uuid: String,
    index: usize,
}

type RecordCacheLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;
type CachedNeuronRecords = OnceCell<Arc<Vec<DiscoverRecord>>>;

struct RecordCache {
    parquet_file: String,
    cache: Mutex<HashMap<String, Arc<CachedNeuronRecords>>>,
    loader: Arc<RecordCacheLoader>,
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
    log_enabled: bool,
    entries: HashMap<String, TargetDiagnosticEntry>,
}

impl TargetDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let log_enabled = std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), TargetDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
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
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    fn emit_logs(&self) {
        if !self.log_enabled {
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

    fn no_candidate_summaries(&self) -> Vec<SynapseNoCandidateSummary> {
        self.entries
            .values()
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                if entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    let reason = match best.reason {
                        RejectionReason::NoSamples => SynapseNoCandidateReason::NoSamples,
                        RejectionReason::ZeroImprovement => {
                            SynapseNoCandidateReason::ZeroImprovement
                        }
                        RejectionReason::BelowThreshold => SynapseNoCandidateReason::BelowThreshold,
                    };
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(SynapseNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            sample_count: Some(best.sample_count),
                            source_record_count: Some(best.source_record_count),
                            improved_count: Some(best.improved_count),
                            worsened_count: Some(best.worsened_count),
                            expected_improvement: Some(best.expected_improvement),
                            threshold: Some(best.threshold),
                            suggested_weight: best.weight,
                        }),
                    };
                }

                SynapseNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: SynapseNoCandidateReason::NoDiagnostics,
                    evaluated_candidates: entry.evaluated_candidates,
                    candidates_with_samples: entry.candidates_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
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
    log_enabled: bool,
    entries: HashMap<String, NeuronDiagnosticEntry>,
}

impl NeuronDiagnostics {
    fn new(targets: &[&String]) -> Self {
        let log_enabled = std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok();
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert(target.to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled,
            entries,
        }
    }

    #[cfg(test)]
    fn new_for_tests(targets: &[&str]) -> Self {
        let mut entries = HashMap::new();
        for target in targets {
            entries.insert((*target).to_string(), NeuronDiagnosticEntry::new(target));
        }
        Self {
            log_enabled: true,
            entries,
        }
    }

    fn set_target_record_count(&mut self, target_uuid: &str, count: usize) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.target_record_count = count;
        }
    }

    fn record_candidate_attempt(&mut self, target_uuid: &str, had_samples: bool) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.evaluated_sources += 1;
            if had_samples {
                entry.sources_with_samples += 1;
            }
        }
    }

    fn record_no_samples(&mut self, target_uuid: &str, source_uuid: &str) {
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
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    fn emit_logs(&self) {
        if !self.log_enabled {
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

    fn no_candidate_summaries(&self) -> Vec<NeuronNoCandidateSummary> {
        self.entries
            .values()
            .filter(|entry| !entry.had_candidate)
            .map(|entry| {
                if entry.evaluated_sources == 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoEligibleSources,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    let reason = match best.reason {
                        NeuronRejectionReason::NoSamples => NeuronNoCandidateReason::NoSamples,
                        NeuronRejectionReason::NotEnoughActivations => {
                            NeuronNoCandidateReason::NotEnoughActivations
                        }
                        NeuronRejectionReason::WeightDegenerate => {
                            NeuronNoCandidateReason::WeightDegenerate
                        }
                        NeuronRejectionReason::BelowThreshold => {
                            NeuronNoCandidateReason::BelowThreshold
                        }
                    };
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: Some(NeuronNoCandidateDetail {
                            source_uuid: Some(best.source_uuid.clone()),
                            orientation: best.orientation.map(|name| name.to_string()),
                            sample_count: Some(best.sample_count),
                            improved_count: Some(best.improved_count),
                            worsened_count: Some(best.worsened_count),
                            expected_improvement: Some(best.expected_improvement),
                            threshold: Some(best.threshold),
                            outgoing_weight: best.outgoing_weight,
                        }),
                    };
                }

                NeuronNoCandidateSummary {
                    target_uuid: entry.target_uuid.clone(),
                    reason: NeuronNoCandidateReason::NoDiagnostics,
                    evaluated_sources: entry.evaluated_sources,
                    sources_with_samples: entry.sources_with_samples,
                    target_record_count: entry.target_record_count,
                    detail: None,
                }
            })
            .collect()
    }
}

impl RecordCache {
    fn new(parquet_file: &str) -> Self {
        Self::with_loader(
            parquet_file,
            Arc::new(|file: &str, neuron_uuid: &str| read_records_from_parquet(file, neuron_uuid)),
        )
    }

    fn with_loader(parquet_file: &str, loader: Arc<RecordCacheLoader>) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(HashMap::new()),
            loader,
        }
    }

    fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        let cache_entry = {
            let mut cache = self.cache.lock().expect("record cache mutex poisoned");
            cache
                .entry(neuron_uuid.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        let loader = Arc::clone(&self.loader);
        let parquet_file = self.parquet_file.clone();
        let context_uuid = neuron_uuid.to_string();
        let load_uuid = context_uuid.clone();

        let arc_records = cache_entry
            .get_or_try_init(move || -> Result<Arc<Vec<DiscoverRecord>>> {
                let mut records = loader(&parquet_file, &load_uuid)?;
                records.sort_by_key(|record| record.obs_index);
                Ok(Arc::new(records))
            })
            .with_context(|| {
                format!("Failed to read discovery records for neuron {context_uuid}")
            })?;

        Ok(Arc::clone(arc_records))
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
    error_squared: f32,
    activation_squared: f32,
    error_activation: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
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
    error_sq_sum: f32,
    activation_sq_sum: f32,
    error_activation_sum: f32,
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
        total_baseline_error_sq: f32,
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

        // Calculate improvement based on magnitude (reduction in squared error)
        // improvement = baseline_sq - new_sq
        // = 2*w*sum(ea) - w^2*sum(aa)
        let improvement_magnitude = 2.0 * outgoing_weight * self.error_activation_sum
            - outgoing_weight * outgoing_weight * self.activation_sq_sum;

        // Normalize by total baseline error of ALL samples (not just active ones)
        let expected_improvement = if total_baseline_error_sq > EPSILON {
            improvement_magnitude / total_baseline_error_sq
        } else {
            0.0
        };

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

fn identity_activation(x: f32) -> f32 {
    x
}

fn bipolar_activation(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn clipped_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

fn absolute_activation(x: f32) -> f32 {
    x.abs()
}

fn inverse_activation(x: f32) -> f32 {
    1.0 - x
}

const ACTIVATION_SPECS: [ActivationCandidateSpec; 11] = [
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
    ActivationCandidateSpec {
        name: "IDENTITY",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: identity_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "BIPOLAR",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: bipolar_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "CLIPPED",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: clipped_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "ABSOLUTE",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: absolute_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "INVERSE",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: inverse_activation,
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
        if sample.activation.abs() <= EPSILON && sample.avg_error.abs() <= EPSILON {
            continue;
        }

        // Always accumulate error stats if there is any error, even if activation is small
        if sample.avg_error.abs() > EPSILON {
            stats.error_sq_sum += sample.avg_error * sample.avg_error;
        }

        if sample.activation.abs() > EPSILON {
            stats.activation_sq_sum += sample.activation * sample.activation;
            stats.error_activation_sum += sample.avg_error * sample.activation;

            // Positive/negative counts are only computed when BOTH activation AND error exceed epsilon
            if sample.avg_error.abs() > EPSILON {
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
            stats.error_sq_sum += contribution.error_squared;
            stats.activation_sq_sum += contribution.activation_squared;
            stats.error_activation_sum += contribution.error_activation;
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
                    stats.error_sq_sum += contribution.error_squared;
                    stats.activation_sq_sum += contribution.activation_squared;
                    stats.error_activation_sum += contribution.error_activation;
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
    let mut total_baseline_error_sq = 0.0;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        total_baseline_error_sq += sample.avg_error * sample.avg_error;

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

    let positive_eval =
        positive_stats.evaluate(source_uuid, target_uuid, threshold, total_baseline_error_sq);
    let negative_eval =
        negative_stats.evaluate(source_uuid, target_uuid, threshold, total_baseline_error_sq);
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

    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

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
            for (sample, output) in samples.iter().zip(outputs.iter()) {
                let new_error = sample.avg_error - outgoing_weight * output;
                if new_error.abs() + EPSILON < sample.avg_error.abs() {
                    improved_count += 1;
                }
            }

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            // Calculate improvement based on magnitude (reduction in squared error)
            // improvement = baseline_sq - new_sq
            // = 2*w*sum(ea) - w^2*sum(aa)
            let improvement_magnitude = 2.0 * outgoing_weight * sum_error_activation
                - outgoing_weight * outgoing_weight * sum_activation_sq;

            let expected_improvement_percentage = if total_baseline_error_sq > EPSILON {
                improvement_magnitude / total_baseline_error_sq
            } else {
                0.0
            };

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

fn analyze_neurons_with_cache(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeNeuronsResult> {
    let require_gpu = input.require_gpu.unwrap_or(cfg!(target_os = "macos"));
    let analyzer = GpuAnalyzer::new(require_gpu)?;

    let threshold = input.improvement_threshold.unwrap_or(0.1);
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let mut order_map: HashMap<&str, usize> = HashMap::new();
    for neuron in &ordered_neurons {
        order_map.insert(neuron.uuid.as_str(), neuron.index);
    }

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    let mut helpful_map: HashMap<(String, String, String), CandidateNeuronJson> = HashMap::new();

    let mut diagnostics = NeuronDiagnostics::new(&unique_focus);

    let deadline = build_deadline(input.analysis_deadline_ms);
    let mut rng = build_shuffle_rng("analyze_synapses", &input.focus_neurons);
    let mut focus_order = unique_focus.clone();
    focus_order.shuffle(&mut rng);
    let mut analysis_timed_out = false;

    for target_uuid in focus_order {
        if deadline_passed(&deadline) {
            analysis_timed_out = true;
            break;
        }

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

        let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
            .collect();
        eligible_sources.shuffle(&mut rng);

        for source in eligible_sources {
            if deadline_passed(&deadline) {
                analysis_timed_out = true;
                break;
            }

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

        if analysis_timed_out {
            break;
        }
    }

    if analysis_timed_out && verbose_enabled() {
        eprintln!("[NEAT-AI-Discovery][verbose] analyse_neurons reached analysis deadline; returning partial results.");
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

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used: analyzer.gpu_used(),
        no_candidate_reasons,
    })
}

pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    let cache = Arc::new(RecordCache::new(&input.parquet_file));
    analyze_neurons_with_cache(input, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ACTIVATION_SPECS;

    #[test]
    fn test_identity_activation() {
        assert_eq!(identity_activation(1.0), 1.0);
        assert_eq!(identity_activation(-1.0), -1.0);
        assert_eq!(identity_activation(0.0), 0.0);
    }

    #[test]
    fn test_bipolar_activation() {
        assert_eq!(bipolar_activation(1.0), 1.0);
        assert_eq!(bipolar_activation(0.0001), 1.0);
        assert_eq!(bipolar_activation(0.0), -1.0);
        assert_eq!(bipolar_activation(-1.0), -1.0);
        assert_eq!(bipolar_activation(-0.0001), -1.0);
    }

    #[test]
    fn test_clipped_activation() {
        assert_eq!(clipped_activation(1.5), 1.0);
        assert_eq!(clipped_activation(0.5), 0.5);
        assert_eq!(clipped_activation(-0.5), -0.5);
        assert_eq!(clipped_activation(-1.5), -1.0);
    }

    #[test]
    fn test_absolute_activation() {
        assert_eq!(absolute_activation(1.0), 1.0);
        assert_eq!(absolute_activation(-1.0), 1.0);
        assert_eq!(absolute_activation(0.0), 0.0);
    }

    #[test]
    fn test_inverse_activation() {
        assert_eq!(inverse_activation(1.0), 0.0);
        assert_eq!(inverse_activation(0.0), 1.0);
        assert_eq!(inverse_activation(-1.0), 2.0);
    }

    #[test]
    fn test_specs_include_new_activations() {
        let names: Vec<&str> = ACTIVATION_SPECS.iter().map(|s| s.name).collect();
        assert!(names.contains(&"IDENTITY"));
        assert!(!names.contains(&"LeakyReLU"));
        assert!(names.contains(&"BIPOLAR"));
        assert!(names.contains(&"CLIPPED"));
        assert!(names.contains(&"ABSOLUTE"));
        assert!(names.contains(&"INVERSE"));
    }
}

pub fn analyze_all(input: &AnalyzeAllInput) -> Result<AnalyzeAllResult> {
    let include_synapse = input.include_synapse_analysis.unwrap_or(true);
    let include_neuron = input.include_neuron_analysis.unwrap_or(true);

    if !include_synapse && !include_neuron {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
        });
    }

    let shared_cache = Arc::new(RecordCache::new(&input.parquet_file));

    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            improvement_threshold: input.improvement_threshold,
            max_candidates: input.max_synapse_candidates,
            require_gpu: input.require_gpu,
            analysis_deadline_ms: input.analysis_deadline_ms,
        })
    } else {
        None
    };

    let neuron_input = if include_neuron {
        Some(AnalyzeNeuronsInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            improvement_threshold: input.improvement_threshold,
            max_candidates: input.max_neuron_candidates,
            require_gpu: input.require_gpu,
            analysis_deadline_ms: input.analysis_deadline_ms,
        })
    } else {
        None
    };

    let (synapse_result, neuron_result) = rayon::join(
        || -> Result<Option<AnalyzeSynapsesResult>> {
            if let Some(inner) = synapse_input.clone() {
                analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache)).map(Some)
            } else {
                Ok(None)
            }
        },
        || -> Result<Option<AnalyzeNeuronsResult>> {
            if let Some(inner) = neuron_input.clone() {
                analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache)).map(Some)
            } else {
                Ok(None)
            }
        },
    );

    Ok(AnalyzeAllResult {
        synapse: synapse_result?,
        neuron: neuron_result?,
    })
}

fn analyze_synapses_with_cache(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeSynapsesResult> {
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

    let mut diagnostics = TargetDiagnostics::new(&unique_focus);

    let deadline = build_deadline(input.analysis_deadline_ms);
    let mut rng = build_shuffle_rng("analyze_neurons", &input.focus_neurons);
    let mut focus_order = unique_focus.clone();
    focus_order.shuffle(&mut rng);
    let mut analysis_timed_out = false;

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

    'target_loop: for target_uuid in focus_order.iter() {
        if deadline_passed(&deadline) {
            analysis_timed_out = true;
            break;
        }

        let target_uuid = *target_uuid;

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

        let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons
            .iter()
            .filter(|neuron| neuron.index < target_index)
            .collect();
        eligible_sources.shuffle(&mut rng);

        // Collect helpful candidates for batching
        for source in eligible_sources {
            if deadline_passed(&deadline) {
                analysis_timed_out = true;
                break 'target_loop;
            }

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

        let improvement_magnitude =
            2.0 * weight * stats.error_activation_sum - weight * weight * stats.activation_sq_sum;

        let expected_improvement_percentage = if stats.error_sq_sum > EPSILON {
            improvement_magnitude / stats.error_sq_sum
        } else {
            0.0
        };

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

    if !analysis_timed_out {
        // Harmful synapses (existing connections) - process per target
        'harmful_targets: for target_uuid in focus_order.iter() {
            if deadline_passed(&deadline) {
                analysis_timed_out = true;
                break 'harmful_targets;
            }

            let target_uuid = *target_uuid;

            let target_records_arc = cache.get(target_uuid)?;
            if target_records_arc.is_empty() {
                continue;
            }
            let target_records = target_records_arc.as_ref();

            if let Some(existing) = synapses_by_target.get(target_uuid.as_str()) {
                for synapse in existing {
                    if deadline_passed(&deadline) {
                        analysis_timed_out = true;
                        break 'harmful_targets;
                    }

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

                    let expected_improvement_percentage = (stats.harmful_count as f32
                        - stats.helpful_count as f32)
                        / total_count as f32;

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
    }

    if analysis_timed_out && verbose_enabled() {
        eprintln!("[NEAT-AI-Discovery][verbose] analyse_synapses reached analysis deadline; returning partial results.");
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

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        gpu_used: analyzer.gpu_used(),
        no_candidate_reasons,
    })
}

pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    let cache = Arc::new(RecordCache::new(&input.parquet_file));
    analyze_synapses_with_cache(input, cache)
}

#[cfg(test)]
mod tests_synapses {
    use super::*;
    use crate::parquet_format::write_records_to_parquet;
    use crate::{CreatureJson, NeuronJson, SynapseJson};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    fn build_deadline_returns_wall_clock_deadline() {
        let future_wall_clock = SystemTime::now() + Duration::from_millis(200);
        let deadline_ms = future_wall_clock
            .duration_since(UNIX_EPOCH)
            .expect("future wall clock timestamp should convert to epoch duration")
            .as_millis() as u64;

        let deadline = build_deadline(Some(deadline_ms)).expect("deadline should be constructed");

        let target_millis = future_wall_clock
            .duration_since(UNIX_EPOCH)
            .expect("future wall clock timestamp should convert to epoch duration")
            .as_millis();
        let computed_millis = deadline
            .duration_since(UNIX_EPOCH)
            .expect("deadline should be convertible back to epoch duration")
            .as_millis();

        assert_eq!(
            computed_millis, target_millis,
            "deadline should preserve the original wall clock timestamp"
        );
    }

    #[test]
    fn deadline_passed_detects_elapsed_wall_clock_deadline() {
        let past_deadline = SystemTime::now() - Duration::from_millis(25);
        assert!(
            deadline_passed(&Some(past_deadline)),
            "past deadlines should be treated as expired immediately"
        );

        let future_deadline = SystemTime::now() + Duration::from_millis(200);
        assert!(
            !deadline_passed(&Some(future_deadline)),
            "future deadlines should not be marked as expired"
        );

        assert!(
            !deadline_passed(&None),
            "missing deadlines should behave as if no timeout was requested"
        );
    }

    #[test]
    fn record_cache_loads_once_per_neuron_under_contention() {
        let load_counter = Arc::new(AtomicUsize::new(0));
        let loader_counter = Arc::clone(&load_counter);
        let loader = Arc::new(
            move |_file: &str, neuron_uuid: &str| -> Result<Vec<DiscoverRecord>> {
                loader_counter.fetch_add(1, AtomicOrdering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                Ok(vec![DiscoverRecord::new(
                    0,
                    neuron_uuid.to_string(),
                    None,
                    0.0,
                    Vec::new(),
                )])
            },
        );

        let cache = Arc::new(RecordCache::with_loader("unused.parquet", loader));
        let worker_count = 4;
        let barrier = Arc::new(Barrier::new(worker_count));
        let mut handles = Vec::new();
        for _ in 0..worker_count {
            let cache_clone = Arc::clone(&cache);
            let barrier_clone = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_clone.wait();
                cache_clone
                    .get("neuron-1")
                    .expect("cache should load neuron records");
            }));
        }

        for handle in handles {
            handle.join().expect("worker thread should exit cleanly");
        }

        assert_eq!(
            load_counter.load(AtomicOrdering::SeqCst),
            1,
            "record cache should only hit the loader once even when multiple threads request the same neuron"
        );
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
        let evaluation = stats.evaluate("source", "target", 2.0, 1.0);
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
    fn target_diagnostics_reports_no_samples_reason() {
        let mut diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 25);
        diagnostics.record_candidate_attempt("output-0", false);
        diagnostics.record_no_samples("output-0", "input-0", 8);

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(
            summaries.len(),
            1,
            "Expected a single diagnostic summary for target without candidates"
        );

        let summary = &summaries[0];
        assert_eq!(
            summary.target_uuid, "output-0",
            "Target UUID should be preserved in summary"
        );
        assert!(
            matches!(summary.reason, SynapseNoCandidateReason::NoSamples),
            "Expected no-samples reason"
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
            analysis_deadline_ms: None,
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
            analysis_deadline_ms: None,
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
            analysis_deadline_ms: None,
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
    fn analyze_synapses_reports_diagnostics_when_no_candidates() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.05],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

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
            parquet_file: parquet_file.clone(),
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should succeed even without candidates");

        assert!(
            result.helpful_synapses.is_empty(),
            "Expected no helpful candidates when there are no eligible sources"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(SynapseNoCandidateReason::NoEligibleSources)),
            "Expected diagnostics to explain missing candidates"
        );
    }

    #[test]
    fn analyze_synapses_stops_harmful_processing_after_deadline() {
        let _gpu_guard = ForceGpuFailureGuard::new();
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![
            false, false, false, false, false, false, true, false, false, false,
        ]);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..16 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-1".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 2,
            neurons: vec![
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-1".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.8,
                },
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "output-1".to_string(),
                    weight: 0.6,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should complete even when the deadline triggers");

        assert!(
            result.harmful_synapses.is_empty(),
            "Harmful synapses should not be evaluated after the deadline is exceeded"
        );
    }

    #[test]
    fn analyze_all_runs_synapse_and_neuron_phases() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
            ));
        }
        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
            }],
        };

        let input = AnalyzeAllInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            harmful_threshold: Some(-0.05),
            max_synapse_candidates: Some(5),
            max_neuron_candidates: Some(5),
            require_gpu: Some(false),
            analysis_deadline_ms: None,
            include_synapse_analysis: Some(true),
            include_neuron_analysis: Some(true),
        };

        let result = analyze_all(&input).expect("Combined analysis should succeed");
        assert!(result.synapse.is_some(), "Synapse phase should run");
        assert!(result.neuron.is_some(), "Neuron phase should run");
    }

    #[test]
    fn analyze_neurons_reports_diagnostics_when_no_candidates() {
        let _guard = ForceGpuFailureGuard::new();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2],
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
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            require_gpu: Some(false),
            analysis_deadline_ms: None,
        };

        let result = analyze_neurons(&input)
            .expect("Neuron analysis should succeed even without candidates");

        assert!(
            result.helpful_neurons.is_empty(),
            "Expected no neuron candidates when the source neuron lacks samples"
        );
        let reason = result
            .no_candidate_reasons
            .first()
            .map(|summary| summary.reason.clone());
        assert!(
            matches!(reason, Some(NeuronNoCandidateReason::NoSamples)),
            "Expected diagnostics to explain missing neuron candidates"
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
            analysis_deadline_ms: None,
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

    #[test]
    fn test_divergence_between_count_and_magnitude() {
        // This test reproduces the scenario where a high percentage of samples improve (count),
        // but one large failure causes the net error reduction (magnitude) to be negative.
        // Old Logic would report positive improvement (e.g. > 80%)
        // New Logic should report negative improvement (e.g. < 0%)

        let mut samples = Vec::new();

        // 10 samples that improve slightly
        for _ in 0..10 {
            samples.push(HelpfulSample {
                activation: 0.1,
                avg_error: 0.1,
            });
        }

        // 1 sample that worsens drastically
        // Activation -10.0. Error 10.0.
        samples.push(HelpfulSample {
            activation: -10.0,
            avg_error: 10.0,
        });

        let stats = cpu_helpful_stats(&samples);

        // Check stats calculation
        assert_eq!(stats.negative_count, 10);
        assert_eq!(stats.positive_count, 1);

        let positive_is_better = stats.positive_count >= stats.negative_count;
        assert!(!positive_is_better);

        let weight = 1.0;

        let improvement_magnitude =
            2.0 * weight * stats.error_activation_sum - weight * weight * stats.activation_sq_sum;

        assert!((stats.error_activation_sum - (-99.9)).abs() < 1e-4);
        assert!((stats.activation_sq_sum - 100.1).abs() < 1e-4);

        assert!(improvement_magnitude < -200.0);

        let expected_improvement_percentage = improvement_magnitude / stats.error_sq_sum;

        assert!((stats.error_sq_sum - 100.1).abs() < 1e-4);
        assert!(expected_improvement_percentage < -2.0); // Should be around -3.0 (-300%)

        println!("Expected improvement (Magnitude): {expected_improvement_percentage}");
    }

    #[test]
    fn test_cpu_gpu_divergence_activation_without_error() {
        // Test case: samples with substantial activation but negligible error (below EPSILON)
        // CPU computes activation_sq_sum and error_activation_sum when activation > epsilon
        // GPU should match this behavior (fixed in shader)

        let samples = vec![
            HelpfulSample {
                activation: 1.0, // Substantial activation
                avg_error: 1e-9, // Negligible error (below EPSILON = 1e-8)
            },
            HelpfulSample {
                activation: -2.0, // Substantial activation
                avg_error: 0.0,   // Zero error
            },
            HelpfulSample {
                activation: 3.0,  // Substantial activation
                avg_error: -1e-9, // Negligible error (below EPSILON)
            },
        ];

        let cpu_stats = cpu_helpful_stats(&samples);

        // CPU computes activation stats even when error is negligible or zero
        // activation_sq_sum = 1.0^2 + (-2.0)^2 + 3.0^2 = 1 + 4 + 9 = 14.0
        assert!((cpu_stats.activation_sq_sum - 14.0).abs() < 1e-6,
            "CPU should compute activation_sq_sum when activation > epsilon, even if error <= epsilon");

        // error_activation_sum should be computed (even if very small)
        // For sample 1: 1.0 * 1e-9 = 1e-9
        // For sample 2: -2.0 * 0.0 = 0.0
        // For sample 3: 3.0 * (-1e-9) = -3e-9
        // Total: -2e-9
        assert!(
            cpu_stats.error_activation_sum.abs() < 1e-6,
            "error_activation_sum should be computed (very small but non-zero)"
        );

        // Now test GPU (if available) - should match CPU after fix
        let analyzer = GpuAnalyzer::new(false);
        if let Ok(analyzer) = analyzer {
            if analyzer.gpu_used() {
                let gpu_stats = analyzer
                    .evaluate_helpful(&samples)
                    .expect("GPU evaluation should succeed");

                // GPU should match CPU behavior after fix
                assert!(
                    (gpu_stats.activation_sq_sum - cpu_stats.activation_sq_sum).abs() < 1e-6,
                    "GPU activation_sq_sum should match CPU: CPU={}, GPU={}",
                    cpu_stats.activation_sq_sum,
                    gpu_stats.activation_sq_sum
                );
                assert!(
                    (gpu_stats.error_activation_sum - cpu_stats.error_activation_sum).abs() < 1e-6,
                    "GPU error_activation_sum should match CPU: CPU={}, GPU={}",
                    cpu_stats.error_activation_sum,
                    gpu_stats.error_activation_sum
                );
            }
        }
    }

    #[test]
    fn test_cpu_gpu_positive_negative_count_divergence() {
        // Test case: samples with substantial activation but negligible error (below EPSILON)
        // Both CPU and GPU should NOT compute positive/negative counts when error <= epsilon,
        // even if activation > epsilon. This matches the GPU shader behavior.

        let samples = vec![
            HelpfulSample {
                activation: 1.0, // Substantial activation
                avg_error: 1e-9, // Negligible error (below EPSILON = 1e-8)
            },
            HelpfulSample {
                activation: -2.0, // Substantial activation
                avg_error: 0.0,   // Zero error
            },
            HelpfulSample {
                activation: 3.0,  // Substantial activation
                avg_error: -1e-9, // Negligible error (below EPSILON)
            },
        ];

        let cpu_stats = cpu_helpful_stats(&samples);

        // CPU should NOT compute positive/negative counts when error <= epsilon
        // even though activation > epsilon
        assert_eq!(cpu_stats.positive_count, 0,
            "CPU should not compute positive_count when error <= epsilon, even if activation > epsilon");
        assert_eq!(cpu_stats.negative_count, 0,
            "CPU should not compute negative_count when error <= epsilon, even if activation > epsilon");
        assert_eq!(
            cpu_stats.positive_improvement_sum, 0.0,
            "CPU should not compute positive_improvement_sum when error <= epsilon"
        );
        assert_eq!(
            cpu_stats.negative_improvement_sum, 0.0,
            "CPU should not compute negative_improvement_sum when error <= epsilon"
        );

        // Now test GPU (if available) - should match CPU
        let analyzer = GpuAnalyzer::new(false);
        if let Ok(analyzer) = analyzer {
            if analyzer.gpu_used() {
                let gpu_stats = analyzer
                    .evaluate_helpful(&samples)
                    .expect("GPU evaluation should succeed");

                // GPU should match CPU behavior
                assert_eq!(
                    gpu_stats.positive_count, cpu_stats.positive_count,
                    "GPU positive_count should match CPU: CPU={}, GPU={}",
                    cpu_stats.positive_count, gpu_stats.positive_count
                );
                assert_eq!(
                    gpu_stats.negative_count, cpu_stats.negative_count,
                    "GPU negative_count should match CPU: CPU={}, GPU={}",
                    cpu_stats.negative_count, gpu_stats.negative_count
                );
            }
        }
    }
}
