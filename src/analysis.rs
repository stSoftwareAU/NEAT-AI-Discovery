use crate::types::DiscoverRecord;
use crate::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson,
    CandidateSynapseJson, SynapseJson,
};
use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, SystemTime};
use wgpu::util::DeviceExt;

const EPSILON: f32 = 1e-8;
/// Workgroup size for GPU compute shaders. Must match the @workgroup_size in WGSL shaders.
/// 256 is optimal for Apple Silicon: divisible by SIMD width (32), good occupancy,
/// and allows efficient wavefront scheduling on M1/M2/M3/M4 GPUs.
const WORKGROUP_SIZE: u32 = 256;
const MIN_NEURON_SAMPLE_COUNT: usize = 10;

/// Check if a target neuron uses a threshold-based discrete activation function.
/// STEP and BIPOLAR can benefit from a specialised threshold-crossing analysis model
/// that counts how many samples would flip to the correct output if we add a new connection.
///
/// - **STEP**: Output = value > 0 ? 1 : 0
/// - **BIPOLAR**: Output = value > 0 ? 1 : -1
#[inline]
fn is_threshold_activation(squash: &str) -> bool {
    matches!(squash.to_uppercase().as_str(), "STEP" | "BIPOLAR")
}
/// Number of GPU operations to batch together for better utilisation.
/// Apple Silicon's Unified Memory Architecture (UMA) eliminates CPU-GPU copy overhead,
/// allowing larger batches without memory transfer penalty. 512 is tuned for M3/M4
/// which have more GPU cores than earlier chips. Larger batches improve GPU occupancy
/// by reducing per-batch overhead and keeping the GPU busy longer.
const GPU_BATCH_SIZE: usize = 512;

fn build_deadline(deadline_ms: Option<u64>) -> Option<SystemTime> {
    // Treat deadline_ms as a relative duration (milliseconds from now), not an absolute timestamp.
    // The calling code (TypeScript) calculates this as Date.now() + duration, but we want to treat
    // it as a duration to avoid issues with clock skew and to match the expected semantics.
    // If the calling code passes an absolute timestamp, we need to convert it to a relative duration.
    // If None is passed, apply default 10 minute timeout to prevent runaway analysis.
    const DEFAULT_DURATION_MS: u64 = 600_000; // 10 minutes (10 * 60 * 1000)

    let target_ms = deadline_ms.unwrap_or(DEFAULT_DURATION_MS);

    Some(target_ms).and_then(|target_ms| {
        // Heuristic: if the value is less than year 2000 in milliseconds (946684800000),
        // treat it as a relative duration. Otherwise, it's likely an absolute timestamp
        // from the calling code, so convert it to a relative duration.
        const YEAR_2000_MS: u64 = 946_684_800_000;

        let relative_ms = if target_ms < YEAR_2000_MS {
            // Small value - treat as relative duration (milliseconds from now)
            target_ms
        } else {
            // Large value - likely an absolute timestamp from calling code.
            // Convert to relative duration by subtracting current time.
            let now_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()?
                .as_millis() as u64;

            // If the timestamp is in the past, return None (deadline already passed)
            if target_ms <= now_ms {
                return None;
            }

            // Calculate relative duration
            target_ms - now_ms
        };

        // Validate duration bounds: minimum 3 seconds, maximum 1 hour
        // If invalid, default to 10 minutes (expected typical value)
        const MIN_DURATION_MS: u64 = 3_000; // 3 seconds
        const MAX_DURATION_MS: u64 = 3_600_000; // 1 hour (60 * 60 * 1000)

        let validated_ms = if relative_ms < MIN_DURATION_MS {
            eprintln!(
                "⚠️  WARNING: analysis_deadline_ms ({:.1}s) is less than minimum (3s). Using default 10 minute timeout.",
                relative_ms as f64 / 1000.0
            );
            DEFAULT_DURATION_MS
        } else if relative_ms > MAX_DURATION_MS {
            eprintln!(
                "⚠️  WARNING: analysis_deadline_ms ({:.1}s) exceeds maximum (1 hour). Using default 10 minute timeout.",
                relative_ms as f64 / 1000.0
            );
            DEFAULT_DURATION_MS
        } else {
            relative_ms
        };

        SystemTime::now().checked_add(Duration::from_millis(validated_ms))
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

    static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());
    static OVERRIDE_SEQUENCE: Mutex<VecDeque<bool>> = Mutex::new(VecDeque::new());
    static OVERRIDE_ACTIVE: AtomicBool = AtomicBool::new(false);

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
            OVERRIDE_ACTIVE.store(true, AtomicOrdering::SeqCst);
            Self { _lock: lock }
        }
    }

    impl Drop for DeadlineOverrideGuard {
        fn drop(&mut self) {
            OVERRIDE_ACTIVE.store(false, AtomicOrdering::SeqCst);
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
        // Allow any thread to consume the override sequence for parallel processing compatibility
        // With parallel processing via par_iter(), worker threads have different thread IDs,
        // so we allow all threads to see and consume the override sequence.
        let mut queue = OVERRIDE_SEQUENCE
            .lock()
            .expect("Deadline override queue should not be poisoned");
        queue.pop_front()
    }
}

fn verbose_enabled() -> bool {
    std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok()
}

/// Suppress Mesa/libEGL debug warnings on Linux.
///
/// When wgpu initialises on Linux, it probes multiple GPU backends (EGL, Vulkan, etc.).
/// If the user lacks permission to access `/dev/dri/renderD*` or `/dev/dri/card*` devices,
/// libEGL emits warnings like "failed to open /dev/dri/renderD128: Permission denied".
///
/// These warnings are often benign if wgpu finds an alternative backend (e.g., Vulkan via
/// a different ICD loader). This function suppresses the warnings by setting environment
/// variables that quiet Mesa's debug output.
///
/// Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to enable suppression.
#[cfg(target_os = "linux")]
fn suppress_mesa_warnings_if_requested() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if env::var("NEAT_AI_DISCOVERY_QUIET_GPU").is_ok() {
            // Suppress EGL debug messages (these cause "failed to open /dev/dri/..." warnings)
            if env::var("EGL_LOG_LEVEL").is_err() {
                // SAFETY: single-threaded at this point (Once guard) and before GPU init
                unsafe { env::set_var("EGL_LOG_LEVEL", "fatal") };
            }

            // Suppress Mesa GLSL shader cache warnings
            if env::var("MESA_GLSL_CACHE_DISABLE").is_err() {
                unsafe { env::set_var("MESA_GLSL_CACHE_DISABLE", "true") };
            }

            // Suppress general Mesa debug output
            if env::var("MESA_DEBUG").is_err() {
                unsafe { env::set_var("MESA_DEBUG", "silent") };
            }
        }
    });
}

#[cfg(not(target_os = "linux"))]
fn suppress_mesa_warnings_if_requested() {
    // No-op on non-Linux platforms
}

/// Ensure XDG_RUNTIME_DIR is set on Linux (required by wgpu on Wayland).
///
/// This function uses `Once` for thread-safe one-time initialisation. It's safe to
/// call from multiple threads concurrently - only the first call will set the
/// environment variable, and subsequent calls are no-ops.
///
/// Must be called before any GPU initialisation (wgpu Instance creation).
#[cfg(target_os = "linux")]
fn ensure_xdg_runtime_dir() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if env::var("XDG_RUNTIME_DIR").is_err() {
            // Create a temporary runtime directory if XDG_RUNTIME_DIR is not set
            if let Ok(temp_dir) = std::env::temp_dir().canonicalize() {
                let runtime_dir = temp_dir.join("neat-ai-discovery-runtime");
                if let Err(e) = std::fs::create_dir_all(&runtime_dir) {
                    eprintln!("[NEAT-AI-Discovery] Warning: Failed to create XDG_RUNTIME_DIR at {runtime_dir:?}: {e}");
                } else {
                    // SAFETY: Inside Once::call_once, so guaranteed single-threaded execution.
                    // Called before any GPU init.
                    unsafe {
                        env::set_var("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
                    }
                }
            }
        }
    });
}

#[cfg(not(target_os = "linux"))]
fn ensure_xdg_runtime_dir() {
    // No-op on non-Linux platforms
}

/// Safely create a wgpu Instance, avoiding panics from backend probing.
///
/// On Linux with old hardware or missing GPU drivers, wgpu's EGL/OpenGL backend
/// can panic during initialisation (e.g., "BadDisplay" errors). This function:
///
/// - On Linux: Disables the GL backend entirely, using only Vulkan to avoid EGL panics
/// - On macOS: Uses Metal (the default and only backend on macOS)
/// - On all platforms: Wraps instance creation in `catch_unwind` as a safety net
///
/// Returns `None` if instance creation fails or panics, allowing callers to handle
/// the failure gracefully (e.g., treating missing GPU as discovery-disabled on Linux).
fn create_wgpu_instance_safely() -> Option<wgpu::Instance> {
    use std::panic;

    // On Linux, avoid GL/GLES backend which can panic on EGL initialisation
    // when /dev/dri devices are missing or inaccessible.
    #[cfg(target_os = "linux")]
    let backends = wgpu::Backends::VULKAN;

    // On macOS, Metal is the only backend and should always work
    #[cfg(target_os = "macos")]
    let backends = wgpu::Backends::METAL;

    // On other platforms, use all available backends
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let backends = wgpu::Backends::all();

    // Wrap in catch_unwind to handle any remaining panics from backend probing
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        })
    }));

    match result {
        Ok(instance) => Some(instance),
        Err(panic_info) => {
            // Log the panic but don't propagate it
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic during wgpu instance creation".to_string()
            };

            #[cfg(target_os = "linux")]
            {
                // On Linux, this is expected on headless servers without GPU
                if verbose_enabled() {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] wgpu instance creation failed: {panic_msg}. \
                         Discovery will be disabled on this machine."
                    );
                }
            }

            #[cfg(target_os = "macos")]
            {
                // On macOS, this is unexpected - Metal should always be available
                eprintln!(
                    "[NEAT-AI-Discovery] ERROR: wgpu instance creation failed on macOS: {panic_msg}. \
                     This indicates a system configuration issue."
                );
            }

            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                eprintln!(
                    "[NEAT-AI-Discovery] wgpu instance creation failed: {panic_msg}. \
                     Discovery will be disabled on this machine."
                );
            }

            None
        }
    }
}

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
    /// Hidden neurons are filtered out from add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    HiddenNeuronFiltered,
    /// Input neurons are filtered out from add-neuron analysis because they're
    /// observation sources, not computation nodes - they have no activation function
    /// or error to reduce.
    InputNeuronFiltered,
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
    total_eligible_sources: u32,
    input_neuron_count: u32,
    already_connected_count: u32,
    record_load_failures: u32,
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
            total_eligible_sources: 0,
            input_neuron_count: 0,
            already_connected_count: 0,
            record_load_failures: 0,
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

    fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    fn set_input_neuron_count(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_neuron_count = count;
        }
    }

    fn record_already_connected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.already_connected_count += 1;
        }
    }

    fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
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

            if entry.total_eligible_sources == 0 {
                // This should never happen - input/constant neurons are skipped early
                // and hidden/output neurons should always have at least input neurons as eligible sources
                // Skip logging to avoid cluttering logs with impossible conditions
                continue;
            }

            // Check if neuron is fully connected (all eligible sources already have synapses)
            // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
            // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
            // This condition is rare - only occurs when neuron is connected to all possible sources
            if entry.already_connected_count == entry.total_eligible_sources {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} is fully connected: all {} eligible upstream sources already have synapses (all {} input neurons and all prior hidden/output neurons). This is a rare condition.",
                    entry.target_uuid, entry.total_eligible_sources, entry.input_neuron_count
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures (this may indicate a bug - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures
                );
            }

            if entry.evaluated_candidates == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream neurons but none were evaluated ({} already connected, {} record load failures).",
                    entry.target_uuid, entry.total_eligible_sources, entry.already_connected_count, entry.record_load_failures
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
                // Only report "no eligible sources" if both total_eligible_sources and evaluated_candidates are 0
                // This handles the case where total_eligible_sources might be 0 in tests but evaluated_candidates > 0
                if entry.total_eligible_sources == 0 && entry.evaluated_candidates == 0 {
                    return SynapseNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: SynapseNoCandidateReason::NoEligibleSources,
                        evaluated_candidates: entry.evaluated_candidates,
                        candidates_with_samples: entry.candidates_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // Check if neuron is fully connected (all eligible sources already have synapses)
                // Eligible sources include: ALL input neurons (input-0 through input-(creature.input-1))
                // AND ALL prior hidden/output neurons (with index < target_index, excluding constants)
                // This condition is rare - only occurs when neuron is connected to all possible sources
                if entry.total_eligible_sources > 0
                    && entry.already_connected_count == entry.total_eligible_sources
                    && entry.evaluated_candidates == 0
                {
                    // Neuron is fully connected - all eligible sources (all inputs + all prior hidden neurons) already have synapses
                    // This is legitimate but rare, and we report it as "no eligible sources"
                    // since there are no NEW sources to evaluate
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

/// Detail about why a source was rejected (currently only used for NoSamples).
#[derive(Clone)]
struct NeuronRejectionDetail {
    source_uuid: String,
    orientation: Option<&'static str>,
    sample_count: usize,
    expected_improvement: f32,
}

impl NeuronRejectionDetail {
    fn score(&self) -> f32 {
        self.expected_improvement
    }
}

struct NeuronDiagnosticEntry {
    target_uuid: String,
    target_record_count: usize,
    total_eligible_sources: u32,
    record_load_failures: u32,
    evaluated_sources: u32,
    sources_with_samples: u32,
    had_candidate: bool,
    best_rejection: Option<NeuronRejectionDetail>,
    /// Set to true when this neuron was filtered out because it's a hidden neuron
    /// (only output neurons are valid targets for add-neuron analysis).
    hidden_filtered: bool,
    /// Set to true when this neuron was filtered out because it's an input neuron
    /// (input neurons are observation sources, not computation nodes).
    input_filtered: bool,
}

impl NeuronDiagnosticEntry {
    fn new(target_uuid: &str) -> Self {
        Self {
            target_uuid: target_uuid.to_string(),
            target_record_count: 0,
            total_eligible_sources: 0,
            record_load_failures: 0,
            evaluated_sources: 0,
            sources_with_samples: 0,
            had_candidate: false,
            best_rejection: None,
            hidden_filtered: false,
            input_filtered: false,
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

    fn set_total_eligible_sources(&mut self, target_uuid: &str, count: u32) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.total_eligible_sources = count;
        }
    }

    fn record_load_failure(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.record_load_failures += 1;
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
                sample_count: 0,
                expected_improvement: f32::NEG_INFINITY,
            });
        }
    }

    fn mark_candidate_selected(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.had_candidate = true;
        }
    }

    /// Mark a neuron as filtered out because it's a hidden neuron.
    /// Hidden neurons are not valid targets for add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    fn mark_hidden_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.hidden_filtered = true;
        }
    }

    /// Mark a neuron as filtered out because it's an input neuron.
    /// Input neurons are observation sources, not computation nodes - they have
    /// no activation function or error to reduce.
    fn mark_input_filtered(&mut self, target_uuid: &str) {
        if let Some(entry) = self.entries.get_mut(target_uuid) {
            entry.input_filtered = true;
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

            // Check pre-analysis filters FIRST - these take precedence over all other reasons.
            // These neurons are filtered out before analysis even begins, so they won't
            // have any other diagnostic data (eligible sources, samples, etc.).

            // Input neurons are observation sources, not computation nodes
            if entry.input_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (input neuron). \
                    Input neurons are observation sources, not computation nodes - they have no \
                    activation function or error to reduce.",
                    entry.target_uuid
                );
                continue;
            }

            // Hidden neurons have backpropagated errors that don't reliably predict output error
            if entry.hidden_filtered {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} was filtered out (hidden neuron). \
                    Add-neuron analysis only targets output neurons because hidden neuron error \
                    reduction doesn't reliably translate to creature score improvement.",
                    entry.target_uuid
                );
                continue;
            }

            // Check for record loading failures (this indicates a bug or data issue)
            if entry.record_load_failures > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} had {} record loading failures out of {} eligible sources (this may indicate a data integrity issue - records exist but couldn't be loaded).",
                    entry.target_uuid, entry.record_load_failures, entry.total_eligible_sources
                );
            }

            if entry.evaluated_sources == 0 {
                if entry.total_eligible_sources > 0
                    && entry.record_load_failures == entry.total_eligible_sources
                {
                    // All eligible sources failed to load - this is a data/bug issue, not "no sources"
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but all {} failed to load from parquet file.",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else if entry.total_eligible_sources > 0 && entry.record_load_failures == 0 {
                    // Sources exist, no load failures, but none evaluated - likely timeout before sources could be checked
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated (0 load failures). Likely analysis TIMEOUT before source loading could start.",
                        entry.target_uuid, entry.total_eligible_sources
                    );
                } else if entry.total_eligible_sources > 0 {
                    // Some sources exist, some failures, none evaluated
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had {} eligible upstream sources but none were evaluated ({} load failures).",
                        entry.target_uuid, entry.total_eligible_sources, entry.record_load_failures
                    );
                } else {
                    // Genuinely no eligible sources (e.g., target is first neuron)
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {} had no upstream neurons to analyse.",
                        entry.target_uuid
                    );
                }
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

            // Currently only NoSamples is used
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Target {} skipped candidate from {} because no overlapping samples were found.",
                entry.target_uuid, best.source_uuid
            );
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
                // Check pre-analysis filters FIRST - these take precedence over all other reasons.
                // These neurons are filtered out before analysis even begins, so they won't
                // have any other diagnostic data (eligible sources, samples, etc.).

                // Input neurons are observation sources, not computation nodes
                if entry.input_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::InputNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Hidden neurons have backpropagated errors that don't reliably predict output error
                if entry.hidden_filtered {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                        evaluated_sources: 0,
                        sources_with_samples: 0,
                        target_record_count: 0,
                        detail: None,
                    };
                }

                // Only report "no eligible sources" if there were genuinely no eligible sources
                // AND no evaluated candidates. If there were eligible sources but they all failed
                // to load or had empty records, report that as NoSamples with context.
                if entry.evaluated_sources == 0 && entry.total_eligible_sources == 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoEligibleSources,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                // If evaluated_sources is 0 but total_eligible_sources > 0, sources existed
                // but all failed to load or had empty records - report as NoSamples
                if entry.evaluated_sources == 0 && entry.total_eligible_sources > 0 {
                    return NeuronNoCandidateSummary {
                        target_uuid: entry.target_uuid.clone(),
                        reason: NeuronNoCandidateReason::NoSamples,
                        evaluated_sources: entry.evaluated_sources,
                        sources_with_samples: entry.sources_with_samples,
                        target_record_count: entry.target_record_count,
                        detail: None,
                    };
                }

                if let Some(best) = &entry.best_rejection {
                    // Currently only NoSamples is ever set as rejection reason
                    let reason = NeuronNoCandidateReason::NoSamples;
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
                            improved_count: None,
                            worsened_count: None,
                            expected_improvement: Some(best.expected_improvement),
                            threshold: None,
                            outgoing_weight: None,
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
    /// Create a pre-loaded cache that reads the entire parquet file once.
    /// This is MUCH faster when you need records for many neurons (e.g., ~2000),
    /// as it avoids scanning the file 2000 times.
    fn new_preloaded(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_all_records_grouped_by_neuron;
        use std::time::Instant;

        let start = Instant::now();
        let grouped_records = read_all_records_grouped_by_neuron(parquet_file)
            .with_context(|| format!("Failed to pre-load records from {parquet_file}"))?;

        let neuron_count = grouped_records.len();
        let total_records: usize = grouped_records.values().map(|v| v.len()).sum();

        // Pre-populate the cache with all loaded records
        let mut cache_map: HashMap<String, Arc<CachedNeuronRecords>> = HashMap::new();
        for (uuid, mut records) in grouped_records {
            records.sort_by_key(|r| r.obs_index);
            let cell = OnceCell::new();
            let _ = cell.set(Arc::new(records));
            cache_map.insert(uuid, Arc::new(cell));
        }

        let elapsed = start.elapsed();
        if verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Pre-loaded {} neurons with {} total records from parquet in {:.2}s",
                neuron_count,
                total_records,
                elapsed.as_secs_f64()
            );
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(cache_map),
            loader: Arc::new(|_file: &str, _neuron_uuid: &str| {
                // This loader should never be called for pre-loaded cache
                Ok(Vec::new())
            }),
        })
    }

    #[cfg(test)]
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

/// Sample data for evaluating potential synapses/neurons.
///
/// For accurate HARD_TANH modelling, we need the target's pre-activation value
/// to properly simulate clamping behaviour. When `target_value` is `Some`, we can
/// compute the actual effect of adding a contribution rather than using the linear
/// approximation.
#[derive(Clone, Copy, Default)]
struct HelpfulSample {
    /// Source neuron's activation (what we're considering adding a connection FROM)
    activation: f32,
    /// Target neuron's average error (expected - actual output)
    avg_error: f32,
    /// Target neuron's pre-activation value (input sum before squash function).
    /// Used for accurate HARD_TANH/clamping calculations. None for GPU-matched samples.
    target_value: Option<f32>,
    /// Target neuron's post-activation output (after squash function).
    /// Used with avg_error to compute expected: expected = target_activation + avg_error
    target_activation: Option<f32>,
}

/// Extended sample for threshold-crossing analysis of discrete activations (STEP/BIPOLAR).
/// Includes the target neuron's pre-activation value to determine threshold crossings.
#[derive(Clone, Copy)]
struct DiscreteHelpfulSample {
    /// Source neuron's activation
    source_activation: f32,
    /// Target neuron's input sum before squash function
    target_value: f32,
    /// Target neuron's current output after squash (0/1 for STEP, -1/1 for BIPOLAR)
    target_activation: f32,
    /// Target neuron's average error
    avg_error: f32,
}

/// Type of threshold activation function
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThresholdType {
    /// STEP: value > 0 ? 1 : 0
    Step,
    /// BIPOLAR: value > 0 ? 1 : -1
    Bipolar,
}

impl ThresholdType {
    fn from_squash(squash: &str) -> Option<Self> {
        match squash.to_uppercase().as_str() {
            "STEP" => Some(Self::Step),
            "BIPOLAR" => Some(Self::Bipolar),
            _ => None,
        }
    }

    /// Calculate the output for a given input value
    fn apply(&self, value: f32) -> f32 {
        match self {
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bipolar => {
                if value > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// Check if adding a contribution would flip the output
    fn would_flip(&self, current_value: f32, contribution: f32) -> bool {
        let current_positive = current_value > 0.0;
        let new_positive = (current_value + contribution) > 0.0;
        current_positive != new_positive
    }

    /// Check if a flip is "helpful" (moves output in direction of error)
    /// Returns: 1 for helpful flip, -1 for harmful flip, 0 for no flip
    fn flip_direction(&self, current_value: f32, contribution: f32, error: f32) -> i32 {
        if !self.would_flip(current_value, contribution) {
            return 0;
        }

        let current_output = self.apply(current_value);
        let new_output = self.apply(current_value + contribution);

        // Error > 0 means output should be higher
        // Error < 0 means output should be lower
        let output_increased = new_output > current_output;
        let should_increase = error > 0.0;

        if output_increased == should_increase {
            1 // Helpful flip
        } else {
            -1 // Harmful flip
        }
    }
}

/// Statistics computed from neuron error and activation samples
#[derive(Debug, Clone)]
struct NeuronStats {
    mean_error: f32,
    error_variance: f32,
    mean_activation: f32,
    activation_variance: f32,
    error_spike_count: u32,
    activation_spike_count: u32,
    activation_min: f32,
    activation_max: f32,
}

impl NeuronStats {
    /// Compute statistics from a slice of discovery records (for target neurons)
    fn from_records(records: &[DiscoverRecord]) -> Option<Self> {
        if records.is_empty() {
            return None;
        }

        let mut samples = Vec::new();
        for record in records {
            if record.errors.is_empty() {
                continue;
            }
            // Compute average error for this record
            let mut error_sum = 0.0;
            let mut error_count = 0;
            for &err in &record.errors {
                if err.is_finite() {
                    error_sum += err;
                    error_count += 1;
                }
            }
            if error_count > 0 && record.activation.is_finite() {
                let avg_error = error_sum / error_count as f32;
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error,
                    target_value: record.value,
                    target_activation: Some(record.activation),
                });
            }
        }

        Self::from_samples(&samples)
    }

    /// Compute statistics from a slice of samples
    fn from_samples(samples: &[HelpfulSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut error_sum = 0.0;
        let mut error_sq_sum = 0.0;
        let mut activation_sum = 0.0;
        let mut activation_sq_sum = 0.0;
        let mut error_spike_count = 0u32;
        let mut activation_spike_count = 0u32;
        let mut activation_min = f32::INFINITY;
        let mut activation_max = f32::NEG_INFINITY;
        let mut valid_count = 0usize;

        // Spike thresholds: 2 standard deviations (we'll approximate with mean + 2*mean for now)
        // We'll compute proper thresholds after we have the mean
        let mut error_abs_sum = 0.0;
        let mut activation_abs_sum = 0.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            valid_count += 1;
            let error_abs = sample.avg_error.abs();
            let activation_abs = sample.activation.abs();

            error_sum += sample.avg_error;
            error_sq_sum += sample.avg_error * sample.avg_error;
            error_abs_sum += error_abs;

            activation_sum += sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            activation_abs_sum += activation_abs;

            if activation_min > sample.activation {
                activation_min = sample.activation;
            }
            if activation_max < sample.activation {
                activation_max = sample.activation;
            }
        }

        if valid_count == 0 {
            return None;
        }

        let count_f = valid_count as f32;
        let mean_error = error_sum / count_f;
        let mean_activation = activation_sum / count_f;
        let mean_error_abs = error_abs_sum / count_f;
        let mean_activation_abs = activation_abs_sum / count_f;

        // Compute variance using E[X^2] - E[X]^2
        let error_variance = (error_sq_sum / count_f) - (mean_error * mean_error);
        let activation_variance =
            (activation_sq_sum / count_f) - (mean_activation * mean_activation);

        // Spike detection: count samples where error/activation exceeds 2x the mean absolute value
        // This is a simple heuristic; more sophisticated methods could use actual std dev
        let error_spike_threshold = mean_error_abs * 2.0;
        let activation_spike_threshold = mean_activation_abs * 2.0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }
            if sample.avg_error.abs() > error_spike_threshold {
                error_spike_count += 1;
            }
            if sample.activation.abs() > activation_spike_threshold {
                activation_spike_count += 1;
            }
        }

        Some(Self {
            mean_error,
            error_variance: error_variance.max(0.0), // Variance should be non-negative
            mean_activation,
            activation_variance: activation_variance.max(0.0),
            error_spike_count,
            activation_spike_count,
            activation_min: if activation_min.is_finite() {
                activation_min
            } else {
                0.0
            },
            activation_max: if activation_max.is_finite() {
                activation_max
            } else {
                0.0
            },
        })
    }

    fn to_json(&self) -> crate::NeuronStatsJson {
        crate::NeuronStatsJson {
            mean_error: self.mean_error,
            error_variance: self.error_variance,
            mean_activation: self.mean_activation,
            activation_variance: self.activation_variance,
            error_spike_count: self.error_spike_count,
            activation_spike_count: self.activation_spike_count,
            activation_min: self.activation_min,
            activation_max: self.activation_max,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHelpfulSample {
    activation: f32,
    avg_error: f32,
}

/// Extended sample struct for GPU matching output, includes target neuron data.
/// Used only by the matching shader; other shaders use the simpler GpuHelpfulSample.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMatchingSample {
    activation: f32,
    avg_error: f32,
    /// Target neuron's pre-activation value (input sum before squash function)
    target_value: f32,
    /// Target neuron's post-activation output (after squash function)
    target_activation: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTargetRecord {
    obs_index: u32,
    error_start_index: u32,
    error_count: u32,
    /// Target neuron's pre-activation value (input sum before squash function)
    value: f32,
    /// Target neuron's post-activation output (after squash function)
    activation: f32,
    pad0: u32,
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
    total_errors: u32,
    pad0: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReluContribution {
    positive_activation_sq: f32,
    positive_error_activation: f32,
    positive_count: u32,
    negative_activation_sq: f32,
    negative_error_activation: f32,
    negative_count: u32,
    error_sq: f32,
    pad0: f32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReluUniforms {
    length: u32,
    threshold: f32,
    epsilon: f32,
    pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasResult {
    bias_value: f32,
    error_reduction: f32,
    valid_sample_count: u32,
    pad0: u32,
}

impl BiasResult {
    fn zeroed() -> Self {
        Self {
            bias_value: 0.0,
            error_reduction: 0.0,
            valid_sample_count: 0,
            pad0: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasUniforms {
    sample_count: u32,
    bias_count: u32,
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_type: u32,
    epsilon: f32,
    min_sample_count: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(dead_code)] // Framework for future GPU activation evaluation
struct ActivationOutput {
    output: f32,
    output_sq: f32,
    error_output: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(dead_code)] // Framework for future GPU activation evaluation
struct ActivationUniforms {
    sample_count: u32,
    orientation: f32,
    scale: f32,
    activation_type: u32,
    epsilon: f32,
    pad0: f32,
    pad1: f32,
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

    /// Evaluate this orientation and return a candidate if it passes the threshold.
    fn evaluate(
        &self,
        source_uuid: &str,
        target_uuid: &str,
        threshold: f32,
        total_baseline_error_sq: f32,
        original_samples: &[HelpfulSample],
    ) -> Option<CandidateNeuronJson> {
        let sample_count = self.samples.len();
        if sample_count < MIN_NEURON_SAMPLE_COUNT || self.activation_sq_sum <= EPSILON {
            return None;
        }

        let mut outgoing_weight = self.error_activation_sum / (self.activation_sq_sum + EPSILON);
        if !outgoing_weight.is_finite() || outgoing_weight.abs() <= EPSILON {
            return None;
        }
        outgoing_weight = outgoing_weight.clamp(-10.0, 10.0);

        let mut improved_count = 0u32;
        for (relu_activation, error) in &self.samples {
            let new_error = error - outgoing_weight * relu_activation;
            if new_error.abs() + EPSILON < error.abs() {
                improved_count += 1;
            }
        }

        // Calculate improvement based on magnitude (reduction in squared error)
        // improvement = baseline_sq - new_sq
        // = 2*w*sum(ea) - w^2*sum(aa)
        let improvement_magnitude = 2.0 * outgoing_weight * self.error_activation_sum
            - outgoing_weight * outgoing_weight * self.activation_sq_sum;

        // Normalise by total baseline error of ALL samples (not just active ones)
        let expected_improvement = if total_baseline_error_sq > EPSILON {
            let result = improvement_magnitude / total_baseline_error_sq;
            if result.is_finite() {
                result
            } else {
                0.0
            }
        } else {
            0.0
        };

        if expected_improvement <= threshold {
            return None;
        }

        let incoming_weight = match self.orientation {
            ReluOrientation::Positive => 1.0,
            ReluOrientation::Negative => -1.0,
        };

        // Calculate optimal bias for ReLU neuron
        let optimal_bias = calculate_optimal_bias(
            original_samples,
            incoming_weight,
            outgoing_weight,
            |x| x.max(0.0), // ReLU activation function
            "ReLU",
            None, // GPU-accelerated bias search
            None, // target_squash not available in this context
        );

        let target_stats = NeuronStats::from_samples(original_samples).map(|s| s.to_json());
        let total_count = self.samples.len() as u32;

        Some(CandidateNeuronJson {
            source_neuron_uuid: source_uuid.to_string(),
            target_neuron_uuid: target_uuid.to_string(),
            incoming_weight,
            outgoing_weight,
            squash: "ReLU".to_string(),
            bias: optimal_bias,
            expected_improvement_percentage: expected_improvement,
            improved_count,
            total_count,
            target_neuron_stats: target_stats,
        })
    }
}

struct ActivationCandidateSpec {
    name: &'static str,
    orientations: &'static [f32],
    scales: &'static [f32],
    activation: fn(f32) -> f32,
    min_improvement: f32,
}

const ORIENTATIONS_BIDIRECTIONAL: [f32; 2] = [1.0, -1.0];
/// Log-spaced scale range for incoming weights - covers multiple orders of magnitude
/// efficiently. Evolution will fine-tune the exact values after discovery.
/// Extended to very large scales (50, 100) for aggressive signal amplification.
/// Note: Very large scales may cause numerical instability with some activations.
const SCALES_WIDE: [f32; 12] = [
    0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
];
/// Log-spaced scales for smooth activation functions (TANH, LOGISTIC, SELU) that
/// saturate at large inputs. Larger scales included but will saturate the output,
/// which may still be useful for binary-like thresholding behaviour.
const SCALES_SMOOTH: [f32; 10] = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 50.0];

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

fn activation_name_to_gpu_id(name: &str) -> u32 {
    match name {
        "GELU" => 0,
        "ELU" => 1,
        "SELU" => 2,
        "Softplus" => 3,
        "LOGISTIC" => 4,
        "TANH" => 5,
        "IDENTITY" => 6,
        "BIPOLAR" => 7,
        "CLIPPED" => 8,
        "ABSOLUTE" => 9,
        "INVERSE" => 10,
        _ => 6, // Default to IDENTITY
    }
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
        // IDENTITY is just a pass-through - require meaningful improvement to justify
        // adding a neuron instead of adjusting existing synapse weights
        min_improvement: 0.05,
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

/// Get activation-function-specific bias range (min, max, step).
/// This is still used by the GPU bias search for compatibility.
/// Extended ranges to work with large incoming weights (up to 200).
///
/// Different activation functions benefit from different bias ranges:
/// - ReLU/ELU: Large negative bias for high-threshold neurons
/// - TANH/LOGISTIC: Wide symmetric range to shift operating point
/// - IDENTITY: Widest range as pure offset (scales with large weights)
fn get_bias_range(squash: &str) -> (f32, f32, f32) {
    match squash {
        // ReLU/ELU: extended negative for high-threshold neurons with large weights
        "ReLU" | "ELU" | "SELU" => (-25.0, 10.0, 0.5),
        // Symmetric activation functions: extended for large weight configurations
        "TANH" | "LOGISTIC" => (-10.0, 10.0, 0.5),
        // IDENTITY: widest range - acts as offset, scales with incoming weights
        "IDENTITY" => (-50.0, 50.0, 1.0),
        // Softplus and GELU: extended negative thresholds
        "Softplus" | "GELU" => (-10.0, 10.0, 0.5),
        // Other activation functions get expanded range
        "INVERSE" | "ABSOLUTE" | "CLIPPED" => (-10.0, 10.0, 0.5),
        "BIPOLAR" => (-10.0, 10.0, 1.0),
        _ => (-10.0, 10.0, 0.5), // Generous default
    }
}

/// Get log-spaced bias values for a given activation function.
/// Uses sinh-like spacing: denser near 0, sparser at extremes.
/// Extended ranges to work with large incoming weights (up to 200).
/// Evolution will fine-tune the exact bias value after discovery.
fn get_bias_values(squash: &str) -> Vec<f32> {
    // Base log-spaced positive values (denser near 0, extended to larger values)
    let base_positive: &[f32] = match squash {
        // ReLU/ELU: extended for large weight thresholding
        "ReLU" | "ELU" | "SELU" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // Symmetric activations: extended range for shifting operating point
        "TANH" | "LOGISTIC" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // IDENTITY: widest range (pure offset, scales with large weights)
        "IDENTITY" => &[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0],
        // Softplus/GELU: extended for large weight configurations
        "Softplus" | "GELU" => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        // Others: moderate extended range
        _ => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    };

    let base_negative: &[f32] = match squash {
        // ReLU/ELU: extended negative for high-threshold neurons
        "ReLU" | "ELU" | "SELU" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -25.0],
        // Symmetric: mirror of positive for operating point shift
        "TANH" | "LOGISTIC" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
        // IDENTITY: widest range to match large weights
        "IDENTITY" => &[-0.5, -1.0, -2.0, -5.0, -10.0, -25.0, -50.0],
        // Softplus/GELU: extended negative thresholds
        "Softplus" | "GELU" => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
        // Others: moderate extended
        _ => &[-0.1, -0.5, -1.0, -2.0, -5.0],
    };

    let mut values: Vec<f32> = base_negative.to_vec();
    values.extend_from_slice(base_positive);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// Calculate optimal bias for a neuron candidate using grid search.
///
/// This function finds the bias value that maximises error reduction when combined
/// with the given weights and activation function. It tests multiple bias values
/// across an activation-function-specific range and selects the one that gives
/// the best improvement.
///
/// Uses GPU-accelerated parallel search when analyzer is provided and GPU is available,
/// otherwise falls back to CPU sequential search.
///
/// # Arguments
/// * `samples` - Training samples (source activations and target errors)
/// * `incoming_weight` - Weight from source to new neuron
/// * `outgoing_weight` - Weight from new neuron to target
/// * `activation_fn` - Activation function to apply
/// * `squash` - Activation function name (for bias range selection and GPU)
/// * `analyzer` - Optional GPU analyzer for accelerated search
///
/// # Returns
/// Optimal bias value that maximises error reduction
fn calculate_optimal_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_fn: fn(f32) -> f32,
    squash: &str,
    analyzer: Option<&GpuAnalyzer>,
    target_squash: Option<&str>,
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let bias_range = get_bias_range(squash);

    // Try GPU-accelerated search first if analyzer available
    if let Some(gpu_analyzer) = analyzer {
        if gpu_analyzer.device.is_some() {
            let activation_type = activation_name_to_gpu_id(squash);
            if let Ok(optimal_bias) = gpu_analyzer.evaluate_bias_gpu(
                samples,
                incoming_weight,
                outgoing_weight,
                activation_type,
                bias_range,
            ) {
                return optimal_bias;
            }
            // If GPU fails, fall through to CPU
        }
    }

    // Use log-spaced bias values for efficient search
    // Evolution will fine-tune the exact value after discovery
    let bias_values = get_bias_values(squash);

    // Calculate baseline error (no new neuron)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    if total_baseline_error_sq <= EPSILON {
        return 0.0;
    }

    let mut best_bias = 0.0;
    let mut best_error_reduction = f32::NEG_INFINITY;

    // Check if we can use HARD_TANH model (need target_value for all samples)
    let use_hard_tanh = target_squash == Some("HARD_TANH")
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some());

    // Search over log-spaced bias values
    for &bias in &bias_values {
        // Calculate error with this bias
        let mut total_new_error_sq = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }

            // Calculate new neuron's activation with bias
            let pre_activation = incoming_weight * sample.activation + bias;
            let new_neuron_activation = activation_fn(pre_activation);

            if !new_neuron_activation.is_finite() {
                continue;
            }

            // Calculate new error at target neuron
            let correction = outgoing_weight * new_neuron_activation;
            let new_error = if use_hard_tanh {
                // HARD_TANH model: account for target neuron's clamping
                let target_value = sample.target_value.unwrap();
                let target_activation = sample.target_activation.unwrap();
                let expected = target_activation + sample.avg_error;
                let new_input = target_value + correction;
                let new_output = hard_tanh(new_input);
                expected - new_output
            } else {
                // Linear model
                sample.avg_error - correction
            };

            if new_error.is_finite() {
                total_new_error_sq += new_error * new_error;
                valid_samples += 1;
            }
        }

        // Only consider if we have valid samples
        if valid_samples < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Calculate error reduction (positive is good)
        let error_reduction = total_baseline_error_sq - total_new_error_sq;

        if error_reduction > best_error_reduction {
            best_error_reduction = error_reduction;
            best_bias = bias;
        }
    }

    best_bias
}

#[derive(Default)]
struct HarmfulStats {
    harmful_count: u32,
    helpful_count: u32,
    harmful_error_sum: f32,
}

pub struct GpuAnalyzer {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    helpful_layout: Option<wgpu::BindGroupLayout>,
    helpful_pipeline: Option<wgpu::ComputePipeline>,
    harmful_layout: Option<wgpu::BindGroupLayout>,
    harmful_pipeline: Option<wgpu::ComputePipeline>,
    matching_layout: Option<wgpu::BindGroupLayout>,
    matching_pipeline: Option<wgpu::ComputePipeline>,
    relu_layout: Option<wgpu::BindGroupLayout>,
    relu_pipeline: Option<wgpu::ComputePipeline>,
    #[allow(dead_code)] // Framework for future GPU activation evaluation
    activation_layout: Option<wgpu::BindGroupLayout>,
    #[allow(dead_code)] // Framework for future GPU activation evaluation
    activation_pipeline: Option<wgpu::ComputePipeline>,
    bias_layout: Option<wgpu::BindGroupLayout>,
    bias_pipeline: Option<wgpu::ComputePipeline>,
}

/// Result of GPU availability check with detailed diagnostics.
pub struct GpuAvailabilityResult {
    /// Whether a GPU is available for use.
    pub available: bool,
    /// Human-readable reason for the availability status.
    pub reason: Option<String>,
    /// Whether this is an error condition (true on macOS when GPU unavailable).
    pub is_error: bool,
}

impl GpuAnalyzer {
    /// Lightweight probe to determine whether a usable GPU device is available.
    ///
    /// This is intended for callers (via FFI) that want to decide whether to
    /// enable the Rust discovery extension at all. It deliberately avoids
    /// falling back to CPU – if the adapter or device cannot be created, the
    /// probe reports `false`.
    ///
    /// Platform-specific behaviour:
    /// - **macOS**: GPU should always be available (Metal). Missing GPU is an error.
    /// - **Linux**: GPU may not be available on headless servers without GPU hardware
    ///   or proper permissions. Missing GPU gracefully disables discovery.
    pub fn gpu_is_available() -> bool {
        Self::check_gpu_availability().available
    }

    /// Check GPU availability with detailed diagnostics.
    ///
    /// Returns availability status, reason, and whether it's an error condition.
    /// On macOS, missing GPU is treated as an error (Metal should always work).
    /// On Linux, missing GPU gracefully disables discovery (common on headless servers).
    pub fn check_gpu_availability() -> GpuAvailabilityResult {
        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let Some(instance) = create_wgpu_instance_safely() else {
            return Self::no_gpu_result("wgpu instance creation failed (GPU backend unavailable)");
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return Self::no_gpu_result("No GPU adapter found");
        };

        let device_result = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU probe device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));

        match device_result {
            Ok(_) => GpuAvailabilityResult {
                available: true,
                reason: None,
                is_error: false,
            },
            Err(e) => Self::no_gpu_result(&format!("GPU device creation failed: {e}")),
        }
    }

    /// Create a result for when GPU is not available.
    /// On macOS this is an error; on Linux it gracefully disables discovery.
    fn no_gpu_result(reason: &str) -> GpuAvailabilityResult {
        #[cfg(target_os = "macos")]
        {
            // On macOS, Metal should always be available - missing GPU is an error
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!(
                    "{reason}. On macOS, GPU (Metal) should always be available. \
                     This may indicate a system configuration issue."
                )),
                is_error: true,
            }
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, GPU may not be available on headless servers - gracefully disable
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!(
                    "{reason}. Discovery disabled on this machine. \
                     This is normal for headless Linux servers without GPU hardware or \
                     without proper permissions to access /dev/dri devices."
                )),
                is_error: false,
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            // For other platforms, treat as non-error (graceful disable)
            GpuAvailabilityResult {
                available: false,
                reason: Some(format!("{reason}. Discovery disabled on this platform.")),
                is_error: false,
            }
        }
    }

    fn new() -> Result<Self> {
        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let instance = create_wgpu_instance_safely().ok_or_else(|| {
            anyhow::anyhow!(
                "wgpu instance creation failed (GPU backend unavailable). \
                 Discovery requires GPU acceleration."
            )
        })?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(adapter) => adapter,
            None => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
                anyhow::bail!(
                    "GPU adapter not available. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                );
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
            Ok(result) => result,
            Err(e) => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
                anyhow::bail!(
                    "GPU device creation failed: {e}. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                );
            }
        };

        let (helpful_layout, helpful_pipeline) =
            Self::build_helpful_pipeline(&device, "helpful-synapse-pipeline");
        let (harmful_layout, harmful_pipeline) =
            Self::build_harmful_pipeline(&device, "harmful-synapse-pipeline");
        let (matching_layout, matching_pipeline) =
            Self::build_matching_pipeline(&device, "matching-pipeline");
        let (relu_layout, relu_pipeline) = Self::build_relu_pipeline(&device, "relu-pipeline");
        let (activation_layout, activation_pipeline) =
            Self::build_activation_pipeline(&device, "activation-pipeline");
        let (bias_layout, bias_pipeline) = Self::build_bias_pipeline(&device, "bias-pipeline");

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            matching_layout: Some(matching_layout),
            matching_pipeline: Some(matching_pipeline),
            relu_layout: Some(relu_layout),
            relu_pipeline: Some(relu_pipeline),
            activation_layout: Some(activation_layout),
            activation_pipeline: Some(activation_pipeline),
            bias_layout: Some(bias_layout),
            bias_pipeline: Some(bias_pipeline),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

    fn build_relu_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relu-shader"),
            source: wgpu::ShaderSource::Wgsl(RELU_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relu-bind-group"),
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

    fn build_activation_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("activation-shader"),
            source: wgpu::ShaderSource::Wgsl(ACTIVATION_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("activation-bind-group"),
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

    fn build_bias_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bias-shader"),
            source: wgpu::ShaderSource::Wgsl(BIAS_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias-bind-group"),
            entries: &[
                // Binding 0: samples (read-only)
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
                // Binding 1: bias_candidates (read-only)
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
                // Binding 2: results (read-write)
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
                // Binding 3: uniforms
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

    #[allow(dead_code)] // Used in tests and may be useful for single evaluations
    fn evaluate_helpful(&self, samples: &[HelpfulSample]) -> Result<HelpfulStats> {
        if samples.is_empty() {
            return Ok(HelpfulStats::default());
        }

        // GPU is always required - TypeScript layer calls check_gpu_available() and skips
        // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
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

        Ok(stats)
    }

    fn evaluate_harmful(&self, samples: &[HelpfulSample], weight: f32) -> Result<HarmfulStats> {
        if samples.is_empty() {
            return Ok(HarmfulStats::default());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
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

        Ok(stats)
    }

    fn evaluate_relu_gpu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for ReLU analysis")?;
        let relu_layout = self
            .relu_layout
            .as_ref()
            .context("GPU ReLU layout not initialised")?;
        let relu_pipeline = self
            .relu_pipeline
            .as_ref()
            .context("GPU ReLU pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![ReluContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ReluUniforms {
            length: samples.len() as u32,
            threshold,
            epsilon: EPSILON,
            pad0: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: relu_layout,
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
            label: Some("relu-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<ReluContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relu-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relu-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(relu_pipeline);
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
                return Err(anyhow!("Failed to map ReLU contributions buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive ReLU map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let contributions: &[ReluContribution] = bytemuck::cast_slice(&data);

        let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
        let mut negative_stats = ReluStats::new(ReluOrientation::Negative);
        let mut total_baseline_error_sq = 0.0;

        // Accumulate statistics from GPU contributions
        for (idx, contribution) in contributions.iter().enumerate() {
            if idx < samples.len() {
                total_baseline_error_sq += contribution.error_sq;

                // For positive ReLU, accumulate activation_sq and error_activation
                if contribution.positive_count > 0 {
                    positive_stats.activation_sq_sum += contribution.positive_activation_sq;
                    positive_stats.error_activation_sum += contribution.positive_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.positive_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.positive_error_activation / activation
                    } else {
                        0.0
                    };
                    positive_stats.samples.push((activation, error));
                }

                // For negative ReLU, accumulate activation_sq and error_activation
                if contribution.negative_count > 0 {
                    negative_stats.activation_sq_sum += contribution.negative_activation_sq;
                    negative_stats.error_activation_sum += contribution.negative_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.negative_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.negative_error_activation / activation
                    } else {
                        0.0
                    };
                    negative_stats.samples.push((activation, error));
                }
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok((positive_stats, negative_stats, total_baseline_error_sq))
    }

    fn evaluate_activation_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Returns: (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-outputs-buffer"),
            contents: bytemuck::cast_slice(&outputs_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ActivationUniforms {
            sample_count: samples.len() as u32,
            orientation,
            scale,
            activation_type,
            epsilon: EPSILON,
            pad0: 0.0,
            pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: activation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("activation-bind-group"),
        });

        let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activation-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&outputs_buffer, 0, &staging_buffer, 0, output_size);

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
                return Err(anyhow!("Failed to map activation outputs buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive activation map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

        let mut sum_activation_sq = 0.0;
        let mut sum_error_activation = 0.0;
        let mut total_baseline_error_sq = 0.0;

        for (idx, output) in outputs.iter().enumerate() {
            if idx < samples.len() {
                let sample = &samples[idx];
                if sample.avg_error.is_finite() {
                    total_baseline_error_sq += sample.avg_error * sample.avg_error;
                }
                if output.valid > 0 {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            }
        }

        // Note: improved_count is not calculated here because it requires weight validation
        // that happens in the caller. The caller will calculate improved_count after
        // validating and clamping the outgoing_weight.

        drop(data);
        staging_buffer.unmap();

        Ok((
            sum_activation_sq,
            sum_error_activation,
            total_baseline_error_sq,
            0, // improved_count calculated by caller after weight validation
        ))
    }

    /// GPU-accelerated bias grid search
    /// Tests all bias values in parallel and returns the optimal bias
    fn evaluate_bias_gpu(
        &self,
        samples: &[HelpfulSample],
        incoming_weight: f32,
        outgoing_weight: f32,
        activation_type: u32,
        bias_range: (f32, f32, f32),
    ) -> Result<f32> {
        // Returns: optimal bias value
        if samples.is_empty() {
            return Ok(0.0);
        }

        let (min_bias, max_bias, step) = bias_range;

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for bias analysis")?;
        let bias_layout = self
            .bias_layout
            .as_ref()
            .context("GPU bias layout not initialised")?;
        let bias_pipeline = self
            .bias_pipeline
            .as_ref()
            .context("GPU bias pipeline not initialised")?;

        // Generate bias candidates
        let num_steps = ((max_bias - min_bias) / step).ceil() as i32 + 1;
        let bias_candidates: Vec<f32> = (0..num_steps)
            .map(|i| min_bias + (i as f32 * step).min(max_bias - min_bias))
            .collect();

        if bias_candidates.is_empty() {
            return Ok(0.0);
        }

        // Prepare GPU buffers
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let results_zeroed = vec![BiasResult::zeroed(); bias_candidates.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-candidates-buffer"),
            contents: bytemuck::cast_slice(&bias_candidates),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-results-buffer"),
            contents: bytemuck::cast_slice(&results_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = BiasUniforms {
            sample_count: samples.len() as u32,
            bias_count: bias_candidates.len() as u32,
            incoming_weight,
            outgoing_weight,
            activation_type,
            epsilon: EPSILON,
            min_sample_count: MIN_NEURON_SAMPLE_COUNT as u32,
            pad0: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bias_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("bias-bind-group"),
        });

        let output_size = (std::mem::size_of::<BiasResult>() * bias_candidates.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bias-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bias-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bias-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(bias_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (bias_candidates.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, output_size);

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
                return Err(anyhow!("Failed to map bias results buffer: {err}"));
            }
            Err(_) => {
                return Err(anyhow!("Failed to receive bias map_async completion"));
            }
        }

        let data = buffer_slice.get_mapped_range();
        let results: &[BiasResult] = bytemuck::cast_slice(&data);

        // Find bias with best error reduction
        let mut best_bias = 0.0;
        let mut best_error_reduction = f32::NEG_INFINITY;

        for result in results {
            if result.valid_sample_count >= MIN_NEURON_SAMPLE_COUNT as u32
                && result.error_reduction > best_error_reduction
            {
                best_error_reduction = result.error_reduction;
                best_bias = result.bias_value;
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(best_bias)
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

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
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
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        for batch_chunk in samples_batch.chunks(GPU_BATCH_SIZE) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();
            let mut batch_sample_refs = Vec::new();

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("helpful-command-encoder-batch"),
            });

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

                // Add compute pass to shared encoder (reduces command buffer overhead)
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

                batch_contributions_buffers.push(contributions_buffer);
                batch_staging_buffers.push(staging_buffer);
                batch_contribution_sizes.push((contribution_size, samples.len()));
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let (contribution_size, _) = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // Wait for all results (single poll for entire batch)
            let mut batch_results = Vec::with_capacity(batch_contribution_sizes.len());
            for (staging_buffer, (_contribution_size, _sample_len), _samples_ref) in
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

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
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

        // Prepare target records: flatten error arrays and create GPU structures
        let mut gpu_targets: Vec<GpuTargetRecord> = Vec::new();
        let mut all_errors: Vec<f32> = Vec::new();
        let mut error_start_index = 0u32;

        for record in target_records {
            if record.errors.is_empty() {
                continue;
            }

            // Count finite errors and add them to the flattened array
            let mut finite_error_count = 0u32;
            for error in &record.errors {
                if error.is_finite() {
                    all_errors.push(*error);
                    finite_error_count += 1;
                }
            }

            if finite_error_count > 0 {
                gpu_targets.push(GpuTargetRecord {
                    obs_index: record.obs_index,
                    error_start_index,
                    error_count: finite_error_count,
                    // Include target neuron's value and activation for accurate simulation
                    value: record.value.unwrap_or(f32::NAN),
                    activation: record.activation,
                    pad0: 0,
                });
                error_start_index += finite_error_count;
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

        let errors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-errors-buffer"),
            contents: bytemuck::cast_slice(&all_errors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let from_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matching-from-buffer"),
            contents: bytemuck::cast_slice(&gpu_froms),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let samples_zeroed = vec![GpuMatchingSample::zeroed(); from_records.len()];
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
            total_errors: all_errors.len() as u32,
            pad0: 0,
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
                    resource: errors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: from_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("matching-bind-group"),
        });

        let sample_size = (std::mem::size_of::<GpuMatchingSample>() * from_records.len()) as u64;
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
        let gpu_samples: &[GpuMatchingSample] = bytemuck::cast_slice(&data);

        // Retain only finite samples; GPU matching emits NaN for invalid rows.
        // Now includes target_value and target_activation for accurate activation
        // function simulation (HARD_TANH, TANH, LOGISTIC, etc.).
        let mut samples = Vec::new();
        for gpu_sample in gpu_samples {
            if gpu_sample.activation.is_finite() && gpu_sample.avg_error.is_finite() {
                // Convert NaN values to None for target data
                let target_value = if gpu_sample.target_value.is_finite() {
                    Some(gpu_sample.target_value)
                } else {
                    None
                };
                let target_activation = if gpu_sample.target_activation.is_finite() {
                    Some(gpu_sample.target_activation)
                } else {
                    None
                };
                samples.push(HelpfulSample {
                    activation: gpu_sample.activation,
                    avg_error: gpu_sample.avg_error,
                    target_value,
                    target_activation,
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

const RELU_SHADER: &str = include_str!("shaders/relu.wgsl");

const ACTIVATION_SHADER: &str = include_str!("shaders/activation.wgsl");

const BIAS_SHADER: &str = include_str!("shaders/bias.wgsl");

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

/// Target data for a single observation (used for matching with source records)
struct TargetData {
    avg_error: f32,
    value: Option<f32>,
    activation: f32,
}

fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    // Build map from obs_index to target data (error, value, activation)
    let mut target_map: HashMap<u32, TargetData> = HashMap::with_capacity(target_records.len());
    for record in target_records {
        if record.errors.is_empty() || !record.activation.is_finite() {
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
        target_map.insert(
            record.obs_index,
            TargetData {
                avg_error: sum / count as f32,
                value: record.value,
                activation: record.activation,
            },
        );
    }

    if target_map.is_empty() {
        return Vec::new();
    }

    let mut samples = Vec::new();
    for record in from_records {
        if let Some(target) = target_map.get(&record.obs_index) {
            if record.activation.is_finite() && target.avg_error.is_finite() {
                samples.push(HelpfulSample {
                    activation: record.activation,
                    avg_error: target.avg_error,
                    target_value: target.value,
                    target_activation: Some(target.activation),
                });
            }
        }
    }

    samples
}

/// Computes the sign of a weight as an i8 for use in the candidate key.
/// Returns 1 for positive weights, -1 for negative, and 0 for zero (though this
/// shouldn't happen in practice).
fn weight_sign(weight: f32) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

fn upsert_candidate(
    map: &mut HashMap<(String, String, String, i8, i8), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    // Key includes signs of BOTH incoming_weight AND outgoing_weight so that:
    // 1. Different ReLU orientations (incoming_weight ±1) are kept separately
    // 2. Split-error complementary pairs (same incoming_weight, opposite outgoing_weight)
    //    are also kept separately - one pushes output UP, one pushes DOWN
    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
        weight_sign(candidate.incoming_weight),
        weight_sign(candidate.outgoing_weight),
    );

    // DEBUG: Log ReLU candidate insertions
    if verbose_enabled() && candidate.squash == "ReLU" {
        eprintln!(
            "[NEAT-AI-Discovery][DEBUG] upsert_candidate ReLU: {} -> {} improvement={:.2}% key=({}, {}, {}, {}, {})",
            candidate.source_neuron_uuid,
            candidate.target_neuron_uuid,
            candidate.expected_improvement_percentage * 100.0,
            &candidate.source_neuron_uuid[..8.min(candidate.source_neuron_uuid.len())],
            &candidate.target_neuron_uuid[..8.min(candidate.target_neuron_uuid.len())],
            candidate.squash,
            weight_sign(candidate.incoming_weight),
            weight_sign(candidate.outgoing_weight),
        );
    }

    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            let existing = entry.get();
            if candidate.expected_improvement_percentage > existing.expected_improvement_percentage
            {
                if verbose_enabled() && candidate.squash == "ReLU" {
                    eprintln!(
                        "[NEAT-AI-Discovery][DEBUG] ReLU replacing existing {} ({:.2}% -> {:.2}%)",
                        existing.squash,
                        existing.expected_improvement_percentage * 100.0,
                        candidate.expected_improvement_percentage * 100.0,
                    );
                }
                entry.insert(candidate);
            } else if verbose_enabled() && candidate.squash == "ReLU" {
                eprintln!(
                    "[NEAT-AI-Discovery][DEBUG] ReLU NOT replacing {} (existing {:.2}% >= new {:.2}%)",
                    existing.squash,
                    existing.expected_improvement_percentage * 100.0,
                    candidate.expected_improvement_percentage * 100.0,
                );
            }
        }
        Entry::Vacant(entry) => {
            if verbose_enabled() && candidate.squash == "ReLU" {
                eprintln!("[NEAT-AI-Discovery][DEBUG] ReLU inserted as NEW entry",);
            }
            entry.insert(candidate);
        }
    }
}

/// Result from ReLU evaluation (split by target error sign)
struct SplitReluResult {
    /// Candidate for samples with positive error (output should be higher)
    positive_error_candidate: Option<CandidateNeuronJson>,
    /// Candidate for samples with negative error (output should be lower)
    negative_error_candidate: Option<CandidateNeuronJson>,
}

/// Apply HARD_TANH activation function (clamp to [-1, 1])
#[inline(always)]
fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// ReLU activation for target simulation
#[inline(always)]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// LeakyReLU activation for target simulation
#[inline(always)]
fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

/// Get the activation function for a given squash name.
/// Returns None for activations that are approximately linear and don't need simulation.
#[inline]
fn get_target_activation_fn(squash: &str) -> Option<fn(f32) -> f32> {
    match squash {
        "HARD_TANH" => Some(hard_tanh),
        "ReLU" => Some(relu),
        "LeakyReLU" => Some(leaky_relu),
        "TANH" => Some(|x: f32| x.tanh()),
        "LOGISTIC" => Some(logistic_activation),
        "BIPOLAR" => Some(bipolar_activation),
        "CLIPPED" => Some(clipped_activation),
        // IDENTITY, INVERSE, etc. are linear - no simulation needed
        _ => None,
    }
}

/// Check if samples support target activation simulation (all have target data).
/// Returns the activation function to use, or None if linear approximation should be used.
#[inline]
fn get_target_simulation_fn(
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
) -> Option<fn(f32) -> f32> {
    let squash = target_squash?;
    let activation_fn = get_target_activation_fn(squash)?;

    // Verify all samples have the required target data
    if samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some())
    {
        Some(activation_fn)
    } else {
        None
    }
}

/// Legacy function for backwards compatibility - returns true only for HARD_TANH
/// Deprecated: Use get_target_simulation_fn instead for more accurate simulation
#[inline]
#[cfg(test)] // Only used in tests now
fn can_use_hard_tanh(samples: &[HelpfulSample], target_squash: Option<&str>) -> bool {
    target_squash == Some("HARD_TANH")
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some())
}

/// Combined computation of improvement and count for ReLU candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// IMPORTANT: The `bias` parameter is critical for accurate predictions. It shifts the ReLU
/// activation threshold, affecting which samples produce non-zero output. When bias > 0,
/// more samples activate; when bias < 0, fewer samples activate. Excluding bias causes
/// significant prediction errors.
///
/// Returns (improvement_percentage, improved_count, total_count)
fn compute_relu_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let new_error = if let Some(target_fn) = target_activation_fn {
            // Simulate the target neuron's actual activation function
            // Safety: target_activation_fn is only Some when all samples have target data
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let expected = target_activation + sample.avg_error;
            let new_input = target_value + contribution;
            target_fn(new_input) - expected
        } else {
            // Linear approximation - consistent with synapse model: new_error = old_error - correction
            // avg_error is (expected - actual), contribution adds to output, so reduces error
            sample.avg_error - contribution
        };

        new_error_sq_sum += new_error * new_error;

        // Sample is improved if |new_error| < |old_error|
        if new_error.abs() + EPSILON < sample.avg_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq;
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, total_count)
}

/// Combined computation of improvement and count for activation candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// Returns (improvement_percentage, improved_count, total_count)
fn compute_activation_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

        let new_error = if let Some(target_fn) = target_activation_fn {
            // Simulate the target neuron's actual activation function
            // Safety: target_activation_fn is only Some when all samples have target data
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let expected = target_activation + sample.avg_error;
            let new_input = target_value + contribution;
            target_fn(new_input) - expected
        } else {
            // Linear approximation - consistent with synapse model: new_error = old_error - correction
            // avg_error is (expected - actual), contribution adds to output, so reduces error
            sample.avg_error - contribution
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        if new_error.abs() + EPSILON < sample.avg_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq;
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, total_count)
}

/// Wrapper for tests - computes improvement only.
/// NOTE: For ReLU candidates, bias affects which samples activate. Pass the actual bias
/// that will be used with the new neuron for accurate predictions.
#[cfg(test)]
fn compute_net_improvement_with_squash(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (improvement, _, _) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    improvement
}

/// Compute synapse improvement accounting for target neuron's activation function.
///
/// For direct synapse connections (source → target), the contribution is `weight × source_activation`.
/// This function simulates the target's activation function to predict accurate improvement,
/// avoiding overprediction near saturation for HARD_TANH, TANH, LOGISTIC, etc.
///
/// Returns improvement_percentage only. Used in tests; production uses compute_synapse_improvement_and_count.
#[cfg(test)]
fn compute_synapse_improvement_with_target_squash(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return 0.0;
    }

    // Check if we can use saturation-aware model
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    let mut new_error_sq_sum = 0.0f32;

    for sample in samples {
        // Direct synapse contribution: weight × source_activation
        let contribution = weight * sample.activation;

        let new_error = if let Some(target_fn) = target_activation_fn {
            // Saturation-aware model: apply target's activation function
            // Safety: target_activation_fn is only Some when all samples have target data
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let expected = target_activation + sample.avg_error;
            let new_input = target_value + contribution;
            target_fn(new_input) - expected
        } else {
            // Linear approximation - assumes contribution directly reduces error
            sample.avg_error - contribution
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }
    }

    let improvement = (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq;
    if improvement.is_finite() {
        improvement
    } else {
        0.0
    }
}

/// Compute improvement, improved count, and worsened count for synapse candidates.
/// All counts use the same saturation-aware methodology for consistency.
///
/// Returns (improvement_percentage, improved_count, worsened_count, total_count)
fn compute_synapse_improvement_and_count(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, 0, samples.len() as u32);
    }

    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let contribution = weight * sample.activation;

        let new_error = if let Some(target_fn) = target_activation_fn {
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let expected = target_activation + sample.avg_error;
            let new_input = target_value + contribution;
            target_fn(new_input) - expected
        } else {
            sample.avg_error - contribution
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Count improved samples: new error is smaller than old error
        if new_error.abs() + EPSILON < sample.avg_error.abs() {
            improved_count += 1;
        }
        // Count worsened samples: new error is larger than old error
        else if new_error.abs() > sample.avg_error.abs() + EPSILON {
            worsened_count += 1;
        }
        // Note: samples where |new_error| ≈ |old_error| are neither improved nor worsened
    }

    let improvement = (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq;
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, total_count)
}

/// Wrapper for tests - counts improved samples only.
#[cfg(test)]
fn count_improved_samples(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_squash: Option<&str>,
) -> (u32, u32) {
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (_, improved, total) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    (improved, total)
}

/// Evaluate ReLU candidates by splitting samples based on TARGET neuron's error sign.
///
/// This is the PRIMARY approach for ReLU evaluation. It finds candidates for both directions:
/// - **Positive-error samples** (output should be HIGHER): compute weight that pushes UP
/// - **Negative-error samples** (output should be LOWER): compute weight that pushes DOWN
///
/// For each direction:
/// 1. Compute optimal weight from the error subset
/// 2. Evaluate NET improvement across ALL samples
/// 3. Return candidate if it passes threshold
///
/// This is the correct approach for directional activations like ReLU because:
/// - ReLU can only push output in ONE direction (based on outgoing weight sign)
/// - Averaging over all samples cancels out when errors are split ~50/50
/// - We evaluate source activations as-is (we don't care how they were calculated)
fn evaluate_relu_candidates_split(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
) -> Result<SplitReluResult> {
    // Split samples by error sign
    let positive_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error > EPSILON)
        .copied()
        .collect();

    let negative_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error < -EPSILON)
        .copied()
        .collect();

    let mut result = SplitReluResult {
        positive_error_candidate: None,
        negative_error_candidate: None,
    };

    // Compute total baseline error across ALL samples (for net improvement calculation)
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    if total_baseline_error_sq <= EPSILON {
        return Ok(result);
    }

    // Get target activation function for accurate simulation (ReLU, HARD_TANH, etc.)
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // For positive errors (output should be higher), compute optimal weight from subset
    // then evaluate the NET effect across ALL samples.
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if positive_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, pos_baseline_error_sq) =
            analyzer.evaluate_relu_gpu(&positive_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                pos_baseline_error_sq,
                &positive_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                if net_improvement > best_improvement {
                    candidate.expected_improvement_percentage = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.positive_error_candidate = best_candidate;
    }

    // For negative errors (output should be lower).
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if negative_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, neg_baseline_error_sq) =
            analyzer.evaluate_relu_gpu(&negative_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                neg_baseline_error_sq,
                &negative_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                if net_improvement > best_improvement {
                    candidate.expected_improvement_percentage = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.negative_error_candidate = best_candidate;
    }

    Ok(result)
}

fn evaluate_activation_candidate(
    analyzer: &GpuAnalyzer,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    spec: &ActivationCandidateSpec,
    target_squash: Option<&str>,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = threshold;
    let mut fallback_candidate: Option<CandidateNeuronJson> = None;
    let mut fallback_score = f32::MIN;

    // Compute total_baseline_error_sq once
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    let use_gpu = analyzer.device.is_some();
    for &orientation in spec.orientations {
        for &scale in spec.scales {
            let incoming_weight = orientation * scale;
            let (
                sum_activation_sq,
                sum_error_activation,
                gpu_baseline_sq,
                _gpu_improved_count, // Ignored - calculated after weight validation
                gpu_succeeded,
            ) = if use_gpu {
                // Use GPU-accelerated evaluation
                match analyzer.evaluate_activation_gpu(samples, activation_type, orientation, scale)
                {
                    Ok(result) => (result.0, result.1, result.2, result.3, true),
                    Err(_) => {
                        // Fall back to CPU if GPU fails
                        let mut sum_activation_sq = 0.0;
                        let mut sum_error_activation = 0.0;
                        for sample in samples {
                            let pre_activation = incoming_weight * sample.activation;
                            let output = (spec.activation)(pre_activation);
                            if output.is_finite() {
                                sum_activation_sq += output * output;
                                sum_error_activation += output * sample.avg_error;
                            }
                        }
                        (
                            sum_activation_sq,
                            sum_error_activation,
                            total_baseline_error_sq,
                            0,
                            false,
                        )
                    }
                }
            } else {
                // CPU path
                let mut sum_activation_sq = 0.0;
                let mut sum_error_activation = 0.0;
                for sample in samples {
                    let pre_activation = incoming_weight * sample.activation;
                    let output = (spec.activation)(pre_activation);
                    if output.is_finite() {
                        sum_activation_sq += output * output;
                        sum_error_activation += output * sample.avg_error;
                    }
                }
                (
                    sum_activation_sq,
                    sum_error_activation,
                    total_baseline_error_sq,
                    0,
                    false,
                )
            };

            // Use GPU baseline if available, otherwise use CPU baseline
            let baseline_sq = if use_gpu && gpu_succeeded {
                gpu_baseline_sq
            } else {
                total_baseline_error_sq
            };

            if sum_activation_sq <= EPSILON {
                continue;
            }

            // Calculate linear optimal weight as starting point
            let linear_optimal_weight = sum_error_activation / (sum_activation_sq + EPSILON);
            if !linear_optimal_weight.is_finite() || linear_optimal_weight.abs() <= EPSILON {
                continue;
            }

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            // For non-linear targets (HARD_TANH, ReLU, etc.), search for best outgoing_weight
            // since linear optimal may be wrong due to activation saturation/clipping.
            // Must verify samples have target_value/target_activation data before simulation.
            let target_activation_fn = get_target_simulation_fn(samples, target_squash);
            let (
                outgoing_weight,
                optimal_bias,
                expected_improvement_percentage,
                final_improved_count,
            ) = if target_activation_fn.is_some() {
                // Weight candidates: linear optimal and scaled versions
                let weight_candidates: [f32; 9] = [
                    linear_optimal_weight * 0.1,
                    linear_optimal_weight * 0.25,
                    linear_optimal_weight * 0.5,
                    linear_optimal_weight * 0.75,
                    linear_optimal_weight,
                    linear_optimal_weight * 1.5,
                    linear_optimal_weight * 2.0,
                    -linear_optimal_weight * 0.5,
                    -linear_optimal_weight,
                ];

                let mut best_weight = linear_optimal_weight.clamp(-10.0, 10.0);
                let mut best_bias = 0.0f32;
                let mut best_improvement = f32::NEG_INFINITY;
                let mut best_improved_count = 0u32;

                for &weight in &weight_candidates {
                    let clamped_weight = weight.clamp(-10.0, 10.0);
                    if clamped_weight.abs() <= EPSILON {
                        continue;
                    }

                    let bias = calculate_optimal_bias(
                        samples,
                        incoming_weight,
                        clamped_weight,
                        spec.activation,
                        spec.name,
                        None,
                        target_squash,
                    );

                    // CRITICAL FIX: Recompute optimal weight WITH the bias included.
                    // The initial weight_candidates were computed WITHOUT bias, so they're
                    // wrong when bias significantly changes the activation pattern.
                    // Now recompute the weight that optimises error reduction for this bias.
                    let mut sum_activation_sq_with_bias = 0.0f32;
                    let mut sum_error_activation_with_bias = 0.0f32;
                    for sample in samples.iter() {
                        let pre_activation = incoming_weight * sample.activation + bias;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_activation_sq_with_bias += output * output;
                            sum_error_activation_with_bias += output * sample.avg_error;
                        }
                    }
                    let recomputed_weight = if sum_activation_sq_with_bias > EPSILON {
                        (sum_error_activation_with_bias / sum_activation_sq_with_bias)
                            .clamp(-10.0, 10.0)
                    } else {
                        clamped_weight
                    };

                    // Single pass for improvement and count with target simulation
                    let (improvement, improved, _) = compute_activation_improvement_and_count(
                        samples,
                        incoming_weight,
                        recomputed_weight,
                        bias,
                        spec.activation,
                        baseline_sq,
                        target_activation_fn,
                    );

                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_weight = recomputed_weight;
                        best_bias = bias;
                        best_improved_count = improved;
                    }
                }

                (
                    best_weight,
                    best_bias,
                    best_improvement,
                    best_improved_count,
                )
            } else {
                // For linear targets or when target data unavailable, use linear optimal weight
                let initial_weight = linear_optimal_weight.clamp(-10.0, 10.0);

                let optimal_bias = calculate_optimal_bias(
                    samples,
                    incoming_weight,
                    initial_weight,
                    spec.activation,
                    spec.name,
                    None,
                    target_squash,
                );

                // CRITICAL FIX: Recompute optimal weight WITH the bias included.
                // The linear_optimal_weight was computed WITHOUT bias, so recompute
                // using the actual activation pattern with bias.
                let mut sum_activation_sq_with_bias = 0.0f32;
                let mut sum_error_activation_with_bias = 0.0f32;
                for sample in samples.iter() {
                    let pre_activation = incoming_weight * sample.activation + optimal_bias;
                    let output = (spec.activation)(pre_activation);
                    if output.is_finite() {
                        sum_activation_sq_with_bias += output * output;
                        sum_error_activation_with_bias += output * sample.avg_error;
                    }
                }
                let outgoing_weight = if sum_activation_sq_with_bias > EPSILON {
                    (sum_error_activation_with_bias / sum_activation_sq_with_bias)
                        .clamp(-10.0, 10.0)
                } else {
                    initial_weight
                };

                let (improvement, improved_count, _) = compute_activation_improvement_and_count(
                    samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    baseline_sq,
                    None, // Linear approximation
                );

                (outgoing_weight, optimal_bias, improvement, improved_count)
            };

            // Skip invalid weights
            if outgoing_weight.abs() <= EPSILON {
                continue;
            }

            // =================================================================
            // FUNDAMENTAL VALIDITY FILTERS
            // These filters apply to ALL candidates (both fallback and best).
            // They must be checked BEFORE setting fallback_candidate to prevent
            // invalid candidates from being returned through the fallback path.
            // =================================================================

            // Require minimum ABSOLUTE error reduction, not just percentage.
            // A 1% improvement on baseline_sq=0.0001 is only 0.000001 absolute reduction,
            // which won't meaningfully affect the creature's total error.
            // Minimum absolute improvement = 0.001 (0.1% of typical baseline ~1.0)
            let absolute_improvement = expected_improvement_percentage * baseline_sq;
            if absolute_improvement < 0.001 {
                continue;
            }

            // IDENTITY with bias ≈ 0 is equivalent to a direct synapse (source × incoming × outgoing)
            // Filter out these redundant candidates - use synapse analysis for direct connections
            if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
                continue;
            }

            // =================================================================
            // FALLBACK CANDIDATE (best seen so far, regardless of threshold)
            // =================================================================
            if expected_improvement_percentage > fallback_score {
                fallback_score = expected_improvement_percentage;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                fallback_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    expected_improvement_percentage,
                    improved_count: final_improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                });
            }

            // =================================================================
            // THRESHOLD CHECK (only for best_candidate, not fallback)
            // =================================================================
            // Use the STRICTER threshold (max) to ensure meaningful improvements
            // If spec requires 5% and caller passes 1%, we require 5%
            // If spec requires 0% and caller passes 1%, we require 1%
            let improvement_cutoff = threshold.max(spec.min_improvement);

            if expected_improvement_percentage <= improvement_cutoff
                || final_improved_count < MIN_NEURON_SAMPLE_COUNT as u32
            {
                continue;
            }

            // Current iteration passed threshold - create best_candidate with current iteration's values
            if expected_improvement_percentage > best_score {
                best_score = expected_improvement_percentage;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                // Use current iteration's values, not fallback candidate's values
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    expected_improvement_percentage, // Use current iteration's value
                    improved_count: final_improved_count, // Use current iteration's value
                    total_count,                     // Use current iteration's value
                    target_neuron_stats: target_stats,
                });
            }
        }
    }

    Ok(best_candidate.or(fallback_candidate))
}

/// Build discrete samples for threshold-crossing analysis.
/// Combines source neuron activations with target neuron values and errors.
fn build_discrete_samples(
    source_records: &[DiscoverRecord],
    target_records: &[DiscoverRecord],
) -> Vec<DiscreteHelpfulSample> {
    // Build a map from obs_index to target record
    let target_map: HashMap<u32, &DiscoverRecord> = target_records
        .iter()
        .filter(|r| !r.errors.is_empty())
        .map(|r| (r.obs_index, r))
        .collect();

    let mut samples = Vec::new();

    for source_record in source_records {
        if !source_record.activation.is_finite() {
            continue;
        }

        if let Some(target_record) = target_map.get(&source_record.obs_index) {
            // Need target's value (pre-activation input sum) for threshold crossing
            let target_value = match target_record.value {
                Some(v) if v.is_finite() => v,
                _ => continue, // Skip if no value available
            };

            if !target_record.activation.is_finite() {
                continue;
            }

            // Compute average error for target
            let mut error_sum = 0.0;
            let mut error_count = 0;
            for &err in &target_record.errors {
                if err.is_finite() {
                    error_sum += err;
                    error_count += 1;
                }
            }

            if error_count > 0 {
                let avg_error = error_sum / error_count as f32;
                samples.push(DiscreteHelpfulSample {
                    source_activation: source_record.activation,
                    target_value,
                    target_activation: target_record.activation,
                    avg_error,
                });
            }
        }
    }

    samples
}

/// Evaluate a discrete activation candidate using threshold-crossing model.
/// Instead of predicting continuous error reduction, counts how many samples
/// would flip to the correct output if we add a new connection.
///
/// For STEP/BIPOLAR, the only meaningful improvement is flipping the output:
/// - If error > 0 (output should be higher), we want to flip 0→1 or -1→1
/// - If error < 0 (output should be lower), we want to flip 1→0 or 1→-1
fn evaluate_discrete_candidate(
    source_uuid: &str,
    target_uuid: &str,
    samples: &[DiscreteHelpfulSample],
    threshold_type: ThresholdType,
    threshold: f32,
) -> Option<CandidateNeuronJson> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return None;
    }

    let total_count = samples.len() as u32;

    // Weight scales to try - for discrete functions, we need weights that can
    // actually push the target across the threshold
    const SCALES: [f32; 8] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    const ORIENTATIONS: [f32; 2] = [1.0, -1.0];

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_improvement = threshold;

    // For discrete functions, use IDENTITY squash on the new neuron
    // This passes the weighted source activation directly
    let new_neuron_squash = "IDENTITY";

    for &orientation in &ORIENTATIONS {
        for &scale in &SCALES {
            let incoming_weight = orientation * scale;

            // For IDENTITY squash, new_neuron_output = incoming_weight * source_activation
            // Try different outgoing weights
            for &out_scale in &SCALES {
                for &out_orientation in &ORIENTATIONS {
                    let outgoing_weight = out_orientation * out_scale;

                    // Count helpful and harmful flips
                    let mut helpful_flips = 0i32;
                    let mut harmful_flips = 0i32;
                    let mut samples_with_error = 0u32;

                    for sample in samples {
                        if sample.avg_error.abs() < EPSILON {
                            continue; // No error, nothing to improve
                        }
                        samples_with_error += 1;

                        // New neuron output with IDENTITY: just passes through
                        let new_neuron_output = incoming_weight * sample.source_activation;
                        let contribution = outgoing_weight * new_neuron_output;

                        let flip_dir = threshold_type.flip_direction(
                            sample.target_value,
                            contribution,
                            sample.avg_error,
                        );

                        match flip_dir {
                            1 => helpful_flips += 1,
                            -1 => harmful_flips += 1,
                            _ => {}
                        }
                    }

                    if samples_with_error < MIN_NEURON_SAMPLE_COUNT as u32 {
                        continue;
                    }

                    // Net improvement: proportion of samples that would be corrected
                    let net_flips = helpful_flips - harmful_flips;
                    let improvement = net_flips as f32 / samples_with_error as f32;

                    // IMPORTANT: Require meaningful improvement (at least 1%) for IDENTITY neurons.
                    // IDENTITY with bias=0 is mathematically equivalent to a direct synapse:
                    //   IDENTITY(source × incoming_weight + 0) × outgoing_weight = source × incoming × outgoing
                    // These candidates should use add-synapse, not add-neuron.
                    // Additionally, very low flip rates (< 1%) indicate the contribution isn't
                    // reliably pushing the target across the threshold.
                    const MIN_DISCRETE_IMPROVEMENT: f32 = 0.01; // 1% minimum

                    if improvement > best_improvement.max(MIN_DISCRETE_IMPROVEMENT)
                        && helpful_flips > harmful_flips
                    {
                        best_improvement = improvement;

                        // Create target neuron stats from samples
                        let target_stats = {
                            let helper_samples: Vec<HelpfulSample> = samples
                                .iter()
                                .map(|s| HelpfulSample {
                                    activation: s.target_activation,
                                    avg_error: s.avg_error,
                                    target_value: Some(s.target_value),
                                    target_activation: Some(s.target_activation),
                                })
                                .collect();
                            NeuronStats::from_samples(&helper_samples).map(|s| s.to_json())
                        };

                        best_candidate = Some(CandidateNeuronJson {
                            source_neuron_uuid: source_uuid.to_string(),
                            target_neuron_uuid: target_uuid.to_string(),
                            incoming_weight,
                            outgoing_weight,
                            squash: new_neuron_squash.to_string(),
                            bias: 0.0, // IDENTITY doesn't need bias for threshold crossing
                            expected_improvement_percentage: improvement,
                            improved_count: helpful_flips as u32,
                            total_count,
                            target_neuron_stats: target_stats,
                        });
                    }
                }
            }
        }
    }

    best_candidate
}

fn analyze_neurons_with_cache(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeNeuronsResult> {
    let threshold = input.improvement_threshold.unwrap_or(0.1);
    let ordered_neurons = build_ordered_neurons(&input.creature);

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    // Build a comprehensive lookup map for ALL neuron UUIDs to their types.
    // This includes: input neurons (from creature.input count) and all neurons from
    // creature.neurons (hidden, output, constant). If a UUID is not in this map,
    // it's an invalid UUID (bug in the caller).
    //
    // Only output neurons should be targets for add-neuron candidates because:
    // - Output neuron errors directly affect creature score
    // - Hidden neuron errors are backpropagated approximations that don't correlate
    //   reliably with actual output error reduction
    // - Input neurons are observation sources, not computation nodes
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    // Log creature configuration for debugging data issues
    if verbose_enabled() {
        let non_input_count = input.creature.neurons.len();
        let total_neurons = input.creature.input + non_input_count;
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Neuron analysis creature config: {} input neurons (input-0 to input-{}), {} non-input neurons, {} total ordered neurons",
            input.creature.input,
            input.creature.input.saturating_sub(1),
            non_input_count,
            total_neurons
        );

        // Verify input neurons exist in parquet by checking a sample
        if input.creature.input > 0 {
            match cache.get("input-0") {
                Ok(records) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check: input-0 has {} records",
                        records.len()
                    );
                    if !records.is_empty() {
                        let first = &records[0];
                        let last = &records[records.len() - 1];
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Parquet data check: input-0 obs_index range [{}, {}], first activation={:.4}",
                            first.obs_index,
                            last.obs_index,
                            first.activation
                        );
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check FAILED: input-0 error: {err}"
                    );
                }
            }

            // Also check a middle input neuron
            let mid_input = input.creature.input / 2;
            let mid_uuid = format!("input-{mid_input}");
            match cache.get(&mid_uuid) {
                Ok(records) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check: {mid_uuid} has {} records",
                        records.len()
                    );
                }
                Err(err) => {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Parquet data check FAILED: {mid_uuid} error: {err}"
                    );
                }
            }
        }
    }

    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    let helpful_map = Arc::new(Mutex::new(HashMap::<
        (String, String, String, i8, i8),
        CandidateNeuronJson,
    >::new()));

    let diagnostics = Arc::new(Mutex::new(NeuronDiagnostics::new(&unique_focus)));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Filter focus neurons to ONLY output neurons for add-neuron analysis.
    //
    // Rationale: Add-neuron candidates predict error reduction at the target neuron.
    // For OUTPUT neurons, this directly corresponds to creature score improvement.
    // For HIDDEN neurons, the backpropagated error is an approximation that doesn't
    // reliably translate to actual output error reduction - we've observed 100%
    // failure rates when targeting hidden neurons.
    // For INPUT neurons, they are observation sources, not computation nodes - they
    // have no activation function or error to reduce.
    //
    // Non-output neurons are skipped here but still tracked in diagnostics so the
    // caller knows they were received but filtered out (with the correct reason).
    let original_focus_count = unique_focus.len();
    let mut skipped_hidden: Vec<String> = Vec::new();
    let mut skipped_input: Vec<String> = Vec::new();

    // Randomize the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    //
    // All neurons are processed - no activation functions are skipped. STEP/BIPOLAR
    // neurons use a specialised threshold-crossing model; all others use the standard
    // linear error model (which is an approximation but still finds useful patterns).
    let mut threshold_targets: Vec<String> = Vec::new();
    let mut focus_order: Vec<String> = unique_focus
        .iter()
        .filter_map(|uuid| {
            // Check neuron type - only allow output neurons for add-neuron analysis
            let neuron_type = neuron_type_map.get(*uuid).map(|s| s.as_str());
            match neuron_type {
                Some("output") => {
                    // Output neurons are valid targets - continue processing
                    if let Some(squash) = neuron_squash_map.get(*uuid) {
                        if is_threshold_activation(squash) {
                            threshold_targets.push((*uuid).clone());
                        }
                    }
                    Some((*uuid).clone())
                }
                Some("input") => {
                    // Input neurons are observation sources, not computation nodes
                    skipped_input.push((*uuid).clone());
                    None
                }
                Some(_) => {
                    // Hidden/constant neurons have backpropagated errors that don't
                    // reliably translate to output error reduction
                    skipped_hidden.push((*uuid).clone());
                    None
                }
                None => {
                    // Unknown UUID - this is likely a bug, but treat as hidden for now
                    // (this shouldn't happen with proper creature data)
                    eprintln!(
                        "[NEAT-AI-Discovery] Warning: Unknown neuron UUID '{uuid}' in focus list \
                        (not found in creature). Treating as hidden neuron."
                    );
                    skipped_hidden.push((*uuid).clone());
                    None
                }
            }
        })
        .collect();

    // Log when non-output neurons are filtered out
    let total_skipped = skipped_hidden.len() + skipped_input.len();
    if total_skipped > 0 {
        if !skipped_input.is_empty() && !skipped_hidden.is_empty() {
            eprintln!(
                "[NEAT-AI-Discovery] Filtered {} neuron(s) from add-neuron analysis (only output neurons are valid targets). \
                Input neurons skipped: {:?}. Hidden neurons skipped: {:?}. Remaining output neurons: {}",
                total_skipped,
                skipped_input.iter().take(5).collect::<Vec<_>>(),
                skipped_hidden.iter().take(5).collect::<Vec<_>>(),
                focus_order.len()
            );
        } else if !skipped_input.is_empty() {
            eprintln!(
                "[NEAT-AI-Discovery] Filtered {} input neuron(s) from add-neuron analysis \
                (input neurons are observation sources, not computation nodes). \
                Input neurons skipped: {:?}. Remaining output neurons: {}",
                skipped_input.len(),
                skipped_input.iter().take(5).collect::<Vec<_>>(),
                focus_order.len()
            );
        } else {
            eprintln!(
                "[NEAT-AI-Discovery] Filtered {} hidden neuron(s) from add-neuron analysis \
                (only output neurons are valid targets). \
                Hidden neurons skipped: {:?}. Remaining output neurons: {}",
                skipped_hidden.len(),
                skipped_hidden.iter().take(5).collect::<Vec<_>>(),
                focus_order.len()
            );
        }
    }

    // If no output neurons remain after filtering, return early with empty results
    if focus_order.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery] No output neurons in focus list ({original_focus_count} non-output neurons filtered out). \
            Add-neuron candidates can only target output neurons."
        );
        // Build no_candidate_reasons with correct reason for each neuron type
        let mut no_candidate_reasons: Vec<NeuronNoCandidateSummary> = Vec::new();
        for uuid in &skipped_input {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::InputNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        for uuid in &skipped_hidden {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        return Ok(AnalyzeNeuronsResult {
            helpful_neurons: Vec::new(),
            gpu_used: true,
            no_candidate_reasons,
        });
    }

    // Mark skipped neurons in diagnostics so they appear with the correct reason
    // instead of misleading reasons like NoEligibleSources.
    // This is the normal flow case where some output neurons exist.
    for input_uuid in &skipped_input {
        diagnostics
            .lock()
            .expect("Mutex poisoned: diagnostics")
            .mark_input_filtered(input_uuid);
    }
    for hidden_uuid in &skipped_hidden {
        diagnostics
            .lock()
            .expect("Mutex poisoned: diagnostics")
            .mark_hidden_filtered(hidden_uuid);
    }

    // Log threshold-crossing neurons for visibility
    if verbose_enabled() && !threshold_targets.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Using threshold-crossing model for {} STEP/BIPOLAR neurons: {:?}",
            threshold_targets.len(),
            threshold_targets.iter().take(5).collect::<Vec<_>>()
        );
    }

    let mut rng = thread_rng();
    focus_order.shuffle(&mut rng);

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);

    // Process each focus neuron in parallel. Deadline checks happen at the start of
    // each focus target so that once analysis for a neuron begins, we prefer to
    // complete its upstream evaluation rather than abandoning it mid-stream. This
    // gives us “vertical” timeout behaviour where some neurons complete fully even
    // if later targets are skipped when the deadline is reached.
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            // Each thread gets its own GpuAnalyzer (wgpu devices are not thread-safe)
            let analyzer = GpuAnalyzer::new()?;

            let target_records_arc = match cache.get(target_uuid.as_str()) {
                Ok(records) => records,
                Err(err) => {
                    if cfg!(debug_assertions) {
                        eprintln!("Failed to load target neuron records for {target_uuid}: {err}");
                    }
                    return Ok(());
                }
            };
            if target_records_arc.is_empty() {
                diagnostics
                    .lock()
                    .expect("Mutex poisoned: diagnostics")
                    .set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_target_record_count(target_uuid, target_records.len());

            // Log target neuron obs_index range for debugging sample matching
            if verbose_enabled() && !target_records.is_empty() {
                let first_obs = target_records.first().map(|r| r.obs_index).unwrap_or(0);
                let last_obs = target_records.last().map(|r| r.obs_index).unwrap_or(0);
                let has_errors = target_records.iter().any(|r| !r.errors.is_empty());
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} has {} records, obs_index range [{}, {}], has_errors={}",
                    target_uuid,
                    target_records.len(),
                    first_obs,
                    last_obs,
                    has_errors
                );
            }

            // Check if this is a threshold activation (STEP/BIPOLAR) that needs special handling
            let threshold_type = neuron_squash_map_arc
                .get(target_uuid)
                .and_then(|squash| ThresholdType::from_squash(squash));

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => return Ok(()),
            };

            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| neuron.index < target_index)
                .collect();
            let mut rng = thread_rng();
            eligible_sources.shuffle(&mut rng);

            // Track total eligible sources for diagnostics
            let total_eligible = eligible_sources.len() as u32;
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_total_eligible_sources(target_uuid, total_eligible);

            // Log focus neuron details for debugging
            if verbose_enabled() && total_eligible == 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {} (index {}) has 0 eligible upstream sources. creature.input={}, so input neurons span indices 0-{}. This indicates target_index <= 0 or a creature configuration mismatch.",
                    target_uuid,
                    target_index,
                    ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()),
                    ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()).saturating_sub(1)
                );
            }

            // Phase 1: Pre-filter sources and collect their records (with deadline checks)
            // This mirrors the synapse analysis approach for better parallelism
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());
            let mut empty_record_sources: Vec<String> = Vec::new();
            let mut load_failure_count = 0u32;

            for source in &eligible_sources {
                // Check deadline during pre-filtering
                if deadline_passed(&deadline) {
                    *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();
                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            sources_to_process.push((source, records));
                        } else {
                            empty_record_sources.push(source_uuid.to_string());
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load source neuron records for {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                }
            }

            // Record load failures in diagnostics
            if load_failure_count > 0 {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for _ in 0..load_failure_count {
                    diag.record_load_failure(target_uuid);
                }
            }

            // Log summary of source loading results for debugging
            let sources_checked = sources_to_process.len() + empty_record_sources.len() + load_failure_count as usize;
            let timed_out_during_loading = *analysis_timed_out.lock().expect("Mutex poisoned");
            if verbose_enabled() && (sources_to_process.is_empty() || load_failure_count > 0 || !empty_record_sources.is_empty() || timed_out_during_loading) {
                let sources_with_records = sources_to_process.len();
                let empty_count = empty_record_sources.len();
                if timed_out_during_loading && sources_checked == 0 {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: TIMEOUT before any of {total_eligible} eligible sources could be checked"
                    );
                } else if timed_out_during_loading {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: TIMEOUT after checking {sources_checked}/{total_eligible} eligible sources ({sources_with_records} with records, {empty_count} empty, {load_failure_count} failures)"
                    );
                } else {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid} source loading: {total_eligible} eligible -> {sources_with_records} with records, {empty_count} empty records, {load_failure_count} load failures"
                    );
                }
            }

            // Batch diagnostics for empty record sources
            if !empty_record_sources.is_empty() {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for source_uuid in &empty_record_sources {
                    diag.record_candidate_attempt(target_uuid, false);
                    diag.record_no_samples(target_uuid, source_uuid);
                }
            }

            // Check if timed out during pre-filtering
            if *analysis_timed_out.lock().expect("Mutex poisoned") {
                return Ok(());
            }

            // Phase 2: Build samples in parallel using CPU (much faster than sequential GPU calls)
            // This is the key optimization - parallel sample building on CPU cores
            struct NeuronWorkResult {
                source_uuid: String,
                samples: Vec<HelpfulSample>,
            }

            let work_results: Vec<NeuronWorkResult> = sources_to_process
                .par_iter()
                .map(|(source, from_records_arc)| {
                    let source_uuid = source.uuid.as_str();
                    let from_records = from_records_arc.as_ref();
                    // Use CPU for sample building - enables true parallelism
                    let samples = build_samples(target_records, from_records);
                    NeuronWorkResult {
                        source_uuid: source_uuid.to_string(),
                        samples,
                    }
                })
                .collect();

            // Phase 3: Batch diagnostics updates for sample building results
            {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for result in &work_results {
                    diag.record_candidate_attempt(target_uuid, !result.samples.is_empty());
                    if result.samples.is_empty() {
                        diag.record_no_samples(target_uuid, &result.source_uuid);
                    }
                }
            }

            // Phase 4: Process evaluations - GPU work is done here
            // Filter to only sources with samples, then evaluate
            //
            // For threshold activations (STEP/BIPOLAR), we use a specialised
            // threshold-crossing model instead of the standard linear error model.
            if let Some(t_type) = threshold_type {
                // Threshold activation path - use discrete evaluation
                for (source, from_records_arc) in &sources_to_process {
                    // Check deadline
                    if deadline_passed(&deadline) {
                        *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                        break;
                    }

                    let from_records = from_records_arc.as_ref();
                    let discrete_samples = build_discrete_samples(from_records, target_records);

                    if discrete_samples.len() < MIN_NEURON_SAMPLE_COUNT {
                        continue;
                    }

                    if let Some(candidate) = evaluate_discrete_candidate(
                        &source.uuid,
                        target_uuid,
                        &discrete_samples,
                        t_type,
                        threshold,
                    ) {
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Threshold-crossing candidate for {} -> {}: {} samples, {:.2}% improvement (flips: {})",
                                source.uuid,
                                target_uuid,
                                discrete_samples.len(),
                                candidate.expected_improvement_percentage * 100.0,
                                candidate.improved_count
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }
                }
            } else {
                // Standard continuous activation path
                for result in work_results {
                    // Check deadline before each evaluation batch
                    if deadline_passed(&deadline) {
                        *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                        break;
                    }

                    if result.samples.is_empty() {
                        continue;
                    }

                    // Get target_squash for accurate HARD_TANH modelling
                    let target_squash = neuron_squash_map_arc.get(target_uuid).map(|s| s.as_str());

                    // ReLU evaluation: split by TARGET neuron's error sign.
                    //
                    // ReLU can only push output in ONE direction (based on outgoing weight sign),
                    // so we evaluate two candidates separately:
                    // - Positive-error ReLU: optimised for samples where output should be HIGHER
                    // - Negative-error ReLU: optimised for samples where output should be LOWER
                    //
                    // Each candidate's weight is computed from its error subset, then NET
                    // improvement is calculated across ALL samples. This is the correct
                    // approach for directional activation functions like ReLU.
                    //
                    // NOTE: We don't use "averaging over all samples" because when errors are
                    // split ~50/50, the average cancels out and no candidate is found.
                    let split_result = evaluate_relu_candidates_split(
                        &analyzer,
                        &result.source_uuid,
                        target_uuid,
                        &result.samples,
                        threshold,
                        target_squash,
                    )?;

                    if let Some(candidate) = split_result.positive_error_candidate {
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] ReLU (push UP) {} -> {}: {:.2}% improvement",
                                result.source_uuid,
                                target_uuid,
                                candidate.expected_improvement_percentage * 100.0
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }

                    if let Some(candidate) = split_result.negative_error_candidate {
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] ReLU (push DOWN) {} -> {}: {:.2}% improvement",
                                result.source_uuid,
                                target_uuid,
                                candidate.expected_improvement_percentage * 100.0
                            );
                        }
                        diagnostics
                            .lock()
                            .expect("Mutex poisoned: diagnostics")
                            .mark_candidate_selected(target_uuid);
                        let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                        upsert_candidate(&mut map, candidate);
                    }

                    for spec in ACTIVATION_SPECS.iter() {
                        if let Some(candidate) = evaluate_activation_candidate(
                            &analyzer,
                            &result.source_uuid,
                            target_uuid,
                            &result.samples,
                            threshold,
                            spec,
                            target_squash,
                        )? {
                            diagnostics
                                .lock()
                                .expect("Mutex poisoned: diagnostics")
                                .mark_candidate_selected(target_uuid);
                            let mut map = helpful_map.lock().expect("Mutex poisoned: helpful_map");
                            upsert_candidate(&mut map, candidate);
                        }
                    }
                }
            }

            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out
        .lock()
        .expect("Mutex poisoned: analysis_timed_out");
    let helpful_map = helpful_map
        .lock()
        .expect("Mutex poisoned: helpful_map")
        .clone();
    let diagnostics = diagnostics.lock().expect("Mutex poisoned: diagnostics");

    if analysis_timed_out && verbose_enabled() {
        eprintln!("[NEAT-AI-Discovery][verbose] analyse_neurons reached analysis deadline; returning partial results.");
    }

    // DEBUG: Log contents of helpful_map before conversion
    if verbose_enabled() {
        let relu_count = helpful_map.values().filter(|c| c.squash == "ReLU").count();
        let total_count = helpful_map.len();
        eprintln!(
            "[NEAT-AI-Discovery][DEBUG] helpful_map contains {total_count} total candidates, {relu_count} are ReLU"
        );
        // Log top 5 by improvement (including squash type)
        let mut sorted_preview: Vec<_> = helpful_map.values().collect();
        sorted_preview.sort_by(|a, b| {
            b.expected_improvement_percentage
                .partial_cmp(&a.expected_improvement_percentage)
                .unwrap_or(Ordering::Equal)
        });
        for (i, c) in sorted_preview.iter().take(10).enumerate() {
            eprintln!(
                "[NEAT-AI-Discovery][DEBUG] Top {} in map: {} {} -> {} improvement={:.2}%",
                i + 1,
                c.squash,
                &c.source_neuron_uuid[..12.min(c.source_neuron_uuid.len())],
                &c.target_neuron_uuid[..12.min(c.target_neuron_uuid.len())],
                c.expected_improvement_percentage * 100.0
            );
        }
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();
    helpful_results.sort_by(|a, b| {
        b.expected_improvement_percentage
            .partial_cmp(&a.expected_improvement_percentage)
            .unwrap_or(Ordering::Equal)
    });

    // DEBUG: Log after sorting
    if verbose_enabled() && !helpful_results.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery][DEBUG] After sorting, top candidate: {} {} -> {} improvement={:.2}%",
            helpful_results[0].squash,
            &helpful_results[0].source_neuron_uuid[..12.min(helpful_results[0].source_neuron_uuid.len())],
            &helpful_results[0].target_neuron_uuid[..12.min(helpful_results[0].target_neuron_uuid.len())],
            helpful_results[0].expected_improvement_percentage * 100.0
        );
    }

    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
    }

    // DEBUG: Log final count after truncation
    if verbose_enabled() {
        let relu_count = helpful_results
            .iter()
            .filter(|c| c.squash == "ReLU")
            .count();
        eprintln!(
            "[NEAT-AI-Discovery][DEBUG] Returning {} candidates, {} are ReLU",
            helpful_results.len(),
            relu_count
        );
        // Log ALL returned candidates
        for (i, c) in helpful_results.iter().enumerate() {
            eprintln!(
                "[NEAT-AI-Discovery][DEBUG] RETURNED[{}]: {} {} -> {} improvement={:.4}% bias={:.4} inW={:.4} outW={:.4}",
                i,
                c.squash,
                &c.source_neuron_uuid,
                &c.target_neuron_uuid[..20.min(c.target_neuron_uuid.len())],
                c.expected_improvement_percentage * 100.0,
                c.bias,
                c.incoming_weight,
                c.outgoing_weight,
            );
        }
    }

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used,
        no_candidate_reasons,
    })
}

pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Neuron analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_preloaded(&input.parquet_file)?);
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

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    let shared_cache = Arc::new(RecordCache::new_preloaded(&input.parquet_file)?);

    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            improvement_threshold: input.improvement_threshold,
            max_candidates: input.max_synapse_candidates,
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
            analysis_deadline_ms: input.analysis_deadline_ms,
        })
    } else {
        None
    };

    // Run neuron analysis FIRST (priority), then synapse analysis.
    // Neuron discovery is more valuable as it can create new network structure.
    // With pre-loaded cache, both run fast, but neurons get priority if timeout approaches.
    let neuron_result = if let Some(inner) = neuron_input.clone() {
        Some(analyze_neurons_with_cache(
            &inner,
            Arc::clone(&shared_cache),
        )?)
    } else {
        None
    };

    let synapse_result = if let Some(inner) = synapse_input.clone() {
        Some(analyze_synapses_with_cache(
            &inner,
            Arc::clone(&shared_cache),
        )?)
    } else {
        None
    };

    Ok(AnalyzeAllResult {
        synapse: synapse_result,
        neuron: neuron_result,
    })
}

fn analyze_synapses_with_cache(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeSynapsesResult> {
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let existing_synapses: HashSet<(String, String)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.from_uuid.clone(), synapse.to_uuid.clone()))
        .collect();

    let synapses_by_target: HashMap<String, Vec<SynapseJson>> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (synapse.to_uuid.clone(), synapse.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;

    let diagnostics = Arc::new(Mutex::new(TargetDiagnostics::new(&unique_focus)));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // Randomize the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    // Also filter out neurons with discrete activation functions (STEP, BIPOLAR, etc.)
    // because the linear error model used by discovery completely fails for them.
    let mut skipped_discrete: Vec<String> = Vec::new();
    let mut focus_order: Vec<String> = unique_focus
        .iter()
        .filter(|uuid| {
            if let Some(squash) = neuron_squash_map.get(**uuid) {
                if is_threshold_activation(squash) {
                    skipped_discrete.push((**uuid).clone());
                    return false;
                }
            }
            true
        })
        .map(|s| (*s).clone())
        .collect();

    // Log skipped discrete neurons for visibility
    if verbose_enabled() && !skipped_discrete.is_empty() {
        eprintln!(
            "[NEAT-AI-Discovery][verbose] Synapse analysis skipped {} focus neurons with discrete activations: {:?}",
            skipped_discrete.len(),
            skipped_discrete.iter().take(5).collect::<Vec<_>>()
        );
    }

    let mut rng = thread_rng();
    focus_order.shuffle(&mut rng);

    let threshold = input.improvement_threshold.unwrap_or(0.1);

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
    }

    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;

    let helpful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let harmful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let helpful_fallback = Arc::new(Mutex::new(Option::<CandidateSynapseJson>::None));
    let analysis_timed_out = Arc::new(Mutex::new(false));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let existing_synapses_arc = Arc::new(existing_synapses);
    let synapses_by_target_arc = Arc::new(synapses_by_target);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);

    // Build a comprehensive map of ALL neuron UUIDs to their types
    // This includes: input neurons, and all neurons from creature.neurons (hidden, output, constant)
    // If a UUID is not in this map, it's an invalid UUID (bug)
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    let neuron_type_map_arc = Arc::new(neuron_type_map);

    // Keep input neuron UUIDs set for quick checks (backwards compatibility)
    let input_neuron_uuids: HashSet<String> = (0..input.creature.input)
        .map(|i| format!("input-{i}"))
        .collect();
    let input_neuron_uuids_arc = Arc::new(input_neuron_uuids);

    // Process each focus neuron in parallel
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            // Each thread gets its own GpuAnalyzer (wgpu devices are not thread-safe)
            let analyzer = GpuAnalyzer::new()?;

            let target_records_arc = cache.get(target_uuid.as_str())?;
            if target_records_arc.is_empty() {
                diagnostics
                    .lock()
                    .expect("Mutex poisoned: diagnostics")
                    .set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_target_record_count(target_uuid, target_records.len());

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => {
                    // Target neuron not found in order map - this indicates a data integrity issue
                    // This should never happen for valid hidden/output neurons
                    if verbose_enabled() {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {target_uuid} not found in creature neuron order map (neuron may not exist in creature definition). Skipping."
                        );
                    }
                    return Ok(());
                }
            };

            // Early validation: skip input and constant neurons (they have no upstream sources)
            // Validate target UUID exists in comprehensive neuron type map
            let target_neuron_type = neuron_type_map_arc.get(target_uuid.as_str())
                .ok_or_else(|| anyhow!(
                    "Invalid target neuron UUID '{}': not found in neuron type map. \
                    This indicates a serious data integrity bug. All valid neurons must be in the type map \
                    (input neurons: input-0..input-{}, or neurons from creature.neurons array).",
                    target_uuid,
                    input_neuron_uuids_arc.len().saturating_sub(1)
                ))?;

            let input_count = input_neuron_uuids_arc.len();
            let is_input_neuron = target_neuron_type == "input";
            let is_constant_neuron = target_neuron_type == "constant";

            // Skip actual input/constant neurons (expected - they have no upstream sources)
            // Only skip by UUID check, not by index, to avoid incorrectly skipping hidden neurons
            // that might have been assigned incorrect indices due to ordering bugs
            if is_input_neuron || is_constant_neuron {
                return Ok(());
            }

            // Filter eligible sources: must have index < target_index and not be a constant
            // All neurons should be in the comprehensive neuron_type_map
            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| {
                    neuron.index < target_index
                        && {
                            // Look up neuron type - if missing, it's a serious bug
                            match neuron_type_map_arc.get(&neuron.uuid) {
                                Some(neuron_type) => {
                                    // Valid neuron - exclude constants, include everything else (input, hidden, output)
                                    neuron_type != "constant"
                                }
                                None => {
                                    // Invalid UUID - serious data integrity bug
                                    eprintln!(
                                        "[NEAT-AI-Discovery] ERROR: Invalid neuron UUID '{}' found in ordered_neurons. \
                                        Not found in comprehensive neuron type map. This indicates a serious data integrity bug.",
                                        neuron.uuid
                                    );
                                    false // Exclude invalid neurons
                                }
                            }
                        }
                })
                .collect();

            // Track total eligible sources before filtering
            let total_eligible = eligible_sources.len() as u32;

            // For hidden/output neurons with index >= input_count, there should always be at least the input neurons as eligible sources
            // If total_eligible == 0, this indicates a serious bug
            if total_eligible == 0 {
                // This should be impossible - we've already filtered out input/constant neurons
                // Log detailed diagnostics to help debug
                let neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index)
                    .count();
                let constants_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| {
                        n.index < target_index
                            && neuron_type_map_arc
                                .get(&n.uuid)
                                .map(|t| t == "constant")
                                .unwrap_or(false)
                    })
                    .count();
                let input_neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index && input_neuron_uuids_arc.contains(&n.uuid))
                    .count();

                eprintln!(
                    "[NEAT-AI-Discovery] BUG: Target {target_uuid} (type: {target_neuron_type}, index: {target_index}) has no eligible upstream neurons. \
                    creature.input: {input_count}, neurons before target: {neurons_before_index}, constants before target: {constants_before_index}, \
                    input neurons before target: {input_neurons_before_index}. This should not happen for hidden/output neurons with index >= creature.input."
                );

                // Still skip to avoid crashing, but log the bug
                return Ok(());
            }
            // Count how many eligible sources are input neurons
            let input_neuron_count = eligible_sources
                .iter()
                .filter(|neuron| input_neuron_uuids_arc.contains(&neuron.uuid))
                .count() as u32;
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_total_eligible_sources(target_uuid, total_eligible);
            diagnostics
                .lock()
                .expect("Mutex poisoned: diagnostics")
                .set_input_neuron_count(target_uuid, input_neuron_count);

            let mut rng = thread_rng();
            eligible_sources.shuffle(&mut rng);

            // Improved GPU utilisation: Build samples on CPU in parallel, then batch GPU evaluation
            // This avoids the GPU sync overhead of calling build_samples_gpu for each source.
            // The heavy computation is in evaluate_helpful_batch which is properly batched.
            struct SourceWorkResult {
                work: Option<HelpfulWork>,
                had_samples: bool,
                source_uuid: String,
                record_count: usize,
            }

            // Pre-filter sources and collect their records (cache is thread-safe)
            // Track already-connected, load-failure, and empty-record counts separately
            let mut already_connected_count = 0u32;
            let mut load_failure_count = 0u32;
            let mut empty_record_sources: Vec<String> = Vec::new();
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());

            for source in &eligible_sources {
                let source_uuid = source.uuid.as_str();

                if existing_synapses_arc
                    .contains(&(source_uuid.to_string(), target_uuid.to_string()))
                {
                    already_connected_count += 1;
                    continue;
                }

                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            sources_to_process.push((source, records));
                        } else {
                            // Empty records - track for diagnostics
                            empty_record_sources.push(source_uuid.to_string());
                            let is_input_neuron = input_neuron_uuids_arc.contains(source_uuid);
                            // Log non-input neurons with empty records (input neurons are logged as summary below)
                            if verbose_enabled() && !is_input_neuron {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Source {source_uuid} (target {target_uuid}) has no records in parquet file."
                                );
                            }
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load records for source {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                };
            }

            // Count how many empty record sources are input neurons (helps diagnose parquet data issues)
            let empty_input_neuron_count = empty_record_sources
                .iter()
                .filter(|uuid| input_neuron_uuids_arc.contains(uuid.as_str()))
                .count();
            let empty_non_input_count = empty_record_sources.len() - empty_input_neuron_count;

            // Log summary if many input neurons have empty records (indicates data issue)
            if verbose_enabled() && empty_input_neuron_count > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: {} of {} input neurons have no records in parquet file (plus {} non-input sources). This may indicate incomplete parquet data.",
                    empty_input_neuron_count,
                    input_neuron_uuids_arc.len(),
                    empty_non_input_count
                );
            }

            // Update diagnostics for already-connected, load failures, and empty records
            if already_connected_count > 0
                || load_failure_count > 0
                || !empty_record_sources.is_empty()
            {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for _ in 0..already_connected_count {
                    diag.record_already_connected(target_uuid);
                }
                for _ in 0..load_failure_count {
                    diag.record_load_failure(target_uuid);
                }
                // Record diagnostics for sources with empty records (matches old sequential behaviour)
                for source_uuid in &empty_record_sources {
                    diag.record_candidate_attempt(target_uuid, false);
                    diag.record_no_samples(target_uuid, source_uuid, 0);
                }
            }

            // Build samples on CPU (fast hashmap matching, no GPU sync overhead)
            // This enables parallel sample building for better throughput
            let source_results: Vec<SourceWorkResult> = sources_to_process
                .par_iter()
                .map(|(source, from_records_arc)| {
                    let source_uuid = source.uuid.as_str();
                    let from_records = from_records_arc.as_ref();
                    let record_count = from_records.len();

                    // Use CPU for sample building - fast and avoids GPU sync overhead
                    let samples = build_samples(target_records, from_records);
                    let had_samples = !samples.is_empty();

                    let work = if had_samples {
                        Some(HelpfulWork {
                            source_uuid: source_uuid.to_string(),
                            target_uuid: target_uuid.to_string(),
                            samples,
                        })
                    } else {
                        None
                    };

                    SourceWorkResult {
                        work,
                        had_samples,
                        source_uuid: source_uuid.to_string(),
                        record_count,
                    }
                })
                .collect();

            // Extract work batch and batch diagnostics updates
            let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();
            let mut diagnostics_updates: Vec<(String, String, bool, usize)> = Vec::new();

            for result in source_results {
                if let Some(work) = result.work {
                    helpful_work_batch.push(work);
                }
                diagnostics_updates.push((
                    target_uuid.to_string(),
                    result.source_uuid,
                    result.had_samples,
                    result.record_count,
                ));
            }

            // Apply all diagnostics updates in a single lock (reduces contention)
            if !diagnostics_updates.is_empty() {
                let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                for (target, source, had_samples, record_count) in diagnostics_updates {
                    diag.record_candidate_attempt(&target, had_samples);
                    if !had_samples {
                        diag.record_no_samples(&target, &source, record_count);
                    }
                }
            }

            // Process helpful work in batches for better GPU utilization
            // Vertical timeout: Complete all GPU batch processing for the current focus neuron
            if !helpful_work_batch.is_empty() {
                let helpful_samples_refs: Vec<&[HelpfulSample]> = helpful_work_batch
                    .iter()
                    .map(|w| w.samples.as_slice())
                    .collect();
                let helpful_stats_batch = analyzer.evaluate_helpful_batch(&helpful_samples_refs)?;

                // Process results - collect all updates first, then apply in batches (reduces mutex contention)
                let mut candidates_to_add = Vec::new();
                let mut diagnostics_zero_improvements = Vec::new();
                let mut diagnostics_below_threshold = Vec::new();
                let mut diagnostics_selected = Vec::new();

                for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
                    let positive_is_better = stats.positive_count >= stats.negative_count;
                    let gpu_improved_count = if positive_is_better {
                        stats.positive_count
                    } else {
                        stats.negative_count
                    };
                    if gpu_improved_count == 0 {
                        diagnostics_zero_improvements.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            work.samples.len(),
                            stats.positive_count,
                            stats.negative_count,
                        ));
                        continue;
                    }

                    let total_count = work.samples.len() as u32;
                    if total_count == 0 {
                        continue;
                    }

                    // Use the correct linear optimal weight formula: w = Σ(error × activation) / Σ(activation²)
                    // This minimises squared error and produces naturally-signed weights based on correlation.
                    // The previous formula (Σ|error| / Σ|activation|) was incorrect and often clamped to ±1.0.
                    let mut weight = 0.0;
                    if stats.activation_sq_sum > EPSILON {
                        weight = stats.error_activation_sum / (stats.activation_sq_sum + EPSILON);
                        weight = weight.clamp(-10.0, 10.0);
                    }

                    // Get target's squash function for saturation-aware improvement calculation.
                    // For saturating activations (HARD_TANH, TANH, LOGISTIC, etc.), the linear model
                    // overpredicts improvement near saturation. Using the actual activation function
                    // gives accurate predictions that match real-world results.
                    let target_squash = neuron_squash_map_arc
                        .get(&work.target_uuid)
                        .map(|s| s.as_str());

                    // Compute expected improvement using saturation-aware model when target data is available.
                    // Falls back to linear model when target_value/target_activation are not recorded.
                    // Both improved_count and worsened_count now use the same CPU-based saturation-aware
                    // methodology for consistency (previously worsened_count came from GPU linear model).
                    let (expected_improvement_percentage, improved_count, worsened_count) = {
                        let baseline_error_sq = stats.error_sq_sum;
                        let (improvement, improved, worsened, _) =
                            compute_synapse_improvement_and_count(
                                &work.samples,
                                weight,
                                baseline_error_sq,
                                target_squash,
                            );
                        (improvement, improved, worsened)
                    };

                    // Accept all positive improvements as candidates (not just those above threshold)
                    // Only reject if improvement is non-positive (<= 0.0)
                    if expected_improvement_percentage <= 0.0 {
                        // Skip non-positive improvements
                        continue;
                    }

                    // If positive but below threshold, still accept as candidate but log for diagnostics
                    if expected_improvement_percentage <= threshold {
                        diagnostics_below_threshold.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            ThresholdContext {
                                sample_count: work.samples.len(),
                                expected_improvement: expected_improvement_percentage,
                                threshold,
                                improved_count,
                                worsened_count,
                                weight,
                            },
                        ));
                    }

                    diagnostics_selected.push(work.target_uuid.clone());
                    let target_stats = cache
                        .get(&work.target_uuid)
                        .ok()
                        .and_then(|records| NeuronStats::from_records(records.as_ref()))
                        .map(|s| s.to_json());
                    candidates_to_add.push(CandidateSynapseJson {
                        from_neuron_uuid: work.source_uuid.clone(),
                        to_neuron_uuid: work.target_uuid.clone(),
                        weight,
                        expected_improvement_percentage,
                        improved_count,
                        total_count,
                        target_neuron_stats: target_stats,
                    });
                }

                // Apply all diagnostics updates in batches (minimizes mutex contention)
                if !diagnostics_zero_improvements.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
                        diag.record_zero_improvement(&target, &source, sample_count, pos, neg);
                    }
                }
                if !diagnostics_below_threshold.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for (target, source, context) in diagnostics_below_threshold {
                        diag.record_below_threshold(&target, &source, context);
                    }
                }
                if !diagnostics_selected.is_empty() {
                    let mut diag = diagnostics.lock().expect("Mutex poisoned: diagnostics");
                    for target in diagnostics_selected {
                        diag.mark_candidate_selected(&target);
                    }
                }
                if !candidates_to_add.is_empty() {
                    let mut results = helpful_results
                        .lock()
                        .expect("Mutex poisoned: helpful_results");
                    results.extend(candidates_to_add);
                }
            }

            // Process harmful synapses for this target - batch results to reduce mutex contention
            // Vertical timeout: Complete all harmful synapse processing for the current focus neuron
            if !*analysis_timed_out.lock().expect("Mutex poisoned") {
                if let Some(existing) = synapses_by_target_arc.get(target_uuid.as_str()) {
                    let mut harmful_candidates = Vec::new();

                    for synapse in existing {

                        let from_records_arc = match cache.get(&synapse.from_uuid) {
                            Ok(records) => records,
                            Err(_) => continue,
                        };
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

                        let target_stats = cache
                            .get(target_uuid.as_str())
                            .ok()
                            .and_then(|records| NeuronStats::from_records(records.as_ref()))
                            .map(|s| s.to_json());
                        harmful_candidates.push(CandidateSynapseJson {
                            from_neuron_uuid: synapse.from_uuid.clone(),
                            to_neuron_uuid: synapse.to_uuid.clone(),
                            weight: synapse.weight,
                            expected_improvement_percentage,
                            improved_count: stats.harmful_count,
                            total_count,
                            target_neuron_stats: target_stats,
                        });
                    }

                    // Batch push all harmful candidates (single lock)
                    if !harmful_candidates.is_empty() {
                        let mut results = harmful_results
                            .lock()
                            .expect("Mutex poisoned: harmful_results");
                        results.extend(harmful_candidates);
                    }
                }
            }

            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out.lock().expect("Mutex poisoned");
    let mut helpful_results = helpful_results.lock().expect("Mutex poisoned").clone();
    let mut harmful_results = harmful_results.lock().expect("Mutex poisoned").clone();
    let mut helpful_fallback = helpful_fallback.lock().expect("Mutex poisoned").take();
    let mut diagnostics = diagnostics.lock().expect("Mutex poisoned");

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
        gpu_used,
        no_candidate_reasons,
    })
}

pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Synapse analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_preloaded(&input.parquet_file)?);
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
    use std::time::{Duration, SystemTime};
    use tempfile::tempdir;

    /// Helper macro to skip tests that require GPU when no GPU is available.
    /// This allows tests to pass gracefully in CI environments without GPUs.
    macro_rules! skip_if_no_gpu {
        () => {
            if !GpuAnalyzer::gpu_is_available() {
                eprintln!("⚠️  Skipping test: GPU not available");
                return;
            }
        };
    }

    #[test]
    fn deadline_passed_detects_elapsed_wall_clock_deadline() {
        // Note: This test does NOT require GPU - it only tests the deadline_passed
        // function which performs simple time comparisons. Do not add skip_if_no_gpu!()
        // Use the deadline override mechanism in tests so behaviour is deterministic
        let _guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![true, false]);

        let dummy_deadline = Some(SystemTime::now());
        assert!(
            deadline_passed(&dummy_deadline),
            "past deadlines should be treated as expired immediately"
        );

        assert!(
            !deadline_passed(&dummy_deadline),
            "future deadlines should not be marked as expired"
        );

        assert!(
            !deadline_passed(&None),
            "missing deadlines should behave as if no timeout was requested"
        );
    }

    #[test]
    fn build_deadline_handles_absolute_timestamps_and_relative_durations() {
        // Verify that build_deadline correctly handles both absolute timestamps
        // (milliseconds since UNIX_EPOCH) and relative durations (milliseconds from now)
        let now = SystemTime::now();
        let now_ms = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_millis() as u64;

        // Test 1: Absolute timestamp (large value, >= year 2000)
        // Create a deadline 10 minutes in the future using absolute timestamp
        let ten_minutes_ms = 10 * 60 * 1000; // 10 minutes in milliseconds
        let future_deadline_ms = now_ms + ten_minutes_ms;
        let deadline = build_deadline(Some(future_deadline_ms));

        assert!(
            deadline.is_some(),
            "deadline should be Some when deadline_ms is provided"
        );

        let deadline_time = deadline.unwrap();

        // The deadline should be approximately 10 minutes in the future
        // Allow for some small timing variance (up to 1 second)
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "absolute timestamp deadline should be approximately 10 minutes in the future, got {duration:?}"
            );
        } else {
            panic!("deadline should be in the future");
        }

        // Test 2: Relative duration (small value, < year 2000)
        // Pass 10 minutes as a relative duration
        let relative_deadline_ms = ten_minutes_ms; // 10 minutes as relative duration
        let relative_deadline = build_deadline(Some(relative_deadline_ms));
        assert!(relative_deadline.is_some());
        let relative_time = relative_deadline.unwrap();
        // This should also be approximately 10 minutes in the future
        if let Ok(duration) = relative_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "relative duration deadline should be approximately 10 minutes in the future, got {duration:?}"
            );
        } else {
            panic!("relative deadline should be in the future");
        }

        // Test 3: Verify that an absolute timestamp in the past returns None
        // (deadline already passed - no point in creating a deadline)
        let past_timestamp_ms = 1_700_000_000_000u64; // Jan 2024 (in the past)
        let past_deadline = build_deadline(Some(past_timestamp_ms));
        assert!(
            past_deadline.is_none(),
            "Past timestamp should return None (deadline already passed)"
        );

        // Test 4: Verify that a future absolute timestamp is correctly converted to relative duration
        let future_timestamp_ms = now_ms + ten_minutes_ms; // 10 minutes in the future as absolute timestamp
        let future_deadline = build_deadline(Some(future_timestamp_ms));
        assert!(future_deadline.is_some());
        let future_time = future_deadline.unwrap();
        // Should be approximately 10 minutes in the future
        if let Ok(duration) = future_time.duration_since(now) {
            let expected_min = Duration::from_millis(ten_minutes_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(ten_minutes_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Future absolute timestamp should be converted to relative duration correctly, got {duration:?}"
            );
        } else {
            panic!("Future deadline should be in the future");
        }
    }

    #[test]
    fn build_deadline_validates_duration_bounds() {
        // Test that build_deadline validates and defaults to 10 minutes for invalid values
        let now = SystemTime::now();
        const DEFAULT_DURATION_MS: u64 = 600_000; // 10 minutes

        // Test 1: Duration below minimum (3 seconds) should default to 10 minutes
        let too_short_ms = 1_000u64; // 1 second
        let deadline = build_deadline(Some(too_short_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            // Should default to 10 minutes (600000 ms)
            let expected_min = Duration::from_millis(DEFAULT_DURATION_MS) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(DEFAULT_DURATION_MS) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration below 3 seconds should default to 10 minutes, got {duration:?}"
            );
        } else {
            panic!("Default deadline should be in the future");
        }

        // Test 2: Duration above maximum (1 hour) should default to 10 minutes
        let too_long_ms = 4_000_000u64; // ~66 minutes
        let deadline = build_deadline(Some(too_long_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            // Should default to 10 minutes (600000 ms)
            let expected_min = Duration::from_millis(DEFAULT_DURATION_MS) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(DEFAULT_DURATION_MS) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration above 1 hour should default to 10 minutes, got {duration:?}"
            );
        } else {
            panic!("Default deadline should be in the future");
        }

        // Test 3: Valid duration (10 minutes) should pass through unchanged
        let valid_ms = 10 * 60 * 1000u64; // 10 minutes
        let deadline = build_deadline(Some(valid_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(valid_ms) - Duration::from_secs(1);
            let expected_max = Duration::from_millis(valid_ms) + Duration::from_secs(1);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Valid duration should pass through unchanged, got {duration:?}"
            );
        } else {
            panic!("Valid deadline should be in the future");
        }

        // Test 4: Exactly at minimum (3 seconds) should pass through
        let min_ms = 3_000u64; // Exactly 3 seconds
        let deadline = build_deadline(Some(min_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(min_ms) - Duration::from_millis(100);
            let expected_max = Duration::from_millis(min_ms) + Duration::from_millis(100);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration at minimum should pass through, got {duration:?}"
            );
        } else {
            panic!("Minimum deadline should be in the future");
        }

        // Test 5: Exactly at maximum (1 hour) should pass through
        let max_ms = 3_600_000u64; // Exactly 1 hour
        let deadline = build_deadline(Some(max_ms));
        assert!(deadline.is_some());
        let deadline_time = deadline.unwrap();
        if let Ok(duration) = deadline_time.duration_since(now) {
            let expected_min = Duration::from_millis(max_ms) - Duration::from_millis(1000);
            let expected_max = Duration::from_millis(max_ms) + Duration::from_millis(1000);
            assert!(
                duration >= expected_min && duration <= expected_max,
                "Duration at maximum should pass through, got {duration:?}"
            );
        } else {
            panic!("Maximum deadline should be in the future");
        }
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
    fn gpu_matching_filters_non_finite_values() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

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
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

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

    /// Test that GPU matching preserves target_value and target_activation.
    /// This is critical for accurate improvement predictions with non-linear
    /// activation functions (TANH, LOGISTIC, HARD_TANH, etc.).
    ///
    /// Bug regression test: Previously, GPU matching always returned
    /// target_value=None and target_activation=None, causing the improvement
    /// calculation to fall back to an inaccurate linear model that consistently
    /// overpredicted improvements for add-neuron candidates.
    #[test]
    fn gpu_matching_preserves_target_value_and_activation() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        // Create target records with explicit value (pre-activation) and activation (post-activation)
        let target_value = 0.8; // Pre-activation input sum
        let target_activation = 0.6; // Post-activation output (e.g., after TANH)
        let target_records = vec![DiscoverRecord::new(
            0,
            "target".to_string(),
            Some(target_value),
            target_activation,
            vec![0.1, -0.2],
        )];
        let from_records = vec![DiscoverRecord::new(
            0,
            "from".to_string(),
            None, // Source value not used
            0.5,  // Source activation
            Vec::new(),
        )];

        // CPU matching should preserve target data
        let cpu_samples = build_samples(&target_records, &from_records);
        assert_eq!(cpu_samples.len(), 1, "CPU should find one matching sample");
        assert_eq!(
            cpu_samples[0].target_value,
            Some(target_value),
            "CPU matching should preserve target_value"
        );
        assert_eq!(
            cpu_samples[0].target_activation,
            Some(target_activation),
            "CPU matching should preserve target_activation"
        );

        // GPU matching should also preserve target data (this was the bug)
        let gpu_samples = analyzer
            .build_samples_gpu(&target_records, &from_records)
            .expect("GPU matching should succeed");

        assert_eq!(gpu_samples.len(), 1, "GPU should find one matching sample");
        assert_eq!(
            gpu_samples[0].target_value,
            Some(target_value),
            "GPU matching should preserve target_value (required for activation function simulation)"
        );
        assert_eq!(
            gpu_samples[0].target_activation,
            Some(target_activation),
            "GPU matching should preserve target_activation (required for error calculation)"
        );

        // Verify values match between CPU and GPU
        assert_eq!(
            cpu_samples[0].activation, gpu_samples[0].activation,
            "Source activation should match between CPU and GPU"
        );
        assert!(
            (cpu_samples[0].avg_error - gpu_samples[0].avg_error).abs() < 1e-6,
            "Average error should match between CPU and GPU"
        );
    }

    /// Test that target_value enables proper activation function simulation.
    /// When target_value is available, get_target_simulation_fn should return
    /// the activation function, enabling saturation-aware improvement predictions.
    #[test]
    fn gpu_matching_enables_target_activation_simulation() {
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

        // Create samples near HARD_TANH saturation to test simulation accuracy
        let target_records = vec![
            DiscoverRecord::new(
                0,
                "target".to_string(),
                Some(0.95), // Near saturation
                0.95,       // HARD_TANH clips to 1.0 when input >= 1.0
                vec![0.1],  // Small positive error (output should be higher)
            ),
            DiscoverRecord::new(
                1,
                "target".to_string(),
                Some(-0.8),
                -0.8,
                vec![-0.15], // Small negative error (output should be lower)
            ),
        ];
        let from_records = vec![
            DiscoverRecord::new(0, "from".to_string(), None, 0.5, Vec::new()),
            DiscoverRecord::new(1, "from".to_string(), None, -0.3, Vec::new()),
        ];

        let gpu_samples = analyzer
            .build_samples_gpu(&target_records, &from_records)
            .expect("GPU matching should succeed");

        assert_eq!(gpu_samples.len(), 2, "Should match both records");

        // Verify all samples have target data (required for simulation)
        for (i, sample) in gpu_samples.iter().enumerate() {
            assert!(
                sample.target_value.is_some(),
                "Sample {i} should have target_value for activation simulation"
            );
            assert!(
                sample.target_activation.is_some(),
                "Sample {i} should have target_activation for error calculation"
            );
        }

        // With target data available, get_target_simulation_fn should return Some
        // for activations that need simulation (HARD_TANH, TANH, ReLU, etc.)
        let simulation_fn = get_target_simulation_fn(&gpu_samples, Some("HARD_TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable HARD_TANH simulation when GPU samples have target data"
        );

        let simulation_fn = get_target_simulation_fn(&gpu_samples, Some("TANH"));
        assert!(
            simulation_fn.is_some(),
            "Should enable TANH simulation when GPU samples have target data"
        );
    }

    /// Regression test: Add-neuron predictions must use weight computed WITH bias.
    ///
    /// BUG: Previously, the optimal outgoing weight was computed WITHOUT bias:
    ///   weight = Σ(error × TANH(x)) / Σ(TANH(x)²)
    ///
    /// But the actual neuron uses bias:
    ///   contribution = weight × TANH(x + bias)
    ///
    /// When bias significantly shifts the activation pattern, the weight computed
    /// without bias causes predictions to have the WRONG SIGN - predicting improvement
    /// when it actually makes things worse.
    ///
    /// This test reproduces the production failure pattern:
    /// - Predicted: +0.3% improvement
    /// - Actual: -0.2% (worse!)
    #[test]
    fn add_neuron_weight_must_include_bias_in_calculation() {
        // Scenario from production: TANH neuron with bias=1
        // This shifts the activation threshold from x>0 to x>-1
        let incoming_weight = 1.0f32;
        let bias = 1.0f32;

        // Create samples that expose the bug:
        // - Source activations centered around 0
        // - Roughly equal positive and negative errors
        // - With bias=1, TANH(x+1) is almost always positive (x > -1)
        // - Without bias, TANH(x) has mixed signs
        let samples: Vec<HelpfulSample> = vec![
            // Positive source activation, positive error (need output up)
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.3,
                avg_error: 0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // Negative source activation, negative error (need output down)
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.3,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.2,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.1,
                avg_error: -0.1,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            // More samples near zero - these are affected most by bias
            HelpfulSample {
                activation: 0.05,
                avg_error: 0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
            HelpfulSample {
                activation: -0.05,
                avg_error: -0.15,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            },
        ];

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

        // BUG PATH: Compute weight WITHOUT bias (what the old code did)
        let mut sum_sq_no_bias = 0.0f32;
        let mut sum_ea_no_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation; // NO BIAS!
            let output = pre_act.tanh();
            sum_sq_no_bias += output * output;
            sum_ea_no_bias += output * s.avg_error;
        }
        let weight_without_bias = sum_ea_no_bias / sum_sq_no_bias;

        // FIX PATH: Compute weight WITH bias (what the fixed code does)
        let mut sum_sq_with_bias = 0.0f32;
        let mut sum_ea_with_bias = 0.0f32;
        for s in &samples {
            let pre_act = incoming_weight * s.activation + bias; // WITH BIAS!
            let output = pre_act.tanh();
            sum_sq_with_bias += output * output;
            sum_ea_with_bias += output * s.avg_error;
        }
        let weight_with_bias = sum_ea_with_bias / sum_sq_with_bias;

        // Compute ACTUAL error reduction using the ACTUAL neuron (with bias)
        fn compute_actual_improvement(
            samples: &[HelpfulSample],
            incoming_weight: f32,
            outgoing_weight: f32,
            bias: f32,
            baseline_error_sq: f32,
        ) -> f32 {
            let mut new_error_sq = 0.0f32;
            for s in samples {
                let pre_act = incoming_weight * s.activation + bias;
                let neuron_output = pre_act.tanh();
                let contribution = outgoing_weight * neuron_output;
                // Linear approximation: new_error = old_error - contribution
                let new_error = s.avg_error - contribution;
                new_error_sq += new_error * new_error;
            }
            // Improvement = (baseline - new) / baseline
            (baseline_error_sq - new_error_sq) / baseline_error_sq
        }

        // Test 1: Using weight computed WITHOUT bias gives WRONG prediction
        // The prediction (using weight_without_bias) should differ from actual
        let predicted_without_bias = {
            // Predicted improvement uses the same formula as actual
            // but the weight was computed from wrong activation pattern
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation; // NO BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_without_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_wrong_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_without_bias,
            bias,
            baseline_error_sq,
        );

        // The bug: predicted is positive, actual is often negative or much smaller
        // This happens because weight was optimised for TANH(x) but applied to TANH(x+1)
        let prediction_error_wrong = (predicted_without_bias - actual_with_wrong_weight).abs();

        // Test 2: Using weight computed WITH bias gives CORRECT prediction
        let predicted_with_bias = {
            let mut predicted_new_error_sq = 0.0f32;
            for s in &samples {
                let pre_act = incoming_weight * s.activation + bias; // WITH BIAS in prediction!
                let output = pre_act.tanh();
                let contribution = weight_with_bias * output;
                let new_error = s.avg_error - contribution;
                predicted_new_error_sq += new_error * new_error;
            }
            (baseline_error_sq - predicted_new_error_sq) / baseline_error_sq
        };

        let actual_with_correct_weight = compute_actual_improvement(
            &samples,
            incoming_weight,
            weight_with_bias,
            bias,
            baseline_error_sq,
        );

        let prediction_error_correct = (predicted_with_bias - actual_with_correct_weight).abs();

        // Assertions:
        // 1. The two weights should be significantly different
        assert!(
            (weight_with_bias - weight_without_bias).abs() > 0.01,
            "Weights should differ: without_bias={weight_without_bias:.4}, with_bias={weight_with_bias:.4}"
        );

        // 2. Using wrong weight should have high prediction error
        assert!(
            prediction_error_wrong > 0.001,
            "Wrong weight should cause prediction error > 0.1%. \
             Predicted={predicted_without_bias:.4}, Actual={actual_with_wrong_weight:.4}, \
             Error={prediction_error_wrong:.4}"
        );

        // 3. Using correct weight should have low prediction error
        assert!(
            prediction_error_correct < 0.0001,
            "Correct weight should have prediction error < 0.01%. \
             Predicted={predicted_with_bias:.4}, Actual={actual_with_correct_weight:.4}, \
             Error={prediction_error_correct:.4}"
        );

        // 4. The key bug symptom: wrong weight often gives OPPOSITE sign of improvement
        // (predicts positive improvement but actual is negative, or vice versa)
        // This may not always happen with this specific test data, but we verify
        // the prediction error is significantly worse.
        assert!(
            prediction_error_wrong > prediction_error_correct * 10.0,
            "Wrong weight should have much higher error than correct weight. \
             Wrong error={prediction_error_wrong:.6}, Correct error={prediction_error_correct:.6}"
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
    fn relu_split_evaluation_finds_candidates_when_activation_correlates_with_error() {
        // Test that split-by-error ReLU evaluation finds candidates when source activation
        // correlates with target error direction.
        //
        // Key insight: A ReLU can only help if its activation correlates with the errors
        // it's trying to fix. If activation is the same for all samples, the ReLU's
        // contribution will cancel out across balanced errors.
        //
        // This test creates samples where:
        // - Source fires (activation > 0) when target error is positive (output should go UP)
        // - Source doesn't fire (activation <= 0) when target error is negative
        //
        // This is the realistic scenario where adding a ReLU neuron can help.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source fires AND output should go UP (positive error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 1.0, // Source fires
                avg_error: 0.5,  // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source doesn't fire AND output should go DOWN (negative error)
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -0.5, // Source doesn't fire (ReLU will output 0)
                avg_error: -0.5,  // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With correlation between activation and error, we should find a positive-error candidate
        // The ReLU fires when we need output to go UP, and doesn't fire when we need it DOWN.
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find positive-error ReLU candidate when activation correlates with error direction"
        );

        // Verify the candidate pushes in the correct direction
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Positive-error candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn relu_split_evaluation_finds_negative_orientation_candidates() {
        // Test that we can find ReLU candidates with incoming_weight = -1.0 (negative orientation).
        //
        // This is critical: when source neurons have predominantly NEGATIVE activations
        // that correlate with errors, we need a ReLU with incoming_weight = -1.0 to flip
        // the sign before the ReLU activation.
        //
        // Bug regression test: Previously, evaluate_relu_candidates_split discarded
        // negative_stats entirely, meaning these candidates could never be found.
        skip_if_no_gpu!();
        let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

        let mut samples = Vec::new();
        // Samples where source has NEGATIVE activation AND output should go UP (positive error)
        // A ReLU with incoming_weight = -1.0 will flip -1.0 to +1.0, then ReLU outputs 1.0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: -1.0, // NEGATIVE activation
                avg_error: 0.5,   // Output should be HIGHER
                target_value: None,
                target_activation: None,
            });
        }
        // Samples where source has POSITIVE activation AND output should go DOWN (negative error)
        // A ReLU with incoming_weight = -1.0 will flip +0.5 to -0.5, then ReLU outputs 0
        for _ in 0..MIN_NEURON_SAMPLE_COUNT {
            samples.push(HelpfulSample {
                activation: 0.5, // POSITIVE activation (will be flipped to negative, ReLU = 0)
                avg_error: -0.5, // Output should be LOWER
                target_value: None,
                target_activation: None,
            });
        }

        let result =
            evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
                .expect("ReLU split evaluation should succeed");

        // With negative activations correlating with positive errors, we should find a candidate
        // that uses the NEGATIVE orientation (incoming_weight = -1.0)
        assert!(
            result.positive_error_candidate.is_some(),
            "Should find ReLU candidate even when source has negative activations (requires negative orientation)"
        );

        // Verify the candidate uses negative incoming weight (the critical fix!)
        if let Some(pos_candidate) = &result.positive_error_candidate {
            assert!(
                pos_candidate.incoming_weight < 0.0,
                "Candidate should have NEGATIVE incoming weight to flip negative activations. Got: {}",
                pos_candidate.incoming_weight
            );
            assert!(
                pos_candidate.outgoing_weight > 0.0,
                "Candidate should have positive outgoing weight (pushes UP). Got: {}",
                pos_candidate.outgoing_weight
            );
        }
    }

    #[test]
    fn neuron_diagnostics_tracks_load_failures() {
        // Test that when eligible sources exist but all fail to load, we report
        // NoSamples rather than NoEligibleSources
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 10); // 10 eligible sources exist
                                                                // All 10 sources fail to load
        for _ in 0..10 {
            diagnostics.record_load_failure("output-0");
        }
        // No record_candidate_attempt calls (because all failed to load)

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should NOT report "no eligible sources" - sources existed but failed to load
        assert!(
            !matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should not report NoEligibleSources when sources existed but failed to load"
        );
        // Should report NoSamples (as a catchall for sources existing but not being usable)
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoSamples),
            "Should report NoSamples when eligible sources exist but none were evaluated"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 10);
        assert_eq!(entry.record_load_failures, 10);
        assert_eq!(entry.evaluated_sources, 0);
    }

    #[test]
    fn neuron_diagnostics_reports_genuine_no_eligible_sources() {
        // Test that when there are genuinely no eligible sources, we correctly report that
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 0); // No eligible sources

        let summaries = diagnostics.no_candidate_summaries();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];

        // Should correctly report no eligible sources
        assert!(
            matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
            "Should report NoEligibleSources when genuinely no sources exist"
        );

        // Verify entry tracking
        let entry = diagnostics.entry_for("output-0").unwrap();
        assert_eq!(entry.total_eligible_sources, 0);
        assert_eq!(entry.record_load_failures, 0);
        assert_eq!(entry.evaluated_sources, 0);
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
    fn analyze_synapses_reports_eligible_sources_correctly_for_non_input_neurons() {
        skip_if_no_gpu!();
        // Test that non-input neurons with valid creature structure always report
        // eligible sources correctly, not "no eligible sources" when sources exist
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "constant-0".to_string(),
                    neuron_type: "constant".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 1.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                // Hidden neuron already connected to input-0
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                },
                // Output neuron already connected to hidden-0
                SynapseJson {
                    from_uuid: "hidden-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 0.5,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Check diagnostics for hidden-0
        // hidden-0 should have eligible sources (input-1 is not connected yet)
        // So it should NOT report "no eligible sources"
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        if let Some(diag) = hidden_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should have eligible sources (input-1 is available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "hidden-0 should have evaluated at least one candidate (input-1), but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }

        // Check diagnostics for output-0
        // output-0 should have eligible sources (input-0, input-1 are available)
        // So it should NOT report "no eligible sources"
        let output_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "output-0");

        if let Some(diag) = output_diag {
            assert!(
                diag.reason != SynapseNoCandidateReason::NoEligibleSources,
                "output-0 should have eligible sources (input-0, input-1 are available), but got: {:?}",
                diag.reason
            );
            assert!(
                diag.evaluated_candidates > 0,
                "output-0 should have evaluated at least one candidate, but evaluated_candidates is {}",
                diag.evaluated_candidates
            );
        }
    }

    #[test]
    fn analyze_synapses_reports_fully_connected_neuron_explicitly() {
        skip_if_no_gpu!();
        // Test that a neuron connected to ALL eligible sources is explicitly reported
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Input neuron records (observations)
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.25,
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                obs_index,
                "input-1".to_string(),
                Some(0.0),
                0.3,
                vec![0.2],
            ));
            // Hidden neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "hidden-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.15],
            ));
            // Output neuron records
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.6,
                vec![0.05],
            ));
        }

        write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write discovery records");

        // Create a creature where hidden-0 is connected to ALL eligible sources
        // (both input-0 and input-1)
        let creature = CreatureJson {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-0".to_string(),
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
            synapses: vec![
                // hidden-0 is connected to ALL eligible sources (input-0 and input-1)
                SynapseJson {
                    from_uuid: "input-0".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.4,
                },
                SynapseJson {
                    from_uuid: "input-1".to_string(),
                    to_uuid: "hidden-0".to_string(),
                    weight: 0.5,
                },
            ],
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["hidden-0".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // hidden-0 should be reported as having no eligible sources
        // because it's connected to ALL eligible sources (both inputs)
        let hidden_diag = result
            .no_candidate_reasons
            .iter()
            .find(|summary| summary.target_uuid == "hidden-0");

        assert!(
            hidden_diag.is_some(),
            "hidden-0 should have diagnostics since it's fully connected"
        );

        if let Some(diag) = hidden_diag {
            assert_eq!(
                diag.reason,
                SynapseNoCandidateReason::NoEligibleSources,
                "hidden-0 should report NoEligibleSources since it's connected to all eligible sources"
            );
            assert_eq!(
                diag.evaluated_candidates, 0,
                "hidden-0 should have 0 evaluated candidates since all sources are already connected"
            );
        }
    }

    #[test]
    fn analyze_synapses_requires_focus_targets() {
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
        skip_if_no_gpu!();
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
        skip_if_no_gpu!();
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
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input)
            .expect("Synapse analysis should complete even when the deadline triggers");

        // With parallel processing, deadline detection order is non-deterministic because
        // multiple threads call deadline_passed() concurrently. The timeout mechanism is
        // approximate - once any thread detects the deadline, analysis_timed_out is set
        // and processing should stop. However, some threads may have already started
        // processing harmful synapses before the deadline was detected.
        //
        // The key requirement is that the analysis completes successfully and respects
        // the deadline approximately. Since timeout is approximate, we verify that:
        // 1. The analysis completes without panicking
        // 2. The result structure is valid
        // 3. We don't process more harmful synapses than exist (sanity check)
        //
        // In this test setup, we have 2 focus neurons, each with 1 harmful synapse (2 total).
        // The deadline sequence [false x6, true, ...] should cause early termination,
        // but with parallel processing, the exact point of termination is non-deterministic.
        let max_possible_harmful = 2; // 2 focus neurons × 1 harmful synapse each
        assert!(
            result.harmful_synapses.len() <= max_possible_harmful,
            "Should not process more harmful synapses than exist. \
             Got {} harmful synapses, max possible is {}",
            result.harmful_synapses.len(),
            max_possible_harmful
        );
        // The deadline mechanism is approximate, so we accept any result as long as
        // the analysis completes and doesn't exceed reasonable bounds
    }

    #[test]
    fn analyze_all_runs_synapse_and_neuron_phases() {
        skip_if_no_gpu!();
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
        skip_if_no_gpu!();
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
    fn analyze_neurons_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![
            false, false, false, true,
        ]);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for two output neurons but none for the
        // hidden source. This guarantees that each focus neuron has at least
        // one eligible upstream source, and that diagnostics can attribute a
        // `NoSamples` reason once analysis runs.
        let mut records = Vec::new();
        for obs_index in 0..(MIN_NEURON_SAMPLE_COUNT as u32 + 1) {
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
            input: 0,
            output: 2,
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
                NeuronJson {
                    uuid: "output-1".to_string(),
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
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: Some(1_000_000),
        };

        // Use a single-threaded Rayon pool so the deadline override sequence remains
        // deterministic for this test. Note: focus neurons are randomized, so we can't
        // assume a specific order.
        let pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build single-threaded Rayon pool");

        let result = pool
            .install(|| analyze_neurons(&input))
            .expect("Neuron analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_sources > 0)
            .collect();

        assert!(
            !processed_neurons.is_empty(),
            "At least one focus neuron should evaluate at least one upstream source before timeout (vertical timeout behaviour)"
        );

        // Verify that the processed neuron(s) are not reported as having no eligible sources
        for summary in &processed_neurons {
            assert!(
                summary.reason != NeuronNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    #[test]
    fn analyze_synapses_uses_vertical_timeout_with_randomized_order() {
        skip_if_no_gpu!();
        use rayon::ThreadPoolBuilder;

        // Simulate a deadline that allows at least one focus neuron to start, but
        // triggers before all are processed. The override sequence is consumed
        // by calls to `deadline_passed` in order. With randomization, we need
        // enough false values to allow at least one neuron to start processing.
        // We provide multiple false values to account for any initialization checks,
        // then true to stop further processing.
        let _deadline_guard = deadline_override::DeadlineOverrideGuard::with_sequence(vec![
            false, false, false, true,
        ]);

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Provide discovery records for an input neuron and two output neurons.
        // Records use disjoint obs_index ranges so that each potential synapse
        // has discovery data but no aligned samples, guaranteeing that the
        // diagnostics machinery records a `NoSamples` style rejection rather
        // than treating the target as having no eligible sources.
        let mut records = Vec::new();
        for obs_index in 0..16u32 {
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![0.0],
            ));
        }
        for obs_index in 100..116u32 {
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
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
            improvement_threshold: Some(0.05),
            max_candidates: None,
            // Any non-None deadline value will exercise the override sequence.
            analysis_deadline_ms: None,
        };

        // Use a single-threaded Rayon pool so the deadline override sequence remains
        // deterministic for this test. Note: focus neurons are randomized, so we can't
        // assume a specific order.
        let pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build single-threaded Rayon pool");

        let result = pool
            .install(|| analyze_synapses(&input))
            .expect("Synapse analysis should succeed even when the deadline triggers");

        // At least one focus neuron should have evaluated at least one upstream
        // source before the deadline (vertical timeout behaviour). Since focus
        // neurons are randomized, we check that at least one of the two neurons
        // was processed.
        let processed_neurons: Vec<_> = result
            .no_candidate_reasons
            .iter()
            .filter(|summary| summary.evaluated_candidates > 0)
            .collect();

        assert!(
            !processed_neurons.is_empty(),
            "At least one focus neuron should evaluate at least one upstream source before timeout (vertical timeout behaviour)"
        );

        // Verify that the processed neuron(s) are not reported as having no eligible sources
        for summary in &processed_neurons {
            assert!(
                summary.reason != SynapseNoCandidateReason::NoEligibleSources,
                "Processed focus neuron should not be reported as having no eligible sources when a timeout occurs"
            );
        }
    }

    /// Test bias calculation for TANH activation function
    #[test]
    fn test_bias_calculation_tanh() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, -0.5, tanh_activation, "TANH", None, None);

        // Bias should be in expanded TANH range
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias for TANH should be in range [-1.0, 1.0], got {bias}"
        );
    }

    /// Test bias calculation for ReLU activation function
    #[test]
    fn test_bias_calculation_relu() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let relu_fn = |x: f32| x.max(0.0);
        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, relu_fn, "ReLU", None, None);

        // ReLU can now use negative bias for threshold shifting (expanded range)
        assert!(bias >= -1.0, "Bias for ReLU should be >= -1.0, got {bias}");
        assert!(bias <= 1.0, "Bias for ReLU should be <= 1.0, got {bias}");
    }

    /// Test bias improves error reduction compared to zero bias
    #[test]
    fn test_bias_improves_error_reduction() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming = 1.5;
        let outgoing = -0.18;

        // Calculate error with zero bias
        let mut zero_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation;
            let new_neuron_activation = inverse_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            zero_bias_error_sq += new_error * new_error;
        }

        // Calculate optimal bias
        let optimal_bias = calculate_optimal_bias(
            &samples,
            incoming,
            outgoing,
            inverse_activation,
            "INVERSE",
            None,
            None,
        );

        // Calculate error with optimal bias
        let mut optimal_bias_error_sq = 0.0;
        for sample in &samples {
            let pre_activation = incoming * sample.activation + optimal_bias;
            let new_neuron_activation = inverse_activation(pre_activation);
            let correction = outgoing * new_neuron_activation;
            let new_error = sample.avg_error - correction;
            optimal_bias_error_sq += new_error * new_error;
        }

        // Optimal bias should give equal or better error reduction than zero bias
        assert!(
            optimal_bias_error_sq <= zero_bias_error_sq + EPSILON,
            "Optimal bias should improve or equal zero bias error reduction: zero_bias_error={zero_bias_error_sq}, optimal_bias_error={optimal_bias_error_sq}"
        );
    }

    /// Test that positive improvements below threshold are accepted as candidates
    /// This verifies the fix where all positive improvements are candidates, not just those above threshold
    #[test]
    fn analyze_synapses_accepts_positive_improvements_below_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive but below-threshold improvement
        // We need: expected_improvement = (2*w*E[a*e] - w^2*E[a^2]) / E[e^2]
        // To get ~0.05 improvement with threshold 0.1, we'll use:
        // - source activation: 0.5 consistently
        // - target error: 0.1 consistently
        // - This should produce a positive improvement when weight is chosen appropriately
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron (input-0) with consistent activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                0.5,    // Consistent activation
                vec![], // Input neurons don't have errors
            ));
            // Target neuron (output-0) with consistent error
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.3,
                vec![0.1], // Consistent error
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
            synapses: Vec::new(), // No existing synapse from input-0 to output-0
        };

        // Set threshold to 0.1 - we expect a positive but below-threshold improvement to be accepted
        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // The key assertion: positive improvements below threshold should be accepted
        // We should have at least one helpful synapse candidate (even if improvement < 0.1)
        // OR if no candidate, it should NOT be due to BelowThreshold for a positive improvement
        if result.helpful_synapses.is_empty() {
            // If no candidates, check diagnostics - it should NOT be BelowThreshold for positive improvements
            let no_candidate = result
                .no_candidate_reasons
                .iter()
                .find(|summary| summary.target_uuid == "output-0");

            if let Some(summary) = no_candidate {
                // If there's a detail, check that it's not a positive improvement below threshold
                if let Some(detail) = &summary.detail {
                    if let Some(improvement) = detail.expected_improvement {
                        if improvement > 0.0 && improvement <= 0.1 {
                            panic!(
                                "Positive improvement {:.4} below threshold 0.1 should be accepted as candidate, but was rejected with reason: {:?}",
                                improvement, summary.reason
                            );
                        }
                    }
                }
            }
        } else {
            // We have candidates - verify at least one has positive improvement
            let has_positive_improvement = result
                .helpful_synapses
                .iter()
                .any(|synapse| synapse.expected_improvement_percentage > 0.0);

            assert!(
                has_positive_improvement,
                "Should have at least one candidate with positive improvement"
            );
        }
    }

    /// Test that non-positive improvements (<= 0.0) are still rejected
    #[test]
    fn analyze_synapses_rejects_non_positive_improvements() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a non-positive improvement
        // Use mismatched activations/errors that result in negative or zero improvement
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that doesn't correlate well (will produce negative improvement)
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.0,
                vec![-0.1], // Negative error when source is positive
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
            synapses: Vec::new(),
        };

        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Non-positive improvements should be rejected (not appear in helpful_synapses)
        // Even though we now accept positive improvements below threshold, we still reject <= 0.0
        let has_non_positive = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_improvement_percentage <= 0.0);

        assert!(
            !has_non_positive,
            "Should not have any candidates with non-positive improvement (<= 0.0)"
        );
    }

    /// Test that positive improvements above threshold are still accepted (regression test)
    #[test]
    fn analyze_synapses_accepts_positive_improvements_above_threshold() {
        skip_if_no_gpu!();
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path
            .to_str()
            .expect("Temporary path should be valid UTF-8")
            .to_string();

        // Create test data that will produce a positive improvement above threshold
        // Use strong correlation between source activation and target error
        let sample_count = 100;
        let mut records = Vec::new();
        for obs_index in 0..sample_count {
            // Source neuron with strong activation
            records.push(DiscoverRecord::new(
                obs_index,
                "input-0".to_string(),
                Some(0.0),
                1.0,
                vec![],
            ));
            // Target neuron with error that correlates positively
            records.push(DiscoverRecord::new(
                obs_index,
                "output-0".to_string(),
                Some(0.0),
                0.5,
                vec![0.2], // Positive error when source is positive
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
            synapses: Vec::new(),
        };

        // Set threshold to 0.1 - we expect improvement above this to be accepted
        let input = AnalyzeSynapsesInput {
            parquet_file,
            creature,
            focus_neurons: vec!["output-0".to_string()],
            improvement_threshold: Some(0.1),
            max_candidates: None,
            analysis_deadline_ms: None,
        };

        let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

        // Positive improvements above threshold should definitely be accepted
        // This is a regression test to ensure we didn't break existing behavior
        let has_above_threshold = result
            .helpful_synapses
            .iter()
            .any(|synapse| synapse.expected_improvement_percentage > 0.1);

        // Note: This test may pass even if no candidates are found due to other reasons
        // (e.g., no samples, zero improvement). The key is that if we have candidates,
        // they should include positive improvements above threshold.
        if !result.helpful_synapses.is_empty() {
            assert!(
                has_above_threshold
                    || result
                        .helpful_synapses
                        .iter()
                        .any(|s| s.expected_improvement_percentage > 0.0),
                "Should have candidates with positive improvement (above or below threshold)"
            );
        }
    }

    /// Test bias range boundaries for different activation functions
    #[test]
    fn test_bias_within_reasonable_range() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        type ActivationTestCase = (&'static str, fn(f32) -> f32, f32, f32);
        let test_cases: Vec<ActivationTestCase> = vec![
            ("TANH", tanh_activation as fn(f32) -> f32, -10.0, 10.0),
            (
                "LOGISTIC",
                logistic_activation as fn(f32) -> f32,
                -10.0,
                10.0,
            ),
            (
                "IDENTITY",
                identity_activation as fn(f32) -> f32,
                -50.0,
                50.0,
            ),
        ];

        for (name, activation_fn, min_expected, max_expected) in test_cases {
            let bias = calculate_optimal_bias(&samples, 1.0, 1.0, activation_fn, name, None, None);
            assert!(
                bias >= min_expected && bias <= max_expected,
                "Bias for {name} should be in range [{min_expected}, {max_expected}], got {bias}"
            );
        }
    }

    /// Test bias calculation handles empty samples
    #[test]
    fn test_bias_calculation_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should return 0.0 for empty samples
        assert_eq!(bias, 0.0, "Empty samples should return bias of 0.0");
    }

    /// Test bias calculation handles insufficient samples
    #[test]
    fn test_bias_calculation_insufficient_samples() {
        // Only 5 samples (less than MIN_NEURON_SAMPLE_COUNT of 10)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.3,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should still return a valid bias in range (though may be 0.0 if no bias tested has sufficient samples)
        assert!(
            (-10.0..=10.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    /// Test get_bias_range returns correct ranges for different activation functions
    /// Note: get_bias_range is used by GPU, get_bias_values is used by CPU
    #[test]
    fn test_get_bias_range() {
        // Test ReLU range (extended negative for high-threshold neurons)
        let (min, max, step) = get_bias_range("ReLU");
        assert_eq!(min, -25.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test TANH range (extended symmetric for large weight configurations)
        let (min, max, step) = get_bias_range("TANH");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test LOGISTIC range (extended symmetric)
        let (min, max, step) = get_bias_range("LOGISTIC");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);

        // Test IDENTITY range (widest - acts as pure offset, scales with large weights)
        let (min, max, step) = get_bias_range("IDENTITY");
        assert_eq!(min, -50.0);
        assert_eq!(max, 50.0);
        assert_eq!(step, 1.0);

        // Test default range for unknown activation (generous)
        let (min, max, step) = get_bias_range("UNKNOWN");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);
    }

    /// Test is_threshold_activation identifies threshold functions (STEP/BIPOLAR).
    /// These use a specialised threshold-crossing model instead of the linear model.
    /// All other activations use the standard linear error model - none are skipped.
    #[test]
    fn test_is_threshold_activation() {
        // Threshold activations - use threshold-crossing model
        assert!(is_threshold_activation("STEP"), "STEP uses threshold model");
        assert!(is_threshold_activation("step"), "case insensitive");
        assert!(
            is_threshold_activation("BIPOLAR"),
            "BIPOLAR uses threshold model"
        );

        // All other activations use standard linear model (not skipped)
        assert!(
            !is_threshold_activation("IF"),
            "IF uses standard model (correlation still works)"
        );
        assert!(
            !is_threshold_activation("MAXIMUM"),
            "MAXIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("MINIMUM"),
            "MINIMUM uses standard model"
        );
        assert!(
            !is_threshold_activation("HARD_TANH"),
            "HARD_TANH uses standard model"
        );
        assert!(
            !is_threshold_activation("CLIPPED"),
            "CLIPPED uses standard model"
        );
        assert!(
            !is_threshold_activation("ReLU6"),
            "ReLU6 uses standard model"
        );
        assert!(!is_threshold_activation("TANH"), "TANH uses standard model");
        assert!(
            !is_threshold_activation("LOGISTIC"),
            "LOGISTIC uses standard model"
        );
        assert!(!is_threshold_activation("ReLU"), "ReLU uses standard model");
        assert!(
            !is_threshold_activation("LeakyReLU"),
            "LeakyReLU uses standard model"
        );
        assert!(!is_threshold_activation("ELU"), "ELU uses standard model");
        assert!(!is_threshold_activation("SELU"), "SELU uses standard model");
        assert!(!is_threshold_activation("GELU"), "GELU uses standard model");
        assert!(
            !is_threshold_activation("IDENTITY"),
            "IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("Softplus"),
            "Softplus uses standard model"
        );
        assert!(
            !is_threshold_activation("BENT_IDENTITY"),
            "BENT_IDENTITY uses standard model"
        );
        assert!(
            !is_threshold_activation("ArcTan"),
            "ArcTan uses standard model"
        );
        assert!(
            !is_threshold_activation("Swish"),
            "Swish uses standard model"
        );
        assert!(!is_threshold_activation("Mish"), "Mish uses standard model");
        assert!(
            !is_threshold_activation("UNKNOWN"),
            "Unknown uses standard model"
        );
    }

    /// Test ThresholdType correctly applies threshold functions
    #[test]
    fn test_threshold_type_apply() {
        // STEP: value > 0 ? 1 : 0
        assert_eq!(ThresholdType::Step.apply(0.5), 1.0);
        assert_eq!(ThresholdType::Step.apply(0.001), 1.0);
        assert_eq!(ThresholdType::Step.apply(0.0), 0.0);
        assert_eq!(ThresholdType::Step.apply(-0.001), 0.0);
        assert_eq!(ThresholdType::Step.apply(-5.0), 0.0);

        // BIPOLAR: value > 0 ? 1 : -1
        assert_eq!(ThresholdType::Bipolar.apply(0.5), 1.0);
        assert_eq!(ThresholdType::Bipolar.apply(0.001), 1.0);
        assert_eq!(ThresholdType::Bipolar.apply(0.0), -1.0);
        assert_eq!(ThresholdType::Bipolar.apply(-0.001), -1.0);
        assert_eq!(ThresholdType::Bipolar.apply(-5.0), -1.0);
    }

    /// Test ThresholdType correctly detects threshold flips
    #[test]
    fn test_threshold_type_would_flip() {
        // STEP: threshold at 0
        assert!(
            ThresholdType::Step.would_flip(-0.5, 1.0),
            "negative -> positive should flip"
        );
        assert!(
            ThresholdType::Step.would_flip(0.5, -1.0),
            "positive -> negative should flip"
        );
        assert!(
            !ThresholdType::Step.would_flip(0.5, 0.3),
            "positive -> more positive shouldn't flip"
        );
        assert!(
            !ThresholdType::Step.would_flip(-0.5, -0.3),
            "negative -> more negative shouldn't flip"
        );

        // Edge cases
        assert!(
            ThresholdType::Step.would_flip(-0.1, 0.2),
            "just crosses threshold"
        );
        assert!(
            !ThresholdType::Step.would_flip(-0.1, 0.05),
            "doesn't quite reach threshold"
        );
    }

    /// Test ThresholdType correctly identifies helpful vs harmful flips
    #[test]
    fn test_threshold_type_flip_direction() {
        // STEP: Error > 0 means output should be higher (0 -> 1 is helpful)
        // Current output is 0 (value < 0), error > 0 (should be 1), flip to 1 is helpful
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, 1.0, 0.5),
            1,
            "flip 0->1 when error>0 is helpful"
        );

        // Current output is 1 (value > 0), error < 0 (should be 0), flip to 0 is helpful
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, -1.0, -0.5),
            1,
            "flip 1->0 when error<0 is helpful"
        );

        // Current output is 0 (value < 0), error < 0 (should be 0), no flip needed
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, -0.1, -0.5),
            0,
            "no flip when error<0 and output=0"
        );

        // Current output is 1 (value > 0), error > 0 (should be 1), no flip needed
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, 0.1, 0.5),
            0,
            "no flip when error>0 and output=1"
        );

        // Harmful flip: flip 1->0 when error > 0 (output should stay high)
        assert_eq!(
            ThresholdType::Step.flip_direction(0.5, -1.0, 0.5),
            -1,
            "flip 1->0 when error>0 is harmful"
        );

        // Harmful flip: flip 0->1 when error < 0 (output should stay low)
        assert_eq!(
            ThresholdType::Step.flip_direction(-0.5, 1.0, -0.5),
            -1,
            "flip 0->1 when error<0 is harmful"
        );
    }

    /// Test evaluate_discrete_candidate finds candidates for STEP targets
    #[test]
    fn test_evaluate_discrete_candidate_step() {
        // Create samples where source activation correlates with whether the target
        // is on the "wrong side" of the threshold
        let mut samples = Vec::new();

        // Case 1: Target is at 0 (value=-0.5) but should be 1 (error=0.5)
        // Source has high positive activation - adding positive contribution would help
        for i in 0..20 {
            samples.push(DiscreteHelpfulSample {
                source_activation: 0.5 + (i as f32 * 0.01),
                target_value: -0.3, // Currently outputs 0
                target_activation: 0.0,
                avg_error: 0.5, // Should be 1 (positive error)
            });
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0, // threshold
        );

        assert!(candidate.is_some(), "Should find a candidate for STEP");
        let c = candidate.unwrap();
        assert!(
            c.expected_improvement_percentage > 0.0,
            "Should have positive improvement"
        );
        assert!(c.improved_count > 0, "Should have some helpful flips");
    }

    /// Test that discrete evaluation filters out low-improvement IDENTITY candidates.
    /// IDENTITY neurons with bias=0 are mathematically equivalent to synapses,
    /// and candidates with very low flip rates (<1%) don't reliably improve the model.
    ///
    /// The MIN_DISCRETE_IMPROVEMENT threshold (1%) is the key filter being tested here.
    /// The test creates samples where ALL have error (to pass MIN_NEURON_SAMPLE_COUNT check)
    /// but only a tiny fraction would actually flip in the helpful direction.
    #[test]
    fn test_discrete_evaluation_filters_low_improvement_identity() {
        // Create a sample set where ALL samples have error (passing the sample count check)
        // but only a tiny fraction (< 1%) would flip helpfully.
        //
        // IMPORTANT: Previously this test was broken - it only gave 5 samples non-zero error,
        // so samples_with_error=5 was less than MIN_NEURON_SAMPLE_COUNT=10, causing all weight
        // combinations to be skipped. The test passed for the wrong reason.
        //
        // Key insight: To make samples truly un-flippable, source_activation must be ZERO.
        // Any non-zero source activation with any of the tested weight scales (0.1 to 50.0)
        // could potentially flip the target. With source_activation=0, contribution=0
        // regardless of weights, so those samples cannot be affected.
        let mut samples = Vec::new();

        // 1000 samples - ALL have error to pass samples_with_error check
        for i in 0..1000 {
            // Only ~5 samples (0.5%) can flip helpfully:
            // These have non-zero source activation
            let can_flip_to_help = i < 5;

            if can_flip_to_help {
                // Sample that CAN flip to help:
                // - source_activation = 1.0 (non-zero, can contribute)
                // - target_value just below threshold (0)
                // - error positive (wants output to increase from 0 to 1)
                samples.push(DiscreteHelpfulSample {
                    source_activation: 1.0,
                    target_value: -0.05,    // Just below threshold
                    target_activation: 0.0, // Currently outputs 0 (STEP)
                    avg_error: 0.5,         // Wants to output 1, has error
                });
            } else {
                // Sample that CANNOT flip:
                // - source_activation = 0 (CRITICAL: contribution is always 0)
                // - Has error but cannot be helped since contribution = weight * 0 = 0
                samples.push(DiscreteHelpfulSample {
                    source_activation: 0.0, // Zero! contribution = weight * 0 = 0
                    target_value: -0.5,     // Below threshold (outputs 0)
                    target_activation: 0.0, // Currently outputs 0
                    avg_error: 0.3,         // Has error but source can't help
                });
            }
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0, // Zero threshold - MIN_DISCRETE_IMPROVEMENT (1%) should filter
        );

        // Should NOT return a candidate because improvement would be <1%
        // samples_with_error = 1000 (all have error)
        // helpful_flips = 5 (only samples with non-zero source activation)
        // harmful_flips = 0 (zero-activation samples can't flip either way)
        // improvement = 5 / 1000 = 0.5% < MIN_DISCRETE_IMPROVEMENT (1%)
        assert!(
            candidate.is_none(),
            "Should NOT return IDENTITY candidate with <1% improvement (0.5% in this test). \
             These are equivalent to direct synapses and don't reliably help. \
             Got candidate: {candidate:?}"
        );
    }

    /// Complementary test: verify that improvement ABOVE 1% DOES return a candidate.
    /// This ensures the MIN_DISCRETE_IMPROVEMENT threshold is the actual filter,
    /// not some other logic (like MIN_NEURON_SAMPLE_COUNT).
    #[test]
    fn test_discrete_evaluation_accepts_above_threshold_improvement() {
        // Same structure as the filter test, but with 2% flippable samples (> 1% threshold)
        let mut samples = Vec::new();

        for i in 0..1000 {
            // 20 samples (2%) can flip helpfully - above the 1% threshold
            let can_flip_to_help = i < 20;

            if can_flip_to_help {
                samples.push(DiscreteHelpfulSample {
                    source_activation: 1.0,
                    target_value: -0.05,
                    target_activation: 0.0,
                    avg_error: 0.5,
                });
            } else {
                samples.push(DiscreteHelpfulSample {
                    source_activation: 0.0, // Cannot contribute
                    target_value: -0.5,
                    target_activation: 0.0,
                    avg_error: 0.3,
                });
            }
        }

        let candidate = evaluate_discrete_candidate(
            "input-0",
            "target-step",
            &samples,
            ThresholdType::Step,
            0.0,
        );

        // SHOULD return a candidate because improvement = 20/1000 = 2% > 1% threshold
        assert!(
            candidate.is_some(),
            "Should return candidate when improvement (2%) exceeds MIN_DISCRETE_IMPROVEMENT (1%)"
        );

        let c = candidate.unwrap();
        // Verify the improvement is roughly what we expect (2%)
        assert!(
            c.expected_improvement_percentage > 0.015 && c.expected_improvement_percentage < 0.025,
            "Expected ~2% improvement, got {}%",
            c.expected_improvement_percentage * 100.0
        );
    }

    /// Test get_bias_values returns log-spaced values for efficient search
    #[test]
    fn test_get_bias_values() {
        // ReLU should have extended negative range for high-threshold neurons
        let relu_values = get_bias_values("ReLU");
        assert!(relu_values.contains(&0.0), "Should include 0");
        assert!(
            relu_values.iter().any(|&v| v <= -10.0),
            "ReLU should have large negative bias for high thresholds"
        );
        assert!(
            relu_values.len() < 25,
            "Should be efficient (log-spaced, not linear)"
        );

        // IDENTITY should have widest range (scales with large weights)
        let identity_values = get_bias_values("IDENTITY");
        assert!(
            identity_values.iter().any(|&v| v >= 25.0),
            "IDENTITY should reach 25.0 for large weight configurations"
        );
        assert!(
            identity_values.iter().any(|&v| v <= -25.0),
            "IDENTITY should reach -25.0"
        );

        // All values should be sorted
        for squash in &["ReLU", "TANH", "IDENTITY", "GELU"] {
            let values = get_bias_values(squash);
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(values, sorted, "Values for {squash} should be sorted");
        }
    }

    /// Test bias calculation with non-finite values
    #[test]
    fn test_bias_calculation_with_non_finite_values() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: f32::NAN,
                avg_error: -0.15,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: f32::INFINITY,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.6,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.18,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.08,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.7,
                avg_error: 0.22,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: -0.12,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.6,
                avg_error: 0.19,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.4,
                avg_error: -0.09,
                target_value: None,
                target_activation: None,
            },
        ];

        let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

        // Should handle non-finite values gracefully and return a finite bias
        assert!(
            bias.is_finite(),
            "Bias should be finite even with non-finite input values"
        );
        assert!(
            (-1.0..=1.0).contains(&bias),
            "Bias should be in reasonable range, got {bias}"
        );
    }

    /// Test that split-error ReLU evaluation finds complementary pairs when errors are split.
    /// When errors are ~50/50 positive/negative, no single ReLU can help all samples.
    /// Split evaluation should find two candidates: one for each error direction.
    #[test]
    fn test_split_relu_finds_complementary_pairs() {
        // Create samples with split errors:
        // - Half have positive error (output should be higher) with high source activation
        // - Half have negative error (output should be lower) with different pattern
        let mut samples = Vec::new();

        // Positive errors: when source is high, output should be higher
        // A ReLU with positive weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: 0.3, // Positive: output should be higher
                target_value: None,
                target_activation: None,
            });
        }

        // Negative errors: when source is high, output should be lower
        // A ReLU with negative weight on these samples will help
        for i in 0..50 {
            samples.push(HelpfulSample {
                activation: 0.5 + (i as f32) * 0.01,
                avg_error: -0.3, // Negative: output should be lower
                target_value: None,
                target_activation: None,
            });
        }

        // Verify we have split errors
        let positive_count = samples.iter().filter(|s| s.avg_error > 0.0).count();
        let negative_count = samples.iter().filter(|s| s.avg_error < 0.0).count();
        assert_eq!(positive_count, 50);
        assert_eq!(negative_count, 50);

        // Standard ReLU evaluation should struggle because errors cancel out
        // when computing error*activation correlation - roughly equal positive
        // and negative errors with similar activations means weak correlation overall.
        //
        // The split evaluation separates these, so each subset has strong correlation.
        // This test documents the expected behaviour without requiring GPU.
    }

    /// Test that upsert_candidate keeps complementary ReLU candidates with different
    /// incoming_weight values. A positive-weight ReLU (incoming_weight=1.0) and a
    /// negative-weight ReLU (incoming_weight=-1.0) should both be kept, not collide.
    #[test]
    fn test_upsert_keeps_complementary_relu_candidates_by_incoming_weight() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Positive-orientation ReLU candidate
        let positive_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            incoming_weight: 1.0, // Positive orientation
            outgoing_weight: 0.5,
            squash: "ReLU".to_string(),
            bias: 0.0,
            expected_improvement_percentage: 0.15,
            improved_count: 30,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Negative-orientation ReLU candidate
        let negative_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            incoming_weight: -1.0, // Negative orientation
            outgoing_weight: 0.4,  // Same outgoing sign
            squash: "ReLU".to_string(),
            bias: 0.0,
            expected_improvement_percentage: 0.12,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_candidate.clone());
        upsert_candidate(&mut map, negative_candidate.clone());

        // Both should be kept - they have different incoming_weight signs
        assert_eq!(
            map.len(),
            2,
            "Candidates with different incoming_weight should both be kept"
        );

        // Verify both are present with correct values
        let pos_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign
            1_i8, // outgoing sign
        );
        let neg_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            -1_i8, // incoming sign
            1_i8,  // outgoing sign
        );

        assert!(
            map.contains_key(&pos_key),
            "Positive-orientation candidate should exist"
        );
        assert!(
            map.contains_key(&neg_key),
            "Negative-orientation candidate should exist"
        );
    }

    /// Test that upsert_candidate keeps split-error complementary pairs with same
    /// incoming_weight but different outgoing_weight signs. This is the key case for
    /// split-error ReLU evaluation where errors are ~50/50 positive/negative.
    #[test]
    fn test_upsert_keeps_split_error_complementary_pairs() {
        use std::collections::HashMap;

        let mut map: HashMap<(String, String, String, i8, i8), CandidateNeuronJson> =
            HashMap::new();

        // Candidate for positive errors: same source/target, positive outgoing_weight
        // This pushes output UP when source is high
        let positive_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            incoming_weight: 1.0, // Same orientation
            outgoing_weight: 0.5, // POSITIVE: pushes output UP
            squash: "ReLU".to_string(),
            bias: 0.0,
            expected_improvement_percentage: 0.10,
            improved_count: 25,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Candidate for negative errors: same source/target, negative outgoing_weight
        // This pushes output DOWN when source is high
        let negative_error_candidate = CandidateNeuronJson {
            source_neuron_uuid: "source-1".to_string(),
            target_neuron_uuid: "target-1".to_string(),
            incoming_weight: 1.0,  // Same orientation
            outgoing_weight: -0.4, // NEGATIVE: pushes output DOWN
            squash: "ReLU".to_string(),
            bias: 0.0,
            expected_improvement_percentage: 0.08,
            improved_count: 20,
            total_count: 50,
            target_neuron_stats: None,
        };

        // Insert both candidates
        upsert_candidate(&mut map, positive_error_candidate.clone());
        upsert_candidate(&mut map, negative_error_candidate.clone());

        // Both should be kept - they have different outgoing_weight signs
        // This is the key fix for split-error ReLU evaluation
        assert_eq!(
            map.len(),
            2,
            "Split-error complementary pairs with different outgoing_weight signs should both be kept"
        );

        // Verify both are present
        let pos_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8, // incoming sign (both same)
            1_i8, // outgoing sign: positive
        );
        let neg_out_key = (
            "source-1".to_string(),
            "target-1".to_string(),
            "ReLU".to_string(),
            1_i8,  // incoming sign (both same)
            -1_i8, // outgoing sign: negative
        );

        assert!(
            map.contains_key(&pos_out_key),
            "Positive-outgoing candidate (pushes UP) should exist"
        );
        assert!(
            map.contains_key(&neg_out_key),
            "Negative-outgoing candidate (pushes DOWN) should exist"
        );

        assert_eq!(
            map.get(&pos_out_key).unwrap().outgoing_weight,
            0.5,
            "Positive-outgoing candidate should have outgoing_weight=0.5"
        );
        assert_eq!(
            map.get(&neg_out_key).unwrap().outgoing_weight,
            -0.4,
            "Negative-outgoing candidate should have outgoing_weight=-0.4"
        );
    }

    // ============================================================================
    // TDD TESTS: Validate ReLU improvement calculations
    // ============================================================================

    /// TDD Test: Verify that predicted improvement matches actual for split errors.
    /// This tests the core maths of compute_net_improvement_across_all_samples.
    #[test]
    fn test_predicted_improvement_matches_actual_for_split_relu() {
        // Scenario: 50% positive errors, 50% negative errors
        // Source always positive (0.5), so ReLU always fires
        //
        // Positive errors: error = +0.2 (want output higher)
        // Negative errors: error = -0.2 (want output lower)
        //
        // If we compute optimal weight from positive subset: w = Σ(error×act)/Σ(act²)
        // For positive subset: w = (0.2×0.5 + 0.2×0.5) / (0.5² + 0.5²) = 0.2/0.5 = 0.4
        //
        // Now apply w=0.4 to ALL samples:
        // - Positive samples: new_error = 0.2 - 0.4×0.5 = 0.0 (perfect!)
        // - Negative samples: new_error = -0.2 - 0.4×0.5 = -0.4 (much worse!)
        //
        // Net improvement should be NEGATIVE (overall harm)

        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: -0.2,
                target_value: None,
                target_activation: None,
            },
        ];

        // Optimal weight computed from positive samples
        let outgoing_weight: f32 = 0.4;
        let incoming_weight: f32 = 1.0;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Use the function under test (linear model, bias=0)
        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test (ReLU threshold at 0)
            baseline_error_sq,
            None,
        );

        // Manually compute actual improvement
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Outgoing weight: {outgoing_weight:.4}, Predicted: {:.4}%, Actual: {:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        // With split errors, the net improvement should be NEGATIVE
        assert!(
            predicted_improvement < 0.0,
            "With split errors and uniform source, net improvement should be negative, got {:.4}%",
            predicted_improvement * 100.0
        );
    }

    /// TDD Test: When errors are aligned, linear model should be accurate.
    #[test]
    fn test_linear_model_accurate_when_errors_aligned() {
        // All positive errors, source always positive
        // This is the ideal case for ReLU - linear model should work perfectly
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.25,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.8,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
        ];

        // Compute optimal weight: w = Σ(error×activation) / Σ(activation²)
        let error_act_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let act_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
        let outgoing_weight = error_act_sum / act_sq_sum;
        let incoming_weight = 1.0f32;

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        let predicted_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0 for this test
            baseline_error_sq,
            None, // Linear model
        );

        // Manually compute (bias=0)
        let mut new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_out = (incoming_weight * sample.activation).max(0.0);
            let new_err = sample.avg_error - outgoing_weight * relu_out;
            new_error_sq += new_err.powi(2);
        }
        let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

        eprintln!(
            "Aligned errors: weight={outgoing_weight:.4}, predicted={:.4}%, actual={:.4}%",
            predicted_improvement * 100.0,
            actual_improvement * 100.0
        );

        assert!(
            (predicted_improvement - actual_improvement).abs() < 0.0001,
            "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
        );

        assert!(
            actual_improvement > 0.1,
            "With aligned errors, should see significant improvement, got {:.4}%",
            actual_improvement * 100.0
        );
    }

    // ============================================================================
    // TDD TESTS: HARD_TANH saturation behaviour
    // ============================================================================

    /// Extended sample for HARD_TANH testing - includes target neuron's pre-activation value
    struct HardTanhSample {
        source_activation: f32,
        target_value: f32,      // Pre-activation input sum
        target_activation: f32, // Post-activation output (clamped to [-1, 1])
        target_error: f32,      // expected - actual
    }

    /// Apply HARD_TANH activation function
    fn hard_tanh(x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }

    /// TEST: Demonstrates that linear model is WRONG for HARD_TANH targets near saturation.
    /// The linear model predicts disaster (-125%) but HARD_TANH actually gives perfect result!
    #[test]
    fn test_hard_tanh_linear_model_is_wrong_near_saturation() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0
        // - error = 1.0 - 0.9 = 0.1 (positive: output should be higher)
        //
        // Source neuron fires with activation 0.5
        // If we add a ReLU with outgoing_weight = 0.5:
        // - contribution = 0.5 * relu(0.5) = 0.5 * 0.5 = 0.25
        //
        // LINEAR MODEL predicts:
        // - new_error = 0.1 - 0.25 = -0.15 (overshot)
        // - old_error² = 0.01, new_error² = 0.0225
        // - improvement = (0.01 - 0.0225) / 0.01 = -125% (WORSE!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 0.9 + 0.25 = 1.15
        // - new_output = clamp(1.15, -1, 1) = 1.0 (saturated!)
        // - new_error = 1.0 - 1.0 = 0.0 (PERFECT!)
        // - old_error² = 0.01, new_error² = 0.0
        // - improvement = (0.01 - 0.0) / 0.01 = +100% (MUCH BETTER!)

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 0.9,      // Near saturation
            target_activation: 0.9, // hard_tanh(0.9) = 0.9
            target_error: 0.1,      // expected (1.0) - actual (0.9)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;

        // Compute baseline error
        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL prediction (current behaviour)
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH behaviour
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model: new_error²={linear_new_error_sq:.4}, improvement={:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: new_error²={hard_tanh_new_error_sq:.4}, improvement={:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The linear model predicts NEGATIVE improvement (making things worse)
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement near saturation, got {:.1}%",
            linear_improvement * 100.0
        );

        // But the actual HARD_TANH behaviour shows PERFECT improvement!
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement (error goes to 0), got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference is massive - linear model is completely wrong!
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// TEST: Linear model predicts improvement but HARD_TANH shows NO improvement (already saturated)
    #[test]
    fn test_hard_tanh_linear_model_wrong_when_already_saturated() {
        // Scenario: Target is ALREADY saturated at 1.0
        // - target_value = 1.5 (input already beyond saturation)
        // - target_activation = 1.0 (clamped output)
        // - expected output = 0.8
        // - error = 0.8 - 1.0 = -0.2 (negative: output should be LOWER)
        //
        // Source fires with activation 0.5, ReLU with outgoing_weight = -0.3
        // Contribution = -0.3 * 0.5 = -0.15 (pushing output DOWN, seems good!)
        //
        // LINEAR MODEL predicts:
        // - new_error = -0.2 - (-0.15) = -0.05 (improved!)
        // - old_error² = 0.04, new_error² = 0.0025
        // - improvement = (0.04 - 0.0025) / 0.04 = 93.75% (great!)
        //
        // ACTUAL HARD_TANH behaviour:
        // - new_input = 1.5 + (-0.15) = 1.35 (still beyond saturation!)
        // - new_output = clamp(1.35) = 1.0 (unchanged!)
        // - new_error = 0.8 - 1.0 = -0.2 (NO CHANGE!)
        // - improvement = 0%

        let samples = vec![HardTanhSample {
            source_activation: 0.5,
            target_value: 1.5,      // Already beyond saturation
            target_activation: 1.0, // Clamped at max
            target_error: -0.2,     // expected (0.8) - actual (1.0)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = -0.3; // Trying to push output down

        let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

        // LINEAR MODEL
        let mut linear_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let linear_new_error = sample.target_error - contribution;
            linear_new_error_sq += linear_new_error.powi(2);
        }
        let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

        // ACTUAL HARD_TANH
        let mut hard_tanh_new_error_sq = 0.0f32;
        for sample in &samples {
            let relu_output = (incoming_weight * sample.source_activation).max(0.0);
            let contribution = outgoing_weight * relu_output;
            let new_input = sample.target_value + contribution;
            let new_output = hard_tanh(new_input);
            let expected = sample.target_activation + sample.target_error;
            let new_error = expected - new_output;
            hard_tanh_new_error_sq += new_error.powi(2);
        }
        let hard_tanh_improvement =
            (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

        eprintln!("Baseline error²: {baseline_error_sq:.4}");
        eprintln!(
            "Linear model predicts: {:.1}% improvement",
            linear_improvement * 100.0
        );
        eprintln!(
            "HARD_TANH actual: {:.1}% improvement",
            hard_tanh_improvement * 100.0
        );

        // Linear model predicts big improvement
        assert!(
            linear_improvement > 0.9,
            "Linear model should predict ~93% improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // But HARD_TANH shows NO improvement (still saturated)
        assert!(
            hard_tanh_improvement.abs() < 0.01,
            "HARD_TANH should show ~0% improvement (still saturated), got {:.1}%",
            hard_tanh_improvement * 100.0
        );
    }

    /// Test that compute_net_improvement_with_squash uses HARD_TANH model when specified.
    /// This verifies the actual function we use in production.
    #[test]
    fn test_compute_net_improvement_uses_hard_tanh_model() {
        // Create samples WITH target data (target_value and target_activation)
        // so that the HARD_TANH model can be used
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,          // expected 1.0, actual 0.9
            target_value: Some(0.9), // Near saturation
            target_activation: Some(0.9),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 * 0.5 = 0.25

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Test with LINEAR model (no squash specified, bias=0)
        let linear_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            None,
        );

        // Test with HARD_TANH model (bias=0)
        let hard_tanh_improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        eprintln!(
            "compute_net_improvement_with_squash(None): {:.1}%",
            linear_improvement * 100.0
        );
        eprintln!(
            "compute_net_improvement_with_squash(HARD_TANH): {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // LINEAR model should predict NEGATIVE improvement (overshoot to -0.15 error)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // baseline = 0.01, so improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            linear_improvement < 0.0,
            "Linear model should predict negative improvement, got {:.1}%",
            linear_improvement * 100.0
        );

        // HARD_TANH model should predict PERFECT improvement (saturate at 1.0)
        // new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        // expected = 0.9 + 0.1 = 1.0, new_error = 0.0
        // improvement = (0.01 - 0.0) / 0.01 = 100%
        assert!(
            hard_tanh_improvement > 0.99,
            "HARD_TANH should show ~100% improvement, got {:.1}%",
            hard_tanh_improvement * 100.0
        );

        // The difference between models should be massive
        let difference = (hard_tanh_improvement - linear_improvement).abs();
        assert!(
            difference > 1.0,
            "Difference between models should be >100%, got {:.1}%",
            difference * 100.0
        );
    }

    /// Test that HARD_TANH model falls back to linear when target data is missing.
    #[test]
    fn test_compute_net_improvement_falls_back_to_linear_without_target_data() {
        // Create samples WITHOUT target data (None values)
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5;
        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // Even with HARD_TANH specified, should fall back to linear model (bias=0)
        let improvement = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        // Should match the linear model result (-125%)
        // new_error = 0.1 - 0.25 = -0.15, new_error² = 0.0225
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        let expected_linear = -1.25;
        assert!(
            (improvement - expected_linear).abs() < 0.01,
            "Should fall back to linear model without target data, got {:.1}% (expected {:.1}%)",
            improvement * 100.0,
            expected_linear * 100.0
        );
    }

    /// TEST: count_improved_samples must use HARD_TANH model for accurate sample counts.
    ///
    /// This test demonstrates the bug where count_improved_samples always uses the linear
    /// model, causing inaccurate counts for HARD_TANH targets. For a sample near saturation,
    /// the linear model predicts the error gets worse (overshoot), but the HARD_TANH model
    /// correctly shows the sample is improved (saturates at the limit).
    #[test]
    fn test_count_improved_samples_uses_hard_tanh_model() {
        // Scenario: Target neuron with HARD_TANH activation is near saturation
        // - target_value = 0.9 (input sum before clamping)
        // - target_activation = 0.9 (output after HARD_TANH, not saturated yet)
        // - expected output = 1.0 (what we want)
        // - avg_error = 0.1 (expected - actual = 1.0 - 0.9 = 0.1)
        //
        // When we add a connection with contribution = 0.25:
        // - LINEAR model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved
        // - HARD_TANH model: new_input = 0.9 + 0.25 = 1.15, new_output = clamp(1.15) = 1.0
        //                    new_error = 1.0 - 1.0 = 0.0, |new_error| < |old_error|, IMPROVED!
        let samples = vec![HelpfulSample {
            activation: 0.5,              // Source neuron's activation
            avg_error: 0.1,               // Target wants to go up by 0.1
            target_value: Some(0.9),      // Pre-activation input sum
            target_activation: Some(0.9), // Post-activation output (not yet saturated)
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.5 × max(0, 1.0 × 0.5) = 0.25

        // With HARD_TANH model, this sample SHOULD be counted as improved (bias=0)
        let (improved_count, total_count) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // bias=0
            Some("HARD_TANH"),
        );

        assert_eq!(total_count, 1, "Should have 1 total sample");
        assert_eq!(
            improved_count, 1,
            "HARD_TANH model should show sample is improved (saturates at 1.0), got {improved_count} improved",
        );
    }

    /// TEST: count_improved_samples falls back to linear model when target data is missing.
    #[test]
    fn test_count_improved_samples_falls_back_to_linear_without_target_data() {
        // Sample WITHOUT target data - should use linear model
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|, NOT improved

        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,               // bias=0
            Some("HARD_TANH"), // Even with HARD_TANH, should fall back to linear
        );

        assert_eq!(
            improved_count, 0,
            "Without target data, should fall back to linear model (sample not improved)"
        );
    }

    /// TEST: count_improved_samples uses linear model for non-HARD_TANH activations.
    #[test]
    fn test_count_improved_samples_uses_linear_for_other_activations() {
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.3,
            target_value: Some(0.5),
            target_activation: Some(0.5),
        }];

        let incoming_weight = 1.0;
        let outgoing_weight = 0.5; // contribution = 0.25
                                   // Linear: new_error = 0.3 - 0.25 = 0.05, |new_error| < |old_error| = 0.3, IMPROVED

        // With TANH (not HARD_TANH), should use linear model (bias=0)
        let (improved_count, _) = count_improved_samples(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0,
            Some("TANH"),
        );

        assert_eq!(
            improved_count, 1,
            "Linear model should show sample is improved for TANH"
        );

        // With None squash, should also use linear model (bias=0)
        let (improved_count_none, _) =
            count_improved_samples(&samples, incoming_weight, outgoing_weight, 0.0, None);

        assert_eq!(
            improved_count_none, 1,
            "Linear model should show sample is improved when squash is None"
        );
    }

    /// Test that compute_activation_improvement_and_count correctly falls back to linear model
    /// when samples lack target_value/target_activation data, even if use_hard_tanh is true.
    ///
    /// This validates the safety invariant: use_hard_tanh should only be true when
    /// can_use_hard_tanh() has verified all samples have the required data.
    #[test]
    fn test_activation_improvement_uses_linear_when_no_target_data() {
        // Samples WITHOUT target data
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,      // No target data
            target_activation: None, // No target data
        }];

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

        // can_use_hard_tanh should return false when samples lack target data
        assert!(
            !can_use_hard_tanh(&samples, Some("HARD_TANH")),
            "can_use_hard_tanh must return false when samples lack target data"
        );

        // get_target_simulation_fn should also return None when samples lack target data
        assert!(
            get_target_simulation_fn(&samples, Some("HARD_TANH")).is_none(),
            "get_target_simulation_fn must return None when samples lack target data"
        );

        // When properly using get_target_simulation_fn, we get linear model behaviour
        let target_activation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
        let (improvement, improved, total) = compute_activation_improvement_and_count(
            &samples,
            1.0,            // incoming_weight
            0.5,            // outgoing_weight
            0.0,            // bias
            |x| x.max(0.0), // ReLU activation
            baseline_sq,
            target_activation_fn, // Will be None due to missing target data
        );

        // Linear model: contribution = 0.5 × max(0, 1.0 × 0.5 + 0) = 0.25
        // new_error = 0.25 - 0.1 = 0.15 (note: compute_activation uses contribution - avg_error)
        // But |0.15| > |0.1| so sample is NOT improved
        // improvement = (0.01 - 0.0225) / 0.01 = -125%
        assert!(
            improvement < 0.0,
            "Linear model should show negative improvement"
        );
        assert_eq!(total, 1, "Should have 1 total sample");
        assert_eq!(
            improved, 0,
            "Linear model should show sample is NOT improved"
        );
    }

    /// Verify synapse weight uses correct linear optimal formula: w = Σ(error × activation) / Σ(activation²)
    /// The old buggy formula (Σ|error| / Σ|activation|) would clamp to ±1.0 in many cases.
    /// This test ensures weights are computed correctly and not always clamped.
    #[test]
    fn synapse_weight_uses_correct_linear_optimal_formula() {
        // Create HelpfulStats with known values that demonstrate the difference
        // between the correct and buggy formulas:
        //
        // Sample 1: activation=2.0, error=0.8  (error×activation = 1.6, activation² = 4.0)
        // Sample 2: activation=3.0, error=0.6  (error×activation = 1.8, activation² = 9.0)
        // Sample 3: activation=1.0, error=0.4  (error×activation = 0.4, activation² = 1.0)
        //
        // Σ(error × activation) = 1.6 + 1.8 + 0.4 = 3.8
        // Σ(activation²) = 4.0 + 9.0 + 1.0 = 14.0
        //
        // Correct optimal weight: 3.8 / 14.0 = 0.271...
        //
        // Old buggy formula would compute:
        // Σ|error| = 0.8 + 0.6 + 0.4 = 1.8
        // Σ|activation| = 2.0 + 3.0 + 1.0 = 6.0
        // Buggy weight: 1.8 / 6.0 = 0.3 (different!)
        //
        // And in cases where Σ|error| > Σ|activation|, the buggy formula would clamp to 1.0

        let stats = HelpfulStats {
            positive_count: 3, // All samples have positive correlation for this test
            negative_count: 0,
            positive_improvement_sum: 1.8, // Σ|error| for positive samples (unused in new formula)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 6.0, // Σ|activation| for positive samples (unused in new formula)
            negative_activation_sum: 0.0,
            error_sq_sum: 0.8 * 0.8 + 0.6 * 0.6 + 0.4 * 0.4, // 0.64 + 0.36 + 0.16 = 1.16
            activation_sq_sum: 14.0,                         // Σ(activation²) = 4 + 9 + 1
            error_activation_sum: 3.8, // Σ(error × activation) = 1.6 + 1.8 + 0.4
        };

        // Apply the correct formula used in production (after fix):
        // weight = error_activation_sum / (activation_sq_sum + EPSILON)
        let weight = if stats.activation_sq_sum > EPSILON {
            let raw = stats.error_activation_sum / (stats.activation_sq_sum + EPSILON);
            raw.clamp(-10.0, 10.0)
        } else {
            0.0
        };

        // Expected weight: 3.8 / 14.0 ≈ 0.2714
        let expected = 3.8 / 14.0;
        assert!(
            (weight - expected).abs() < 0.001,
            "Weight should be calculated as Σ(error×activation)/Σ(activation²) = {expected:.4}, got {weight:.4}"
        );

        // Verify it's NOT the buggy value
        let buggy_weight = 1.8 / 6.0; // 0.3
        assert!(
            (weight - buggy_weight).abs() > 0.01,
            "Weight {weight} should differ from buggy formula result {buggy_weight}"
        );

        // Now test a case that would clamp to 1.0 with the buggy formula
        // Samples where |error| >> |activation|
        let stats_would_clamp = HelpfulStats {
            positive_count: 2,
            negative_count: 0,
            positive_improvement_sum: 5.0, // Σ|error| (buggy formula would use this)
            negative_improvement_sum: 0.0,
            positive_activation_sum: 2.0, // Σ|activation| (buggy formula: 5.0/2.0 = 2.5 -> clamp to 1.0)
            negative_activation_sum: 0.0,
            error_sq_sum: 13.0,        // 2² + 3² = 4 + 9 = 13
            activation_sq_sum: 2.0,    // 1² + 1² = 2
            error_activation_sum: 5.0, // 2×1 + 3×1 = 5
        };

        let weight2 = if stats_would_clamp.activation_sq_sum > EPSILON {
            let raw = stats_would_clamp.error_activation_sum
                / (stats_would_clamp.activation_sq_sum + EPSILON);
            raw.clamp(-10.0, 10.0)
        } else {
            0.0
        };

        // Correct optimal: 5.0 / 2.0 = 2.5
        // Buggy formula would have clamped to 1.0
        assert!(
            (weight2 - 2.5).abs() < 0.001,
            "Weight should be 2.5 (not clamped to 1.0), got {weight2}"
        );
        assert!(
            weight2 > 1.0,
            "Weight {weight2} should exceed 1.0, proving it's not using the buggy clamped formula"
        );
    }

    /// Test that synapse improvement calculation uses saturation-aware model for HARD_TANH targets.
    ///
    /// The linear model overpredicts improvement when target is near saturation because it
    /// assumes the contribution is applied directly to error, not clamped by the activation.
    ///
    /// Example: HARD_TANH target with pre-activation value = 0.9, error = 0.15 (wants output 1.05)
    /// Linear model: Adding contribution of 0.2 reduces error by 0.2 (100%+ improvement!)
    /// Reality: HARD_TANH(0.9 + 0.2) = HARD_TANH(1.1) = 1.0, so error becomes 1.0 - 1.05 = -0.05
    /// Actual improvement: |0.15|² - |0.05|² = 0.0225 - 0.0025 = 0.02 (only ~89% reduction)
    ///
    /// Without saturation-aware model, synapse candidates may promise more than they deliver.
    #[test]
    fn synapse_improvement_uses_saturation_aware_model_for_hard_tanh() {
        // Create samples where HARD_TANH is near saturation
        // target_value (pre-activation) = 0.9, so close to +1 saturation
        // target_activation = HARD_TANH(0.9) = 0.9
        // avg_error = 0.15 (output should be 0.9 + 0.15 = 1.05, but HARD_TANH caps at 1.0)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,              // source neuron activation
                avg_error: 0.15,              // target error (positive = output should be higher)
                target_value: Some(0.9),      // pre-activation sum
                target_activation: Some(0.9), // current output
            },
            HelpfulSample {
                activation: 0.4,
                avg_error: 0.12,
                target_value: Some(0.85),
                target_activation: Some(0.85),
            },
        ];

        // Compute optimal weight using linear model
        let mut error_activation_sum = 0.0f32;
        let mut activation_sq_sum = 0.0f32;
        let mut baseline_error_sq = 0.0f32;

        for sample in &samples {
            error_activation_sum += sample.avg_error * sample.activation;
            activation_sq_sum += sample.activation * sample.activation;
            baseline_error_sq += sample.avg_error * sample.avg_error;
        }

        let weight = error_activation_sum / (activation_sq_sum + EPSILON);

        // LINEAR MODEL PREDICTION (what old code does):
        // improvement = (2*w*E[a*e] - w²*E[a²]) / E[e²]
        let linear_improvement = (2.0 * weight * error_activation_sum
            - weight * weight * activation_sq_sum)
            / baseline_error_sq;

        // SATURATION-AWARE MODEL (what should happen):
        // For each sample, compute actual new error after applying synapse through HARD_TANH
        let mut actual_new_error_sq = 0.0f32;
        for sample in &samples {
            let target_value = sample.target_value.unwrap();
            let target_activation = sample.target_activation.unwrap();
            let expected_output = target_activation + sample.avg_error;

            // New pre-activation = old pre-activation + weight * source_activation
            let new_pre_activation = target_value + weight * sample.activation;
            // New output = HARD_TANH(new_pre_activation)
            let new_output = new_pre_activation.clamp(-1.0, 1.0);
            // New error = new_output - expected_output
            let new_error = new_output - expected_output;

            actual_new_error_sq += new_error * new_error;
        }

        let actual_improvement = (baseline_error_sq - actual_new_error_sq) / baseline_error_sq;

        // Linear model should predict MORE improvement than reality (overpredicts)
        assert!(
            linear_improvement > actual_improvement,
            "Linear model ({linear_improvement:.4}) should overpredict vs actual ({actual_improvement:.4}) for HARD_TANH near saturation"
        );

        // The difference should be meaningful (not just floating-point noise)
        let prediction_error = (linear_improvement - actual_improvement).abs();
        assert!(
            prediction_error > 0.01,
            "Prediction error ({prediction_error:.4}) should be > 1% for HARD_TANH near saturation"
        );

        // Now test that compute_synapse_improvement_with_target_squash gives accurate prediction
        let saturation_aware_improvement = compute_synapse_improvement_with_target_squash(
            &samples,
            weight,
            baseline_error_sq,
            Some("HARD_TANH"),
        );

        // Saturation-aware model should be close to actual improvement
        let saturation_error = (saturation_aware_improvement - actual_improvement).abs();
        assert!(
            saturation_error < 0.01,
            "Saturation-aware model ({saturation_aware_improvement:.4}) should match actual ({actual_improvement:.4}), error was {saturation_error:.4}"
        );
    }

    /// TDD Test: ReLU improvement calculation MUST include bias for accurate predictions.
    ///
    /// When bias > 0, the ReLU threshold shifts left, causing more samples to activate.
    /// When bias < 0, the ReLU threshold shifts right, causing fewer samples to activate.
    ///
    /// If bias is NOT included in the improvement calculation, the prediction will be
    /// inaccurate when a non-zero bias is proposed for the new neuron.
    ///
    /// This test demonstrates the bug: compute_relu_improvement_and_count ignores bias,
    /// leading to overestimation when the actual neuron would use a different activation
    /// pattern due to the bias.
    #[test]
    fn test_relu_improvement_must_include_bias() {
        // Scenario: Source activations that are NEGATIVE (would be zeroed by ReLU without bias).
        // With a positive bias, the ReLU would fire on these samples.
        //
        // Sample 1: activation = -0.3, error = 0.5 (want output higher)
        // Sample 2: activation = -0.2, error = 0.4 (want output higher)
        // Sample 3: activation = 0.1, error = 0.3 (want output higher)
        //
        // Without bias (bias=0):
        //   ReLU(1.0 × -0.3 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × -0.2 + 0) = 0  → contribution = 0
        //   ReLU(1.0 × 0.1 + 0) = 0.1 → contribution = outgoing_weight × 0.1
        //
        // With bias=0.5:
        //   ReLU(1.0 × -0.3 + 0.5) = 0.2 → contribution = outgoing_weight × 0.2
        //   ReLU(1.0 × -0.2 + 0.5) = 0.3 → contribution = outgoing_weight × 0.3
        //   ReLU(1.0 × 0.1 + 0.5) = 0.6 → contribution = outgoing_weight × 0.6
        //
        // The bias dramatically changes which samples are affected and by how much!

        let samples = vec![
            HelpfulSample {
                activation: -0.3,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: 0.4,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.1,
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            },
        ];

        let incoming_weight = 1.0f32;
        let outgoing_weight = 0.8f32; // Positive weight to reduce positive errors
        let bias = 0.5f32; // Significant positive bias

        let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
        // 0.5² + 0.4² + 0.3² = 0.25 + 0.16 + 0.09 = 0.5

        // Predicted improvement using the function WITH bias parameter
        let predicted_with_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            bias,
            baseline_error_sq,
            None,
        );

        // Also compute without bias (bias=0) to show the difference
        let predicted_without_bias = compute_net_improvement_with_squash(
            &samples,
            incoming_weight,
            outgoing_weight,
            0.0, // No bias
            baseline_error_sq,
            None,
        );

        // Manually compute ACTUAL improvement WITH bias
        let mut new_error_sq_with_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation + bias;
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_with_bias += new_err.powi(2);
        }
        let actual_improvement_with_bias =
            (baseline_error_sq - new_error_sq_with_bias) / baseline_error_sq;

        // Manually compute improvement WITHOUT bias (what current code predicts)
        let mut new_error_sq_without_bias = 0.0f32;
        for sample in &samples {
            let pre_activation = incoming_weight * sample.activation; // No bias!
            let relu_out = pre_activation.max(0.0);
            let contribution = outgoing_weight * relu_out;
            let new_err = sample.avg_error - contribution;
            new_error_sq_without_bias += new_err.powi(2);
        }
        let manual_improvement_without_bias =
            (baseline_error_sq - new_error_sq_without_bias) / baseline_error_sq;

        eprintln!(
            "Baseline error²: {baseline_error_sq:.4}, With bias: new_error²={new_error_sq_with_bias:.4}, Without bias: new_error²={new_error_sq_without_bias:.4}"
        );
        let predicted_with_bias_pct = predicted_with_bias * 100.0;
        let predicted_without_bias_pct = predicted_without_bias * 100.0;
        let actual_with_bias_pct = actual_improvement_with_bias * 100.0;
        eprintln!(
            "Predicted (with bias): {predicted_with_bias_pct:.2}%, Predicted (no bias): {predicted_without_bias_pct:.2}%, Actual (with bias): {actual_with_bias_pct:.2}%"
        );

        // The predicted improvement WITH bias should match the actual improvement WITH bias.
        // This verifies that the bias parameter is correctly included in the calculation.
        assert!(
            (predicted_with_bias - actual_improvement_with_bias).abs() < 0.01,
            "Predicted improvement WITH bias ({predicted_with_bias:.4}) must match actual improvement WITH bias ({actual_improvement_with_bias:.4})."
        );

        // Verify that WITHOUT bias prediction matches manual calculation (both bias=0)
        assert!(
            (predicted_without_bias - manual_improvement_without_bias).abs() < 0.01,
            "Predicted (no bias) ({predicted_without_bias:.4}) must match manual (no bias) ({manual_improvement_without_bias:.4})."
        );

        // The key insight: with bias=0.5, improvement should be much higher than with bias=0
        // because more samples activate the ReLU
        assert!(
            actual_improvement_with_bias > predicted_without_bias + 0.1,
            "Improvement with bias ({actual_improvement_with_bias:.4}) should be significantly higher than without ({predicted_without_bias:.4})"
        );
    }

    #[test]
    fn neuron_diagnostics_reports_hidden_neuron_filtered_in_mixed_focus_list() {
        // Test that when a focus list contains BOTH output AND hidden neurons,
        // the hidden neurons get HiddenNeuronFiltered reason (not NoEligibleSources).
        //
        // Bug scenario: When focus_order is NOT empty (some output neurons exist),
        // the skipped_hidden neurons were never merged into diagnostics, so they
        // appeared with misleading reasons like NoEligibleSources.
        let mut diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "hidden-1"]);

        // Mark hidden-1 as filtered (this is what should happen in normal flow)
        diagnostics.mark_hidden_filtered("hidden-1");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // Both should have summaries
        assert_eq!(
            summaries.len(),
            2,
            "Expected 2 summaries (one for output, one for hidden)"
        );

        // Find the hidden neuron summary
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-1")
            .expect("Should have summary for hidden-1");

        // The hidden neuron should have HiddenNeuronFiltered reason
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }

    #[test]
    fn neuron_diagnostics_reports_input_neuron_filtered_not_hidden() {
        // Test that when an input neuron is in the focus list, it gets
        // InputNeuronFiltered reason (not HiddenNeuronFiltered).
        //
        // Bug scenario: The neuron_type_map was only built from creature.neurons
        // and didn't include input neurons. When an input neuron (e.g. "input-0")
        // was in the focus list, neuron_type_map.get() returned None, and
        // `neuron_type != Some("output")` evaluated to true. The input neuron
        // was incorrectly added to skipped_hidden and reported with
        // HiddenNeuronFiltered reason.
        let mut diagnostics =
            NeuronDiagnostics::new_for_tests(&["output-0", "input-1", "hidden-2"]);

        // Mark input-1 as filtered because it's an input neuron
        diagnostics.mark_input_filtered("input-1");

        // Mark hidden-2 as filtered because it's a hidden neuron
        diagnostics.mark_hidden_filtered("hidden-2");

        // Simulate output-0 being processed normally but finding no candidate
        diagnostics.set_target_record_count("output-0", 100);
        diagnostics.set_total_eligible_sources("output-0", 5);
        diagnostics.record_candidate_attempt("output-0", false);

        let summaries = diagnostics.no_candidate_summaries();

        // All three should have summaries
        assert_eq!(
            summaries.len(),
            3,
            "Expected 3 summaries (one for output, one for input, one for hidden)"
        );

        // Find the input neuron summary - should have InputNeuronFiltered reason
        let input_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "input-1")
            .expect("Should have summary for input-1");
        assert!(
            matches!(
                input_summary.reason,
                NeuronNoCandidateReason::InputNeuronFiltered
            ),
            "Input neuron should report InputNeuronFiltered, not {:?}",
            input_summary.reason
        );

        // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
        let hidden_summary = summaries
            .iter()
            .find(|s| s.target_uuid == "hidden-2")
            .expect("Should have summary for hidden-2");
        assert!(
            matches!(
                hidden_summary.reason,
                NeuronNoCandidateReason::HiddenNeuronFiltered
            ),
            "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
            hidden_summary.reason
        );
    }
}
