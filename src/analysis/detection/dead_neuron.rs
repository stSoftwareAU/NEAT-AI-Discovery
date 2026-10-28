//! Dead neuron detection module (Issue #341).
//!
//! Identifies neurons that have become effectively dead (always outputting zero or
//! near-zero activation) and recommends their removal. Dead neurons waste computation
//! without contributing to the network's output.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Dead Neuron Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A neuron is "dead" if:
//! 1. **Near-zero activation**: Mean absolute activation < threshold (e.g., 1e-6) across
//!    all samples.
//! 2. **Zero variance**: Standard deviation of activation ≈ 0 (always outputs the same value).
//! 3. **Only hidden neurons**: Output and input neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When a dead neuron is detected, we recommend:
//! 1. **Remove the neuron**: Emit a `RemoveNeuron` operation via `CoordinatedStructuralCandidateJson`.
//!
//! Dead neurons consume GPU resources during both training and inference without contributing
//! useful information, so removal is the primary recommendation.
//!
//! ## Bounded reachability (Issue #2370)
//!
//! Each dead neuron's connected outputs are read from one shared, memoised
//! strongly-connected-component pass over the fan-out graph, instead of one
//! depth-first search (with fresh allocations) per dead neuron. The cost is
//! O(V + E) plus the sizes of the output sets that have to be merged: a component
//! with a single closed successor and no direct output edge shares that
//! successor's set without copying, so a long chain above a wide fan-out does not
//! copy the fan-out set per chain node. The elements merged are totalled and
//! capped by a ceiling (`MAX_REACHABILITY_MERGE_WORK` on the production path),
//! which bounds the memo's memory (CWE-400).
//!
//! The pass checks the discovery deadline and global cancellation once per step,
//! and every stop is reported via [`DeadNeuronScan::truncation`]: a stop during
//! the hidden-neuron scan returns no candidates, while a stop during the
//! reachability pass (deadline, cancellation or the merge ceiling, which reports
//! `ScanTruncation::CandidateCeiling`) keeps only the candidates whose outputs
//! were already complete.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::{collections::HashMap, rc::Rc, time::SystemTime};

use super::helpers::{
    ConfidenceFactor, build_record_map, compute_activation_stats, compute_mean_abs_activation,
    sort_candidates_by_score_gain, weighted_confidence,
};

use crate::analysis::recommendation::epistatic::ScanTruncation;
use crate::analysis::utils::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::topology_cache::CreatureTopologyCache;

// MIN_SAMPLES_FOR_DEAD_DETECTION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DEAD_DETECTION;

/// Threshold for mean absolute activation to consider a neuron dead.
/// Activations below this are effectively zero.
const DEAD_ACTIVATION_THRESHOLD: f32 = 1e-6;

/// Cap on the output-set elements the reachability memo may merge and retain
/// (Issue #2370). Each element is one `usize` id, so the memo holds at most about
/// 64 MB of ids; reaching the cap stops the scan with
/// `ScanTruncation::CandidateCeiling`, keeping the candidates already completed
/// (CWE-400 memory bound).
const MAX_REACHABILITY_MERGE_WORK: usize = 8_000_000;

/// Maximum activation standard deviation for a dead neuron.
/// If output varies significantly, the neuron is not truly dead.
const MAX_DEAD_STD_DEV: f32 = 1e-6;

/// Minimum fraction of samples where a neuron must be active to avoid being
/// flagged as dead. A neuron active on even a small fraction of samples with
/// meaningful activation is not dead.
const MIN_ACTIVE_FRACTION: f32 = 0.01;

/// Activation magnitude threshold for considering a single sample "active".
const ACTIVE_SAMPLE_THRESHOLD: f32 = 0.01;

/// Result of detecting a dead neuron.
#[derive(Debug, Clone)]
pub struct DeadNeuronCandidate {
    /// UUID of the dead neuron.
    pub neuron_uuid: String,
    /// Mean absolute activation across all samples.
    pub mean_abs_activation: f32,
    /// Standard deviation of activation across samples.
    pub activation_std_dev: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// UUIDs of output neurons reachable from this neuron.
    pub connected_outputs: Vec<String>,
    /// Confidence that removing this neuron is safe (0.0 to 1.0).
    pub removal_confidence: f32,
    /// Estimated creature score improvement from removing this neuron.
    pub estimated_improvement: f32,
}

/// Result of a deadline- and cancellation-bounded dead-neuron scan (Issue #2370).
#[derive(Debug, Clone)]
pub struct DeadNeuronScan {
    /// Candidates whose connected outputs are complete, sorted by removal
    /// confidence (highest first).
    pub candidates: Vec<DeadNeuronCandidate>,
    /// `Some` when the scan stopped before finishing every dead neuron:
    /// `DeadlinePassed` for the deadline or cancellation, `CandidateCeiling` when
    /// the reachability merge work reached its ceiling. Candidates completed
    /// before the stop are kept, except that a stop during the hidden-neuron scan
    /// returns none.
    pub truncation: Option<ScanTruncation>,
}

/// A dead hidden neuron whose activation statistics are known but whose
/// connected outputs are not yet computed (Issue #2370).
struct DeadNeuronDraft<'a> {
    uuid: &'a str,
    mean_abs_activation: f32,
    activation_std_dev: f32,
    sample_count: usize,
    removal_confidence: f32,
}

/// Detect dead neurons from the creature topology and recorded activations.
///
/// Unbounded by deadline; use [`detect_dead_neurons_with_deadline`] on the
/// discovery hot path.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `topo` - Optional pre-computed topology cache (Issue #754). When `None`,
///   topology maps are built locally (backward-compatible path).
///
/// # Returns
/// A list of `DeadNeuronCandidate` for neurons that are dead,
/// sorted by removal confidence (highest first).
pub fn detect_dead_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    topo: Option<&CreatureTopologyCache>,
) -> Vec<DeadNeuronCandidate> {
    detect_dead_neurons_with_deadline(creature, neuron_records, topo, &None).candidates
}

/// [`detect_dead_neurons`] that stops at `deadline` or on global cancellation,
/// and reports why it stopped via [`DeadNeuronScan::truncation`] (Issue #2370).
pub fn detect_dead_neurons_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    topo: Option<&CreatureTopologyCache>,
    deadline: &Option<SystemTime>,
) -> DeadNeuronScan {
    detect_dead_neurons_observed(
        creature,
        neuron_records,
        topo,
        deadline,
        MAX_REACHABILITY_MERGE_WORK,
        || {},
    )
}

/// [`detect_dead_neurons_with_deadline`] with `on_reachability_step` called once
/// per edge examined and once per element merged by the shared reachability
/// pass, so a test can assert on how the work grows without timing it
/// (Issue #2370).
///
/// The reachability cost is O(V + E) plus the sizes of the merged output sets.
/// `merge_ceiling` caps the total elements merged: once a merge would exceed it,
/// the scan stops with `ScanTruncation::CandidateCeiling` and keeps the
/// candidates completed so far.
pub fn detect_dead_neurons_observed(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    topo: Option<&CreatureTopologyCache>,
    deadline: &Option<SystemTime>,
    merge_ceiling: usize,
    mut on_reachability_step: impl FnMut(),
) -> DeadNeuronScan {
    // Use pre-computed cache or build locally.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Phase 1: classify hidden neurons by their activation statistics.
    let mut dead: Vec<DeadNeuronDraft<'_>> = Vec::with_capacity(topo.hidden_uuids.len());

    for uuid in &topo.hidden_uuids {
        // Issue #2370: checked before each hidden neuron. Outputs cannot be
        // computed for a partial hidden scan, so stop with no candidates.
        if deadline_passed(deadline) {
            return DeadNeuronScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }

        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_DEAD_DETECTION {
            continue;
        }

        let n = records.len() as f32;

        // Compute mean absolute activation (Issue #941: shared helper)
        let mean_abs_activation = compute_mean_abs_activation(records);

        // Compute activation standard deviation (Issue #941: shared helper)
        let activation_std_dev = compute_activation_stats(records).std_dev;

        // Check if neuron is dead: near-zero mean absolute activation AND low variance
        if mean_abs_activation >= DEAD_ACTIVATION_THRESHOLD {
            continue;
        }

        if activation_std_dev >= MAX_DEAD_STD_DEV {
            continue;
        }

        // Check active fraction: if the neuron fires meaningfully on even a small
        // fraction of samples, it is not dead (prevents false positives).
        let active_count = records
            .iter()
            .filter(|r| r.activation.abs() >= ACTIVE_SAMPLE_THRESHOLD)
            .count();
        let active_fraction = active_count as f32 / n;

        if active_fraction >= MIN_ACTIVE_FRACTION {
            continue;
        }

        // Compute removal confidence based on how dead the neuron is
        let removal_confidence =
            compute_removal_confidence(mean_abs_activation, activation_std_dev, n);

        dead.push(DeadNeuronDraft {
            uuid: uuid.as_str(),
            mean_abs_activation,
            activation_std_dev,
            sample_count: records.len(),
            removal_confidence,
        });
    }

    // Phase 2: connected outputs for each dead neuron, from one shared pass
    // (Issue #2370). Completed candidates are kept if the pass stops early.
    let mut reachability = OutputReachability::new(topo, merge_ceiling);
    let mut candidates = Vec::with_capacity(dead.len());
    let mut truncation = None;

    for draft in dead {
        if let Err(t) = reachability.explore(draft.uuid, deadline, &mut on_reachability_step) {
            truncation = Some(t);
            break;
        }

        // Estimated improvement: removing a dead neuron saves computation.
        // The improvement is small but positive (reduced overhead).
        let estimated_improvement = draft.removal_confidence * 0.001;

        candidates.push(DeadNeuronCandidate {
            neuron_uuid: draft.uuid.to_string(),
            mean_abs_activation: draft.mean_abs_activation,
            activation_std_dev: draft.activation_std_dev,
            sample_count: draft.sample_count,
            connected_outputs: reachability.connected_outputs(draft.uuid),
            removal_confidence: draft.removal_confidence,
            estimated_improvement,
        });
    }

    // Sort by removal confidence (highest first)
    candidates.sort_by(|a, b| b.removal_confidence.total_cmp(&a.removal_confidence));

    DeadNeuronScan {
        candidates,
        truncation,
    }
}

/// Sentinel for "not yet visited" (`index`) and "not yet in a closed component"
/// (`comp`) in [`OutputReachability`] (Issue #2370).
const UNSET: usize = usize::MAX;

/// One explicit-stack frame of the iterative Tarjan walk (Issue #2370).
struct TarjanFrame<'a> {
    node: usize,
    edges: &'a [String],
    next_edge: usize,
}

/// Memoised output reachability over the fan-out graph (Issue #2370).
///
/// Runs an iterative Tarjan strongly-connected-components pass, lazily from each
/// dead neuron and sharing its memo across them. When a component closes, its
/// output set is computed once from its members' direct edges and from the
/// already-closed components they reach. A component with no direct output edge
/// and exactly one closed successor shares that successor's set (`Rc`, no copy);
/// any other component merges, and the merged size is charged against a ceiling.
/// A node's connected outputs are the
/// output set of its component: outputs reachable via at least one edge, through
/// any node type, with cycles allowed.
///
/// The Tarjan walk uses an explicit call stack, so a corrupt, cyclic or very deep
/// creature cannot overflow the thread stack.
struct OutputReachability<'a> {
    topo: &'a CreatureTopologyCache,
    /// Lazily interned node names mapped to dense ids.
    ids: HashMap<&'a str, usize>,
    names: Vec<&'a str>,
    /// Tarjan discovery index per node (`UNSET` = not yet visited).
    index: Vec<usize>,
    /// Tarjan low-link per node.
    low: Vec<usize>,
    /// Closed component per node (`UNSET` = not yet in a closed component).
    comp: Vec<usize>,
    /// Tarjan node stack.
    stack: Vec<usize>,
    /// Next Tarjan discovery index.
    counter: usize,
    /// Output-node ids reachable from each closed component (sorted, deduplicated).
    comp_sets: Vec<Rc<Vec<usize>>>,
    /// Upper bound of the elements merged so far (the memo's retained size).
    merge_work: usize,
    /// Cap on `merge_work`; a merge that would exceed it is refused.
    merge_ceiling: usize,
}

impl<'a> OutputReachability<'a> {
    fn new(topo: &'a CreatureTopologyCache, merge_ceiling: usize) -> Self {
        Self {
            topo,
            ids: HashMap::new(),
            names: Vec::new(),
            index: Vec::new(),
            low: Vec::new(),
            comp: Vec::new(),
            stack: Vec::new(),
            counter: 0,
            comp_sets: Vec::new(),
            merge_work: 0,
            merge_ceiling,
        }
    }

    /// Return the dense id for `name`, interning it on first sight.
    fn intern(&mut self, name: &'a str) -> usize {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len();
        self.ids.insert(name, id);
        self.names.push(name);
        self.index.push(UNSET);
        self.low.push(UNSET);
        self.comp.push(UNSET);
        id
    }

    /// Start a Tarjan visit of node `id`.
    fn open(&mut self, id: usize) {
        self.index[id] = self.counter;
        self.low[id] = self.counter;
        self.counter += 1;
        self.stack.push(id);
    }

    /// Make the outputs reachable from `root` available.
    ///
    /// Returns `Err(DeadlinePassed)` if the deadline passed or cancellation was
    /// requested before the walk finished, or `Err(CandidateCeiling)` if a merge
    /// would exceed the merge ceiling. After an `Err` the memo is left mid-walk
    /// (open nodes on the Tarjan stack) and must not be reused.
    /// `deadline_passed` is checked once per main-loop iteration.
    fn explore(
        &mut self,
        root: &'a str,
        deadline: &Option<SystemTime>,
        on_step: &mut impl FnMut(),
    ) -> Result<(), ScanTruncation> {
        let root_id = self.intern(root);
        if self.comp[root_id] != UNSET {
            return Ok(());
        }

        let topo = self.topo;
        let mut call: Vec<TarjanFrame<'a>> = Vec::new();
        self.open(root_id);
        call.push(TarjanFrame {
            node: root_id,
            edges: topo.fan_out_for(root),
            next_edge: 0,
        });

        while let Some(top) = call.last_mut() {
            if deadline_passed(deadline) {
                return Err(ScanTruncation::DeadlinePassed);
            }

            let v = top.node;
            let edges = top.edges;
            if top.next_edge < edges.len() {
                let target: &'a str = edges[top.next_edge].as_str();
                top.next_edge += 1;
                on_step();

                let w = self.intern(target);
                if self.index[w] == UNSET {
                    self.open(w);
                    call.push(TarjanFrame {
                        node: w,
                        edges: topo.fan_out_for(target),
                        next_edge: 0,
                    });
                } else if self.comp[w] == UNSET {
                    // `w` is visited and still on the Tarjan stack.
                    self.low[v] = self.low[v].min(self.index[w]);
                }
                // Otherwise `w` is in a closed component: nothing to propagate.
            } else {
                call.pop();
                if self.low[v] == self.index[v] {
                    self.close(v, on_step)?;
                }
                if let Some(parent) = call.last() {
                    let p = parent.node;
                    self.low[p] = self.low[p].min(self.low[v]);
                }
            }
        }

        Ok(())
    }

    /// Close the component whose root is `v`, computing its output set once.
    ///
    /// Returns `Err(CandidateCeiling)`, before copying anything, if merging would
    /// push the total merge work past the ceiling.
    fn close(&mut self, v: usize, on_step: &mut impl FnMut()) -> Result<(), ScanTruncation> {
        let c = self.comp_sets.len();

        // Assign every member before computing the set, so an edge inside the
        // component is recognised by `comp == c` and skipped.
        let mut members: Vec<usize> = Vec::new();
        while let Some(x) = self.stack.pop() {
            self.comp[x] = c;
            members.push(x);
            if x == v {
                break;
            }
        }

        let topo = self.topo;
        let mut direct: Vec<usize> = Vec::new();
        let mut succ: Vec<usize> = Vec::new();
        for &m in &members {
            for target in topo.fan_out_for(self.names[m]) {
                on_step();
                let t = self.ids[target.as_str()];
                if topo.output_uuids.contains(target.as_str()) {
                    direct.push(t);
                }
                let tc = self.comp[t];
                if tc != UNSET && tc != c {
                    succ.push(tc);
                }
            }
        }
        succ.sort_unstable();
        succ.dedup();

        // Nothing to merge: share the single closed successor's set, or use an
        // empty one. No copy, no per-element step, no budget charge.
        if direct.is_empty() {
            match succ.as_slice() {
                [] => {
                    self.comp_sets.push(Rc::new(Vec::new()));
                    return Ok(());
                }
                [only] => {
                    let shared = Rc::clone(&self.comp_sets[*only]);
                    self.comp_sets.push(shared);
                    return Ok(());
                }
                _ => {}
            }
        }

        // Charge the upper bound before copying, so the ceiling bounds memory.
        let bound = direct.len() + succ.iter().map(|&s| self.comp_sets[s].len()).sum::<usize>();
        if self.merge_work.saturating_add(bound) > self.merge_ceiling {
            return Err(ScanTruncation::CandidateCeiling);
        }
        self.merge_work += bound;

        let mut set = direct;
        for &sc in &succ {
            for &o in self.comp_sets[sc].iter() {
                on_step();
                set.push(o);
            }
        }
        set.sort_unstable();
        set.dedup();
        self.comp_sets.push(Rc::new(set));
        Ok(())
    }

    /// Output names reachable from `root`, sorted ascending.
    ///
    /// Only meaningful after `explore(root)` has returned `Ok`.
    fn connected_outputs(&self, root: &str) -> Vec<String> {
        let Some(&id) = self.ids.get(root) else {
            return Vec::new();
        };
        let c = self.comp[id];
        if c == UNSET {
            return Vec::new();
        }
        let mut outputs: Vec<String> = self.comp_sets[c]
            .iter()
            .map(|&o| self.names[o].to_string())
            .collect();
        outputs.sort_unstable();
        outputs.dedup();
        outputs
    }
}

/// Compute removal confidence based on activation statistics.
///
/// Higher confidence when:
/// - Mean absolute activation is closer to zero
/// - Standard deviation is closer to zero
/// - More samples were analysed
///
/// Issue #941: refactored to use `weighted_confidence` shared helper.
fn compute_removal_confidence(
    mean_abs_activation: f32,
    activation_std_dev: f32,
    sample_count: f32,
) -> f32 {
    let activation_factor = 1.0 - (mean_abs_activation / DEAD_ACTIVATION_THRESHOLD).min(1.0);
    let variance_factor = 1.0 - (activation_std_dev / MAX_DEAD_STD_DEV).min(1.0);
    let sample_factor = (sample_count / 1000.0).min(1.0);

    weighted_confidence(
        &[
            ConfidenceFactor {
                value: activation_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: variance_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: sample_factor,
                weight: 0.2,
            },
        ],
        0.5,
        1.0,
    )
}

/// Convert dead neuron candidates into coordinated structural candidates.
///
/// Each dead neuron produces a `RemoveNeuron` coordinated candidate.
/// The NEAT-AI controller will validate the removal through ablation testing
/// before actually applying it.
pub fn dead_neurons_to_coordinated_candidates(
    candidates: &[DeadNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: c.neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Dead neuron {}: mean abs activation {:.2e}, std dev {:.2e}, {} samples → remove to reduce wasted computation",
                c.neuron_uuid, c.mean_abs_activation, c.activation_std_dev, c.sample_count
            )),
        });
    }

    // Sort by expected improvement (best first) (Issue #941: shared helper)
    sort_candidates_by_score_gain(&mut results);

    results
}

#[cfg(test)]
#[path = "dead_neuron_deadline_tests.rs"]
mod deadline_tests;
