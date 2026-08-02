# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

#### `remove-low-impact` ranks on a live activation-weighted signal (Issue #1923)

The structural removal path built every candidate with `mean_activation: 0.0`
and a reason string calling the activation-weighted gate "deferred to analysis";
nothing downstream ever resolved the deferral. The #1920 cache study measured
the cost — `removalCandidate.impact` correlated with realised gain at r = −0.036
and `meanActivation` had zero variance across all 67 cached records, so the
strategy behind 79% of realised gain picked arbitrarily from its eligible pool.
The gate now resolves in the same call, from a two-column
(`neuron_uuid` + `activation`) streaming Parquet pass that materialises no
records and runs only after focus selection is fixed, so it does not reintroduce
the Issue #1766 focus stall. Candidates are ranked on
`removalSavings − activationWeightedImpact`, and Issue #892's active-neuron gate
is now reachable on this path. When records cannot be read the fields stay
unmeasured, the `reason` says so, a WARN is emitted, and unmeasured candidates
rank below every measured one.

#### `append_discovery_records` no longer acknowledges a cancelled session (Issue #1876)

`append_records` could return `Ok(records_in_batch)` for a session that
`cancel_session` or the TTL sweep had already removed, reporting records as
written moments before the session's `.parquet.tmp` file was deleted. Sessions
now carry a lock-free cancellation tombstone that `append_records` checks after
taking the per-session lock and again after its writes, so the call fails with
`Session cancelled: <sessionId>` instead. Cancelling still never waits on an
in-flight Parquet write.

### Removed

#### `ModuleStarvationTracker` deleted rather than wired (Issue #1793)

The per-(creature, module) starvation cooldown (Issue #1273) was never populated
in production: the sole production caller reached it through a wrapper that
hard-coded `starvation_tracker = None, current_epoch = 0`, and the tracker could
have no producer at its documented per-`analyze_all` scope. The live
`ModuleOutcomeTracker` gate already suppresses persistently-failing modules from
real data, so the dead layer was removed instead of duplicated.

- Removed `analysis::module_starvation_tracker`, the
  `prepare_and_detect_discovery_modules_with_starvation` /
  `detect_discovery_modules_parallel_with_starvation` wrappers, and the
  `DiscoveryModuleDetectionEntry.starved` flag.
- **Breaking (FFI payload):** `droughtDiagnostic.starvedModuleCount` is no
  longer serialised — it could only ever be `0`. The `module_starved` rejection
  reason is gone from `ALL_REJECTION_REASONS` for the same reason.
- **Breaking (config):** `NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK`
  and `NEAT_AI_DISCOVERY_MODULE_STARVATION_COOLDOWN_EPOCHS` are removed; they
  configured a tracker that no longer exists.

### Added

#### Fail-loud candidate reconciliation on every pass (Issue #1802)

Issues #1796–#1801 wired the six silent drop paths the #1782 diagnosis found into
`RejectionBreakdown`, but nothing stopped the seventh being added the same way.
Each surface now owns a per-pass `CandidateLedger` and asserts
`considered == accounted` where its breakdown is finalised. See
[`docs/analysis/candidate-reconciliation-1802.md`](docs/analysis/candidate-reconciliation-1802.md).

- On a mismatch the residual is recorded under the new stable reason
  `unaccounted_drop`, one `tracing::warn!` names the surface and the delta, and a
  `debug_assert!` fires under strict mode — on by default for debug builds, so a
  new silent drop path fails CI. Override with
  `NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION`.
- `synapseMetadata` / `neuronMetadata` carry a `candidate_reconciliation` payload
  so callers can confirm the invariant positively rather than by the absence of a
  failure marker.
- Wiring the ledger exposed six further bare `continue`s, all now counted:
  `below_improved_ratio` (three sites), `target_saturated` (candidate squash
  compounding a near-saturated target), `cpu_pre_reject_no_signal` (no usable
  weight could be fitted), `zero_improvement` (non-positive post-evaluation
  improvement), and the new reason `degenerate_weight_update` (two sites where a
  clamped weight-update delta collapsed to a no-op).
- A balanced pass logs nothing, records nothing, and costs two relaxed atomic
  loads per surface.

### Fixed

#### Structural removal triage collapsed into one implementation (Issue #1805)

Two near-identical copies of the least-impact removal criterion existed side by
side and had already drifted once. The criterion — savings-vs-impact, the
`REMOVAL_CANDIDATE_BOOST` application point, the non-finite-impact policy
(#1804), the noise-floor re-gate (#1142), the hidden-only filter and the
net-improvement-descending sort — now lives only in
`identify_structural_removal_candidates`.

- `focus::triage_removal_candidates` is retained as a documented thin **adapter**
  over that single implementation (no public items removed, so no breaking
  change). Its per-candidate `reason` string now carries the shipped path's
  wording, and its pass is `rayon`-parallel rather than serial.
- **Fixed:** the shipped FFI path (`rank_focus_neurons`) took `costOfGrowth` raw,
  so a non-positive value silently produced zero candidates and a NaN value
  emitted every hidden neuron — including high-impact ones — with NaN savings.
  It now shares the adapter's validation: non-finite or non-positive falls back
  to the `1e-7` default with a WARN.

#### Quality-skipped candidates count as an abundance rejection (Issue #1799)

Quality-based module skipping (Issue #1074) discarded the remaining discovery
modules' candidates without incrementing any rejection counter, so the drop was
invisible to `candidate_starvation::classify`, which reads the
`RejectionBreakdown` alone.

- New stable reason `module_skipped_quality_satisfied`, recorded with the
  skipped module's `candidates_produced` (unit: **candidates**, not modules) so
  `signals_from_breakdown` totals stay meaningful.
- Classified as an **abundance** rejection alongside `budget_truncated` and
  `per_target_cap` — the skip fires because the pass already holds enough
  high-quality candidates, so a quality-skipping pass is never classified
  `CandidateStarved` on the strength of these drops.
- Skipping behaviour, `modulesSkippedByQuality` metadata, and the per-module
  stats are unchanged — observability only.

#### Failure-cache entries expire; fingerprint skip gains an escape hatch (Issue #1781)

Persisted suppression state could hold a creature in drought indefinitely.

- `FailureCacheEntry` accepts an optional `ageEpochs` (alias
  `epochsSinceRecorded`). Entries at or beyond 20 passes suppress nothing; a
  coarse (target-agnostic) entry keeps its wildcard reach for only 5 passes, so
  one `coordinated-structural` failure can no longer suppress every coordinated
  candidate forever. An entry with no reported age matches exactly and gets no
  wildcard reach.
- The `previousNeuronFingerprints` skip is bypassed after 3 consecutive empty
  passes — the structural fingerprint cannot change during a drought, so the
  cache was skipping every focus neuron regardless of new recorded data.
- A whole-pass fingerprint drop now records one `fingerprint_unchanged`
  rejection per skipped neuron, surfaced through
  `zeroCandidateSummary.rejectionBreakdown` and fed to the starvation
  classifier as an upstream (starvation) reason.

### Security

#### Parquet decode is now bounded, not predicted (Issue #1869)

The pre-load admission decision was `compressed_file_size × 3`, and the only
memory-budget check ran *after* the allocation it was meant to prevent. Parquet's
dictionary and RLE encodings routinely beat 3:1 on this schema's repeated neuron
UUIDs, so a file projected at "300 MB, fits comfortably" could decode into many
gigabytes and abort the host process.

- `estimate_parquet_in_memory_bytes` now projects from the Parquet **footer** —
  the exact decompressed row count and `errors` value count — with the old
  `file_size × 3` heuristic kept only as a floor.
- The reader charges every materialised record against a cumulative
  `DecodeBudget` **inside** its batch loop, alongside the existing cancellation
  and deadline checks, and fails loud with a typed `MemoryExhausted` error the
  moment the ceiling is reached. The analysis phase forwards its
  `max_analysis_memory_mb`; budget-free paths use
  `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB`, defaulting to half of total RAM.
  This closes the `max_obs` gap — that limit caps distinct *observations*, so a
  file whose rows all share one `obs_index` was unbounded even at `maxObs: 1`.
- The snapshot exporter's dense `O(neurons × observations)` grid is checked
  against the same ceiling before the first vector is allocated.

#### Pre-commit gate no longer bypasses the dependency quarantine (Issue #1865)

`./quality.sh` — the documented pre-commit step — ran `cargo upgrade
--incompatible` followed by `cargo update`, force-upgrading every direct and
transitive crate with no age check. That bypassed both the Renovate
`minimumReleaseAge` window and the `VIBE_BUMP_QUARANTINE_HOURS` gate, so a crate
poisoned minutes earlier was pulled in and its `build.rs` executed locally.

- Removed the upgrade/update step from `quality.sh`: the quality gate verifies
  the tree and never mutates its dependency graph. Bumps go through
  `./bump-deps.sh` or Renovate.
- `bump-deps.sh` now enforces the quarantine against the **resolved lockfile**,
  not just the manifest requirement strings. `Cargo.lock` is diffed against its
  pre-bump state, every in-quarantine change — transitive included — is pinned
  back with `cargo update --precise`, and a newly-pulled package inside the
  window fails the run loud (exit 8) because there is no earlier version to pin
  back to. Unknown publish times fail closed.
- `--no-network` now also skips the lockfile refresh: an offline re-resolve
  cannot be age-checked.

### Changed

#### CI `version-increment` no longer re-resolves the lockfile (Issue #1878)

The `version-increment` job ran a bare `cargo update` after bumping the patch
version, then committed and pushed the result with the `ACTIONS_PUSH` PAT. That
re-resolved every direct **and** transitive dependency to the newest published
version with no age check, so a crate published minutes earlier landed in the
committed `Cargo.lock` on every PR — the same bypass as Issue #1865, on the CI
path rather than the pre-commit path.

- The job now runs `cargo update --workspace`, which records only this crate's
  own new version and leaves every dependency resolution untouched.
- A fail-loud guard asserts the `Cargo.lock` diff is exactly the crate's own
  version bump; anything else aborts the job rather than pushing an
  unquarantined dependency graph. Dependency movement belongs to
  `./bump-deps.sh` (which enforces `VIBE_BUMP_QUARANTINE_HOURS`) and Renovate.

#### Remove private-repo links and mentions from archived PR summaries (Issue #1726)

The archived PR summaries under `docs/archive/pr-summaries/` named private
`stSoftwareAU` repositories — and in several places linked directly to private
issues, commits, and checkout paths that 404 for the public. Every such file
ships in every public clone and is indexed by search engines.

- Reworded all name-level mentions across 65 archived summaries to concept level
  ("the production discovery cache", "a large production creature",
  "production-cluster fixture"); dropped the direct private issue/commit links
  and the private checkout-path references, keeping commit hashes, creature
  ids, and metrics intact.
- Added `tests/issue_1726_archive_no_private_repo_names.rs` — the regression gate
  that fails loudly if an archived summary reintroduces a private repository
  name (the archive counterpart to the #1723 active-docs gate).

#### Reword private-repo references in test comments, rename the strip-pattern test file (Issue #1725)

Test doc comments, helper names, and one test *file* name cited private
`stSoftwareAU` deployments (production sampler-cache commits, a production
corruption-log name, and several numbered production deployments) as their
motivating evidence. `cargo test` prints those file and test names on every
run, pointing public contributors at evidence they cannot reach.

- Reworded every such citation to concept level ("production failure-cache
  evidence", "a large production creature"); internal issue numbers preserve
  maintainer traceability. No test behaviour changed.
- Renamed the FFI strip-pattern test file that carried a private-deployment name
  to `issue_1188_strip_pattern_rejection.rs` and updated the `tests/ffi/main.rs`
  harness reference; the private-named `*_rejects_*` test names became
  behaviour-describing.
- Extended `tests/source_free_of_private_repo_names.rs` to walk `tests/` as
  well, match case-insensitively (catching lower-case identifiers), and guard
  file names — the regression gate that keeps private names out.

### Added

#### Two-stage per-target source budget (Issue #1542)

Caps the number of upstream source neurons that reach the expensive
sample-building + GPU evaluation stage for each focus target, controlled by the
new `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET` environment variable. On sparse
deep creatures (thousands of inputs × several focus targets) unbounded source
enumeration is the dominant wall-clock multiplier that micro-optimisations
cannot address.

- Stage 1 is the existing cheap CPU pre-score (`order_eligible_sources`), which
  already places the highest-priority sources first (unused-input bias,
  input-index bias, hidden interleaving). Stage 2 is the new
  `apply_source_budget`, which deterministically truncates the ordered list to
  the top-K, dropping only the low-priority tail before any GPU work. Candidate
  scoring semantics for the sources that *are* evaluated are unchanged.
- Applied to both synapse analysis (`target_analysis`) and neuron analysis
  (`neuron::preparation`) so the two share one budget.
- `0`, unset, empty, or invalid values preserve the pre-#1542 unlimited
  behaviour (back-compat, and the shipped default).
- New `benches/source_budget.rs` A/B measures end-to-end `analyze_synapses` for a
  large fan-in target at unlimited vs K ∈ {64, 128, 256}.

#### Propagation-aware change-squash gain estimate (Issue #1532)

Extends the #1518 propagation-aware approach — which fixed the **remove-neuron**
estimate — to the **change-squash** estimate path, the second estimate path
cited in the production discovery cache at commit `2596f073`. For the recorded failure
(`neuron-1481550544`, `SELU → SQUARE`) the pipeline emitted a near-zero
placeholder gain of `+8.6e-10` while the measured effect was `-0.000341` —
~400,000× too small and the wrong sign.

- New `analysis::change_squash_gain` module exporting
  `estimate_change_squash_gain(creature, neuron_uuid, current_local_error,
  proposed_local_error)`. It reuses `compute_impacts_public` for the neuron's
  propagation-aware downstream influence (DRY with the remove-neuron estimator)
  and scales it by the local perturbation the swap induces (the reduction in the
  neuron's local error). The signed estimate is non-positive: on a converged
  network, re-fitting a neuron's activation disrupts the downstream layers
  trained around its original behaviour.
- New production-scale guards in `tests/change_squash_propagation.rs` against the
  committed production-scale fixture: the estimate matches the measured actual in sign
  and within one order of magnitude (the #1529 pass criterion), and the near-zero
  placeholder path is never re-emitted.

#### Deprioritise remove-neuron candidates during a search-exhaustion drought (Issue #1448)

On the plateaued production creature (#1418, 1673 neurons at score
~0.4224) the destructive remove-neuron path dominated the failure cache (bucket
`247b83ab`: 9 of 11 files) with low-impact proposals that never pass scoring.
The #1425 failure-cache calibration shrinks remove-neuron predictions but only
*after* the failures are cached, so it cannot stop the first wave of
over-confident proposals on a creature whose search is already exhausted.

- New `analysis::remove_neuron_drought` module. When the trailing-failure streak
  reaches the (task-calibrated) drought threshold **and** the drought classifies
  as `search_exhaustion` (reusing the #1421/#1424 environmental-vs-exhaustion
  disambiguation), single-op `RemoveNeuron` coordinated candidates have their
  `expectedCreatureScoreGain` multiplied by a deprioritisation factor (default
  `0.1`). The demoted gains sort below the constructive change types
  (add-synapse / squash / multi-op coordinated) and the most over-confident ones
  fall through the existing coordinated noise floor, so the destructive module
  yields budget to the constructive ones during a plateau.
- Environmental droughts (memory / GPU gated passes) are left untouched — only a
  genuine search-exhaustion drought triggers the deprioritisation.
- New `NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR` lever (clamped
  `[0.001, 1.0]`; `1.0` disables). Demotions are counted under the new
  `remove_neuron_drought_deprioritised` rejection reason and a single
  `tracing::warn!` reports how many candidates were demoted.

#### Zero-candidate summary on the analysis response (Issue #1446)

When a discovery pass found nothing, operators saw an unhelpful "Built 0
candidates" block with no visible reason, even though the Rust side already
populated `rejectionBreakdown`, `droughtDiagnostic`, and `creatureDroughtAlarm`
in analysis metadata. Those diagnostics had to be hunted for in `.discovery/`
JSON sidecars or by enabling verbose logging.

- `analyze_parallel` now attaches a `zeroCandidateSummary` object to the
  top-level response **only when the pass produced no candidates of any kind**.
  It carries `dominantRejectionReason` (merged across synapse and neuron
  analysis), the full `rejectionBreakdown`, the `droughtDiagnostic` /
  `creatureDroughtAlarm` (when active), and `environmentalGates` (memory / GPU /
  cancellation flags) so the dominant reason is identifiable at a glance.
- For genuinely-empty passes (not environmentally gated) a single
  `tracing::warn!` event names the dominant rejection reason and drought streak.
- New public types `ZeroCandidateSummary` and `EnvironmentalGatesJson` plus the
  `build_zero_candidate_summary` helper, documented in `docs/FFI_API.md`.

#### Diversity-aware focus selection (Issue #1445)

On a plateaued mature network a single high-impact neuron could hold ~98.5% of
the focus-selection roulette weight (a large production creature), so the weighted
roulette collapsed to a single target and discovery revisited the same
neighbourhood every pass. Impact-weighted ranking only ordered neurons; it did
not enforce diversity in the final focus set.

- New `src/focus/selection.rs` (`select_focus_neurons`) adds a deterministic
  selection layer over the ranked list with a **diversity floor** (stratified
  pick across the ranked bands when one neuron exceeds its even `1/N` share) and
  **drought-aware round-robin rotation** across the top `K × N` neurons once
  `epochsSinceLastAcceptedCandidate` reaches the drought threshold
  (`NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD`, #1202).
- `rank_focus_neurons` now returns a `focusSelection` block
  (`selected`, `rawWeightConcentrationRatio`, `weightConcentrationRatio`,
  `diversityFloorApplied`, `rotationApplied`, `poolSize`) and each ranked neuron
  carries its `weightedScore`. A `focus_selection_weight_concentration_high`
  WARN fires when the raw concentration exceeds `0.5`.
- New optional FFI inputs `epochsSinceLastAcceptedCandidate` and `focusSetSize`
  (default 6); both are backwards compatible when omitted.

### Changed

#### Evidence citations in source, benches, and the example reworded to concept level (Issue #1724)

Comments across `src/`, all four `benches/`, and `examples/generate_snapshot.rs`
cited private downstream repositories by name — commit hashes and creature ids
as the derivation evidence for scoring constants and bench workloads. Public
readers cannot inspect any of it, so the citations carried no verification value
while continuously naming private infrastructure.

- Evidence is now cited at concept level ("production discovery-cache
  analysis", "production scale", "the calling host layer"). The internal issue
  numbers already carried in most comments preserve traceability for
  maintainers; no constant, threshold, or behaviour changed.
- `examples/generate_snapshot.rs` documents its inputs generically (a
  discovery-data parquet plus the creature JSON) instead of local checkout paths
  inside a private sibling repository, so the printed usage is reproducible.
- New `tests/source_free_of_private_repo_names.rs` is the regression gate that
  keeps shipped source free of private repository names.

#### Deadline-breakdown log marker renamed to `DEADLINE-BREAKDOWN` (Issue #1723)

The consolidated per-cycle deadline-consumption summary and its `STARVED`
warning were tagged with a marker named after a private downstream deployment,
which meant the public troubleshooting guide in `README.md` could not describe
the signal without naming that deployment.

- The `marker` field and the summary-line prefix are now the deployment-neutral
  `DEADLINE-BREAKDOWN`, exported as
  `analysis::deadline_breakdown::DEADLINE_BREAKDOWN_MARKER` so the string has a
  single source of truth.
- **Operator action:** log greps for the old marker must be updated. Nothing
  else about the event changed — same fields, same values, same emission points.

### Fixed

#### Harmful-neuron (remove-neuron) failure-cache calibration correction (Issue #1425)

Harmful-neuron (`remove-neuron`) candidates over-predicted their score gain by
~800× (failure bucket `247b83ab`): predicted `+0.166` vs actual `≈0`. Because
`expected_creature_score_gain` is the candidate ranking key, these inflated
predictions crowded the top of the candidate list every pass, failed scoring,
and landed in the failure cache only to be regenerated next time — sustaining
the discovery drought.

The failure-cache calibration correction (`CalibrationCorrection::correction_for`,
Issues #1131 / #1162) was applied to the add-neuron and add-synapse paths but
not to the harmful-neuron path.

- A single-op `RemoveNeuron` coordinated candidate is now keyed under the new
  `remove-neuron` change type (`CHANGE_TYPE_REMOVE_NEURON`) when its predicted
  gain is calibrated, so repeated remove-neuron over-predictions shrink future
  remove-neuron predictions via the failure-cache EWMA.
- Multi-op coordinated candidates and non-removal single ops keep the generic
  `coordinated-structural` correction — regression preserved.

#### Wall-clock budget on focus ranking with graceful fallback (Issue #1375)

Focus ranking previously had no wall-clock bound. In the #1373 incident it ran
for 1h 11m and contributed to the whole discovery task overrunning its 3h budget
and being killed. The per-chunk Rust FFI analysis already enforces a budget;
focus ranking now has the same safety net.

- New `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` env var (default `120000` =
  2 minutes) bounds every `focus::rank_focus_neurons*` run. `0` disables the
  bound; other values clamp to `[1000, 3600000]`.
- The budget is checked between passes and inside the per-neuron loops
  (record verification and the parallel ranking map). On exceed the run aborts
  with a structured `DiscoveryError::Timeout` (classified retryable) so the
  TypeScript caller routes it into the existing local-ranking fallback instead
  of running unbounded.
- Fast / preload runs are unchanged — the per-neuron check is a single
  `Instant::now()` comparison with negligible overhead.

## [v0.74.74]

### Fixed

#### Squash-bounded impact attribution and consumer-gate model (Issue #1300)

Mirrors `NEAT-AI-Explore#266`. Three correctness gaps in the impact calculation:

- **Squash-bounded contribution.** Per-synapse contribution now respects the
  downstream squash's emit magnitude (`squash_emit_magnitude` in
  `src/activations.rs`). For threshold squashes (STEP/BIPOLAR) the previous
  behaviour returned the full `child_impact` for every inbound synapse —
  overstating influence by `N×` for `N` inbound synapses. Threshold squashes
  now normalise by total inbound weight and apply the emit cap, same as Linear
  bounded squashes (TANH, LOGISTIC, HARD_TANH, ...).
- **Min-gate awareness.** New `ConsumerContract` / `OutputGate` API and
  `compute_impacts_with_contract` let callers declare downstream
  `min(output, constant)` / `max(output, constant)` gates. Output impacts are
  scaled by the gate's pass-through probability computed from records.
- **Auto-derived regime thresholds.** New
  `derive_regime_threshold_from_records` helper picks a percentile from the
  recorded distribution so callers do not need to hard-code constants.

See `docs/IMPACT_CALCULATION.md` for the updated semantics.

## [v0.72.34]

### Added

#### Memory budget enforcement (Issue #1028)

Analysis phase now accepts `max_analysis_memory_mb` to cap Rust-side memory
consumption. When exceeded, analysis returns early with
`memory_budget_exceeded: true`, preventing unbounded memory growth on
constrained machines.

#### Deadline enforcement for detection modules (Issue #1029)

Detection modules now respect `analysis_deadline_ms`, aborting early when the
wall-clock deadline is reached. Partial results are returned and coverage
improves over repeated runs.

#### Rust-side memory usage FFI (Issue #1027)

New `discovery_memory_usage_bytes()` FFI symbol exposes the Rust allocator's
current heap usage. Callers can combine this with `Deno.memoryUsage().heapUsed`
for accurate total-process memory monitoring.

#### MCMC-inspired candidate selection (Issues #1017–#1021)

- **MCMC pipeline audit** (#1017): documented that the pipeline is deterministic
  (not true MCMC) but can benefit from MCMC-inspired techniques.
- **Metropolis-Hastings acceptance** (#1018): optional probabilistic acceptance
  via `NEAT_AI_DISCOVERY_MH_TEMPERATURE` environment variable.
- **Adaptive proposal distribution** (#1019): Gaussian proposals replace the
  fixed 9-variant weight grid, with per-target-type sigma adaptation.
- **Temperature scheduling** (#1020): linear and exponential cooling schedules
  for exploration-exploitation balance.
- **MCMC diagnostics** (#1021): lock-free acceptance rate tracking and diversity
  metrics for monitoring candidate selection quality.

### Performance

#### Concurrent synapse/neuron analysis (Issue #1002)

Synapse and neuron analyses now run concurrently via `rayon::join` with a shared
GPU queue, reducing wall-clock time by overlapping CPU-bound sample building.

#### Parallel candidate compression (Issue #1003)

Identity and nonlinear candidate compression now run in parallel via
`rayon::join`, both operating on immutable references.

#### Overlapped compression and detection (Issue #1004)

Candidate compression now overlaps with discovery module detection using a
split-detect-merge pattern, further reducing wall-clock time.

#### New benchmarks (Issues #1001, #1006, #1009)

- Pipeline wall-clock utilisation benchmarks (#1001)
- SIMD baseline micro-benchmarks for hot numerical loops (#1006)
- Compiler auto-vectorisation audit benchmarks (#1009)

## [v0.2.18]

### Fixed

#### Synapse analysis for all squash types

STEP neurons now get proper simulation functions, matching BIPOLAR handling.

Previously, `target_simulation_fn` returned `None` for STEP (claiming "handled via threshold-
crossing model"), but returned `Some` for BIPOLAR. This inconsistency meant:
- **BIPOLAR**: Got accurate simulation predicting output flips (-1 ↔ 1)
- **STEP**: Fell back to linear error model (inaccurate for threshold functions)

The fix adds a proper simulation function for STEP: `|x| if x > 0.0 { 1.0 } else { 0.0 }`.
Both STEP and BIPOLAR now use simulation that accurately predicts when synapse contributions
will cross the zero threshold and flip the discrete output.

## [v0.2.17]

### Added

#### Synapse-friendly discovery

Synapse analysis now runs first when a deadline is set, preventing starvation.

Previously, neuron analysis ran first and could consume the entire `analysisDeadlineMs` budget,
leaving zero time for synapse analysis. This caused "no add-synapses candidates" even when
many potential synapses existed.

**Analysis ordering (v0.2.17+)**:
| Deadline Set | Analysis Order | Rationale |
|--------------|----------------|-----------|
| Yes | Synapses → Neurons | Prevents synapse starvation |
| No | Neurons → Synapses | Original behaviour preserved |

**New metadata fields (v0.2.17+)**:

The analysis output now includes metadata to diagnose prediction issues:

**Synapse analysis metadata** (`synapseMetadata`):
| Field | Description |
|-------|-------------|
| `targetValueAvailable` | Whether pre-activation `value` data was in recordings |
| `saturationAwareSimulationUsed` | Whether saturation-aware simulation was used |
| `candidatesFound` | Total candidates found before truncation |
| `candidatesReturned` | Candidates returned after `maxCandidates` limit |

**Neuron analysis metadata** (`neuronMetadata`):
| Field | Description |
|-------|-------------|
| `candidatesFound` | Total candidates including paired variants, before truncation |
| `candidatesReturned` | Candidates returned after `maxCandidates` limit |

**Why this matters**:
- If `targetValueAvailable = false`, predictions may be inaccurate for saturating activations
  (HARD_TANH, TANH, etc.) because the linear fallback model is used
- If `candidatesFound > candidatesReturned`, increase `maxSynapseCandidates`/`maxNeuronCandidates`
- If synapse candidates are still zero, check if deadline is too short or focus neurons are filtered

## [v0.2.3] - Issue #132

### Fixed

#### Configurable costOfGrowth

**CRITICAL BUG FIX**: The `costOfGrowth` threshold was incorrectly hardcoded to `0.01`
(since v0.1.145) based on a false assumption that "TypeScript used 0.01". **NEAT-AI has
always used `1e-7`** - it was never 0.01. This bug caused **418 false removal candidates**
in production!

The threshold is now **configurable** with the correct default of `1e-7`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `costOfGrowth` | `Option<f32>` | `1e-7` | Threshold for removal candidates |

**JSON input example**:
```json
{
  "parquetFile": "/path/to/records.parquet",
  "creature": { "..." : "..." },
  "maxResults": 100,
  "costOfGrowth": 1e-7
}
```

**Behaviour**:
- Neurons with `activation_weighted_impact < costOfGrowth` are removal candidates
- Default `1e-7` matches NEAT-AI's Score.ts complexity formula
- NEAT-AI should pass its configured `costOfGrowth` value for consistency

**Common costOfGrowth values**:
| Value | Purpose |
|-------|---------|
| `1e-7` | Default - standard complexity penalty per neuron |
| `1e-9` or lower | Encourages creature expansion for evolution on new neurons |
| Higher values | More aggressive pruning (use with caution) |

**Why the old hardcoded `0.01` was wrong**: With threshold `0.01`, neurons with impact `1e-5`
were incorrectly flagged for removal even though they contribute meaningfully to output.
The correct default `1e-7` ensures only truly negligible neurons are removal candidates,
while still allowing users to override for specific use cases like expansion.

```
activation_weighted_impact = structural_impact × mean_absolute_activation
```

Where:
- `structural_impact` = NORMALISED impact through the network
- `mean_absolute_activation` = sum(|finite activation|) / finite_record_count

**Note**: Non-finite activation values (NaN, Infinity) are filtered out when computing
`mean_absolute_activation` to prevent corruption of the removal candidate ranking.

**Normalised impact calculation**:

For each synapse from neuron A to target B with weight w:
```
A's contribution to B = |w| / total_inbound_to_B × B's_impact
```

This is **recursive** - if B has 100 inputs, A only contributes 1/100th of B's signal.

## [v0.2.2] - Issue #130

### Fixed

#### Source variance discounting

**CRITICAL BUG FIX**: Predictions were massively over-estimated when source neurons had
constant or near-constant activation. A constant source cannot reduce error correlation
because it only adds a fixed offset to the target - like adjusting the bias.

**Production example**: `input-1244` had variance 0.000000 (completely constant), yet the
model predicted 29.6% error reduction. Actual result was 0%.

| Source std dev | Discount factor | Effect |
|---------------|-----------------|--------|
| ≥ 0.05 | 1.0 (100%) | Full prediction |
| 0.025 | 0.5 (50%) | Half prediction |
| 0.01 | 0.2 (20%) | Heavy discount |
| 0.0 | 0.0 (0%) | Skip entirely |

**How it works**:
1. Compute source activation variance from recorded samples
2. Calculate discount factor: `min(1.0, source_std_dev / 0.05)`
3. Apply discount to predicted improvements

**Why 0.05 threshold?** Production analysis showed sources with std dev < 0.05 consistently
produced unreliable predictions. Sources need meaningful variation to correlate with
target error.

**Key insight**: The prediction model assumes `weight × source_activation` correlates with
target error. If `source_activation` is constant, the correlation is zero regardless of weight.
TypeScript already replaces constant neurons with constants - this fix makes the Rust
predictions match that reality.

## [v0.2.1] - Issue #130

### Fixed

#### Normalised impact calculation

**BUG FIX**: Hidden neurons were incorrectly getting `targetNeuronImpact = 1.0`
(same as output neurons) when they had large weights to outputs. This caused
predictions to NOT be discounted, leading to massive overestimation.

**The problem**: The impact formula was using absolute weights:
```
contribution = |weight| × child_impact
```

With weight 3.0 to output: `3.0 × 1.0 = 3.0` (then clamped to 1.0).
Result: Hidden neuron treated like output → NO discounting applied.

**The fix**: Use normalised weights as documented:
```
contribution = |weight| / total_inbound_weight × child_impact
```

This measures **attribution** (fraction of influence), not sensitivity:

| Scenario | Normalised Impact | Meaning |
|----------|-------------------|---------|
| Sole input to output | 1.0 | 100% influence |
| 50% of output's input weight | 0.5 | 50% influence |
| 1% of output's input weight | 0.01 | 1% influence |

**Key properties**:
- Output neurons: impact = 1.0 (always)
- Hidden neurons: impact ≤ 1.0 (depending on fraction of total input weight)
- Multiple competing inputs: impact dilutes proportionally
- Sum of all inputs to a neuron = 1.0 (fractions sum to whole)

**Zero-weight edge case (v0.2.2)**: When all inbound synapses to a neuron have
`weight == 0.0`, the normalised formula would compute `0.0 / 0.0 = NaN`. This is
now handled: zero total weight means zero contribution, so impact = 0.0.

## [v0.1.169] - Issue #128

### Changed

#### Creature-level metrics

**CRITICAL CHANGE**: All discovery candidates now return **creature-level** metrics instead
of neuron-level percentages. The old `expectedImprovementPercentage` field has been
**removed** and replaced with clearer creature-level fields.

**The goal of discovery is to improve the CREATURE'S SCORE.** Every candidate now includes:

| Field | Description | Value Range |
|-------|-------------|-------------|
| `targetNeuronImpact` | Impact of target neuron on creature (output=1.0, hidden<1.0) | 0.0 - 1.0 |
| `expectedCreatureErrorReduction` | Expected reduction in creature's error | 0.0 - 1.0 (ratio) |
| `expectedCreatureScoreGain` | Expected improvement in creature's score | 0.0 - 1.0 (ratio) |

**Why the change?** The old `expectedImprovementPercentage` was measuring the **target
neuron's** error reduction, not the **creature's** error reduction. A hidden neuron with
0.0001 impact and 33% neuron-level improvement would contribute only 0.0033% to the
creature's score - less than the cost of growth! The new fields make this transparent.

**How it works**:
- `targetNeuronImpact` shows the target neuron's weighted path to outputs
- `expectedCreatureErrorReduction` = neuron-level improvement × targetNeuronImpact
- `expectedCreatureScoreGain` = expectedCreatureErrorReduction (since score = 1 - error)

**Example**: A candidate improving a hidden neuron's error by 70% where the neuron has impact 0.1:
```
targetNeuronImpact: 0.1
expectedCreatureErrorReduction: 0.07  (70% × 0.1 = 7% creature-level)
expectedCreatureScoreGain: 0.07
```

**Candidates are sorted by `expectedCreatureScoreGain`** (highest first), so the best
candidates for the creature appear at the top.

**TypeScript usage**:
```typescript
// Use the creature-level score gain directly
const creatureLevelImprovement = candidate.expectedCreatureScoreGain;

// The impact is also available for transparency
console.log(`Target impact: ${candidate.targetNeuronImpact}`);

// Display as percentage
console.log(`Expected score gain: ${creatureLevelImprovement * 100}%`);
```

### Fixed

#### Field naming and percentage calculations (Issues #124, #128)

**IMPORTANT**: Rust returns creature-level metrics that are already ratios (0.0 - 1.0).
Multiply by 100 to display as a percentage.

| Field | Type | Value Range | To Display as % |
|-------|------|-------------|-----------------|
| `expectedCreatureScoreGain` | **Ratio** | 0.0 - 1.0 | `value × 100` |
| `expectedCreatureErrorReduction` | **Ratio** | 0.0 - 1.0 | `value × 100` |
| `targetNeuronImpact` | **Ratio** | 0.0 - 1.0 | `value × 100` |

**These fields are creature-level ratios**, calculated as:
- `expectedCreatureScoreGain` = (baseline_creature_error - new_creature_error) / baseline_creature_error
- This accounts for the target neuron's impact on the creature

```typescript
// Display as percentage - just multiply by 100
const scoreGainPercent = candidate.expectedCreatureScoreGain * 100;
console.log(`Expected score gain: ${scoreGainPercent.toFixed(2)}%`);

// The impact shows how much the target neuron affects the creature
const targetImpactPercent = candidate.targetNeuronImpact * 100;
console.log(`Target neuron contributes ${targetImpactPercent.toFixed(1)}% to creature output`);
```

**Recommendation**: Use `expectedCreatureScoreGain` for ranking and displaying candidates.
All fields are already creature-level, so no additional calculations are needed.

## [v0.1.167] - Issue #123

### Fixed

#### Saturation detection for add-neuron candidates

**CRITICAL FIX**: Candidates with saturated neurons are now detected and rejected before being
returned. This prevents massive prediction failures where expected improvement is orders of
magnitude higher than actual improvement.

**The problem** (Issue #123):
- Expected error reduction: 4.67%
- Actual error reduction: ~0% (2.5×10⁻¹⁴)
- This is a 12 orders of magnitude prediction error!

**Root cause**: When bias is large relative to the input range, the neuron saturates and outputs
nearly-constant values regardless of input:

| Activation | Bias | Input Range | Output Range | Status |
|------------|------|-------------|--------------|--------|
| SOFTSIGN | 5.0 | [-1, 1] | [0.78, 0.85] | **Saturated** ❌ |
| TANH | 10.0 | [-1, 1] | [0.9999, 1.0] | **Saturated** ❌ |
| TANH | 1.0 | [-1, 1] | [-0.76, 0.96] | Normal ✓ |

A constant-output neuron **cannot** reduce error correlation - it just adds a fixed offset.
The prediction model incorrectly assumes output varies with input.

**The fix**: Before returning a candidate, the library now checks if:
1. The INPUT activations have variance (source data varies)
2. The OUTPUT would have variance (neuron not saturated)

If input varies but output doesn't, the candidate is rejected. This is implemented via
`has_sufficient_output_variance()` with threshold `MIN_NEURON_OUTPUT_STD_DEV = 0.01`.

**Note**: If the input itself is constant (all samples have same source activation), the
candidate is NOT rejected - constant output is expected and predictions remain valid.

**Tests added**:
- `test_saturation_detection_rejects_constant_output`: SOFTSIGN with bias=5 rejected
- `test_saturation_detection_tanh_large_bias`: TANH with bias=10 rejected
- `test_saturation_detection_allows_constant_input`: Constant input not incorrectly rejected

## [v0.1.162]

### Fixed

#### Removal candidate expected error reduction fix (Issue #117)

**CRITICAL BUG FIX**: The `expectedErrorReduction` for removal candidates was
completely wrong. It was showing the **neuron's error** instead of the **creature's expected
error change** from removing the neuron.

| Field | Old (WRONG) | New (CORRECT) |
|-------|-------------|---------------|
| `expectedErrorReduction` | Neuron's `totalError` | `activationWeightedImpact` |
| Example value | 27% (neuron's error) | 0.000001% (actual impact) |
| Actual error change | ~0% | ~0% |

**The bug**: TypeScript was using `totalError` (the neuron's average error from recorded
samples) as the expected error reduction for the creature. This led to predictions like
"removing this neuron will reduce error by 27%" when the actual reduction was ~0%.

**The fix**: A new `expectedErrorReduction` field is now provided that reflects the
**actual expected creature-level error change**. For removal candidates (low-impact neurons),
this is approximately equal to `activationWeightedImpact` - the actual contribution the
neuron makes to the output.

**Why this makes sense**: Removal candidates have low `activationWeightedImpact` by definition
(that's why they're candidates for removal). Removing them changes the creature's error by
approximately this small amount. The neuron's own error (`totalError`) is irrelevant because
the neuron doesn't significantly affect the output.

**Removal candidate JSON response** now includes:
- `expectedErrorReduction`: The creature-level expected error change (based on impact)
- `totalError`: The neuron's error (for reference, but NOT for prediction)

**TypeScript should use `expectedErrorReduction`** for predictions:
```typescript
// CORRECT: Use the impact-based prediction
const expectedChange = candidate.expectedErrorReduction;

// WRONG: Don't use neuron error as the prediction!
// const expectedChange = candidate.totalError;  // BUG!
```

## [v0.1.145]

### Fixed

#### Impact calculation fix: Absolute not normalised

**CRITICAL BUG FIX**: The impact calculation for removal candidates was massively
underestimating actual impact by up to **145 billion times**.

| Calculated Impact | Actual Error Increase | Underestimation |
|-------------------|----------------------|-----------------|
| 1.39e-12          | 20% (0.2)            | 145 billion x   |
| 4.94e-13          | 0.0054%              | 100 million x   |
| 1.87e-10          | 0.44%                | 24 million x    |

**Root cause**: The old formula normalised by total incoming weights:
```
impact = |weight| / total_inbound × child_impact  // WRONG
```

This answered "what fraction of blame?" not "what happens when removed?"

**The fix**: Use absolute impact (weight × downstream):
```
impact = |weight| × child_impact  // CORRECT
```

**BUG INTRODUCED**: `COST_OF_GROWTH` was incorrectly changed from `1e-7` to `0.01` in this version
based on a false assumption. **NEAT-AI has always used `1e-7`** (never 0.01). This caused 418
false removal candidates in production. Reverted in v0.2.3 (Issue #132).

**Tests added**: `tests/impact_calculation_production.rs` captures the production failure
patterns and verifies the fix prevents regression.

## [v0.1.142]

### Fixed

#### Root cause identified: Sample representativeness

**CRITICAL FINDING**: Synthetic integration tests (`tests/prediction_validation.rs`) have
identified the root cause of prediction inversion in production:

**The sampled data may not be representative of the full training data.**

When TypeScript samples 7.5% of training data for discovery:
1. Rust analyses this sampled subset and finds candidates that improve the samples
2. TypeScript evaluates candidates on 100% of training data
3. If the sample has different error patterns than the full data, predictions invert

| Data Set | Pattern | Candidate Effect |
|----------|---------|------------------|
| Sampled (7.5%) | Positive correlation | +100% improvement |
| Full (100%) | Different/opposite | -261% (worse!) |

**Test demonstrating this**: `test_sample_vs_full_evaluation_mismatch` in
`tests/prediction_validation.rs` creates a scenario where sampled data has one pattern
but full data has the opposite, causing prediction inversion.

**Other tests verify the formula is correct**:
- `test_perfect_correlation_positive_weight_helps`: 99.99% match between prediction and simulation
- `test_split_error_production_like_scenario`: Direction correct even with 50/50 split errors
- `test_hard_tanh_saturation_aware_prediction`: Saturation-aware model works correctly

**Potential solutions** (future work):
1. **Stratified sampling**: Ensure sampled data is representative of error distribution
2. **Cross-validation**: Evaluate candidates on a held-out validation set before returning
3. **Increase sample rate**: Use more data for more representative samples
4. **Confidence bounds**: Only return candidates with high-confidence predictions

## [v0.1.141]

### Fixed

#### GPU shader activation function fix

**CRITICAL BUG FIX**: New activation functions added in v0.1.139 were not working correctly
on GPU. The GPU shaders (`activation.wgsl` and `bias.wgsl`) only handled activation IDs 0-10,
but the new activations were assigned IDs 11-18.

| Activation | GPU ID | Status Before | Status After |
|------------|--------|---------------|--------------|
| LeakyReLU | 11 | ❌ IDENTITY fallback | ✅ Correct |
| Mish | 12 | ❌ IDENTITY fallback | ✅ Correct |
| Swish | 13 | ❌ IDENTITY fallback | ✅ Correct |
| HARD_TANH | 14 | ❌ IDENTITY fallback | ✅ Correct |
| SOFTSIGN | 15 | ❌ IDENTITY fallback | ✅ Correct |
| BENT_IDENTITY | 16 | ❌ IDENTITY fallback | ✅ Correct |
| ArcTan | 17 | ❌ IDENTITY fallback | ✅ Correct |
| ReLU6 | 18 | ❌ IDENTITY fallback | ✅ Correct |

**The bug**: GPU shaders used `default: { return x; }` for unknown IDs, silently returning
IDENTITY results instead of the correct activation function. This caused incorrect weight
calculations without any error.

**Example of the bug (LeakyReLU)**:
- Pre-activation = -1.0
- LeakyReLU(-1.0) = -0.01 (correct)
- IDENTITY(-1.0) = -1.0 (100x wrong!)

Weight calculations based on these wrong outputs would be completely incorrect, producing
candidates that fail when applied in production.

**The fix**: Added all 8 new activation functions to both `activation.wgsl` and `bias.wgsl`
shaders with correct implementations:
- LeakyReLU: `x if x >= 0, else 0.01 * x`
- Mish: `x * tanh(softplus(x))`
- Swish: `x * sigmoid(x)`
- HARD_TANH: `clamp(x, -1, 1)`
- SOFTSIGN: `x / (1 + |x|)`
- BENT_IDENTITY: `(sqrt(x² + 1) - 1) / 2 + x`
- ArcTan: `atan(x)`
- ReLU6: `clamp(x, 0, 6)`

**Test added**: `tests/gpu_activation_shaders.rs` verifies GPU shader correctness for all
new activation functions.

## [v0.1.140]

### Fixed

#### Prediction validation

**INVESTIGATION**: With ~100k samples, predictions should be accurate. Production data shows
predictions are inverted (~84% in wrong direction).

**Finding from synthetic tests**: The prediction formula is **mathematically correct**!
All 6 synthetic tests pass with predictions matching manual simulation to within 0.01%.
This means the issue is in **sample collection or interpretation**, not the formula.

| Test Scenario | Predicted | Manual | Match? |
|---------------|-----------|--------|--------|
| Linear region | 75.00% | 75.00% | ✓ |
| Near saturation | 100.00% | 100.00% | ✓ |
| Negative error | 75.00% | 75.00% | ✓ |
| Mixed errors | 7.10% | 7.10% | ✓ |
| TypeScript simulation | 66.38% | 66.38% | ✓ |

**Key insight**: Large bias values (e.g., 10) combined with large incoming weights cause saturation,
making the neuron behave like a constant. This appears "optimal" on small samples but fails to generalise.
See `tests/fixed_vs_optimised_params.rs` for investigation tests.

## [v0.1.139]

### Added

#### Expanded activation functions and discrete weight fix

**MAJOR FEATURE**: Added 8 new activation functions based on analysis of successful
discoveries. Many successful neurons evolved TO activations we weren't trying!

| New Activation | Evidence |
|----------------|----------|
| **LeakyReLU** | 4 successful discoveries evolved ReLU → LeakyReLU! |
| **Mish** | 2 successful discoveries evolved TO Mish (from ELU, Softplus) |
| **Swish** | 1 successful discovery evolved ReLU → Swish |
| **HARD_TANH** | 1 successful discovery evolved CLIPPED → HARD_TANH |
| **SOFTSIGN** | Successful discovery neuron with SOFTSIGN |
| **BENT_IDENTITY** | 1 successful discovery evolved LeakyReLU → BENT_IDENTITY |
| **ArcTan** | Similar to SOFTSIGN, bounded output |
| **ReLU6** | Capped ReLU, useful for bounded outputs |

Total activations now: **19** (was 11).

**Philosophy change**: The goal is finding MORE successful candidates, not filtering
out failures. Failed candidates are excluded after evaluation anyway. "Kiss more frogs
to find more princes."

### Fixed

#### Discrete evaluation generating huge outgoing weights

The `evaluate_discrete_candidate` function (for STEP/BIPOLAR targets with IDENTITY neurons)
was generating outgoing weights up to ±50, far exceeding `MAX_OUTGOING_WEIGHT` (0.1).

| Before | After |
|--------|-------|
| OUTGOING_SCALES: [0.1..50.0] | OUTGOING_SCALES: [0.01..0.1] |

**Production evidence**: 455 out of 793 large-weight failed candidates were IDENTITY neurons
from this code path. None produced real improvements.

## [v0.1.138]

### Fixed

#### Tighter outgoing weight clamp

**CRITICAL IMPROVEMENT**: Analysis of 2030 failed add-neuron candidates vs ~22 successful
discoveries revealed that outgoing weights were being computed far too large.

**Successful discoveries (survived evolution):**
- |outgoing_weight|: 0.00002 to 0.03 (all < 0.05)
- incoming/outgoing ratio: 71x to 104,000x
- Example: incoming=100, outgoing=-0.00096 (ratio 104,000x)

**Failed discoveries:**
- 36% had |outgoing_weight| > 0.05 (up to 50!)
- Many had ratio < 10x (even 1:1)
- Previous clamp: [-10.0, 10.0] was far too loose

**The fix**: Three-part improvement to weight calculation:

1. **Tighter outgoing weight clamp**: Changed from `[-10.0, 10.0]` to `[-0.1, 0.1]`.
   New neurons should contribute a SMALL correction, not dominate the network.

2. **Weight ratio validation**: For add-neuron candidates where `incoming_weight > 1.0`,
   we now validate that `incoming/outgoing >= 50`. Candidates with nearly equal incoming
   and outgoing weights are rejected as unreliable predictions.

3. **Shared weight function**: Created `calculate_optimal_outgoing_weight()` to ensure
   consistent weight calculation across add-synapse and add-neuron analysis (DRY).

**Also fixed**: Split-error ReLU evaluation was computing bias from the error subset only,
which could produce large positive biases that made the ReLU fire for ALL samples (defeating
the purpose of split-error). Now uses bias=0 for split-error ReLU candidates.

**Expected impact**: ~36% of failed candidates (with |outgoing_weight| > 0.05) will now
produce tighter, more accurate predictions. The remaining candidates may still fail due
to other factors (sample overfitting, bias-weight interaction, activation saturation)
which can be addressed in follow-up improvements.

## [v0.1.136]

### Fixed

#### Split-error fallback candidate fix

**BUG FIX #1**: The split-error evaluation introduced in v0.1.135 had a threshold bug that
broke the fallback mechanism. Candidates with small positive improvements (below threshold)
were silently dropped instead of being returned as fallbacks.

**Root cause**: `evaluate_activation_for_subset` initialised `best_net_improvement` to
`threshold`, meaning candidates with `0 < improvement <= threshold` failed the comparison
check and were never returned.

**Fix**: Changed `best_net_improvement` initialisation from `threshold` to `0.0`.

**BUG FIX #2**: When split-error evaluation was attempted (both positive and negative error
subsets had enough samples) but found NO candidates with positive net improvement, the code
incorrectly fell back to all-samples evaluation. This produced unreliable small-improvement
predictions (~0.05%) that consistently failed in production.

**Root cause**: When errors are truly split ~50/50 AND source activations don't correlate
with error sign, there's NO good weight. Any weight helps one group but hurts the other
equally. Split-error correctly rejects these candidates. But all-samples would then compute
a weak weight (due to error cancellation) and return small positive predictions that were
within the model's error margin - essentially noise.

**Fix**: Track whether split-error evaluation was properly attempted. If both subsets had
enough samples but NEITHER produced candidates, return None instead of falling through
to all-samples. The all-samples fallback is now ONLY used when errors aren't clearly split
(e.g., all positive, all negative, or one subset too small).

**Test added**: `tests/split_error_fallback_candidates.rs` verifies the fix.

## [v0.1.135]

### Fixed

#### Split-error evaluation for all activations

**BUG FIX**: When target neuron errors are split ~50/50 between positive and negative,
the standard linear model would predict small positive improvements that were actually
negative in practice. This caused systematic prediction failures for add-neuron candidates.

**Root cause**: Computing optimal weight from ALL samples averages out when errors
are balanced. The model predicts +0.08% but actual result is -0.08% because helping
one group hurts the other equally.

**Fix**: Extended ReLU's split-error handling to ALL activations:
1. Split samples by error sign (positive vs negative)
2. For EACH subset, compute optimal weight from that subset
3. Evaluate NET improvement across ALL samples
4. Only return candidates where net improvement > 0

**Result**: The improvement metric is now the TRUE net improvement across
all samples, not just a subset prediction. Candidates that would hurt one group more
than they help the other are filtered out automatically.

**Test added**: `tests/split_error_all_activations.rs` verifies the fix.

## [v0.1.134]

### Changed

#### Simplified candidate filtering

**SIMPLIFICATION**: Removed all arbitrary percentage thresholds. The creature's score
is the **only** measure that matters.

**Rust's job**:
1. Find ALL candidates with positive expected error reduction
2. Apply impact discounting (convert to creature-level predictions)
3. Sort by expected improvement (best first)
4. If `analysisDeadlineMs` is set, randomise within the top-K and preserve that diversified
   ordering through truncation to avoid category starvation across repeated runs
5. Return candidates to TypeScript

**TypeScript's job**:
1. Select top N candidates based on available CPUs
2. Apply each candidate mutation and measure ACTUAL score change
3. Keep candidates that improve the score
4. Return improved creatures to the population

**Evolution's job**:
- Successful mutations compete in the population
- Unsuccessful mutations get bred out naturally
- No manual filtering needed

**Why no thresholds?** Previous versions had arbitrary thresholds (2%, 0.1%) that
filtered candidates before TypeScript could evaluate them. This was wrong:
- The cost of growth is ~1e-7 per neuron, ~1e-8 per synapse
- Any measurable improvement easily exceeds this cost
- The old 2% threshold was **10,000x too aggressive**
- Candidates that looked "too small" in Rust could still improve the actual score

**Current behaviour**: Return everything positive. Let TypeScript measure. Let evolution decide.

## [v0.1.132]

### Changed

#### Squash-aware impact calculation

**ENHANCEMENT**: The impact calculation is now **squash-aware**. Different squash
functions use different impact formulas to avoid underestimating impact.

| Squash Category | Functions | Impact Formula | Rationale |
|-----------------|-----------|----------------|-----------|
| **Linear** | IDENTITY, TANH, LOGISTIC, etc. | `\|w\| / total_inbound × child` | Sum of weighted inputs |
| **Threshold** | STEP, BIPOLAR | `child_impact` (full, not normalised) | Any synapse can flip output |
| **Selection** | MINIMUM, MAXIMUM, IF | `P(winning) × child_impact` | Actual win probability from activations |

**Why this matters**:

- **STEP/BIPOLAR**: A tiny weight (1e-8) feeding into a STEP neuron could flip the
  output from 0→1 if the neuron is near its threshold. The old sum-based formula
  would calculate impact ≈ 0, but the actual effect could be 1.0!

- **MINIMUM/MAXIMUM (v0.1.143+)**: Uses recorded activation data to compute actual
  selection probabilities. If a synapse wins MINIMUM 90% of the time in the recorded
  samples, it gets 90% of the impact. This is more accurate than the previous 1/N
  equal probability fallback.

- **IF neurons (v0.1.143+)**: Synapse types (`"condition"`, `"positive"`, `"negative"`)
  are now used to compute accurate impact. Condition synapses always contribute (100%),
  while positive/negative synapses share impact based on how often each branch is taken.

For detailed explanation with diagrams, see [Impact Calculation](docs/IMPACT_CALCULATION.md).

**Example**: Output has 107 incoming synapses with total |weight| = 343.
A neuron with weight 3.0 to output contributes: `3.0 / 343 ≈ 0.9%` of output.

This correctly captures that removing a neuron with many competing inputs
has a small effect on the downstream signal.

**Removal candidate JSON response** now includes:
- `incomingSynapses` / `outgoingSynapses`: synapse counts used in calculation
- `removalSavings`: the raw savings value from NEAT-AI formula
- Candidates sorted by activation_weighted_impact ascending (lowest first = safest to remove)

The `calculate_removal_savings(incoming, outgoing, growth_cost)` function is
available for use in other analyses and is tested against the NEAT-AI formula.

If verbose logging is enabled (`NEAT_AI_DISCOVERY_VERBOSE=1`), you'll see
messages like:

```
[NEAT-AI-Discovery][verbose] Using threshold-crossing model for 2 STEP/BIPOLAR neurons: [...]
```

## [v0.1.127]

### Fixed

#### Dynamic removal threshold based on synapse counts

**BUG FIX**: After the v0.1.126 impact calculation fix, ZERO removal candidates
were being found. The fix implements a dynamic threshold based on NEAT-AI's
actual Score.ts complexity formula.

**The issue**: The removal candidate threshold was a static value that didn't
account for the complexity savings from removing the neuron's synapses.

**NEAT-AI's Score.ts formula** (the authoritative source):
```typescript
const complexityPenalty = hiddenNeuronCount * growthCost +
    creature.synapses.length * growthCost / 10 +
    penalty * growthCost / 100;
```

**So removing a neuron with N incoming and M outgoing synapses saves:**
```
savings = growthCost × (1 + (N + M) / 10)
```

**The fix**: ALL neurons with `activation_weighted_impact < costOfGrowth` are
returned as removal candidates, sorted by impact ascending.

## [v0.1.126]

### Fixed

#### Impact calculation fix

**CRITICAL BUG FIX**: The neuron impact calculation was severely underestimating
impact by normalising weights. This caused ~75% of "low-impact" removal
candidates to actually INCREASE error when removed.

**The bug**: Impact was computed as `weight / total_inbound × child_impact` which
gave the "fraction of downstream's input from this neuron" instead of the actual
contribution to output.

**Example of the bug**:
- Neuron A → Target (weight 0.001), Other → Target (weight 100)
- Old (normalised): impact = 0.001 / 100.001 × 1.0 ≈ **1e-5**
- New (absolute): impact = 0.001 × 1.0 = **0.001**

The normalised formula underestimated by **100x** in this case! For deep networks
with many competing inputs at each layer, the underestimation compounds to
**1000x or more**.

**Production evidence**: Neurons with calculated impact 1e-10 to 1e-17 caused
score deltas of 1e-5 to 1e-2 when removed - off by 5-15 orders of magnitude.

**The fix**: Impact now uses absolute weight products along paths to outputs:
```
impact = weight × downstream_impact
```

This matches the actual contribution: `activation × weight × downstream_impact`.

## [v0.1.123]

### Added

#### Hidden neuron impact discounting

Hidden neurons are valid targets for add-neuron and add-synapse analysis.
Predictions are discounted by the neuron's impact score to give creature-level
expected improvements.

**How it works**:
- **Output neurons**: Impact = 1.0 (direct contribution to score). No discount applied.
- **Hidden neurons**: Impact = normalised path weight to outputs. Predictions are
  discounted by impact factor.

For a hidden neuron with impact 0.5:
- Raw predicted improvement: 10%
- Discounted improvement: 10% × 0.5 = 5%

This discounting ensures hidden neuron predictions reflect their actual contribution
to the creature's score based on their position in the network topology.

## [v0.1.120]

### Fixed

#### ACTIVATION domain consistency

**CRITICAL BUG FIX**: When using target activation function simulation (to handle
saturation in HARD_TANH, TANH, etc.), the improvement calculation was comparing
errors from **different domains**:

- **Baseline error**: VALUE domain (`avg_error²`)
- **New error**: ACTIVATION domain (`(expected - new_output)²`)

Near saturation, VALUE domain errors are much larger than ACTIVATION domain errors
(because the activation function compresses them). This caused **massive
overprediction** of improvements.

**Example of the bug:**

| Value | Computation | Result |
|-------|-------------|--------|
| target_value | Pre-activation input | 0.9 |
| avg_error | VALUE domain error | 0.3 |
| desired_value | target_value + avg_error | 1.2 |
| expected | HARD_TANH(1.2) | 1.0 (saturated) |
| target_activation | Current output | 0.9 |
| contribution | Weight × new_neuron_output | 0.05 |
| new_input | target_value + contribution | 0.95 |
| new_output | HARD_TANH(0.95) | 0.95 |

**Buggy calculation (mixed domains):**
- Baseline error² = 0.3² = 0.09 (VALUE domain)
- New error² = (1.0 - 0.95)² = 0.0025 (ACTIVATION domain)
- Improvement = (0.09 - 0.0025) / 0.09 = **97%** ❌

**Correct calculation (consistent ACTIVATION domain):**
- Baseline error² = (1.0 - 0.9)² = 0.01 (ACTIVATION domain)
- New error² = (1.0 - 0.95)² = 0.0025 (ACTIVATION domain)
- Improvement = (0.01 - 0.0025) / 0.01 = **75%** ✓

The fix computes **both baseline and new error** in the same domain (ACTIVATION
when simulating, VALUE for linear approximation). This is verified by
`improvement_calculation_uses_consistent_domains`.

**Fixed functions:**
- `compute_synapse_improvement_and_count`
- `compute_relu_improvement_and_count`
- `compute_activation_improvement_and_count`

## [v0.1.117]

### Fixed

#### VALUE domain error interpretation

**CRITICAL BUG FIX**: The NEAT-AI TypeScript library stores errors in the **VALUE
domain** (pre-activation), not the ACTIVATION domain (post-squash). This affects
how the Rust library interprets and uses error data for improvement predictions.

**TypeScript error calculation (NEAT-AI `Neuron.record()`):**
```typescript
const targetValue = unSquash(desiredActivation);  // Convert desired output to pre-activation
const error = targetValue - currentValue;         // VALUE domain error
```

**Previous (incorrect) Rust interpretation:**
```rust
// WRONG: Treated error as activation domain
let expected = target_activation + avg_error;  // Mixing ACTIVATION + VALUE domains!
```

**Corrected Rust interpretation (v0.1.117):**
```rust
// CORRECT: Error is in VALUE domain, so compute expected via squash
let desired_value = target_value + avg_error;
let expected = squash(desired_value);  // Convert to ACTIVATION domain
```

**Why this matters for saturation:**

| Scenario | Current Value | Error (VALUE) | Old Formula | Correct Formula |
|----------|---------------|---------------|-------------|-----------------|
| Near saturation | 0.8 | 0.3 | `expected = 0.8 + 0.3 = 1.1` | `expected = clamp(1.1) = 1.0` |
| In saturation | 1.5 | -1.0 | `expected = 1.0 + (-1.0) = 0.0` | `expected = clamp(0.5) = 0.5` |

The old formula produced incorrect `expected` values when the target neuron was
near or in saturation, causing predictions to be wildly inaccurate.

**Fixed locations:**
- `compute_net_improvement_new` (HARD_TANH model)
- `compute_activation_improvement_and_count` (all 4 activation paths)
- `compute_synapse_improvement_with_target_squash`
- `count_improved_samples_with_target_squash`

This fix ensures predictions match actual results when candidates are applied,
resolving the "add-neuron candidates always fail" production issue.

## [v0.1.115]

### Fixed

#### Bias-aware weight calculation

For **add-neuron** candidates, the optimal outgoing weight must be computed using
the new neuron's **actual activation pattern** (which includes bias). Previously,
the weight was computed without bias, then a separate bias optimisation was
performed. This caused predictions to fail when bias significantly shifted the
activation threshold.

**Example failure scenario (now fixed):**
- New TANH neuron with `bias=1`
- Without bias: `TANH(x)` fires when x > 0 (~50% of samples)
- With bias: `TANH(x+1)` fires when x > -1 (almost always!)
- The optimal weight for these two patterns is completely different

**The fix**: After finding the optimal bias, the library now **recomputes** the
optimal outgoing weight using the actual activation pattern (with bias). This
ensures predictions match reality.

This is verified by unit tests: `add_neuron_weight_must_include_bias_in_calculation`
and integration test: `test_add_neuron_with_hard_tanh_target_uses_bias_aware_weight`.
