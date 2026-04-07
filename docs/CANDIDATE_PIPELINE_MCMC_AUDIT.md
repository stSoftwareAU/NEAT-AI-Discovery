# Candidate Selection Pipeline: MCMC Applicability Audit

Issue #1017 — Part of #1016

## 1. Executive Summary

After thorough analysis of the candidate selection pipeline, the finding is that
**there is no explicit MCMC implementation**, and **MCMC is not the correct
framework** for this system. The pipeline is a **one-shot optimisation search**
(propose candidates, evaluate, filter, rank) rather than a Markov chain sampling
procedure. The poor synapse candidate success rate (0% in GRQ-sampler cache) is
best addressed through calibration improvements to the existing deterministic
pipeline, not by introducing MCMC machinery.

## 2. Pipeline Stages Mapped to MCMC Concepts

The table below maps each pipeline stage to its nearest MCMC equivalent, then
identifies where the analogy breaks down.

| Pipeline Stage | Code Location | MCMC Analogue | Gap / Divergence |
|----------------|---------------|---------------|------------------|
| **Source enumeration** | `orchestration.rs` — `order_focus_targets()` | Proposal distribution | Deterministic neuron ordering with interleaving (Issue #907), not a stochastic proposal kernel |
| **Weight grid search** | `evaluation.rs:160–170` — 9-variant `weight_candidates` | Parameter proposal | Fixed grid of {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5, -1.0} × optimal weight — not a continuous proposal distribution |
| **GPU evaluation** | `evaluation.rs:78–94` — `positive_count`/`negative_count` | Likelihood computation | Counts improved/worsened samples as a quality signal — analogous to likelihood evaluation |
| **Hold-out validation** | `holdout_validation.rs` — `split_samples_holdout()` | — (no MCMC analogue) | Cross-validation to combat overfitting from the 9-variant search; a statistical regularisation technique, not an MCMC concept |
| **Accept/reject** | `evaluation.rs:258–289` — threshold + `MIN_IMPROVED_RATIO` | Acceptance criterion | **Purely deterministic**: accept if `improvement > 0` AND `improved_ratio >= 0.6`. No probabilistic acceptance (no Metropolis-Hastings α). Worse candidates are always rejected |
| **Pessimism discount** | `discounting.rs` — `apply_synapse_pessimism_discount()` | — (no MCMC analogue) | Concave power-curve scaling of predictions to correct systematic overestimation. A calibration correction, not a transition probability |
| **Prediction calibration** | `candidate_scoring.rs:557` — `SYNAPSE_PREDICTION_CALIBRATION = 0.001` | — (no MCMC analogue) | Per-type multiplicative correction for 100–10,000× overestimation. Calibration, not sampling |
| **Impact discounting** | `post_processing.rs` — `apply_impact_to_helpful()` | — (no MCMC analogue) | Structural network-topology weighting (distance to outputs) |
| **Diversification** | `deadline.rs:342` — `shuffle_within_top_k(DIVERSIFY_TOP_K=64)` | Chain mixing / exploration | Shuffles top-64 candidates for diversity across runs — closest to MCMC exploration, but applied post-hoc to a sorted list rather than as part of a chain's transition kernel |
| **Ranking & truncation** | `post_processing.rs:386–453` | — (no MCMC analogue) | Sort by `expected_creature_score_gain`, truncate to `max_candidates`. Pure optimisation ranking |
| **Convergence diagnostics** | None | R-hat, ESS, trace plots | **Completely absent**. No chain state is maintained across invocations. Each analysis call is independent |

## 3. Why This Is Not MCMC

MCMC (Markov chain Monte Carlo) requires four properties that this system lacks:

### 3.1 No Markov Chain State

Each call to `analyze_synapses_with_cache_impl()` is stateless — there is no
chain state carried between invocations. MCMC requires a current state `x_t` that
is updated to `x_{t+1}` via a transition kernel. The pipeline evaluates all
candidates independently and returns the best ones.

### 3.2 No Probabilistic Acceptance

The accept/reject logic in `evaluation.rs:258–289` is deterministic:

```
if neuron_error_improvement <= 0.0 → reject
if improved_ratio < MIN_IMPROVED_RATIO (0.6) → reject
if neuron_error_improvement <= threshold → reject (but still tracked)
otherwise → accept
```

In Metropolis-Hastings MCMC, the acceptance probability is:

```
α = min(1, π(x') × q(x|x') / (π(x) × q(x'|x)))
```

This allows accepting *worse* states with some probability, enabling exploration
of the posterior. The current pipeline never accepts worse candidates — it is a
greedy filter.

### 3.3 No Detailed Balance

MCMC requires the transition kernel to satisfy detailed balance with respect to
the target distribution. The pipeline has no target distribution to sample from
— it is searching for the single best candidate, not sampling from a posterior
over candidates.

### 3.4 No Ergodicity / Convergence

MCMC chains must be ergodic (able to reach any state from any other state). The
pipeline's deterministic source ordering and fixed weight grid cannot explore
the full candidate space. The `shuffle_within_top_k` provides some exploration
diversity but only within the top-64 of an already-filtered list.

## 4. Root Cause Analysis: Why 0% Synapse Success Rate

The 0% success rate (0/31 in GRQ-sampler cache) and 100–10,000× prediction
overestimation are **not caused by the absence of MCMC**. They stem from:

### 4.1 Prediction-to-Reality Gap

The improvement calculation in `improvement.rs` measures the fraction of a
single target neuron's squared error explained by sampled data:

```
improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq
```

This neuron-level metric does not translate to creature-level score gain because:
- It measures improvement on **sampled** data, not the full evaluation set
- It measures improvement at **one neuron**, not the whole network
- Non-linear activation functions mean local improvements do not compose linearly

The pipeline already addresses this with:
- Error fraction scaling (`post_processing.rs:80–90`)
- Pessimism discounting (floor=0.05, exponent=0.85 for synapses)
- Prediction calibration (×0.001 for synapses)

### 4.2 Multi-Weight Search Overfitting

The 9-variant weight grid search (`evaluation.rs:160–170`) selects the
best-looking weight from 9 options, creating selection bias. Hold-out validation
(`holdout_validation.rs`) was added to combat this, but only activates when
sample count >= 20 (`HOLDOUT_MIN_SAMPLE_COUNT`).

### 4.3 Small Sample Size

With only 31 synapse candidates in the cache, the 0% rate has a wide confidence
interval. A Beta(1, 32) posterior gives a 95% credible interval of [0%, 12.7%]
— the true success rate could be non-trivial.

## 5. MCMC Techniques Evaluated

### 5.1 Metropolis-Hastings Acceptance

**Would it help?** No. Probabilistic acceptance is designed for **sampling from
a posterior distribution**, not for **finding the best candidate**. Accepting
worse candidates with some probability would:
- Return lower-quality candidates to the caller (NEAT-AI)
- Not improve the prediction calibration problem
- Add complexity without addressing the root cause

The pipeline's goal is to return the K best candidates for ablation testing by
NEAT-AI. Ablation testing already provides the "exploration" that MH acceptance
would give — NEAT-AI tests each candidate and keeps only those that actually
improve the creature.

### 5.2 Simulated Annealing

**Would it help?** Partially, but it is the wrong abstraction. Simulated
annealing is useful when searching a single continuous parameter space and the
objective has many local optima. The pipeline's "parameter space" is discrete
(which source neuron to connect to which target neuron) with a continuous
sub-problem (what weight to use). The weight sub-problem is already addressed
by the 9-variant grid search.

Simulated annealing could theoretically be applied to the source ordering to
explore more diverse candidates early on, but `shuffle_within_top_k` already
provides this effect without the complexity of a cooling schedule.

### 5.3 Adaptive Proposals

**Would it help?** This is the most promising MCMC-inspired technique, and
the pipeline **already implements a version of it**:
- `candidate_cache.rs` tracks success/failure outcomes per candidate type
- `module_weights.rs` adjusts per-module weights based on historical success
- `scale_outcomes.rs` tracks per-scale success rates for weight variants
- Source-type boosts (`INPUT_SOURCE_BOOST = 1.5`) are calibrated from cache data
- Activation-function boosts are Bayesian-smoothed from GRQ-sampler evidence

These are functionally equivalent to adaptive proposal distributions in MCMC.

### 5.4 Temperature Scheduling

**Would it help?** No. Temperature scheduling modulates the exploration-vs-
exploitation trade-off over the course of a chain. Since there is no chain
(each invocation is independent), there is nothing to anneal. The diversity
mechanism (`DIVERSIFY_TOP_K = 64`) provides run-to-run exploration without
requiring a temperature parameter.

## 6. Recommendation

**Do not introduce MCMC machinery.** The pipeline is correctly structured as a
one-shot optimisation search with post-hoc diversity. The correct framework is
**calibrated prediction with ablation testing**, not **posterior sampling**.

### 6.1 What Is Already Working

The pipeline has progressively improved its calibration through:
1. Hold-out validation (Issue #893) — combats multi-weight overfitting
2. Per-type pessimism discounting (Issues #506, #789, #791) — corrects optimistic
   predictions using concave power curves
3. Prediction calibration factors (Issue #891) — per-type multiplicative correction
4. Error fraction scaling (Issue #730) — converts neuron-level to creature-level
5. Adaptive boosts from cache evidence (Issues #465, #468, #887) — learned priors

### 6.2 Where to Focus Instead

If synapse candidate success rates need improvement, the following calibration
approaches are more promising than MCMC:

1. **Expand the evaluation dataset**: 31 synapse candidates is too small for
   reliable success rate estimation. More data will clarify whether the 0% rate
   is a true signal or sampling noise.

2. **Tighten hold-out validation**: Lower `HOLDOUT_MIN_SAMPLE_COUNT` from 20
   if possible, or increase `HOLDOUT_VALIDATION_FRACTION` from 0.3 to ensure
   more validation data.

3. **Refine pessimism parameters**: The synapse-specific parameters
   (floor=0.05, exponent=0.85) were set based on the 0% success rate. As more
   data accumulates, these can be recalibrated.

4. **Weight search refinement**: The fixed 9-variant grid could be replaced with
   a finer-grained search around the optimal weight, or an adaptive grid based
   on historical successful weights.

## 7. Key Files Reviewed

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/analysis/synapse/target_analysis/evaluation.rs` | Core accept/reject logic, weight grid search | Central to the pipeline |
| `src/analysis/synapse/orchestration.rs` | Pipeline orchestration, parallel processing | Confirms stateless per-invocation design |
| `src/analysis/diagnostics/rejection.rs` | Rejection tracking and reporting | Confirms deterministic rejection reasons |
| `src/analysis/constants/candidate_scoring.rs` | Scoring constants, boosts, calibration | Documents calibration approach |
| `src/analysis/neuron/post_processing.rs` | Neuron candidate post-processing | Shows pessimism + calibration pipeline |
| `src/analysis/synapse/holdout_validation.rs` | Hold-out validation for overfitting | Key anti-overfitting mechanism |
| `src/analysis/synapse/scoring/discounting.rs` | Pessimism discount implementation | Core calibration mechanism |
| `src/analysis/synapse/scoring/improvement.rs` | Improvement calculation | Source of prediction overestimation |
| `src/analysis/synapse/post_processing.rs` | Full post-processing pipeline | Impact, boosts, calibration, diversification |
| `src/analysis/constants/detection_thresholds.rs` | `MIN_IMPROVED_RATIO` (0.6) | Deterministic acceptance threshold |
