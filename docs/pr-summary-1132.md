# Issue #1132 — Adapt discovery strategy when recent creature-level success rate is near zero

Closes #1132

## Summary

Adds a creature-level rolling success-rate signal to `analyze_all` so the pipeline can detect when a specific creature has been unable to discover any useful candidate for several passes in a row, and biases the candidate mix away from high-failure-rate structural modules while the drought persists.

## What changed

### New module `src/analysis/discovery_mode.rs`
- `DiscoveryOutcomeLog { outcomes: Vec<bool> }` — caller-supplied chronological log of per-pass outcomes. Library is stateless; the host persists the log across runs.
- Rolling window of the last `ROLLING_WINDOW = 10` outcomes.
- `decide_mode(&log, threshold, max_epochs) -> DiscoveryMode` returns `Conservative` iff the rolling success rate is **strictly below** `threshold` (default `0.2`) and the trailing consecutive-failure streak has not exceeded `max_epochs` (default `20`). After that cooldown the library reverts to `Normal` because the bias clearly wasn't helping.
- `biased_tracker_for_conservative_mode(&ModuleOutcomeTracker)` boosts the success-rate signal of low-risk modules (activation substitution, bias drift, weight adjustment, …) by `1.5×` and attenuates the signal of high-risk structural modules (coordinated-structural, add-neurons, …) by the inverse factor. The biased tracker feeds every downstream allocator (`apply_module_boost_to_candidates`, ensemble scoring, budget allocator).
- `coordinated_gain_multiplier_for_mode(mode, multiplier)` returns `1.0` in `Normal` mode and the configured (≥ 1.0) multiplier in `Conservative` mode. A new `apply_coordinated_gain_floor_with_multiplier` in `candidate_aggregation.rs` applies this to the post-discount noise floor, so only obviously-promising structural candidates survive when the rolling rate is low.
- 11 unit tests covering empty log, windowed rate, boundary at `0.2`, cooldown entry/exit, serialisation, biasing idempotence.

### Orchestration integration (`src/analysis/orchestration.rs`)
- Computes `discovery_mode` + `rolling_success_rate` from `input.discovery_outcome_log` before the tracker is consumed.
- In `Conservative` mode:
  - the tracker passed to `module_weights::apply_module_boost_to_candidates` and friends is the biased copy;
  - `apply_coordinated_gain_floor_with_multiplier` tightens the post-discount floor by `conservative_gain_multiplier()` (default `10×`).
- Populates the new metadata fields on both `synapse_result.metadata` and `neuron_result.metadata` immediately before the final `Ok(AnalyzeAllResult …)` return, so callers can observe the mode regardless of early-return paths.

### FFI surface
- `AnalyzeParallelInput` and `AnalyzeAllInput` gain `discovery_outcome_log: Option<DiscoveryOutcomeLog>` (camelCase, serde `default`).
- `SynapseAnalysisMetadataJson` and `NeuronAnalysisMetadataJson` now expose `discoveryMode: "normal" | "conservative"` and `rollingSuccessRate: f32`.

### Config (`src/config/user_facing.rs`)
Three new env-var accessors, documented in the module-level summary table:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` | f32 | `0.2` | Rolling success-rate threshold below which conservative mode engages |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` | u32 | `20` | Max consecutive failed passes before abandoning conservative mode |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` | f32 | `10.0` | Multiplier applied to the coordinated-gain floor while conservative |

`conservative_gain_multiplier` clamps below `1.0` to `1.0` so the floor can never be relaxed.

### Integration test (`tests/analysis/issue_1132_conservative_mode.rs`)
- `ten_failures_triggers_conservative_mode_in_metadata` — a log of ten consecutive failures causes both the synapse and neuron result metadata to report `Conservative` with rolling rate `0.0`.
- `no_outcome_log_keeps_metadata_in_normal_mode` — absence of a log leaves metadata at the default `Normal` / `1.0`.
- `mode_decision_transitions_at_documented_boundaries` — non-GPU cross-check of `decide_mode` at the `0.2` threshold and the 20-epoch cooldown.
- `conservative_bias_raises_low_risk_above_high_risk` — direct integration check of the weight shift.

## Design notes

- **Stateless library**: the host persists the rolling log; the library recomputes the mode every call. Matches the pattern used by `module_outcome_tracker` and `failure_cache`.
- **Exit semantics**: conservative mode exits once the rolling window naturally climbs above threshold, or once the cooldown elapses. A single success does **not** force-exit — the rolling rate governs, which avoids pinball transitions under noisy signals. The `cooldown_exit_after_max_epochs` test documents the hard fallback.
- **Symmetric factor**: `CONSERVATIVE_HIGH_RISK_PENALTY = 1 / CONSERVATIVE_LOW_RISK_BOOST = 1 / 1.5`. Keeps reasoning about the net effect on the allocator simple.

## Verification

- `cargo test --lib discovery_mode` — 11/11 ✅
- `cargo test --test analysis issue_1132` — 4/4 ✅
- `./quality.sh < /dev/null` — ✅ (full suite: fmt, clippy, test, docs, release build)

## Pre-PR security self-check

- [x] **Input validation**: `DiscoveryOutcomeLog.outcomes` is a plain `Vec<bool>` — no parsing or injection surface. `low_success_rate_threshold`/`conservative_mode_max_epochs`/`conservative_gain_multiplier` filter to finite numbers in sensible ranges and fall back to defaults.
- [x] **Secrets**: no `.config*.json` or credential files touched.
- [x] **Injection surface**: no new SQL/shell/FS calls.
- [x] **Output encoding**: values are serialised through existing serde path with the same camelCase convention as surrounding fields.
- [x] **Authentication/authorisation**: n/a — pure library change.
- [x] **Error handling**: no panics added; all casts are clamped.
- [x] **Dependencies**: none added.
