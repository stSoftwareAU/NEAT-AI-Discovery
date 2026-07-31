# ⚙️ Configuration Reference

This is the **single, authoritative** reference for every runtime environment
variable that tunes `neat_ai_discovery`. The code-side origin is
[`src/config/`](../src/config/).

> **Single source of truth.** Do not copy this table into `README.md`,
> `AGENTS.md`, or anywhere else. Both of those documents link here. When you add
> or change an environment variable, update **this file only** — see
> [CONTRIBUTING.md](../CONTRIBUTING.md).

All discovery variables use the `NEAT_AI_DISCOVERY_` prefix. Boolean toggles
accept `1`/`true`/`yes` to enable unless noted otherwise; unset means the
documented default applies. Invalid or out-of-range values fall back to the
default rather than aborting.

## Library & logging

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_LIB_PATH` | `~/.cargo/lib/` | Path to the compiled library. |
| `RUST_LOG` | `warn` | Structured log level via `tracing` (e.g. `neat_ai_discovery=info`). |
| `NEAT_AI_DISCOVERY_VERBOSE` | off | Enable verbose logging (`1` to enable). |

## GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` | auto | Override GPU batch size (64–4096). |
| `NEAT_AI_DISCOVERY_GPU_TIMING` | off | Enable GPU kernel profiling. |
| `NEAT_AI_DISCOVERY_QUIET_GPU` | off | Suppress Mesa/libEGL debug output. |
| `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` | 3 | Maximum consecutive device-lost recovery attempts on the GPU work queue before the pass fails. Accepted range `0–10`; out-of-range or invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_ZERO_COPY` | auto-detect | Force-enable (`1`/`true`) or force-disable (`0`/`false`) zero-copy GPU buffers, overriding hardware auto-detection. Unset lets the library decide from the adapter. |

## Streaming & Parquet

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` | adaptive | Max blocks in the streaming Parquet cache. |
| `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` | 2 | Streaming prefetch depth. |
| `NEAT_AI_DISCOVERY_PRELOAD_ALL` | off | Disable streaming and use full Parquet preload. |
| `NEAT_AI_DISCOVERY_BLOCK_SIZE` | 10000 | Streaming block size in records. |
| `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` | 3600 | Streaming session TTL for orphan cleanup. |

## Focus selection & ranking

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` | off | Enable outlier-focused analysis. |
| `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` | 90 | Outlier identification threshold. |
| `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY` | off | Force output-only focus targets. |
| `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` | off | Bias toward newer input indices. |
| `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS` | off | Prioritise unused input neurons. |
| `NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS` | off | Exclude functionally-constant hidden neurons (zero activation variance) from focus-slot eligibility so they stop wasting slots on candidates that can never succeed; they remain available to the constant-neuron removal path. Slot-waste is surfaced as `focus_ineligible_constant` (Issue #1624). |
| `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE` | off | Gate out neurons whose structural impact magnitude is below `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD` from focus-slot eligibility. Complements the constant filter above by catching *non-constant* near-zero-impact neurons (~31.6% of neurons had `\|impact\| < 1e-6` in production, Issue #1631) that no change can move. Gated count is surfaced as `focus_ineligible_low_impact` (Issue #1635). |
| `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD` | 1e-6 | Impact-magnitude gate: neurons with `\|impact\| <` this value are gated out when the gate above is enabled (retain-on-equal at the boundary). Positive finite values only; invalid or non-positive values fall back to the default (Issue #1635). |
| `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` | dynamic | Constant-source folding threshold. |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` | unset | Cap focus-ranking eager pre-load size in MB. When set and projected size (file × 3) exceeds the budget, lazy mode is used with a structured `info` log (Issue #1172). |
| `NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB` | half of total RAM | Hard ceiling on the records decoded from a single parquet file. Every materialised record is charged against the ceiling **inside** the reader's batch loop, so a file that decodes far larger than its compressed size suggested aborts mid-decode with a typed `MemoryExhausted` error instead of exhausting the host. The analysis phase's own memory budget takes precedence; this variable bounds the paths that carry no caller budget (visualisation snapshot export, focus gradients) and also caps the exporter's dense neuron × observation grid. Unset / invalid / `0` falls back to half of total system RAM (Issue #1869). |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB` | 1024 | Safety margin (MB) reserved from OS-available memory when deciding eager-vs-lazy focus-ranking pre-load. `0` reserves no margin; unset / invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` | 120000 (eager); scaled for lazy | Wall-clock budget for focus ranking; a run that exceeds it aborts with a structured `Timeout` error so the caller falls back to local ranking. When **unset**, the default is scaled by loading mode and projected dataset size — eager keeps the flat 120 s, while a *lazy* pass earns `4 × 120 s + 20 ms per projected MB` so a legitimate lazy fallback finishes instead of aborting into degraded recorded-error aggregation (Issue #3172). An explicit value **wins verbatim** and is never scaled; `0` disables the bound; other values clamp to `[1000, 3600000]` (Issue #1375). |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS` | 60000 | Perf-cliff threshold for a *lazy* focus-ranking pass; a lazy pass at or above this emits one explicit perf-cliff `WARN` naming the neuron count and projected dataset size. Preload never trips it. `0` disables the warning (Issue #1377). |
| `NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH` | off | Fold each neuron's mean reconstruction-activation delta into its focus score as an additive term. Opt-in (`1`/`true`/`yes`) so the throughput shift can be validated on a reference snapshot first (Issue #1634). |
| `NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH_WEIGHT` | 0.1 | Additive weight applied to the mean reconstruction delta when the signal above is enabled. Non-negative finite values only (`0.0` disables the contribution); invalid or negative values fall back to the default (Issue #1634). |

## Detection thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` | 2.0 | Noise-to-signal ratio threshold for the noise-signal detection module. Parsed as `f32`; unset uses the module default. |
| `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` | 2.0 | Input-dominance threshold for input-sensitivity detection. Parsed as `f32`; unset uses the module default. |
| `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` | 10.0 | Gradient threshold for input-sensitivity detection. Parsed as `f32`; unset uses the module default. |

## Analysis budget & deadlines

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` | 20 | Overall wall-clock cap for discovery time in minutes (range 1–120). |
| `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS` | 60000 | Guaranteed minimum window (ms) reserved for synapse/neuron analysis so focus selection + Parquet loading cannot starve it (Issue #1408). Parquet loading is curtailed at `deadline − reserve`; if less than 1s would remain, `analyze_all` fails fast with an actionable error instead of analysing 0/N targets. `0` disables the reserve; other values clamp to `[1, 3600000]`. |
| `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION` | 0.5 | Fraction of the remaining discovery window the reserve may claim (Issue #1408). Effective reserve = `min(ANALYSIS_RESERVE_MS, remaining × fraction)`, so tight budgets are split rather than starving loading. Honoured in `(0.0, 0.9]`; invalid values fall back to `0.5`. |
| `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` | off | Abort if no progress for N seconds. |
| `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` | 2 | Delay between the diagnostic dump and abort. |

## Candidate generation & pruning

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET` | 3 | Max add-neuron candidates per target within a single batch (range 1–32) (Issue #1140). |
| `NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET` | 3 | Max coordinated-structural candidates per final-operation target neuron within a single batch (range 1–32). Mirrors the per-target add-neuron cap; applied after the post-discount expected-gain floor and before any cross-target diversity reordering (Issue #1271). |
| `NEAT_AI_DISCOVERY_MIN_DISTINCT_TARGETS_PER_BATCH` | 3 | Minimum distinct target neurons in an emitted add-neuron batch when the candidate pool supports it. The top of the gain-sorted list is reordered to cover this many distinct targets before the per-target cap is applied (range 1–32) (Issue #1193). |
| `NEAT_AI_DISCOVERY_MAX_ACTIVATION_CONFIGS_PER_TARGET` | 0 (disabled) | Cap on (orientation × scale) activation configs scanned per (source, target) pair for hidden add-neuron evaluation, after squash-family filtering (Issue #1545). A positive value keeps the `N` configs whose scale is closest to `1.0` and drops numerically-unstable extreme scales first. |
| `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET` | unlimited | Cap the number of priority-ordered source neurons evaluated (sample-build + GPU) per focus target (Issue #1542). `0`/unset/invalid = unlimited (back-compat). |
| `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN` | `1e-5` | Minimum **pre-calibration** predicted creature error reduction for emitted add-neuron / add-synapse candidates. Predictions below this screen are dominated by floating-point round-off in the downstream evaluator. The filters compare post-calibration gains, so the screen is converted into that scale by the per-type calibration constant first — the effective floors are `3e-8` (add-neuron) and `3e-9` (add-synapse), never below the `1e-9` noise backstop. Clamped to `[0.0, 1e-2]`; `0.0` disables the floor (Issues #1191, #1778). |
| `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE` | 0.01 | Minimum absolute bypass-synapse weight required to emit a 1-in/1-out hidden-neuron collapse candidate. Bypass weights below this floor mean the chain `a→h→b` contributed nothing meaningful through `h`, so the 4-op coordinated collapse is rejected. Clamped to `[0.0, 0.1]` (Issue #1270). |
| `NEAT_AI_DISCOVERY_CPU_PRE_REJECT` | on | CPU pre-reject screen that runs before the helpful GPU submit (Issue #1544); set `0`/`false`/`no` to disable. Drops helpful add-synapse candidates with no least-squares signal without changing which candidates survive. |
| `NEAT_AI_DISCOVERY_HIDDEN_SQUASH_PRUNE` | on | Squash-aware hidden-target activation scan pruning (Issue #1545); set `0`/`false`/`no` to always scan the full cross-product. A drought escalates the scan back to the full set so pruning can never permanently starve a plateaued creature. |
| `NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD` | 1000 | Hidden-neuron count above which **expensive**-tier discovery modules are skipped at dispatch on non-escalation passes (Issue #1547). Suppressed during drought / novelty-escalation passes so the full set re-enables. `0` disables it entirely. |
| `NEAT_AI_DISCOVERY_MH_TEMPERATURE` | off | Metropolis-Hastings temperature for probabilistic acceptance. |
| `NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL` | off | Re-enable the disabled batch-successful module (Issue #1059). |
| `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR` | unset | **Absolute** override for the minimum `net_improvement` a remove-low-impact candidate must clear to reach the FFI response. Applied verbatim, ignoring `costOfGrowth`, and takes precedence over the `_UNITS` form below. Non-negative finite values only; `0.0` disables the floor (test path); invalid values fall back to the denominated default. Sensible band `1e-8`–`1e-3` at the production `costOfGrowth` (Issues #1142, #1814). |
| `NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR_UNITS` | `1.0` | The remove-low-impact noise floor **in units of `costOfGrowth`** — the screened quantity (`boostedSavings − contribution`) is linear in `costOfGrowth`, so the floor is too (Issue #1814). Effective floor is `units × costOfGrowth`, clamped below by the `1e-9` gain-floor noise backstop. Must stay in `(0.664, 1.5)`: below `0.664` the #1142 numerical-noise class leaks through, at or above `1.5` a zero-contribution orphan can no longer be pruned. `0.0` disables the floor; invalid values fall back to `1.0`. |
| `NEAT_AI_DISCOVERY_COORDINATED_NOISE_FLOOR_MULTIPLIER` | 1.0 | **Test-only escape hatch.** Multiplier applied to every per-op-count coordinated-structural noise floor; production callers leave it unset. Finite `> 0.0` values only, clamped to `[1e-3, 100.0]`; invalid values fall back to `1.0` (Issue #1142). |

## Drought & novelty escalation

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` | 5 | Consecutive trailing empty discovery passes at which the drought diagnostic warn log fires and the `droughtDiagnostic` payload populates on `synapseMetadata` / `neuronMetadata`. Must be `>= 1` (Issue #1202). |
| `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` | 50 | Operator escape hatch: force a one-shot reset of failed-candidate cache entries and active target cooldowns after this many consecutive empty discovery passes (Issue #1205). Armed by default (Issue #1422); set to `0` to disable. |
| `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS` | 100 | Epochs since the creature last accepted a candidate at which a single, durable creature-level drought alarm fires (Issue #1424). Emits a `tracing::warn!` line and a `creatureDroughtAlarm` field carrying the creature uuid, epochs-since-last-acceptance, and an environmental-vs-search-exhaustion classification. Fires once; set to `0` to disable. |
| `NEAT_AI_DISCOVERY_NOVELTY_SUPPRESSION_RATIO` | 0.8 | Fraction of the considered candidate pool that must be cache-suppressed before novelty/diversification escalation engages on a plateaued creature (Issue #1423). Honoured in `(0.0, 1.0]`. |
| `NEAT_AI_DISCOVERY_NOVELTY_GAIN_RELAXATION` | 0.5 | Multiplier applied to the coordinated-structural expected-gain floor when novelty escalation engages, loosening it so structurally-novel candidates survive. Honoured in `(0.0, 1.0]`; never raises the floor above the base constant (Issue #1423). |
| `NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR` | 0.1 | Multiplier applied to a single-op `RemoveNeuron` coordinated candidate's `expectedCreatureScoreGain` while the creature is in a **search-exhaustion** drought (Issue #1448). Engages only when the trailing-failure streak reaches the drought threshold **and** the drought classifies as `search_exhaustion`. Clamped to `[0.001, 1.0]`; `1.0` disables it. |
| `NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR` | 0.25 | Cold-start calibration prior multiplier for non-invertible / periodic target activations (SINE, COSINE, GAUSSIAN, SQUARE, ABSOLUTE). Applied while the per-(`change_type`, `target_squash`) bucket has fewer than three failure samples. Clamped to `[0.001, 1.0]` (Issue #1192). |
| `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` | 0.2 | Rolling success-rate threshold below which the creature enters **Conservative** discovery mode (Issue #1132). Honoured in `(0.0, 1.0]`; invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` | 20 | Maximum consecutive failed passes Conservative mode persists before it is abandoned and Normal mode resumes. Must be `>= 1`; invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` | 10.0 | Multiplier applied to the coordinated-structural expected-gain floor (`COORDINATED_MIN_EXPECTED_GAIN`) while Conservative mode is active. Values `< 1.0` are clamped up to `1.0`; the floor never relaxes below the base constant. |
| `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES` | 3 | Consecutive per-target failures before a target neuron enters cooldown. Must be `>= 1`; invalid values fall back to the default (Issue #1273). |
| `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS` | 10 | Cooldown duration in epochs for a skipped target neuron. Must be `>= 1`; invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT` | 1 | Within-batch failures on a single target before subsequent same-target candidates in that batch are short-circuited. Must be `>= 1`; invalid values fall back to the default. |
| `NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR` | 2 | Divisor that shrinks the target-cooldown epoch window while in Conservative mode so failing targets re-enter focus sooner. Clamped to `[1, 64]`. |
| `NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR` | 4 | Divisor applied to the target-cooldown window during an extended drought (last-ditch escape hatch; effective cooldown floored at 2 epochs). Clamped to `[1, 64]`. |

## Memory & recording gates

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` | 0.5 macOS / 1.0 Linux | Minimum available memory (GB) below which discovery is gated off (Issue #1420). Lower it (e.g. `0.1`) so a small-but-capable ~8GB host can proceed; `0` disables the available-memory gate. Invalid / out-of-range (`0.0–64.0`) values fall back to the platform default. The 4GB total-memory minimum is unaffected. |
| `NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION` | 1.0 | Fraction of selected focus neurons that must have **zero** Parquet rows before the fail-fast insufficient-recording gate skips synapse/neuron analysis (Issue #1444). The gate detects this with a cheap record-count scan *before* GPU work and surfaces `insufficient_recording` as the dominant rejection reason. Honoured in `(0.0, 1.0]`; `0` disables the gate. |

## Observability & diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_TIMING` | off | Print phase timing to stderr when set to any value (convention `=1`). |
| `NEAT_AI_DISCOVERY_PROFILE` | off | Emit a structured profile to stderr. The only recognised value is `json` (case-insensitive); anything else disables profiling. |
| `NEAT_AI_DISCOVERY_GPU_METRICS` | off | Print GPU metrics to stderr when set to any value (convention `=1`). |
| `NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD` | 10.0 | `actual/expected` ratio above which prediction-vs-actual calibration mismatches are logged via `tracing::warn!`. Must be finite and `> 1.0`; invalid values fall back to the default so the log channel cannot be silenced by a malformed value. |
| `NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION` | on for debug builds, off for release builds | Trip a `debug_assert!` when a discovery pass cannot account for every considered candidate (Issue #1802), so a newly-added silent drop path fails CI. Set to `0` to force warn-only. The `unaccounted_drop` rejection-breakdown entry and the `tracing::warn!` naming the surface and delta are emitted regardless, so a mismatch is never silent. See [docs/analysis/candidate-reconciliation-1802.md](analysis/candidate-reconciliation-1802.md). |
| `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` | `sample` | Path override for the macOS `sample` binary used for thread-dump diagnostics. Diagnostics/tooling only — not a discovery-tuning knob. |

## Related guides

- [README.md — Troubleshooting](../README.md#troubleshooting)
- [docs/GPU_GUIDE.md](GPU_GUIDE.md) — GPU performance tuning and debugging
- [docs/DROUGHT_PLAYBOOK.md](DROUGHT_PLAYBOOK.md) — diagnosing "no successful candidates" droughts
- [docs/FOCUS_SELECTION.md](FOCUS_SELECTION.md) — focus-selection design and env knobs
- [docs/analysis/candidate-reconciliation-1802.md](analysis/candidate-reconciliation-1802.md) — the fail-loud candidate-reconciliation invariant
