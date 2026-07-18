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
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` | 120000 | Wall-clock budget for focus ranking; a run that exceeds it aborts with a structured `Timeout` error so the caller falls back to local ranking. `0` disables the bound; other values clamp to `[1000, 3600000]` (Issue #1375). |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS` | 60000 | Perf-cliff threshold for a *lazy* focus-ranking pass; a lazy pass at or above this emits one explicit perf-cliff `WARN` naming the neuron count and projected dataset size. Preload never trips it. `0` disables the warning (Issue #1377). |

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
| `NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN` | `1e-5` | Absolute minimum `expected_creature_score_gain` for emitted add-neuron / add-synapse candidates. Predictions below this floor are dominated by floating-point round-off in the downstream evaluator. Clamped to `[0.0, 1e-2]` (Issue #1191). |
| `NEAT_AI_DISCOVERY_MIN_BYPASS_WEIGHT_FOR_COLLAPSE` | 0.01 | Minimum absolute bypass-synapse weight required to emit a 1-in/1-out hidden-neuron collapse candidate. Bypass weights below this floor mean the chain `a→h→b` contributed nothing meaningful through `h`, so the 4-op coordinated collapse is rejected. Clamped to `[0.0, 0.1]` (Issue #1270). |
| `NEAT_AI_DISCOVERY_CPU_PRE_REJECT` | on | CPU pre-reject screen that runs before the helpful GPU submit (Issue #1544); set `0`/`false`/`no` to disable. Drops helpful add-synapse candidates with no least-squares signal without changing which candidates survive. |
| `NEAT_AI_DISCOVERY_HIDDEN_SQUASH_PRUNE` | on | Squash-aware hidden-target activation scan pruning (Issue #1545); set `0`/`false`/`no` to always scan the full cross-product. A drought escalates the scan back to the full set so pruning can never permanently starve a plateaued creature. |
| `NEAT_AI_DISCOVERY_MODULE_TIERING_HIDDEN_THRESHOLD` | 1000 | Hidden-neuron count above which **expensive**-tier discovery modules are skipped at dispatch on non-escalation passes (Issue #1547). Suppressed during drought / novelty-escalation passes so the full set re-enables. `0` disables it entirely. |
| `NEAT_AI_DISCOVERY_MH_TEMPERATURE` | off | Metropolis-Hastings temperature for probabilistic acceptance. |
| `NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL` | off | Re-enable the disabled batch-successful module (Issue #1059). |

## Drought & novelty escalation

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` | 5 | Consecutive trailing empty discovery passes at which the drought diagnostic warn log fires and the `droughtDiagnostic` payload populates on `synapseMetadata` / `neuronMetadata`. Must be `>= 1` (Issue #1202). |
| `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` | 50 | Operator escape hatch: force a one-shot reset of failed-candidate cache entries and active target cooldowns after this many consecutive empty discovery passes (Issue #1205). Armed by default (Issue #1422); set to `0` to disable. |
| `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS` | 100 | Epochs since the creature last accepted a candidate at which a single, durable creature-level drought alarm fires (Issue #1424). Emits a `tracing::warn!` line and a `creatureDroughtAlarm` field carrying the creature uuid, epochs-since-last-acceptance, and an environmental-vs-search-exhaustion classification. Fires once; set to `0` to disable. |
| `NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK` | 15 | Consecutive per-(creature, module) failure count at which a single discovery module is temporarily disabled for that creature. Clamped to `[1, 1000]` (Issue #1273). |
| `NEAT_AI_DISCOVERY_MODULE_STARVATION_COOLDOWN_EPOCHS` | 10 | Epochs that a starved module remains disabled before being re-armed. Clamped to `[1, 10_000]` (Issue #1273). |
| `NEAT_AI_DISCOVERY_NOVELTY_SUPPRESSION_RATIO` | 0.8 | Fraction of the considered candidate pool that must be cache-suppressed before novelty/diversification escalation engages on a plateaued creature (Issue #1423). Honoured in `(0.0, 1.0]`. |
| `NEAT_AI_DISCOVERY_NOVELTY_GAIN_RELAXATION` | 0.5 | Multiplier applied to the coordinated-structural expected-gain floor when novelty escalation engages, loosening it so structurally-novel candidates survive. Honoured in `(0.0, 1.0]`; never raises the floor above the base constant (Issue #1423). |
| `NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR` | 0.1 | Multiplier applied to a single-op `RemoveNeuron` coordinated candidate's `expectedCreatureScoreGain` while the creature is in a **search-exhaustion** drought (Issue #1448). Engages only when the trailing-failure streak reaches the drought threshold **and** the drought classifies as `search_exhaustion`. Clamped to `[0.001, 1.0]`; `1.0` disables it. |
| `NEAT_AI_DISCOVERY_RISKY_SQUASH_PRIOR` | 0.25 | Cold-start calibration prior multiplier for non-invertible / periodic target activations (SINE, COSINE, GAUSSIAN, SQUARE, ABSOLUTE). Applied while the per-(`change_type`, `target_squash`) bucket has fewer than three failure samples. Clamped to `[0.001, 1.0]` (Issue #1192). |

## Memory & recording gates

| Variable | Default | Description |
|----------|---------|-------------|
| `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` | 0.5 macOS / 1.0 Linux | Minimum available memory (GB) below which discovery is gated off (Issue #1420). Lower it (e.g. `0.1`) so a small-but-capable ~8GB host can proceed; `0` disables the available-memory gate. Invalid / out-of-range (`0.0–64.0`) values fall back to the platform default. The 4GB total-memory minimum is unaffected. |
| `NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION` | 1.0 | Fraction of selected focus neurons that must have **zero** Parquet rows before the fail-fast insufficient-recording gate skips synapse/neuron analysis (Issue #1444). The gate detects this with a cheap record-count scan *before* GPU work and surfaces `insufficient_recording` as the dominant rejection reason. Honoured in `(0.0, 1.0]`; `0` disables the gate. |

## Related guides

- [README.md — Troubleshooting](../README.md#troubleshooting)
- [docs/GPU_GUIDE.md](GPU_GUIDE.md) — GPU performance tuning and debugging
- [docs/DROUGHT_PLAYBOOK.md](DROUGHT_PLAYBOOK.md) — diagnosing "no successful candidates" droughts
- [docs/FOCUS_SELECTION.md](FOCUS_SELECTION.md) — focus-selection design and env knobs
