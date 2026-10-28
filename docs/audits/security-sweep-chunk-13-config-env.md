# Security sweep — chunk `13`: Config / env-var surface — src/config

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `13`
- **Sweep date:** `2026-10-28`
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2`
- **Exposure:** `local`
- **Swept by:** Issue #2224 (scaffold for chunk 13 of the #2083 overflow tracker; regions filled by later sub-issues); 13c filled by Issue #2265
- **Tracker issue:** #2096 (parent #2121)

### Sweep status — IN PROGRESS

The index carries a `last_swept` date because the ledger contract (`tests/issue_2088_sweep_ledger_contract.rs`) requires a record for any index entry with a record. This record is partly filled: region 13c is swept (Issue #2265), and every other outcome below still reads `pending`. The date is not a completed-sweep claim until the rows are decided, the same approach as the chunk 8b record (Issue #2103). The date will stand as a completed sweep only once no row reads `pending`.

**Re-check note.** HEAD differs from the baseline commit. Three commits since the baseline touch `src/config`: `7694130` (#2195, removed `outlier_analysis` and `outlier_percentile`), `8c5639e` (#2402, GPU_RETRY_LIMIT warning, for #2364) and `2660968` (#2407, for #2352). Net: `user_facing.rs` 1396 → 1445 lines and `mod.rs` 426 → 416 lines at HEAD at the time of writing. All line counts in this record are baseline counts unless labelled HEAD, and later citations must say which they are.

## Files swept

Line counts as at the baseline commit.

### small modules

<!-- section: 13a -->

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/config/helpers.rs` | 102 | pending |
| `src/config/detection.rs` | 70 | pending |
| `src/config/drought_mitigation.rs` | 106 | pending |
| `src/config/observability.rs` | 88 | pending |
| `src/config/mod.rs` | 426 | pending |

### user_facing L1-664

<!-- section: 13b -->

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/config/user_facing.rs` | 664 | pending |

### user_facing L665-end

<!-- section: 13c -->

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/config/user_facing.rs` | 732 | swept (Issue #2265) — one finding filed: #2460 (low, `analysis/utils/deadline.rs::cap_deadline_to_wall_clock` multiplies the unbounded FFI `max_discovery_wall_clock_minutes` by 60 unchecked, probe (e)). Two findings recorded on existing issues: zero-spelling detail on #2213 and doc drift on #2125. Probe (a) is safe (`FocusDeadline::resolve` takes the earlier deadline, pinned by #2264). Probe (f) `SAMPLE_PROGRAM` has no new finding (#2251, #2118). All 18 variables in the range have a row below; the range runs from `DEFAULT_FOCUS_RANKING_BUDGET_MS` to `drought_reset_after_epochs` and excludes `focus_ranking_memory_margin_mb`, which sits in 13b |

The `user_facing.rs` split is by line: L1–664 is row 13b and L665–1396 at baseline is row 13c.

**13c range at HEAD.** The 13c range maps to HEAD from `DEFAULT_FOCUS_RANKING_BUDGET_MS` to the end of `user_facing.rs::drought_reset_after_epochs`. Code is cited by symbol, not line number, because HEAD has drifted from the baseline (CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number").

### readers outside src/config

<!-- section: 13d -->

| Path | Lines | Outcome |
| --- | --- | --- |

Rows for readers outside `src/config` are added by Issue #2271.

## Defect classes probed

1. Unbounded numeric config driving allocation or thread creation.
2. Silent fallback on invalid input that disables a guard.
3. Path-valued settings.
4. Boolean or flag settings that relax security.
5. Precedence and shadowing (env vs FFI vs default).
6. Secret or sensitive-value logging.
7. Documentation drift.

The classes are probed by later sub-issues; none has been probed yet.

## Settings table

One row per environment variable read under `src/config`, filled region by region by the sub-issue that owns it. Each region starts with its header row and no data rows.

The `reader file:line` column is fixed by the agreed schema. Each sub-issue must give the symbol (`file.rs::symbol`) beside the line and say whether the line is baseline or HEAD, because lines drift (CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number", Issue #1942).

### small modules

<!-- section: 13a -->

| variable | reader file:line | type | default | accepted range / validation | on invalid input | 0 / opt-out semantics | can weaken a guard? (which) | cached (OnceLock) or live | logged how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### user_facing L1-664

<!-- section: 13b -->

| variable | reader file:line | type | default | accepted range / validation | on invalid input | 0 / opt-out semantics | can weaken a guard? (which) | cached (OnceLock) or live | logged how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### user_facing L665-end

<!-- section: 13c -->

The reader column gives `file.rs::symbol` only, the header keeps the agreed schema name. Every accessor is live (it reads `std::env::var` on each call); none is cached in a `OnceLock`. "Class" below means the silent guard opt-out class, #2213 (planning issue, closed) and #2276 (the open issue that files it). Hostile-value behaviour is pinned by `tests/issue_2096_config_user_facing_b.rs` (#2263).

| variable | reader file:line | type | default | accepted range / validation | on invalid input | 0 / opt-out semantics | can weaken a guard? (which) | cached (OnceLock) or live | logged how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` | `user_facing.rs::focus_ranking_budget_override` (via `focus_ranking_budget_ms` and `effective_focus_ranking_budget_ms`) | `u64` ms | 120000 (`DEFAULT_FOCUS_RANKING_BUDGET_MS`); an unset value is scaled for lazy runs by `scale_default_focus_ranking_budget_ms` (4 x default plus 20 ms per projected MB, clamped) | trimmed; a positive value is clamped to 1000-3600000 ms; an explicit value is never scaled | empty, non-numeric, negative or fractional: armed default (scaled); `u64::MAX` parses and clamps to 3600000 | `0` disables the budget (`None`), as do `00` and `+0`; `0.0`, `-0` and `0e5` keep the armed default; garbage keeps it active (#2263); class #2213 / #2276 | yes: the focus-ranking wall-clock net, and with `0` and no shared deadline focus ranking is unbounded. `FocusDeadline::resolve` still takes the earlier of the budget and the caller's shared deadline, so the env value cannot outlive it (#2264) | live | `debug` when `0`, when clamped and on invalid input |
| `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS` | `user_facing.rs::analysis_reserve_ms` (consumer `orchestration.rs::analyze_all`) | `u64` ms | 60000 | a positive value is clamped to 1-3600000 ms | empty, non-numeric, negative or fractional: default 60000 | `0` disables the reserve, as do `00` and `+0`; `0.0`, `-0` and `0e5` keep the default; garbage keeps it active (#2263); class #2213 / #2276 | yes: the #1408 guaranteed analysis window and its fail-fast shortfall error; with `0` focus ranking and Parquet loading may consume the whole window | live | `debug` when `0` and on invalid input |
| `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION` | `user_facing.rs::analysis_reserve_fraction` (consumer `orchestration.rs::analyze_all`) | `f64` | 0.5 | finite and above 0; a value above 0.9 is clamped to 0.9 (`ANALYSIS_RESERVE_FRACTION_MAX`) | empty, non-finite, zero, negative or non-numeric: default 0.5 | none: `0` is not an opt-out, it falls back to 0.5 | no: the fraction only splits the window between loading and analysis, and it is capped at 0.9 | live | `debug` on invalid input |
| `NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION` | `user_facing.rs::resolve_insufficient_recording_fraction` (via `insufficient_recording_fraction`, consumer `orchestration.rs::analyze_all`) | `f64`, as `Option` | 1.0 | finite, above 0 and at most 1 | empty, non-finite, negative, above 1 or non-numeric: default 1.0 (gate armed) | `0` disables the fail-fast gate (`None`); all six spellings (`0`, `00`, `+0`, `0.0`, `-0`, `0e5`) disable it; garbage keeps it active (#2263); class #2213 / #2276 | yes: the #1444 insufficient-recording gate; with `0` analysis always runs, even when no focus neuron has recorded rows | live | none (no log in any branch, invalid input included) |
| `NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS` | `user_facing.rs::focus_ranking_perf_cliff_ms` (consumer `focus/ranking/mod.rs::maybe_warn_perf_cliff`) | `u64` ms | 60000 | any `u64`, no clamp; `u64::MAX` is accepted (#2263) | empty, non-numeric or negative: default 60000 | `0` disables the warning (`lazy_pass_exceeds_perf_cliff` requires a threshold above 0) | no: it gates one `warn!` only, and a huge value silences that warning and nothing else | live | `debug` on invalid input; none for `0` |
| `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` | `user_facing.rs::session_ttl_secs` (via `parse_env`, consumer `streaming.rs::cleanup_stale_sessions`) | `u64` s | 3600 | clamped to 60-86400 | empty or non-numeric: default 3600 (silent) | none: `0` clamps up to 60, it does not switch cleanup off | no: cleanup cannot be disabled, and the range bounds how long an orphaned session is kept | live | none (`parse_env` logs nothing) |
| `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` | `user_facing.rs::max_wall_clock_minutes` (via `parse_env`, consumer `orchestration.rs::analyze_all`) | `u64` minutes | 20 | clamped to 1-120 | empty or non-numeric: default 20 (silent) | none: `0` clamps up to 1 | no from the env, which is only the fallback. The FFI `max_discovery_wall_clock_minutes` wins over it and has no 1-120 bound (finding #2460), so the env var cannot loosen a caller-supplied cap | live | none |
| `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` | `user_facing.rs::sample_program_override` (via `sample_program`, consumer `debug/sample_capture.rs::sampler_program`) | `String`, a program name or path | unset: `sample` on macOS, none elsewhere | any value that is non-blank after `trim`; returned untrimmed (`"  x  "` keeps its padding, #2263); no path or character validation | blank: treated as unset | none | no: it picks the executable the SIGUSR1 thread dump spawns. `sample_capture.rs::run_external_command_with_timeout` uses `Command::new`, so a bare name resolves through `PATH`, nothing runs through a shell, the argv is fixed, there is no privilege elevation, and whoever sets the variable already runs code as that user (probe (f), #2251, #2118) | live | the accessor logs nothing; `sample_capture.rs::capture` echoes the program name into the dump text (stderr) |
| `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` | `user_facing.rs::low_success_rate_threshold` (via `parse_env`) | `f32` | 0.2 (`DEFAULT_LOW_SUCCESS_RATE_THRESHOLD`) | finite, above 0 and at most 1 | otherwise the default (silent) | none: `0` falls back to 0.2 | no: an in-range value only moves when conservative mode engages | live | none |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` | `user_facing.rs::conservative_mode_max_epochs` (via `parse_env`) | `u32` | 20 (`DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`) | at least 1 | otherwise the default (silent) | none: `0` falls back to 20 | no: it only bounds how long the risk bias lasts | live | none |
| `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` | `user_facing.rs::conservative_gain_multiplier` (via `parse_env`) | `f32` | 10.0 (`DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER`) | finite, then raised to at least 1.0 | non-finite or non-numeric: the default (silent); a finite value below 1 clamps up to 1.0 | none | no: the floor is never relaxed below the base constant | live | none |
| `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` | `user_facing.rs::drought_log_threshold` (via `parse_env`) | `u32` | 5 (`DEFAULT_DROUGHT_LOG_THRESHOLD`) | at least 1 | otherwise the default (silent) | none: `0` falls back to 5 | no: it only sets when the drought diagnostic logs | live | none |
| `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS` | `user_facing.rs::resolve_drought_alarm_epochs` (via `drought_alarm_epochs`, consumer `orchestration.rs::analyze_all`) | `u32`, as `Option` | 100 (`DEFAULT_DROUGHT_ALARM_EPOCHS`) | any `u32`, trimmed | empty, non-numeric, negative or above `u32::MAX`: the armed default | `0` disables the alarm (`None`), as do `00` and `+0`; `0.0`, `-0` and `0e5` keep the default; garbage keeps it active (#2263); class #2213 / #2276 | yes: the creature-level drought alarm, which is observability only, nothing is gated on it | live | none |
| `NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR` | `user_facing.rs::resolve_remove_neuron_drought_factor` (via `remove_neuron_drought_factor`) | `f32` | 0.1 (`DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR`) | finite, clamped to 0.001-1.0 | empty, non-numeric or non-finite: default 0.1; a finite out-of-range value clamps instead of falling back (`0` and `-1` become 0.001, 5 becomes 1.0, #2263) | none: `1.0` is the documented neutral value that disables the deprioritisation, and `0` clamps to 0.001 | no: the clamp keeps the factor from zeroing or inflating a prediction | live | none |
| `NEAT_AI_DISCOVERY_NOVELTY_SUPPRESSION_RATIO` | `user_facing.rs::novelty_suppression_ratio` (via `parse_env`) | `f64` | 0.8 (`DEFAULT_SUPPRESSION_RATIO_THRESHOLD`) | finite, above 0 and at most 1 | otherwise the default (silent) | none: `0` falls back to 0.8 | no: it only sets when novelty escalation engages | live | none |
| `NEAT_AI_DISCOVERY_NOVELTY_GAIN_RELAXATION` | `user_facing.rs::novelty_gain_relaxation` (via `parse_env`) | `f32` | 0.5 (`DEFAULT_GAIN_FLOOR_RELAXATION`) | finite, above 0 and at most 1 | otherwise the default (silent) | none: `0` falls back to 0.5 | loosens the coordinated-structural gain floor by design, as a multiplier of at most 1.0 so it never raises the floor; a tiny positive value such as `1e-30` loosens it close to zero, which is the documented operator lever and only applies while novelty escalation is engaged | live | none |
| `NEAT_AI_DISCOVERY_MIN_AVAILABLE_MEMORY_GB` | `user_facing.rs::resolve_min_available_memory_gb` (via `min_available_memory_gb`, consumer `analysis/utils/memory.rs`) | `f64` GB | platform default `DEFAULT_MIN_AVAILABLE_MEMORY_GB` (0.5 macOS, 1.0 Linux) | finite, 0.0-64.0 (`MAX_MIN_AVAILABLE_MEMORY_GB`) | empty, non-finite, negative, above 64 or non-numeric: the platform default, which is the armed floor | `0` disables the available-memory gate; all six spellings do; garbage keeps it active (#2263); class #2213 / #2276 | yes: the available-memory discovery gate (#1420) | live | `debug` on invalid input; none for `0` |
| `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` | `user_facing.rs::drought_reset_after_epochs` (consumer `orchestration.rs::analyze_all`) | `u32`, as `Option` | 50 (`DEFAULT_DROUGHT_RESET_AFTER_EPOCHS`) | any `u32`, trimmed | unparsable: the armed default 50, the exemplar that invalid input must not disable a guard | `0` disables the lever (`None`), as do `00` and `+0`; `0.0`, `-0` and `0e5` keep the armed default; garbage keeps it active (#2263); class #2213 / #2276 | yes: the operator escape hatch, a one-shot target-cooldown reset after this many empty passes | live | none |

### readers outside src/config

<!-- section: 13d -->

| variable | reader file:line | type | default | accepted range / validation | on invalid input | 0 / opt-out semantics | can weaken a guard? (which) | cached (OnceLock) or live | logged how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Issues filed

Write `negative-result` when the sweep found nothing. An empty section is indistinguishable from an unfinished record.

**Class findings (small modules)**

<!-- section: 13a-findings -->

- pending — no findings recorded yet

**Small modules (13a)**

<!-- section: 13a -->

- pending — no findings recorded yet

**user_facing L1-664 (13b)**

<!-- section: 13b -->

- #2404 (low) — `config/user_facing.rs::watchdog_abort_delay` accepts any `u64` and `watchdog.rs::watchdog_loop` sleeps that long before `std::process::abort()`, so the abort becomes a hang. The fix lives in the accessor, not in `watchdog.rs`. Cross-links #2119 and #2252.
- #2405 (medium) — `watchdog.rs::watchdog_loop` truncates `config.stall_timeout.as_millis() as u64`, so a huge `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` yields a near-zero threshold and aborts a healthy process (CWE-197). Cross-links #2252.

**user_facing L665-end (13c)**

<!-- section: 13c -->

- #2460 (low, proposed `severity:low`, `confidence:high`, exposure FFI caller only) — `analysis/utils/deadline.rs::cap_deadline_to_wall_clock` evaluates `minutes * 60` inside `Duration::from_secs(...)` with no overflow check, so the `checked_add` does not protect it. The FFI `max_discovery_wall_clock_minutes: Option<u64>` reaches it through `orchestration.rs::analyze_all` unbounded, unlike the env fallback clamped to 1-120. `307445734561825861` wraps to a 44 s cap in release builds (no `overflow-checks`) and panics in debug builds. Probe (e).
- #2213 (existing, comment [#2213 comment](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2213#issuecomment-6467974556)) — which zero spellings disable each of the six guard opt-outs: `00` and `+0` disable the integer guards, `0.0`, `-0` and `0e5` keep them armed, all six disable the two float guards; garbage never disables any. The open class issue is #2276. No new issue.
- #2125 (existing, comment [#2125 comment](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2125#issuecomment-6467975056)) — the `src/ffi_types/requests.rs` doc says `None` for `max_discovery_wall_clock_minutes` means no cap, but `analyze_all` falls back to the env value or the 20-minute default. See also the doc-drift section below.
- Tracker comment on #2096: [#2096 comment](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2096#issuecomment-6467998495) — lists the three items above.

**Readers outside src/config (13d)**

<!-- section: 13d -->

- pending — no findings recorded yet

## Doc drift observed

Documentation that disagrees with the code under `src/config`, recorded per region.

**Small modules (13a)**

<!-- section: 13a -->

- pending — no findings recorded yet

**user_facing L1-664 (13b)**

<!-- section: 13b -->

- pending — no findings recorded yet

**user_facing L665-end (13c)**

<!-- section: 13c -->

- `src/ffi_types/requests.rs` (`max_discovery_wall_clock_minutes`): "When `None`, no overall cap is enforced" is wrong. `orchestration.rs::analyze_all` uses the env value (clamped 1-120) or 20 minutes. Reported on #2125.
- `docs/CONFIGURATION.md`, `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` row: does not say that an FFI-supplied `max_discovery_wall_clock_minutes` takes precedence over it. Reported on #2125.
- `docs/CONFIGURATION.md`, `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION` row: says "Honoured in `(0.0, 0.9]`; invalid values fall back to `0.5`", but `user_facing.rs::analysis_reserve_fraction` clamps a value above 0.9 to 0.9 instead of falling back. For #2125 to reconcile.
- `docs/CONFIGURATION.md`, `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` row: does not state the 60-86400 clamp that `user_facing.rs::session_ttl_secs` applies. For #2125 to reconcile.
- `user_facing.rs::remove_neuron_drought_factor` doc comment: says out-of-range values "fall back to the default" and, in the same paragraph, that the value is clamped. A finite out-of-range value clamps (`0` and `-1` become 0.001, #2263); only empty, non-numeric and non-finite values fall back. Not a security finding, so no issue; for #2125 to reconcile.

**Readers outside src/config (13d)**

<!-- section: 13d -->

- pending — no findings recorded yet

## Probe dispositions (13c)

Probes (a)–(k) are from #2123. Code is cited by symbol. Re-verified at HEAD (`b2ca056`) before recording.

- **(a) Focus budget versus the shared deadline — safe, no finding.** `focus/ranking/mod.rs::FocusDeadline::resolve` builds a budget deadline from `effective_focus_ranking_budget_ms` and a shared deadline from `from_shared_deadline`, and keeps the one that expires first. An explicit env value, even the 3600000 ms maximum, cannot outlive the caller's `analysisDeadlineMs`. The `0` opt-out leaves focus ranking unbounded only when the caller supplies no shared deadline. Pinned by `focus/ranking/mod.rs::focus_deadline_tests::resolve_explicit_max_budget_is_still_bounded_by_shared_deadline`, `resolve_disabled_budget_still_honours_shared_deadline` and `resolve_disabled_budget_without_shared_deadline_is_unbounded` (#2264). The accessors are pinned by `tests/issue_2096_config_user_facing_b.rs::focus_ranking_budget_ms_clamps_disables_and_falls_back` and `effective_focus_ranking_budget_ms_scales_only_the_default` (#2263).
- **(b) Analysis reserve — no finding, class opt-out.** `analysis_reserve_ms` honours `0` as the opt-out (`debug` log) and clamps a positive value to 1-3600000; `analysis_reserve_fraction` honours a finite value above 0, clamps a value above 0.9 to 0.9, and falls back to 0.5 for empty, non-finite, zero, negative or non-numeric input. Consumer: `orchestration.rs::analyze_all`. Garbage keeps the reserve active: `tests/issue_2096_config_user_facing_b.rs::garbage_never_disables_any_zero_opt_out`; ranges in `analysis_reserve_ms_clamps_and_falls_back` and `analysis_reserve_fraction_clamps_and_falls_back`. Class issue #2213 / #2276, log level `debug`.
- **(c) Insufficient-recording gate — no finding, class opt-out.** `0` (any spelling) disables the gate with no log in any branch. Pinned by `insufficient_recording_fraction_disables_on_zero_only`, `resolve_insufficient_recording_fraction_matches_the_table` and `float_opt_outs_accept_every_zero_spelling`. Log level: none. Class issue #2213 / #2276.
- **(d) Perf-cliff threshold — no finding.** `focus_ranking_perf_cliff_ms` is an unbounded `u64` that gates one `warn!`; `u64::MAX` silences that warning and nothing else (`focus_ranking_perf_cliff_ms_accepts_huge_unclamped`).
- **(e) Wall-clock cap precedence — one finding.** `orchestration.rs::analyze_all` prefers the FFI `max_discovery_wall_clock_minutes` and falls back to `max_wall_clock_minutes` (clamped 1-120, default 20). The env var cannot loosen a caller-supplied cap. The FFI value has no bound and `cap_deadline_to_wall_clock` multiplies it by 60 unchecked: finding #2460. `session_ttl_secs` (60-86400) is recorded in the settings table; no finding. Doc drift on #2125. Chunk 4 (#2090) owns FFI input validation in general, so #2460 cross-links rather than audits it.
- **(f) `SAMPLE_PROGRAM` — no new finding.** The only path-valued setting in `src/config`. `sample_capture.rs::sampler_program` takes `sample_program_override`; `run_external_command_with_timeout` runs it with `Command::new`, so a bare name resolves through `PATH`; there is no shell, the argv is fixed, stdout and stderr go to null, the child is killed after the timeout, and there is no privilege elevation. The value is not trimmed (`sample_program_override_keeps_a_padded_value_untrimmed`) and has no path validation, so a padded name fails to spawn and the dump says `Failed to run '<program>'`. The program name is echoed into the dump text on stderr, which is the process's own log. Whoever sets the variable already runs code as that user. Already audited in the chunk-11 ledger under #2251 (and #2118, closed); the predictable-path hint is tracked in #2266. Nothing is re-filed here.
- **(g) Drought and mode levers — no finding.** Ranges are in the rows above. `low_success_rate_threshold` (0-1], `conservative_mode_max_epochs` at least 1, `conservative_gain_multiplier` finite then at least 1.0, `drought_log_threshold` at least 1, `drought_alarm_epochs` (`0` disables, unparsable keeps the armed default; class opt-out, log level none), `remove_neuron_drought_factor` clamped to 0.001-1.0.
- **(h) Novelty levers — no finding.** Both honoured in (0, 1]; `novelty_gain_relaxation` only ever loosens the floor, by design.
- **(i) Available-memory floor — no finding, class opt-out.** `0` disables the gate with no log; range 0-64 GB; invalid input lands on the platform default, which is the armed floor. Log level: none for `0`, `debug` for invalid input. Class issue #2213 / #2276.
- **(j) Drought reset — no finding, class opt-out.** Unparsable input keeps the armed default 50 (the exemplar for invalid input not disabling a guard); `0` disables the lever with no log. Pinned by `drought_reset_after_epochs_disables_on_zero_only`. Class issue #2213 / #2276.
- **(k) Fallback scan — no site loosens a guard.** Every invalid-input fallback in the range lands on the default: `focus_ranking_budget_override`, `analysis_reserve_ms`, `analysis_reserve_fraction`, `resolve_insufficient_recording_fraction`, `focus_ranking_perf_cliff_ms`, `session_ttl_secs`, `max_wall_clock_minutes`, `low_success_rate_threshold`, `conservative_mode_max_epochs`, `conservative_gain_multiplier`, `drought_log_threshold`, `resolve_drought_alarm_epochs`, `resolve_remove_neuron_drought_factor`, `novelty_suppression_ratio`, `novelty_gain_relaxation`, `resolve_min_available_memory_gb` and `drought_reset_after_epochs`. The platform default for the memory floor and the armed default 50 for the drought reset are the tightening choices.

**Behaviour differences listed in the #2263 PR summary.** (a) Non-literal zero spellings disable a guard: comment on #2213, no new issue. (b) `sample_program_override` returns a padded value untrimmed: recorded under probe (f), no finding. (c) `focus_ranking_perf_cliff_ms` accepts `u64::MAX` unclamped: probe (d), no finding; `remove_neuron_drought_factor` clamps `0` and `-1` to 0.001 instead of falling back: doc drift above, no security finding.

## Outcome

The sweep is in progress. Region 13c, `user_facing.rs` L665–end, is swept (Issue #2265): one finding filed (#2460) and two recorded on existing issues (#2213, #2125); the probe dispositions are under "Probe dispositions (13c)". The other regions have no recorded outcome yet: 13a small modules, 13b `user_facing.rs` L1–664 (findings #2404 and #2405 are filed, rows not yet decided) and 13d readers outside `src/config`.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a sweep and never justify a non-null `last_swept`.

- #1385 — Reconcile focus-ranking budget clamp: code doesn't clamp but README claims [1000, 3600000]
- #1408 — Reserve a guaranteed minimum analysis budget so focus selection + parquet load cannot starve synapse/neuron analysis
- #1420 — Low-memory gate disables discovery on ~8 GB hosts — make the available-memory floor configurable / reclaim-aware
- #1869 — Parquet pre-load bound is estimated from compressed size and only checked after the allocation (src/analysis/utils/memory.rs)
- #1873 — Safe GPU setup fns assert an unenforceable thread-free invariant around unsafe env writes
- #1904 — XDG_RUNTIME_DIR fallback adopts a pre-existing /tmp directory of any ownership or mode (src/analysis/utils/platform.rs)
- #2006 — Numeric env-override parse chain duplicated across 35 accessors with inconsistent trimming

## Verify this record

```bash
git diff b85a551ed2521ed327469b20eb88aeda828357d2..HEAD -- src/config
```

An empty diff means this record still describes the current code. The diff is not empty at HEAD: see the re-check note under Record.
