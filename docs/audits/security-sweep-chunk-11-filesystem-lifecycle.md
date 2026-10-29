# Security sweep — chunk `11`: Filesystem lifecycle

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `11` — matches the `id` in the index.
- **Human name:** Filesystem lifecycle — `src/discovery_cleanup.rs`,
  `src/debug.rs` + `src/debug/`, `src/watchdog.rs`, `src/tracking_alloc.rs`,
  `src/discovery_history.rs`.
- **Sweep date:** `2026-10-05`
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2`
- **Exposure:** `local`
- **Swept by:** Issue #2095 (chunk 11 of the #2083 overflow tracker), split
  across its audit sub-issues #2117 (11a), #2118 (11b), #2119 (11c) and #2120
  (11d); scaffolded by Issue #2233.
- **Tracker issue:** `#2095`

### Sweep status — COMPLETE

All 8 files in this chunk have been swept, across four slices: #2234 (PR #2257), #2251 (PR #2268), #2252 (PR #2287) and #2253 (PR #2394). Issue #2254 then reconciled the tables below against a fresh grep inventory of the chunk and wrote the `## Outcome` and `## Issues filed` sections. The index's `last_swept` date now marks this finished sweep.

### Why the finding tables cite symbols, not line numbers

Issue #2233 specified `file:line` columns. These tables use `file.rs::symbol`
instead, per **CONTRIBUTING.md § Cite Code by Symbol, Never by Line Number**
(Issue #1942), matching the chunk 8b record: a line number rots at the next
refactor, while a symbol survives it and a test can check it still exists.

## Files swept

Line counts as at the baseline commit.

### discovery_cleanup

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/discovery_cleanup.rs` | 798 | findings filed — #2255 (lossy `display()` path in the orphan sweep bypasses the #1903 age floor) and #2256 (`discovery.lock` has no in-tree writer and the host spells it `.discovery.lock`); every mutation is otherwise symlink-safe or deliberately follows a caller-owned root |

Probe dispositions (Issue #2234):

- **Probe (a) — symlinked base root: accepted.** A `base_dir` symlink whose
  final component contains `.discovery` passes the marker gate in
  `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since`, and its
  `exists()`, `is_dir()` and `read_dir` follow it; each child is
  `<link>/<child>`, so `remove_discovery_dir`'s `symlink_metadata` and
  `remove_dir_all` resolve through the intermediate link. Acceptable under
  the local co-tenant model: the root defaults to `.discovery` in the
  caller's own working directory (NEAT-AI `discoveryBaseDirectory`), so only
  the caller can plant the link, and a link to a scratch volume is a
  legitimate layout. The blast radius stays the target's lock-less child
  directories — never the target, its parent, or a child link. Pinned by
  `tests/issue_2095_cleanup_symlinked_root.rs`: the canary outside both the
  link and its target, the target, a locked session and the link all survive.
- **Probe (b) — TOCTOU between the `entry.file_type()` check and
  `remove_dir_all`: no finding.** A child swapped for a symlink after the scan
  loop's `file_type()` check is re-probed by `remove_discovery_dir`'s
  `fs::symlink_metadata`, which refuses a symlink and, via its `is_dir()`
  check, any non-directory. The residual window between that probe and
  `fs::remove_dir_all` is harmless: the `std::fs::remove_dir_all`
  documentation states it "does **not** follow symbolic links and it will
  simply remove the symbolic link itself" at the top-level path, and does not follow
  symlinks within the tree either (it unlinks them). The race is not
  deterministically testable, so there is no race test;
  `tests/issue_1903_orphan_sweep_lock_recheck.rs` stays the regression surface
  for the lock window.
- **Probe (c) — crafted entry names: one finding (#2255).**
  `assert_is_discovery_dir` refuses `Component::ParentDir` before any probe,
  and `read_dir` yields single-component names joined under the root, so an
  entry can never be absolute or contain `..`. Pinned by
  `tests/issue_2095_cleanup_entry_names.rs`: unicode, `...`, dash-led,
  newline and bidi-override orphans are removed and a lock-less canary beside
  the root survives. The `..` refusal and its surviving target are already
  pinned by
  `tests/issue_1866_cleanup_dir_path_guard.rs::test_cleanup_rejects_parent_dir_traversal`. The lossy case is a finding: the sweep passes
  `path.display().to_string()` to `cleanup_orphaned_discovery_dir`, so a
  non-UTF-8 name becomes U+FFFD and the removal targets a literal-U+FFFD
  sibling that skipped the age floor (reproduced: sibling touched after
  `scan_started` removed), while the real orphan is never removed and is
  reported as `already_gone`. Filed as #2255; its fix passes the `Path`
  through and ships the failing-first test.
- **Probe (d) — dangling-symlink lock: covered.**
  `tests/issue_1903_orphan_sweep_lock_recheck.rs::dangling_symlink_lock_is_not_orphaned`
  asserts `is_directory_orphaned` is `false`, then `removed == 0`, no errors,
  and that the session directory still exists after the sweep. That surviving
  session directory is this probe's canary: the dangling lock has no real
  target to protect.
- **`discovery.lock` writer: finding filed (#2256).** `LOCK_FILE_NAME` has no
  in-tree production writer — only tests create it. The contract is the
  caller's: `docs/FFI_API.md` § "Discovery Directory Cleanup" and
  § "Orphaned Directory Sweep", and the doc comment on
  `ffi/utilities.rs::clean_orphaned_discovery_dirs` ("orphaned when it has no
  `discovery.lock` file"). The in-tree sweep guard is the #1903 age floor plus
  the fail-closed lock probe. The ownership signal is unenforced: the only
  known host, NEAT-AI `src/discovery/DiscoveryCleanup.ts`, writes
  `.discovery.lock`, so the FFI sweep would read every live NEAT-AI session
  older than the scan as orphaned (reproduced). NEAT-AI does not bind the FFI
  sweep today, so there is no live path; filed as #2256.

### debug + sampler

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/debug.rs` | 604 | no finding — no filesystem mutation and no process spawn (a grep for `fs::`, `File::`, `OpenOptions`, `DirBuilder`, `Command::new`, `.exists()` and `remove_dir_all` has no hits); SIGUSR1 is only deliverable by the same uid or root, and SIGUSR1 dumps are serialised on the one signal-handler thread; concurrent `render_thread_dump` callers each get their own `SampleDir` (signal/lifecycle probe) |
| `src/debug/sample_capture.rs` | 509 | finding filed — #2266 (`sample_capture.rs::write_manual_hint` tells the operator to run the sampler with `-file /tmp/sample.txt`, re-creating the predictable shared-tmp path #1905 removed); the spawn itself is shell-free with a fixed argv and a program only the same uid can choose |
| `src/debug/sample_dir.rs` | 285 | no finding — exclusive, non-recursive `0700` directory creation under `std::env::temp_dir()`; `read_guarded` refuses a symlink or a foreign-owned file, and its stat→read window sits inside the euid-owned `0700` directory |
| `src/debug/process_state.rs` | 150 | no finding — in-memory only: renders the breaker, heartbeat, outstanding-request and local-backtrace state into a `String`, with no filesystem or process access |

Probe dispositions (Issue #2251):

- **External process — no finding.** `sample_capture.rs::sampler_program`
  picks the non-empty `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` override
  (`config::sample_program_override`), else `"sample"` on macOS (resolved via
  `PATH`), else none. `sample_capture.rs::run_external_command_with_timeout`
  spawns it through `Command::new` — no shell — with the fixed argv
  `[pid, "1", "-mayDie", "-file", <capture path>]` and `Stdio::null()` on
  stdout and stderr, and kills it after `SAMPLE_TIMEOUT_SECS` (5 s) plus
  `SAMPLE_KILL_GRACE_MS` (500 ms). Only the process's own environment chooses
  the program, and same-uid control of env or `PATH` is outside the attacker
  model. The config reader is not re-audited here: the override surface is
  #2096 (closed), with #2123 and #2124 as its open follow-ups. A child still
  alive after the grace window is returned with `status: None` and not
  reaped — a zombie until process exit, a hygiene point, not a security one.
- **Output handling — same-uid disclosure only.**
  `sample_capture.rs::read_capture` and
  `sample_capture.rs::write_filtered_sample_output` copy the process's own
  stack symbols into the dump, which goes to its own stderr and so to the
  operator's log. A refusal from `read_guarded` is written into the dump as a
  WARNING rather than swallowed.
- **`/tmp/sample.txt` manual hint — finding filed (#2266).**
  `sample_capture.rs::write_manual_hint` prints
  `Try manually: sample <pid> 1 -mayDie -file /tmp/sample.txt` on every
  fallback once a sampler program has resolved (with none resolved,
  `sample_capture.rs::capture` prints a `gdb` hint off macOS instead), steering the operator to exactly the predictable path in a
  world-writable directory that #1905 removed from the automated path.
  `tests/issue_1934_sample_fallback.rs` pins the `Try manually: sample`
  prefix, so the fix keeps that prefix.
- **Signal/lifecycle — no finding.** `debug.rs::init_debug_handlers` runs once
  (a `OnceLock`, from the library's one-time initialisation) and
  `debug.rs::install_signal_handler` registers SIGUSR1 through
  `signal_hook::iterator::Signals`, so `debug.rs::dump_all_threads` runs on the
  `signal-handler` thread, outside signal context, one dump at a time. Only
  the same uid or root can send SIGUSR1. The public
  `debug.rs::render_thread_dump` can run concurrently with a signalled dump,
  but each call gets its own capture directory (nanoseconds plus a
  process-local counter in `sample_dir.rs::unique_suffix`), so two dumps never
  share a path. `debug.rs::shutdown_debug_handlers` (FFI
  `cleanup_discovery_lib`) closes the signal iterator and joins each thread
  with a 10 s bound, so shutdown cannot hang on a wedged dump.
- **Tmp directory and read TOCTOU — refuted.** A co-tenant can predict the
  `neat_ai_discovery.sample.<pid>.<nanos>.<seq>.<attempt>.d` name and
  pre-create it, but `sample_dir.rs::create_private_dir` is a non-recursive
  `mkdir` that fails with `AlreadyExists` on anything present — a symlink
  included — so `sample_dir.rs::SampleDir::create` moves to the next name.
  Squatting all eight attempts is a denial of the capture only, and it
  degrades loudly: a WARNING, the manual hint and a `NoBacktraces` banner. The
  window between `read_guarded`'s `symlink_metadata` and its `read_to_string`
  is not exploitable: the capture path's parent is the euid-owned `0700`
  directory, so no other uid can swap the entry.
- **Test-only predictable names — noted, not a finding.** The
  `sample_capture.rs` test module writes helper scripts at predictable
  `$TMPDIR` names. It never ships in the library, and a co-tenant racing a
  developer's test run is outside the sweep's production scope.

### watchdog + tracking_alloc + discovery_history

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/watchdog.rs` | 364 | audited — no PID, process-spawn or filesystem surface (it only raises SIGUSR1 at its own process and aborts); the uncapped abort delay and the stall-timeout truncation are already tracked by open #2259 under #2122, so nothing new is filed |
| `src/tracking_alloc.rs` | 234 | audited — the `GlobalAlloc` hooks cannot panic, allocate or recurse; the counter wraps only on a caller layout mismatch, which is undefined behaviour; neither consumer of `tracking_alloc.rs::TrackingAlloc::allocated` can spuriously cancel or return early |
| `src/discovery_history.rs` | 626 | audited — no filesystem I/O, and its JSON arrives only as the `get_calibration_summary` FFI string; #1906 holds on that path; two findings filed: #2391 (`record_attempt` overflows `u32` at the ceiling, Rust API only) and #2392 (finite observations yield non-finite calibration metrics, returned as `null` under `success: true`) |

Probe dispositions (Issue #2252):

- **PID and filesystem — none.** A grep of `src/watchdog.rs` and
  `src/tracking_alloc.rs` for `fs::`, `File`, `remove`, `OpenOptions`,
  `Command`, `pid` and `process::id` has one hit: the test module's
  `std::env::remove_var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS")`.
  `watchdog.rs::watchdog_loop` signals only its own process
  (`signal_hook::low_level::raise`, not `kill(pid, …)`), so no PID is read,
  stored or reused. An abort skips every `Drop`, so a live session directory
  is left for the orphan sweep; the #1903 guards on that sweep are
  re-verified by #2234 in the discovery_cleanup section, not here.
- **SIGUSR1 — a handler is installed on the FFI path.** The analysis FFI
  entries run `lib.rs::log_version_once` → `debug.rs::init_debug_handlers` →
  `debug.rs::install_signal_handler`, which registers SIGUSR1 through
  `signal_hook::iterator::Signals` before `analysis/orchestration.rs::analyze_all`
  calls `watchdog.rs::start_from_env`. The raise therefore dumps the threads,
  and `std::process::abort` follows `abort_delay` later. After
  `cleanup_discovery_lib` → `debug.rs::shutdown_debug_handlers` closes the
  iterator, signal-hook-registry keeps its low-level handler installed, so a
  later raise is a no-op and the abort still follows. Only when `Signals::new`
  failed (logged as a WARNING) or a direct Rust caller skipped the one-time
  initialisation does SIGUSR1 keep its default disposition, which terminates
  the process at once without the dump. The stall is still logged at ERROR
  first, so this costs diagnosis, not safety. Observation only: the
  `debug.rs::install_signal_handler` doc comment says SIGUSR1 has "no default
  action", but POSIX's default action for it is to terminate the process.
- **Discarded `raise` result — hides nothing.** `let _ = raise(SIGUSR1)`
  drops a result that can only be an error for an invalid signal number, and
  SIGUSR1 is a constant.
- **Abort delay — cross-referenced, not refiled.**
  `config/user_facing.rs::watchdog_abort_delay` accepts any `u64` seconds, so
  a huge `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` makes
  `watchdog_loop`'s `thread::sleep(config.abort_delay)` turn the abort into
  the very hang the watchdog exists to end. This is #2122's finding, tracked
  by open #2259 as its finding A.
- **Stall-timeout truncation — cross-referenced, not refiled.**
  `watchdog_loop` compares against `config.stall_timeout.as_millis() as u64`;
  a stall of `18446744073709552` seconds truncates to about 384 ms and fires
  on a healthy run (CWE-197). Tracked by open #2259 as its finding B.
- **Lifecycle — no finding.** `watchdog.rs::WatchdogConfig::from_env` returns
  `None` unless the stall timeout is positive. `watchdog.rs::Watchdog::start`
  publishes its state in `ACTIVE` (the last start wins) and spawns the
  `hang-watchdog` thread with `.ok()`, so a failed spawn silently runs with no
  watchdog — a hygiene point, not a security one. `watchdog.rs::Drop::drop`
  sets the stop flag, clears `ACTIVE` only when `Arc::ptr_eq` matches (a
  stale handle cannot clear a newer watchdog) and joins without holding
  `ACTIVE`; the loop polls at most every 5 s, so the join is bounded. Once a
  stall is detected the loop no longer checks the stop flag, so the abort is
  inevitable and a concurrent join waits for it — by design.
  `watchdog.rs::heartbeat_snapshot` takes both locks with a 50 ms
  `try_lock_for`, so a thread dump never blocks on them; the stall path's
  blocking `stage.lock()` contends only with `beat`, which holds it for one
  assignment.
- **Allocator hooks — no panic, no allocation, no recursion.** `alloc`,
  `alloc_zeroed`, `dealloc` and `realloc` call only `System` and a Relaxed
  `fetch_add` or `fetch_sub`: no formatting, no locking, no logging and no
  `Vec`, so a hook can neither re-enter the allocator nor panic inside it.
  Relaxed is enough for a statistics counter that orders nothing else.
- **Counter wrap — reachable only through undefined behaviour.** `dealloc`
  subtracts `layout.size()` unconditionally, and `realloc` subtracts the
  shrink delta, so the counter can wrap only when a caller frees with a
  layout other than the one it allocated with. The `GlobalAlloc` contract
  makes that undefined behaviour, so no sound test can exercise the wrap —
  which is why none does — and safe Rust cannot reach it.
- **Consumer `ffi/utilities.rs::discovery_memory_usage_bytes` — no spurious
  zero.** It returns `tracking_alloc.rs::TrackingAlloc::allocated` through
  a `catch_unwind` whose `.unwrap_or(0)` fallback is dead code: the closure is a single atomic load
  and cannot panic, so the FFI never reports a false 0.
- **Consumer `analysis/utils/memory.rs::is_memory_budget_exceeded` →
  `analysis/orchestration.rs::analyze_all` — no spurious cancel.** Without
  undefined behaviour the counter never over-counts, so the budget cannot
  trip early. It measures the whole process, by design (#1028). The two
  early returns in `analyze_all` set `memory_budget_exceeded: true` and log a
  WARNING, so a cancel is reported, never silent; the third check skips only
  post-processing and keeps the candidates. A budget of 0 is the caller's
  explicit request. The three `allocated_bytes` log fields in `analyze_all`
  only read the counter.
- **No other consumers.** A grep of `src`, `benches`, `fuzz` and `tests`
  for `.allocated()` finds only the sites above (benches and tests would be
  out of scope anyway).

Probe dispositions (Issue #2253):

- **Filesystem — none; one FFI entry point.** A grep of
  `src/discovery_history.rs` for `fs::`, `File`, `OpenOptions`, `Command`,
  `remove_`, `std::io` and `Path` has no hit: the module only computes and
  (de)serialises. Its one FFI entry is
  `ffi/utilities.rs::get_calibration_summary` →
  `ffi_internal/analysis.rs::get_calibration_summary_internal`, which parses
  the caller's `discoveryHistory` string with `serde_json::from_str` into a
  `DiscoveryHistory` (a parse error is returned as `InvalidInput`) and calls
  only `DiscoveryHistory::calibration_summary`. The scoring reader,
  `focus/ranking/mod.rs::rank_focus_neurons_with_history`, has no caller in
  `src/` outside the `focus` module, so
  `NeuronDiscoveryHistory::bayesian_score` is reached only through the
  public Rust API.
- **#1906 — re-verified, holds on the live path.** `NeuronDiscoveryHistory`
  carries `#[serde(try_from = "NeuronDiscoveryHistoryWire")]`, so every
  deserialise — each map entry of a `DiscoveryHistory` included — runs
  `discovery_history.rs::NeuronDiscoveryHistory::try_from`, which rejects
  `successes > attempts`; no other `Deserialize` path builds the type.
  `discovery_history.rs::NeuronDiscoveryHistory::bayesian_score` computes
  failures as `attempts.saturating_sub(successes)`. The regression tests
  `test_deserialize_rejects_successes_exceeding_attempts`,
  `test_bayesian_score_invalid_counts_saturates` and
  `test_deserialize_accepts_valid_counts` exist and pass.
- **`record_attempt` overflow — reachable through the Rust API only, finding
  filed (#2391).** `discovery_history.rs::NeuronDiscoveryHistory::record_attempt`
  adds with a plain `+= 1`. `try_from` accepts
  `attempts == successes == u32::MAX`; one more failed attempt then panics in
  debug (`attempt to add with overflow`) or, with `[profile.release]` leaving
  `overflow-checks` off, wraps `attempts` to 0 while `successes` stays at
  `u32::MAX`, so the saved history fails its next load. No FFI path reaches
  it: nothing in `src/` calls `DiscoveryHistory::record`, and the FFI entry
  above never records. Pinned by the ignored failing-first unit test
  `test_record_attempt_keeps_successes_within_attempts_at_u32_max`.
- **Calibration NaN/inf — reaches the FFI summary, finding filed (#2392).**
  JSON carries no NaN or inf literal, and serde_json refuses an out-of-range
  number such as `1e400` (pinned by the passing unit test
  `test_deserialise_rejects_non_finite_json_numbers`), but finite values
  overflow in the arithmetic. `predicted = 1.7e308, actual = -1.7e308` makes
  the mean absolute error in `CalibrationTracker::calibration_summary` `inf`,
  and adding the opposite pair makes its bias NaN. Two ratios of
  `±1e308 / 1e-9` make `discovery_history.rs::compute_calibration_factor` sum
  `+inf` and `-inf` to NaN, which `f64::clamp` passes through, so
  `DiscoveryHistory::calibration_factor` and the summary's factor leave the
  documented `[0.1, 10.0]`. serde_json writes a non-finite `f64` as `null`, so
  the FFI answers `success: true` with `null` metrics.
  `DiscoveryHistory::record_calibration` also stores NaN or inf from a Rust
  caller unchecked. Pinned by the ignored failing-first unit tests
  `test_calibration_summary_stays_finite_for_extreme_finite_observations` and
  `test_calibration_factor_is_never_nan_for_finite_observations`.
- **`record_calibration` bound — unbounded, no finding.**
  `CalibrationTracker::record_prediction` pushes onto a `Vec` with no cap,
  and `DiscoveryHistory::prune` never trims calibration. Nothing in `src/`
  calls `record_calibration`, so on the FFI path the list is exactly what the
  caller's own JSON carried: memory and the single pass in
  `calibration_summary` are linear in that input, and the response holds one
  entry per key the caller supplied — no amplification. A Rust host that
  records without end grows only its own history.
- **`prune` — no finding.** `discovery_history.rs::DiscoveryHistory::prune`
  collects the current UUIDs into a `HashSet<&str>` and `retain`s the map
  entries whose key is in it: duplicate UUIDs are harmless, an empty slice
  removes every entry (no neuron is current), and nothing in it can panic.
  Observation only: deserialisation does not check that a map key matches
  its entry's `uuid` field, but `prune`, `get` and `bayesian_score_for` all
  key on the map key, and nothing in `src/` outside the module's tests reads
  `NeuronDiscoveryHistory::uuid`, so a mismatch changes no decision.
- **Calibration key — observation only.** `calibration_key` joins module and
  candidate type with `::` and `parse_calibration_key` splits at the first
  `::`, so module `a::b` with type `c` and module `a` with type `b::c` share
  one bucket. Nothing in `src/` records calibration, and on the FFI path the
  caller writes the keys itself, so this is noted, not filed.
- **#1902 — re-verified, holds on the live path.** The guard lives in
  `src/streaming.rs`: `streaming.rs::finish_session` sets
  `preserve_tmp_on_drop` after its `records_written == 0` exit and before
  both `writer.finish()` and `fs::rename`, and `streaming.rs::Drop::drop`
  returns before `fs::remove_file` when the flag is set. See its rows in
  "Filesystem mutation sites" and "Re-verified remediations".

## Defect classes probed

- Symlink following — a mutation or read that resolves an attacker-placed
  symlink.
- TOCTOU — a check (existence, lock, ownership) separated from the act it
  guards.
- Tmp-file creation — predictable names, pre-existing paths, unsafe modes.
- External-process execution — spawned commands, their arguments and search
  path.
- PID handling — signalling or inspecting a process by a PID that may be
  stale or reused.
- Allocation accounting — counters that can underflow, overflow or drift from
  the real allocator.
- Cleanup blast radius — how much a delete can remove beyond what the process
  created.

## Filesystem mutation sites

| Site (`file.rs::symbol`) | Operation | Path root | Symlink-safe (yes/no/why) |
| --- | --- | --- | --- |
| `discovery_cleanup.rs::assert_is_discovery_dir` | `LOCK_FILE_NAMES.iter().any(\|name\| path.join(name).exists())` (both `discovery.lock` and the host's `.discovery.lock`, Issue #2256) combined with `path.join(DISCOVERY_DATA_FILE_NAME).exists()` — read-only gate | caller-supplied `temp_dir` after the empty and `..` checks, or a swept `<base_dir>/<child>` via `remove_discovery_dir` | follows symlinks — yes, read-only: a followed link can only admit a path, and `remove_discovery_dir` refuses a symlinked `temp_dir` next |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::symlink_metadata(path)` | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` | yes — does not follow the final component; a symlink is refused (`InvalidInput`) and so is a non-directory. Intermediate components resolve, which probe (a) accepts for a caller-owned root |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::canonicalize(path)` — audit log only | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` | follows symlinks — yes, log only: the resolved path is never acted on, and an error falls back to the raw path |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::remove_dir_all(path)` | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` and the `symlink_metadata` refusal | yes — std does not follow a top-level symlink nor symlinks within the tree (probe b); only intermediate components resolve (probe a) |
| `discovery_cleanup.rs::is_directory_orphaned` | `LOCK_FILE_NAMES.iter().all(\|name\| ...)` probing each lock spelling with `fs::symlink_metadata(dir.join(name))` (Issue #2256) | a swept child of `base_dir`, or `temp_dir` during the #1903 re-check | yes — the lock itself is not followed; fails closed on any error but `NotFound` for every name (probe d) |
| `discovery_cleanup.rs::directory_touched_since` | `fs::symlink_metadata(dir)?.modified()` | a swept child of `base_dir` after the marker gate | yes — reads the entry's own mtime, not a link target's; errors propagate |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.exists()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a): the root is caller-owned |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.is_dir()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `fs::read_dir(base_path)` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a); yields single-component names only (probe c) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `entry.file_type()` | a child of `base_dir` | yes — `DirEntry::file_type` does not follow; a symlinked child is skipped |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `path.is_dir()` | a child of `base_dir`, already known not to be a symlink | follows, but only after the `file_type()` symlink skip; a later swap is caught by `remove_discovery_dir`'s re-probe (probe b) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `remove_discovery_dir(&path, LockRecheck::Enforce)` | a child of `base_dir` | yes — fixed by #2255: the `Path` is passed through unchanged, `display()` is used only for logging, and `remove_discovery_dir` re-probes the exact entry with `symlink_metadata` rather than a lossy re-rendered string |
| `sample_dir.rs::create_private_dir` | `DirBuilder` with `mode(0o700)` and `recursive(false)`, then `set_permissions(0o700)` (unix); a mode-less non-recursive `create` elsewhere | `std::env::temp_dir()` joined with the per-invocation name from `SampleDir::create` | yes — `mkdir` does not follow the final component, so a planted symlink or any existing entry fails with `AlreadyExists`; `set_permissions` runs only on the directory this call just created |
| `sample_dir.rs::SampleDir::create` | up to `MAX_CREATE_ATTEMPTS` (8) calls to `create_private_dir` on `neat_ai_discovery.sample.<pid>.<nanos>.<seq>.<attempt>.d` | `std::env::temp_dir()` | yes — every attempt is exclusive; a co-tenant squatting all eight names only denies the capture, which is reported (WARNING, manual hint, `NoBacktraces`) |
| `sample_dir.rs::Drop::drop` | `fs::remove_dir_all(dir)`; `NotFound` ignored, any other error printed as a WARNING | the directory `SampleDir::create` made | yes — std does not follow a top-level symlink nor symlinks within the tree, and the directory is euid-owned `0700`, so no other uid can plant inside it; a same-uid swap inside it is the same principal deleting its own files, not a boundary crossing |
| `sample_dir.rs::read_guarded` | `fs::symlink_metadata(path)`, refuse a non-regular file (`InvalidData`) or a uid other than the euid (`PermissionDenied`), then `fs::read_to_string(path)` | `<SampleDir>/sample.txt` | yes — the final component is not followed; the stat→read TOCTOU is refuted because the parent is the euid-owned `0700` directory |
| `sample_capture.rs::run_external_command_with_timeout` | process spawn: `Command::new(program)` with the argv `[pid, "1", "-mayDie", "-file", <capture path>]` and null stdout/stderr; killed after 5 s plus a 500 ms grace | the sampler writes `<SampleDir>/sample.txt` | yes — no shell; the output path is inside the private directory, and the program comes only from the process's own env or `PATH` (same uid) |
| none — `src/watchdog.rs`, `src/tracking_alloc.rs` (Issue #2252) | no filesystem operation or process spawn: a grep for `fs::`, `File`, `remove`, `OpenOptions`, `Command`, `pid` and `process::id` hits only a test-module `remove_var` | n/a | n/a — no path is touched; `src/discovery_history.rs` is swept by 11c-2 |
| none — `src/discovery_history.rs` (Issue #2253) | no filesystem operation or process spawn: a grep for `fs::`, `File`, `OpenOptions`, `Command`, `remove_`, `std::io` and `Path` has no hit | n/a — the history arrives as an FFI string | n/a — no path is touched |
| `streaming.rs::Drop::drop` (`impl Drop for RecordingSession`, cited for #1902) | `fs::remove_file("<parquet_path>.tmp")`; skipped when `finished` or `preserve_tmp_on_drop` is set; `NotFound` ignored, any other error logged at WARNING | `<temp_dir>/discovery_data.parquet.tmp`, under the session's caller-supplied `temp_dir` | yes for the final component — `remove_file` unlinks a symlink rather than its target; intermediate components resolve under the caller-owned `temp_dir` |

**Grep reconciliation.** Chunk 11's non-test code was re-grepped for
every filesystem- or process-mutating verb, with each file cut at its
top-level `#[cfg(test)] mod tests` before the search:

```
rg -n 'remove_(file|dir_all)|create_dir|rename|symlink|canonicalize|set_permissions|\.exists\(\)|Command::new' src/discovery_cleanup.rs src/discovery_history.rs src/debug.rs src/debug/ src/watchdog.rs src/tracking_alloc.rs
```

- Every hit that mutates the filesystem or spawns a process already has a row
  in the table above:
  `discovery_cleanup.rs::remove_discovery_dir` (`fs::remove_dir_all`);
  `sample_dir.rs::Drop::drop` (`remove_dir_all`);
  `sample_dir.rs::create_private_dir` (bare `set_permissions`);
  `sample_capture.rs::run_external_command_with_timeout` (`Command::new`).
- Every other hit is a read-only probe, a doc comment or an unrelated word
  match, grouped by file below:
  - **src/discovery_cleanup.rs:**
    - `assert_is_discovery_dir` — the `.exists()` marker probe is a read-only
      precondition.
    - Doc and comment text in `cleanup_discovery_dir`, `remove_discovery_dir`,
      `is_directory_orphaned` and `clean_orphaned_discovery_dirs_since` —
      prose only.
    - `remove_discovery_dir` — the `fs::symlink_metadata` probe plus the
      `is_symlink()` refusal are a read-only guard. The `fs::canonicalize`
      call resolves the path for the log line only.
    - `is_directory_orphaned` — the `fs::symlink_metadata` lock probe is a
      read.
    - `directory_touched_since` — the `fs::symlink_metadata(..).modified()`
      mtime read.
    - `clean_orphaned_discovery_dirs_since` — the `base_path.exists()` early
      return is a read, and the `ft.is_symlink()` skip refuses to follow a
      symlink.
  - **src/discovery_history.rs:** six `#[serde(rename_all = "camelCase")]`
    attributes. This is serde field renaming, with no filesystem use.
  - **src/debug/sample_capture.rs:** the doc comment on `write_manual_hint`
    (the word "symlink").
  - **src/debug/sample_dir.rs:**
    - The module doc (`//!`) — prose.
    - The `use std::fs::{DirBuilder, Permissions, set_permissions};` import
      inside `create_private_dir` — not a call. The call is covered by the
      row above.
    - `read_guarded` — its doc comment and its `std::fs::symlink_metadata`
      probe are a read-only refusal guard.
    - `describe` — `metadata.is_symlink()` only labels the refusal message.
  - **src/debug.rs, src/debug/process_state.rs, src/watchdog.rs,
    src/tracking_alloc.rs:** no production hits.
- **Bare-call blind spot.** The regex matches verb text, so it catches an
  imported verb called bare only when the verb's own name is in the regex —
  e.g. `set_permissions`, imported in `debug/sample_dir.rs` via
  `use std::fs::{DirBuilder, Permissions, set_permissions};` and called bare
  as `set_permissions(dir, ...)`, is caught. An imported or bare-called verb
  the regex does not name slips past:
  `DirBuilder::new().mode(..).create(dir)` right beside it in
  `sample_dir.rs::create_private_dir` does not match, nor would
  `File::create`, `OpenOptions::open`, `fs::write`, or any verb imported under
  a `use … as` alias. The slice full-file audits (#2234, #2251, #2252, #2253)
  read every line of their files rather than relying on the grep, so they
  cover this gap — the `create_private_dir` row above already records the
  `DirBuilder` create.

## Re-verified remediations

| Issue | Guard | Citing site (`file.rs::symbol`) | On live path? |
| --- | --- | --- | --- |
| #1903 | lock re-read immediately before removal (`LockRecheck::Enforce` → `CleanupOutcome::Claimed`); fail-closed lock probe (only `NotFound` means "no lock"); age floor (a child touched at or after `scan_started` is counted `claimed`) | `discovery_cleanup.rs::remove_discovery_dir` (re-check), `discovery_cleanup.rs::is_directory_orphaned` (fail-closed probe), `discovery_cleanup.rs::directory_touched_since` and the age-floor match in `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since`; regression surface `tests/issue_1903_orphan_sweep_lock_recheck.rs` | yes — FFI `ffi/utilities.rs::clean_orphaned_discovery_dirs` → `discovery_cleanup.rs::clean_orphaned_discovery_dirs` → `clean_orphaned_discovery_dirs_since` → `remove_discovery_dir` (`LockRecheck::Enforce`). Holds for UTF-8 names; this sweep found #2255 (the age floor could be bypassed for a non-UTF-8 entry) and #2256 (the guard keyed on a name the known host never writes), both since fixed: #2255 by commit 2ef5eba, #2256 by PR #2269 |
| #1905 | owner-only per-invocation capture directory removed on `Drop` on every exit path, including kill-on-timeout; a symlink or foreign-owned file at the capture path is refused, not followed; guard test `tests/issue_1905_sample_temp_dir.rs` | `sample_dir.rs::SampleDir::create`, `sample_dir.rs::create_private_dir`, `sample_dir.rs::Drop::drop`, `sample_dir.rs::read_guarded` (via `sample_capture.rs::read_capture`); fallback-text surface `tests/issue_1934_sample_fallback.rs` | yes — SIGUSR1 → `debug.rs::dump_all_threads` → `debug.rs::render_thread_dump` → `sample_capture.rs::capture`. The manual fallback hint named `/tmp/sample.txt` (#2266), since fixed by PR #2275: `sample_capture.rs::write_manual_hint` now prints `"$(mktemp -d)/sample.txt"` |
| #1904 | `RUNTIME_DIR_MODE` (`0700`) and `prepare_runtime_dir` in `src/analysis/utils/platform.rs`: create owner-only, re-apply the mode past the umask, refuse a pre-existing world-writable directory or a symlink | `platform.rs::prepare_runtime_dir`, pinned by its `xdg_runtime_dir_*` unit tests; the sampler only mirrors the pattern in `sample_dir.rs::create_private_dir` and does not call it | yes on Linux (`platform.rs::ensure_xdg_runtime_dir`); not on the sampler path — no `src/debug*` file references it |
| #1906 | `successes > attempts` refused on every deserialise; the failure count saturates when scoring | `discovery_history.rs::NeuronDiscoveryHistory::try_from` (the `TryFrom<NeuronDiscoveryHistoryWire>` impl behind `#[serde(try_from)]`), `discovery_history.rs::NeuronDiscoveryHistory::bayesian_score`; regression tests `test_deserialize_rejects_successes_exceeding_attempts`, `test_bayesian_score_invalid_counts_saturates` and `test_deserialize_accepts_valid_counts` in `src/discovery_history.rs` | yes — FFI `ffi/utilities.rs::get_calibration_summary` → `ffi_internal/analysis.rs::get_calibration_summary_internal` → `serde_json::from_str` into `DiscoveryHistory` runs `try_from` per entry, and a corrupt entry fails the call as `InvalidInput`. `bayesian_score` is off the FFI path (Rust API only). The invariant holds on the way in, but `record_attempt` can break it at the `u32` ceiling (#2391) |
| #1902 | `preserve_tmp_on_drop` set before `writer.finish()` and the rename, so `Drop` keeps the complete `.tmp` after a failed finalise | `streaming.rs::finish_session` (sets the flag), `streaming.rs::Drop::drop` (returns before `fs::remove_file` when it is set); regression tests `test_failed_finish_preserves_tmp` and `test_empty_session_finish_removes_tmp` in the `src/streaming.rs` tests module | yes — FFI `ffi/recording.rs::finish_discovery_session` → `streaming.rs::finish_session`; the flag is set after the `records_written == 0` exit (an empty recording is still deleted, by design) and before both fallible steps |

## Outcome

The five 2025-era remediations #1902–#1906 hold on their live paths. Five
findings were filed:

- #2255 (CLOSED, fixed) — the lossy-path orphan sweep bypassed the #1903 age
  floor for a non-UTF-8 directory name.
- #2256 (CLOSED, fixed) — the liveness probe keyed on an unwritten lock name.
- #2266 (CLOSED, fixed) — the manual thread-dump hint re-taught the
  predictable `/tmp/sample.txt` path.
- #2391 (OPEN) — `record_attempt` can overflow `u32` at the ceiling #1906
  accepts.
- #2392 (OPEN) — non-finite calibration metrics can reach the FFI under
  `success: true`.

Two of the five findings remain open.

## Issues filed

- #2255 — Orphan sweep removes a lossy-rendered path, bypassing the #1903 age
  floor for non-UTF-8 entries (src/discovery_cleanup.rs::clean_orphaned_discovery_dirs_since)
  (CLOSED, fixed by commit 2ef5eba)
- #2256 — Orphan sweep liveness keys on discovery.lock, which nothing in-tree
  writes and NEAT-AI spells .discovery.lock
  (src/discovery_cleanup.rs::is_directory_orphaned) (CLOSED, fixed by PR #2269)
- #2266 — Thread-dump manual hint re-teaches the predictable /tmp/sample.txt
  path #1905 removed (src/debug/sample_capture.rs::write_manual_hint) (CLOSED,
  fixed by PR #2275)
- #2391 — NeuronDiscoveryHistory::record_attempt overflows u32 at the ceiling
  the #1906 guard accepts, breaking successes <= attempts
  (src/discovery_history.rs::NeuronDiscoveryHistory::record_attempt) (OPEN)
- #2392 — Calibration metrics go non-finite from finite inputs and reach the
  get_calibration_summary FFI as null under success: true
  (src/discovery_history.rs::compute_calibration_factor) (OPEN)

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- #1902 — a failed `finish_session` no longer deletes the complete recording.
- #1903 — the orphan sweep no longer deletes a directory a session claimed
  after its `discovery.lock` check.
- #1904 — the `XDG_RUNTIME_DIR` fallback no longer adopts a pre-existing `/tmp`
  directory of any ownership or mode.
- #1905 — macOS sample output no longer uses a predictable `/tmp` path.
- #1906 — `NeuronDiscoveryHistory` deserialisation now rejects
  `successes > attempts`.

## Verify this record

```bash
git diff b85a551ed2521ed327469b20eb88aeda828357d2..HEAD -- \
  src/discovery_cleanup.rs src/debug.rs src/debug/sample_capture.rs \
  src/debug/sample_dir.rs src/debug/process_state.rs src/watchdog.rs \
  src/tracking_alloc.rs src/discovery_history.rs
```

An empty diff means this record still describes the current code.

On 2026-10-05 this diff is not empty — `git diff --stat
b85a551ed2521ed327469b20eb88aeda828357d2..HEAD -- src/discovery_cleanup.rs
src/debug.rs src/debug/sample_capture.rs src/debug/sample_dir.rs
src/debug/process_state.rs src/watchdog.rs src/tracking_alloc.rs
src/discovery_history.rs` shows three files touched — but the conclusions
above still hold. `src/discovery_cleanup.rs` changed for the #2255 and #2256
fixes: the orphan sweep now threads the `Path` itself through
`remove_discovery_dir`, and `LOCK_FILE_NAMES` is probed with
`fs::symlink_metadata` for both lock spellings rather than one, still
fail-closed. `src/debug/sample_capture.rs` changed for the #2266 fix:
`write_manual_hint` now prints `"$(mktemp -d)/sample.txt"` instead of a
predictable path. `src/discovery_history.rs` carries only a test-only hunk —
the #2253 regression tests — with no production code changed. None of these
hunks weakens a guard this record relies on.
