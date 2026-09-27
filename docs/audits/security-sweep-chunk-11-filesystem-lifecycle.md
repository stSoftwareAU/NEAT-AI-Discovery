# Security sweep — chunk `11`: Filesystem lifecycle

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `11` — matches the `id` in the index.
- **Human name:** Filesystem lifecycle — `src/discovery_cleanup.rs`,
  `src/debug.rs` + `src/debug/`, `src/watchdog.rs`, `src/tracking_alloc.rs`,
  `src/discovery_history.rs`.
- **Sweep date:** `2026-09-27`
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2`
- **Exposure:** `local`
- **Swept by:** Issue #2095 (chunk 11 of the #2083 overflow tracker), split
  across its audit sub-issues #2117 (11a), #2118 (11b), #2119 (11c) and #2120
  (11d); scaffolded by Issue #2233.
- **Tracker issue:** `#2095`

### Sweep status — IN PROGRESS

This record is a scaffold. Every file row below reads `pending` until its
owning audit sub-issue sweeps it; the index's `last_swept` date marks when the
scaffold was cut, not a finished sweep. Each sub-issue edits only its own `###`
section and its own marked region of the two finding tables below, so
concurrent PRs do not conflict.

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
| `src/debug.rs` | 604 | no finding — no filesystem mutation and no process spawn (a grep for `fs::`, `File::`, `OpenOptions`, `DirBuilder`, `Command::new`, `.exists()` and `remove_dir_all` has no hits); SIGUSR1 is only deliverable by the same uid or root, and dumps are serialised on one thread (signal/lifecycle probe) |
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
  fallback, steering the operator to exactly the predictable path in a
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
| `src/watchdog.rs` | 364 | pending |
| `src/tracking_alloc.rs` | 234 | pending |
| `src/discovery_history.rs` | 626 | pending |

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
<!-- section: discovery_cleanup -->
| `discovery_cleanup.rs::assert_is_discovery_dir` | `path.join(LOCK_FILE_NAME).exists()` and `path.join(DISCOVERY_DATA_FILE_NAME).exists()` — read-only gate | caller-supplied `temp_dir` after the empty and `..` checks, or a swept `<base_dir>/<child>` via `cleanup_orphaned_discovery_dir` | follows symlinks — yes, read-only: a followed link can only admit a path, and `remove_discovery_dir` refuses a symlinked `temp_dir` next |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::symlink_metadata(path)` | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` | yes — does not follow the final component; a symlink is refused (`InvalidInput`) and so is a non-directory. Intermediate components resolve, which probe (a) accepts for a caller-owned root |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::canonicalize(path)` — audit log only | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` | follows symlinks — yes, log only: the resolved path is never acted on, and an error falls back to the raw path |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::remove_dir_all(path)` | caller-supplied `temp_dir`, or a swept `<base_dir>/<child>`, after `assert_is_discovery_dir` and the `symlink_metadata` refusal | yes — std does not follow a top-level symlink nor symlinks within the tree (probe b); only intermediate components resolve (probe a) |
| `discovery_cleanup.rs::is_directory_orphaned` | `fs::symlink_metadata(dir.join(LOCK_FILE_NAME))` | a swept child of `base_dir`, or `temp_dir` during the #1903 re-check | yes — the lock itself is not followed; fails closed on any error but `NotFound` (probe d) |
| `discovery_cleanup.rs::directory_touched_since` | `fs::symlink_metadata(dir)?.modified()` | a swept child of `base_dir` after the marker gate | yes — reads the entry's own mtime, not a link target's; errors propagate |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.exists()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a): the root is caller-owned |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.is_dir()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `fs::read_dir(base_path)` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a); yields single-component names only (probe c) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `entry.file_type()` | a child of `base_dir` | yes — `DirEntry::file_type` does not follow; a symlinked child is skipped |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `path.is_dir()` | a child of `base_dir`, already known not to be a symlink | follows, but only after the `file_type()` symlink skip; a later swap is caught by `remove_discovery_dir`'s re-probe (probe b) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `cleanup_orphaned_discovery_dir(&path.display().to_string())` | a child of `base_dir` | **no — finding #2255**: the lossy string can name a different, literal-U+FFFD sibling that skipped the age floor |

<!-- section: debug + sampler -->
| `sample_dir.rs::create_private_dir` | `DirBuilder` with `mode(0o700)` and `recursive(false)`, then `set_permissions(0o700)` (unix); a mode-less non-recursive `create` elsewhere | `std::env::temp_dir()` joined with the per-invocation name from `SampleDir::create` | yes — `mkdir` does not follow the final component, so a planted symlink or any existing entry fails with `AlreadyExists`; `set_permissions` runs only on the directory this call just created |
| `sample_dir.rs::SampleDir::create` | up to `MAX_CREATE_ATTEMPTS` (8) calls to `create_private_dir` on `neat_ai_discovery.sample.<pid>.<nanos>.<seq>.<attempt>.d` | `std::env::temp_dir()` | yes — every attempt is exclusive; a co-tenant squatting all eight names only denies the capture, which is reported (WARNING, manual hint, `NoBacktraces`) |
| `sample_dir.rs::Drop::drop` | `fs::remove_dir_all(dir)`; `NotFound` ignored, any other error printed as a WARNING | the directory `SampleDir::create` made | yes — std does not follow a top-level symlink nor symlinks within the tree, and the directory is euid-owned `0700`, so no other uid can plant inside it |
| `sample_dir.rs::read_guarded` | `fs::symlink_metadata(path)`, refuse a non-regular file (`InvalidData`) or a uid other than the euid (`PermissionDenied`), then `fs::read_to_string(path)` | `<SampleDir>/sample.txt` | yes — the final component is not followed; the stat→read TOCTOU is refuted because the parent is the euid-owned `0700` directory |
| `sample_capture.rs::run_external_command_with_timeout` | process spawn: `Command::new(program)` with the argv `[pid, "1", "-mayDie", "-file", <capture path>]` and null stdout/stderr; killed after 5 s plus a 500 ms grace | the sampler writes `<SampleDir>/sample.txt` | yes — no shell; the output path is inside the private directory, and the program comes only from the process's own env or `PATH` (same uid) |

<!-- section: watchdog + tracking_alloc + discovery_history -->

## Re-verified remediations

| Issue | Guard | Citing site (`file.rs::symbol`) | On live path? |
| --- | --- | --- | --- |
<!-- section: discovery_cleanup -->
| #1903 | lock re-read immediately before removal (`LockRecheck::Enforce` → `CleanupOutcome::Claimed`); fail-closed lock probe (only `NotFound` means "no lock"); age floor (a child touched at or after `scan_started` is counted `claimed`) | `discovery_cleanup.rs::remove_discovery_dir` (re-check), `discovery_cleanup.rs::is_directory_orphaned` (fail-closed probe), `discovery_cleanup.rs::directory_touched_since` and the age-floor match in `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since`; regression surface `tests/issue_1903_orphan_sweep_lock_recheck.rs` | yes — FFI `ffi/utilities.rs::clean_orphaned_discovery_dirs` → `discovery_cleanup.rs::clean_orphaned_discovery_dirs` → `clean_orphaned_discovery_dirs_since` → `cleanup_orphaned_discovery_dir` (`LockRecheck::Enforce`). Holds for UTF-8 names; #2255 bypasses the age floor for a non-UTF-8 entry, and #2256 means the guard keys on a name the known host never writes |

<!-- section: debug + sampler -->
| #1905 | owner-only per-invocation capture directory removed on `Drop` on every exit path, including kill-on-timeout; a symlink or foreign-owned file at the capture path is refused, not followed; guard test `tests/issue_1905_sample_temp_dir.rs` | `sample_dir.rs::SampleDir::create`, `sample_dir.rs::create_private_dir`, `sample_dir.rs::Drop::drop`, `sample_dir.rs::read_guarded` (via `sample_capture.rs::read_capture`); regression surface for the fallback text `tests/issue_1934_sample_fallback.rs` | yes — SIGUSR1 → `debug.rs::dump_all_threads` → `debug.rs::render_thread_dump` → `sample_capture.rs::capture`. The manual fallback hint still names `/tmp/sample.txt` (#2266) |
| #1904 | `RUNTIME_DIR_MODE` (`0700`) and `prepare_runtime_dir` in `src/analysis/utils/platform.rs`: create owner-only, re-apply the mode past the umask, refuse a pre-existing world-writable directory or a symlink | `platform.rs::prepare_runtime_dir`, pinned by its `xdg_runtime_dir_*` unit tests; the sampler only mirrors the pattern in `sample_dir.rs::create_private_dir` and does not call it | yes on Linux (`platform.rs::ensure_xdg_runtime_dir`); not on the sampler path — no `src/debug*` file references it |

<!-- section: watchdog + tracking_alloc + discovery_history -->

## Outcome

Pending — written by the finalisation sub-issue, #2120.

## Issues filed

Placeholder — this list is finalised by #2120.

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
