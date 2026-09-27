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
  documentation states it "does **not** follow symbolic links and will simply
  remove the symbolic link itself" at the top-level path, and does not follow
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
  the root survives; a `..`-climbing `temp_dir` is refused and its canary
  survives. The lossy case is a finding: the sweep passes
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
| `src/debug.rs` | 604 | pending |
| `src/debug/sample_capture.rs` | 509 | pending |
| `src/debug/sample_dir.rs` | 285 | pending |
| `src/debug/process_state.rs` | 150 | pending |

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
| `discovery_cleanup.rs::assert_is_discovery_dir` | `path.join(LOCK_FILE_NAME).exists()` and `path.join(DISCOVERY_DATA_FILE_NAME).exists()` — read-only gate | caller-supplied `temp_dir`, after the empty and `..` checks | follows symlinks — yes, read-only: a followed link can only admit a path, and `remove_discovery_dir` refuses a symlinked `temp_dir` next |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::symlink_metadata(path)` | caller-supplied `temp_dir` after `assert_is_discovery_dir` | yes — does not follow the final component; a symlink is refused (`InvalidInput`) and so is a non-directory. Intermediate components resolve, which probe (a) accepts for a caller-owned root |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::canonicalize(path)` — audit log only | caller-supplied `temp_dir` after `assert_is_discovery_dir` | follows symlinks — yes, log only: the resolved path is never acted on, and an error falls back to the raw path |
| `discovery_cleanup.rs::remove_discovery_dir` | `fs::remove_dir_all(path)` | caller-supplied `temp_dir` after `assert_is_discovery_dir` and the `symlink_metadata` refusal | yes — std does not follow a top-level symlink nor symlinks within the tree (probe b); only intermediate components resolve (probe a) |
| `discovery_cleanup.rs::is_directory_orphaned` | `fs::symlink_metadata(dir.join(LOCK_FILE_NAME))` | a swept child of `base_dir`, or `temp_dir` during the #1903 re-check | yes — the lock itself is not followed; fails closed on any error but `NotFound` (probe d) |
| `discovery_cleanup.rs::directory_touched_since` | `fs::symlink_metadata(dir)?.modified()` | a swept child of `base_dir` after the marker gate | yes — reads the entry's own mtime, not a link target's; errors propagate |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.exists()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a): the root is caller-owned |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `base_path.is_dir()` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `fs::read_dir(base_path)` | `base_dir` after the marker gate | follows a symlinked root — accepted, probe (a); yields single-component names only (probe c) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `entry.file_type()` | a child of `base_dir` | yes — `DirEntry::file_type` does not follow; a symlinked child is skipped |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `path.is_dir()` | a child of `base_dir`, already known not to be a symlink | follows, but only after the `file_type()` symlink skip; a later swap is caught by `remove_discovery_dir`'s re-probe (probe b) |
| `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since` | `cleanup_orphaned_discovery_dir(&path.display().to_string())` | a child of `base_dir` | **no — finding #2255**: the lossy string can name a different, literal-U+FFFD sibling that skipped the age floor |

<!-- section: debug + sampler -->

<!-- section: watchdog + tracking_alloc + discovery_history -->

## Re-verified remediations

| Issue | Guard | Citing site (`file.rs::symbol`) | On live path? |
| --- | --- | --- | --- |
<!-- section: discovery_cleanup -->
| #1903 | lock re-read immediately before removal (`LockRecheck::Enforce` → `CleanupOutcome::Claimed`); fail-closed lock probe (only `NotFound` means "no lock"); age floor (a child touched at or after `scan_started` is counted `claimed`) | `discovery_cleanup.rs::remove_discovery_dir` (re-check), `discovery_cleanup.rs::is_directory_orphaned` (fail-closed probe), `discovery_cleanup.rs::directory_touched_since` and the age-floor match in `discovery_cleanup.rs::clean_orphaned_discovery_dirs_since`; regression surface `tests/issue_1903_orphan_sweep_lock_recheck.rs` | yes — FFI `ffi/utilities.rs::clean_orphaned_discovery_dirs` → `discovery_cleanup.rs::clean_orphaned_discovery_dirs` → `clean_orphaned_discovery_dirs_since` → `cleanup_orphaned_discovery_dir` (`LockRecheck::Enforce`). Holds for UTF-8 names; #2255 bypasses the age floor for a non-UTF-8 entry, and #2256 means the guard keys on a name the known host never writes |

<!-- section: debug + sampler -->

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
