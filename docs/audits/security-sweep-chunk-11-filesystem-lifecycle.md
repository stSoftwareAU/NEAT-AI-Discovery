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
  across audit sub-issues of #2117, #2118, #2119 and #2120; scaffolded by
  Issue #2233.
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
| `src/discovery_cleanup.rs` | 798 | pending |

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

<!-- section: debug + sampler -->

<!-- section: watchdog + tracking_alloc + discovery_history -->

## Re-verified remediations

| Issue | Guard | Citing site (`file.rs::symbol`) | On live path? |
| --- | --- | --- | --- |

<!-- section: discovery_cleanup -->

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
