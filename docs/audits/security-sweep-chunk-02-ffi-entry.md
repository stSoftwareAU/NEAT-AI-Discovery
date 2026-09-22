# Security sweep — chunk `2`: FFI entry points (`src/ffi` + `src/ffi_internal`)

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `2` — matches the `id` in the index.
- **Human name:** FFI entry points — `src/ffi` + `src/ffi_internal`.
- **Sweep date:** `2026-09-22`
- **Baseline commit:** `a2440e746d64b163e432c7521bef00b7e58db1ea`
  — `git diff b85a551..a2440e7 -- src/ffi src/ffi_internal` is empty, so this
  record also describes the `b85a551` baseline the tracker issue named.
- **Exposure:** `internal` — the crate ships as a `cdylib`/`rlib` driven over
  Deno FFI. These 22 `extern "C"` symbols are the only surface a caller reaches
  directly.
- **Swept by:** Issue #2089 (chunk 2 of the #2083 overflow tracker).
- **Tracker issue:** `#2083`

## Files swept

3,849 lines, read in full. Line counts as at the baseline commit.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/ffi/mod.rs` | 51 | clean — single null-safe, unwind-guarded free entry point |
| `src/ffi/analysis.rs` | 224 | defect — 4 entry points had no `catch_unwind`; fixed in this sweep |
| `src/ffi/utilities.rs` | 571 | clean |
| `src/ffi/gpu.rs` | 41 | clean |
| `src/ffi/recording.rs` | 447 | clean |
| `src/ffi/helpers.rs` | 362 | defect — panic response was not valid JSON; fixed in this sweep |
| `src/ffi_internal/mod.rs` | 377 | clean — re-exports plus `#[cfg(test)]` only |
| `src/ffi_internal/analysis.rs` | 1209 | clean |
| `src/ffi_internal/utilities.rs` | 241 | clean |
| `src/ffi_internal/gpu.rs` | 229 | clean |
| `src/ffi_internal/recording.rs` | 97 | clean |

## Defect classes probed

- **Raw pointer handling** — every `*const c_char` / `*mut` parameter: null
  check before deref, and whether each `SAFETY` comment states a condition the
  caller is required and able to honour.
- **UTF-8 / `CStr` decoding** — `CStr::from_ptr` on a non-UTF-8 buffer; lossy
  conversion that truncates rather than rejecting.
- **Ownership contract** — exactly one free entry point, null-safety of that
  free, reachability of double-free / use-after-free from a plausible caller
  mistake.
- **Unwind safety** — `panic::catch_unwind` present on *each* `extern "C"`
  entry point, and the wrapper itself unable to panic (payload formatting,
  error-string allocation).
- **Integer handling** — `as` casts of caller-supplied lengths/counts, `usize`
  truncation on 32-bit, capacity arithmetic overflow.
- **Error paths** — leaked allocations, absolute filesystem paths or internal
  state echoed into caller-visible error strings.

**Not covered by this sweep:** the business logic the `*_internal` functions
delegate into (`analysis::analyze_all`, `record::*`, `streaming::*`,
`parquet_format::*`, `focus::*`, `export::*`) — those are chunks 7, 8a, 8b, 9
and 11. This sweep read the boundary, not what lies behind it.

## `extern "C"` entry-point enumeration

All 22 exports in `src/ffi/`. "As found" is the baseline state; "after" is the
state this sweep leaves behind.

| # | Entry point | Location | Pointer params | Null-checked | `catch_unwind` as found | `catch_unwind` after |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `free_discovery_result` | `src/ffi/mod.rs:38` | `*mut c_char` | yes (`ptr.is_null()`) | yes | yes |
| 2 | `cancel_analysis` | `src/ffi/analysis.rs:24` | none | n/a | **no** | yes |
| 3 | `cancel_analysis_memory_pressure` | `src/ffi/analysis.rs:47` | none | n/a | **no** | yes |
| 4 | `reset_cancellation` | `src/ffi/analysis.rs:64` | none | n/a | **no** | yes |
| 5 | `is_analysis_active` | `src/ffi/analysis.rs:88` | none | n/a | **no** | yes |
| 6 | `rank_focus_neurons` | `src/ffi/analysis.rs:106` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 7 | `analyze_parallel` | `src/ffi/analysis.rs:174` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 8 | `check_gpu_available` | `src/ffi/gpu.rs:8` | none | n/a | yes | yes |
| 9 | `record_discovery` | `src/ffi/recording.rs:19` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 10 | `start_discovery_session` | `src/ffi/recording.rs:106` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 11 | `append_discovery_records` | `src/ffi/recording.rs:215` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 12 | `finish_discovery_session` | `src/ffi/recording.rs:305` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 13 | `cancel_discovery_session` | `src/ffi/recording.rs:393` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 14 | `merge_discovery_parquet` | `src/ffi/utilities.rs:20` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 15 | `read_discovery_records_ffi` | `src/ffi/utilities.rs:75` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 16 | `export_visualisation_snapshot` | `src/ffi/utilities.rs:152` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 17 | `get_calibration_summary` | `src/ffi/utilities.rs:230` | `*const c_char` | yes (`validate_c_str_input_with_fields`) | yes | yes |
| 18 | `discovery_memory_usage_bytes` | `src/ffi/utilities.rs:288` | none | n/a | yes | yes |
| 19 | `cleanup_discovery_lib` | `src/ffi/utilities.rs:311` | none | n/a | yes | yes |
| 20 | `cleanup_discovery_dir` | `src/ffi/utilities.rs:346` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 21 | `clean_orphaned_discovery_dirs` | `src/ffi/utilities.rs:451` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 22 | `get_library_version` | `src/ffi/utilities.rs:538` | none | n/a | yes | yes |

Every pointer-taking entry point routes its input through the single shared
guard in `src/ffi/helpers.rs` (`validate_c_str_input`, Issue #2045), which
rejects null and **rejects** invalid UTF-8 rather than converting lossily. No
entry point takes a caller-supplied length, so there is no pointer/length pair
to mis-trust and no length cast to overflow.

## Outcome

Two defects, both fixed under this issue; nothing else survived triage.

### SEC-2089-01 — the FFI panic response was not valid JSON

`panic_to_ffi_json` (`src/ffi/helpers.rs:130`) is the last-resort error channel
of every `extern "C"` entry point: `catch_unwind` hands it the caught payload
and its output is what the host receives. It built the response by
string-formatting and escaped only `\` and `"`:

```rust
let error_json = format!(
    "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
    msg.replace('\\', "\\\\").replace('"', "\\\"")
);
```

JSON forbids raw control characters inside a string, and every `assert!` /
`assert_eq!` panic message carries newlines — so the response for the most
common panic shape was unparsable. `serde_json` rejects it with
*"control character (`\u0000`-`\u001F`) found while parsing a string"*. The
documented `{"success":false,"error":…}` contract that `AGENTS.md` requires
every controller to branch on therefore degraded into unparsable text at
exactly the moment a caller needs it, so a panicking analysis pass surfaced to
the host as a JSON parse error with the real cause discarded — a silent failure
of the boundary's own fail-loud channel.

**Fix:** serialise the response through `serde_json` so the escaping is total,
keeping the non-panicking fallbacks intact.

**Regression test:**
`tests/ffi/issue_2089_panic_response_json.rs::assertion_style_panic_message_yields_parsable_json`
(plus five siblings). Observed failing against the unfixed formatter with the
serde control-character error above, and passing after the fix.

### SEC-2089-02 — four `extern "C"` exports had no unwind guard

`cancel_analysis`, `cancel_analysis_memory_pressure`, `reset_cancellation` and
`is_analysis_active` (`src/ffi/analysis.rs:24`, `:47`, `:64`, `:88`) called
straight into `crate::cancellation::*` with no `panic::catch_unwind`. An unwind
out of an `extern "C"` function terminates the process; the host cannot catch
it, so a panic on these paths takes the Deno process down rather than returning
a verdict.

No panic is reachable on these paths today — three are pure atomic stores and
loads, and the two cancellation calls additionally emit a `tracing` event that
dispatches into whatever subscriber is installed — so this is the same
defence-in-depth the house already applies to `cleanup_discovery_lib`
(`src/ffi/utilities.rs:311`), which is wrapped despite being equally unlikely
to panic. It is recorded here as a defect because the issue's stated standard
is that *every* `extern "C"` function carries the wrapper.

`is_analysis_active` answers `1` ("active") rather than `0` if its guard ever
fires: the host uses that verdict to decide whether deleting the Parquet temp
directory is safe, so the fail-safe answer is the one that makes it wait.

**Test:** `tests/infrastructure/issue_2089_ffi_unwind_guard.rs` pins that each
guarded entry point still delegates — a wrapper that silently turned one into a
no-op fails loudly there. It is a behaviour pin, **not** a red/green regression
test: with no reachable panic there is no input that fails against the unfixed
code, and a test that forced one would abort the test binary rather than report
a failure.

### Examined and found clean

- **Ownership contract.** Exactly one free entry point
  (`free_discovery_result`, `src/ffi/mod.rs:38`); it is null-safe and
  unwind-guarded. Every `*mut c_char` handed back originates from
  `CString::into_raw` in `to_ffi_json`, `ffi_error_literal` or
  `panic_to_ffi_json`. Double-free and use-after-free remain caller
  obligations, correctly stated in the `# Safety` block — they are not
  preventable from this side of a C boundary.
- **`ffi_error_literal` returning null.** The documented last resort
  (`src/ffi/helpers.rs:36`) is only ever called with compile-time literals,
  which cannot contain an interior NUL, so the null branch is unreachable;
  `free_discovery_result` is null-safe regardless.
- **Panic-payload bounds.** `truncate_panic_msg` caps the embedded message at
  4 KiB on a UTF-8 char boundary (Issue #1365), so a large payload cannot drive
  an unbounded allocation in the error path.
- **Integer handling.** The only casts are
  `crate::ALLOCATOR.allocated() as u64` (`src/ffi/utilities.rs:293`, a widening
  or identity cast, never a truncating one) and
  `start.elapsed().as_millis().min(u64::MAX as u128) as u64`
  (`src/ffi_internal/analysis.rs:778`, explicitly clamped first). Every
  `usize → u32` conversion in `src/ffi_internal/analysis.rs` uses
  `u32::try_from(..).unwrap_or(u32::MAX)` and every count uses
  `saturating_add`.
- **Error paths.** Caller-visible error strings echo only paths the caller
  itself supplied (`tempDir`, `baseDir`, `parquetFile`, `outFile`) plus serde's
  own parse diagnostics. No internal path, environment value or host state is
  disclosed.
- **Allocation leaks on error paths.** Every early return from a pointer-taking
  entry point returns an owned `*mut c_char`; the guard's error pointer is
  returned directly rather than dropped, so no branch leaks or double-allocates.
- **`src/ffi_internal/*`.** These take `&str` and return `Result<String>`; they
  hold no raw pointers, no `unsafe`, and no `extern "C"` symbol. They honour
  their documented contract — caller-input failures come back as a
  `success: false` payload, `Err` is reserved for a failure to serialise the
  response — and each analysis/record/export path that accepts a creature calls
  `validate_creature` before any business logic, as `AGENTS.md` requires.

### Out of scope, confirmed not bypassable on this surface

The guards named in the issue were re-checked from the entry point inwards and
none is bypassable here, so none is refiled:

- Creature input/output bounds (#1867, #2020, #2078) — `validate_creature` runs
  before business logic on every creature-accepting path:
  `src/ffi_internal/recording.rs:52`, `src/ffi_internal/analysis.rs:74` and
  `:607`, `src/ffi_internal/utilities.rs:116`, and `src/ffi/recording.rs:146`
  for the streaming session, which captures the creature for the lifetime of
  its append/finish calls.
- Tmp-directory handling (#1904, #1905) and the `cleanup_discovery_dir` path
  guard (#1866) — enforced inside `discovery_cleanup`, reached only through
  `src/ffi/utilities.rs:346`/`:451`, which pass the caller path through
  unmodified and classify an `InvalidInput` rejection as non-retryable.
- Streaming lifecycle (#1902) — session ids are opaque keys looked up in
  `streaming`; the FFI layer neither constructs nor interprets them.
- Parquet decode budget (#1869) — enforced in `parquet_format`, below this
  boundary.

## Issues filed

Both findings were fixed under #2089 itself, so neither survived triage as an
open issue:

- `SEC-2089-01` — invalid-JSON panic response — fixed in this sweep's PR.
- `SEC-2089-02` — four unguarded `extern "C"` exports — fixed in this sweep's PR.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- `#2045` — the shared null / invalid-UTF-8 input guard every pointer-taking
  entry point now routes through.
- `#1365` — the 4 KiB cap on a panic message embedded in an FFI error response.
- `#772` — the `to_ffi_json` / `ffi_error_literal` helpers that removed
  `unwrap()` from the boundary.
- `#711` — `# Safety` markers on the `unsafe extern "C"` entry points.
- `#1866` — the discovery-directory path allowlist behind `cleanup_discovery_dir`.

## Verify this record

```bash
git diff a2440e746d64b163e432c7521bef00b7e58db1ea..HEAD -- src/ffi src/ffi_internal
```

An empty diff means this record still describes the current code. The two fixes
above land on top of that baseline, so the first non-empty diff is expected to
be exactly them.
