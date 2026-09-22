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
| 1 | `free_discovery_result` | `src/ffi/mod.rs::free_discovery_result` | `*mut c_char` | yes (`ptr.is_null()`) | yes | yes |
| 2 | `cancel_analysis` | `src/ffi/analysis.rs::cancel_analysis` | none | n/a | **no** | yes |
| 3 | `cancel_analysis_memory_pressure` | `src/ffi/analysis.rs::cancel_analysis_memory_pressure` | none | n/a | **no** | yes |
| 4 | `reset_cancellation` | `src/ffi/analysis.rs::reset_cancellation` | none | n/a | **no** | yes |
| 5 | `is_analysis_active` | `src/ffi/analysis.rs::is_analysis_active` | none | n/a | **no** | yes |
| 6 | `rank_focus_neurons` | `src/ffi/analysis.rs::rank_focus_neurons` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 7 | `analyze_parallel` | `src/ffi/analysis.rs::analyze_parallel` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 8 | `check_gpu_available` | `src/ffi/gpu.rs::check_gpu_available` | none | n/a | yes | yes |
| 9 | `record_discovery` | `src/ffi/recording.rs::record_discovery` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 10 | `start_discovery_session` | `src/ffi/recording.rs::start_discovery_session` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 11 | `append_discovery_records` | `src/ffi/recording.rs::append_discovery_records` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 12 | `finish_discovery_session` | `src/ffi/recording.rs::finish_discovery_session` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 13 | `cancel_discovery_session` | `src/ffi/recording.rs::cancel_discovery_session` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 14 | `merge_discovery_parquet` | `src/ffi/utilities.rs::merge_discovery_parquet` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 15 | `read_discovery_records_ffi` | `src/ffi/utilities.rs::read_discovery_records_ffi` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 16 | `export_visualisation_snapshot` | `src/ffi/utilities.rs::export_visualisation_snapshot` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 17 | `get_calibration_summary` | `src/ffi/utilities.rs::get_calibration_summary` | `*const c_char` | yes (`validate_c_str_input_with_fields`) | yes | yes |
| 18 | `discovery_memory_usage_bytes` | `src/ffi/utilities.rs::discovery_memory_usage_bytes` | none | n/a | yes | yes |
| 19 | `cleanup_discovery_lib` | `src/ffi/utilities.rs::cleanup_discovery_lib` | none | n/a | yes | yes |
| 20 | `cleanup_discovery_dir` | `src/ffi/utilities.rs::cleanup_discovery_dir` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 21 | `clean_orphaned_discovery_dirs` | `src/ffi/utilities.rs::clean_orphaned_discovery_dirs` | `*const c_char` | yes (`validate_c_str_input`) | yes | yes |
| 22 | `get_library_version` | `src/ffi/utilities.rs::get_library_version` | none | n/a | yes | yes |

Every pointer-taking entry point routes its input through the single shared
guard in `src/ffi/helpers.rs` (`validate_c_str_input`, Issue #2045), which
rejects null and **rejects** invalid UTF-8 rather than converting lossily. No
entry point takes a caller-supplied length, so there is no pointer/length pair
to mis-trust and no length cast to overflow.

## Outcome

Two defects, both fixed under this issue; nothing else survived triage.

### SEC-2089-01 — the FFI panic response was not valid JSON

`src/ffi/helpers.rs::panic_to_ffi_json` is the last-resort error channel of
every `extern "C"` entry point: `catch_unwind` hands it the caught payload
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

**Attacker model:** any party who can drive the crate into a panic. Exposure is
internal, so that is the Deno host itself plus whatever reaches it — a
malformed recording, a corrupt Parquet file, a wedged GPU, or a host-installed
`tracing` layer that faults. No privilege is needed beyond making one FFI call.

**Trigger:** a panic anywhere inside a pointer-returning entry point whose
payload contains a control character. `assert!` / `assert_eq!` is the common
case: its message embeds `\n` unconditionally.

**Exploit sketch:** not a memory-safety or disclosure exploit — an availability
and diagnosability one. The host's `JSON.parse` of the response throws, so the
structured `errorKind` / `retryable` fields it branches on never arrive. A
caller that treats an unparsable response as a transport fault retries a
deterministic panic indefinitely, and the panic message naming the real cause
is discarded rather than logged, so the fault is invisible in triage. This is
the FFI boundary's own fail-loud channel failing silently.

**Fix:** serialise the response through `serde_json` so the escaping is total,
keeping the non-panicking fallbacks intact.

**Regression test:**
`tests/issue_2089_panic_response_json.rs::a_panic_caught_at_the_ffi_boundary_reaches_the_host_as_parsable_json`.
It drives the panic through the shipped entry point — a host-installed
`tracing` subscriber that panics, unwinding out of the analysis path into
`src/ffi/analysis.rs::rank_focus_neurons`'s own `catch_unwind` — per the #1806
convention, rather than calling the crate-private formatter. Observed failing
against the unfixed formatter with the serde control-character error above, and
passing after the fix.

### SEC-2089-02 — four `extern "C"` exports had no unwind guard

`cancel_analysis`, `cancel_analysis_memory_pressure`, `reset_cancellation` and
`is_analysis_active` (all four in `src/ffi/analysis.rs`) called
straight into `crate::cancellation::*` with no `panic::catch_unwind`. An unwind
out of an `extern "C"` function terminates the process; the host cannot catch
it, so a panic on these paths takes the Deno process down rather than returning
a verdict.

No panic is reachable on these paths today — three are pure atomic stores and
loads, and the two cancellation calls additionally emit a `tracing` event that
dispatches into whatever subscriber is installed — so this is the same
defence-in-depth the house already applies to `cleanup_discovery_lib`
(`src/ffi/utilities.rs::cleanup_discovery_lib`), which is wrapped despite being
equally unlikely to panic. It is recorded here as a defect because the issue's stated standard
is that *every* `extern "C"` function carries the wrapper.

**Attacker model:** the Deno host, or anything that can fault a component these
four paths touch — a host-installed `tracing` subscriber is the only
non-atomic thing on them. Internal exposure; one FFI call is the whole
interaction.

**Trigger:** any panic raised inside these four calls. None is reachable today
(see below), which is why this is graded defence-in-depth.

**Exploit sketch:** the host process terminates instead of receiving a verdict.
There is no partial-result path and no error channel on a `void` export, so the
failure mode is a hard stop of the discovery run — recoverable only by restart,
with in-flight recording state left behind.

`is_analysis_active` answers `1` ("active") rather than `0` if its guard ever
fires: the host uses that verdict to decide whether deleting the Parquet temp
directory is safe, so the fail-safe answer is the one that makes it wait. This
rests on the crate keeping the default unwind panic strategy — `Cargo.toml`
sets no `panic = "abort"`, under which `catch_unwind` never returns and every
wrapper here, old and new, would be inert.

**Test:** `tests/infrastructure/issue_2089_ffi_unwind_guard.rs` pins that each
guarded entry point still delegates — a wrapper that silently turned one into a
no-op fails loudly there. It is a behaviour pin, **not** a red/green regression
test: with no reachable panic there is no input that fails against the unfixed
code, and a test that forced one would abort the test binary rather than report
a failure.

### Examined and found clean

- **Ownership contract.** Exactly one free entry point
  (`src/ffi/mod.rs::free_discovery_result`); it is null-safe and
  unwind-guarded. Every `*mut c_char` handed back originates from
  `CString::into_raw` in `to_ffi_json`, `ffi_error_literal` or
  `panic_to_ffi_json`. Double-free and use-after-free remain caller
  obligations, correctly stated in the `# Safety` block — they are not
  preventable from this side of a C boundary.
- **`ffi_error_literal` returning null.** The documented last resort
  (`src/ffi/helpers.rs::ffi_error_literal`) is only ever called with
  compile-time literals,
  which cannot contain an interior NUL, so the null branch is unreachable;
  `free_discovery_result` is null-safe regardless.
- **Panic-payload bounds.** `truncate_panic_msg` caps the embedded message at
  4 KiB on a UTF-8 char boundary (Issue #1365), so a large payload cannot drive
  an unbounded allocation in the error path.
- **Integer handling.** The only casts are
  `crate::ALLOCATOR.allocated() as u64`
  (`src/ffi/utilities.rs::discovery_memory_usage_bytes`, a widening or identity
  cast, never a truncating one) and
  `start.elapsed().as_millis().min(u64::MAX as u128) as u64`
  (`src/ffi_internal/analysis.rs::rank_focus_neurons_internal`, explicitly
  clamped first). Every
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
  `src/ffi_internal/recording.rs::record_discovery_internal`,
  `src/ffi_internal/analysis.rs::analyze_parallel_internal` and
  `::rank_focus_neurons_internal`,
  `src/ffi_internal/utilities.rs::export_visualisation_snapshot_internal`, and
  `src/ffi/recording.rs::start_discovery_session` for the streaming session,
  which captures the creature for the lifetime of its append/finish calls.
- Tmp-directory handling (#1904, #1905) and the `cleanup_discovery_dir` path
  guard (#1866) — enforced inside `discovery_cleanup`, reached only through
  `src/ffi/utilities.rs::cleanup_discovery_dir` and
  `::clean_orphaned_discovery_dirs`, which pass the caller path through
  unmodified and classify an `InvalidInput` rejection as non-retryable.
- Streaming lifecycle (#1902) — session ids are opaque keys looked up in
  `streaming`; the FFI layer neither constructs nor interprets them.
- Parquet decode budget (#1869) — enforced in `parquet_format`, below this
  boundary.

## Issues filed

`negative-result` for *open* issues: both findings were remediated inside this
sweep, so neither survived triage as an issue to file. The deliverable's
"one issue per surviving finding" rule has no surviving finding to apply to;
each is recorded above with the same evidence a filed issue would carry.

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
