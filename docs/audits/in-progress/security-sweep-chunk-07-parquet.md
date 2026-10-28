# Security sweep — chunk `7`: Parquet reader/writer/schema (`src/parquet_format`)

Ledger rules: [`README.md`](../README.md). Index entry:
[`lib-sweep-coverage.json`](../lib-sweep-coverage.json) (chunk `7`).

## Record

- **Chunk id:** `7` — matches the `id` in the index.
- **Human name:** Parquet reader/writer/schema.
- **Sweep date:** `2026-10-28` (date this staged record was written; the sweep
  date proper is set by #2145 when the sweep completes).
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2` — every line
  count and `file:line` citation below is a claim about this commit.
  `git diff b85a551..HEAD -- src/parquet_format` is empty at the time of
  writing.
- **Exposure:** `internal` — as recorded in the index.
- **Swept by:** Issue #2279 staged this record. The read side (`footer.rs`, `schema.rs`, `decode_budget.rs`, `reader.rs`, `batch_columns.rs`, `activation_summary.rs`) was swept by Issue #2143; the write side and cache (`mod.rs`, `writer.rs`, `shared_records.rs`) is pending #2144.
- **Tracker issue:** `#2091` (chunk 7), under the overflow tracker `#2083`.

### Citation convention — a deliberate departure

`CONTRIBUTING.md` → "Cite Code by Symbol, Never by Line Number" (Issue #1942)
prefers symbols. This record cites `file:line` **alongside** the symbol because
it is pinned to one baseline commit and carries the command that falsifies it
(see *Verify this record*): a citation here is a claim about `b85a551`, not
about HEAD. Where a file has drifted since (see #1048 below), the lines are
those of `b85a551`. Issue #2279 explicitly requires the `File:line` column and the guard and comment line numbers, so the departure is required, not elective.

### Sweep status — IN PROGRESS

This is a **partly swept** record: the header, the verified related remediations and
the decode-entry-point inventory are filled, and #2143 has filled the read-side rows, probes, outcome and issues. #2144 (write side and cache) fills what is still marked `pending #2144`. The file lives in `docs/audits/in-progress/` on purpose, because
`tests/issue_2088_sweep_ledger_contract.rs` scans only the top level of
`docs/audits/` for `security-sweep-chunk-*.md` records and would reject a record
whose index entry still reads `last_swept: null`. #2145 later runs `git mv` to
the top level and flips chunk `7` in `lib-sweep-coverage.json`. Until then the
index entry keeps `last_swept`, `baseline_commit` and `record` as `null`, and
**nothing here counts as sweep coverage**.

## Files swept

Line counts as at the baseline commit `b85a551`. The Outcome column is `<outcome> — <reason>`; the three write-side rows stay `pending #2144`.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/parquet_format/mod.rs` | 393 | pending #2144 |
| `src/parquet_format/footer.rs` | 106 | finding #2447 — negative footer num_rows / total_byte_size / num_values read as 0 (footer.rs L79, L83, L98) and the file row count is not cross-checked against the row groups; the missing .parquet.tmp rejection (L73) affects no caller |
| `src/parquet_format/schema.rs` | 95 | clean — validate_neuron_uuid was probed at 0, 1, 100 and 101 bytes and multi-byte UTF-8 at the boundary; truncate_utf8 is reached only for the 120-byte error preview (it truncates nothing at those lengths); the "characters" wording reports bytes (cosmetic, no outcome changes) |
| `src/parquet_format/decode_budget.rs` | 135 | finding #2446 — the budget is charged only after the batch is materialised, so the module doc's "before the allocation that would exhaust the host completes" (L7-8) is false at batch granularity; the arithmetic itself is saturating and clean |
| `src/parquet_format/reader.rs` | 558 | finding #2446 — build() at L315-318, L411-414 and L508-511 has no footer pre-check and no batch bound; charge_record (L370, L435, L552) runs after decode_row |
| `src/parquet_format/batch_columns.rs` | 324 | clean — by-name column resolution, ErrorLists offset checks (L58-77) and NaN/inf pass-through probed; no panic or wrong value |
| `src/parquet_format/writer.rs` | 596 | pending #2144 |
| `src/parquet_format/shared_records.rs` | 276 | pending #2144 |
| `src/parquet_format/activation_summary.rs` | 159 | finding #2446 — no decode budget and no batch bound, so one hostile string column is materialised unchecked (L93-141); non-finite activations are correctly skipped (L134) |
| **Total** | **2,642** | |

## Decode entry points

Every function in `src/parquet_format/` that decodes a Parquet file, or that
the decode paths hand off to, at `b85a551`. `Y (via <callee>)` means the check
is inherited from the named callee. Where a wrapper passes `None`, the cell
says so.

| Function | File:line | Visibility | Schema validated (profile) | Decode budget | Cancellation | Deadline |
| --- | --- | --- | --- | --- | --- | --- |
| `read_all_records_grouped_by_neuron` | `reader.rs:237` | `pub` | Y (via `read_all_records_grouped_by_neuron_with_deadline`, which reaches `_bounded`; profile `Full`) | Y (via `_bounded`; `budget_mb` is `None`, so `resolve` falls back to the env or half-RAM ceiling) | Y (via `_bounded`) | N (passes deadline `None` to `_with_deadline`) |
| `read_all_records_grouped_by_neuron_with_profile` | `reader.rs:245` | `pub` | Y (via `_with_deadline_and_profile`, which reaches `_bounded`; caller's profile) | Y (via `_bounded`; `budget_mb` `None`) | Y (via `_bounded`) | N (passes deadline `None` to `_with_deadline_and_profile`) |
| `read_all_records_grouped_by_neuron_with_deadline` | `reader.rs:259` | `pub` | Y (via `_with_deadline_and_profile`; profile `Full`) | Y (via `_bounded`; `budget_mb` `None`) | Y (via `_bounded`) | Y (via `_bounded`; caller's deadline forwarded) |
| `read_all_records_grouped_by_neuron_with_deadline_and_profile` | `reader.rs:276` | `pub` | Y (via `_bounded`; caller's profile) | Y (via `_bounded`; passes `budget_mb` `None`) | Y (via `_bounded`) | Y (via `_bounded`; caller's deadline forwarded) |
| `read_all_records_grouped_by_neuron_bounded` | `reader.rs:294` | `pub` | Y (`validate_parquet_schema` at L312, caller's profile) | Y (`DecodeBudget::resolve` at L323, `charge_record` at L370; caller's `budget_mb`) | Y (`is_cancelled` at L328, each batch) | Y (L300-304 before opening; L338-350 each batch) |
| `read_records_from_parquet` | `reader.rs:389` | `pub` | Y (via `read_records_from_parquet_with_profile`; profile `Full`) | Y (via `_with_profile`; `resolve(None)`) | N (no cancellation check on the flat path) | N (no deadline parameter) |
| `read_records_from_parquet_with_profile` | `reader.rs:398` | `pub` | Y (`validate_parquet_schema` at L408, caller's profile) | Y (`resolve(None)` at L419, `charge_record` at L435) | N (no cancellation check) | N (no deadline parameter) |
| `read_all_records_from_parquet` | `reader.rs:448` | `pub` | Y (via `read_records_from_parquet_with_limit`, then `_with_limit_and_budget`; profile `Full`) | Y (via `_with_limit_and_budget`; `budget_mb` `None`) | N (no cancellation check) | N (no deadline parameter) |
| `read_records_from_parquet_with_limit` | `reader.rs:466` | `pub` | Y (via `_with_limit_and_profile`, then `_with_limit_and_budget`; profile `Full`) | Y (via `_with_limit_and_budget`; `budget_mb` `None`) | N (no cancellation check) | N (no deadline parameter) |
| `read_records_from_parquet_with_limit_and_profile` | `reader.rs:475` | `pub` | Y (via `_with_limit_and_budget`; caller's profile) | Y (via `_with_limit_and_budget`; passes `budget_mb` `None`) | N (no cancellation check) | N (no deadline parameter) |
| `read_records_from_parquet_with_limit_and_budget` | `reader.rs:489` | `pub` | Y (`validate_parquet_schema` at L505, caller's profile; not reached when `max_obs == Some(0)` returns an empty `Vec` early) | Y (`resolve(budget_mb)` at L517, `charge_record` at L552; not reached on the `max_obs == Some(0)` early return) | N (no cancellation check) | N (no deadline parameter) |
| `open_parquet_file` | `reader.rs:80` | `pub(crate)` | N (opens the file only; rejects a `.parquet.tmp` suffix, schema checked by the caller) | N (opens only, decodes nothing) | N (no cancellation check) | N (no deadline parameter) |
| `validate_parquet_schema` | `reader.rs:140` | `pub(crate)` | Y (this is the check; takes the `profile`) | N (reads schema metadata only, decodes no rows) | N (no cancellation check) | N (no deadline parameter) |
| `read_mean_abs_activation_by_neuron` | `activation_summary.rs:68` | `pub` | Y (`validate_parquet_schema` at L90, profile `ActivationOnly`; skipped when `wanted` is empty, which returns an empty map at L73-75 before opening the file) | N (no `DecodeBudget`; accumulators are keyed by the caller's `wanted` set) | N (no cancellation check) | Y (L77 before opening, L101 each batch) |
| `read_footer_stats` | `footer.rs:72` | `pub` | N (metadata only: reads the footer, never decodes rows) | N (metadata only; feeds `projected_in_memory_bytes` for callers) | N (metadata only) | N (metadata only) |
| `merge_into_temp_file` | `writer.rs:363` | private | Y (`validate_parquet_schema` at L377, profile `Full`, per input file) | N (no `DecodeBudget`) | N (no cancellation check) | N (no deadline parameter) |
| `merge_parquet_files` | `writer.rs:411` | `pub` | Y (via `merge_into_temp_file`; profile `Full`) | N (inherits `merge_into_temp_file`, which has none) | N (inherits `merge_into_temp_file`, which has none) | N (inherits `merge_into_temp_file`, which has none) |
| `load_grouped_records_shared` | `shared_records.rs:112` | `pub` | Y on a miss (via `load_grouped_records_shared_with_budget`, which reaches `_bounded`; profile `Full`); a hit decodes nothing | Y on a miss (via `_with_budget`; passes `budget_mb` `None`) | Y on a miss (via `_bounded`) | Y on a miss (caller's deadline forwarded); a hit decodes nothing |
| `load_grouped_records_shared_with_budget` | `shared_records.rs:124` | `pub` | Y on a miss (calls `_bounded` with profile `Full`); a hit decodes nothing | Y on a miss (caller's `budget_mb` forwarded to `_bounded`) | Y on a miss (via `_bounded`) | Y on a miss (caller's deadline forwarded); a hit decodes nothing |
| `load_shared_with_decoder` | `shared_records.rs:141` | private | Y on a miss (inherits the decoder closure; production passes `_bounded` with `Full`); a hit decodes nothing | Y on a miss (inherits the decoder closure) | Y on a miss (inherits the decoder closure) | Y on a miss (inherits the decoder closure) |
| `DiscoveryBatchColumns::resolve` | `batch_columns.rs:113` | `pub(crate)` | N (not a file entry point: resolves columns of an already-built `RecordBatch`; a missing or wrongly typed column is a recoverable error, and callers validate the file schema first) | N (not a file entry point; the caller charges the budget) | N (not a file entry point; the caller checks) | N (not a file entry point; the caller checks) |
| `DiscoveryBatchColumns::decode_row` | `batch_columns.rs:145` | `pub(crate)` | N (not a file entry point: decodes one row of an already-built batch; the caller validated the schema) | N (not a file entry point; the caller calls `charge_record` per decoded row) | N (not a file entry point; the caller checks) | N (not a file entry point; the caller checks) |

Out of the table: the writer builders (`write_records_to_parquet`,
`write_records_to_parquet_with_limit`, `ParquetRecordWriter`), `schema.rs` and
`mod.rs`. None of them decodes a Parquet file: the builders only encode
in-memory records, `schema.rs` defines the Arrow schema and `mod.rs` declares and re-exports the sub-modules (plus in-crate tests).

## Defect classes probed

Probes use hand-built Parquet files written with the `write_custom_parquet` / `fields_with` / `empty_errors_column` / `path_in` pattern of `tests/issue_1901_parquet_schema_nullability.rs`; probes that found nothing are recorded here and not committed; findings carry their regression test in the finding's issue.

### Header-declared sizes

| File:line | Input | Outcome |
| --- | --- | --- |
| `footer.rs:79` | valid 50-row file, footer `FileMetaData.num_rows` patched 50 to -1 | `read_footer_stats` returns `num_rows: 0` (the `unwrap_or(0)`), the reader still decodes 50 rows — finding #2447 |
| `footer.rs:79` | same file, `num_rows` patched to 1 | stats report `num_rows: 1`, the reader decodes 50 rows; the file count is not cross-checked against the row groups — finding #2447 |
| `footer.rs:83`, `footer.rs:98` | `total_byte_size` / `num_values` | same `u64::try_from(..).unwrap_or(0)` shape, covered by static reading and by #2447 (not patched separately) |
| `footer.rs:73` | the same file renamed `*.parquet.tmp` | `read_footer_stats` returns stats while `open_parquet_file` (reader.rs:82) rejects it; the only caller is `src/analysis/utils/memory.rs:420`, which treats an error as a logged fallback, so no caller is affected — no finding |
| `footer.rs:75` | footer truncated by 20 bytes | `Corrupt footer` error — clean |
| `memory.rs:420-421` + `decode_budget.rs:91` | understated footer | the projection is `max(file_size x 3, footer projection)`; `charge_record` still bounds the decode per record (subject to #2446) — weakened estimate, not a bypass |

### Decode-budget completeness

| File:line | Input | Outcome |
| --- | --- | --- |
| `reader.rs:370`, `:435`, `:552` | every reader | each charges the record before it is inserted or pushed (grouped reader: charge at L370, insert at L375-378) — clean |
| `activation_summary.rs:129-140` | no `charge_record` | memory is O(wanted neurons), accumulators only — clean for retained memory; batch materialisation falls under the next row |
| `reader.rs:315-318` (batch before charge) | one-row file, `errors` list of 20,000,000 zeros: 1,643 bytes on disk, 80,000,000 bytes decoded, read with a 1 MB budget | `MemoryExhausted` as designed, but the process high-water mark (`VmHWM`, reset through `/proc/self/clear_refs`) grew by 240,586,752 bytes before the first `charge_record` — finding #2446 |
| `reader.rs:314-318` | answer to the external-crate question | yes: the `parquet` crate (60.0.0, `Cargo.lock`) builds the whole `RecordBatch` before the first `charge_record`, with a default of 1,024 rows per batch; `with_batch_size` bounds rows but not one huge row, and the crate has no per-list cap, so a crate-level fix is out of scope. The repo can still refuse from the footer already held by the builder, which #2446 asks for. The batch is built when the reader is iterated (reader.rs L325 and L360), not inside `build()`; the lines cited are where the reader is configured. |
| `reader.rs:370` (under-charge) | 300,000 rows with distinct 36-byte UUIDs read with a 1,000,000 MB budget; 300,000 rows with one UUID | retained `VmRSS` growth 47,673,344 bytes against a 30,000,000-byte estimate (1.59x), and 35,864,576 bytes (1.20x) for one UUID. Coarse (RSS, allocator slack), within the documented "floor" and the half-of-RAM ceiling — no finding |
| `decode_budget.rs:43-49` | unknown total RAM | unbounded only when the total is 0; `get_memory_info` defaults to 8 GB on both Linux (`memory.rs:280`) and macOS (`memory.rs:139`), so 0 is not reachable in practice (it would need a literal zero from `/proc/meminfo` or `sysctl`) — clean |

### Cancellation / deadline

| File:line | Input | Outcome |
| --- | --- | --- |
| Flat readers (`read_records_from_parquet*`, `_with_limit*`) check neither cancellation nor deadline; `read_mean_abs_activation_by_neuron` checks the deadline only | callers traced: `src/ffi_internal/utilities.rs:202` (host-supplied `parquet_file`), `src/analysis/cache/lru_cache.rs:147` and `:190`, `compressed_cache.rs:132`, `cache/mod.rs:220`, `src/export/snapshot.rs:35`, `src/streaming.rs:876` (a file the process just wrote), `src/focus/ranking/record_providers.rs:125`; cancellable/deadline-aware: `cache/mod.rs:264`, `focus/ranking/mod.rs:553` and `:605`, `focus/gradient.rs:402` (cancellation only), `focus/ranking/activation_weighting.rs:118` | every flat read is one linear scan bounded by the file and the decode budget, so none is unbounded in the unterminated sense; the scan cost of a hostile file is the #2446 shape. No separate finding |

### Column pruning vs schema trust

| File:line | Input | Outcome |
| --- | --- | --- |
| `reader.rs:186-229` | extra sixth column | accepted, the projection ignores it — clean |
| `reader.rs:188-201` | columns reordered (`neuron_uuid` first) | rejected: expected `obs_index` at index 0 — clean |
| `reader.rs:204-216` | `errors` as Utf8, read under `WithoutErrors` and `ActivationOnly` | accepted (not projected, never decoded); under `Full` rejected with a type error — clean by design |
| `reader.rs:204-216` | `obs_index` as UInt64 | accepted under `ActivationOnly`, rejected under `Full` — clean by design |
| `reader.rs:117-125` | `errors` as `List<Float64>` | rejected under `Full`, accepted under `WithoutErrors` — clean |
| `batch_columns.rs:58-77` | negative or out-of-range list offsets | `offset_at` rejects a negative or missing offset and `values.get(start..end)` rejects a span past the child array; Arrow's `OffsetBuffer` refuses to construct non-monotonic or negative offsets and the parquet reader builds valid ones, so a malformed array cannot be built through the safe API — covered by static reading, not by a probe |
| `batch_columns.rs:259` | sliced batch | `a_sliced_batch_decodes_each_row_from_its_own_offsets` already covers it — clean |
| `batch_columns.rs:81-91` | missing or wrongly typed column | recoverable errors (unit tests at L275 and L299) — clean |

### NaN / infinity / null coercion

| File:line | Input | Outcome |
| --- | --- | --- |
| `batch_columns.rs:145-164` | `value` NaN, +inf, -inf and null; `activation` NaN, +inf, -inf; `errors` containing NaN, +inf, -inf | the reader passes every non-finite value through unchanged (`value: Some(NaN)`, `activation: NaN`, `errors: [NaN, 1.0]`) and null `value` decodes as `None`; null in any other column is rejected by schema validation (#1901) — no panic |
| `activation_summary.rs:134` | the same file | non-finite activations are skipped; the neuron whose only samples are non-finite is omitted, the neuron with one finite sample reports `mean_abs_activation: 2.0`, `sample_count: 1` — clean |
| Downstream | where decoded values land | `export_visualisation_snapshot` drops non-finite errors (`tests/export/export_visualisation_snapshot_non_finite_errors.rs`), `src/focus/selection.rs:111-113` clamps non-finite weights; detection and scoring consumers belong to the chunk-8a/8b sweeps. Not traced exhaustively here, and no non-finite value was found that changes a result or panics on this read side — no finding |

### UUID handling

| File:line | Input | Outcome |
| --- | --- | --- |
| `schema.rs:68-95` | writer-side check at 0, 1, 100 and 101 ASCII bytes | 0 rejected ("0 characters"), 1 and 100 accepted, 101 rejected ("101 characters") — clean |
| `schema.rs:46-66` | 34 x 3-byte (102 bytes), 33 x 3-byte (99 bytes), 99 ASCII + one 2-byte (101 bytes), 98 ASCII + one 2-byte (100 bytes) | the 102- and 101-byte ones are rejected on length, the 99- and 100-byte ones on the invalid-character rule; the length message calls bytes "characters" (a 34-character string reports "102 characters"). Cosmetic: the outcome is right, only the unit in the message is wrong — noted, not filed |
| `reader.rs` | foreign file with an empty, a 5,000-byte and a non-charset UUID | the reader does not apply `validate_neuron_uuid`: all three are accepted as group keys (the 5,000-byte key is charged by `charge_record`). The bounds are a write-side contract; no consumer was found that trusts them — no finding |

Write side (#2144): pending #2144.

## Outcome

Read side (#2143): `footer.rs` finding #2447; `decode_budget.rs`, `reader.rs` and `activation_summary.rs` finding #2446; `batch_columns.rs` and `schema.rs` clean. Write side and cache (#2144): pending #2144; `mod.rs`, `writer.rs` and `shared_records.rs` are not claimed clean.

## Issues filed

| Issue | Severity | Defect |
| --- | --- | --- |
| #2446 (SEC-2091-01) | medium | decode budget charged after the Arrow batch is materialised |
| #2447 (SEC-2091-02) | low | `read_footer_stats` maps negative footer sizes to 0 |

Write side and cache (#2144): pending #2144. Write `negative-result` there if that sweep finds nothing.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`. All lines are as at `b85a551`.

- **#1900** — merge output aliasing an input. Guard:
  `writer.rs::resolve_for_alias_check` L325-344 and `merge_parquet_files`
  L411-442. Comments citing the issue: L320 and L400.
- **#1901** — schema type and nullability check. Guard:
  `reader.rs::data_type_matches` L117-125 and `validate_parquet_schema`
  L140-232. Comments: L112, L132 and L203.
- **#1907** — cache key describes the bytes actually cached. Guard:
  `shared_records.rs::CacheKey::for_path` L62-76, and the re-stat in
  `load_shared_with_decoder` (L172-183). Comments: L17, L61 and L140.
- **#1869** — cumulative decode budget. Guard: `decode_budget.rs::resolve` L75
  and `charge_record` L91; `footer.rs::projected_in_memory_bytes` L53-64;
  `charge_record` calls in `reader.rs` at L370, L435 and L552. Comments in
  `reader.rs`: L285, L322, L418, L484 and L516.
- **#1406** — process-wide single-decode cache. Guard: `shared_records.rs::CACHE`
  L99, `load_grouped_records_shared` L112 and
  `load_grouped_records_shared_with_budget` L124. Comments: L1 and L205.
- **#1048** — no surviving code in `src/parquet_format`. The fix lives in
  `src/analysis/cache/`: `compressed_cache.rs` L46 and L58; `lru_cache.rs` L114
  and L126; `mod.rs` L95, L110, L211, L287 and L607 (lines as at `b85a551`;
  `src/analysis/cache/mod.rs` has since drifted, so verify against `b85a551`,
  not HEAD). `reader.rs` L326 cites **#1047** (cancellation at each batch
  boundary), not #1048.

## Verify this record

Run from the repository root. Every command reads `b85a551` directly, so the
result does not depend on HEAD.

```bash
# Line counts (Files swept) — expect 393 106 95 135 558 324 596 276 159, total 2,642
for f in mod footer schema decode_budget reader batch_columns writer shared_records activation_summary; do
  printf '%s ' "$f"; git show "b85a551:src/parquet_format/$f.rs" | wc -l
done

# Related remediations (comment lines)
for n in 1900 1901 1907 1869 1406 1047; do
  git grep -n "#$n" b85a551 -- src/parquet_format
done
git grep -n '#1048' b85a551 -- src/analysis/cache

# Decode entry points (definitions and checks)
git grep -n -e 'pub fn read_' -e 'fn merge_' -e 'fn load_' -e 'fn open_parquet_file' \
  -e 'fn validate_parquet_schema' b85a551 -- src/parquet_format
git grep -n -e 'validate_parquet_schema(' -e 'DecodeBudget::resolve' \
  -e 'charge_record(' -e 'is_cancelled' -e 'deadline' b85a551 -- \
  src/parquet_format/reader.rs src/parquet_format/activation_summary.rs \
  src/parquet_format/writer.rs src/parquet_format/shared_records.rs

# Guard definitions cited under Related remediations and the batch decoder
git grep -n -e 'fn resolve_for_alias_check' -e 'fn data_type_matches' \
  -e 'fn for_path' -e 'static CACHE' -e 'fn resolve' -e 'fn charge_record' \
  -e 'fn projected_in_memory_bytes' -e 'fn decode_row' b85a551 -- src/parquet_format

# Staleness — an empty diff means this record still describes the current code
git diff b85a551..HEAD -- src/parquet_format
```
