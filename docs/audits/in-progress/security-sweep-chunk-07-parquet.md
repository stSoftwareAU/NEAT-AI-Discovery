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
- **Swept by:** Issue #2279 (the chunk-7 staging sub-issue of #2215); the sweep
  itself is pending #2143 and #2144.
- **Tracker issue:** `#2091` (chunk 7), under the overflow tracker `#2083`.

### Citation convention — a deliberate departure

`CONTRIBUTING.md` → "Cite Code by Symbol, Never by Line Number" (Issue #1942)
prefers symbols. This record cites `file:line` **alongside** the symbol because
it is pinned to one baseline commit and carries the command that falsifies it
(see *Verify this record*): a citation here is a claim about `b85a551`, not
about HEAD. Where a file has drifted since (see #1048 below), the lines are
those of `b85a551`. Issue #2279 explicitly requires the `File:line` column and the guard and comment line numbers, so the departure is required, not elective.

### Sweep status — IN PROGRESS

This is the **staged** record: the header, the verified related remediations and
the decode-entry-point inventory are filled; the sweep itself is not. #2143
(read side) and #2144 (write side and cache) fill the sections marked
`pending`. The file lives in `docs/audits/in-progress/` on purpose, because
`tests/issue_2088_sweep_ledger_contract.rs` scans only the top level of
`docs/audits/` for `security-sweep-chunk-*.md` records and would reject a record
whose index entry still reads `last_swept: null`. #2145 later runs `git mv` to
the top level and flips chunk `7` in `lib-sweep-coverage.json`. Until then the
index entry keeps `last_swept`, `baseline_commit` and `record` as `null`, and
**nothing here counts as sweep coverage**.

## Files swept

Line counts as at the baseline commit `b85a551`. The Outcome column is
`pending` until #2143 / #2144 land.

| Path | Lines | Outcome |
| --- | --- | --- |
| `src/parquet_format/mod.rs` | 393 | pending #2143 |
| `src/parquet_format/footer.rs` | 106 | pending #2143 |
| `src/parquet_format/schema.rs` | 95 | pending #2143 |
| `src/parquet_format/decode_budget.rs` | 135 | pending #2143 |
| `src/parquet_format/reader.rs` | 558 | pending #2143 |
| `src/parquet_format/batch_columns.rs` | 324 | pending #2143 |
| `src/parquet_format/writer.rs` | 596 | pending #2144 |
| `src/parquet_format/shared_records.rs` | 276 | pending #2144 |
| `src/parquet_format/activation_summary.rs` | 159 | pending #2143 |
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

pending #2143 (read side) / #2144 (write side and cache). Not yet filled: until
those land, this record states nothing about which defect classes were looked
for.

## Outcome

pending #2143 / #2144. No per-file outcome is recorded yet, and no `src/parquet_format`
file is claimed clean.

## Issues filed

pending #2143 / #2144. Write `negative-result` here when a sweep finds nothing.

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
