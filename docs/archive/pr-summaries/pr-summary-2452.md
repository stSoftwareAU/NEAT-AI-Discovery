## Summary

The shared-decode cache key `(path, dev/ino, len, mtime_nanos)` did not change when a file was rewritten in place (same inode, same length) and its mtime restored, so a stale decode was served. The key now also carries `ctime` on unix, and `record_discovery_data` calls `invalidate()` once the Parquet file is finalised. Closes #2452.

Version bump: `Cargo.toml` and the `neat_ai_discovery` entry in `Cargo.lock` bumped from `0.74.325` to `0.74.326`.

## Spec

### Intent and Rationale

- `ctime` cannot be set by a user, so `cp -p`, `rsync --inplace -t` and `touch -r` can no longer make a rewritten file reproduce its old key.
- `invalidate()` after `finish` means the cycle that re-records a path never serves its previous decode, whatever the filesystem timestamps say.

### Essential Design Decisions

- `ctime` is `Option<(i64, i64)>` (secs, nsec): `None` on non-unix, mirroring the existing `dev_ino` helper.
- Residual accepted and documented in the module docs: a same-length rewrite inside one coarse filesystem timestamp tick is not distinguished. Truncating and rewriting a recording takes far longer than a tick on supported filesystems.
- Reading the Parquet footer would not help, since a same-shape rewrite changes only the data pages.

### Undiscoverable Facts

- The issue decided the fifth component should be `ctime` and that the same-tick residual is accepted.
- `docs/audits/security-sweep-chunk-07-parquet.md` is a point-in-time audit record and was deliberately left unchanged.

## Evidence

Backend-only change; no UI files touched. Tests in `tests/issue_2452_shared_cache_inplace_rewrite.rs`:

- `inplace_rewrite_with_restored_mtime_is_not_a_cache_hit` — loads a 1-record file (activation 1.0), rewrites it in place with the same shape (activation 9.0), asserts inode, length and restored mtime are unchanged and ctime moved, then asserts the second load is not `Arc::ptr_eq` to the first and returns 9.0. It failed on the unfixed code (`an in-place rewrite with a restored mtime must not be served from the cache`) and passes after the fix.
- `record_discovery_data_invalidates_the_shared_decode_cache` — records, loads the file into the cache (`decodes_for_path == 1`), re-records, asserts the slot is gone (`decodes_for_path == 0`). It failed on the unfixed code (`left: 1, right: 0`).

## Reproduction

- **symptom** — the second `load_grouped_records_shared` returned the first `Arc` with activation 1.0 while the file held 9.0.
- **status** — `verified` — both regression tests were observed failing against the unfixed code and passing after the fix.
- **regression test** — `tests/issue_2452_shared_cache_inplace_rewrite.rs::inplace_rewrite_with_restored_mtime_is_not_a_cache_hit`

## Test Plan

- Added `tests/issue_2452_shared_cache_inplace_rewrite.rs` (two tests above). Passed on the head along with `tests/issue_1907_shared_records_cache_key.rs`, `tests/issue_1406_shared_parquet_decode.rs`, `tests/issue_2067_record_module_docs.rs` and `shared_records::tests` (lib).
- Branch outcomes: `ctime` operand of `CacheKey` equality — reached by `inplace_rewrite_with_restored_mtime_is_not_a_cache_hit` (red without the field); `invalidate()` call in `record_discovery_data` — reached by `record_discovery_data_invalidates_the_shared_decode_cache` (red without the call).
- No existing test was edited, so no assertion was removed.
- Callers checked: `CacheKey::for_path` is used only by `load_shared_with_decoder` (both the pre-decode and post-decode stat), so the new component applies to both ends of the #1907 window.
- Docs: module docs, `CacheKey` doc, `load_grouped_records_shared` doc and `invalidate` doc updated; README, `docs/CACHE_TUNING.md`, `docs/CONFIGURATION.md` and `docs/FFI_API.md` do not list the key.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- Not checked: non-unix builds (the `ctime` helper falls back to `None` there).

Original trigger closed: an in-place rewrite that restores mtime still changes `ctime`, which no user-space call can set, and the re-recording path additionally drops the slot explicitly, so the reported input no longer yields a stale decode; only the documented same-tick case remains.
