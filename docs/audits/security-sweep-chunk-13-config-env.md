# Security sweep — chunk `13`: Config / env-var surface — src/config

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `13`
- **Sweep date:** `2026-10-28`
- **Baseline commit:** `b85a551ed2521ed327469b20eb88aeda828357d2`
- **Exposure:** `local`
- **Swept by:** Issue #2224 (scaffold for chunk 13 of the #2083 overflow tracker; regions filled by later sub-issues)
- **Tracker issue:** #2096 (parent #2121)

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
| `src/config/user_facing.rs` | 732 | pending |

The `user_facing.rs` split is by line: L1–664 is row 13b and L665–1396 at baseline is row 13c.

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

| variable | reader file:line | type | default | accepted range / validation | on invalid input | 0 / opt-out semantics | can weaken a guard? (which) | cached (OnceLock) or live | logged how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

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

- pending — no findings recorded yet

**user_facing L665-end (13c)**

<!-- section: 13c -->

- pending — no findings recorded yet

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

- pending — no findings recorded yet

**Readers outside src/config (13d)**

<!-- section: 13d -->

- pending — no findings recorded yet

## Outcome

The sweep is in progress. Outcomes are pending per region: 13a small modules, 13b `user_facing.rs` L1–664, 13c `user_facing.rs` L665–end, and 13d readers outside `src/config`. No region has a recorded outcome yet.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a sweep and never justify a non-null `last_swept`.

- #1385 — Reconcile focus-ranking budget clamp: code doesn't clamp but README claims [1000, 3600000]
- #1408 — Reserve a guaranteed minimum analysis budget so focus selection + parquet load cannot starve synapse/neuron analysis
- #1420 — Low-memory gate disables discovery on ~8 GB hosts — make the available-memory floor configurable / reclaim-aware
- #1869 — Parquet pre-load bound is estimated from compressed size and only checked after the allocation (src/analysis/utils/memory.rs:402-405)
- #1873 — Safe GPU setup fns assert an unenforceable thread-free invariant around unsafe env writes
- #1904 — XDG_RUNTIME_DIR fallback adopts a pre-existing /tmp directory of any ownership or mode (src/analysis/utils/platform.rs:163-183)
- #2006 — Numeric env-override parse chain duplicated across 35 accessors with inconsistent trimming

## Verify this record

```bash
git diff b85a551ed2521ed327469b20eb88aeda828357d2..HEAD -- src/config
```

An empty diff means this record still describes the current code. The diff is not empty at HEAD: see the re-check note under Record.
