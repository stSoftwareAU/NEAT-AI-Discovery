# Security sweep — chunk `<NN>`: `<human name>`

Copy this file to `docs/audits/security-sweep-chunk-<NN>-<slug>.md`, fill every
field, and update the matching entry in `docs/audits/lib-sweep-coverage.json` in
the same commit. Rules: [`README.md`](README.md).

## Record

- **Chunk id:** `<2 / 8a / 16 / …>` — must match the `id` in the index.
- **Sweep date:** `<YYYY-MM-DD>`
- **Baseline commit:** `<full SHA of the commit read>` — a record with no
  commit SHA is worthless.
- **Exposure:** `<internal / local / network>`
- **Swept by:** `<who or which run>`
- **Tracker issue:** `<#NNNN>`

## Files swept

Line counts as at the baseline commit.

| Path | Lines | Outcome |
| --- | --- | --- |
| `<src/path/file.rs>` | `<N>` | `<clean / defect — see #NNNN>` |

## Defect classes probed

List what was actually looked for, so a later reader knows what this sweep does
**not** cover.

- `<e.g. unchecked FFI pointer/length pairs>`
- `<e.g. path traversal in caller-supplied filenames>`
- `<e.g. integer overflow in size arithmetic>`

## Outcome

`<One paragraph: what was found overall, and what was deliberately left out of
scope.>`

## Issues filed

- `<#NNNN — one line each>`

Write `negative-result` when the sweep found nothing. An empty section is
indistinguishable from an unfinished record.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- `<#NNNN — what it fixed>`

## Verify this record

```bash
git diff <baseline_commit>..HEAD -- <paths swept>
```

An empty diff means this record still describes the current code.
