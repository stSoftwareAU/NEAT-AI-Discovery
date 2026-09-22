# Security sweep ledger

This directory is the **sweep-coverage ledger**: the record of which parts of
this crate have actually been read for security defects, when, and against which
commit. Without it there is no way to tell a swept surface from an unswept one —
every overflow tracker restarts from zero, the same file gets swept twice, and
another is never swept at all.

A ledger entry answers one question and answers it falsifiably: *when was chunk
N last swept, and against which commit?*

## What lives here

| File | Purpose |
| --- | --- |
| `README.md` | These rules. |
| `lib-sweep-coverage.json` | Machine-readable index — one entry per chunk, so an automated run can answer the coverage question without reading prose. |
| `security-sweep-TEMPLATE.md` | The skeleton every per-chunk record starts from. |
| `security-sweep-chunk-<NN>-<slug>.md` | One prose record per swept chunk. |

## When a sweep must write to the ledger

Any run that reads a chunk for security defects — a `security-scan` run, a
chunk issue from the overflow tracker, or a manual audit — writes **both**:

1. a new `docs/audits/security-sweep-chunk-<NN>-<slug>.md` record, and
2. the matching entry update in `lib-sweep-coverage.json`.

A sweep that writes neither did not happen as far as the next run is concerned.
A sweep that writes only the prose record is invisible to automation; the parity
gate below rejects it.

## Required fields of a chunk record

Every record carries all of these. A record missing any of them cannot be
checked by a later reader and is not coverage:

- **Chunk id** — the id used in `lib-sweep-coverage.json` (`2`, `8a`, `16`, …).
- **Human name** — what the chunk actually is, in words.
- **File list** — each path swept, with its line count at the baseline commit.
- **Exposure** — `internal`, `local`, or `network`.
- **Baseline commit SHA** — the commit the sweep read.
- **Sweep date** — ISO `YYYY-MM-DD`.
- **Defect classes probed** — what was actually looked for.
- **Outcome per file** — what was found in each file, or explicitly nothing.
- **Issues filed** — the issue numbers raised, or `negative-result` when the
  sweep found nothing.

## One file per chunk — never a shared document

Records are written **one file per chunk**:
`docs/audits/security-sweep-chunk-<NN>-<slug>.md`, where `<NN>` is the
zero-padded chunk id including any letter suffix (`02`, `08a`, `13`).

Never append chunk records to a single shared document. The chunk issues run in
parallel, so a shared append-only file guarantees merge conflicts on every
concurrent sweep.

`lib-sweep-coverage.json` is the **only** shared file. Keep each chunk entry on
its own line with the keys in this order:

```text
id, name, exposure, issue, last_swept, baseline_commit, record
```

Two concurrent sweeps then conflict on one line rather than on the whole
document.

## A record with no commit SHA is worthless

The baseline commit SHA is what makes a record falsifiable. Without it a reader
cannot tell whether the swept code still exists, so "swept" degrades into an
unverifiable claim.

With it, anyone can check how far the sweep has drifted:

```bash
git diff <baseline_commit>..HEAD -- src/config
```

An empty diff means the sweep still describes the current code. A large diff
means the chunk needs re-sweeping, whatever the ledger says.

For the same reason, an entry with no `last_swept` date carries `null` for both
`baseline_commit` and `record`. An explicit "never recorded" is a datum; half a
claim is not.

## Prose and index must match both ways

`tests/issue_2088_sweep_ledger_contract.rs` enforces the ledger's integrity and
runs from both `./quality.sh` and `./scripts/doc-check.sh`. It fails when:

- `lib-sweep-coverage.json` does not parse, or an overflow chunk has no entry;
- a `security-sweep-chunk-*.md` record has no index entry — that record is
  invisible to the next automated run;
- an index entry names a record file that does not exist;
- a claimed sweep has no baseline commit SHA, or a null sweep date still claims
  a baseline commit or record;
- an entry drops a required key or spells the keys out of order.

## Prior remediation is not sweep coverage

The earlier remediation cycle (#1867, #1900–#1918, #2020, #2054, #2078) fixed
specific defects; it did not systematically read these chunks for new ones.
Closing an issue is not a sweep.

Individual records may list those issues under **Related remediations (not
sweep coverage)**, clearly separated from the sweep itself. They never justify a
non-null `last_swept`: only an actual sweep, pinned to the commit it read, does
that.
