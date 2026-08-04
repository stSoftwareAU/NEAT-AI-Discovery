# Documentation Archive

This directory contains archived documentation that is retained for historical
reference.

## Contents

- **[pr-summaries/](pr-summaries/)** — Historical PR summary files documenting
  changes, evidence, and test plans for each pull request.

## What goes where (Issue #1941)

Three documentation tiers, three different obligations. Putting a document in
the wrong tier is how a superseded fact ends up being read as current.

| Tier | Role | Obligation when the facts change |
|------|------|----------------------------------|
| `docs/*.md` | **Live reference** — the single source of truth an operator or agent reads to learn current behaviour. | Rewrite in place. A live document must never state a superseded fact, and must never contradict itself. |
| `docs/analysis/*.md` | **Point-in-time studies** — a diagnosis or measurement as at one commit. Written once, cited afterwards. | Do **not** rewrite. Add a dated "as at" header and annotate each superseded claim inline with the issue that closed it, so a later reader can still follow the original reasoning. |
| `docs/archive/pr-summaries/*.md` | **Transient** — one PR's evidence and test plan. | Fold anything of lasting value into the live reference, then delete the summary — capture is the precondition for deletion, never the other way round (see [pr-summaries/README.md](pr-summaries/README.md)). |

A study in `docs/analysis/` is often cited by name in PR summaries long after it
was written, so an un-annotated superseded claim there propagates further than
one in a PR summary. Annotate on the change that supersedes it, not later.

## Direction of fix — wire it, do not delete it (Issue #1937)

When a live document describes a capability the code does not honour, the doc is
not automatically the side that is wrong. A half-wired documented capability gets
**wired, not deleted**. Decide by asking whether the capability is **absent** or
merely **half-wired**:

- **Half-wired** — the behaviour is implemented and tested, and only one call
  site fails to forward it. Wire the missing site; the documented behaviour
  becomes true. `docs/FFI_API.md` promised `includeSynapseAnalysis` /
  `includeNeuronAnalysis`; the phases really did honour both flags and only
  `AnalyzeParallelInput` lacked the fields, so serde discarded the caller's keys.
  Adding the fields kept a working capability on the published surface — deleting
  the section would have removed it.
- **Absent** — nothing implements the behaviour and nothing can emit it (a type
  with no `Serialize`, a module wired into nothing). Correct the doc.

Deleting documented behaviour is the fallback, not the reflex.
