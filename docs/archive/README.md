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
| `docs/archive/pr-summaries/*.md` | **Transient** — one PR's evidence and test plan. | Fold anything of lasting value into the live reference, then leave the summary alone; it is a record of one change, not a reference. |

A study in `docs/analysis/` is often cited by name in PR summaries long after it
was written, so an un-annotated superseded claim there propagates further than
one in a PR summary. Annotate on the change that supersedes it, not later.
