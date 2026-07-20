# Archived PR Summaries

This directory contains historical PR summary files (`pr-summary-*.md`) that
document the changes, evidence, and test plans for each pull request.

These files were consolidated here from `docs/` and `docs/archive/` to keep the
main documentation directory clean and navigable. They are not actively
maintained.

## Retention rule — fold, then delete (Issue #1682)

A PR summary is **not** kept indefinitely. It is retained only until its durable
learnings have been folded into the live docs; once captured there, the summary
is deleted. Capture is the **precondition** for deletion — never delete a summary
whose learning has not yet landed in a live doc.

- **Durable learnings** — anything an agent would otherwise re-derive: measured
  trade-offs and benchmark numbers, upgrade traps, design decisions, and
  especially **negative results** (approaches proven fruitless). Negative results
  are the learnings most at risk, because nothing in the code fails when they are
  lost — the project just silently re-spends the effort.
- **Fold each learning into the live doc that owns the topic** — e.g.
  optimisation outcomes and benchmark trade-offs into
  [`docs/BENCHMARKS.md`](../../BENCHMARKS.md); GPU-stack traps into
  [`docs/GPU_GUIDE.md`](../../GPU_GUIDE.md); quality-gate traps into
  [`AGENTS.md`](../../../AGENTS.md).
- **No negative result may be dropped.** If a summary records a negative result,
  fold it before deleting the summary.
- **Then delete** the summary. The live-doc entry is now the single source of
  truth.

New PR summaries continue to be created at
`docs/archive/pr-summaries/pr-summary-<ISSUE>.md` as part of each pull request
(see [CONTRIBUTING.md](../../../CONTRIBUTING.md) for details).
