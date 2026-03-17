## Summary

Moved 89 PR summary files (`pr-summary-607.md` through `pr-summary-838.md`) from
`docs/` to `docs/archive/pr-summaries/` to consolidate all PR summaries in a
single archive location. Closes #840.

The archive now contains 251 PR summaries (PR #164 through #838), keeping the
main `docs/` directory clean and navigable.

No broken links were found — the only external reference (in `CONTRIBUTING.md`)
already points to the archive path.

## Evidence

- Verified zero `pr-summary-*.md` files remain in `docs/`
- Verified 251 PR summaries consolidated in `docs/archive/pr-summaries/`
- No documentation links reference the old `docs/pr-summary-*.md` paths

## Test Plan

- `quality.sh` passes (documentation-only change, no code affected)
