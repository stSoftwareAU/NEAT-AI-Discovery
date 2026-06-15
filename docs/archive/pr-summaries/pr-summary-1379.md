## Summary

Fixed two pre-existing Mermaid quality-gate failures tracked by the baseline
carryover tracker. Both diagrams began their fenced ` ```mermaid ` block with a
YAML frontmatter section (`--- ... ---`), which the Mermaid validator misreads
as the diagram type, reporting `Unknown Mermaid diagram type: ---`. Closes #1379.

The carried-over findings were:

- `docs/IMPACT_CALCULATION.md:157` — frontmatter setting a cosmetic
  `themeVariables.fontSize`. Removed the frontmatter so the block opens with the
  recognised `graph TD` diagram type. The font-size override was purely
  cosmetic and is not required for the diagram to render.
- `docs/discoveries/monotonicity.md:18` — frontmatter setting a `title`.
  Moved the title into the chart body using the inline `title "..."` directive
  that `xychart-beta` supports natively, so no information is lost and the block
  opens with the recognised `xychart-beta` diagram type.

## Evidence

This is a documentation-only change (no Rust or shell code touched), so there is
no UI or benchmark evidence. Verification was performed by scanning every
` ```mermaid ` block under `docs/` for blocks whose first non-blank content line
is `---`:

```text
=== remaining frontmatter mermaid blocks (should be none) ===
NONE FOUND - good
```

Before the fix, that scan reported the two carried-over findings:

```text
docs/IMPACT_CALCULATION.md:158: ---
docs/discoveries/monotonicity.md:19: ---
```

Both diagrams now begin with a valid diagram-type token (`graph TD` and
`xychart-beta` respectively), which is what the Mermaid validator keys on.

`quality.sh` was not run: it performs `cargo upgrade --incompatible` plus a full
release build, which would pull unrelated dependency bumps into a
documentation-only PR (violating change scope). No Rust or shell sources were
modified.

## Test Plan

- Scanned all `docs/**/*.md` Mermaid blocks for a leading `---` frontmatter line
  and confirmed zero remain (the same check that reproduced the two findings).
- Confirmed the two edited blocks now open with a recognised Mermaid diagram
  type and that the `xychart-beta` title is preserved via the inline `title`
  directive.
