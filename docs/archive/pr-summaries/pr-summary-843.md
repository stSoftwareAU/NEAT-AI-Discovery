## Summary

Convert all ASCII art diagrams in `docs/` reference guides to Mermaid diagrams and apply
emoji/styling to section headings across all 6 target files. Closes #843.

### Changes

- **IMPACT_CALCULATION.md**: Converted 8 ASCII art network diagrams (simple chain,
  branching network, multiple outputs, deep network, STEP/BIPOLAR functions, threshold
  crossing, MINIMUM/MAXIMUM selection, activation-weighted impact) to coloured Mermaid
  `graph LR` and `sequenceDiagram` diagrams with styled nodes
- **DISCOVERY_TYPES.md**: Converted 2 ASCII art diagrams (Rust/TypeScript workflow and
  redundant path pruning example) to Mermaid diagrams; added emojis to all major section
  headings
- **FFI_API.md**: Converted the streaming API workflow ASCII art to a Mermaid sequence
  diagram; added emojis to all section headings
- **GPU_GUIDE.md**: Added emojis to all section headings (no ASCII art found)
- **ANALYSIS_DEEP_DIVE.md**: Added emojis to all section headings (no ASCII art found)
- **BENCHMARKS.md**: Added emojis to all section headings (no ASCII art found)

### Acceptance Criteria

- [x] No ASCII art box-drawing diagrams remain in any `docs/*.md` reference file
- [x] All diagrams are Mermaid format and render correctly on GitHub
- [x] Diagrams are visually appealing and factually accurate
- [x] Emojis and styling used appropriately
- [x] Australian English throughout
- [x] `quality.sh` passes

## Evidence

- `quality.sh` passes with all 124 tests passing
- Grep for box-drawing characters (`┌┐└┘├┤┬┴┼`) confirms zero matches in the 6 target files

## Test Plan

- Documentation-only changes; no code modifications
- Verified `quality.sh` passes (includes cargo build, fmt, clippy, test, doc build)
- Visual verification of Mermaid diagrams by reviewing rendered markdown
