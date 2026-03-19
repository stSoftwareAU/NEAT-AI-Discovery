## Summary

Converted all ASCII art diagrams in core documentation to Mermaid diagrams and
applied styling with emojis and colour-coded sections. Closes #842.

### Changes

- **README.md**: Converted the Discovery → Evolution Pipeline ASCII art diagram
  to a Mermaid flowchart with colour-coded stages (blue for Rust, orange for
  TypeScript, green for Evolution). Added emojis to all section headings.
- **docs/discoveries/README.md**: Converted the Traditional NEAT vs NEAT with
  Discovery comparison ASCII art to a side-by-side Mermaid flowchart (red for
  traditional, green for discovery). Converted the 4-step pipeline ASCII art to
  a Mermaid flowchart with four colour-coded stages. Replaced the ASCII
  production success rates table with a proper Markdown table. Added emojis to
  all section headings.
- **CONTRIBUTING.md**: Added emojis to all section headings for visual
  consistency with the other core docs.
- Australian English used throughout all modified files.

## Evidence

- No ASCII art remains in README.md, CONTRIBUTING.md, or docs/discoveries/README.md
- All diagrams use Mermaid format with `style` directives for colour coding
- `quality.sh` passes cleanly

## Test Plan

- Documentation-only changes; no code or tests modified
- Verified no ASCII box-drawing characters remain in the three target files
- `quality.sh` passes with all 124 tests passing
