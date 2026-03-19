## Summary

Convert all ASCII art diagrams to Mermaid and apply fun/informative styling (emojis, colours, Australian English) in the 10 Pruning and Data Quality discovery scenario guides. Closes #844.

### Files Updated

**Pruning Discoveries (7):**
1. `docs/discoveries/dead-neuron.md`
2. `docs/discoveries/dormant-synapse.md`
3. `docs/discoveries/opposing-synapse.md`
4. `docs/discoveries/redundant-path.md`
5. `docs/discoveries/remove-low-impact.md`
6. `docs/discoveries/co-adaptation.md`
7. `docs/discoveries/noise-signal.md`

**Data Quality Discoveries (3):**
8. `docs/discoveries/bounded-range.md`
9. `docs/discoveries/sentinel-gating.md`
10. `docs/discoveries/observation-utilisation.md`

### Changes Per File
- Converted all ASCII art network diagrams to Mermaid flowcharts and graph diagrams
- Applied colour-coded styling: red for problematic components, green for healthy/fixed, blue for inputs, purple for hidden neurons, orange for warnings/gates
- Added emojis to all section headings and key callouts
- Converted inline examples to formatted tables where appropriate
- Used callout blocks (blockquotes) for key insights
- Ensured Australian English spelling throughout
- Preserved all factual content — detection criteria, thresholds, and technical details unchanged

## Evidence
- `quality.sh` passes cleanly (all 124 tests pass)
- No ASCII art remains in any of the 10 files (verified via grep)
- Mermaid diagrams use GitHub-compatible syntax with `style` directives for colours

## Test Plan
- Documentation-only change — no code modified
- Verified `quality.sh` passes with all checks green
- Verified no ASCII art box-drawing characters remain in converted files
