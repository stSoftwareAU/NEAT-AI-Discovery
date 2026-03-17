## Summary

Convert all ASCII art diagrams to Mermaid and apply fun/informative styling (emojis, colours, Australian English) in the 13 Repair discovery scenario guides under `docs/discoveries/`. Closes #845.

### Changes

All 13 files updated:
- `saturated-neuron.md`, `output-bias-drift.md`, `oscillating-neuron.md`, `activation-mismatch.md`, `unbounded-capping.md`, `restricted-range.md`, `operating-point.md`, `bias-perturbation.md`, `squash-weight-rescale.md`, `activation-recommendation.md`, `symmetry-breaking.md`, `error-plateau.md`, `output-squash-mismatch.md`

For each file:
1. Converted all ASCII art network diagrams to Mermaid (flowcharts and graph diagrams)
2. Added colour styling — red for problematic components, green for fixed/healthy, blue for inputs, orange for warnings
3. Added emojis to section headings and key callouts
4. Applied blockquote callout styling for key insights
5. Converted inline examples to structured tables for readability
6. Ensured Australian English spelling throughout (colour, behaviour, organisation, utilisation, recentring, etc.)
7. All detection criteria, thresholds, and factual content preserved unchanged

## Evidence

- `quality.sh` passes cleanly (all 124 tests pass, clippy/fmt/doc build all green)
- No ASCII art (```` ``` ```` code blocks with box-drawing characters) remains in any of the 13 files
- All diagrams use Mermaid format with GitHub-compatible syntax and colour styling

## Test Plan

- Verified `quality.sh` passes with no regressions
- Documentation-only change — no code modified, no new tests required
