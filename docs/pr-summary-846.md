## Summary

Convert all ASCII art diagrams to Mermaid and apply fun/informative styling (emojis, colours, Australian English) in the Growth, Topology, Synapse Weight, and Scoring & Recommendation discovery scenario guides. Closes #846.

### Files Updated (17 files)

**Growth & Topology Discoveries (10):**
1. `docs/discoveries/bottleneck-neuron.md`
2. `docs/discoveries/correlated-error.md`
3. `docs/discoveries/multi-hop.md`
4. `docs/discoveries/add-neuron.md`
5. `docs/discoveries/add-synapse.md`
6. `docs/discoveries/skip-connection.md`
7. `docs/discoveries/topology-diversification.md`
8. `docs/discoveries/output-conflict.md`
9. `docs/discoveries/hard-sample-cluster.md`
10. `docs/discoveries/output-range-compression.md`

**Synapse Weight Discoveries (5):**
11. `docs/discoveries/gradient-discovery.md`
12. `docs/discoveries/weight-coherence.md`
13. `docs/discoveries/weight-magnitude-reset.md`
14. `docs/discoveries/weight-polarity-flip.md`
15. `docs/discoveries/input-sensitivity.md`

**Scoring & Recommendation (2):**
16. `docs/discoveries/sample-weighted.md`
17. `docs/discoveries/monotonicity.md`

**Note:** `docs/discoveries/topology.md` was listed in the issue but does not exist in the repository and was skipped.

### Changes Per File

For each file:
- Converted all ASCII art network diagrams to Mermaid (flowcharts and graph diagrams)
- Used colours to distinguish healthy (green), problematic (red), input (blue), new (purple), process (light blue), and decision (orange) components
- Added emojis to section headings and key callouts
- Applied styling (callout blocks, bold, formatting)
- Ensured Australian English spelling throughout
- Kept content factually accurate — no detection criteria or thresholds were changed

## Evidence

- No ASCII art remains in any of the 17 updated files (verified via grep for box-drawing characters)
- All diagrams use Mermaid format with consistent colour scheme
- `quality.sh` passes cleanly

## Test Plan

- Documentation-only change — no code modified
- Verified no ASCII box-drawing characters remain in any updated file
- `quality.sh` passes with all 124 tests passing
