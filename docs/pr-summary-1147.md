## Summary
Added a `## 🔗 Related Repositories` section to the NEAT-AI-Discovery README using the canonical markdown block defined in [stSoftwareAU/NEAT-AI-core#18](https://github.com/stSoftwareAU/NEAT-AI-core/issues/18). The section lists all seven public NEAT-AI-* repositories with one-line descriptions and includes the canonical Mermaid dependency diagram, with a small framing note indicating this repository is NEAT-AI-Discovery. Placed before the `## 📚 Additional Documentation` section so it sits alongside other project-level navigation content. Closes #1147.

## Evidence
This is a documentation-only change; no UI or performance impact.

- README renders the new section with the seven-repo table and the Mermaid dependency graph.
- Canonical repo list, descriptions, and diagram are preserved verbatim from the source issue.
- Repo quality gate (`./quality.sh`) exercises Rust build/lint/test — README-only change does not affect it.

## Test Plan
- [x] README opens to the new `🔗 Related Repositories` section with all seven repos and working GitHub links.
- [x] Mermaid block is valid and matches the canonical diagram (same nodes and edges as the source issue).
- [x] `./quality.sh` run locally.
