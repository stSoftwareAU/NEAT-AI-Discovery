## Summary

Created comprehensive, visual documentation for all 13 discovery scenarios in `docs/discoveries/`. Each scenario has its own README with:

- **The Problem**: Plain-language explanation of the network pathology, with ASCII diagrams
- **How We Detect It**: Algorithm summary with thresholds and decision flow
- **How We Fix It**: Before/after diagrams showing the proposed structural change
- **Example**: Worked example with concrete numbers
- **References**: Links to relevant Wikipedia articles and academic papers

Added a discovery scenarios index page (`docs/discoveries/README.md`) that provides a bird's-eye view of the entire discovery pipeline and categorises scenarios into Pruning, Repair, and Growth discoveries. Updated the main `README.md` to link to this guide.

## Evidence

Unable to generate screenshot: This is a documentation-only change with no visual interface. The documentation uses ASCII art diagrams rendered in code blocks, which display correctly in any Markdown renderer (GitHub, VS Code, etc.).

### Files Created

| File | Discovery Scenario |
|------|-------------------|
| `docs/discoveries/README.md` | Index page with pipeline overview |
| `docs/discoveries/saturated-neuron.md` | Saturated neuron detection |
| `docs/discoveries/bottleneck-neuron.md` | Bottleneck neuron detection |
| `docs/discoveries/dead-neuron.md` | Dead neuron detection |
| `docs/discoveries/dormant-synapse.md` | Dormant synapse detection |
| `docs/discoveries/opposing-synapse.md` | Opposing synapse detection |
| `docs/discoveries/output-bias-drift.md` | Output bias drift detection |
| `docs/discoveries/oscillating-neuron.md` | Oscillating neuron detection |
| `docs/discoveries/correlated-error.md` | Correlated error pattern detection |
| `docs/discoveries/multi-hop.md` | Multi-hop candidate analysis |
| `docs/discoveries/redundant-path.md` | Redundant path pruning |
| `docs/discoveries/add-neuron.md` | Add neuron discovery |
| `docs/discoveries/add-synapse.md` | Add synapse discovery |
| `docs/discoveries/remove-low-impact.md` | Remove low-impact neuron detection |

### Files Modified

| File | Change |
|------|--------|
| `README.md` | Added "Discovery Scenarios" section linking to the guide; added entry in Additional Documentation table |

## Test Plan

- Added `tests/issue_410_discovery_scenario_docs.rs` with 11 tests:
  - `each_discovery_scenario_doc_exists_with_heading` — verifies all 13 docs exist with correct headings
  - `each_discovery_scenario_doc_has_problem_section` — verifies "The Problem" section
  - `each_discovery_scenario_doc_has_detection_section` — verifies "How We Detect It" section
  - `each_discovery_scenario_doc_has_fix_section` — verifies "How We Fix It" section
  - `each_discovery_scenario_doc_has_example` — verifies "Example" section
  - `each_discovery_scenario_doc_has_references` — verifies "References" section
  - `each_discovery_scenario_doc_has_diagrams` — verifies ASCII diagrams are present
  - `each_discovery_scenario_doc_links_back_to_index` — verifies back-link to index
  - `discovery_index_exists_and_links_to_all_scenarios` — verifies index links all scenarios
  - `discovery_index_contains_pipeline_overview` — verifies pipeline documentation
  - `readme_links_to_discovery_scenarios_guide` — verifies main README linkage
