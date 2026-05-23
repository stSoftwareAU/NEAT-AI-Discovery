## Summary

Reworded `docs/discoveries/add-neuron.md:56` (and three sibling claims surfaced
by the audit) to stop overclaiming that "expected improvement" equals the
network's MSE reduction. The figure is sum-of-squared-error (SSE) reduction —
exact only when NEAT-AI's cost function is `MSE`, and a ranking signal for the
other six built-in costs (`MAE`, `MAPE`, `MSLE`, `HINGE`, `CROSS_ENTROPY`,
`CATEGORICAL_ERROR`). Closes #1251.

## Evidence

Docs-only change — no code paths affected. Source references cross-checked
against [`docs/COST_FUNCTION_NOTES.md`](../../COST_FUNCTION_NOTES.md) §4 and
§6.

Files updated:

- `docs/discoveries/add-neuron.md:56` — reworded the flowchart node and added
  a `NOTE` callout that links to `COST_FUNCTION_NOTES.md` for the per-cost
  contract.
- `docs/discoveries/add-synapse.md:183` — example expected-improvement line
  reworded ("12% reduction in O1's sum-of-squared error (exact for `MSE`; a
  ranking signal for other costs)").
- `docs/discoveries/redundant-path.md:47` — improvement-estimation flowchart
  node now says SSE original vs renormalised with the MSE caveat.
- `docs/discoveries/redundant-path.md:121` — example wording reworded from
  "MSE comparison" to "sum-of-squared-error comparison" with the same caveat.
- `docs/discoveries/output-bias-drift.md:108-110` — Wikipedia reference
  softened from "the loss metric that output bias drift directly inflates" to
  acknowledge that the inflation is exact for `MSE` and a ranking signal for
  the other costs.

Audit completed across `docs/discoveries/*.md` for `MSE` / `mean squared
error` / `sum-of-squared` / `loss reduction` / `least squares` claims. The
remaining `least squares` references describe the regression method itself,
not a loss claim, and remain accurate.

## Test Plan

- [x] Docs-only change; `quality.sh`'s spell-check / markdown lint will run
      on CI.
- [x] Verified all four discovery markdown files render with their existing
      Mermaid blocks intact (no syntax changes inside fenced ` ```mermaid `
      blocks beyond extending an existing flowchart-node label).
- [x] Cross-referenced the new wording against `docs/COST_FUNCTION_NOTES.md`
      §4 (per-cost behaviour) and §6 (concrete bugs surfaced).
