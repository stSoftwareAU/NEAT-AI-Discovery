# Remove-neuron reachability — measured reference table (Issue #1810)

Sub-issue of the #1785 milestone: the evidence base every other sub-issue is
measured against.

Issue #1785 cites a reproduction that does not exist on `Develop` —
`tests/issue_1777_discovery_diagnostic.rs` is not a file in this repository, and
[`candidate-rate-diagnosis-1777.md`](candidate-rate-diagnosis-1777.md) has no
"Fresh-run evidence" section. The numbers it quotes were therefore not
re-derivable, and nothing pinned the end-to-end zero yield the milestone is
trying to change.

This document is that evidence base. Every number below is **produced by a
committed test**, not quoted:

```bash
cargo test --test issue_1785_remove_neuron_reachability -- --nocapture --test-threads=1
```

The test is a **characterisation pin, not an invariant**. A fix to either gate is
expected to break it loudly; when that happens, update the pin *and* this table
together.

## Measurement fixture

`tests/fixtures/remove_neuron_reachability/network.json` — hand-authored and
synthetic (see its
[README](../../tests/fixtures/remove_neuron_reachability/README.md) for the
construction rule). 6 inputs, **36 hidden** neurons (12/10/8/3 across four
feed-forward layers plus 3 orphans), 2 outputs, 102 synapses, maximum hidden
degree **12**, all weights `1.0`, all squashes `IDENTITY`, `costOfGrowth = 1e-7`
(the production default, `DEFAULT_COST_OF_GROWTH`).

## The two gates

```mermaid
flowchart TD
    C["Fixture creature<br/>36 hidden neurons"] --> G1
    C --> G2
    subgraph G1["Gate 1 — analysis path"]
        A1["36 sole-op RemoveNeuron candidates"] --> A2["apply_honest_remove_neuron_gain (#1530)<br/>honest gain = −impact ≤ 0"]
        A2 --> A3["coordinated_post_discount_noise_floor(1) = 5e-7"]
        A3 --> A4["0 candidates clear the floor"]
        A4 --> A5["Escape hatch: #1622 promotion<br/>0 flags, 0 promoted"]
    end
    subgraph G2["Gate 2 — focus / FFI path"]
        B1["identify_structural_removal_candidates(creature, 1e-7)"] --> B2["boosted_savings ≤ contribution<br/>33 silent, uncounted drops"]
        B1 --> B3["net &lt; REMOVE_LOW_IMPACT_NOISE_FLOOR (1e-5)<br/>3 reported rejections"]
        B2 --> B4["0 surviving candidates"]
        B3 --> B4
    end
    A5 --> Z["End-to-end yield: 0 remove-neuron candidates"]
    B4 --> Z
```

## Block 1 — Gate 1, analysis path

`apply_honest_remove_neuron_gain` (`src/analysis/discovery_dispatch.rs:141`) over
one sole-op `RemoveNeuron` candidate per hidden neuron.

| Measurement | Value |
|---|---|
| Removable neurons (candidate set size) | **36** |
| Neurons whose gain the #1530 override replaced | 36 |
| Honest gain — min | `−1.000000e0` |
| Honest gain — median | `−1.666667e-1` |
| Honest gain — max (best) | `−0e0` |
| `coordinated_post_discount_noise_floor(1)` | `5e-7` |
| Neurons clearing the floor | **0** |

The honest gain is `−impact` (`estimate_remove_neuron_gain`,
`src/analysis/remove_neuron_gain.rs:131`), so it is non-positive by construction
while the floor is strictly positive: **no** remove-neuron candidate can clear
Gate 1, on this fixture or any other. That is stronger than #1785's quoted
"0 neurons clearing the floor" — it is structural, not incidental.

## Block 2 — the promotion escape hatch

| Measurement | Value |
|---|---|
| `functionally_constant_neuron_uuids(&creature)` | **empty** (unwired seam, `remove_neuron_constant_promotion.rs:117`) |
| Candidates carrying a `constant_neuron_bias_fold` | **0** |
| `bias_folded_constant_neuron_uuids(&candidates)` | **empty** |
| Candidates promoted to `CONSTANT_NEURON_PRIORITY_GAIN` | **0** |

The only route past Gate 1 is #1622 promotion. Its structural flag source is a
documented, still-unwired seam that returns an empty set, and its live #1779
source needs a candidate carrying an *accepted* bias fold — which the
structure-only candidate set never attaches. The escape hatch therefore yields
zero, matching #1785's "0 promotion entries".

## Block 3 — Gate 2, focus / FFI path

`identify_structural_removal_candidates(&creature, 1e-7)`
(`src/focus/ranking/removal_candidates.rs:462`), reached through the shipped FFI
entry point `rank_focus_neurons_internal` (`src/ffi_internal/analysis.rs:707`)
because the function is `pub(crate)` (the Issue #1806 convention).

| Measurement | Value |
|---|---|
| Hidden neurons considered | 36 |
| Surviving candidates | **0** |
| `noise_floor_rejections` (reported under `rejectionBreakdown`) | **3** |
| Silent `boosted_savings <= contribution` drops (uncounted) | **33** |
| Best boosted savings across the fixture | **`3.3e-7`** |
| `REMOVE_LOW_IMPACT_NOISE_FLOOR` | `1e-5` |

The two drop paths split cleanly by structure: the 33 connected hidden neurons
carry a structural contribution orders of magnitude above their boosted savings,
so they hit the bare `return None` at
`src/focus/ranking/removal_candidates.rs:387` — counted **nowhere**, which is why
the test measures it as the residue. The 3 orphans have contribution `0.0`, so
they reach the noise-floor gate and are reported. Even they fall short by a
factor of ~30 (`3.3e-7` at best, against `1e-5`).

## Block 4 — the 657-synapse break-even

Searched over the shipped `calculate_removal_savings`
(`src/focus/ranking/removal_candidates.rs:73`) rather than quoted.

| Measurement | Value |
|---|---|
| `costOfGrowth` | `1e-7` |
| `REMOVAL_CANDIDATE_BOOST` | `1.5` |
| `REMOVE_LOW_IMPACT_NOISE_FLOOR` | `1e-5` |
| Smallest synapse count clearing the floor | **657** |
| Boosted savings at 657 synapses | `1.0005e-5` |
| Boosted savings at 656 synapses | `9.99e-6` |

A zero-contribution neuron needs `1.5 × 1e-7 × (1 + n/10) >= 1e-5`, i.e.
`n >= 656.7` — **657 synapses** on a single neuron before pruning it is worth
reporting. Confirmed.

## Reconciliation with the numbers quoted in #1785

| #1785 quotes | Status here | Value on `Develop` (`a36e9ba`) |
|---|---|---|
| 36 removable neurons | Reproduced (fixture sized to match) | 36 |
| Best honest gain `−1.84e-2` | **Not reproducible** — it is a property of the unavailable production creature, not of the pipeline. The signed, non-positive shape is reproduced. | best `−0e0`, median `−1.67e-1` on this fixture |
| 0 neurons clearing the floor | Reproduced, and shown to be structural | 0 |
| 0 promotion entries | Reproduced | 0 |
| Best boosted savings `3.30e-7` | Reproduced (degree-12 hub) | `3.3e-7` |
| 657 synapses needed | Reproduced by construction | 657 |

### Corrected source references

Verified against `Develop` at `a36e9ba`. #1785's table was written against
`06403d7`; several line numbers have since moved, and two pointed at the wrong
code.

| #1785 says | Correct on `a36e9ba` |
|---|---|
| `remove_neuron_constant_promotion.rs:112-114` | `remove_neuron_constant_promotion.rs:117-119` — `functionally_constant_neuron_uuids` returns an empty `HashSet` |
| `candidate_scoring.rs:1470` (`REMOVE_LOW_IMPACT_NOISE_FLOOR`) | `candidate_scoring.rs:1469`; `REMOVAL_CANDIDATE_BOOST` at `candidate_scoring.rs:1440` |
| `removal_candidates.rs:160-170` (silent drop site) | that range is `RemovalCandidateOutcome::rejection_breakdown` (`:161-171`); the silent `boosted_savings <= contribution` drop is at `removal_candidates.rs:387-389` |
| triage lives in `removal_triage.rs` | the shipped copy is `removal_candidates.rs::identify_structural_removal_candidates` (`:462`), called from `ffi_internal/analysis.rs:707`. `removal_triage.rs` retains only the thin `triage_removal_candidates` adapter over the same criterion (#1805) |
| `discovery_dispatch.rs:142` (honest-gain override) | `discovery_dispatch.rs:141` |

Line numbers move. The function and constant names above are the durable
references; treat the numbers as a snapshot of `a36e9ba`.

## What a fix has to change

For a remove-neuron candidate to reach NEAT-AI, **both** gates must yield:

- **Gate 1** needs the floor comparison to admit a non-positive honest gain (or
  the promotion seam to be wired), otherwise the 5e-7 floor rejects everything
  by construction.
- **Gate 2** needs the savings scale to reach the `1e-5` floor — today that takes
  657 synapses on one neuron — and needs the `boosted_savings <= contribution`
  drop to be counted, since 33 of 36 neurons currently vanish there with no
  diagnostic at all.

Both are out of scope for #1810, which only measures and pins. They belong to
the other sub-issues of #1785, and each should re-run the command at the top of this
document and update this table as part of its closing checklist.
