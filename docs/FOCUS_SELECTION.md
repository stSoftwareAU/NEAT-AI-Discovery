# Focus Selection

This document captures the **focus-selection design end-to-end** — why each
discovery run focuses on a small subset of neurons, how the focus set is now
chosen (a **structure-weighted random draw**, with **no parquet on the focus
path**), and why the **seconds bar** is a hard invariant.

It exists so the next "please confirm my understanding of the focus logic"
request is a single doc link rather than a code spelunk
(Issues #1382, #1386, #1766).

---

## 1. Why a focus subset at all

Discovery cannot evaluate *every* neuron in a network within the per-run
evaluation budget — the cost grows with the number of candidate targets. Each
run therefore **focuses** on a small set (about half a dozen, ~6) of *selectable*
neurons and concentrates the analysis budget there.

A "selectable" neuron is one that can usefully be tuned: **input** and
**constant** neurons are excluded (see `is_selectable_type` in
[`src/focus/ranking/mod.rs`](../src/focus/ranking/mod.rs)). Output neurons **are**
selectable — and, seeded at impact `1.0`, they are the highest-impact targets by
definition (`output-0` is the canonical example).

## 2. The product rule — structure-weighted random (Issue #1766)

**Choosing the focus set is derived from creature structure alone.** It never
opens or decodes the discovery parquet.

```text
Creature JSON → structural impact map → weighted-random by impact → focus set N
```

1. **Structural impact from topology only.** The impact map is computed with
   [`compute_impacts_public`](../src/focus/impact.rs) — path-weight products over
   the creature graph, with output neurons seeded at `1.0`. No discovery records
   are read.
2. **Weighted-random draw.** [`select_focus_by_structural_impact`](../src/focus/selection.rs)
   draws `min(N, pool)` selectable neurons **without replacement** by roulette
   over those impacts. Large creatures still explore beyond pure greed while
   mostly landing on high-impact neurons; output neurons dominate the weight mass
   naturally, so they are almost always chosen without a hard-coded "only
   output-0" policy.
3. **Deterministic and reproducible.** The draw is seeded by the caller's
   monotonic `focusSelectionCursor` (falling back to
   `epochsSinceLastAcceptedCandidate`, then `0`). A fixed cursor reproduces the
   same set; advancing the cursor reseeds the draw so successive passes sweep a
   fresh tail.
4. **Zero-weight fallback.** Neurons with zero / negative / non-finite impact
   carry zero weight; when the remaining pool is all-zero the draw falls back to
   a uniform pick so the focus set still fills.

```mermaid
flowchart LR
    C[Creature JSON] --> I["Structural impact map<br/>compute_impacts_public<br/>(topology only, no records)"]
    I --> W[Positive weights per<br/>selectable neuron<br/>outputs seed at 1.0]
    W --> D["Weighted-random draw<br/>without replacement<br/>seed = focusSelectionCursor"]
    D --> F[Focus set of N neurons]
    F --> A["Analysis phase<br/>(parquet decoded HERE, after focus is chosen)"]
```

## 3. The seconds bar — a hard invariant

**If choosing the focus set takes more than seconds, we have shot ourselves in
the foot.** The structural draw is `O(neurons + synapses)` and completes in
milliseconds even on a creature with a multi-GB recording — because the parquet
is never touched to pick the focus set.

This invariant exists because of the **focus-stall incident** that motivated
this issue: a trivial 16-hidden, `maxNeurons=6` creature accumulated ~12.5 GB of
discovery data, and the
then-current parquet-coupled ranking projected an 18.7 GB in-memory load, scaled
its own budget to ~14 min, and ran for **~2 h** in `build_lazy_provider` /
`read_all_records_grouped_by_neuron` before aborting — after which the instant
structure-only fallback finished in one shot (`6 candidates, 6 selected`). That
proved selection is cheap once parquet is out of the path; coupling focus choice
to multi-GB I/O burned the analysis budget before any useful work started.

The regression is locked by
[`tests/ffi/issue_1766_structural_focus_selection.rs`](../tests/ffi/issue_1766_structural_focus_selection.rs):
a reference-shaped creature (16 hidden feeding one output) selects successfully
in well under the seconds bar even when the `parquetFile` path **does not
exist**, proving no parquet is opened.

**Not the fix:** scaling `FOCUS_RANKING_BUDGET_MS` / lazy budgets, a multi-GB
warm pass, or "fail fast to a fallback while still attempting parquet-coupled
ranking as the happy path." The happy path is structure-only, full stop.

## 4. Parquet is for analysis, not for choosing focus

Parquet is still decoded — but **after** the focus set is chosen, by the
**analysis** phase (`analyze_parallel`) that scores concrete synapse/neuron
candidates on the chosen neurons. The focus FFI therefore omits the
`loadingMode` / `projectedMb` observability fields (no parquet was loaded) and
`constantNeuronRemovals` (constant folding needs recorded activation variance —
see §4.2). Removal *triage*, however, now runs **structure-only** at focus time
on the near-opposite axis to focus — see §4.1.

### 4.1 Removal is the near-opposite axis to focus (Issue #1767)

Focus and removal are **opposite axes over the same structural impact map**:

| Concern | Structural criterion | Parquet at focus time |
|---------|----------------------|-----------------------|
| **Focus** | **High** structural impact — weighted-random draw, outputs seed at `1.0` (§2) | **Never** |
| **Removal** | Near-**opposite**: **low** structural contribution vs the complexity **savings** of pruning the neuron and its synapses | Not while **choosing** the candidate set; an activation-only pass then measures the survivors (§4.2) |

[`identify_structural_removal_candidates`](../src/focus/ranking/removal_candidates.rs)
flags a **hidden** neuron for removal when the boosted complexity savings of
pruning it exceed its structural contribution (its path-weight impact on the
outputs):

```text
savings          = costOfGrowth × (1 + (incoming + outgoing) / 10)      # NEAT-AI Score.ts
boostedSavings   = savings × REMOVAL_CANDIDATE_BOOST (1.5×)
contribution     = structural impact on outputs (compute_impacts_public, topology only)
removalCandidate ⟺ boostedSavings > contribution  AND  (boostedSavings − contribution) ≥ noiseFloor
```

- It is `O(neurons + synapses)` and reads **no** discovery records — so it never
  reintroduces the focus-time parquet dependency §3 removed.
- Only **hidden** neurons are considered: outputs (impact `1.0`) and
  input / constant neurons are never removal targets.
- This is *low contribution*, **not** "negate the focus score". The #414
  philosophy — *high error ≠ remove* — is untouched: error is never read here.
- The #1142 noise-floor gate is reused, so boost-inflated near-zero wins are
  dropped and surfaced under `rejectionBreakdown`
  (`REJECTION_REMOVAL_BELOW_NOISE_FLOOR`).

#### Rejection accounting (Issue #1808)

Every hidden neuron entering triage ends in exactly one bucket — emitted as a
candidate, or counted under a named reason — so `candidates + rejections`
always equals the hidden neurons considered. Both gates report through the same
`rejectionBreakdown` map, so a caller never has to hard-code a reason string:

| Verdict | Reported as |
|---------|-------------|
| `boostedSavings > contribution` and margin ≥ noise floor | a `removalCandidates[]` entry |
| `boostedSavings ≤ contribution` | `removal_savings_below_impact` |
| margin < noise floor | `removal_below_noise_floor` |
| mean activation above threshold | `removal_active_neuron` |

```mermaid
flowchart LR
    H["hidden neuron"] --> G1{"boostedSavings<br/>&gt; contribution?"}
    G1 -- no --> R1["removal_savings_below_impact"]
    G1 -- yes --> G2{"margin ≥<br/>noiseFloor?"}
    G2 -- no --> R2["removal_below_noise_floor"]
    G2 -- yes --> C["removalCandidates[]"]
    R1 --> B["rejectionBreakdown"]
    R2 --> B
```

`triage_removal_candidates` returns the same shape:
`StructuralRemovalTriage::rejection_breakdown()` plus
`hidden_neurons_considered`, so the structure-only adapter and the shipped FFI
path can no longer diverge on what they report.

### 4.2 The activation-weighted gate resolves after selection (Issue #1923)

Structural triage picks the candidate **set**; it cannot rank it. Until #1923
every candidate shipped with `meanActivation: 0.0`, a hard-coded value the
reason string called "deferred to analysis" — and nothing downstream ever
resolved the deferral. The #1920 cache study measured the cost:
`removalCandidate.impact` correlated with realised gain at **r = −0.036**, and
`meanActivation` had **zero variance** across all 67 cached records, so
`remove-low-impact` — 53% of every cached candidate and 79% of all realised gain
— picked arbitrarily from its eligible pool.

The gate now resolves in the same pass, **after** selection is complete:

```mermaid
flowchart LR
    C["creature topology"] --> S["focus selection<br/>(structure only, §3)"]
    C --> T["structural removal triage<br/>(§4.1, no parquet)"]
    T --> W["activation-only parquet pass<br/>neuron_uuid + activation, no records materialised"]
    W --> G{"gate:<br/>meanActivation ≤ 0.04?<br/>savings &gt; awi ≥ noiseFloor?"}
    G -- no --> B["rejectionBreakdown"]
    G -- yes --> R["removalCandidates[]<br/>ranked by savings − awi"]
```

- The measurement pass projects **two** columns and materialises **no**
  `DiscoverRecord`s, so it never decodes the `errors` list that dominates the
  file — it is not the multi-GB record warm §3 removed. It runs strictly after
  the focus set is fixed and is bounded by the shared discovery deadline, so it
  cannot delay or alter focus selection.
- `activationWeightedImpact = impact × meanActivation` is now live, and
  candidates are ranked on `removalSavings − activationWeightedImpact` —
  the same criterion the record-derived path uses.
- `expectedErrorReduction` carries the activation-weighted contribution the
  removal gives up (#117), not a structural first-pass estimate.

**Constant-neuron folding** (`constantNeuronRemovals`, #306) still stays off
this path: it needs recorded activation *variance* and per-record bias folding,
not a single aggregate, so it remains in the analysis phase.

#### When the gate cannot resolve

An unreadable parquet, or a candidate with no recorded rows, leaves
`meanActivation` / `activationWeightedImpact` at `0.0` — but never silently:

- the candidate's `reason` says `activation-weighted gate pending — unresolved:
  <error>` or `— no recorded activation samples`;
- an I/O failure also emits a WARN naming the file and the error; and
- unmeasured candidates rank **below** every measured one, so a zero that means
  "not measured" can never be mistaken for the best available removal.

## 5. What the FFI surfaces

The `rank_focus_neurons` FFI response carries a `focusSelection` block plus a
structure-only `neurons[]` ranked pool (impact-descending). Each ranked neuron's
`impact` and `weightedScore` both carry the structural impact used as the draw
weight; `totalError` / `meanActivation` are record-derived and are `0.0` on the
focus path.

| Field | Meaning |
|-------|---------|
| `selected` | The chosen focus uuids (impact-weighted draw). |
| `rawWeightConcentrationRatio` | max weight ÷ sum over the eligible pool — exposes single-target impact collapse. |
| `weightConcentrationRatio` | Concentration over the **selected** weights. |
| `exploitationCount` / `explorationCount` | Every slot is an impact-weighted draw, so `exploitationCount == selected.len()` and `explorationCount == 0` (the #1662 split no longer applies). |
| `explorationCursor` | The seed used for the draw (the monotonic per-creature cursor). |
| `eligiblePoolSize` / `poolSize` | Selectable neurons available for the draw. |
| `cumulativeCoverage` | Neurons drawn this pass. |
| `droughtActive` | Always `false` on the structure-only path. |

When `rawWeightConcentrationRatio` exceeds `0.5` the crate emits a single
`focus_selection_weight_concentration_high` WARN so a pathologically
single-target impact profile stays visible.

## 6. History — random → parquet-coupled ranking → structure-weighted random

The focus list began as a **uniform-random** pick, then moved to a
**parquet-coupled impact-weighted ranking** (`rank_focus_neurons*`, scoring
`error × impact^γ × gradient × frequency [+ signals]` over recorded discovery
data). That ranking is where the ~2h focus stall came from: it forced a parquet
decode before any neuron was picked. Issue #1766 replaces the focus-choosing path
with the structure-weighted random draw of §2 — keeping the "land mostly on
high-impact neurons, but still explore" intent while removing the multi-GB I/O.

The wall-clock guard (`FocusDeadline`, #1375/#3172), the eager-vs-lazy loader
(#1172/#1376), the perf-cliff WARN (#1377), and the #1662 exploit/explore
allocator were all mitigations for the *parquet-coupled* ranking. They no longer
bound the FFI focus path (there is nothing multi-GB left to bound). The
`rank_focus_neurons*` Rust functions and their record-derived signals (§7, §8)
are retained for **analysis-time** ranking, not for choosing the focus set.

## 7. Environment knobs

Every focus knob — with its default, valid range, and description — is documented
once in the single authoritative reference,
[docs/CONFIGURATION.md § Focus selection & ranking](CONFIGURATION.md#focus-selection--ranking).
The structure-only focus draw of §2 needs **none** of them: `focusSetSize` and
`focusSelectionCursor` are request fields, not env knobs. The remaining
`NEAT_AI_DISCOVERY_FOCUS_*` knobs (ranking budget, memory budget/margin,
perf-cliff, reconstruction mismatch, impact gate) govern the retained
record-derived ranking (§8, §9), not the focus path.

---

## 8. Reconstruction-mismatch focus signal (Issue #1634) — record-derived ranking

> Applies to the retained record-derived `rank_focus_neurons*` ranking, not the
> structure-only focus path of §2.

Impact-weighted ranking measures *how much a change would move the output* but
not *which neurons the current model of the creature fails to explain*. The
per-neuron **reconstruction mismatch** measures the latter: for each selectable
neuron we reconstruct its activation from its inbound synapses and compare it to
the recorded activation.

```text
reconstructedValue      = bias + Σ (from_activation × weight)   over inbound synapses
reconstructedActivation = squash(reconstructedValue)
reconstructionMismatch  = mean |recordedActivation − reconstructedActivation|
```

When enabled, the mismatch is folded into the record-derived score as an
**additive** term, applied *after* the multiplicative factors:

```text
weightedScore = error × (impact + ε)^γ × gradientFactor × frequencyFactor × historyMultiplier
              + weight × reconstructionMismatch          # Issue #1634, additive
```

The signal is **opt-in**
(`NEAT_AI_DISCOVERY_FOCUS_RECONSTRUCTION_MISMATCH=1`) with a tunable weight
(`…_WEIGHT`, default `0.1`); disabled it is byte-identical to the pre-#1634 path.

## 9. Impact-magnitude gate (Issue #1635) — record-derived ranking

> Applies to the retained record-derived `rank_focus_neurons*` ranking, not the
> structure-only focus path of §2 (which already draws by impact, so near-zero
> impact neurons are naturally almost never selected).

The constant-neuron filter (`NEAT_AI_DISCOVERY_FOCUS_EXCLUDE_CONSTANT_NEURONS`,
Issue #1624) removes only neurons whose activation *never varies*. Snapshot
mining on the production creature (Issue #1631) found a larger waste class:
neurons that **vary** yet carry a near-zero downstream impact (**31.6%** with
`|impact| < 1e-6`). The **impact-magnitude gate** drops any neuron whose
structural impact magnitude is strictly below the configured threshold:

```text
gated  ⟺  |impact| < NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE_THRESHOLD   # default 1e-6
```

- **Boundary rule — retain-on-equal.** A neuron *exactly* at the gate is kept.
- **Never silently dropped.** The gated count is logged
  (`focus_ineligible_low_impact`) and surfaced on `RankFocusStats`.
- **Opt-in.** Enabled via `NEAT_AI_DISCOVERY_FOCUS_IMPACT_GATE=1`.

---

## See also

- [docs/IMPACT_CALCULATION.md](IMPACT_CALCULATION.md) — how neuron impact is
  estimated (the structural map the focus draw weights by).
- [`src/focus/selection.rs`](../src/focus/selection.rs) — the structure-weighted
  random focus draw (`select_focus_by_structural_impact`) and the retained
  exploit/explore allocator (`select_focus_neurons`).
- [`src/focus/`](../src/focus/) — the ranking implementation
  (`ranking/mod.rs`, `impact.rs`, `gradient.rs`, `layers.rs`, `allocation.rs`,
  `selection.rs`).
- Issues: #1373 (a comparable parquet-stall incident), #1374–#1377, #3172 (parquet-coupled
  guard work, now off the focus path); #1445 / #1662 (record-derived
  exploit/explore, superseded for focus by #1766); **#1766** (structure-weighted
  random focus, no parquet, seconds-bar invariant).
```

