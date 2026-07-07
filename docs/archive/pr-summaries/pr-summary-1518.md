## Summary

Replaced the fabricated floor-at-`0.1` placeholder remove-neuron gain with a
**propagation-aware estimator** in NEAT-AI-Discovery (Rust). The old NEAT-AI
(Deno) `#2483` sink turned a large squash error into a large positive "gain"
(`0.1 + (log10(err) − 10)/10 × 0.4`, clamped `[0.1, 0.5]`) regardless of
topology. For a neuron many layers from the output(s), the activation is
attenuated / squashed repeatedly on the way to the output, so the true effect of
removal is tiny — the placeholder claimed `+0.17882921` for `neuron-1802938338`
while the measured effect was only `-0.000194` (~920× too large and opposite in
sign).

The new `estimate_remove_neuron_gain(creature, neuron_uuid)` reuses the existing
propagation-aware impact machinery (`compute_impacts_public`, which walks every
downstream edge weight and squash bound to the output) and turns the unsigned
influence fraction into a signed honest gain:

- **Magnitude** = the neuron's propagation-aware influence on the output(s).
  Deep neurons attenuate to ~`1e-4`, not the fabricated `0.1+`.
- **Sign** = non-positive. Removing a neuron that still carries genuine
  downstream influence removes that contribution, so the trained network's score
  is expected to drop by roughly its influence. A neuron with no downstream
  influence attenuates to ~`0`.

This keeps over-threshold "harmful" neurons removal-eligible (their honest gain
is ≈0 or slightly negative) without a fabricated large positive gain crowding
out realistic (~`1e-4`) candidates.

Closes #1518.

## Evidence

Backend/CLI change — no web interface to screenshot. Verified via the committed
production-depth fixture (`neuron-1802938338` in the GRQ-cluster creature) and
synthetic-topology unit tests.

Data flow of the estimator:

```mermaid
flowchart LR
    A[Remove-neuron candidate<br/>neuron_uuid] --> B[compute_impacts_public]
    B --> C[Walk downstream edges:<br/>weight × squash bound<br/>to output(s)]
    C --> D[Influence fraction I in 0..1<br/>deep neuron → tiny]
    D --> E[Signed honest gain = −I]
    E --> F{Within one order of<br/>measured actual<br/>& correct sign?}
    F -->|yes| G[Regression gate green]
```

Estimator result on the recorded failure fixture:

| Quantity | Value |
|----------|-------|
| Old placeholder gain | `+0.17882921` |
| Measured actual effect | `-0.000194` |
| Propagation-aware estimate | small **negative** value within one order of magnitude of `-1.9e-4` |

## Test Plan

- `tests/remove_neuron_propagation.rs`
  - `remove_neuron_effect_at_production_depth` — **`#[ignore]` removed**; now the
    permanent regression gate. Asserts `estimate_remove_neuron_gain` matches the
    measured actual (`-0.000194`) within one order of magnitude and in sign on
    the committed fixtures.
  - `placeholder_gain_is_wrong_at_depth` — **inverted** (per #1516/#1517 done-check):
    now asserts the estimator does **not** emit a value in the retired
    `[0.1, 0.5]` placeholder floor range for the deep neuron, so an accidental
    resurfacing of the placeholder path turns CI red.
- `tests/remove_neuron_gain.rs` (new) — synthetic-topology unit tests:
  - `shallow_hidden_neuron_has_small_negative_gain` — 1-of-2 inbound neuron → `~-0.5`.
  - `deeper_neuron_attenuates_below_shallower_neuron` — deeper neuron attenuates
    below a shallower one (propagation behaviour the placeholder lacked).
  - `output_neuron_returns_none`, `unknown_neuron_returns_none` — edge cases.
  - `disconnected_neuron_has_zero_gain` — no downstream path → `~0` gain.

### Note on dependency upgrade

`quality.sh`'s `cargo upgrade --incompatible` step bumps `wgpu`/`naga` from
`29 → 30`, a major version with breaking API changes (`BufferView` /
`MapRangeError`) that does not compile. That bump is out of scope for this issue
and was reverted so this PR stays isolated; the `wgpu 30` migration is a
separate piece of work. All other quality checks (fmt, clippy `-D warnings`,
check, doc, tests) pass at the pinned `wgpu 29`.
