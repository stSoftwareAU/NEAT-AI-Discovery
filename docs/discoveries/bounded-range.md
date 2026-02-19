# Bounded Range Detection

[Back to Discovery Index](README.md) | **Source:** [`src/analysis/detection/bounded_range.rs`](../../src/analysis/detection/bounded_range.rs) | **Issue:** [#395](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/395)

---

## The Problem

Some input and hidden neurons receive data where a significant fraction of
samples sit at a **sentinel boundary value** (typically -1, 0, or +1 meaning
"missing", "off", or "maximum"). These sentinel values carry no useful signal
but are mixed in with genuine data, confusing downstream neurons that cannot
distinguish "missing" from "actually zero".

```
  Input neuron activations:
  ──────────────────────────
  -1, -1, 0.3, -1, 0.7, -1, 0.5, -1, 0.4, -1
       ↑                    ↑
  Sentinel values           Useful values
  (40% of samples)          (in [0.3, 0.7] range)

  Gap between sentinel (-1) and useful range (0.3):
  ────────────────────────────────────────────────
  -1.0         0.0         0.3    0.5    0.7
   ╔═╗          │           ╔══════════════╗
   ║S║          │           ║  useful      ║
   ╚═╝          │           ║  range       ║
   sentinel     gap=1.3     ╚══════════════╝
```

### Why It Hurts the Creature's Score

- **Signal contamination**: The sentinel value is treated as a real data
  point, biasing learned weights.
- **Wasted capacity**: The neuron cannot learn separate responses for
  "missing data" vs "data with value near the sentinel".
- **Downstream confusion**: Neurons receiving this signal cannot distinguish
  meaningful values from sentinel markers.

---

## How We Detect It

```
  ┌────────────────────────────────────────────────────────────────┐
  │  For each input and hidden neuron:                             │
  │                                                                │
  │  1. Collect activation samples (minimum 20)                    │
  │  2. For each sentinel candidate (-1, 0, +1):                   │
  │     a. Count fraction of samples within tolerance of sentinel  │
  │     b. If fraction >= 20% → potential sentinel                 │
  │     c. Measure gap between sentinel and nearest useful value   │
  │     d. If gap exceeds minimum → sentinel confirmed             │
  │  3. If sentinel detected → bounded range candidate             │
  └────────────────────────────────────────────────────────────────┘
```

### Confidence Scoring

```
  Confidence = 0.5 + (fraction_factor × 0.4
                    + gap_factor × 0.4
                    + sample_factor × 0.2) × 0.5

  Higher sentinel fraction and wider gap increase confidence.
```

---

## How We Fix It

The fix adds a **gating neuron** that learns to suppress the sentinel values
while passing useful data through:

```
  BEFORE                                 AFTER
  ┌─────────┐                            ┌─────────┐
  │ Input   │──────────────→ H1          │ Input   │──────────────→ H1
  │ [-1,0.5]│                            │ [-1,0.5]│
  └─────────┘                            └─────────┘
  sentinel mixed                                │
  with useful data                              │
                                                ▼
                                         ┌──────────┐
                                         │ Gate     │
                                         │ RELU     │───→ H1
                                         │ bias =   │
                                         │ -centre  │  Suppresses
                                         └──────────┘  sentinel!
```

| Candidate | Operations | Detail |
|-----------|-----------|--------|
| **Add gating neuron** | `addNeuron` + `addSynapse` | RELU gate with bias set to suppress the sentinel value |

The gate neuron's bias is set to `-useful_centre`, causing sentinel values
to produce zero output while useful values pass through.

---

## Example

```
  Input neuron I5 has 500 samples:
  - 180 samples (36%) at value = -1.0 (sentinel for "missing")
  - 320 samples in range [0.2, 0.8] (useful data)
  - Gap between sentinel and useful range: 1.2

  Fix: Add RELU gate neuron
    Bias = -(0.2 + 0.8)/2 = -0.5
    Sentinel input: RELU(-1.0 - 0.5) = RELU(-1.5) = 0 (suppressed)
    Useful input:   RELU(0.5 - 0.5) = RELU(0.0) = 0.0 (borderline)
                    RELU(0.8 - 0.5) = RELU(0.3) = 0.3 (passes through)

  Result: Downstream neurons receive 0 for missing data
  and proportional values for real data.
```

---

## References

- **Source module**: [`src/analysis/detection/bounded_range.rs`](../../src/analysis/detection/bounded_range.rs)
- **DISCOVERY_TYPES.md**: [Technical reference](../DISCOVERY_TYPES.md)
- **Related**: [Sentinel Value Gating](sentinel-gating.md) — similar concept
  for input neurons with additional error-correlation checks
- **Related**: [Observation Utilisation](observation-utilisation.md) — flags
  underutilised input ranges dominated by sentinels
