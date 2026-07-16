## Summary

The playful project term **"creature"** was used from the first paragraph of
`README.md` (and pervasively across `AGENTS.md` and `docs/**`) but was never
defined in plain English on first use, reading as unexplained jargon that slows
onboarding for new readers and agents (documentation audit, check 7).

This PR adds a one-line, plain-English gloss immediately after the first use of
"creature" in `README.md`, defining it as *an individual candidate neural
network (a genome) in the evolving NEAT population* and linking the underlying
algorithm to the [Neuroevolution of augmenting topologies](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
Wikipedia article. Downstream docs can now point back to this single definition.

Closes #1614.

## Evidence

Documentation-only change — no web interface to screenshot and no Rust code
touched. Verified with `markdownlint-cli2`, the configured markdown quality
gate:

```
Summary: 0 error(s)
```

The gloss as it now reads at first use in `README.md`:

> **creature**: an individual candidate neural network (a genome) in the evolving
> [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
> population. NEAT-AI evolves a population of creatures; this library helps decide
> which mutations to a creature are worth trying.

## Test Plan

- Ran `markdownlint-cli2 "README.md"` — 0 errors (also passes the repo-wide
  64-file lint).
- No Rust source changed, so no unit/integration tests are affected.
