# AGENTS.md — Agent-Only Pointers for This Repository

This is a **thin pointer file** for AI coding agents. It carries only the
agent-specific rules and invariants that have no better home. Everything else
lives in the human docs — go there first:

- **[README.md](README.md)** — project overview, mission, GPU requirement, and
  the [Minimum System Requirements](README.md#minimum-system-requirements).
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — the canonical coding conventions
  (including the Australian English requirement), testing doctrine, the
  `./quality.sh` quality gate, and the CI pipeline.
- **[docs/FFI_API.md](docs/FFI_API.md)** — full FFI API reference.
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — the authoritative
  `NEAT_AI_DISCOVERY_*` environment-variable list.
- **[docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md)** — the full candidate-type
  reference (types, operations, success/failure rates).

The source layout is not mirrored here — it drifts. Read `src/` directly; each
module's `mod.rs` documents its own responsibilities.

All dependencies must be Apache-2.0 compatible: the authoritative allowed-licence
list is the `[licenses]` table in [`deny.toml`](deny.toml), enforced by
`cargo deny check`.

---

## Do Not Modify CI Without Approval

**Do NOT modify `.github/workflows/ci.yml` without explicit approval.** The CI
triggers, the auto-format job, and the `version-increment` job (which uses the
`ACTIONS_PUSH` PAT so its push re-triggers workflows) are load-bearing.

The PAT must stay off disk: every checkout sets `persist-credentials: false` and
the PAT is bound to `ACTIONS_PUSH_TOKEN` only on the steps that talk to the
remote, which use an explicit authenticated URL (Issue #1868). Do not reintroduce
`token: ${{ secrets.ACTIONS_PUSH }}` on a checkout step.

---

## Dependency Bumps — `./bump-deps.sh` only, never the quality gate

`./quality.sh` does **not** upgrade dependencies (Issue #1865) — a pre-commit
gate verifies the tree, it must not mutate its dependency graph. Bump with
`./bump-deps.sh`, which age-checks every change to the resolved `Cargo.lock`
(transitive packages included) against `VIBE_BUMP_QUARANTINE_HOURS` (default
24h) and pins in-quarantine versions back. The same window is enforced on every
tracked manifest — root and `fuzz/Cargo.toml` — across every dependency table,
`[build-dependencies]` and `[target.<spec>.*]` included (Issue #1908). Do not
reintroduce `cargo upgrade` / `cargo update` into `quality.sh`.

`cargo upgrade --incompatible` — which `bump-deps.sh` uses only to *discover*
candidates — force-bumps dependencies across **major** versions, which can break
unrelated source. The **wgpu**/naga **29 → 30** bump (changed
`Buffer::get_mapped_range()` to return `Result<BufferView, MapRangeError>` and
added `RequestAdapterOptions::apply_limit_buckets`) broke `src/analysis/gpu/*`
and was independently rediscovered and reverted in ~ten PRs before the migration
finally landed (Issue #1594). If a major bump breaks code outside your issue's
scope, **either migrate it in the same PR or revert the bump** — never commit a
half-migrated build.

---

## Version Bumps

**The version in `Cargo.toml` must be incremented on any code change.** Remote
and unattended machines cache the compiled library by version number; without a
bump they keep running the old library. CI's `version-increment` job auto-bumps
the patch version on every PR, but if you commit outside that flow you must bump
it manually (e.g. `0.43.8` → `0.43.9`). Confirm the loaded version with
`get_library_version()`.

---

## Dead Levers — Delete the Component *and* Its Config Surface

**An operator lever that silently does nothing is worse than no lever** — it is
logged at startup, documented, and tuned during an incident, and changes
nothing. The rule, applied three times running (Issues #1792, #1793, #1818):

1. **Delete a never-constructed component unless a concrete writer can be
   named** — an existing inbound surface carrying the data the component is
   keyed on, not "could be wired one day".
2. **Delete its config surface in the same change** — constants, env reads,
   config fields, the startup `info!` line, and the `docs/CONFIGURATION.md` /
   `docs/DROUGHT_PLAYBOOK.md` rows. #1792 deleted `CandidateOutcomeCache` but
   left its staleness env vars read, logged and documented while controlling
   nothing, so #1818 had to delete the same failure mode a second time.
3. **More suppression is the wrong direction** — the pipeline already
   *over*-suppresses (`docs/analysis/candidate-rate-diagnosis-1777.md`).

**Do not re-add these** (the negative result behind #1792/#1793/#1818): the only
inbound per-candidate history is the caller-supplied `failureCache` — failures
only, no `source_uuid` — so nothing can key a
`(source_uuid, target_uuid, operation_type)` cache or a staleness window off it,
and a process-global alternative leaks per-creature suppression between
creatures. `ModuleOutcomeTracker` already covers what `ModuleStarvationTracker`
was meant to do. `src/analysis/candidate_starvation.rs` is a **different, live**
component, untouched by this note.

**Mermaid: never use a bare `;` in unquoted note or message text**
(Issue #1817) — Mermaid parses it as a statement separator, so a `Note over A,B:`
containing one breaks the diagram. Use a comma. The enforcing gate lives outside
this repo (the worker's `mermaid_validator.ts`), so `./quality.sh` cannot catch
it locally, and this repo carries ~30 Mermaid blocks. HTML entities (`&gt;`,
`&le;`) and a `;` inside a quoted label are fine.

---

## Key Invariants — Must Not Be Violated

### Forward-only Activation Order

Discovery assumes **forward-only** networks (no recurrent feedback):

- A neuron may only read activations from **earlier** neurons in the creature's
  evaluation order.
- Synapses must point from an earlier neuron to a later neuron.
- New neurons must be inserted at the correct index (not appended).
- No cross-sample state — each recorded activation/error is for a single
  training sample.

#### Validated FFI Surface (Issue #1184, #1188, #1867)

`validate_forward_only_synapses` (in `src/ffi_types/forward_only_validation.rs`)
and `validate_creature_input_bounds` (in `src/ffi_types/creature_bounds.rs`) are
the defence-in-depth gates. They run immediately after JSON deserialisation and
before any business logic on every FFI entry point that accepts a
`CreatureJson`. A violation returns a structured `DiscoveryError::InvalidInput`
with `error_kind: "data_validation"`.

`validate_creature_input_bounds` caps `creature.input` at
`MAX_CREATURE_INPUT_NEURONS` (1,000,000). The count is caller-supplied and sizes
allocations in both the recording and analysis paths, so an unbounded value
aborts the process via `handle_alloc_error` — an abort `panic::catch_unwind`
cannot intercept (Issue #1867). The cap is absolute, **not** relative to
`creature.neurons.len()`: input neurons are implied by the count and are not
listed in `creature.neurons`.

| FFI entry point | Accepts `CreatureJson` | Validates | Notes |
|-----------------|------------------------|-----------|-------|
| `record_discovery` | yes | yes | via `record_discovery_internal` |
| `start_discovery_session` | yes | yes | validated in the FFI handler before `streaming::start_session` (Issue #1188) |
| `append_discovery_records` | no | n/a | references session by ID; creature is captured at session start |
| `finish_discovery_session` | no | n/a | session ID only |
| `cancel_discovery_session` | no | n/a | session ID only |
| `analyze_parallel` | yes | yes | via `analyze_parallel_internal` |
| `rank_focus_neurons` | yes | yes | via `rank_focus_neurons_internal` |
| `export_visualisation_snapshot` | yes | yes | via `export_visualisation_snapshot_internal` (Issue #1188) |
| `merge_discovery_parquet` | no | n/a | parquet I/O only |
| `read_discovery_records_ffi` | no | n/a | parquet I/O only |
| `get_calibration_summary` | no | n/a | discovery-history JSON only |
| `cleanup_discovery_dir` | no | n/a | filesystem cleanup |
| `clean_orphaned_discovery_dirs` | no | n/a | filesystem cleanup |
| `discovery_memory_usage_bytes` | no | n/a | atomic counter read |
| `cleanup_discovery_lib` | no | n/a | shutdown hook |
| `get_library_version` | no | n/a | constant string |
| `check_gpu_available` | no | n/a | hardware probe |
| `cancel_analysis` / `cancel_analysis_memory_pressure` / `reset_cancellation` / `is_analysis_active` | no | n/a | cancellation flags |
| `free_discovery_result` | no | n/a | memory free |

There are intentionally **no validation-bypassing paths** for creature input.
Any new FFI entry point that accepts a `CreatureJson` must call
`validate_forward_only_synapses` **and** `validate_creature_input_bounds` before
any business logic and update this table.

### Atomic Record Writes

For each discovery record, all data (observations, activations, errors) **must**
come from the same training record:

- The analysis phase matches records by `obs_index`.
- Parallelisation across training records is allowed.
- Per-record atomicity is required (activate → collect all neurons → write
  atomically).
- Never mix data from different training records within a single write.

### FFI Memory Management

Every FFI call returning a `char*` **must** be freed with
`free_discovery_result()`. Failure to do so will leak memory.

### VALUE Domain Errors

All FFI functions return structured JSON with a `success` field. When `success`
is `false`, the `error` field contains a descriptive message. Controllers must
check this field before processing results.

---

## Candidate Types

Reuse existing candidate types whenever possible. If a new type is truly
required, it must be documented in README.md and a corresponding handler added
to NEAT-AI. See [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) for the full
reference with operations and success/failure rates.
