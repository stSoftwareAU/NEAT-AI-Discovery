# AGENTS.md — Agent-Only Notes for AI Coding Agents

This is a **thin pointer file**. The shared conventions, architecture, testing
doctrine and quality gate now live in the human docs — this file carries only
the invariants and rules that are specific to AI agents working in this repo.

- **User-facing overview, GPU requirements, documentation index** —
  [README.md](README.md).
- **Coding conventions, testing philosophy, quality gate, CI pipeline,
  version management, project structure** —
  [CONTRIBUTING.md](CONTRIBUTING.md). It is the canonical home for that
  material; do not re-copy it here.

Read both before making changes. The sections below are the delta an agent
must know on top of them.

---

## Where things live

| Topic | Canonical home |
|-------|----------------|
| Project mission, GPU requirement, docs index | [README.md](README.md) |
| Coding conventions (KISS/DRY, Australian English) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Testing philosophy ("what" vs "how", benchmarks vs tests) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Quality gate (`./quality.sh`) and CI pipeline | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Build/install (`./scripts/runlib.sh`) and versioning | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Source layout | `src/` (the tree is authoritative; do not mirror it here) |
| FFI API reference | [docs/FFI_API.md](docs/FFI_API.md) |
| Candidate types | [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) |
| Environment variables | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| GPU tuning | [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) |

The source tree changes constantly — read `src/` directly rather than trusting
any hand-maintained mirror. There is intentionally no directory listing in this
file.

---

## CI is off-limits

**Do NOT modify `.github/workflows/ci.yml` without explicit human approval.**
CI treats warnings as errors; always run `./quality.sh` locally before
committing (see [CONTRIBUTING.md](CONTRIBUTING.md) for the full gate).

---

## Key Invariants

These invariants **must not** be violated by any code change.

### Forward-only Activation Order

Discovery assumes **forward-only** networks (no recurrent feedback):

- A neuron may only read activations from **earlier** neurons in the creature's
  evaluation order.
- Synapses must point from an earlier neuron to a later neuron.
- New neurons must be inserted at the correct index (not appended).
- No cross-sample state — each recorded activation/error is for a single
  training sample.

#### Validated FFI Surface (Issue #1184, #1188)

`validate_forward_only_synapses` (in `src/ffi_types/forward_only_validation.rs`)
is the defence-in-depth gate. It runs immediately after JSON deserialisation and
before any business logic on every FFI entry point that accepts a
`CreatureJson`. A violation returns a structured `DiscoveryError::InvalidInput`
with `error_kind: "data_validation"`.

| FFI entry point | Accepts `CreatureJson` | Validates | Notes |
|-----------------|------------------------|-----------|-------|
| `record_discovery` | yes | yes | via `record_discovery_internal` |
| `start_discovery_session` | yes | yes | validated in the FFI handler before `streaming::start_session` (Issue #1188) |
| `append_discovery_records` | no | n/a | references session by ID; creature captured at session start |
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
`validate_forward_only_synapses` before any business logic and update this table.

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
`free_discovery_result()`. Failure to do so leaks memory.

### VALUE Domain Errors

All FFI functions return structured JSON with a `success` field. When `success`
is `false`, the `error` field contains a descriptive message. Controllers must
check this field before processing results.

---

## GPU Requirement

**This library requires a GPU.** There is no CPU fallback. See
[README.md — Minimum System Requirements](README.md#minimum-system-requirements)
for hardware requirements and [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) for tuning
and troubleshooting.

---

## Candidate Types

Reuse existing candidate types whenever possible. A genuinely new type must be
documented in README.md and given a corresponding handler in NEAT-AI. The full
reference — with operations, success/failure rates and descriptions — lives in
[docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md).
