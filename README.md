# NEAT-AI-Discovery

A high-performance Rust companion library for
[`stSoftwareAU/NEAT-AI`](https://github.com/stSoftwareAU/NEAT-AI). It records
neuron activations and errors during discovery runs, then analyses the captured
samples to recommend structural upgrades (new synapses or neurons) that reduce
error. Controllers call into the library via Deno FFI to power
`Creature.discoveryDir()` workflows.

## Why use this library?

- **Production-ready discovery** – Handles millions of observations without the
  memory blow-outs that limit the TypeScript implementation.
- **Single-file artefacts** – Writes per-run Parquet files so results are easy to
  transfer, archive, or inspect with standard tooling.
- **Drop-in for NEAT-AI** – Exposes the `libneat_ai_discovery` symbol set expected
  by the TypeScript bindings in `NEAT-AI`.

## Quick start

1. Install prerequisites (`rustup`, `cargo`, build tools, and `jq`). The
   `scripts/runlib.sh` helper will guide you if anything is missing.
2. Build the library:
   ```bash
   cargo build --release --lib
   # or use the helper that also installs into ~/.cargo/lib
   ./scripts/runlib.sh
   ```
3. Confirm the artefact exists (`target/release/libneat_ai_discovery.*`).
4. Run the quality gate before committing:
   ```bash
   ./quality.sh
   ```

## Using the library with NEAT-AI

1. Place the compiled artefact where Deno can load it:
   - Copy `libneat_ai_discovery.*` into `~/.cargo/lib`, **or**
   - Export `NEAT_AI_DISCOVERY_LIB_PATH=/absolute/path/to/libneat_ai_discovery.*`.
2. Grant FFI permissions when running discovery jobs:
   ```bash
   deno run --allow-env --allow-ffi --allow-read your-script.ts
   ```
3. From your controller, guard calls with
   `isRustDiscoveryEnabled()` so the job fails fast if the module cannot be
   loaded.
4. Follow the end-to-end discovery orchestration documented in the
   [`DiscoveryDir` guide](https://github.com/stSoftwareAU/NEAT-AI/blob/main/docs/DiscoveryDir.md).
   The guide covers safe-write practices, worker loops, and how to persist the
   improved creatures that this library exports.

## Verifying the installation

Use the NEAT-AI helper script after copying the library:

```bash
cd /path/to/NEAT-AI
./scripts/check_discovery.ts
```

If the script reports that discovery is enabled, you are ready to schedule
`Creature.discoveryDir()` jobs against your sampled datasets. Otherwise revisit
`NEAT_AI_DISCOVERY_LIB_PATH` and the permissions passed to `deno run`.

## Troubleshooting

- **Library not found**: Double-check the artefact path, file extension (e.g.
  `.dylib` on macOS, `.so` on Linux), and `NEAT_AI_DISCOVERY_LIB_PATH`.
- **FFI permission errors**: Ensure discovery workers launch with
  `--allow-ffi --allow-env --allow-read --allow-write` and only point to trusted
  library locations.
- **Empty Parquet output**: Confirm the caller supplies the sampled discovery
  dataset and that each record bundles observations, activations, and errors for
  the same training index.

## Existing reference material

The sections below capture the original project brief, scale targets, and
engineering standards. They remain authoritative for contributors and are linked
here for convenience:

- [Project goal](#goal)
- [Problem statement](#problem-statement)
- [Performance requirements](#performance-requirements)
- [Features](#features)
- [Development guidelines](#development)
- [File format](#file-format)
- [JSON interface](#json-interface)
- [Code quality expectations](#code-quality)
- [Cross-platform support](#cross-platform-support)
- [Distributed build & versioning](#distributed-build--versioning)

---

## Goal

The goal is to record neuron activations and errors during the discovery
training phase, then scan this recorded data to identify beneficial new
synapses/neurons that would reduce error. **The current DenoJS implementation has
severe performance and memory issues that make discovery unviable for larger
models.** This Rust library must solve these performance/memory problems while
maintaining the same functional behavior.

## Problem Statement

The current DenoJS implementation requires extreme filtering of the training
data (millions of records) to make discovery work in reasonable time. The
DenoJS has severe performance and memory issues that make discovery unviable
for larger models. This library aims to solve these problems while maintaining
the same functional behavior.

**Target Scale:**
- Training records: Millions (not hard-coded, but that's the scale)
- Observations per record: 1,486 (float32 values - this is the input size)
- Neurons: 447
- Synapses: 16,012

## Performance Requirements

- Must handle millions of training records efficiently without memory issues
- Must process significantly more data than DenoJS can handle (DenoJS requires extreme filtering to work)
- Must be significantly faster than TypeScript implementation
- Must use minimal memory (avoid loading all data into memory at once)
- Files are temporary (deleted after discovery phase)
- Only needs compatibility within the Rust discovery phase
- Goal: Process full dataset (or much more) compared to filtered subset in DenoJS

## Features

- Record neuron activations and errors during discovery training phase
- Single Parquet file format (eliminates many-small-files problem)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging
- Cross-platform support (macOS, Ubuntu, AWS Linux)

## Development

### Development Guidelines

**IMPORTANT: All development must follow these mandatory practices:**

1. **Test-Driven Development (TDD)**: Always write tests first before implementing features
   - Write a failing test for the new feature
   - Implement the feature to make the test pass
   - Refactor if needed while keeping tests green
   - All new tests should pass after implementation
   - **Always read this README before making any changes**

2. **Code Quality Enforcement**: **MUST run quality checks after EVERY code change**
   - **CRITICAL**: Execute `./quality.sh` after making ANY code modifications
   - This script runs formatting, linting, type checking, and all tests
   - Fix all linting issues automatically before committing
   - Ensure code formatting and quality standards are maintained
   - **Never commit code without running `./quality.sh` first**

### Prerequisites

**User-installable (automatically handled by `runlib.sh`):**
- Rust (latest stable version) - automatically installed by `runlib.sh` if missing
- Cargo - automatically installed by `runlib.sh` if missing

**System packages (must be installed by administrator):**
- **jq** - must be installed system-wide (required for build scripts)
- **Build tools (gcc/cc)** - required on Linux systems:
  - **Ubuntu/Debian**: `sudo apt-get install -y build-essential`
  - **RHEL/CentOS/Amazon Linux**: `sudo yum groupinstall -y "Development Tools" && sudo yum install -y gcc`
  - **Fedora**: `sudo dnf groupinstall -y "Development Tools" && sudo dnf install -y gcc`
- **macOS**: Xcode Command Line Tools (typically already installed, or can be installed via `xcode-select --install` without sudo)

### Building

```bash
cargo build
```

Build library for release:

```bash
cargo build --release --lib
```

### Building with runlib.sh

The library can be built and installed using the `scripts/runlib.sh` script:

```bash
./scripts/runlib.sh
```

This will build the library and install it to `~/.cargo/lib/` with version tracking.

**Note:** The script automatically installs Rust and Cargo if missing (no sudo required). However, system packages must be installed by an administrator:
- **jq** must be installed system-wide
- **Build tools (gcc/cc)** must be installed on Linux systems (see Prerequisites above)
- If build tools are missing, the script will display clear error messages with installation instructions for the administrator

### Testing

```bash
# Run all tests
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests only
cargo test --test '*'
```

## File Format

### Single Parquet File

Instead of many small CSV files (one per neuron), we use a single Parquet file:

- File location: `.discovery/{creature_uuid}_{random}/discovery_data.parquet`
- Schema:
  - `obs_index: u32` - Observation index (training record index) for ordering
  - `neuron_uuid: string` - Neuron identifier
  - `value: f32` - Neuron value (optional, can be null)
  - `activation: f32` - Neuron activation
  - `errors: list<f32>` - Array of error values

**Benefits:**
- Single file handle (eliminates small-file problems)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging
- Good performance for single file (overhead acceptable)
- Widely supported format

### Debugging Parquet Files

Parquet files can be viewed with standard tools:

**Python:**
```python
import pandas as pd
df = pd.read_parquet('discovery_data.parquet')
print(df.head())
```

**DuckDB:**
```sql
SELECT * FROM 'discovery_data.parquet' LIMIT 10;
```

**Command-line:**
- `parquet-tools` (Java-based)
- `parquet-cli` (Rust-based)

Many data tools support Parquet natively (Tableau, Apache Spark, etc.)

## CRITICAL REQUIREMENT: Atomic Record Writes

**For each discovery record, all data (observations, activations, errors) MUST come from the same training record.** This is essential because:
- The analysis phase matches records by index (record 0 from neuron A corresponds to record 0 from neuron B)
- When evaluating synapse candidates, records must align - all data in record i must be from the same training record
- If observations, activations, and errors don't line up from the same training record, analysis will be incorrect

**Implementation Requirements:**
- **Atomic writes**: For each training record, activate creature, collect ALL neuron data (activations, errors), then write ALL neuron rows together
- **Parallelisation allowed**: Since training dataset is already randomised, we CAN process different training records in parallel
- **Per-record atomicity**: Each parallel task must process one complete training record (activate → collect all neurons → write all neurons atomically)
- **Cross-neuron alignment**: Records with the same `obs_index` across different neurons correspond to the same training record
- **No mixing**: Never mix data from different training records within a single discovery record write
- **Matching by obs_index**: TypeScript matches records across neurons by `obs_index` (not by array position), so record order from Rust doesn't matter

## JSON Interface

### Input Format

```json
{
  "creature": {
    "neurons": [
      {
        "uuid": "hidden-1",
        "type": "hidden",
        "squash": "TANH",
        "bias": 0.0
      }
    ],
    "synapses": [
      {
        "from_uuid": "input-0",
        "to_uuid": "hidden-1",
        "weight": 0.5
      }
    ],
    "input": 20,
    "output": 2
  },
  "training_data": [
    {"input": [0.1, 0.2, ...], "output": [0.5, 0.3]},
    ...
  ],
  "temp_dir": ".discovery/abc123_456789",
  "binary_file_path": "/path/to/binary.bin",  // optional
  "record_indices": [0, 5, 10, ...],  // optional
  "timeout_seconds": 300  // optional
}
```

### Output Format

Success:
```json
{
  "success": true,
  "temp_dir": ".discovery/abc123_456789",
  "file": "discovery_data.parquet"
}
```

Error:
```json
{
  "success": false,
  "error": "Error message here"
}
```

## Code Quality

```bash
# Format code
cargo fmt

# Lint code
cargo clippy

# Check code
cargo check

# Run quality checks
./quality.sh
```

## Cross-Platform Support

The library must work on:
- **macOS** (primary target)
- Ubuntu
- AWS Linux (x86_64 and ARM64)

All dependencies build automatically on remote, unattended machines.

## Distributed Build & Versioning

- Versions are managed in `Cargo.toml` and are automatically incremented by CI on pull requests when files in `src/` change.
- Local and remote runs use a distributed build pattern via `scripts/runlib.sh`:
  - The library is installed to `~/.cargo/lib/` and tracked with a version marker at `~/.cargo/lib/.neat_ai_discovery.version`.
  - On run, if the installed version differs from `Cargo.toml`, the library is rebuilt and reinstalled; otherwise it runs silently without rebuilding.
- Do not manually edit version numbers; CI handles patch bumps when source changes are detected.

## License

This project is licensed under the terms specified in the LICENSE file.
