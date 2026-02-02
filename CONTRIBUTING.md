# Contributing to NEAT-AI-Discovery

Thank you for your interest in contributing to NEAT-AI-Discovery! This guide
covers everything you need to get started.

> **For AI agents**: Machine-readable coding conventions and invariants live in
> [AGENTS.md](AGENTS.md). This file is the human-readable contributor guide.

---

## Getting Started

### Prerequisites

**User-installable (automatically handled by `scripts/runlib.sh`):**
- Rust (latest stable version)
- Cargo

**System packages (must be installed by an administrator):**
- **jq** — required for build scripts
- **Build tools (gcc/cc)** — required on Linux:
  - **Ubuntu/Debian**: `sudo apt-get install -y build-essential`
  - **RHEL/CentOS/Amazon Linux**: `sudo yum groupinstall -y "Development Tools" && sudo yum install -y gcc`
  - **Fedora**: `sudo dnf groupinstall -y "Development Tools" && sudo dnf install -y gcc`
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

**GPU requirement**: This library requires a GPU (Metal on macOS, Vulkan on
Linux). There is no CPU fallback. GPU-dependent tests are skipped automatically
on machines without a GPU.

### Building

```bash
./scripts/runlib.sh
```

This script installs Rust and Cargo if missing (no sudo required), builds the
library in release mode, installs it to `~/.cargo/lib/` with version tracking,
and signs it on macOS for FFI compatibility.

### Running Tests

```bash
# Run all tests (unit + integration)
cargo test --lib --tests --all-features -- --test-threads=1

# Run specific test file
cargo test --test <test_name>

# Run tests matching a pattern
cargo test test_hidden_neuron

# Run benchmarks
cargo bench --bench <bench_name>
```

Tests run sequentially (`--test-threads=1`) due to shared global state (deadline
overrides, GPU failure guards, environment variables).

---

## Development Workflow

### Test-Driven Development (TDD)

We follow strict TDD:

1. **Write a failing test** that defines the expected behaviour.
2. **Implement the feature** to make the test pass.
3. **Refactor** if needed while keeping tests green.

### Quality Gate

**Always run `./quality.sh` before committing.** CI treats warnings as errors.

`./quality.sh` performs these checks in order:

1. Bash syntax check (all `.sh` files)
2. `cargo build` (debug, quick feedback)
3. `cargo fmt --all` (auto-formatting)
4. `cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args`
5. `cargo check --all-targets --all-features`
6. `cargo test --lib --tests --all-features -- --test-threads=1`
7. `cargo build --release --lib`

If any step fails, fix the issue and re-run. Do **not** commit code that fails
`./quality.sh`.

### CI Pipeline

GitHub Actions runs on every pull request to `Develop`:

- `auto-format` — applies `rustfmt` and commits fixes
- `version-increment` — auto-bumps patch version when `src/` changes
- `quality` — fmt check, Clippy, cargo check, tests, build
- `shell-checks` — validates bash script syntax
- `spell-check` — runs codespell on the codebase
- `validation` — checks required files and `Cargo.toml`
- `security` — runs security audit workflow

**GPU tests are skipped in CI** (no GPU available). For full coverage, run
`./quality.sh` locally before pushing.

**Do NOT modify `.github/workflows/ci.yml` without explicit approval.**

---

## Code Style

### Australian English

Use Australian English throughout all code, comments, and documentation:

- colour, behaviour, organisation, favour, metre, centre, analyse, minimise,
  optimise, serialise, initialise, utilise, recognise, emphasise, prioritise,
  licence (noun), defence, modelling, travelling

### Formatting and Linting

- **Formatting**: `cargo fmt --all` (applied automatically by `./quality.sh`)
- **Linting**: `cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args`

### Coding Principles

- **KISS** — Favour simplicity; avoid unnecessary complexity.
- **DRY** — Avoid duplication; maintain a single source of truth.
- **Boy Scout Rule** — Leave the code cleaner than you found it.
- **Prefer smaller files** — Favour many smaller, focused source files over
  large monolithic ones (Single Responsibility Principle).
- **Avoid over-engineering** — Only make changes that are directly requested or
  clearly necessary. Do not add features, refactoring, or "improvements" beyond
  what was asked.

### Rust Best Practices

- Follow standard Rust idioms (`Result`, `Option`, pattern matching).
- Use `anyhow` for error handling in application code.
- Prefer Rust standard library features over external crates when possible.
- All dependencies must be Apache-2.0 compatible (see
  [AGENTS.md](AGENTS.md#rust-best-practices) for the full list of allowed
  licences).

---

## Testing Guidelines

### Unit Tests vs Benchmarks

| Concern | Unit Tests | Benchmarks |
|---------|-----------|------------|
| **Purpose** | Verify correctness | Measure performance |
| **Location** | `tests/` (integration) or `src/` with `#[cfg(test)]` | `benches/` |
| **Run with** | `cargo test` | `cargo bench --bench <name>` |
| **Should not** | Measure performance or print timing | Verify correctness |

**Never reduce iteration counts to make "performance tests" faster in unit
tests.** If you need to confirm performance, create proper benchmarks.

### Test Organisation

- **Prefer `tests/` directory** (integration tests) over inline unit tests.
- Only place tests under `src/` when the behaviour cannot be exercised cleanly
  via the public API.
- If unit tests in `src/` grow large, extract them into a dedicated `tests.rs`
  module file.
- Do not make APIs public just for testing.
- Group related tests by concern (e.g., all weight tests in one file).

### Test Outcomes, Not Implementation

Tests verify **what** the system does, not **how** it does it. The same test
should pass regardless of whether we use GPU, CPU, or TPU internally.

```rust
// GOOD: Tests the outcome
#[test]
fn test_low_impact_neurons_are_detected() {
    let creature = create_test_creature();
    let impacts = compute_impacts(&creature);
    assert!(impacts["far-from-output"] < 0.1);
    assert!(impacts["close-to-output"] > 0.9);
}

// BAD: Tests implementation details
#[test]
fn test_gpu_kernel_computes_impacts() {
    // Don't test HOW we compute, test WHAT we compute
}
```

### Test Change Significance

- **New test file** — generally good (more coverage).
- **Modified test** — requires justification (did requirements change?).
- **Removed/skipped test** — red flag; must be justified.

---

## Pull Request Process

### PR Summary File

Every PR must include a summary file at `docs/pr-summary-<ISSUE>.md` containing
the following sections. (Older PR summaries are archived in `docs/archive/`.)

1. **Summary** — brief description of what was changed and why
2. **Evidence** — screenshots for UI changes, benchmark results for performance
   changes, or test references for bug fixes
3. **Test Plan** — list of tests added or modified

### Commit Messages

- Reference the issue number (e.g., `Add CONTRIBUTING.md (#372)`)
- Keep the first line concise (under 72 characters)
- Use the imperative mood ("Add feature" not "Added feature")

### Version Management

Do not manually bump versions. CI increments `Cargo.toml` patch versions when
`src/` changes are detected. Call `get_library_version()` to confirm what a
worker has loaded.

### Deployment Checklist

1. Run `./quality.sh` in this repository
2. If changes affect NEAT-AI integration, also run `./quality.sh` in the
   NEAT-AI repository
3. Verify all tests pass before committing

---

## Project Structure

```
src/
├── lib.rs                    # FFI entry points (no_mangle extern "C")
├── types.rs                  # Core type definitions
├── activations.rs            # Activation function calculations
├── record.rs                 # Discovery data recording
├── streaming.rs              # Streaming session management
├── parquet_format.rs         # Parquet I/O
├── export.rs                 # Data export utilities
├── discovery_history.rs      # Historical tracking
├── debug.rs                  # Debugging utilities
├── observability.rs          # Observability / logging
├── intern.rs                 # Neuron UUID interning
├── watchdog.rs               # Signal handling (SIGUSR1)
├── focus.rs                  # Focus neuron selection
│
├── analysis/                 # Core analysis engine
│   ├── mod.rs                # Module organisation
│   ├── shared.rs             # Common types, results, diagnostics
│   ├── synapse.rs            # Synapse analysis
│   ├── neuron.rs             # Neuron analysis
│   ├── activation.rs         # Activation function analysis
│   ├── samples.rs            # Sample data structures
│   ├── gpu/                  # GPU infrastructure
│   └── utils/                # Utilities (memory, deadline, platform)
│
└── shaders/                  # WGSL compute shaders
```

| Directory | Purpose |
|-----------|---------|
| `tests/` | Integration tests |
| `benches/` | Criterion benchmarks |
| `examples/` | Standalone examples |
| `scripts/` | Build and install helpers |
| `docs/` | Supplementary documentation and PR summaries |

For the full source layout with all files, see
[AGENTS.md — Source Layout](AGENTS.md#source-layout).

---

## Further Reading

- [README.md](README.md) — User-facing documentation, FFI API, troubleshooting
- [AGENTS.md](AGENTS.md) — Detailed coding conventions for AI agents
- [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md) — Discovery type reference
- [docs/IMPACT_CALCULATION.md](docs/IMPACT_CALCULATION.md) — Impact scoring algorithm
- [CHANGELOG.md](CHANGELOG.md) — Version-by-version history
