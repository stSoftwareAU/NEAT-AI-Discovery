# 🤝 Contributing to NEAT-AI-Discovery

Thank you for your interest in contributing to NEAT-AI-Discovery! This guide
covers everything you need to get started.

> **For AI agents**: Machine-readable coding conventions and invariants live in
> [AGENTS.md](AGENTS.md). This file is the human-readable contributor guide.

---

## 🚀 Getting Started

### 📋 Prerequisites

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

### 🔨 Building

```bash
./scripts/runlib.sh
```

This script installs Rust and Cargo if missing (no sudo required), builds the
library in release mode, installs it to `~/.cargo/lib/` with version tracking,
and signs it on macOS for FFI compatibility.

### 🧪 Running Tests

```bash
# Run all tests (unit + integration)
cargo test --lib --tests --all-features -- --test-threads=2

# Run specific test file
cargo test --test <test_name>

# Run tests matching a pattern
cargo test test_hidden_neuron

# Run benchmarks
cargo bench --bench <bench_name>
```

Tests that mutate shared global state (environment variables, deadline overrides,
watchdog) are marked with `#[serial]` from the `serial_test` crate and will not
run concurrently with each other. All other tests run in parallel (`--test-threads=2`).

---

## 💻 Development Workflow

### 🔴🟢🔵 Test-Driven Development (TDD)

We follow strict TDD:

1. **Write a failing test** that defines the expected behaviour.
2. **Implement the feature** to make the test pass.
3. **Refactor** if needed while keeping tests green.

### ✅ Quality Gate

**Always run `./quality.sh` before committing.** CI treats warnings as errors,
so do not skip this step. `./quality.sh` performs these checks in order:

1. `./quality/bash_syntax.sh` — `bash -n` syntax gate over every `.sh` file
   (Issue #1755); the same committed script CI runs on pull requests
2. `shellcheck` over every `.sh` file (hard-fails if `shellcheck` is not
   installed)
3. `./scripts/check-pr-summary-location.sh` — PR summaries must stay in
   `docs/archive/pr-summaries/` (Issue #1613)
4. `cargo upgrade --incompatible` + `cargo update` (dependency upgrade)
5. `cargo deny check` (licence and dependency audit)
6. `cargo build` (debug, quick feedback)
7. `cargo fmt --all` (auto-formatting)
8. `cargo clippy --all-targets --all-features -- -D warnings`
9. `cargo check --all-targets --all-features`
10. `cargo test --lib --tests --all-features -- --test-threads=2`
11. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` (documentation build)
12. `cargo build --release --lib`

If any step fails, fix the issue and re-run. Do **not** commit code that fails
`./quality.sh`.

**GPU tests are skipped in CI** (no GPU available). For full coverage, run
`./quality.sh` locally before pushing.

#### CI Pipeline

GitHub Actions runs the checker on every pull request into `Develop` **and**
into `milestone/*` feature branches (`.github/workflows/ci.yml`, Issue #1651), so
the gate runs on milestone sub-issue PRs too, not just the rollup into `Develop`:

- `auto-format` — applies `rustfmt` and commits fixes
- `version-increment` — auto-bumps the patch version on every PR when changes
  exist (uses the `ACTIONS_PUSH` PAT so the push re-triggers workflows)
- `quality` — fmt check, Clippy, cargo check, doc build, tests, build
- `spell-check` — runs codespell on the codebase
- `validation` — checks required files and `Cargo.toml`
- `security` — runs the security audit workflow
- `shellcheck` (separate workflow `.github/workflows/shellcheck.yml`) — runs the
  committed `quality/bash_syntax.sh` (`bash -n`) gate, then lints bash scripts
  via ShellCheck (Issue #1755)

**Do NOT modify `.github/workflows/ci.yml` without explicit approval.**

---

## 🎨 Code Style

### Language

**Use Australian English** throughout all code, comments, and documentation:
colour, behaviour, organisation, favour, metre, centre, analyse, minimise,
optimise, serialise, initialise, utilise, recognise, emphasise, prioritise,
licence (noun), defence, modelling, travelling, programme (except "program" in
computing contexts).

### Principles

- **KISS** — Favour simplicity; avoid unnecessary complexity.
- **DRY** — Avoid duplication; maintain a single source of truth.
- **Boy Scout Rule** — Leave the code cleaner than you found it.
- **Prefer smaller files** — Favour many smaller, focused source files over large
  monolithic ones (Single Responsibility Principle). Target individual source
  files under ~1,500 lines; consider splitting at ~2,000 lines.
- **Separate concerns** — GPU infrastructure, business logic, and types belong in
  separate files.

### Rust Best Practices

- Follow standard Rust idioms (`Result`, `Option`, pattern matching).
- Use `anyhow` for error handling in application code.
- Prefer Rust standard library features over external crates when possible.
- All dependencies must be Apache-2.0 compatible (see the `[licenses]` allow-list
  in [`deny.toml`](deny.toml), enforced by `cargo deny check`).

### Avoid Over-engineering

- Only make changes that are directly requested or clearly necessary.
- Do not add features, refactor code, or make "improvements" beyond what was
  asked.
- Do not add error handling for scenarios that cannot happen.
- Do not create helpers or abstractions for one-time operations.
- Do not add docstrings, comments, or type annotations to code you did not
  change.

---

## 🧪 Testing Guidelines

**The quality of tests is what makes a good system.** (For the TDD loop itself,
see the Test-Driven Development steps under Development Workflow above.)

### Test Outcomes, Not Implementation ("What" vs "How")

Tests verify **what** the system does, not **how** it does it. The same test
should pass regardless of whether we use GPU, CPU, or TPU internally.

**"What" tests (GOOD)** — test observable outcomes:
- Call a function with test data, assert on the result.
- Verify the system produces correct candidates, detects patterns, returns the
  expected JSON structure, etc.
- Will still pass if we switch quick sort to bubble sort, HashMap to BTreeMap, or
  GPU to CPU.

**"How" tests (BAD)** — test implementation details:
- Assert that a specific internal function is called.
- Check that a particular data structure is used internally.
- Verify iteration order, internal cache state, or call counts for non-observable
  behaviour.
- Break on any refactor even when behaviour is unchanged.

**Benchmarks disguised as tests (BAD)** — measure performance in unit tests:
- Loop N times and assert on elapsed time.
- Compare durations between two code paths.
- Use `Instant::now()` / `.elapsed()` to validate speed.
- These always produce unreliable results because tests run in parallel with
  other system activity.

### Unit Tests vs Benchmarks

| Concern | Unit Tests | Benchmarks |
|---------|-----------|------------|
| **Purpose** | Verify correctness | Measure performance |
| **Location** | `tests/` (integration) or `src/` with `#[cfg(test)]` | `benches/` |
| **Run with** | `cargo test` | `cargo bench --bench <name>` |
| **Asserts on** | Results, structure, correctness | Timing, throughput |
| **Must not** | Use `Instant`/`elapsed` for pass/fail | Verify correctness |

Put timing assertions in `benches/`, not `tests/`. **Never reduce iteration
counts to make "performance tests" faster in unit tests** — write a proper
benchmark instead.

### Test Organisation

- **Prefer the `tests/` directory** (integration tests) over inline unit tests.
- Only place tests under `src/` when the behaviour cannot be exercised cleanly
  via the public API.
- Do not make APIs public just for testing.
- Group related tests by concern (e.g. all weight tests in one file).

Tests that mutate shared global state (environment variables, deadline
overrides, watchdog) are marked with `#[serial]` from the `serial_test` crate.
GPU-dependent tests include `skip_without_gpu!()` and are skipped automatically
on machines without a GPU.

---

## ⚙️ Environment Variables — One Source of Truth

Every `NEAT_AI_DISCOVERY_*` environment variable is documented in **exactly one
place**: [docs/CONFIGURATION.md](docs/CONFIGURATION.md). `README.md` and
`AGENTS.md` only link to it. When you add, rename, or change the default of a
variable, update `docs/CONFIGURATION.md` — do **not** re-copy the table into any
other file. Duplicated tables drift apart (Issue #1611); a single reference
cannot.

---

## 📬 Pull Request Process

### 📝 PR Summary File

Every PR must include a summary file at `docs/archive/pr-summaries/pr-summary-<ISSUE>.md`
containing the following sections.

1. **Summary** — brief description of what was changed and why
2. **Evidence** — screenshots for UI changes, benchmark results for performance
   changes, or test references for bug fixes
3. **Test Plan** — list of tests added or modified

### 👥 Code Owners & Branch Protection

High-blast-radius paths are owned by the admin maintainers
(`@Green-Beret @nleck @stservice`) in
[`.github/CODEOWNERS`](.github/CODEOWNERS): the CI workflows (which hold the
`ACTIONS_PUSH` PAT plus `SEMGREP_APP_TOKEN` and `CODECOV_TOKEN`), the
dependency manifests (`Cargo.toml` / `Cargo.lock`), and the security policy.
A pull request touching any of these requires maintainer review. Individual
maintainers are named (rather than a team) because no org team holds direct
write access to this repo, so a team owner would not enforce; switch to a team
reference once one is granted write access.

`CODEOWNERS` only takes effect once branch protection enforces it. A repository
admin must enable the following on the default branch (`Develop`):

- **Require a pull request before merging** — with **Require review from Code
  Owners**.
- **Block direct pushes and force-pushes** to the protected branch.
- **Require linear history** (no merge commits).
- Confirm the required status checks (the `quality.sh` gate) are green before
  merge.

As defence-in-depth, consider **Require signed commits**. An admin can apply
these settings via **Settings → Branches → Branch protection rules**, or with
the GitHub CLI (`gh api -X PUT repos/stSoftwareAU/NEAT-AI-Discovery/branches/Develop/protection ...`).
These are repository-level settings that cannot be committed as files.

### 💬 Commit Messages

- Reference the issue number (e.g., `Add CONTRIBUTING.md (#372)`)
- Keep the first line concise (under 72 characters)
- Use the imperative mood ("Add feature" not "Added feature")

### 🔢 Version Management

CI auto-increments the `Cargo.toml` patch version on **every pull request**
(unless the PR branch already carries a bump), not only when `src/` changes — so
in the normal PR workflow you do not need to bump it yourself. If you commit
**directly** (outside the PR workflow, where CI does not run), you must manually
increment the patch version. The authoritative version-bump policy lives in the
README [Distributed Build & Versioning](README.md#-distributed-build--versioning)
section. Call `get_library_version()` to confirm what a worker has loaded.

### 🚀 Deployment Checklist

1. Run `./quality.sh` in this repository
2. If changes affect NEAT-AI integration, also run `./quality.sh` in the
   NEAT-AI repository
3. Verify all tests pass before committing

---

## 🏗️ Project Structure

The crate produces both a `cdylib` (for FFI via Deno) and an `rlib` (for Rust
integration). Rather than maintain a hand-written mirror of the tree — which
drifts as files move (Issue #1683) — browse the source directly:

- **`src/`** — the library source. Key areas: `src/ffi/` (FFI entry points),
  `src/ffi_types/` (JSON request/response structs), `src/analysis/` (the core
  analysis engine, including `detection/`, `recommendation/`, and `scoring/`),
  `src/focus/` (focus-neuron selection), and `src/parquet_format/` (Parquet I/O).
- **The module column of [docs/DISCOVERY_TYPES.md](docs/DISCOVERY_TYPES.md)** —
  the authoritative map from each discovery type to its source module.
- **`tests/`** — integration tests, **`benches/`** — Criterion benchmarks,
  **`examples/`** — standalone examples, **`scripts/`** — build/install helpers
  (`runlib.sh`), **`docs/`** — supplementary documentation.

See `Cargo.toml` for the full dependency list.

---

## 📚 Further Reading

See the [Additional Documentation](README.md#additional-documentation) table in
README.md for a comprehensive index of all project documentation.
