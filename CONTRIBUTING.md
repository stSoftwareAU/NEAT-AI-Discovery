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

#### Build profiles (Issue #2017)

Cargo profiles live in the root [`Cargo.toml`](Cargo.toml) and follow the
fleet rule from [VibeCoding#4159](https://github.com/stSoftwareAU/VibeCoding/issues/4159):

| Profile | Goal | Settings |
|---------|------|----------|
| `dev` (`cargo build`) | Compile as fast as possible | `debug = "line-tables-only"` (panic file:line kept; full DWARF dropped). Default `opt-level = 0` and incremental stay. |
| `release` (`cargo build --release` / `./scripts/runlib.sh`) | Most optimised artefact | `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`. Compile time is irrelevant. |

Stable Rust only — no nightly, no `-Zthreads`, no Cranelift.
`-C target-cpu=native` is **not** set: this crate ships a `cdylib`/`rlib`
consumed via Deno FFI on other hosts, not a same-host binary.

The Rust bootstrap goes through `./scripts/install-rustup.sh`, which downloads
the pinned `rustup-init` binary for the host target and executes it **only**
when its SHA-256 matches the digest committed in `scripts/rustup-init.sha256`.
Nothing is ever piped from the network into a shell: a mismatch, a failed
download, or an unpinned host target aborts non-zero without executing the
downloaded file (Issue #1911). To bump rustup, change `RUSTUP_VERSION` in
`scripts/install-rustup.sh` and replace every digest in
`scripts/rustup-init.sha256` with the corresponding `rustup-init.sha256`
published by the Rust project for the new version — the two files must move
together.

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

### 🔌 Cargo Features

Test scaffolding must not ship in the release `cdylib`/`rlib`. Such modules live
behind **off-by-default** cargo features, so `cargo build --release --lib` never
compiles them (Issue #1877):

| Feature | Gates | Consumer |
|---------|-------|----------|
| `regression-harness` | `src/analysis/production_discovery_regression.rs` | `tests/production_discovery_regression.rs` |

Always run tests with `--all-features` (as `./quality.sh` and CI do) — a plain
`cargo test` **skips** targets that declare `required-features`. To run a gated
suite on its own, name the feature:

```bash
cargo test --features regression-harness --test production_discovery_regression
```

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
2. `./quality/shellcheck.sh` — ShellCheck lint over every `.sh` file
   (Issue #1898); the same committed script CI runs on pull requests
   (hard-fails if `shellcheck` is not installed)
3. `./quality/cargo_install_pinning.sh` — every `cargo install` call site must
   pass both `--locked` and `--version` (Issue #1912, enforcing Issue #1223)
4. `./scripts/check-pr-summary-location.sh` — PR summaries must stay in
   `docs/archive/pr-summaries/` (Issue #1613)
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

**The gate never upgrades dependencies** (Issue #1865). It used to run
`cargo upgrade --incompatible` + `cargo update`, which pulled crates published
minutes earlier and bypassed the 24h quarantine window. Bump dependencies with
`./bump-deps.sh` — it age-checks every change to the resolved `Cargo.lock`,
transitive packages included, plus every dependency table of every tracked
manifest (`[build-dependencies]`, `[target.<spec>.*]` and `fuzz/Cargo.toml`
included, Issue #1908) — or let Renovate raise the PR. `./bump-deps.sh`
requires `cargo-deny`: its audit gate exits 9 rather than skipping when the tool
is missing, so the bump can never pass unaudited (Issue #1870).

**GPU tests are skipped in CI** (no GPU available). For full coverage, run
`./quality.sh` locally before pushing.

#### CI Pipeline

GitHub Actions runs the checker on every pull request into `Develop` **and**
into `milestone/*` feature branches (`.github/workflows/ci.yml`, Issue #1651), so
the gate runs on milestone sub-issue PRs too, not just the rollup into `Develop`:

- `auto-format` — applies `rustfmt` and commits fixes
- `version-increment` — auto-bumps the patch version on every PR when changes
  exist (uses the `ACTIONS_PUSH` PAT so the push re-triggers workflows). It
  refreshes only this crate's own `Cargo.lock` entry (`cargo update
  --workspace`) and fails loud if any dependency resolution moves — dependency
  bumps go through `./bump-deps.sh` or Renovate, which enforce the quarantine
  window (Issue #1878)
- `quality` — fmt check, Clippy, library build (`cargo build --lib`), an
  intermediate-artefact cleanup, then tests. It runs **neither** `cargo check`
  nor a doc build: no workflow builds the docs (see
  [`docs/ci-doc-build-step.md`](docs/ci-doc-build-step.md)), so run
  `./scripts/doc-check.sh` — or the full `./quality.sh` — locally. Its test
  command is
  `cargo test --lib --tests --bins --all-features --verbose -- --test-threads=2`
  — the extra `--bins` and `--verbose` are the only delta from the documented
  local command, and `--bins` selects nothing because the crate declares no
  `[[bin]]` targets
- `spell-check` — runs codespell on the codebase. codespell is installed with
  `pip install --user --require-hashes -r
  .github/requirements/codespell-requirements.txt`, so the version and the
  SHA-256 of every artefact are pinned; Renovate's `pip_requirements` manager
  keeps that file current under the standard 24h quarantine (Issue #1913)
- `validation` — checks required files, `Cargo.toml`, and documentation (its
  `Check documentation` step asserts `README.md` is not a stub and warns when
  `src/lib.rs` carries no `///` comments — it does **not** build the docs)
- `security` — runs the security audit workflow: `cargo audit` (RustSec
  advisories), `cargo deny check` (the `deny.toml` licence, ban and
  dependency-source policy, enforced in CI since Issue #1870), and
  `dependency-review`
- `shellcheck` (separate workflow `.github/workflows/shellcheck.yml`) — runs the
  committed `quality/bash_syntax.sh` (`bash -n`) gate, then the committed
  `quality/shellcheck.sh` lint gate (Issues #1755, #1898). The ShellCheck binary
  is installed straight from upstream `koalaman/shellcheck` releases by
  SHA-pinned `taiki-e/install-action` — no third-party wrapper action
- `renovate-config-validator` (separate workflow
  `.github/workflows/renovate-validate.yml`) — runs the upstream
  `renovate-config-validator --strict` on PRs that touch `renovate.json`
  (Issue #1916). `renovate.json` carries the 24h supply-chain quarantine, and a
  deprecated or removed config key does not fail loudly: Renovate either
  rejects the file or silently treats the rule as non-matching, so a control can
  disappear without any signal. `--strict` also fails on keys Renovate would
  otherwise auto-migrate. Note that `packageRules` **ordering is load-bearing**
  — the last matching rule wins, so the internal `stSoftwareAU/*` bypass must
  stay the final entry
- `Coverage` (separate workflow `.github/workflows/cargo-quality.yml`) — builds
  and uploads Codecov coverage. It carries no fmt or Clippy step: those live in
  `ci.yml/quality` and running them twice doubled CI time for an identical
  result (Issue #1636)
- `Markdown Lint` (separate workflow `.github/workflows/markdown-lint.yml`) —
  `markdownlint-cli2` over every `**/*.md`, configured by
  `.markdownlint-cli2.jsonc`
- `Semgrep` (separate workflow `.github/workflows/semgrep.yml`) — SAST scan over
  the tree, in a SHA-pinned `semgrep/semgrep` container
- `Gitleaks` (separate workflow `.github/workflows/gitleaks.yml`) — secret-scans
  the PR diff
- `actionlint` (separate workflow `.github/workflows/actionlint.yml`) — lints
  every workflow file under `.github/workflows/`, so a bad expression or an
  unsupported runner label fails on the PR rather than after merge (Issue #1292)

Every one of these separate workflows triggers on `pull_request` against both
`"*"` and `milestone/*` — the `*` glob does not cross `/`, so milestone
sub-issue PRs need the explicit second pattern or the gate skips them.

Every job that needs Rust installs it with the committed
`./scripts/install-rust-toolchain.sh [TOOLCHAIN] [COMPONENT...]`, which drives
the runner's preinstalled `rustup` and retries transient failures. It replaced
`dtolnay/rust-toolchain`, whose tarball was fetched from `codeload.github.com`
during the runner's *Prepare all required actions* phase — a fetch with a fixed
100 s timeout and a 3-attempt retry policy that a workflow cannot tune, so a
codeload stall failed the job before any repository code ran (Issue #1891).

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

### Cite Code by Symbol, Never by Line Number (Issue #1942)

When documentation points at a call site, cite it as `<file>.rs::<function>` —
never `<file>.rs:<line>`. Line numbers rot on the next refactor, silently, and a
whole catalogue of them rotted at once in `docs/COST_FUNCTION_NOTES.md`. A symbol
survives the refactor, and a test can verify the symbol still exists; nothing can
verify a stale line number. This applies to every doc, comment, and PR summary in
the repository, not just the catalogue where the rot was found.

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

**A process-wide singleton needs an injectable value seam — `#[serial]` alone is
not enough (Issues #1929, #1930).** `#[serial]` orders tests; it does not undo
what one of them latched. The GPU circuit breaker is a one-way latch, so the
first test to trip the global instance refused GPU work for every test that ran
afterwards and cascaded into 12 unrelated failures. Serialising them changed
nothing, because the damage outlives the test that caused it.

So write the singleton as a **value** and let production hold the global:

- Production calls `global_gpu_breaker()` — one instance, so a trip anywhere
  stops GPU work everywhere.
- Tests construct their own `GpuCircuitBreaker` and point the subject at it, so
  a tripped state cannot escape the test that created it.
- The `RequestEvaluator` / `EvaluatorFactory` seam (#1929) is the same pattern:
  it is what lets the queue tests assert the analyser is *never* called, on CI
  machines with no GPU.

Reach for `#[serial]` for state a test can genuinely restore (an environment
variable it sets and unsets). For anything latched, one-way, or expensive to
reset, inject the value instead.

### Guard Wiring at the Shipped Entry Point (Issues #1795, #1806, #1815)

**A unit test that builds its own subject cannot detect a missing production
caller.** This is the root diagnosis of the whole #1780 bug class: every
suppression store had thorough unit tests that constructed the store directly,
so `TargetFailureTracker` stayed green with zero production writers, the
whole #1767 removal-triage suite pinned a `focus::` helper that nothing in `src/`
called, and two independently green units shipped with the composition between
them broken (#1815).

A unit test proves a component *can* work. It says nothing about whether the
shipped path *reaches* it. So:

- **Drive the shipped entry point** — the FFI function
  (`rank_focus_neurons`, `analyze_parallel`, …) or the real pass entry point
  (`analyze_all`), not a `pub(crate)` helper. This is the **#1806 convention**:
  when a guard needs a crate-private helper, reach it *through* the FFI entry
  point rather than widening the crate export or moving the suite in-tree.
- **Assert on the FFI response shape** — the serialised keys
  (`removalCandidates`, `rejectionBreakdown`, the stable reason keys), not an
  intermediate vector. An intermediate-vector refactor must not be able to keep
  the guard green while the candidate is dropped downstream.
- **Never make an API public just to test it** (see Test Organisation above) —
  driving the shipped path is what replaces the test-only export.
- **Guard the composition, not only the units.** When two units each pass, add
  the end-to-end case that asserts what happens when they are composed.

### An Assertion That Holds Either Way Is Not Coverage (Issue #1799)

A test whose assertion is true whether or not the path under test fires proves
nothing. Two quality-skip tests passed vacuously for exactly this reason: their
fixtures targeted a single neuron, the per-target coordinated cap (Issue #1271)
silently shrank the candidate set below the threshold under test, and quality
skipping never fired — while `total <= N` and "stats are recorded" both stayed
green.

**Pin a positive precondition** so the test fails loudly when the path stops
firing:

- Assert the path was actually reached — e.g. `considered > 0`, a non-empty
  candidate list, or a counter that only the path under test increments.
- Assert on the **reason key**, not just a count, so a renamed or mis-keyed
  reason is caught.
- Assert the fixture still satisfies its own preconditions (≥ N hidden neurons,
  non-empty candidate set) so the suite cannot pass on a hollowed-out fixture.
- Watch for caps and floors that can silently shrink a fixture below the
  threshold it was built to cross.

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
The same block also owns the scripts that *enforce* a supply-chain control —
`bump-deps.sh` (the quarantine gate), `quality.sh` (the audit gate), and the
toolchain installers `scripts/runlib.sh` and `scripts/fuzz-ci.sh` — because
editing the enforcement is equivalent to editing the declaration (Issue #1914).
The block's inclusion criterion is "files that enforce or bypass a supply-chain
control". A pull request touching any of these requires maintainer review.
Individual
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
increment the patch version. A PR version that is **strictly behind**
`origin/Develop` fails CI rather than being treated as “already bumped”
(Issue #2015) — never resolve a merge conflict by taking the older Develop
token. The authoritative version-bump policy lives in the README
[Distributed Build & Versioning](README.md#-distributed-build--versioning)
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
