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

**Always run `./quality.sh` before committing.** CI treats warnings as errors.
For the full list of checks performed by `./quality.sh` and details of the CI
pipeline, see [AGENTS.md — Quality Gate](AGENTS.md#5-quality-gate).

**GPU tests are skipped in CI** (no GPU available). For full coverage, run
`./quality.sh` locally before pushing.

---

## 🎨 Code Style

For the full coding conventions — Australian English requirements, formatting,
linting, coding principles, and Rust best practices — see
[AGENTS.md — Coding Conventions](AGENTS.md#3-coding-conventions).

---

## 🧪 Testing Guidelines

For the full testing philosophy — TDD workflow, unit tests vs benchmarks, test
organisation, and test outcomes vs implementation — see
[AGENTS.md — Testing Philosophy](AGENTS.md#4-testing-philosophy).

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

Do not manually bump versions. CI increments `Cargo.toml` patch versions when
`src/` changes are detected. Call `get_library_version()` to confirm what a
worker has loaded.

### 🚀 Deployment Checklist

1. Run `./quality.sh` in this repository
2. If changes affect NEAT-AI integration, also run `./quality.sh` in the
   NEAT-AI repository
3. Verify all tests pass before committing

---

## 🏗️ Project Structure

For the full source layout with all files and directory descriptions, see
[AGENTS.md — Architecture](AGENTS.md#2-architecture).

---

## 📚 Further Reading

See the [Additional Documentation](README.md#additional-documentation) table in
README.md for a comprehensive index of all project documentation.
