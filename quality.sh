#!/bin/bash
set -euo pipefail

# Source cargo environment if available (needed for non-login shells)
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

export RUSTFLAGS="-D warnings"
echo "🔍 Pre-deployment Quality Check"
echo "================================"

# Committed `bash -n` gate, shared with CI (Issue #1755)
echo "📝 Checking bash script syntax..."
./quality/bash_syntax.sh .

# Committed ShellCheck gate, shared with CI (Issue #1898)
echo "🐚 Running shellcheck on bash scripts..."
./quality/shellcheck.sh .

# Committed `cargo install` pinning gate (Issue #1912, enforcing Issue #1223).
echo "📌 Checking cargo install pinning..."
./quality/cargo_install_pinning.sh .

# PR summaries must stay in their canonical archive dir (Issue #1613).
echo "📄 Checking PR summary layout..."
./scripts/check-pr-summary-location.sh

# Dependency bumps deliberately do NOT happen here (Issue #1865). This gate is
# the documented pre-commit step, so upgrading here pulled crates published
# minutes earlier — bypassing both the Renovate `minimumReleaseAge` window and
# the `VIBE_BUMP_QUARANTINE_HOURS` gate in ./bump-deps.sh, and executing a
# freshly-poisoned crate's build.rs on the contributor's machine. A quality gate
# verifies the tree; it must not mutate its dependency graph. Bump with
# `./bump-deps.sh` (quarantine-gated) or let Renovate raise the PR.

# Licence and dependency audit
echo "📜 Running licence and dependency audit..."
cargo deny check

# Use workspace for faster builds
echo "🏗️ Building (debug) for quick feedback..."
cargo build

echo "🪄 Auto-formatting code..."
cargo fmt --all

echo "🔧 Running linter..."
# Lint rules are configured in Cargo.toml [lints.clippy] — do not add -D/-W flags here (Issue #876)
cargo clippy --all-targets --all-features -- -D warnings

echo "✅ Running type checks..."
cargo check --all-targets --all-features

echo "🧪 Running tests..."
# Tests that mutate shared global state (env vars, deadline overrides, watchdog) are marked
# with #[serial] from the serial_test crate and will not run concurrently with each other.
# Note: Exclude benchmarks (--benches) since criterion benchmarks use custom harness and fail with --test-threads
cargo test --lib --tests --all-features -- --test-threads=2

echo "📖 Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

echo "🏗️ Building release library..."
cargo build --release --lib

echo "✅ All quality checks passed!"

