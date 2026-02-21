#!/bin/bash
set -euo pipefail

# Source cargo environment if available (needed for non-login shells)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

export RUSTFLAGS="-D warnings"
echo "🔍 Pre-deployment Quality Check"
echo "================================"

# Check bash script syntax
echo "📝 Checking bash script syntax..."
find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*" -exec bash -n {} \;

# Licence and dependency audit
echo "📜 Running licence and dependency audit..."
cargo deny check

# Use workspace for faster builds
echo "🏗️ Building (debug) for quick feedback..."
cargo build

echo "🪄 Auto-formatting code..."
cargo fmt --all

echo "🔧 Running linter..."
cargo clippy --all-targets --all-features -- -D warnings -D clippy::uninlined_format_args

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

