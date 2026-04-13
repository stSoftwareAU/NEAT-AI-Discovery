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

# Check bash script syntax
echo "📝 Checking bash script syntax..."
find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*" -exec bash -n {} \;

echo "Running shellcheck on bash scripts..."
if ! command -v shellcheck &> /dev/null; then
    echo "shellcheck is required — install: https://github.com/koalaman/shellcheck#installing"
    exit 1
fi
SHELLCHECK_FAILED=0
while IFS= read -r script; do
    echo "  shellcheck: $script"
    if ! shellcheck -s bash "$script"; then
        SHELLCHECK_FAILED=1
    fi
done < <(find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*")
if [[ "$SHELLCHECK_FAILED" -ne 0 ]]; then
    echo "shellcheck: FAILED"
    exit 1
fi
echo "shellcheck: all scripts passed"

# Update dependencies to latest versions (including incompatible upgrades)
echo "📦 Upgrading Rust library dependencies..."
if command -v cargo-upgrade &> /dev/null; then
    cargo upgrade --incompatible
    cargo update
else
    echo "⚠️  cargo-edit not installed — skipping dependency upgrade"
    echo "   Install with: cargo install cargo-edit"
fi

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

