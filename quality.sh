#!/bin/bash
set -euo pipefail

export RUSTFLAGS="-D warnings"
echo "🔍 Pre-deployment Quality Check"
echo "================================"

# Check bash script syntax
echo "📝 Checking bash script syntax..."
find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*" -exec bash -n {} \;

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
cargo test --all-targets --all-features

echo "🏗️ Building release library..."
cargo build --release --lib

echo "✅ All quality checks passed!"

