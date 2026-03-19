#!/bin/bash
set -euo pipefail

# Fuzzing CI script — runs both fuzz targets for a configurable duration.
#
# Usage:
#   ./scripts/fuzz-ci.sh [max_total_time_per_target]
#
# Default: 30 seconds per target (60 seconds total).
# Requires: nightly Rust toolchain and cargo-fuzz.
#
# This script is intended to be called from CI (e.g., a GitHub Actions fuzzing
# job) or locally to exercise the fuzz targets before merge.

MAX_TIME="${1:-30}"

echo "🔍 Fuzzing CI — ${MAX_TIME}s per target"
echo "========================================="

# Ensure nightly toolchain is available
if ! rustup run nightly rustc --version >/dev/null 2>&1; then
    echo "Installing nightly toolchain..."
    rustup install nightly
fi

# Ensure cargo-fuzz is installed
if ! cargo +nightly fuzz --version >/dev/null 2>&1; then
    echo "Installing cargo-fuzz..."
    cargo +nightly install cargo-fuzz
fi

FUZZ_TARGETS=(
    "fuzz_ffi_deserialisation"
    "fuzz_ffi_entry_points"
)

FAILED=0

for target in "${FUZZ_TARGETS[@]}"; do
    echo ""
    echo "▶ Running fuzz target: ${target} (max ${MAX_TIME}s)"
    echo "---------------------------------------------------"
    if cargo +nightly fuzz run "${target}" -- -max_total_time="${MAX_TIME}"; then
        echo "✅ ${target} completed without crashes"
    else
        echo "❌ ${target} failed"
        FAILED=1
    fi
done

echo ""
if [ "${FAILED}" -eq 0 ]; then
    echo "✅ All fuzz targets passed"
else
    echo "❌ Some fuzz targets failed"
    exit 1
fi
