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

# Ensure nightly toolchain is available.
#
# The channel deliberately floats rather than pinning `nightly-YYYY-MM-DD`
# (Issue #1912): cargo-fuzz builds the targets with `-Z sanitizer`, and a dated
# nightly goes stale against the sanitiser and `libfuzzer-sys` support the fuzz
# targets need. The toolchain comes from rustup's signed channel, not crates.io,
# so it carries no third-party `build.rs` — the risk the tool pin below closes.
if ! rustup run nightly rustc --version >/dev/null 2>&1; then
    echo "Installing nightly toolchain..."
    rustup install nightly
fi

# Ensure cargo-fuzz is installed.
#
# Pinned with both `--locked` and `--version` (Issue #1223; this call site was
# missed until Issue #1912). Without `--version` whatever is latest on crates.io
# at run time is fetched; without `--locked` cargo re-resolves the full
# transitive graph, so a freshly poisoned dependency would execute its
# `build.rs` on this runner. Bump the pin deliberately — `quality/cargo_install_pinning.sh`
# enforces that both flags stay present.
if ! cargo +nightly fuzz --version >/dev/null 2>&1; then
    echo "Installing cargo-fuzz 0.13.2..."
    cargo +nightly install --locked --version 0.13.2 cargo-fuzz
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
    # `--locked` keeps the run on the committed `fuzz/Cargo.lock` resolution the
    # README promises; without it cargo re-resolves the graph and the run is no
    # longer reproducible (Issue #1992).
    if cargo +nightly fuzz run --locked "${target}" -- -max_total_time="${MAX_TIME}"; then
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
