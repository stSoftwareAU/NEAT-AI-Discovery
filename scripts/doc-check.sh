#!/bin/bash
set -euo pipefail

# Documentation build check — verifies that cargo doc builds without warnings.
#
# Usage:
#   ./scripts/doc-check.sh
#
# This treats all documentation warnings as errors, catching broken doc links,
# malformed doc comments, and missing documentation for public items.

echo "📖 Building documentation (warnings as errors)..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Security sweep ledger integrity (Issue #2088): the index must parse, and prose
# records and index entries must match both ways — a record with no index entry
# is invisible to the next automated sweep. ./quality.sh runs the same test.
echo "🔐 Checking security sweep ledger..."
cargo test --test issue_2088_sweep_ledger_contract

echo "✅ Documentation build passed — no warnings"
